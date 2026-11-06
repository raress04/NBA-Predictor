"""
ml/shadow_logger.py
====================
P1-3: Shadow ML Pipeline — logging-only module.

For every accepted live prop pick that passes all gates in live_ingestion.py,
this module runs XGB calibration inference SILENTLY and logs the result to
ml_shadow_picks in bet_tracker.db.

USE_ML_CALIBRATION stays False. This is pure data collection.
Goal: ~200 resolved picks (~4 weeks) for evaluate_shadow.py (P1-4).
"""

import os
import json
import sqlite3
import numpy as np
from datetime import datetime, timezone

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'database', 'bet_tracker.db')
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'models')

# Cache loaded models/meta so we only hit disk once per process
_xgb_models = {}   # cat -> xgb.Booster
_xgb_meta   = {}   # cat -> dict with feature_cols, medians
_models_loaded = False


def _load_models():
    """Lazy-load XGB calibration models (silently — errors are non-fatal)."""
    global _models_loaded
    if _models_loaded:
        return
    _models_loaded = True

    try:
        import xgboost as xgb
    except ImportError:
        return  # XGBoost not installed — shadow logging skipped silently

    for cat in ['POINTS', 'REBOUNDS', 'ASSISTS']:
        model_path = os.path.join(_MODELS_DIR, f'xgb_calibration_{cat}_v2.json')
        meta_path  = model_path.replace('.json', '_meta.json')
        if os.path.exists(model_path) and os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    _xgb_meta[cat] = json.load(f)
                bst = xgb.Booster()
                bst.load_model(model_path)
                _xgb_models[cat] = bst
            except Exception:
                pass  # silently skip broken model


def _xgb_predict(cat: str, feature_vec: dict) -> float:
    """
    Run XGB calibration inference. Returns probability of OVER (0-1),
    or None if model not available.
    """
    _load_models()
    if cat not in _xgb_models:
        return None

    try:
        import xgboost as xgb
        meta = _xgb_meta[cat]
        feat_cols = meta['feature_cols']
        medians   = meta['medians']

        row = [float(feature_vec.get(c, medians.get(c, 0.0)) or medians.get(c, 0.0))
               for c in feat_cols]
        X = np.array(row, dtype=np.float32).reshape(1, -1)
        dm = xgb.DMatrix(X, feature_names=feat_cols)
        prob = float(_xgb_models[cat].predict(dm)[0])
        return max(0.0, min(1.0, prob))
    except Exception:
        return None


_ML_SHADOW_COLUMNS_OK = set()   # DB paths already migrated in this process


def _ensure_ml_shadow_columns(conn) -> None:
    """Idempotent migration: conf_stage records which scale sim_confidence is on
    ('raw_dir' = direction-specific raw sim frequency, Step 1 of CRITICAL_BUGS_FOUND_06.11)."""
    if _DB_PATH in _ML_SHADOW_COLUMNS_OK:
        return
    cols = {r[1] for r in conn.execute("PRAGMA table_info(ml_shadow_picks)")}
    if cols and 'conf_stage' not in cols:
        conn.execute("ALTER TABLE ml_shadow_picks ADD COLUMN conf_stage TEXT")
    _ML_SHADOW_COLUMNS_OK.add(_DB_PATH)


def log_ml_shadow_pick(
    game_date: str,
    player: str,
    stat_category: str,
    direction: str,
    line: float,
    projection: float,
    sim_confidence: float,
    edge_pct: float,
    # Features for XGB inference
    base_features: dict = None,
    over_odds: float = 1.91,
    under_odds: float = 1.91,
    model_version: str = 'ERA3',
    cohort: str = 'selected',
    conf_stage: str = None,
) -> bool:
    """
    Called after a pick passes all gates in live_ingestion.py.
    Runs XGB inference, logs (sim_confidence, xgb_confidence) to ml_shadow_picks.
    Returns True if logged, False if skipped/error.
    Does NOT affect actual pick output.
    """
    cat = stat_category.upper()
    if cat not in ('POINTS', 'REBOUNDS', 'ASSISTS'):
        return False  # only log props we have XGB models for

    xgb_confidence = None

    if base_features is not None:
        # Build inference vector matching xgb_engine feature space
        vec = dict(base_features)
        vec['projected_value'] = float(projection) if projection else 0.0
        vec['bookmaker_line']  = float(line) if line else 0.0
        vec['edge_pct']        = float(edge_pct) if edge_pct else 0.0
        vec['conf']            = float(sim_confidence) if sim_confidence else 50.0
        vec['gap']             = vec['projected_value'] - vec['bookmaker_line']
        oi = (1.0 / over_odds)  if over_odds  > 0 else 0.5
        ui = (1.0 / under_odds) if under_odds > 0 else 0.5
        vec['over_implied']   = oi
        vec['under_implied']  = ui
        vec['vig']            = max(0.0, oi + ui - 1.0)
        vec['odds_skew']      = oi - ui
        vec['direction_bin']  = 1.0 if direction.upper() == 'OVER' else 0.0
        for c in ('cat_points', 'cat_rebounds', 'cat_assists'):
            vec[c] = 0.0
        vec[f'cat_{cat.lower()}'] = 1.0

        prob_over = _xgb_predict(cat, vec)
        if prob_over is not None:
            # For UNDER picks, flip to get probability the pick wins
            if direction.upper() == 'UNDER':
                xgb_confidence = (1.0 - prob_over) * 100.0
            else:
                xgb_confidence = prob_over * 100.0

    try:
        conn = sqlite3.connect(_DB_PATH)
        _ensure_ml_shadow_columns(conn)
        conn.execute("""
            INSERT INTO ml_shadow_picks
              (game_date, player, stat_category, direction, line, projection,
               sim_confidence, xgb_confidence, edge_pct, model_version, cohort,
               conf_stage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_date, player, cat, direction.upper(),
            float(line) if line else None,
            float(projection) if projection else None,
            float(sim_confidence) if sim_confidence else None,
            round(xgb_confidence, 2) if xgb_confidence is not None else None,
            float(edge_pct) if edge_pct else None,
            model_version,
            cohort,
            conf_stage,
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        # Never crash the pipeline
        print(f"  [shadow_logger] WARN: failed to log {player} {cat}: {e}")
        return False


def resolve_ml_shadow_picks(game_date: str, force: bool = False,
                            db_path: str = None, nba_db: str = None):
    """
    Called by result-resolution scripts (prepare_analyze, repair scripts).
    Grades ml_shadow_picks on game_date against REAL box scores (nba_data.db).

    Fix 2026-11-06 (CRITICAL_BUGS_FOUND_06.11 — C4): this used to join
    projection_outcomes, whose 2026-27 rows are 'live_pending' placeholders with
    actual_value = 0.0, so every OVER was graded a loss and every UNDER a win.

    Players with no box-score row (DNP / data not yet ingested) stay unresolved
    so a later run can pick them up. Pushes are marked resolved with is_win NULL.

    Args:
        force: re-grade rows that were already resolved on this date.
    """
    from utils.box_actuals import ActualsIndex, actual_for, grade, load_box_actuals

    try:
        actuals = load_box_actuals(game_date, nba_db=nba_db)
        if not actuals:
            print(f"[shadow_logger] No box scores for {game_date} yet — nothing resolved.")
            return 0
        index = ActualsIndex(actuals)

        conn = sqlite3.connect(db_path or _DB_PATH)
        where_unresolved = "" if force else "AND resolved_at IS NULL"
        rows = conn.execute(f"""
            SELECT id, player, stat_category, direction, line
            FROM ml_shadow_picks
            WHERE game_date = ? {where_unresolved}
        """, (game_date,)).fetchall()

        updated = 0
        # UTC, same as SQLite CURRENT_TIMESTAMP / datetime('now') used by the other tables
        resolved_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        for row_id, player, cat, direction, line in rows:
            actual = actual_for(index, player, cat)
            if actual is None or line is None:
                if force:
                    conn.execute(
                        "UPDATE ml_shadow_picks SET actual_value=NULL, is_win=NULL, resolved_at=NULL WHERE id=?",
                        (row_id,))
                continue
            conn.execute("""
                UPDATE ml_shadow_picks
                SET actual_value=?, is_win=?, resolved_at=?
                WHERE id=?
            """, (actual, grade(direction, actual, line), resolved_at, row_id))
            updated += 1

        conn.commit()
        conn.close()
        if updated:
            print(f"[shadow_logger] Resolved {updated} ml_shadow_picks for {game_date} (box scores)")
        return updated
    except Exception as e:
        print(f"[shadow_logger] resolve error: {e}")
        return 0
