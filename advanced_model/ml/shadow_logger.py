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
from datetime import datetime

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
        conn.execute("""
            INSERT INTO ml_shadow_picks
              (game_date, player, stat_category, direction, line, projection,
               sim_confidence, xgb_confidence, edge_pct, model_version, cohort)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_date, player, cat, direction.upper(),
            float(line) if line else None,
            float(projection) if projection else None,
            float(sim_confidence) if sim_confidence else None,
            round(xgb_confidence, 2) if xgb_confidence is not None else None,
            float(edge_pct) if edge_pct else None,
            model_version,
            cohort,
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        # Never crash the pipeline
        print(f"  [shadow_logger] WARN: failed to log {player} {cat}: {e}")
        return False


def resolve_ml_shadow_picks(game_date: str):
    """
    Called by result-resolution scripts (e.g. db_manager.resolve_picks).
    Marks actual_value + is_win for all ml_shadow_picks on game_date
    that have a matching resolved entry in projection_outcomes.
    """
    try:
        conn = sqlite3.connect(_DB_PATH)

        # Join ml_shadow_picks against projection_outcomes in nba_data.db
        nba_db = os.path.join(os.path.dirname(_DB_PATH), 'nba_data.db')
        conn.execute(f"ATTACH DATABASE '{nba_db}' AS nba")

        rows = conn.execute("""
            SELECT m.id, m.player, m.stat_category, m.direction, m.line,
                   po.actual_value
            FROM ml_shadow_picks m
            JOIN nba.projection_outcomes po
              ON po.player  = m.player
             AND po.game_date = m.game_date
             AND po.category  = m.stat_category
            WHERE m.game_date = ?
              AND m.resolved_at IS NULL
              AND po.actual_value IS NOT NULL
        """, (game_date,)).fetchall()

        updated = 0
        for row_id, player, cat, direction, line, actual in rows:
            if line is None or actual is None:
                continue
            is_win = 1 if (
                (direction == 'OVER'  and actual > line) or
                (direction == 'UNDER' and actual < line)
            ) else 0
            conn.execute("""
                UPDATE ml_shadow_picks
                SET actual_value=?, is_win=?, resolved_at=?
                WHERE id=?
            """, (actual, is_win, game_date, row_id))
            updated += 1

        conn.commit()
        conn.execute("DETACH DATABASE nba")
        conn.close()
        if updated:
            print(f"[shadow_logger] Resolved {updated} ml_shadow_picks for {game_date}")
        return updated
    except Exception as e:
        print(f"[shadow_logger] resolve error: {e}")
        return 0
