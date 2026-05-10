"""
xgb_engine.py
Consolidates XGBoost model training and inference.
"""

"""
advanced_model/ml.train_calibration_model.py
======================================================
Train per-category XGBoost classifiers to calibrate win probability
(did actual_value > bookmaker_line?). Compares against:
  1. Raw sim-based confidence (baseline)
  2. Platt scaling (if enough data)

Saves model only if it beats BOTH alternatives on Brier score.

Output:
  models/xgb_calibration_POINTS.json   (if gate passes)
  models/xgb_calibration_REBOUNDS.json (if gate passes)
  models/xgb_calibration_ASSISTS.json  (if gate passes)
  logs/xgb_calibration_training.txt

Usage:
    python3 -m ml.train_calibration_model
"""

import os
import json
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

MODEL_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH  = os.path.join(MODEL_DIR, 'database', 'train_v2.parquet')
VAL_PATH    = os.path.join(MODEL_DIR, 'database', 'val_v2.parquet')
MODELS_DIR  = os.path.join(MODEL_DIR, 'models')
LOG_PATH    = os.path.join(MODEL_DIR, 'logs', 'xgb_calibration_training_v2.txt')

CATEGORIES  = ['POINTS', 'REBOUNDS', 'ASSISTS']
BRIER_IMPROVEMENT = 0.005   # must beat both baselines by this absolute margin

EXCLUDE_COLS = {
    'game_date', 'player', 'category', 'season',
    'model_bias',          # regression residual — not directly useful for classification
    'actual_value',        # leakage
    'actual_derived',      # leakage — computed from model_bias + projected
    'y',                   # leakage — this IS the target
    'over_odds', 'under_odds',
    'opp_team_id',
    'cat_points', 'cat_rebounds', 'cat_assists',
    'direction',
}


def get_calibration_features(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE_COLS
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]


def brier_score(y_true, y_prob):
    return float(np.mean((y_prob - y_true) ** 2))


def platt_scale(y_train, y_prob_train, y_prob_val):
    """Fit logistic Platt scaling on train, apply to val."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(y_prob_train.reshape(-1, 1), y_train)
    return lr.predict_proba(y_prob_val.reshape(-1, 1))[:, 1]


def train_calibration_model():
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("[!] xgboost not installed. Run: pip install xgboost")
        raise
    from sklearn.calibration import calibration_curve

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    print("=" * 60)
    print("  XGBoost CALIBRATION MODEL TRAINING  (ML.4)")
    print("=" * 60)

    train_df = pd.read_parquet(TRAIN_PATH)
    val_df   = pd.read_parquet(VAL_PATH)

    lines = []
    lines.append("=" * 60)
    lines.append("  XGBoost CALIBRATION MODEL — TRAINING RESULTS")
    lines.append("=" * 60)

    results = {}

    for cat in CATEGORIES:
        print(f"\n── {cat} ──────────────────────────────────────────────")
        lines.append(f"\n── {cat} ─────────────────────────────────────────────")

        tr = train_df[train_df['category'] == cat].dropna(subset=['model_bias']).copy()
        va = val_df  [val_df  ['category'] == cat].dropna(subset=['model_bias']).copy()

        # Binary target: did the OVER hit? (actual > bookmaker_line)
        # We need bookmaker_line + actual → re-derive from projected + model_bias
        # actual = projected_value + model_bias
        tr['actual_derived'] = tr['projected_value'] + tr['model_bias']
        va['actual_derived'] = va['projected_value'] + va['model_bias']
        tr['y'] = (tr['actual_derived'] > tr['bookmaker_line']).astype(int)
        va['y'] = (va['actual_derived'] > va['bookmaker_line']).astype(int)

        feature_cols = get_calibration_features(tr)
        medians = tr[feature_cols].median()
        tr[feature_cols] = tr[feature_cols].fillna(medians)
        va[feature_cols] = va[feature_cols].fillna(medians)

        X_tr, y_tr = tr[feature_cols].values, tr['y'].values
        X_va, y_va = va[feature_cols].values, va['y'].values

        lines.append(f"   Train rows  : {len(X_tr):,}  (OVER rate: {y_tr.mean()*100:.1f}%)")
        lines.append(f"   Val rows    : {len(X_va):,}  (OVER rate: {y_va.mean()*100:.1f}%)")
        lines.append(f"   Features    : {len(feature_cols)}")

        # ── Baseline 1: raw sim confidence (conf column, scaled 0–1) ──
        raw_conf_tr = (tr['conf'].fillna(50.0) / 100.0).clip(0.01, 0.99).values
        raw_conf_va = (va['conf'].fillna(50.0) / 100.0).clip(0.01, 0.99).values
        brier_raw   = brier_score(y_va, raw_conf_va)
        lines.append(f"\n   Baseline 1 — Raw sim confidence Brier : {brier_raw:.5f}")
        print(f"   Baseline 1 Brier (raw conf): {brier_raw:.5f}")

        # ── Baseline 2: Platt scaling on sim confidence ──
        try:
            platt_va  = platt_scale(y_tr, raw_conf_tr, raw_conf_va)
            brier_platt = brier_score(y_va, platt_va)
            lines.append(f"   Baseline 2 — Platt scaled Brier       : {brier_platt:.5f}")
            print(f"   Baseline 2 Brier (Platt):    {brier_platt:.5f}")
        except Exception as e:
            brier_platt = brier_raw
            lines.append(f"   Baseline 2 — Platt skipped ({e})")

        best_baseline = min(brier_raw, brier_platt)

        # ── XGBoost Calibration Classifier ──
        clf = XGBClassifier(
            n_estimators          = 500,
            max_depth             = 3,
            learning_rate         = 0.05,
            subsample             = 0.8,
            colsample_bytree      = 0.8,
            reg_alpha             = 0.5,
            reg_lambda            = 2.0,
            early_stopping_rounds = 30,
            eval_metric           = 'logloss',
            use_label_encoder     = False,
            verbosity             = 0,
            random_state          = 42,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        xgb_probs  = clf.predict_proba(X_va)[:, 1]
        brier_xgb  = brier_score(y_va, xgb_probs)
        n_trees    = clf.best_iteration + 1 if hasattr(clf, 'best_iteration') else clf.n_estimators

        improvement_vs_raw   = brier_raw   - brier_xgb
        improvement_vs_platt = brier_platt - brier_xgb
        beats_both = (improvement_vs_raw   >= BRIER_IMPROVEMENT and
                      improvement_vs_platt >= BRIER_IMPROVEMENT)

        lines.append(f"\n   XGBoost Brier (val)            : {brier_xgb:.5f}")
        lines.append(f"   Δ vs raw sim conf              : {improvement_vs_raw:+.5f}")
        lines.append(f"   Δ vs Platt                     : {improvement_vs_platt:+.5f}")
        lines.append(f"   Gate (≥0.005 improvement both) : {'PASS' if beats_both else 'FAIL'}")
        lines.append(f"   Best n_estimators              : {n_trees}")
        print(f"   XGBoost Brier: {brier_xgb:.5f}  Δraw={improvement_vs_raw:+.5f}  Δplatt={improvement_vs_platt:+.5f}")

        # ── Calibration reliability curve (5 bins) ──
        lines.append(f"\n   Reliability curve (XGBoost, 5 bins):")
        try:
            prob_true, prob_pred = calibration_curve(y_va, xgb_probs, n_bins=5, strategy='quantile')
            for pt, pp in zip(prob_true, prob_pred):
                lines.append(f"     predicted={pp:.3f}  actual={pt:.3f}  diff={pt-pp:+.3f}")
        except Exception as e:
            lines.append(f"     (Could not compute: {e})")

        # ── Top features ──
        imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
        lines.append(f"\n   Top 10 feature importances:")
        for feat, score in imp.head(10).items():
            lines.append(f"     {feat:<35} {score:.4f}")

        if beats_both:
            model_path = os.path.join(MODELS_DIR, f'xgb_calibration_{cat}_v2.json')
            clf.save_model(model_path)
            meta = {
                'category':          cat,
                'feature_cols':      feature_cols,
                'medians':           medians[feature_cols].to_dict(),
                'brier_raw':         float(brier_raw),
                'brier_platt':       float(brier_platt),
                'brier_xgb':         float(brier_xgb),
                'improvement_vs_raw':   float(improvement_vs_raw),
                'improvement_vs_platt': float(improvement_vs_platt),
            }
            with open(model_path.replace('.json', '_meta.json'), 'w') as f:
                json.dump(meta, f, indent=2)
            lines.append(f"\n   [✅] GATE PASSED — model saved → {model_path}")
            print(f"   [✅] GATE PASSED — saved {model_path}")
            results[cat] = 'SAVED'
        else:
            lines.append(f"\n   [⚠️]  GATE FAILED — no calibration model for {cat}")
            print(f"   [⚠️]  GATE FAILED")
            results[cat] = 'NO_CALIBRATION'

    # ── Summary ──
    lines.append("\n" + "=" * 60)
    lines.append("  SUMMARY")
    lines.append("=" * 60)
    for cat, verdict in results.items():
        lines.append(f"  {cat:<12} → {verdict}")

    report = "\n".join(lines)
    print("\n" + report)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n[+] Log saved → {LOG_PATH}")
    print("\n[✅] ML.4 Complete.")


if __name__ == '__main__':
    train()


# --- FROM train_residual_model.py ---

"""
advanced_model/ml.train_residual_model.py
===================================================
Train per-category XGBoost regressors that predict model_bias
(actual - projected). Only saves a model if it achieves ≥3% MAE
improvement over the zero-correction baseline on the validation set.

Output:
  models/xgb_residual_POINTS.json
  models/xgb_residual_REBOUNDS.json
  models/xgb_residual_ASSISTS.json  (only where gate passes)
  logs/xgb_residual_training.txt

Usage:
    python3 -m ml.train_residual_model
"""

import os
import json
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

MODEL_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_PATH  = os.path.join(MODEL_DIR, 'database', 'train.parquet')
VAL_PATH    = os.path.join(MODEL_DIR, 'database', 'val.parquet')
MODELS_DIR  = os.path.join(MODEL_DIR, 'models')
LOG_PATH    = os.path.join(MODEL_DIR, 'logs', 'xgb_residual_training.txt')

CATEGORIES  = ['POINTS', 'REBOUNDS', 'ASSISTS']
IMPROVEMENT_GATE = 0.97   # model_mae must be <= baseline_mae * 0.97

# Features to exclude from X (leakage, identifiers, or the target itself)
EXCLUDE_COLS = {
    'game_date', 'player', 'category', 'season',
    'model_bias',          # TARGET — do not leak
    'actual_value',        # leakage
    'over_odds', 'under_odds',  # raw odds already represented via implied probs
    'opp_team_id',         # categorical int — too sparse without encoding; opp_def_* captures effect
    'cat_points', 'cat_rebounds', 'cat_assists',  # redundant when training per-category
    'direction',           # raw string, direction_bin already encodes it
}


def get_residual_features(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE_COLS
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]


def train_residual_model():
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("[!] xgboost not installed. Run: pip install xgboost")
        raise

    from sklearn.metrics import mean_absolute_error

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    print("=" * 60)
    print("  XGBoost RESIDUAL CORRECTION TRAINING  (ML.3)")
    print("=" * 60)

    train_df = pd.read_parquet(TRAIN_PATH)
    val_df   = pd.read_parquet(VAL_PATH)

    lines = []
    lines.append("=" * 60)
    lines.append("  XGBoost RESIDUAL MODEL — TRAINING RESULTS")
    lines.append("=" * 60)

    results = {}

    for cat in CATEGORIES:
        print(f"\n── {cat} ──────────────────────────────────────────────")
        lines.append(f"\n── {cat} ─────────────────────────────────────────────")

        tr = train_df[train_df['category'] == cat].dropna(subset=['model_bias']).copy()
        va = val_df  [val_df  ['category'] == cat].dropna(subset=['model_bias']).copy()

        feature_cols = get_residual_features(tr)
        # Fill any remaining NaNs with column median from train set
        medians = tr[feature_cols].median()
        tr[feature_cols] = tr[feature_cols].fillna(medians)
        va[feature_cols] = va[feature_cols].fillna(medians)

        X_tr, y_tr = tr[feature_cols].values, tr['model_bias'].values
        X_va, y_va = va[feature_cols].values, va['model_bias'].values

        lines.append(f"   Train rows : {len(X_tr):,}")
        lines.append(f"   Val rows   : {len(X_va):,}")
        lines.append(f"   Features   : {len(feature_cols)}")

        # Baseline: predict 0 (no correction)
        baseline_mae = mean_absolute_error(y_va, np.zeros_like(y_va))
        lines.append(f"   Baseline MAE (zero correction) : {baseline_mae:.4f}")

        # Train XGBoost
        model = XGBRegressor(
            n_estimators      = 500,
            max_depth         = 4,
            learning_rate     = 0.05,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            reg_alpha         = 0.1,
            reg_lambda        = 1.0,
            early_stopping_rounds = 30,
            eval_metric       = 'mae',
            verbosity         = 0,
            random_state      = 42,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        preds      = model.predict(X_va)
        model_mae  = mean_absolute_error(y_va, preds)
        threshold  = baseline_mae * IMPROVEMENT_GATE
        passes     = model_mae <= threshold
        n_trees    = model.best_iteration + 1 if hasattr(model, 'best_iteration') else model.n_estimators

        improvement_pct = (baseline_mae - model_mae) / baseline_mae * 100

        lines.append(f"   Model MAE (val)                : {model_mae:.4f}")
        lines.append(f"   Gate threshold (97% baseline)  : {threshold:.4f}")
        lines.append(f"   Improvement                    : {improvement_pct:+.2f}%")
        lines.append(f"   Best n_estimators              : {n_trees}")

        if passes:
            model_path = os.path.join(MODELS_DIR, f'xgb_residual_{cat}.json')
            model.save_model(model_path)
            # Also save feature column names and medians for inference
            meta = {
                'category':    cat,
                'feature_cols': feature_cols,
                'medians':     medians[feature_cols].to_dict(),
                'baseline_mae': float(baseline_mae),
                'model_mae':    float(model_mae),
                'improvement_pct': float(improvement_pct),
            }
            meta_path = model_path.replace('.json', '_meta.json')
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            lines.append(f"   [✅] GATE PASSED — model saved to {model_path}")
            print(f"   [✅] GATE PASSED — MAE {model_mae:.4f} vs baseline {baseline_mae:.4f} ({improvement_pct:+.2f}%)")
            results[cat] = 'SAVED'
        else:
            lines.append(f"   [⚠️]  GATE FAILED — no ML correction for {cat}")
            lines.append(f"         model_mae={model_mae:.4f} > gate={threshold:.4f}")
            print(f"   [⚠️]  GATE FAILED — MAE {model_mae:.4f} did not beat {threshold:.4f}")
            results[cat] = 'NO_ML_CORRECTION'

        # Top 10 feature importances
        imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        lines.append(f"\n   Top 10 feature importances:")
        for feat, score in imp.head(10).items():
            lines.append(f"     {feat:<35} {score:.4f}")

    # Summary
    lines.append("\n" + "=" * 60)
    lines.append("  SUMMARY")
    lines.append("=" * 60)
    for cat, verdict in results.items():
        lines.append(f"  {cat:<12} → {verdict}")

    report = "\n".join(lines)
    print("\n" + report)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n[+] Log saved → {LOG_PATH}")
    print("\n[✅] ML.3 Complete.")


if __name__ == '__main__':
    train()


# --- FROM inference.py (live ML corrections) ---

from ml.feature_pipeline import get_player_features, build_inference_vector
"""
advanced_model/ml/inference.py
==============================
Live inference module for XGBoost Calibration and Residual Correction models.
Computes features on the fly using sqlite queries.
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from config import settings

_models_calibration = {}
_models_residual = {}
_metadata_calibration = {}
_metadata_residual = {}
_initialized = False

def init_models():
    global _initialized
    if _initialized: return
    _initialized = True

    try:
        import xgboost as xgb
    except ImportError:
        print("[!] XGBoost not installed. ML features disabled.")
        return

    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), settings.ML_MODEL_PATH)
    
    for cat in ['POINTS', 'REBOUNDS', 'ASSISTS']:
        # Load Calibration Models
        if settings.USE_ML_CALIBRATION:
            calib_path = os.path.join(model_dir, f'xgb_calibration_{cat}_v2.json')
            meta_path = calib_path.replace('.json', '_meta.json')
            if os.path.exists(calib_path) and os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    _metadata_calibration[cat] = json.load(f)
                bst = xgb.Booster()
                bst.load_model(calib_path)
                _models_calibration[cat] = bst

        # Load Residual Models
        if settings.USE_ML_RESIDUAL_CORRECTION:
            resid_path = os.path.join(model_dir, f'xgb_residual_{cat}_v2.json')
            meta_path = resid_path.replace('.json', '_meta.json')
            if os.path.exists(resid_path) and os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    _metadata_residual[cat] = json.load(f)
                bst = xgb.Booster()
                bst.load_model(resid_path)
                _models_residual[cat] = bst


def apply_ml_corrections(player_name: str, stat_category: str, game_date: str, opp_team_name: str, is_home: bool,
                         projected: float, line: float, edge_pct: float, conf: float, over_odds: float, under_odds: float):
    init_models()
    cat = stat_category.upper()
    
    result = {
        'ml_correction_applied': 'NONE',
        'projected': projected,
        'confidence': conf
    }
    
    if not settings.USE_ML_CALIBRATION and not settings.USE_ML_RESIDUAL_CORRECTION:
        return result
        
    if cat not in _metadata_calibration and cat not in _metadata_residual:
        return result
        
    import xgboost as xgb
    base_features = get_player_features(player_name, game_date, opp_team_name, is_home)
    
    applied = []
    
    # 1. Residual Correction
    if settings.USE_ML_RESIDUAL_CORRECTION and cat in _models_residual:
        meta = _metadata_residual[cat]
        X = build_inference_vector(cat, base_features, projected, line, edge_pct, conf, over_odds, under_odds, meta)
        dmatrix = xgb.DMatrix(X, feature_names=meta['feature_cols'])
        bias = _models_residual[cat].predict(dmatrix)[0]
        result['projected'] = projected + float(bias)
        applied.append('RESIDUAL')
        # Re-eval features for calibration with new projection
        projected = result['projected']
        
    # 2. Calibration
    if settings.USE_ML_CALIBRATION and cat in _models_calibration:
        meta = _metadata_calibration[cat]
        X = build_inference_vector(cat, base_features, projected, line, edge_pct, conf, over_odds, under_odds, meta)
        dmatrix = xgb.DMatrix(X, feature_names=meta['feature_cols'])
        prob = _models_calibration[cat].predict(dmatrix)[0]
        
        # The model predicts probability of OVER.
        # If the pick is UNDER (projected < line), the confidence is 1 - prob.
        if projected < line:
            prob = 1.0 - prob
            
        result['confidence'] = float(prob) * 100.0
        applied.append('CALIBRATION')
        
    if applied:
        result['ml_correction_applied'] = 'BOTH' if len(applied) == 2 else applied[0]
        
    return result
