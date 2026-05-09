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


def get_features(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE_COLS
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]


def train():
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

        feature_cols = get_features(tr)
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
