"""
Retrain residual model on v2 expanded dataset.
Reads train_v2/val_v2, saves models with _v2 suffix if gate passes.
"""
import os, json, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

MODEL_DIR  = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(MODEL_DIR, 'database', 'train_v2.parquet')
VAL_PATH   = os.path.join(MODEL_DIR, 'database', 'val_v2.parquet')
MODELS_DIR = os.path.join(MODEL_DIR, 'models')
LOG_PATH   = os.path.join(MODEL_DIR, 'logs', 'xgb_residual_training_v2.txt')

CATEGORIES = ['POINTS', 'REBOUNDS', 'ASSISTS']
IMPROVEMENT_GATE = 0.97

EXCLUDE_COLS = {
    'game_date', 'player', 'category', 'season',
    'model_bias', 'actual_value',
    'over_odds', 'under_odds',
    'opp_team_id',
    'cat_points', 'cat_rebounds', 'cat_assists',
    'direction',
}

def get_features(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]

def main():
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error

    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 60)
    print("  XGBoost RESIDUAL v2 — ML.3b-5")
    print("=" * 60)

    train_df = pd.read_parquet(TRAIN_PATH)
    val_df   = pd.read_parquet(VAL_PATH)

    lines = []
    lines.append("=" * 60)
    lines.append("  XGBoost RESIDUAL v2 — TRAINING RESULTS")
    lines.append("=" * 60)

    results = {}

    # Also load v1 results for comparison
    v1_mae = {
        'POINTS':   (4.8193, 4.7097),
        'REBOUNDS': (1.9552, 1.9093),
        'ASSISTS':  (1.4619, 1.4475),
    }

    for cat in CATEGORIES:
        print(f"\n── {cat} ──────────────────────────────────────────────")
        lines.append(f"\n── {cat} ─────────────────────────────────────────────")

        tr = train_df[train_df['category'] == cat].dropna(subset=['model_bias']).copy()
        va = val_df  [val_df  ['category'] == cat].dropna(subset=['model_bias']).copy()

        feature_cols = get_features(tr)
        medians = tr[feature_cols].median()
        tr[feature_cols] = tr[feature_cols].fillna(medians)
        va[feature_cols] = va[feature_cols].fillna(medians)

        X_tr, y_tr = tr[feature_cols].values, tr['model_bias'].values
        X_va, y_va = va[feature_cols].values, va['model_bias'].values

        lines.append(f"   Train rows : {len(X_tr):,}")
        lines.append(f"   Val rows   : {len(X_va):,}")
        lines.append(f"   Features   : {len(feature_cols)}")

        baseline_mae = mean_absolute_error(y_va, np.zeros_like(y_va))
        threshold    = baseline_mae * IMPROVEMENT_GATE

        model = XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            early_stopping_rounds=30, eval_metric='mae',
            verbosity=0, random_state=42,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        preds     = model.predict(X_va)
        model_mae = mean_absolute_error(y_va, preds)
        passes    = model_mae <= threshold
        n_trees   = model.best_iteration + 1

        improvement = (baseline_mae - model_mae) / baseline_mae * 100

        v1_base, v1_model = v1_mae[cat]
        v1_imp = (v1_base - v1_model) / v1_base * 100

        lines.append(f"   Baseline MAE                   : {baseline_mae:.4f}")
        lines.append(f"   Model MAE (v2)                 : {model_mae:.4f}")
        lines.append(f"   Gate threshold (97%)           : {threshold:.4f}")
        lines.append(f"   Improvement v2                 : {improvement:+.2f}%")
        lines.append(f"   Improvement v1 (for comparison): {v1_imp:+.2f}%")
        lines.append(f"   Best n_estimators              : {n_trees}")

        print(f"   Baseline: {baseline_mae:.4f}  Model: {model_mae:.4f}  "
              f"Improvement: {improvement:+.2f}%  Gate: {'PASS' if passes else 'FAIL'}")

        if passes:
            path = os.path.join(MODELS_DIR, f'xgb_residual_{cat}_v2.json')
            model.save_model(path)
            meta = {
                'category': cat, 'feature_cols': feature_cols,
                'medians': medians[feature_cols].to_dict(),
                'baseline_mae': float(baseline_mae), 'model_mae': float(model_mae),
                'improvement_pct': float(improvement),
            }
            with open(path.replace('.json', '_meta.json'), 'w') as f:
                json.dump(meta, f, indent=2)
            lines.append(f"   [✅] GATE PASSED — model saved: {path}")
            results[cat] = f'SAVED ({improvement:+.2f}%)'
        else:
            lines.append(f"   [⚠️]  GATE FAILED")
            results[cat] = f'NO_ML_CORRECTION ({improvement:+.2f}%)'

        imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        lines.append(f"\n   Top 10 feature importances:")
        for feat, score in imp.head(10).items():
            lines.append(f"     {feat:<35} {score:.4f}")

    lines.append("\n" + "=" * 60)
    lines.append("  SUMMARY")
    lines.append("=" * 60)
    for cat, verdict in results.items():
        lines.append(f"  {cat:<12} → {verdict}")

    report = "\n".join(lines)
    print("\n" + report)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n[+] Log → {LOG_PATH}")
    print("\n[✅] ML.3b-5 Complete.")

if __name__ == '__main__':
    main()
