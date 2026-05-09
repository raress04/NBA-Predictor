"""
advanced_model/ml.train_test_split.py
===============================================
Chronological train / validation / test split for ML residual models.

Boundaries (strictly date-filtered, no shuffling):
  Train      : 2025-10-21 → 2026-01-31   (Oct–Dec 2025 + Jan 2026)
  Validation : 2026-02-01 → 2026-03-15
  Test       : 2026-03-16 → last available date

Output files (saved alongside feature_matrix.parquet):
  database/train.parquet
  database/val.parquet
  database/test.parquet

Usage:
    python3 -m ml.train_test_split
"""

import os
import pandas as pd

MODEL_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FM_PATH    = os.path.join(MODEL_DIR, 'database', 'feature_matrix.parquet')
OUT_DIR    = os.path.join(MODEL_DIR, 'database')
LOG_PATH   = os.path.join(MODEL_DIR, 'logs',     'train_test_split.txt')

TRAIN_END  = pd.Timestamp('2026-02-01')   # exclusive upper bound for train
VAL_END    = pd.Timestamp('2026-03-16')   # exclusive upper bound for val
CATEGORIES = ['POINTS', 'REBOUNDS', 'ASSISTS']

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def split():
    print("=" * 60)
    print("  CHRONOLOGICAL TRAIN / VAL / TEST SPLIT")
    print("=" * 60)

    df = pd.read_parquet(FM_PATH)
    df['game_date'] = pd.to_datetime(df['game_date'])

    # Drop rows where target is missing
    before = len(df)
    df = df.dropna(subset=['model_bias'])
    dropped = before - len(df)
    if dropped:
        print(f"[!] Dropped {dropped:,} rows with missing model_bias")

    # Strict chronological split — no shuffling
    train = df[df['game_date'] <  TRAIN_END].copy()
    val   = df[(df['game_date'] >= TRAIN_END) & (df['game_date'] < VAL_END)].copy()
    test  = df[df['game_date'] >= VAL_END].copy()

    splits = {'train': train, 'val': val, 'test': test}

    lines = []
    lines.append("=" * 60)
    lines.append("  TRAIN / VAL / TEST SPLIT SUMMARY")
    lines.append(f"  Feature matrix : {FM_PATH}")
    lines.append(f"  Total rows     : {len(df):,}")
    lines.append("=" * 60)

    all_ok = True
    for name, subset in splits.items():
        lines.append(f"\n── {name.upper()} ({len(subset):,} rows) "
                     f"[{subset['game_date'].min().date()} → {subset['game_date'].max().date()}]")
        for cat in CATEGORIES:
            cat_df = subset[subset['category'] == cat]
            n = len(cat_df)
            lines.append(f"   {cat:<10}: {n:>6,} rows")
            if n < 50:
                lines.append(f"   [!] WARNING: very low count for {cat} in {name}")
                all_ok = False

        # Unique players in this split
        lines.append(f"   Unique players: {subset['player'].nunique():,}")

    # Check: no player exclusively in test (severe sparsity indicator)
    train_players = set(train['player'].unique())
    val_players   = set(val['player'].unique())
    test_players  = set(test['player'].unique())

    only_in_test = test_players - train_players - val_players
    lines.append(f"\n── PLAYER COVERAGE CHECK")
    lines.append(f"   Players in train only (not val/test) : {len(train_players - val_players - test_players):,}")
    lines.append(f"   Players in test only (not train/val) : {len(only_in_test):,}")
    if only_in_test:
        lines.append(f"   [!] Players exclusively in test (data sparsity): {', '.join(sorted(only_in_test)[:10])}")
        lines.append(f"       → Noted as a limitation; model will fall back to zero-correction for these.")
    else:
        lines.append(f"   [✅] No players exclusively in test set.")

    lines.append(f"\n── VERDICT")
    if all_ok:
        lines.append("   [✅] All splits have non-trivial counts per category.")
    else:
        lines.append("   [!] Some splits have low counts — check warnings above.")

    report = "\n".join(lines)
    print(report)

    # Save splits
    for name, subset in splits.items():
        path = os.path.join(OUT_DIR, f'{name}.parquet')
        subset.to_parquet(path, index=False)
        print(f"\n[+] Saved {name}.parquet — {len(subset):,} rows")

    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n[+] Log saved → {LOG_PATH}")
    print("\n[✅] ML.2 Complete.")


if __name__ == '__main__':
    split()
