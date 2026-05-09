"""
advanced_model/ml.build_training_dataset.py
==================================================
Merges Oct-Dec 2025 and Jan-Apr 2026 retrospective projection_outcomes
into a single master training database.

Usage:
    python3 -m ml.build_training_dataset
"""
import sqlite3
import os
import sys
import pandas as pd

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_OCT_DEC  = os.path.join(MODEL_DIR, 'database', 'retrospective_2025_26_oct_dec.db')
DB_JAN_APR  = os.path.join(MODEL_DIR, 'database', 'retrospective_db.db')
DB_MASTER   = os.path.join(MODEL_DIR, 'database', 'master_training_data.db')

LABEL_OCT_DEC = '2025_26_oct_dec'
LABEL_JAN_APR = '2025_26_jan_apr'


def load_outcomes(db_path: str, season_label: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        print(f"[!] DB not found: {db_path}")
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM projection_outcomes", conn)
    conn.close()
    df['season'] = season_label
    print(f"[+] {season_label}: {len(df):,} rows loaded from {os.path.basename(db_path)}")
    return df


def verify_schemas(df1: pd.DataFrame, df2: pd.DataFrame):
    """Warn on mismatched columns; fill missing ones with NULL."""
    cols1 = set(df1.columns) - {'id', 'season'}
    cols2 = set(df2.columns) - {'id', 'season'}
    missing_in_2 = cols1 - cols2
    missing_in_1 = cols2 - cols1
    if missing_in_2 or missing_in_1:
        print(f"[!] Schema difference detected (will fill with NULL):")
        if missing_in_2:
            print(f"    Missing in jan_apr → filling: {missing_in_2}")
            for col in missing_in_2:
                df2[col] = None
        if missing_in_1:
            print(f"    Missing in oct_dec → filling: {missing_in_1}")
            for col in missing_in_1:
                df1[col] = None
    else:
        print("[✅] Schemas match.")


def build():
    print(f"[+] Building master training dataset...")

    df_oct = load_outcomes(DB_OCT_DEC, LABEL_OCT_DEC)
    df_jan = load_outcomes(DB_JAN_APR, LABEL_JAN_APR)

    verify_schemas(df_oct, df_jan)

    # Drop auto-increment id, will regenerate in master
    for df in [df_oct, df_jan]:
        if 'id' in df.columns:
            df.drop(columns=['id'], inplace=True)

    combined = pd.concat([df_oct, df_jan], ignore_index=True)
    print(f"[+] Combined total: {len(combined):,} rows")

    # Write to master DB
    conn = sqlite3.connect(DB_MASTER)
    conn.execute("DROP TABLE IF EXISTS projection_outcomes")
    combined.to_sql('projection_outcomes', conn, index=True, index_label='id', if_exists='replace')

    # Unique index to prevent duplicates
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_outcome
        ON projection_outcomes (game_date, player, category, source_file)
    """)
    conn.commit()

    # Verify
    rows = conn.execute("SELECT season, COUNT(*) AS n FROM projection_outcomes GROUP BY season").fetchall()
    total = conn.execute("SELECT COUNT(*) FROM projection_outcomes").fetchone()[0]
    conn.close()

    print(f"\n[+] Master DB written to: {DB_MASTER}")
    print(f"{'─'*40}")
    for season, n in rows:
        print(f"  {season:<25} {n:>8,} rows")
    print(f"  {'TOTAL':<25} {total:>8,} rows")
    print(f"{'─'*40}")

    if total < 40_000:
        print(f"[!] WARNING: total {total:,} rows < expected 40,000")
    else:
        print(f"[✅] Gate passed: {total:,} rows >= 40,000")


if __name__ == '__main__':
    build()
