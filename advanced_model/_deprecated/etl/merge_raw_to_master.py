"""
advanced_model/etl/merge_raw_to_master.py
==========================================
Merges resolved projection_outcomes_raw rows into master_training_data.db.
Deduplicates against existing rows via the unique index.
Adds season label '2025_26_live_raw' for the new rows.

Usage:
    python3 -m etl.merge_raw_to_master
"""

import os
import sqlite3
import pandas as pd

MODEL_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BT_DB      = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')
MASTER_DB  = os.path.join(MODEL_DIR, 'database', 'master_training_data.db')

SEASON_LABEL = '2025_26_live_raw'


def main():
    print("=" * 60)
    print("  MERGE RAW PREDICTIONS INTO MASTER DB  (ML.3b-3)")
    print("=" * 60)

    conn_bt     = sqlite3.connect(BT_DB)
    conn_master = sqlite3.connect(MASTER_DB)

    # Load resolved raw rows
    new_rows = pd.read_sql_query(
        """SELECT
            game_date,
            player,
            category,
            projected_value,
            bookmaker_line,
            actual_value,
            model_bias,
            direction,
            confidence      AS conf,
            edge_pct,
            is_valuable,
            source_file,
            source_mode
           FROM projection_outcomes_raw
           WHERE actual_value IS NOT NULL
             AND category IN ('POINTS','REBOUNDS','ASSISTS')""",
        conn_bt
    )
    conn_bt.close()
    print(f"[*] Resolved raw rows to merge: {len(new_rows):,}")

    # Add season label
    new_rows['season'] = SEASON_LABEL
    # Rename confidence to match master schema
    new_rows = new_rows.rename(columns={'confidence': 'conf'}) \
        if 'conf' not in new_rows.columns else new_rows

    # Load existing keys from master to deduplicate
    existing_keys = set(
        pd.read_sql_query(
            "SELECT game_date || '|' || player || '|' || category || '|' || source_file AS k "
            "FROM projection_outcomes",
            conn_master
        )['k']
    )
    print(f"[*] Existing rows in master: {len(existing_keys):,}")

    new_rows['_key'] = (new_rows['game_date'] + '|' + new_rows['player'] + '|' +
                        new_rows['category'] + '|' + new_rows['source_file'])
    before = len(new_rows)
    new_rows = new_rows[~new_rows['_key'].isin(existing_keys)].drop(columns=['_key'])
    print(f"[*] After dedup: {len(new_rows):,} truly new rows (removed {before - len(new_rows):,} dupes)")

    if new_rows.empty:
        print("[!] Nothing new to insert — master already up to date.")
        conn_master.close()
        return

    # Align columns with master schema (drop extras, fill missing)
    master_cols = [r[1] for r in conn_master.execute("PRAGMA table_info(projection_outcomes)")]
    for col in master_cols:
        if col not in new_rows.columns and col != 'id':
            new_rows[col] = None
    cols_to_insert = [c for c in master_cols if c != 'id' and c in new_rows.columns]
    new_rows = new_rows[cols_to_insert]

    new_rows.to_sql('projection_outcomes', conn_master, if_exists='append', index=False)
    conn_master.commit()

    # Verify final counts
    print("\n[*] Master DB final counts by season:")
    counts = pd.read_sql_query(
        "SELECT season, category, COUNT(*) AS n FROM projection_outcomes "
        "WHERE category IN ('POINTS','REBOUNDS','ASSISTS') "
        "GROUP BY season, category ORDER BY season, category",
        conn_master
    )
    print(counts.to_string(index=False))

    total = conn_master.execute(
        "SELECT COUNT(*) FROM projection_outcomes WHERE category IN ('POINTS','REBOUNDS','ASSISTS')"
    ).fetchone()[0]
    print(f"\n  TOTAL P/R/A rows: {total:,}")

    conn_master.close()
    print("\n[✅] ML.3b-3 Complete.")


if __name__ == '__main__':
    main()
