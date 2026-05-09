"""
advanced_model/etl/resolve_actuals_raw.py
==========================================
Joins projection_outcomes_raw with box_scores from nba_data.db
to populate actual_value and model_bias.

Uses unicodedata normalization for fuzzy player name matching.
Falls back to difflib for near-misses.

Usage:
    python3 -m etl.resolve_actuals_raw
"""

import os
import sqlite3
import unicodedata
import difflib
import pandas as pd

MODEL_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BT_DB      = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')
NBA_DB     = os.path.join(MODEL_DIR, 'database', 'nba_data.db')
LOG_PATH   = os.path.join(MODEL_DIR, 'logs', 'ml3b_unresolved_actuals.txt')
CAT_COL    = {'POINTS': 'pts', 'REBOUNDS': 'reb', 'ASSISTS': 'ast'}


def normalize(name: str) -> str:
    nfkd = unicodedata.normalize('NFKD', str(name))
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def main():
    print("=" * 60)
    print("  RESOLVE ACTUALS (ML.3b-2)")
    print("=" * 60)

    conn_bt  = sqlite3.connect(BT_DB)
    conn_nba = sqlite3.connect(NBA_DB)

    # Load unresolved rows
    raw = pd.read_sql_query(
        "SELECT id, game_date, player, category, projected_value "
        "FROM projection_outcomes_raw WHERE actual_value IS NULL",
        conn_bt
    )
    print(f"[*] Rows needing resolution: {len(raw):,}")

    if raw.empty:
        print("[✅] Nothing to resolve.")
        conn_bt.close()
        conn_nba.close()
        return

    # Load box scores Jan–Apr 2026 + Oct–Dec 2025
    box = pd.read_sql_query(
        "SELECT SUBSTR(game_date,1,10) AS game_date, player_name AS player, pts, reb, ast "
        "FROM box_scores",
        conn_nba
    )
    print(f"[*] Box score rows loaded: {len(box):,}")

    # Build lookup: (game_date, player_norm) → {pts, reb, ast}
    box['player_norm'] = box['player'].apply(normalize)
    raw['player_norm'] = raw['player'].apply(normalize)

    # Index for fast lookup
    box_idx = box.set_index(['game_date', 'player_norm'])

    resolved   = []
    unresolved = []

    # Build fuzzy candidates per game_date
    date_players = box.groupby('game_date')['player_norm'].apply(list).to_dict()

    for _, row in raw.iterrows():
        col = CAT_COL.get(row['category'])
        if not col:
            continue

        gd  = row['game_date']
        pn  = row['player_norm']
        key = (gd, pn)

        # Exact match
        if key in box_idx.index:
            actual = float(box_idx.loc[key, col])
            resolved.append({'id': int(row['id']),
                             'actual_value': actual,
                             'model_bias': actual - row['projected_value']})
            continue

        # Fuzzy match within same game_date
        candidates = date_players.get(gd, [])
        close = difflib.get_close_matches(pn, candidates, n=1, cutoff=0.85)
        if close:
            fkey = (gd, close[0])
            if fkey in box_idx.index:
                actual = float(box_idx.loc[fkey, col])
                resolved.append({'id': int(row['id']),
                                 'actual_value': actual,
                                 'model_bias': actual - row['projected_value']})
                continue

        unresolved.append({'game_date': row['game_date'],
                           'player':    row['player'],
                           'category':  row['category']})

    print(f"[+] Resolved  : {len(resolved):,}")
    print(f"[!] Unresolved: {len(unresolved):,} "
          f"({len(unresolved)/len(raw)*100:.1f}%)")

    # Batch update
    conn_bt.executemany(
        "UPDATE projection_outcomes_raw "
        "SET actual_value=?, model_bias=? WHERE id=?",
        [(r['actual_value'], r['model_bias'], r['id']) for r in resolved]
    )
    conn_bt.commit()

    # Sanity check on bias
    print("\n[*] Bias sanity check (post-resolution):")
    bias_check = pd.read_sql_query(
        "SELECT category, "
        "ROUND(AVG(model_bias),3) AS mean_bias, "
        "ROUND(AVG(ABS(model_bias)),3) AS mae, "
        "COUNT(*) AS n "
        "FROM projection_outcomes_raw "
        "WHERE actual_value IS NOT NULL "
        "GROUP BY category",
        conn_bt
    )
    print(bias_check.to_string(index=False))

    # Check for suspiciously large bias (column mapping error)
    for _, r in bias_check.iterrows():
        if abs(r['mean_bias']) > 2.0:
            print(f"\n[!!!] CRITICAL: {r['category']} mean_bias={r['mean_bias']} > 2.0 "
                  f"— possible column mapping error. STOPPING.")
            conn_bt.close()
            conn_nba.close()
            return

    # Resolution rate gate
    total   = len(raw)
    res_pct = len(resolved) / total * 100
    print(f"\n[*] Resolution rate: {res_pct:.1f}%")
    if res_pct < 85:
        print(f"[!] WARNING: Resolution rate {res_pct:.1f}% < 85% target")
    else:
        print(f"[✅] Resolution gate passed: {res_pct:.1f}% ≥ 85%")

    # Save unresolved log
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write(f"Unresolved actuals ({len(unresolved)} rows)\n")
        f.write("=" * 50 + "\n")
        for u in unresolved:
            f.write(f"{u['game_date']}  {u['player']:<30}  {u['category']}\n")
    print(f"[+] Unresolved log → {LOG_PATH}")

    conn_bt.close()
    conn_nba.close()
    print("\n[✅] ML.3b-2 Complete.")


if __name__ == '__main__':
    main()
