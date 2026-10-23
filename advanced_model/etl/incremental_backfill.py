"""
Safe incremental backfill for 2026-27 season box scores.
Does NOT wipe existing data. Only appends new season rows.
Skips game_ids already present in the DB.

NOTE: Preseason games (before Oct 20, 2026) are included in the DB but
filtered out by live_ingestion.py at query time via REGULAR_SEASON_STARTS.
Do not manually delete preseason rows — they are harmless and the filter handles them.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import sqlite3
import pandas as pd
import time
import os

from nba_api.stats.endpoints import playergamelogs

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database', 'nba_data.db')
DB_PATH = os.path.normpath(DB_PATH)

SEASON = '2026-27'

def run():
    print(f"[+] Incremental backfill for season {SEASON}")
    print(f"[+] DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)

    # Get already-present game_ids for this season to avoid dupes
    existing_games = set(
        r[0] for r in conn.execute(
            "SELECT game_id FROM games WHERE season = ?", (SEASON,)
        ).fetchall()
    )
    print(f"[+] Existing game_ids for {SEASON} in DB: {len(existing_games)}")

    try:
        print(f"[+] Fetching player game logs from NBA API...")
        logs = playergamelogs.PlayerGameLogs(season_nullable=SEASON).get_data_frames()[0]
        print(f"    -> Got {len(logs)} rows total from API")
    except Exception as e:
        print(f"[-] API fetch failed: {e}")
        conn.close()
        return

    if logs.empty:
        print("[!] No data returned for this season yet. Season may not have started in API.")
        conn.close()
        return

    # Filter out already-present games
    new_logs = logs[~logs['GAME_ID'].isin(existing_games)]
    print(f"    -> New rows to insert (excluding existing): {len(new_logs)}")

    if new_logs.empty:
        print("[+] DB already up to date. Nothing to insert.")
        conn.close()
        return

    # --- Insert into games table ---
    games_df = new_logs[['GAME_ID', 'GAME_DATE', 'MATCHUP', 'SEASON_YEAR']].drop_duplicates(subset=['GAME_ID'])
    games_df = games_df.rename(columns={
        'GAME_ID': 'game_id',
        'GAME_DATE': 'game_date',
        'MATCHUP': 'matchup',
        'SEASON_YEAR': 'season'
    })
    games_df.to_sql('games', conn, if_exists='append', index=False)
    print(f"    -> Inserted {len(games_df)} new game records")

    # --- Handle minutes column ---
    if 'MIN' in new_logs.columns:
        new_logs = new_logs.copy()
        new_logs['minutes'] = new_logs['MIN']

    # --- Insert into box_scores table ---
    box_df = new_logs.rename(columns={
        'SEASON_YEAR': 'season',
        'GAME_ID': 'game_id',
        'GAME_DATE': 'game_date',
        'TEAM_ID': 'team_id',
        'PLAYER_ID': 'player_id',
        'PLAYER_NAME': 'player_name',
        'FGM': 'fgm', 'FGA': 'fga',
        'FG3M': 'fg3m', 'FG3A': 'fg3a',
        'FTM': 'ftm', 'FTA': 'fta',
        'OREB': 'oreb', 'DREB': 'dreb', 'REB': 'reb',
        'AST': 'ast', 'STL': 'stl', 'BLK': 'blk',
        'TOV': 'tov', 'PF': 'pf', 'PTS': 'pts',
        'PLUS_MINUS': 'plus_minus'
    })

    cols = ['season', 'game_id', 'game_date', 'team_id', 'player_id', 'player_name', 'minutes',
            'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta', 'oreb', 'dreb', 'reb',
            'ast', 'stl', 'blk', 'tov', 'pf', 'pts', 'plus_minus']
    cols = [c for c in cols if c in box_df.columns]
    box_df[cols].to_sql('box_scores', conn, if_exists='append', index=False)
    print(f"    -> Inserted {len(box_df)} box score rows")

    conn.commit()
    conn.close()

    print(f"\n[SUCCESS] Incremental backfill complete for {SEASON}")
    print(f"  Games added: {len(games_df)}")
    print(f"  Box score rows added: {len(box_df)}")
    print(f"  Date range: {new_logs['GAME_DATE'].min()} -> {new_logs['GAME_DATE'].max()}")

if __name__ == '__main__':
    run()
