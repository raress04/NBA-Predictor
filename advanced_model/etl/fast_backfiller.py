import sqlite3
import pandas as pd
import time
import os
from nba_api.stats.endpoints import playergamelogs

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'database', 'nba_data.db')

def setup_db(conn):
    cursor = conn.cursor()
    # Wipe the old slow tables to prevent schema clashes
    cursor.execute('DROP TABLE IF EXISTS box_scores')
    cursor.execute('DROP TABLE IF EXISTS games')
    cursor.execute('DROP TABLE IF EXISTS play_by_play') # Deprecated locally
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            game_date TEXT,
            matchup TEXT,
            season TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS box_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season TEXT,
            game_id TEXT,
            game_date TEXT,
            team_id INTEGER,
            player_id INTEGER,
            player_name TEXT,
            minutes TEXT,
            fgm INTEGER,
            fga INTEGER,
            fg3m INTEGER,
            fg3a INTEGER,
            ftm INTEGER,
            fta INTEGER,
            oreb INTEGER,
            dreb INTEGER,
            reb INTEGER,
            ast INTEGER,
            stl INTEGER,
            blk INTEGER,
            tov INTEGER,
            pf INTEGER,
            pts INTEGER,
            plus_minus INTEGER
        )
    ''')
    conn.commit()

def run_fast_backfiller(seasons):
    print(f"[+] Starting High-Speed Bulk Backfill for {len(seasons)} seasons")
    conn = sqlite3.connect(DB_PATH)
    setup_db(conn)

    for season in seasons:
        print(f"[+] Fetching player game logs for season {season}...")
        try:
            # 1 single API call = ~30,000 player box scores
            logs = playergamelogs.PlayerGameLogs(season_nullable=season).get_data_frames()[0]
            print(f"    -> Extracted {len(logs)} Box Scores in bulk.")
            
            # Insert into games table (Unique game_id)
            games_df = logs[['GAME_ID', 'GAME_DATE', 'MATCHUP', 'SEASON_YEAR']].drop_duplicates(subset=['GAME_ID'])
            games_df = games_df.rename(columns={
                'GAME_ID': 'game_id',
                'GAME_DATE': 'game_date',
                'MATCHUP': 'matchup',
                'SEASON_YEAR': 'season'
            })
            games_df.to_sql('games', conn, if_exists='append', index=False)
            
            # Handle float vs MM:SS minutes column format safely
            if 'MIN' in logs.columns:
                logs['minutes'] = logs['MIN'] 
            
            # Map columns for box_scores
            box_scores_df = logs.rename(columns={
                'SEASON_YEAR': 'season',
                'GAME_ID': 'game_id',
                'GAME_DATE': 'game_date',
                'TEAM_ID': 'team_id',
                'PLAYER_ID': 'player_id',
                'PLAYER_NAME': 'player_name',
                'FGM': 'fgm',
                'FGA': 'fga',
                'FG3M': 'fg3m',
                'FG3A': 'fg3a',
                'FTM': 'ftm',
                'FTA': 'fta',
                'OREB': 'oreb',
                'DREB': 'dreb',
                'REB': 'reb',
                'AST': 'ast',
                'STL': 'stl',
                'BLK': 'blk',
                'TOV': 'tov',
                'PF': 'pf',
                'PTS': 'pts',
                'PLUS_MINUS': 'plus_minus'
            })
            
            cols_to_keep = ['season', 'game_id', 'game_date', 'team_id', 'player_id', 'player_name', 'minutes',
                            'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta', 'oreb', 'dreb', 'reb', 
                            'ast', 'stl', 'blk', 'tov', 'pf', 'pts', 'plus_minus']
            
            # Only keep the columns that actually exist to avoid crash if API changed
            cols_to_keep = [c for c in cols_to_keep if c in box_scores_df.columns]
            
            # Save to SQLite
            box_scores_df[cols_to_keep].to_sql('box_scores', conn, if_exists='append', index=False)
            
            print(f"    -> Successfully appended to SQLite Database.")
            time.sleep(1) # Slight Politeness delay
            
        except Exception as e:
            print(f"[-] Error parsing {season}: {e}")
            
    conn.close()
    print("\\n[SUCCESS] Fast Backfill Complete! Your custom DB now contains ~7 years of complete data.")

if __name__ == '__main__':
    print("================================================================")
    print(" WARNING: DANGER ZONE")
    print("================================================================")
    print("This script will WIPE your entire existing database and")
    print("rebuild it from scratch by downloading the last 7 seasons.")
    print("Make sure this is what you want to do!")
    print("================================================================")
    
    confirm = input("\nType 'YES' (all caps) to proceed, or anything else to abort: ")
    if confirm != 'YES':
        print("\nAborted. Your database was not modified.")
        exit()

    # The last 7 years of NBA data, downloaded in around 30 seconds total.
    seasons = [
        '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26'
    ]
    run_fast_backfiller(seasons)
