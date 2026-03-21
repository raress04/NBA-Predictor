import sqlite3
import pandas as pd
import time
import os
import sys
import argparse
from datetime import datetime, timedelta
from nba_api.stats.endpoints import playergamelogs

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(MODEL_DIR)

from etl import analyze_predictions

DB_PATH = os.path.join(MODEL_DIR, 'database', 'nba_data.db')

def fetch_yesterday(target_date_str=None):
    if target_date_str:
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
    else:
        target_date = datetime.now() - timedelta(days=1)
        
    yesterday_str = target_date.strftime('%m/%d/%Y')  # Format MM/DD/YYYY for NBA API
    db_date_str = target_date.strftime('%Y-%m-%d')    # Format YYYY-MM-DD for standard DB

    
    print(f"[+] Fetching player game logs for yesterday ({yesterday_str})...")
    
    conn = sqlite3.connect(DB_PATH)
    try:
        # Fetch logs specifically for yesterday (with retries for API timeout)
        logs = None
        for attempt in range(3):
            try:
                logs = playergamelogs.PlayerGameLogs(
                    season_nullable='2025-26',
                    date_from_nullable=yesterday_str,
                    date_to_nullable=yesterday_str,
                    timeout=30  # generous timeout
                ).get_data_frames()[0]
                break
            except Exception as e:
                print(f"    [!] API Timeout/Error on attempt {attempt+1}: {e}")
                time.sleep(2)
        
        if logs is None:
            print("    -> Failed to fetch logs after 3 attempts.")
            return

        if len(logs) == 0:
            print("    -> No games/box scores found for yesterday.")
            return

        print(f"    -> Extracted {len(logs)} Box Scores.")
        
        # Insert into games table (Unique game_id)
        games_df = logs[['GAME_ID', 'GAME_DATE', 'MATCHUP', 'SEASON_YEAR']].drop_duplicates(subset=['GAME_ID'])
        games_df = games_df.rename(columns={
            'GAME_ID': 'game_id',
            'GAME_DATE': 'game_date',
            'MATCHUP': 'matchup',
            'SEASON_YEAR': 'season'
        })
        
        cursor = conn.cursor()
        for _, row in games_df.iterrows():
            cursor.execute(
                "INSERT OR IGNORE INTO games (game_id, game_date, matchup, season) VALUES (?, ?, ?, ?)",
                (row['game_id'], row['game_date'], row['matchup'], row['season'])
            )
        
        # Format minutes
        if 'MIN' in logs.columns:
            logs['minutes'] = logs['MIN'] 
        
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
        cols_to_keep = [c for c in cols_to_keep if c in box_scores_df.columns]
        
        # Add index to dramatically speed up the SELECT 1 lookup below
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bs_game_player ON box_scores(game_id, player_id);")

        inserted_count = 0
        for _, row in box_scores_df.iterrows():
            # Check if this player log for this game already exists
            cursor.execute(
                "SELECT 1 FROM box_scores WHERE game_id = ? AND player_id = ?",
                (row['game_id'], row['player_id'])
            )
            if not cursor.fetchone():
                vals = [row[c] for c in cols_to_keep]
                placeholders = ",".join(["?"] * len(cols_to_keep))
                cursor.execute(
                    f"INSERT INTO box_scores ({','.join(cols_to_keep)}) VALUES ({placeholders})",
                    vals
                )
                inserted_count += 1
                
        conn.commit()
        print(f"    -> Successfully inserted {inserted_count} new box scores to SQLite Database.")
        
        # Write results to date-specific file if provided, otherwise default
        month_name = target_date.strftime('%B').lower()
        filename = f'yesterday_results_{db_date_str}.txt' if target_date_str else 'yesterday_results.txt'
        
        # Always save in results/month_folder
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', month_name)
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, filename)
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write(f"--- YESTERDAY'S RESULTS ({yesterday_str}) ---\n\n")
            if 'MATCHUP' in logs.columns and len(logs) > 0:
                # To group by unique games properly instead of each team's view of the matchup
                # (since 'MATCHUP' will be "ATL vs. BOS" for ATL players and "BOS @ ATL" for BOS players),
                # we group by GAME_ID to get all players in the same game block.
                grouped = logs.groupby('GAME_ID')
                for game_id, group in grouped:
                    home_team_matchups = group[group['MATCHUP'].str.contains(' vs. ', na=False)]
                    if len(home_team_matchups) > 0:
                        matchup = home_team_matchups.iloc[0]['MATCHUP']
                    else:
                        matchup = group.iloc[0]['MATCHUP']
                        
                    # Calculate team scores directly from player logs
                    team_scores = group.groupby('TEAM_ABBREVIATION')['PTS'].sum()
                    if len(team_scores) == 2:
                        if ' vs. ' in matchup:
                            home, away = matchup.split(' vs. ')
                        elif ' @ ' in matchup:
                            away, home = matchup.split(' @ ')
                        else:
                            teams = list(team_scores.index)
                            home, away = teams[0], teams[1]
                            
                        # Format mathematically aligned with analyze_predictions.py parser
                        score_home = team_scores.get(home, 0)
                        score_away = team_scores.get(away, 0)
                        matchup_str = f"{home} vs. {away}    {score_home} - {score_away}"
                    else:
                        matchup_str = matchup

                    f.write(f"{'='*50}\n")
                    f.write(f"MATCHUP: {matchup_str}\n")
                    f.write(f"{'='*50}\n")
                    for _, player in group.sort_values('PTS', ascending=False).iterrows():
                        pts = player.get('PTS', 0)
                        reb = player.get('REB', 0)
                        ast = player.get('AST', 0)
                        fg3m = player.get('FG3M', 0)
                        name = player.get('PLAYER_NAME', 'Unknown')
                        f.write(f"{name:22} | {pts:2} PTS, {reb:2} REB, {ast:2} AST, {fg3m:2} 3PM\n")
                    f.write("\n")
            else:
                f.write("No matchup data found.\n")
        print(f"[+] Output written to {results_path}")
        
    except Exception as e:
        print(f"[-] Error fetching yesterday's logs: {e}")
        
    conn.close()
    
    # ── Run the automated prediction analysis module (ONLY if not historical) ──
    if not target_date_str:
        print("\n")
        predict_filename = 'predicts.txt'
        predict_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), predict_filename)
        analyze_predictions.generate_analysis(predicts_path=predict_path, results_path=results_path, write_db=True)

    # ── Phase 5.D: Resolve shadow_picks outcomes for this date ────────────────
    try:
        import sqlite3 as _sqlite3
        resolve_date = db_date_str if target_date_str else yesterday_str
        box_scores = {}  # player -> {POINTS, REBOUNDS, ASSISTS}
        if logs is not None and len(logs) > 0:
            for _, row in logs.iterrows():
                name = row.get('PLAYER_NAME', '')
                if name:
                    box_scores[name] = {
                        'POINTS': int(row.get('PTS', 0)),
                        'REBOUNDS': int(row.get('REB', 0)),
                        'ASSISTS': int(row.get('AST', 0)),
                    }

        if box_scores:
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'database', 'bet_tracker.db')
            conn_sp = _sqlite3.connect(db_path)
            cur_sp = conn_sp.cursor()
            cur_sp.execute("SELECT id, player, stat_category, direction, line FROM shadow_picks WHERE game_date = ? AND actual_result IS NULL", (resolve_date,))
            rows = cur_sp.fetchall()
            updated = 0
            for sid, player, cat, direction, line in rows:
                actual = box_scores.get(player)
                if not actual or line is None:
                    continue
                actual_val = actual.get(cat)
                if actual_val is None:
                    continue
                is_win = 1 if (direction == 'OVER' and actual_val > line) or (direction == 'UNDER' and actual_val < line) else 0
                cur_sp.execute("UPDATE shadow_picks SET actual_value = ?, actual_result = ? WHERE id = ?", (actual_val, is_win, sid))
                updated += 1
            conn_sp.commit()
            conn_sp.close()
            print(f"[+] Phase 5.D: Resolved {updated} shadow_picks outcomes for {resolve_date}.")

        # ── Phase 5.D: Resolve parlay outcomes for this date ─────────────────
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'database', 'bet_tracker.db')
        conn_p = _sqlite3.connect(db_path)
        cur_p = conn_p.cursor()
        cur_p.execute("SELECT id, stake, combined_odds FROM parlays WHERE game_date = ? AND result IS NULL", (resolve_date,))
        parlays_to_resolve = cur_p.fetchall()
        resolved_parlays = 0
        for parlay_id, stake, combined_odds in parlays_to_resolve:
            cur_p.execute("""
                SELECT pl.id, pr.is_win
                FROM parlay_legs pl
                LEFT JOIN pick_results pr ON pr.pick_id = pl.pick_id
                WHERE pl.parlay_id = ?
            """, (parlay_id,))
            legs = cur_p.fetchall()
            if not legs or any(is_win is None for _, is_win in legs):
                continue
            won = all(is_win == 1 for _, is_win in legs)
            result = 'WIN' if won else 'LOSS'
            payout = round((stake or 0) * (combined_odds or 1.0), 6) if won else 0.0
            roi = round((payout - (stake or 0)) / stake, 4) if stake else 0.0
            cur_p.execute("UPDATE parlays SET result = ?, payout = ?, roi = ? WHERE id = ?", (result, payout, roi, parlay_id))
            resolved_parlays += 1
        conn_p.commit()
        conn_p.close()
        if resolved_parlays:
            print(f"[+] Phase 5.D: Resolved {resolved_parlays} parlay outcomes for {resolve_date}.")
    except Exception as e:
        print(f"[-] Phase 5.D resolution error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fetch box scores for a specific date")
    parser.add_argument('--date', type=str, default=None, help="Target date YYYY-MM-DD")
    args = parser.parse_args()
    fetch_yesterday(args.date)
