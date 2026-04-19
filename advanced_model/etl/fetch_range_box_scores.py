import sqlite3
import pandas as pd
import time
import os
import sys
import argparse
import difflib
from datetime import datetime, timedelta
from nba_api.stats.endpoints import playergamelogs

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(MODEL_DIR)

from etl import analyze_predictions

DB_PATH = os.path.join(MODEL_DIR, 'database', 'nba_data.db')
TRACKER_DB = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')

def resolve_outcomes(date_iso, logs_df):
    """
    Resolves shadow_picks and parlay outcomes for a specific date in bet_tracker.db.
    """
    if logs_df is None or len(logs_df) == 0:
        return

    print(f"    [*] Resolving outcomes for {date_iso}...")
    
    # player -> {POINTS, REBOUNDS, ASSISTS}
    box_scores = {}
    for _, row in logs_df.iterrows():
        name = row.get('PLAYER_NAME', '')
        if name:
            box_scores[name] = {
                'POINTS': int(row.get('PTS', 0)),
                'REBOUNDS': int(row.get('REB', 0)),
                'ASSISTS': int(row.get('AST', 0)),
            }

    try:
        conn = sqlite3.connect(TRACKER_DB)
        cursor = conn.cursor()

        # 1. Resolve shadow_picks (Player Props only)
        cursor.execute("""
            SELECT id, player, stat_category, direction, line 
            FROM shadow_picks 
            WHERE game_date = ? 
              AND actual_result IS NULL
              AND stat_category NOT IN ('TOTAL', 'SPREAD')
        """, (date_iso,))
        
        shadow_rows = cursor.fetchall()
        updated_shadow = 0
        for sid, player, cat, direction, line in shadow_rows:
            actual = box_scores.get(player)
            if not actual:
                # Fuzzy match on player name (cutoff=0.85 avoids false positives)
                close = difflib.get_close_matches(player, box_scores.keys(), n=1, cutoff=0.85)
                if close:
                    actual = box_scores[close[0]]
            if not actual or line is None:
                continue
            actual_val = actual.get(cat.upper())
            if actual_val is None:
                continue
            
            # Logic: OVER means actual > line, UNDER means actual < line
            is_win = 1 if (direction.upper() == 'OVER' and actual_val > line) or \
                          (direction.upper() == 'UNDER' and actual_val < line) else 0
            
            cursor.execute("""
                UPDATE shadow_picks
                SET actual_value = ?, actual_result = ?, resolved_at = datetime('now')
                WHERE id = ?
            """, (actual_val, is_win, sid))
            updated_shadow += 1

        # 2. Resolve parlays
        cursor.execute("SELECT id, stake, combined_odds FROM parlays WHERE game_date = ? AND result IS NULL", (date_iso,))
        parlays = cursor.fetchall()
        updated_parlays = 0
        for pid, stake, odds in parlays:
            # Check if all legs are resolved in pick_results (which is populated by analyze_predictions)
            cursor.execute("""
                SELECT pl.id, pr.is_win
                FROM parlay_legs pl
                LEFT JOIN pick_results pr ON pr.pick_id = pl.pick_id
                WHERE pl.parlay_id = ?
            """, (pid,))
            legs = cursor.fetchall()
            
            if not legs or any(is_win is None for _, is_win in legs):
                continue
                
            won = all(is_win == 1 for _, is_win in legs)
            result = 'WIN' if won else 'LOSS'
            payout = round((stake or 0) * (odds or 1.0), 6) if won else 0.0
            roi = round((payout - (stake or 0)) / stake, 4) if stake and stake > 0 else 0.0
            
            cursor.execute("UPDATE parlays SET result = ?, payout = ?, roi = ? WHERE id = ?", 
                         (result, payout, roi, pid))
            updated_parlays += 1

        conn.commit()
        conn.close()
        if updated_shadow or updated_parlays:
            print(f"    [OK] Resolved {updated_shadow} shadow picks and {updated_parlays} parlays.")

        # Trigger ban re-evaluation every time we cross a multiple of 50 resolved shadow picks
        if updated_shadow > 0:
            try:
                _eval_conn = sqlite3.connect(TRACKER_DB)
                total_resolved = _eval_conn.execute(
                    "SELECT COUNT(*) FROM shadow_picks WHERE actual_result IS NOT NULL"
                ).fetchone()[0]
                _eval_conn.close()
                if (total_resolved // 50) > ((total_resolved - updated_shadow) // 50):
                    evaluate_ban_status()
            except Exception:
                pass
    except Exception as e:
        print(f"    [!] Error resolving outcomes: {e}")


def evaluate_ban_status():
    """
    For each banned category+direction, prints a recommendation if shadow win rate > 52.5%.
    Never auto-lifts bans — always requires human confirmation.
    Run monthly or automatically after every 50 new shadow picks resolved.
    """
    try:
        conn = sqlite3.connect(TRACKER_DB)
        rows = conn.execute("""
            SELECT stat_category, direction,
                   COUNT(*) as n,
                   ROUND(AVG(actual_result) * 100, 1) as win_rate_pct
            FROM shadow_picks
            WHERE actual_result IS NOT NULL
              AND ban_reason = 'BANNED_CATEGORY'
            GROUP BY stat_category, direction
            HAVING COUNT(*) >= 50
            ORDER BY win_rate_pct DESC
        """).fetchall()
        conn.close()

        if not rows:
            return

        print("\n[BAN STATUS REVIEW]")
        for cat, direction, n, wr in rows:
            if wr > 52.5:
                print(f"  ⚠️  LIFT CANDIDATE: {cat} {direction} — {wr}% win rate (n={n}) — consider lifting ban")
            else:
                print(f"  ✅  BAN JUSTIFIED:  {cat} {direction} — {wr}% win rate (n={n})")
    except Exception as e:
        print(f"    [!] evaluate_ban_status error: {e}")

def save_daily_results_file(date_dt, logs_df):
    """
    Generates the human-readable yesterday_results_YYYY-MM-DD.txt file.
    """
    iso_date = date_dt.strftime('%Y-%m-%d')
    month_name = date_dt.strftime('%B').lower()
    
    results_dir = os.path.join(MODEL_DIR, 'results', month_name)
    os.makedirs(results_dir, exist_ok=True)
    filename = f'yesterday_results_{iso_date}.txt'
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"--- RESULTS FOR {iso_date} ---\n\n")
        if logs_df is not None and not logs_df.empty:
            grouped = logs_df.groupby('GAME_ID')
            for _, group in grouped:
                matchup = group.iloc[0]['MATCHUP']
                
                # Attempt to get score-formatted matchup
                team_scores = group.groupby('TEAM_ABBREVIATION')['PTS'].sum()
                if len(team_scores) == 2:
                    t1, t2 = list(team_scores.index)
                    matchup_str = f"{t1} @ {t2} | {team_scores[t1]} - {team_scores[t2]}"
                else:
                    matchup_str = matchup

                f.write(f"{'='*50}\n")
                f.write(f"MATCHUP: {matchup_str}\n")
                f.write(f"{'='*50}\n")
                
                for _, player in group.sort_values('PTS', ascending=False).iterrows():
                    name = player.get('PLAYER_NAME', 'Unknown')
                    pts = int(player.get('PTS', 0))
                    reb = int(player.get('REB', 0))
                    ast = int(player.get('AST', 0))
                    f.write(f"{name:25} | {pts:2} PTS, {reb:2} REB, {ast:2} AST\n")
                f.write("\n")
        else:
            f.write("No box score data found.\n")
    return filepath

def sync_range(start_date_str, end_date_str, run_analysis=False):
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    current = start_date
    conn = sqlite3.connect(DB_PATH)
    
    while current <= end_date:
        ds_nba = current.strftime('%m/%d/%Y')
        ds_db = current.strftime('%Y-%m-%d')
        print(f"\n[*] Syncing Date: {ds_db}")
        
        logs = None
        for attempt in range(3):
            try:
                logs = playergamelogs.PlayerGameLogs(
                    season_nullable='2025-26',
                    date_from_nullable=ds_nba,
                    date_to_nullable=ds_nba,
                    timeout=30
                ).get_data_frames()[0]
                break
            except Exception as e:
                print(f"    [!] nba_api Error: {e}")
                time.sleep(2)
        
        if logs is not None and len(logs) > 0:
            print(f"    [+] Fetched {len(logs)} box scores.")
            
            # 1. Save to box_scores table
            logs_rename = logs.rename(columns={
                'SEASON_YEAR': 'season', 'GAME_ID': 'game_id', 'GAME_DATE': 'game_date',
                'TEAM_ID': 'team_id', 'PLAYER_ID': 'player_id', 'PLAYER_NAME': 'player_name',
                'PTS': 'pts', 'REB': 'reb', 'AST': 'ast', 'STL': 'stl', 'BLK': 'blk',
                'TOV': 'tov', 'PF': 'pf', 'FGM': 'fgm', 'FGA': 'fga', 'FG3M': 'fg3m',
                'FG3A': 'fg3a', 'FTM': 'ftm', 'FTA': 'fta', 'OREB': 'oreb', 'DREB': 'dreb',
                'PLUS_MINUS': 'plus_minus', 'MIN': 'minutes'
            })
            
            cols = ['season', 'game_id', 'game_date', 'team_id', 'player_id', 
                    'player_name', 'minutes', 'fgm', 'fga', 'fg3m', 'fg3a', 
                    'ftm', 'fta', 'oreb', 'dreb', 'reb', 'ast', 'stl', 
                    'blk', 'tov', 'pf', 'pts', 'plus_minus']
            
            cursor = conn.cursor()
            for _, row in logs_rename[cols].iterrows():
                cursor.execute(f"INSERT OR REPLACE INTO box_scores ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})", tuple(row[cols]))
            
            # 2. Update games table
            games_df = logs[['GAME_ID', 'GAME_DATE', 'MATCHUP', 'SEASON_YEAR']].drop_duplicates(subset=['GAME_ID'])
            for _, row in games_df.iterrows():
                cursor.execute("INSERT OR IGNORE INTO games (game_id, game_date, matchup, season) VALUES (?, ?, ?, ?)",
                             (row['GAME_ID'], row['GAME_DATE'], row['MATCHUP'], row['SEASON_YEAR']))
            conn.commit()
            print(f"    [OK] DB update complete.")

            # 3. Save human-readable results file
            results_path = save_daily_results_file(current, logs)
            print(f"    [OK] Results saved to {os.path.basename(results_path)}")

            # 4. Run Analyze Predictions (Hits/Losses)
            if run_analysis:
                print(f"    [*] Running prediction analysis...")
                # Search for prediction file for this date
                month_sub = current.strftime('%B').lower()
                pred_file = os.path.join(MODEL_DIR, 'predictions', month_sub, f'predicts_{ds_db}.txt')
                if not os.path.exists(pred_file):
                    # Try legacy root folder predicts.txt (only if it matches this date)
                    pred_file = os.path.join(MODEL_DIR, 'predicts.txt')
                
                if os.path.exists(pred_file):
                    analyze_predictions.generate_analysis(predicts_path=pred_file, results_path=results_path, write_db=True)
                    print(f"    [OK] Hit rates updated in pick_results.")
                else:
                    print(f"    [!] No prediction file found for {ds_db}. Skipping analysis.")

            # 5. Resolve Shadow Picks & Parlays
            resolve_outcomes(ds_db, logs)

        else:
            print(f"    [!] No data found for this date.")
            
        current += timedelta(days=1)
        if current <= end_date:
            time.sleep(1.5) # Avoid API rate limits
            
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synchronize box scores and resolve outcomes for a range of dates.")
    parser.add_argument('start', type=str, nargs='?', default=None, help="Start date YYYY-MM-DD (defaults to yesterday)")
    parser.add_argument('end', type=str, nargs='?', default=None, help="End date YYYY-MM-DD (defaults to yesterday)")
    
    args = parser.parse_args()
    
    if not args.start:
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        sync_range(yesterday, yesterday)
    elif not args.end:
        sync_range(args.start, args.start)
    else:
        sync_range(args.start, args.end)
