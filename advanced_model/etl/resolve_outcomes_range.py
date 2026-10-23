import os
import sys
import argparse
import sqlite3
import pandas as pd
import subprocess
from datetime import datetime, timedelta

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(MODEL_DIR)

from etl.fetch_range_box_scores import resolve_outcomes

NBA_DB = os.path.join(MODEL_DIR, 'database', 'nba_data.db')
TRACKER_DB = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')
LOGS_DIR = os.path.join(MODEL_DIR, 'logs')

def process_date(date_iso, dry_run, log_file, mode='live', force=False):
    nba_conn = sqlite3.connect(NBA_DB)
    tracker_conn = sqlite3.connect(TRACKER_DB)
    
    # GUARD 1: Check nbadata.db
    c_nba = nba_conn.cursor()
    c_nba.execute("SELECT COUNT(*) FROM box_scores WHERE game_date LIKE ?", (date_iso + '%',))
    box_scores_count = c_nba.fetchone()[0]
    
    if box_scores_count == 0:
        log_line = f"[{date_iso}] SKIP — no box scores in nbadata.db"
        print(log_line)
        log_file.write(log_line + "\n")
        nba_conn.close()
        tracker_conn.close()
        return 'SKIP', 0, 0, 0
        
    # GUARD 2: Check bettracker.db
    c_trk = tracker_conn.cursor()
    c_trk.execute("""
        SELECT COUNT(*) FROM daily_picks dp
        LEFT JOIN pick_results pr ON dp.id = pr.pick_id
        WHERE dp.date = ?
          AND dp.synthetic_flag IS NULL
          AND pr.pick_id IS NULL
    """, (date_iso,))
    picks_pending = c_trk.fetchone()[0]
    
    if picks_pending == 0 and not force:
        log_line = f"[{date_iso}] SKIP — already fully resolved"
        print(log_line)
        log_file.write(log_line + "\n")
        nba_conn.close()
        tracker_conn.close()
        return 'SKIP_RESOLVED', box_scores_count, 0, 0

    # BUILD logs_df
    query = """
        SELECT 
            b.player_name AS PLAYER_NAME, 
            t.abbreviation AS TEAM_ABBREVIATION, 
            CAST(b.pts AS INTEGER) AS PTS,
            CAST(b.reb AS INTEGER) AS REB, 
            CAST(b.ast AS INTEGER) AS AST,
            b.game_id AS GAME_ID,
            g.matchup AS MATCHUP
        FROM box_scores b
        LEFT JOIN teams t ON b.team_id = t.team_id
        LEFT JOIN games g ON b.game_id = g.game_id
        WHERE b.game_date LIKE ?
    """
    logs_df = pd.read_sql(query, nba_conn, params=(date_iso + '%',))
    
    # BUILD yesterday_results.txt using centralized paths
    from utils.prediction_paths import get_results_filepath
    results_path = get_results_filepath(date_iso, mode=mode, ext='txt', base_dir=MODEL_DIR)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    games = logs_df.groupby('GAME_ID')
    results_text = []
    
    for game_id, group in games:
        # Extract Matchup safely
        matchup_str = group['MATCHUP'].iloc[0] if 'MATCHUP' in group.columns and pd.notnull(group['MATCHUP'].iloc[0]) else ""
        
        if ' @ ' in matchup_str:
            away_team, home_team = [x.strip() for x in matchup_str.split(' @ ')]
        elif ' vs. ' in matchup_str:
            home_team, away_team = [x.strip() for x in matchup_str.split(' vs. ')]
        else:
            # Fallback if matchup format is unrecognized or missing
            team_abbrevs = group['TEAM_ABBREVIATION'].dropna().unique()
            if len(team_abbrevs) >= 2:
                away_team, home_team = team_abbrevs[0], team_abbrevs[1]
            else:
                continue

        away_pts = group[group['TEAM_ABBREVIATION'] == away_team]['PTS'].sum()
        home_pts = group[group['TEAM_ABBREVIATION'] == home_team]['PTS'].sum()
        
        results_text.append(f"MATCHUP: {away_team} vs. {home_team}    {away_pts} - {home_pts}\n")
        
        sorted_players = group.sort_values(by='PTS', ascending=False)
        for _, p in sorted_players.iterrows():
            player = p['PLAYER_NAME']
            pts = p['PTS']
            reb = p['REB']
            ast = p['AST']
            results_text.append(f"{player:<22} | {pts} PTS, {reb} REB,  {ast} AST\n")
        results_text.append("\n")

    if dry_run:
        num_matchups = len(games)
        num_players = len(logs_df)
        log_line = f"[{date_iso}] DRY-RUN — would write {num_matchups} game matchups, {num_players} player lines, resolve {picks_pending} picks"
        print(log_line)
        log_file.write(log_line + "\n")
        nba_conn.close()
        tracker_conn.close()
        return 'DRY_RUN', box_scores_count, picks_pending, 0

    # Write the file permanently 
    with open(results_path, 'w', encoding='utf-8') as f:
        f.writelines(results_text)
        
    error_flag = False
    
    # CALL resolve_outcomes
    try:
        resolve_outcomes(date_iso, logs_df)
    except Exception as e:
        log_file.write(f"[{date_iso}] ERROR in resolve_outcomes: {e}\n")
        error_flag = True
        
    # CALL analyze_predictions using subprocess (Fix 1 applied)
    try:
        from utils.prediction_paths import resolve_prediction_file
        pred_file = resolve_prediction_file(date_iso, mode='live', base_dir=MODEL_DIR, allow_legacy=True)
        if not pred_file:
            pred_file = resolve_prediction_file(date_iso, mode='backtest', base_dir=MODEL_DIR, allow_legacy=True)
        if not pred_file or not os.path.exists(pred_file):
            pred_file = os.path.join(MODEL_DIR, 'predicts.txt')
            
        result = subprocess.run(
            [sys.executable, "-m", "etl.analyze_predictions",
             "--input", pred_file,
             "--actuals", results_path,
             "--write-db"],
            cwd=MODEL_DIR,
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
    except Exception as e:
        log_file.write(f"[{date_iso}] ERROR in analyze_predictions: {e}\n")
        error_flag = True

    # LOG result line
    err_str = 'YES' if error_flag else 'NO'
    log_line = f"[{date_iso}] OK boxscores={box_scores_count} | picks_pending={picks_pending} | resolved={picks_pending} | errors={err_str}"
    print(log_line)
    log_file.write(log_line + "\n")
    
    nba_conn.close()
    tracker_conn.close()
    
    return 'SUCCESS', box_scores_count, picks_pending, 1 if error_flag else 0

def main():
    parser = argparse.ArgumentParser(description="Resolve outcomes safely without API calls.")
    parser.add_argument('--start-date', required=True, help='YYYY-MM-DD')
    parser.add_argument('--end-date', required=True, help='YYYY-MM-DD')
    parser.add_argument('--mode', choices=['live', 'backtest'], default='live', help='Mode: live or backtest')
    parser.add_argument('--force', action='store_true', help='Force regenerate results and run analysis even if resolved')
    parser.add_argument('--dry-run', action='store_true', help='Dry run only')
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')

    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOGS_DIR, f'resolve_outcomes_range_{timestamp}.log')
    
    stats = {
        'processed': 0,
        'skipped_no_data': 0,
        'skipped_resolved': 0,
        'resolved': 0,
        'errors': 0
    }
    
    current_date = start_date
    with open(log_path, 'w', encoding='utf-8') as log_file:
        while current_date <= end_date:
            date_iso = current_date.strftime('%Y-%m-%d')
            status, bs_count, picks_pending, has_error = process_date(date_iso, args.dry_run, log_file, mode=args.mode, force=args.force)
            
            stats['processed'] += 1
            if status == 'SKIP':
                stats['skipped_no_data'] += 1
            elif status == 'SKIP_RESOLVED':
                stats['skipped_resolved'] += 1
            elif status in ('SUCCESS', 'DRY_RUN'):
                stats['resolved'] += picks_pending
                stats['errors'] += has_error
                
            current_date += timedelta(days=1)
            
        summary = f"""
================================================
resolve_outcomes_range COMPLETE
Date range      : {args.start_date} to {args.end_date}
Dates processed : {stats['processed']}
Dates skipped   : {stats['skipped_no_data'] + stats['skipped_resolved']} (no box scores: {stats['skipped_no_data']}, already resolved: {stats['skipped_resolved']})
Picks resolved  : {stats['resolved']}
Errors          : {stats['errors']}
Run mode        : {'DRY-RUN' if args.dry_run else 'LIVE'}
================================================
"""
        print(summary)
        log_file.write(summary)

if __name__ == '__main__':
    main()
