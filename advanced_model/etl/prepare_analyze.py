"""
etl/prepare_yesterday.py
========================
Idempotent ETL preparation step for yesterday's results and prediction analysis.

Ensures that before the main model and prediction run execute:
1. Yesterday's completed-game box scores exist (or are fetched).
2. Yesterday's results file (yesterday_results_YYYY-MM-DD.txt) exists and is valid.
3. Yesterday's analysis report (analyze_report_YYYY-MM-DD.pdf) exists and is valid.

CLI:
    python -m etl.prepare_yesterday
    python -m etl.prepare_yesterday --date YYYY-MM-DD
    python -m etl.prepare_yesterday --date YYYY-MM-DD --mode [live|backtest]
    python -m etl.prepare_yesterday --force
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import re
import argparse
import sqlite3
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
import pandas as pd

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from utils.prediction_paths import (
    get_nba_season,
    get_results_filepath,
    get_results_dir,
    get_analyze_report_filepath,
    get_analyze_dir,
    resolve_prediction_file,
    resolve_results_file,
    resolve_analyze_file,
)
from etl.fetch_range_box_scores import resolve_outcomes
from etl.analyze_predictions import generate_evaluate, parse_predicts, parse_results

NBA_DB = os.path.join(MODEL_DIR, 'database', 'nba_data.db')
TRACKER_DB = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')


def compute_target_date(date_str: Optional[str] = None) -> str:
    """
    Computes and validates target date string (YYYY-MM-DD).
    If date_str is provided, validates format.
    Otherwise, returns the previous US-Eastern slate date (utils.slate_date).
    """
    if date_str:
        try:
            dt = datetime.strptime(date_str.strip(), '%Y-%m-%d')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Invalid date format '{date_str}'. Expected YYYY-MM-DD.")
    from utils.slate_date import previous_slate_date
    return previous_slate_date()


def validate_results_file(filepath: Optional[str], target_date: str) -> Tuple[bool, str]:
    """
    Validates yesterday_results_YYYY-MM-DD.txt:
    - File exists and is non-empty (>0 bytes)
    - Date in filename matches target_date (if present in name)
    - Contains at least 1 MATCHUP line (e.g. 'MATCHUP: CLE vs. PHI 106 - 114')
    - Contains at least 1 valid player stat row ('Player | PTS PTS, REB REB, AST AST')
    - Does not contain placeholder error text
    """
    if not filepath or not os.path.exists(filepath):
        return False, f"File does not exist: {filepath}"

    if os.path.getsize(filepath) == 0:
        return False, f"File is empty (0 bytes): {filepath}"

    basename = os.path.basename(filepath)
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', basename)
    if date_match and date_match.group(1) != target_date:
        return False, f"Date mismatch in filename '{basename}' vs target '{target_date}'"

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return False, f"Cannot read results file: {e}"

    if not content.strip():
        return False, "Results file content is whitespace only"

    if "[!] No data" in content or "[-] ERROR" in content or "ERROR in" in content:
        return False, "Results file contains error placeholder text"

    actual_games, actual_props = parse_results(filepath)
    if not actual_games:
        return False, "No valid MATCHUP headers found in results file"
    if not actual_props:
        return False, "No valid player stat lines found in results file"

    return True, "Valid results file"


def validate_analysis_file(filepath: Optional[str], target_date: str) -> Tuple[bool, str]:
    """
    Validates analyze_report_YYYY-MM-DD.pdf:
    - File exists
    - File size is reasonable for compiled PDF (>= 5000 bytes)
    - Starts with PDF magic bytes (%PDF-)
    - Date in filename matches target_date
    """
    if not filepath or not os.path.exists(filepath):
        return False, f"Analysis file does not exist: {filepath}"

    file_size = os.path.getsize(filepath)
    if file_size < 5000:
        return False, f"Analysis file suspiciously small ({file_size} bytes, min 5000 bytes)"

    basename = os.path.basename(filepath)
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', basename)
    if date_match and date_match.group(1) != target_date:
        return False, f"Date mismatch in analysis filename '{basename}' vs target '{target_date}'"

    try:
        with open(filepath, 'rb') as f:
            header = f.read(1024)
        if b"%PDF-" not in header:
            return False, "File lacks valid %PDF- magic header"
    except Exception as e:
        return False, f"Cannot read analysis file: {e}"

    return True, "Valid analysis PDF report"


def validate_prediction_file(filepath: Optional[str], target_date: str) -> Tuple[bool, str]:
    """
    Validates prediction file (predicts_YYYY-MM-DD.txt):
    - File exists and is non-empty
    - Contains at least 1 matchup or projected player/score
    """
    if not filepath or not os.path.exists(filepath):
        return False, f"Prediction file does not exist: {filepath}"

    if os.path.getsize(filepath) == 0:
        return False, f"Prediction file is empty (0 bytes): {filepath}"

    try:
        games, props, medians = parse_predicts(filepath)
    except Exception as e:
        return False, f"Failed to parse prediction file: {e}"

    if not games and not props and not medians:
        return False, "Prediction file contains no matchups, props, or medians"

    return True, "Valid prediction file"


def check_games_and_box_scores(target_date: str, db_path: Optional[str] = None) -> Tuple[int, int, List[str]]:
    """
    Queries nba_data.db for scheduled games and box scores for target_date.
    Returns: (games_count, box_scores_count, matchups_list)
    """
    db = db_path or NBA_DB
    if not os.path.exists(db):
        return 0, 0, []

    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT game_id, matchup FROM games WHERE game_date LIKE ?", (f"{target_date}%",))
    game_rows = cur.fetchall()
    games_count = len(game_rows)
    matchups = [r[1] for r in game_rows if r[1]]

    cur.execute("SELECT COUNT(*) FROM box_scores WHERE game_date LIKE ?", (f"{target_date}%",))
    box_scores_count = cur.fetchone()[0]

    conn.close()
    return games_count, box_scores_count, matchups


def fetch_box_scores_if_needed(target_date: str, db_path: Optional[str] = None) -> bool:
    """
    Ensures box scores for target_date are in nba_data.db.
    If already present, returns True immediately.
    If missing, fetches from NBA API using playergamelogs and stores them.
    """
    db = db_path or NBA_DB
    _, existing_box_count, _ = check_games_and_box_scores(target_date, db_path=db)
    if existing_box_count > 0:
        return True

    print(f"[*] Box scores for {target_date} missing from DB. Fetching from NBA API...")
    try:
        from nba_api.stats.endpoints import playergamelogs
        season_full = get_nba_season(target_date)  # e.g. '2026-2027'
        parts = season_full.split('-')
        season_api = f"{parts[0]}-{parts[1][-2:]}"  # e.g. '2026-27'

        logs_endpoint = playergamelogs.PlayerGameLogs(season_nullable=season_api)
        dfs = logs_endpoint.get_data_frames()
        if not dfs or dfs[0].empty:
            print(f"[!] NBA API returned no game logs for season {season_api}.")
            return False

        logs = dfs[0]
        # Filter for rows matching target_date
        date_logs = logs[logs['GAME_DATE'].astype(str).str.startswith(target_date)]
        if date_logs.empty:
            print(f"[!] NBA API has no completed game logs for {target_date} yet.")
            return False

        conn = sqlite3.connect(db)
        existing_games = set(
            r[0] for r in conn.execute("SELECT game_id FROM games WHERE season = ?", (season_api,)).fetchall()
        )

        new_games_df = date_logs[['GAME_ID', 'GAME_DATE', 'MATCHUP', 'SEASON_YEAR']].drop_duplicates(subset=['GAME_ID'])
        new_games_to_insert = new_games_df[~new_games_df['GAME_ID'].isin(existing_games)].rename(columns={
            'GAME_ID': 'game_id',
            'GAME_DATE': 'game_date',
            'MATCHUP': 'matchup',
            'SEASON_YEAR': 'season'
        })
        if not new_games_to_insert.empty:
            new_games_to_insert.to_sql('games', conn, if_exists='append', index=False)

        if 'MIN' in date_logs.columns:
            date_logs = date_logs.copy()
            date_logs['minutes'] = date_logs['MIN']

        box_df = date_logs.rename(columns={
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
        # Idempotent insert (CRITICAL_BUGS_FOUND_06.11): re-running a date no longer duplicates rows
        from utils.box_scores_io import insert_box_scores
        n_written = insert_box_scores(conn, box_df, cols)
        conn.commit()
        conn.close()

        print(f"[+] Ingested {n_written} new box score rows for {target_date} "
              f"({len(box_df) - n_written} already present).")
        return True

    except Exception as e:
        print(f"[-] Box score fetch failed: {e}")
        return False


def generate_results_file(
    target_date: str,
    mode: str = 'live',
    base_dir: Optional[str] = None,
    db_path: Optional[str] = None
) -> Optional[str]:
    """
    Generates yesterday_results_YYYY-MM-DD.txt from box_scores in nba_data.db.
    Writes atomically to a temporary file before renaming to target path.
    Also calls resolve_outcomes() to resolve bet_tracker.db shadow picks/parlays.
    Returns path of generated results file on success, None on failure.
    """
    db = db_path or NBA_DB
    base = base_dir or MODEL_DIR

    if not os.path.exists(db):
        print(f"[-] Database not found: {db}")
        return None

    conn = sqlite3.connect(db)
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
    logs_df = pd.read_sql(query, conn, params=(f"{target_date}%",))
    conn.close()

    if logs_df.empty:
        print(f"[-] No box score records found in DB for {target_date}.")
        return None

    results_path = get_results_filepath(target_date, mode=mode, ext='txt', base_dir=base)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    games = logs_df.groupby('GAME_ID')
    results_text = []

    for game_id, group in games:
        matchup_str = group['MATCHUP'].iloc[0] if 'MATCHUP' in group.columns and pd.notnull(group['MATCHUP'].iloc[0]) else ""
        if ' @ ' in matchup_str:
            away_team, home_team = [x.strip() for x in matchup_str.split(' @ ')]
        elif ' vs. ' in matchup_str:
            home_team, away_team = [x.strip() for x in matchup_str.split(' vs. ')]
        else:
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

    # Atomic write pattern: write to .tmp in same directory, validate, then os.replace
    tmp_path = f"{results_path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.writelines(results_text)

        is_valid, reason = validate_results_file(tmp_path, target_date)
        if not is_valid:
            print(f"[-] Generated results failed validation: {reason}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return None

        os.replace(tmp_path, results_path)
        print(f"[+] Results file written atomically: {results_path}")
    except Exception as e:
        print(f"[-] Atomic write failed: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return None

    # Resolve outcomes in bet_tracker.db
    try:
        resolve_outcomes(target_date, logs_df)
    except Exception as e:
        print(f"[!] Warning: resolve_outcomes reported an error: {e}")

    return results_path


def generate_analysis_report(
    target_date: str,
    mode: str = 'live',
    predicts_path: Optional[str] = None,
    results_path: Optional[str] = None,
    base_dir: Optional[str] = None
) -> Optional[str]:
    """
    Generates analyze_report_YYYY-MM-DD.pdf using generate_evaluate().
    Returns path of generated PDF on success, None on failure.
    """
    base = base_dir or MODEL_DIR
    out_dir = get_analyze_dir(target_date, mode=mode, base_dir=base)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[*] Generating analysis report for {target_date} (mode={mode})...")
    try:
        generate_evaluate(
            predicts_path=predicts_path,
            results_path=results_path,
            output_dir=out_dir,
            write_db=True,
            generate_pdf=True,
            clean_txt=True
        )
    except Exception as e:
        print(f"[-] generate_evaluate failed: {e}")
        return None

    report_pdf = resolve_analyze_file(target_date, mode=mode, base_dir=base, allow_legacy=True)
    is_valid, reason = validate_analysis_file(report_pdf, target_date)
    if not is_valid:
        print(f"[-] Generated analysis failed validation: {reason}")
        return None

    print(f"[+] Analysis report successfully generated: {report_pdf}")
    return report_pdf


def prepare_analyze(
    target_date: Optional[str] = None,
    mode: str = 'live',
    force: bool = False,
    base_dir: Optional[str] = None,
    db_path: Optional[str] = None
) -> int:
    """
    Main orchestration step:
    1. Resolve target date (yesterday if omitted).
    2. Check prediction file.
       - If missing and verified no-game day: success no-op (return 0).
       - If missing and games were scheduled/played:
           live     -> WARNING "live record has a gap", skip, return 0 (pipeline continues;
                       backtest predictions are never substituted — H1, 2026-11-06)
           backtest -> actionable failure (return 1): an explicit backtest request
                       for a date with no backtest predictions.
    3. Check results and analysis validity.
    4. Execute required steps according to dependency rules:
       - Both valid and not force: skip (return 0).
       - Results valid, analysis invalid/missing: generate analysis only (return 0).
       - Results invalid/missing: fetch box scores -> generate results -> generate analysis (return 0).
    Returns 0 on success, 1 on failure.
    """
    base = base_dir or MODEL_DIR
    date_iso = compute_target_date(target_date)

    print("==================================================")
    print(f"[+] ETL PREPARATION STEP -- Target Date: {date_iso} (Mode: {mode})")
    print("==================================================")

    # 1. Resolve predictions file
    predicts_file = resolve_prediction_file(date_iso, mode=mode, base_dir=base, allow_legacy=True)
    is_pred_valid = False
    if predicts_file:
        is_pred_valid, pred_reason = validate_prediction_file(predicts_file, date_iso)

    # Check for verified no-game day if predictions are missing or invalid
    games_count, box_count, matchups = check_games_and_box_scores(date_iso, db_path=db_path)

    if not predicts_file or not is_pred_valid:
        if games_count == 0 and box_count == 0:
            print(f"[+] Verified no-game day for {date_iso}: 0 games scheduled and 0 predictions found.")
            print("[+] Successful no-op. Nothing to prepare.")
            return 0

        if mode == 'live':
            # H1 decision (Plans/CRITICAL_BUGS_FOUND_06.11.md, owner chose option B on 2026-11-06):
            # a missed live run must NOT stop tonight's pipeline, and must NOT be filled with
            # backtest predictions (live and backtest are never mixed). Record the gap and move on.
            # Backtest grading for such a date is a separate, explicit run: --mode backtest --date D.
            print(f"[!] WARNING: no live predictions for {date_iso} — live record has a gap "
                  f"({games_count} games were played). Live grading skipped for this date; "
                  f"the pipeline continues.")
            return 0
        else:
            print(f"[-] ACTIONABLE ERROR: Prediction file missing or invalid for {date_iso} (mode={mode}),")
            print(f"    but {games_count} games were recorded ({box_count} box score rows).")
            print("    The pipeline will NEVER fabricate predictions or re-run the model retroactively.")
            print("    Aborting to protect temporal integrity.")
            return 1

    # 2. Check current state of artifacts
    results_file = resolve_results_file(date_iso, mode=mode, base_dir=base, allow_legacy=True)
    analysis_file = resolve_analyze_file(date_iso, mode=mode, base_dir=base, allow_legacy=True)

    results_valid, res_reason = validate_results_file(results_file, date_iso)
    analysis_valid, ana_reason = validate_analysis_file(analysis_file, date_iso)

    print(f"[*] Artifact status for {date_iso}:")
    print(f"    -> Predictions : {'VALID' if is_pred_valid else 'INVALID'} ({predicts_file})")
    print(f"    -> Results     : {'VALID' if results_valid else 'INVALID'} ({res_reason})")
    print(f"    -> Analysis    : {'VALID' if analysis_valid else 'INVALID'} ({ana_reason})")

    # CASE A: Both valid
    if results_valid and analysis_valid and not force:
        print(f"[+] Both results and analysis artifacts exist and are valid.")
        print(f"[+] Reusing existing artifacts. No fetch or regeneration required.")
        # Resolve ml_shadow_picks outcomes for the date being prepared
        try:
            from ml.shadow_logger import resolve_ml_shadow_picks
            resolved_ml = resolve_ml_shadow_picks(date_iso)
            if resolved_ml:
                print(f"[+] ML shadow: resolved {resolved_ml} rows for {date_iso}.")
        except Exception as _e:
            print(f"[!] ML shadow resolution non-fatal error: {_e}")
        return 0

    # CASE B: Results valid, analysis missing or invalid
    if results_valid and (not analysis_valid or force):
        print(f"[+] Results artifact is valid. Generating missing/invalid analysis report only...")
        new_analysis = generate_analysis_report(
            date_iso,
            mode=mode,
            predicts_path=predicts_file,
            results_path=results_file,
            base_dir=base
        )
        if not new_analysis:
            print(f"[-] Analysis report generation failed for {date_iso}.")
            return 1
        print(f"[+] Preparation completed successfully (analysis generated).")
        # Resolve ml_shadow_picks outcomes for the date being prepared
        try:
            from ml.shadow_logger import resolve_ml_shadow_picks
            resolved_ml = resolve_ml_shadow_picks(date_iso)
            if resolved_ml:
                print(f"[+] ML shadow: resolved {resolved_ml} rows for {date_iso}.")
        except Exception as _e:
            print(f"[!] ML shadow resolution non-fatal error: {_e}")
        return 0

    # CASE C: Results missing or invalid
    print(f"[+] Results artifact is missing or invalid. Initiating full preparation chain...")

    # Step C.1: Ensure box scores exist
    has_box_scores = fetch_box_scores_if_needed(date_iso, db_path=db_path)
    if not has_box_scores:
        g_count, b_count, m_list = check_games_and_box_scores(date_iso, db_path=db_path)
        if g_count > 0:
            print(f"[-] PENDING GAMES: {g_count} games scheduled for {date_iso}, but completed box scores unavailable.")
            print(f"    Matchups: {', '.join(m_list)}")
            print("    Exiting nonzero so pipeline can retry when games conclude.")
            return 1
        else:
            print(f"[-] Box scores unavailable for {date_iso}. Aborting.")
            return 1

    # Step C.2: Generate results file
    new_results = generate_results_file(date_iso, mode=mode, base_dir=base, db_path=db_path)
    if not new_results:
        print(f"[-] Failed to generate results file for {date_iso}.")
        return 1

    # Step C.3: Generate analysis report (must be regenerated because results dependency was updated)
    print(f"[+] Generating analysis report using newly prepared results file...")
    new_analysis = generate_analysis_report(
        date_iso,
        mode=mode,
        predicts_path=predicts_file,
        results_path=new_results,
        base_dir=base
    )
    if not new_analysis:
        print(f"[-] Failed to generate analysis report for {date_iso}.")
        return 1

    print(f"[+] Preparation completed successfully for {date_iso}.")
    # Resolve ml_shadow_picks outcomes for the date being prepared
    try:
        from ml.shadow_logger import resolve_ml_shadow_picks
        resolved_ml = resolve_ml_shadow_picks(date_iso)
        if resolved_ml:
            print(f"[+] ML shadow: resolved {resolved_ml} rows for {date_iso}.")
    except Exception as _e:
        print(f"[!] ML shadow resolution non-fatal error: {_e}")
    return 0




# Backward compatibility alias
prepare_yesterday = prepare_analyze

def main():
    parser = argparse.ArgumentParser(
        description="ETL preparation step: Validate or prepare yesterday's results and analysis before running model."
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help="Target date to prepare (YYYY-MM-DD). Defaults to yesterday."
    )
    parser.add_argument(
        '--mode',
        choices=['live', 'backtest'],
        default='live',
        help="Pipeline execution mode (default: live)."
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force regeneration of artifacts even if currently valid."
    )
    args = parser.parse_args()

    try:
        exit_code = prepare_analyze(
            target_date=args.date,
            mode=args.mode,
            force=args.force
        )
        sys.exit(exit_code)
    except Exception as e:
        print(f"[-] Unexpected fatal error in prepare_analyze: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
