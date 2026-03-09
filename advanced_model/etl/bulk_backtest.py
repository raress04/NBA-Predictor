import os
import sys
import argparse
import subprocess
import re
import sqlite3
import shutil
import math
import concurrent.futures
from datetime import datetime, timedelta

# Ensure parent is in path
MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(MODEL_DIR)

def generate_dates(start_date_str, end_date_str):
    """Generate a list of YYYY-MM-DD strings between start and end inclusive."""
    start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    dates = []
    curr = start_dt
    while curr <= end_dt:
        dates.append(curr.strftime('%Y-%m-%d'))
        curr += timedelta(days=1)
    return dates

def get_predicts_file_path(date_str):
    """Determine the predicts.txt path based on the month/year."""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    month_name = dt.strftime('%B').lower()
    
    folder_path = os.path.join(MODEL_DIR, 'predictions', month_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        
    return os.path.join(folder_path, f'predicts_{date_str}.txt')

def run_script(script_path, args):
    """Run a python script via subprocess."""
    cmd = [sys.executable, script_path] + args
    subprocess.run(cmd, check=True)

def get_season_averages(date_str):
    """Query nba_data.db for season-to-date averages for all players."""
    db_path = os.path.join(MODEL_DIR, 'database', 'nba_data.db')
    if not os.path.exists(db_path):
        return {}
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Simple query to get average P/R/A per player up to date_str
    cur.execute("""
        SELECT player_name, 
               AVG(pts) as avg_pts, 
               AVG(reb) as avg_reb, 
               AVG(ast) as avg_ast
        FROM box_scores
        WHERE game_date < ?
        GROUP BY player_name
    """, (f"{date_str}T00:00:00",))
    
    averages = {}
    for row in cur.fetchall():
        p_name, p, r, a = row
        averages[p_name] = {'POINTS': p, 'REBOUNDS': r, 'ASSISTS': a}
    
    conn.close()
    return averages

def process_single_date(date_str, scraper_script, fetch_script):
    """Independent worker function for a single date."""
    print(f"\n[Worker] 🚀 PROCESSING DATE: {date_str}")
    
    predicts_txt = get_predicts_file_path(date_str)
    results_txt = os.path.join(MODEL_DIR, f'yesterday_results_{date_str}.txt')

    # If both files exist, we skip the heavy simulation part
    if not (os.path.exists(predicts_txt) and os.path.exists(results_txt)):
        # 1. Fetch Yesterday's Box Scores
        print(f"  [{date_str}] Fetching Results...")
        try:
            run_script(fetch_script, ['--date', date_str])
        except Exception as e:
            print(f"  [!] Failed fetching results for {date_str}: {e}")

        # 2. Run Simulations
        print(f"  [{date_str}] Running Simulations...")
        try:
            run_script(scraper_script, ['--date', date_str])
        except Exception as e:
            print(f"  [!] Failed simulations for {date_str}: {e}")

        # Move root predicts to folder if needed
        root_predicts = os.path.join(MODEL_DIR, f'predicts_{date_str}.txt')
        if os.path.exists(root_predicts) and not os.path.exists(predicts_txt):
            shutil.move(root_predicts, predicts_txt)

    # 3. Grading & Database Ingestion
    if not os.path.exists(results_txt):
        gen_res = os.path.join(MODEL_DIR, 'yesterday_results.txt')
        if os.path.exists(gen_res):
            shutil.copy(gen_res, results_txt)

    if os.path.exists(predicts_txt) and os.path.exists(results_txt):
        db_path = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')
        conn = sqlite3.connect(db_path, timeout=60)
        conn.execute("PRAGMA busy_timeout = 60000")
        cur = conn.cursor()
        
        # Load Baseline Lines (Market)
        cur.execute("""
            SELECT player_or_team, stat_category, line 
            FROM daily_picks 
            WHERE date IN ('2026-03-03', '2026-03-04', '2026-03-05') 
            AND bet_type = 'prop'
            GROUP BY player_or_team, stat_category
        """)
        baseline_lines = {}
        for row in cur.fetchall():
            p_name, s_cat, l_val = row
            if p_name not in baseline_lines: baseline_lines[p_name] = {}
            baseline_lines[p_name][s_cat] = l_val

        std_averages = get_season_averages(date_str)

        # Parse Predictions & Medians
        medians = []
        curr_game = None
        with open(predicts_txt, 'r', encoding='utf-8') as f:
            for line in f:
                mg = re.search(r'MATCHUP:\s+(.*?)\s+@\s+(.*)', line)
                if mg: curr_game = f"{mg.group(1).strip()} @ {mg.group(2).strip()}"
                mm = re.search(r"^\s*([A-Za-z\s\.\'-]+)\s*->\s*(\d+)\s*PTS,\s*(\d+)\s*REB,\s*(\d+)\s*AST(.*)", line)
                if mm and curr_game:
                    medians.append({
                        'player': mm.group(1).strip(),
                        'game': curr_game,
                        'PTS': int(mm.group(2)),
                        'REB': int(mm.group(3)),
                        'AST': int(mm.group(4)),
                        'extra': mm.group(5)
                    })
        
        # Parse Real Results
        actuals = {}
        with open(results_txt, 'r', encoding='utf-8') as f:
            for line in f:
                sm = re.search(r"^([A-Za-z\s\.\'-]+)\s*\|\s*(\d+)\s*PTS,\s*(\d+)\s*REB,\s*(\d+)\s*AST", line)
                if sm:
                    actuals[sm.group(1).strip()] = {'PTS': int(sm.group(2)), 'REB': int(sm.group(3)), 'AST': int(sm.group(4))}
        
        valid_inserts = []
        stat_types = ['PTS', 'REB', 'AST']
        stat_mapping = {'PTS': 'POINTS', 'REB': 'REBOUNDS', 'AST': 'ASSISTS'}
        
        for m in medians:
            p = m['player']
            if p not in actuals: continue
            
            for s_type in stat_types:
                proj_val = float(m[s_type])
                act_val = float(actuals[p][s_type])
                s_cat = stat_mapping[s_type]
                
                # Confidence Extraction
                file_conf = 0.0
                stat_term = s_cat.lower()
                conf_match = re.search(rf"O/U \d+\.?\d* {stat_term} VALUABLE: (?:Under|Over) \((\d+)% conf", m['extra'])
                if conf_match:
                    file_conf = float(conf_match.group(1))

                # Line Selection Logic
                if p in baseline_lines and s_cat in baseline_lines[p]:
                    line_val = baseline_lines[p][s_cat]
                    source_label = 'baseline'
                    use_strict = False
                elif p in std_averages and s_cat in std_averages[p]:
                    raw_val = std_averages[p][s_cat]
                    v1 = math.floor(raw_val) + 0.5
                    v2 = v1 - 1.0 if raw_val < v1 else v1 + 1.0
                    line_val = v1 if abs(raw_val - v1) <= abs(raw_val - v2) else v2
                    source_label = 'season_avg'
                    use_strict = False
                else:
                    line_val = proj_val
                    source_label = 'median_fallback'
                    use_strict = True
                
                diff = act_val - line_val
                abs_diff = abs(diff)
                is_valuable = False
                if not use_strict:
                    if proj_val > line_val and act_val > line_val: direction, is_win, is_valuable = "OVER", 1, True
                    elif proj_val < line_val and act_val < line_val: direction, is_win, is_valuable = "UNDER", 1, True
                    elif proj_val > line_val and act_val < line_val: direction, is_win, is_valuable = "OVER", 0, True
                    elif proj_val < line_val and act_val > line_val: direction, is_win, is_valuable = "UNDER", 0, True
                else:
                    margin = 3 if s_type == 'PTS' else 2
                    if abs_diff >= margin:
                        direction = "OVER" if act_val > line_val else "UNDER"
                        is_win, is_valuable = 1, True
                        
                if is_valuable:
                    valid_inserts.append({
                        'date': date_str,
                        'bet_type': 'prop',
                        'stat_category': s_cat,
                        'player_or_team': p,
                        'game': m['game'],
                        'direction': direction,
                        'line': line_val,
                        'actual_value': act_val,
                        'is_win': is_win,
                        'diff_val': diff,
                        'diff_pct': (diff / line_val * 100) if line_val > 0 else 0,
                        'source': source_label,
                        'confidence': file_conf
                    })

        # SQLite Insertion
        inserted_count = 0
        for r in valid_inserts:
            cur.execute("""
                SELECT id FROM daily_picks
                WHERE date=? AND player_or_team=? AND stat_category=? AND direction=? AND line=?
            """, (date_str, r['player_or_team'], r['stat_category'], r['direction'], r['line']))
            existing = cur.fetchone()
            
            if existing:
                pick_id = existing[0]
                if r['confidence'] > 0:
                    cur.execute("UPDATE daily_picks SET sim_confidence=? WHERE id=? AND (sim_confidence IS NULL OR sim_confidence=0)", (r['confidence'], pick_id))
            else:
                s_lab = r['source']
                if not s_lab.startswith('bulk_backtest_'): s_lab = f"bulk_backtest_{s_lab}"
                cur.execute("""
                    INSERT INTO daily_picks (date, bet_type, stat_category, player_or_team, game, direction, line, source_file, sim_confidence)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (date_str, 'prop', r['stat_category'], r['player_or_team'], r['game'], r['direction'], r['line'], s_lab, r['confidence']))
                pick_id = cur.lastrowid
            
            cur.execute("SELECT id FROM pick_results WHERE pick_id=?", (pick_id,))
            if not cur.fetchone():
                cur.execute("""
                    INSERT INTO pick_results (pick_id, actual_value, is_win, diff_val, diff_pct, resolved_at)
                    VALUES (?,?,?,?,?,?)
                """, (pick_id, r['actual_value'], r['is_win'], r['diff_val'], r['diff_pct'], date_str))
                inserted_count += 1
            
        conn.commit()
        conn.close()
        print(f"  [{date_str}] Done. Added {inserted_count} new results.")
    return date_str

def main():
    parser = argparse.ArgumentParser(description='NBA Parallel Bulk Backtest')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    args = parser.parse_args()
    
    dates = generate_dates(args.start, args.end)
    print(f"🚀 STARTING PARALLEL BACKTEST ({len(dates)} days, {args.workers} workers)")
    
    scraper_script = os.path.join(MODEL_DIR, 'etl', 'live_scraper.py')
    fetch_script = os.path.join(MODEL_DIR, 'etl', 'fetch_yesterday.py')
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_date, d, scraper_script, fetch_script): d for d in dates}
        for future in concurrent.futures.as_completed(futures):
            d = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  [!] Worker failed for {d}: {e}")
        
    print(f"\n✅ All tasks completed.")

if __name__ == "__main__":
    main()
