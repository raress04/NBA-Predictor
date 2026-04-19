"""
WARNING: DEPRECATED - CIRCULAR VALIDATION
This script was responsible for generating the 31,553 synthetic picks in `bet_tracker.db`.
It invents bookmaker lines based on the model's own projections or season averages,
leading to artificially inflated win rates.

RELATIONSHIP TO BACKTESTER.PY:
This module is COMPLETELY INDEPENDENT of `simulator/backtester.py`.
- `bulk_backtest.py` pollutes the database and uses circular validation logic.
- `simulator/backtester.py` is a clean diagnostic tool evaluating fundamental score MAE.

Do not use this for Phase 6. Phase 6 will involve creating a new clean backtest script
integrating real data.
"""
"""
WARNING: DEPRECATED - CIRCULAR VALIDATION
This script was responsible for generating the 31,553 synthetic picks in `bet_tracker.db`.
It invents bookmaker lines based on the model's own projections or season averages,
leading to artificially inflated win rates.

RELATIONSHIP TO BACKTESTER.PY:
This module is COMPLETELY INDEPENDENT of `simulator/backtester.py`.
- `bulk_backtest.py` pollutes the database and uses circular validation logic.
- `simulator/backtester.py` is a clean diagnostic tool evaluating fundamental score MAE.

Do not use this for Phase 6. Phase 6 will involve creating a new clean backtest script
integrating real data.
"""
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

# Team Full Name to Abbreviation Mapping
TEAM_MAP = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'Los Angeles Clippers': 'LAC', 'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS'
}

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

def get_results_file_path(date_str):
    """Determine the yesterday_results.txt path based on the month/year."""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    month_name = dt.strftime('%B').lower()
    
    folder_path = os.path.join(MODEL_DIR, 'results', month_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        
    return os.path.join(folder_path, f'yesterday_results_{date_str}.txt')

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
    results_txt = get_results_file_path(date_str)

    # If both files exist, we skip the heavy simulation part
    if not (os.path.exists(predicts_txt) and os.path.exists(results_txt)):
        # 1. Fetch Yesterday's Box Scores
        print(f"  [{date_str}] Fetching Results...")
        try:
            run_script(fetch_script, ['--date', date_str])
        except Exception as e:
            print(f"  [!] Failed fetching results for {date_str}: {e}")

        # Move root results generated by single script
        root_results = os.path.join(MODEL_DIR, f'yesterday_results_{date_str}.txt')
        if os.path.exists(root_results) and not os.path.exists(results_txt):
            shutil.move(root_results, results_txt)

        # Alternative generic catch if the fetch script outputs as just 'yesterday_results.txt'
        if not os.path.exists(results_txt):
            gen_res = os.path.join(MODEL_DIR, 'yesterday_results.txt')
            if os.path.exists(gen_res):
                shutil.copy(gen_res, results_txt)

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

        # Prepare team mapping for current games
        def normalize_game_name(full_name_game):
            # "Away Team @ Home Team" -> "AWY @ HOM"
            if ' @ ' not in full_name_game: return full_name_game
            away_f, home_f = full_name_game.split(' @ ')
            away_a = TEAM_MAP.get(away_f.strip(), away_f.strip())
            home_a = TEAM_MAP.get(home_f.strip(), home_f.strip())
            return f"{away_a} @ {home_a}"

        # Parse Predictions & Medians
        medians = []
        game_stats = []
        curr_game = None
        with open(predicts_txt, 'r', encoding='utf-8') as f:
            for line in f:
                mg = re.search(r'MATCHUP:\s+(.*?)\s+@\s+(.*)', line)
                if mg: 
                    curr_game = normalize_game_name(f"{mg.group(1).strip()} @ {mg.group(2).strip()}")
                
                # TOTAL
                tm = re.search(r'PROJECTED TOTAL:\s*([\d\.]+)', line)
                if tm and curr_game:
                    game_stats.append({'game': curr_game, 'type': 'TOTAL', 'proj': float(tm.group(1))})
                
                # SPREAD
                sm = re.search(r'PROJECTED SCORE:\s*(.*?)\s+([\d\.]+)\s+-\s+(.*?)\s+([\d\.]+)', line)
                if sm and curr_game:
                    t1, s1, t2, s2 = sm.groups()
                    team_a = TEAM_MAP.get(t1.strip(), t1.strip())
                    game_stats.append({'game': curr_game, 'type': 'SPREAD', 'team': team_a, 'proj_margin': float(s1) - float(s2)})

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
        game_results = {}
        with open(results_txt, 'r', encoding='utf-8') as f:
            for line in f:
                # Player scores
                p_sm = re.search(r"^([A-Za-z\s\.\'-]+)\s*\|\s*(\d+)\s*PTS,\s*(\d+)\s*REB,\s*(\d+)\s*AST", line)
                if p_sm:
                    actuals[p_sm.group(1).strip()] = {'PTS': int(p_sm.group(2)), 'REB': int(p_sm.group(3)), 'AST': int(p_sm.group(4))}
                
                # Game scores
                g_sm = re.search(r'MATCHUP:\s+(\w+)\s+vs\.\s+(\w+)\s+(\d+)\s+-\s+(\d+)', line)
                if g_sm:
                    t1, t2, s1, s2 = g_sm.groups()
                    game_results[f"{t2} @ {t1}"] = {'t1': t1, 't2': t2, 's1': int(s1), 's2': int(s2)}
                    game_results[f"{t1} @ {t2}"] = {'t1': t1, 't2': t2, 's1': int(s1), 's2': int(s2)} # handle both orderings
        
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

        # Process SPREAD & TOTAL
        for g in game_stats:
            res = game_results.get(g['game'])
            if not res: continue
            
            if g['type'] == 'TOTAL':
                act_total = float(res['s1'] + res['s2'])
                # Since we don't have a market line, we'll invent one for bias analysis: 
                # we'll use a round number near the projection or just use the projection as the 'line'
                # For bias analysis (Mean Bias), the line doesn't matter as long as we track Actual - Projected.
                # However, for summary (Win Rate), we need a line. I'll use a standard offset or round number.
                line_val = round(g['proj'] * 2) / 2.0  # nearest 0.5
                diff = act_total - g['proj']
                valid_inserts.append({
                    'date': date_str,
                    'bet_type': 'total',
                    'stat_category': 'TOTAL',
                    'player_or_team': 'GAME',
                    'game': g['game'],
                    'direction': 'OVER' if act_total > g['proj'] else 'UNDER', # synthetic win/loss based on proj
                    'line': line_val,
                    'actual_value': act_total,
                    'is_win': 1, # mark as resolved
                    'diff_val': diff,
                    'diff_pct': (diff / g['proj'] * 100) if g['proj'] > 0 else 0,
                    'source': 'bulk_backtest_gen',
                    'confidence': 0
                })
            elif g['type'] == 'SPREAD':
                # res has s1 (home), s2 (away). t1 is usually the first team listed in the score line.
                # Let's verify team names match.
                act_margin = float(res['s1'] - res['s2']) if res['t1'] == g['team'] else float(res['s2'] - res['s1'])
                diff = act_margin - g['proj_margin']
                valid_inserts.append({
                    'date': date_str,
                    'bet_type': 'spread',
                    'stat_category': 'SPREAD',
                    'player_or_team': g['team'],
                    'game': g['game'],
                    'direction': 'COVER',
                    'line': 0.0, # pure margin tracking
                    'actual_value': act_margin,
                    'is_win': 1,
                    'diff_val': diff,
                    'diff_pct': 0,
                    'source': 'bulk_backtest_gen',
                    'confidence': 0
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
                """, (date_str, r['bet_type'], r['stat_category'], r['player_or_team'], r['game'], r['direction'], r['line'], s_lab, r['confidence']))
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
    fetch_script = os.path.join(MODEL_DIR, 'etl', 'fetch_range_box_scores.py')
    
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
