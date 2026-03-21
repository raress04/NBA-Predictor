import re
import os
import sqlite3
from datetime import datetime
from typing import List, Dict

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def parse_parlays_from_file(file_path: str) -> Dict[str, List[List[str]]]:
    """
    Parses parlay legs from a predicts_YYYY-MM-DD.txt file.
    Returns: {'parlay1': [[pick_text, game_text], ...], 'parlay2': [...]}
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    results = {'parlay1': [], 'parlay2': []}
    
    # Simple regex to split sections
    p1_section = re.search(r'PARLAY RECOMMENDATION #1.*?={60}', content, re.DOTALL)
    p2_section = re.search(r'PARLAY RECOMMENDATION #2.*?={60}', content, re.DOTALL)

    def extract_picks(section_text):
        picks = []
        # Pattern: Pick N: [Game] Description @ Odds
        pick_lines = re.findall(r'Pick \d+: \[(.*?)\] (.*?) @ ([\d\.]+)', section_text)
        for game, desc, odds in pick_lines:
            picks.append({'game': game, 'desc': desc, 'odds': float(odds)})
        return picks

    if p1_section:
        results['parlay1'] = extract_picks(p1_section.group(0))
    if p2_section:
        results['parlay2'] = extract_picks(p2_section.group(0))
        
    return results

def match_leg_to_db(conn, date_str, leg_info):
    """
    Tries to find the outcome of a leg in the DB.
    """
    cur = conn.cursor()
    # Problem: The 'desc' in parlay text is slightly different from 'daily_picks' store.
    # Daily picks stores: player_or_team, stat_category, direction, line.
    
    # We need a fuzzy match or a more robust parser.
    # Example Desc: "Los Angeles Lakers -11.5 -> -8.5 (reduced 3.0pts) | Proj margin: +18.0pts"
    # Example Desc: "Over 239.5 (Projected: 264.0)"
    # Example Desc: "LeBron James Over 22.5 POINTS (Sim: 88%)"
    
    desc = leg_info['desc']
    
    # Try Prop Match
    prop_m = re.search(r'(.*?) (Over|Under) ([\d\.]+) (POINTS|REBOUNDS|ASSISTS|PTS|REB|AST|TOTAL|SPREAD)', desc, re.IGNORECASE)
    if prop_m:
        player, direction, line, stat = prop_m.groups()
        stat_map = {'PTS': 'POINTS', 'REB': 'REBOUNDS', 'AST': 'ASSISTS'}
        stat = stat_map.get(stat.upper(), stat.upper())
        
        cur.execute("""
            SELECT pr.is_win FROM daily_picks dp
            JOIN pick_results pr ON dp.id = pr.pick_id
            WHERE dp.date=? AND dp.player_or_team LIKE ? AND dp.stat_category=? 
              AND dp.direction=? AND dp.line=?
        """, (date_str, f"%{player.strip()}%", stat, direction.upper(), float(line)))
        row = cur.fetchone()
        if row: return row[0]

    # Try Spread Match
    spread_m = re.search(r'(.*?) ([+-][\d\.]+) \u2192 ([+-][\d\.]+)', desc)
    if spread_m:
        team, market_line, safe_line = spread_m.groups()
        cur.execute("""
            SELECT pr.is_win FROM daily_picks dp
            JOIN pick_results pr ON dp.id = pr.pick_id
            WHERE dp.date=? AND dp.player_or_team LIKE ? AND dp.stat_category='SPREAD'
        """, (date_str, f"%{team.strip()}%"))
        row = cur.fetchone()
        if row: return row[0]

    # Try Total Match
    total_m = re.search(r'(Over|Under) ([\d\.]+)', desc)
    if total_m:
        direction, line = total_m.groups()
        cur.execute("""
            SELECT pr.is_win FROM daily_picks dp
            JOIN pick_results pr ON dp.id = pr.pick_id
            WHERE dp.date=? AND dp.stat_category='TOTAL' AND dp.direction=? AND dp.line=?
        """, (date_str, direction.upper(), float(line)))
        row = cur.fetchone()
        if row: return row[0]

    return None

def analyze_all():
    conn = get_db_connection()
    
    pred_files = []
    for root, dirs, files in os.walk(os.path.join(MODEL_DIR, 'predictions')):
        for f in files:
            if f.startswith('predicts_') and f.endswith('.txt'):
                pred_files.append(os.path.join(root, f))
    
    pred_files.sort()
    
    reports = {'parlay1': {'wins': 0, 'losses': 0, 'pending': 0, 'total': 0},
               'parlay2': {'wins': 0, 'losses': 0, 'pending': 0, 'total': 0}}

    print("\n" + "="*60)
    print("  📈  PARLAY HISTORICAL PERFORMANCE ANALYSIS")
    print("="*60)
    print(f"{'DATE':<12} | {'PARLAY #1 (2-3 PICKS)':<22} | {'PARLAY #2 (4 PICKS)':<20}")
    print("-" * 60)

    for fpath in pred_files:
        basename = os.path.basename(fpath)
        date_str = re.search(r'(\d{4}-\d{2}-\d{2})', basename).group(1)
        
        parlays = parse_parlays_from_file(fpath)
        
        day_results = []
        for p_key in ['parlay1', 'parlay2']:
            legs = parlays[p_key]
            if not legs:
                day_results.append("NO DATA")
                continue
                
            leg_results = []
            for leg in legs:
                res = match_leg_to_db(conn, date_str, leg)
                leg_results.append(res)
            
            if None in leg_results:
                day_results.append("PENDING")
                reports[p_key]['pending'] += 1
            elif all(r == 1 for r in leg_results):
                day_results.append("WON")
                reports[p_key]['wins'] += 1
                reports[p_key]['total'] += 1
            else:
                day_results.append("LOST")
                reports[p_key]['losses'] += 1
                reports[p_key]['total'] += 1

        print(f"{date_str:<12} | {day_results[0]:<22} | {day_results[1]:<20}")

    conn.close()
    
    print("\n" + "="*60)
    print("  📊 AGGREGATE PARLAY STATISTICS")
    print("="*60)
    for p_key, label in [('parlay1', 'Parlay #1 (2-3 Legs)'), ('parlay2', 'Parlay #2 (4 Legs / System)')]:
        stats = reports[p_key]
        total = stats['total']
        if total > 0:
            win_rate = (stats['wins'] / total) * 100
            print(f"  {label:<25} Rate: {stats['wins']}/{total} ({win_rate:.1f}%)")
        else:
            print(f"  {label:<25} No resolved parlays found.")
    print("="*60 + "\n")

if __name__ == "__main__":
    analyze_all()
