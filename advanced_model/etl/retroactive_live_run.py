import sqlite3
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Add parent dir to path
MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(MODEL_DIR)

from simulator.matrix_builder import (
    get_player_stats_matrix, get_team_pace, 
    get_team_efficiency, get_team_rebound_efficiency
)
from simulator.markov_engine import MarkovSimulator
from etl.bias_corrections import get_tiered_bias

NBA_DB = os.path.join(MODEL_DIR, 'database', 'nba_data.db')
TRACKER_DB = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')

def get_actual_results(date_str):
    conn = sqlite3.connect(NBA_DB)
    # Convert YYYYMMDD to YYYY-MM-DD
    iso_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    # Get all player stats for the date
    query = """
    SELECT player_name, team_id, pts, (oreb + dreb) as reb, ast, game_id
    FROM box_scores
    WHERE game_date LIKE ?
    """
    df = pd.read_sql_query(query, conn, params=(f"{iso_date}%",))
    
    # Get team totals for the date
    team_query = """
    SELECT team_id, game_id, SUM(pts) as team_pts
    FROM box_scores
    WHERE game_date LIKE ?
    GROUP BY team_id, game_id
    """
    team_df = pd.read_sql_query(team_query, conn, params=(f"{iso_date}%",))
    
    # Get game totals (sum of both teams)
    game_totals = team_df.groupby('game_id')['team_pts'].sum().to_dict()
    
    conn.close()
    return df, team_df, game_totals

def get_historical_lines(date_str):
    conn = sqlite3.connect(TRACKER_DB)
    query = "SELECT player_name, stat_category, line, event_id FROM historical_odds_cache WHERE date = ?"
    df = pd.read_sql_query(query, conn, params=(date_str,))
    conn.close()
    
    # Organized as { (player, category): line }
    lines = {}
    event_ids = {} # { (player, category): eid }
    for _, row in df.iterrows():
        lines[(row['player_name'], row['stat_category'])] = row['line']
        event_ids[(row['player_name'], row['stat_category'])] = row['event_id']
    return lines, event_ids

MONTH_MAP = {
    '01': 'january', '02': 'february', '03': 'march', '04': 'april'
}

def save_report(date_str, report_lines):
    month_name = MONTH_MAP.get(date_str[4:6], 'other')
    iso_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    dir_path = os.path.join(MODEL_DIR, 'new_gen_prediction', month_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    file_path = os.path.join(dir_path, f"predicts_{iso_date}.txt")
    with open(file_path, 'w') as f:
        f.write("\n".join(report_lines))

TEAM_ID_TO_NAME = {
    1610612737: 'Atlanta Hawks', 1610612738: 'Boston Celtics',
    1610612751: 'Brooklyn Nets', 1610612766: 'Charlotte Hornets',
    1610612741: 'Chicago Bulls', 1610612739: 'Cleveland Cavaliers',
    1610612742: 'Dallas Mavericks', 1610612743: 'Denver Nuggets',
    1610612765: 'Detroit Pistons', 1610612744: 'Golden State Warriors',
    1610612745: 'Houston Rockets', 1610612754: 'Indiana Pacers',
    1610612746: 'LA Clippers', 1610612747: 'Los Angeles Lakers',
    1610612763: 'Memphis Grizzlies', 1610612748: 'Miami Heat',
    1610612749: 'Milwaukee Bucks', 1610612750: 'Minnesota Timberwolves',
    1610612740: 'New Orleans Pelicans', 1610612752: 'New York Knicks',
    1610612760: 'Oklahoma City Thunder', 1610612753: 'Orlando Magic',
    1610612755: 'Philadelphia 76ers', 1610612756: 'Phoenix Suns',
    1610612757: 'Portland Trail Blazers', 1610612758: 'Sacramento Kings',
    1610612759: 'San Antonio Spurs', 1610612761: 'Toronto Raptors',
    1610612762: 'Utah Jazz', 1610612764: 'Washington Wizards',
}

from config.settings import PROJECTION_TIERS
from simulator.parlay_builder import EDGE_THRESHOLDS
from simulator.markov_engine import compute_posterior_confidence

def run_retroactive_day(date_str):
    print(f"\n[>>>] RUNNING RETROACTIVE: {date_str}")
    actual_players, actual_teams, game_totals = get_actual_results(date_str)
    lines, event_ids = get_historical_lines(date_str)
    
    if actual_players.empty or not lines:
        print(f"  [!] Missing data for {date_str} (Players: {len(actual_players)}, Lines: {len(lines)})")
        return

    conn_tracker = sqlite3.connect(TRACKER_DB)
    cursor = conn_tracker.cursor()

    processed = 0
    iso_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    report_lines = [f"NBA MODEL PREDICTIONS - {iso_date}", "="*60, ""]
    game_ids = actual_players['game_id'].unique()
    
    for gid in game_ids:
        game_players = actual_players[actual_players['game_id'] == gid]
        team_ids = game_players['team_id'].unique()
        if len(team_ids) < 2: continue
        t1_id, t2_id = team_ids[0], team_ids[1]
        t1_name = TEAM_ID_TO_NAME.get(t1_id, f"Team {t1_id}")
        t2_name = TEAM_ID_TO_NAME.get(t2_id, f"Team {t2_id}")
        
        # 1. Build Rosters (using before_date)
        rosters = {t1_id: {}, t2_id: {}}
        for _, p in game_players.iterrows():
            try:
                # Use only players with significant minutes or a line
                if p['pts'] + p['reb'] + p['ast'] > 5 or (p['player_name'], 'POINTS') in lines:
                    matrix = get_player_stats_matrix(p['player_name'], before_date=date_str)
                    rosters[p['team_id']][p['player_name']] = matrix
            except: pass
            
        if not rosters[t1_id] or not rosters[t2_id]: continue

        # 2. Get Pace
        pace1 = get_team_pace(t1_id, before_date=date_str)
        pace2 = get_team_pace(t2_id, before_date=date_str)
        game_pace = (pace1 + pace2) / 2

        # 3. Simulate Game
        sim = MarkovSimulator(rosters[t1_id], rosters[t2_id])
        num_sims = 500
        aggregated = {} # {player: {stat: [values]}}
        game_scores = [] # [total_pts]
        
        for _ in range(num_sims):
            res = sim.run_full_game(pace=game_pace)
            game_scores.append(res['home_score'] + res['away_score'])
            for p_name, p_stats in res['player_stats'].items():
                if p_name not in aggregated:
                    aggregated[p_name] = {'pts': [], 'reb': [], 'ast': []}
                aggregated[p_name]['pts'].append(p_stats['PTS'])
                aggregated[p_name]['reb'].append(p_stats['REB'])
                aggregated[p_name]['ast'].append(p_stats['AST'])
        
        # Start Matchup Report
        report_lines.append(f"\n============================================================")
        report_lines.append(f" MATCHUP: {t1_name} vs. {t2_name}")
        report_lines.append(f"============================================================")
        
        proj_total_raw = np.mean(game_scores)
        total_bias = get_tiered_bias('TOTAL', proj_total_raw)
        proj_total_corr = proj_total_raw + total_bias
        
        report_lines.append(f"PROJECTED TOTAL: {proj_total_raw:.1f}")
        total_line = lines.get(('Game Total', 'TOTAL'))
        if total_line:
            val_total = ""
            edge_total = proj_total_corr - total_line
            thresh_total = EDGE_THRESHOLDS['totals']['trigger']
            if abs(edge_total) >= thresh_total:
                direction = "Over" if edge_total > 0 else "Under"
                overs = sum(1 for s in game_scores if s > total_line)
                conf, _ = compute_posterior_confidence(overs, num_sims)
                val_total = f" VALUABLE: {direction} ({conf*100:.0f}% conf, {abs(edge_total):.1f} edge)"
            
            report_lines.append(f"  [BIAS CORRECTION] TOTAL | Neutral | median={proj_total_raw:.1f} | offset={total_bias:+.2f} | corrected={proj_total_corr:.1f}")
            report_lines.append(f"  -> GAME TOTAL: O/U {total_line} {val_total if val_total else 'ignore'}")

        report_lines.append("-" * 40)
        report_lines.append("   TOP PLAYER PROP PROJECTIONS (MEDIAN)")
        report_lines.append("-" * 40)

        # 4. Store Player Outcomes
        stat_col_map = {'POINTS': 'pts', 'REBOUNDS': 'reb', 'ASSISTS': 'ast'}
        for tid in [t1_id, t2_id]:
            report_lines.append(f"\n[{'HOME' if tid == t1_id else 'AWAY'} TEAM]")
            t_players = game_players[game_players['team_id'] == tid]
            for _, p_row in t_players.iterrows():
                p_name = p_row['player_name']
                if p_name not in aggregated: continue
                
                # Projections & Bias Correction logs
                p_projs_raw = {
                    'POINTS': np.mean(aggregated[p_name]['pts']),
                    'REBOUNDS': np.mean(aggregated[p_name]['reb']),
                    'ASSISTS': np.mean(aggregated[p_name]['ast'])
                }
                
                corrected = {}
                for cat in ['POINTS', 'REBOUNDS', 'ASSISTS']:
                    raw_val = p_projs_raw[cat]
                    offset = get_tiered_bias(cat, raw_val)
                    corr_val = raw_val + offset
                    corrected[cat] = corr_val
                    
                    # Log bias correction if there's a line
                    if (p_name, cat) in lines:
                        tier_thresh = PROJECTION_TIERS.get(cat, 10)
                        tier = "Starter" if raw_val >= tier_thresh else "Bench"
                        report_lines.append(f"  [BIAS CORRECTION] {cat:8} | {tier:7} | median={raw_val:4.1f} | offset={offset:+5.2f} | corrected={corr_val:4.1f}")

                proj_summary = f"{p_name:20} -> {corrected['POINTS']:.0f} PTS, {corrected['REBOUNDS']:.0f} REB, {corrected['ASSISTS']:.0f} AST"
                
                line_details = []
                for cat in ['POINTS', 'REBOUNDS', 'ASSISTS']:
                    line = lines.get((p_name, cat))
                    if line is None: continue
                    
                    # Edge Tag Logic
                    edge = corrected[cat] - line
                    trigger = EDGE_THRESHOLDS[cat.lower()]['trigger']
                    if abs(edge) >= trigger:
                        direction = "Over" if edge > 0 else "Under"
                        # Calc confidence
                        samples = aggregated[p_name][stat_col_map[cat]]
                        overs = sum(1 for s in samples if s > line)
                        conf, _ = compute_posterior_confidence(overs, num_sims)
                        tag = f"VALUABLE: {direction} ({conf*100:.0f}% conf, {abs(edge/line*100):.1f}% edge)"
                    else:
                        tag = "ignore"
                    
                    line_details.append(f"O/U {line} {cat.lower()} {tag}")
                    
                    # DB Storage
                    col = stat_col_map[cat]
                    actual = p_row[col]
                    bias = actual - p_projs_raw[cat] # Recalibrate original raw bias
                    eid = event_ids.get((p_name, cat), f"AN_{gid}")
                    
                    cursor.execute("""
                        INSERT INTO projection_outcomes 
                        (game_date, player, category, projected_value, bookmaker_line, actual_value, model_bias, source_mode, source_file, synthetic_flag)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        date_str, p_name, cat, p_projs_raw[cat], line, actual, 
                        bias, 'historical', eid, '0'
                    ))
                    processed += 1
                
                if line_details:
                    proj_summary += f" ({', '.join(line_details)})"
                report_lines.append(f"  {proj_summary}")

        # 5. Store Game Total Outcome (DB)
        if total_line:
            actual_total = game_totals.get(gid, 0)
            proj_total = np.mean(game_scores)
            bias_total = actual_total - proj_total
            eid = event_ids.get(('Game Total', 'TOTAL'), f"AN_{gid}")
            
            cursor.execute("""
                INSERT INTO projection_outcomes 
                (game_date, player, category, projected_value, bookmaker_line, actual_value, model_bias, source_mode, source_file, synthetic_flag)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date_str, 'Game Total', 'TOTAL', proj_total, total_line, actual_total, 
                bias_total, 'historical', eid, '0'
            ))
            processed += 1

    conn_tracker.commit()
    conn_tracker.close()
    save_report(date_str, report_lines)
    print(f"  [✓] Processed {processed} outcome rows. Detailed report saved.")

def main(start_date, end_date):
    current = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    
    while current <= end:
        ds = current.strftime("%Y%m%d")
        run_retroactive_day(ds)
        current += timedelta(days=1)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python3 etl/retroactive_live_run.py YYYYMMDD YYYYMMDD")
