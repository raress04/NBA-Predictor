import os
import sqlite3
import argparse
from datetime import datetime
from advanced_model.etl.parse_historical_reports import parse_historical_file

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(MODEL_DIR, 'database')
BET_TRACKER_DB = os.path.join(DB_DIR, 'bet_tracker.db')
NBA_DATA_DB = os.path.join(DB_DIR, 'nba_data.db')

def _get_conn(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

def populate_from_live():
    """Populate projection_outcomes from live picks."""
    print("[*] Populating from live picks...")
    conn = _get_conn(BET_TRACKER_DB)
    cur = conn.cursor()
    
    cur.execute("SELECT DISTINCT date, source_file FROM daily_picks WHERE source_file LIKE 'predicts_2026-03%'")
    rows = cur.fetchall()
    
    total_inserted = 0
    n_conn = _get_conn(NBA_DATA_DB)
    n_cur = n_conn.cursor()

    for i, row in enumerate(rows):
        game_date = row['date']
        source_file = row['source_file']
        month = datetime.strptime(game_date, "%Y-%m-%d").strftime("%B").lower()
        filepath = os.path.join(MODEL_DIR, 'predictions', month, source_file)
        
        if not os.path.exists(filepath):
            continue
            
        print(f"    Processing {source_file} ({i+1}/{len(rows)})...")
        projections = parse_historical_file(filepath)
        
        day_inserted = 0
        for p in projections:
            actual_val = get_actual_value(n_cur, game_date, p['player'], p['category'], p['matchup'])
            if actual_val is None:
                continue
                
            model_bias = actual_val - p['projected_value']
            bookmaker_line = get_bookmaker_line(game_date, p['player'], p['category'])
            
            cur.execute("""
                INSERT OR IGNORE INTO projection_outcomes 
                (game_date, player, category, projected_value, bookmaker_line, actual_value, model_bias, source_mode, source_file)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (game_date, p['player'], p['category'], p['projected_value'], bookmaker_line, actual_val, model_bias, 'live', source_file))
            total_inserted += cur.rowcount
            day_inserted += cur.rowcount
        
        conn.commit()
        print(f"      - Inserted {day_inserted} outcomes for {game_date}")
            
    n_conn.close()
    conn.commit()
    conn.close()
    print(f"[+] Inserted {total_inserted} live projection outcomes.")

def populate_from_historical(directory):
    """Populate projection_outcomes from historical reports in directory."""
    print(f"[*] Populating from historical reports in {directory}...")
    import glob
    files = glob.glob(os.path.join(directory, "predicts_*.txt"))
    
    conn = _get_conn(BET_TRACKER_DB)
    cur = conn.cursor()
    
    total_inserted = 0
    
    n_conn = _get_conn(NBA_DATA_DB)
    n_cur = n_conn.cursor()
    
    for i, f in enumerate(files):
        print(f"    Processing {os.path.basename(f)} ({i+1}/{len(files)})...")
        projections = parse_historical_file(f)
        day_inserted = 0
        for p in projections:
            actual_val = get_actual_value(n_cur, p['game_date'], p['player'], p['category'], p['matchup'])
            if actual_val is None:
                continue
            
            model_bias = actual_val - p['projected_value']
            
            cur.execute("""
                INSERT OR IGNORE INTO projection_outcomes 
                (game_date, player, category, projected_value, bookmaker_line, actual_value, model_bias, source_mode, source_file)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (p['game_date'], p['player'], p['category'], p['projected_value'], None, actual_val, model_bias, 'historical', p['source_file']))
            total_inserted += cur.rowcount
            day_inserted += cur.rowcount
        
        conn.commit()
        print(f"      - Inserted {day_inserted} outcomes for {os.path.basename(f)}")
            
    n_conn.close()
    conn.close()
    print(f"[+] Inserted {total_inserted} historical projection outcomes.")

def get_actual_value(cur, game_date, player, category, matchup):
    """Fetch actual value from nba_data.db using provided cursor."""
    val = None
    # Normalize date for query (match 'YYYY-MM-DD%')
    date_pattern = f"{game_date}%"
    
    if category in ['POINTS', 'REBOUNDS', 'ASSISTS']:
        cur.execute("""
            SELECT pts, reb, ast 
            FROM box_scores 
            WHERE game_date LIKE ? AND player_name = ?
        """, (date_pattern, player))
        row = cur.fetchone()
        if row:
            if category == 'POINTS': val = float(row['pts'])
            elif category == 'REBOUNDS': val = float(row['reb'])
            elif category == 'ASSISTS': val = float(row['ast'])
    
    elif category == 'TOTAL' or category == 'SPREAD':
        # Find game_id first
        cur.execute("SELECT game_id, matchup FROM games WHERE game_date LIKE ?", (date_pattern,))
        games = cur.fetchall()
        game_id = None
        
        # Matchup name search
        report_teams = matchup.replace(" @ ", " vs. ").split(" vs. ")
        for g in games:
            g_matchup = g['matchup'].replace(" @ ", " vs. ").split(" vs. ")
            # Check if common elements exist (fuzzy)
            match_count = 0
            for rt in report_teams:
                for gt in g_matchup:
                    if rt.lower() in gt.lower() or gt.lower() in rt.lower():
                        match_count += 1
                        break
            if match_count >= 2:
                game_id = g['game_id']
                break
        
        if game_id:
            if category == 'TOTAL':
                cur.execute("SELECT SUM(pts) as total FROM box_scores WHERE game_id = ?", (game_id,))
                row = cur.fetchone()
                if row and row['total'] is not None:
                    val = float(row['total'])
            
            elif category == 'SPREAD':
                # 'player' is team abbreviation like 'LAC'
                # Find team_id for this player
                cur.execute("SELECT team_id FROM teams WHERE abbreviation = ?", (player,))
                t_row = cur.fetchone()
                if t_row:
                    team_id = t_row['team_id']
                    # Get this team's pts
                    cur.execute("SELECT SUM(pts) as team_pts FROM box_scores WHERE game_id = ? AND team_id = ?", (game_id, team_id))
                    t_pts_row = cur.fetchone()
                    # Get total pts
                    cur.execute("SELECT SUM(pts) as total_pts FROM box_scores WHERE game_id = ?", (game_id,))
                    total_pts_row = cur.fetchone()
                    
                    if t_pts_row and total_pts_row and t_pts_row['team_pts'] is not None:
                        team_pts = float(t_pts_row['team_pts'])
                        opp_pts = float(total_pts_row['total_pts']) - team_pts
                        val = team_pts - opp_pts

    return val

def get_bookmaker_line(game_date, player, category):
    conn = _get_conn(BET_TRACKER_DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT line FROM daily_picks 
        WHERE date = ? AND player_or_team = ? AND stat_category = ?
    """, (game_date, player, category))
    row = cur.fetchone()
    conn.close()
    return float(row['line']) if row else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['live', 'historical', 'all'], default='all')
    args = parser.parse_args()
    
    # Init DB schema
    from advanced_model.database.bet_tracker_schema import init_db
    init_db()
    
    if args.mode in ['live', 'all']:
        populate_from_live()
    if args.mode in ['historical', 'all']:
        populate_from_historical(os.path.join(MODEL_DIR, 'predictions', 'january'))
        populate_from_historical(os.path.join(MODEL_DIR, 'predictions', 'february'))
