import os
import re
import sqlite3
from datetime import datetime

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')
print(f"[DEBUG] Attempting to open DB at: {db_path}")
conn = sqlite3.connect(db_path)
cur = conn.cursor()

def get_predicts_file_path(date_str):
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    month = dt.strftime('%B').lower()
    return os.path.join(MODEL_DIR, 'predictions', month, f'predicts_{date_str}.txt')

# Get all unique dates from the DB
cur.execute("SELECT DISTINCT date FROM daily_picks WHERE date BETWEEN '2026-01-01' AND '2026-03-05'")
dates = [r[0] for r in cur.fetchall()]

total_updated = 0
for date_str in dates:
    fpath = get_predicts_file_path(date_str)
    if not os.path.exists(fpath): continue
    
    medians = {}
    with open(fpath, 'r', encoding='utf-8') as f:
        curr_game = None
        for line in f:
            mg = re.search(r'MATCHUP:\s+(.*?)\s+@\s+(.*)', line)
            if mg: curr_game = f"{mg.group(1).strip()} @ {mg.group(2).strip()}"
            mm = re.search(r"^\s*([A-Za-z\s\.\'-]+)\s*->\s*(\d+)\s*PTS,\s*(\d+)\s*REB,\s*(\d+)\s*AST(.*)", line)
            if mm:
                p = mm.group(1).strip()
                medians[p] = {
                    'POINTS': float(mm.group(2)), 
                    'REBOUNDS': float(mm.group(3)), 
                    'ASSISTS': float(mm.group(4)),
                    'extra': mm.group(5)
                }

    cur.execute("SELECT id, player_or_team, stat_category, line FROM daily_picks WHERE date = ?", (date_str,))
    picks = cur.fetchall()
    
    for pick_id, p_name, s_cat, line in picks:
        if p_name in medians:
            m_data = medians[p_name]
            proj = m_data.get(s_cat, 0.0)
            margin = abs(proj - line)
            
            # Extract confidence from 'extra' text
            conf = 0.0
            stat_term = s_cat.lower()
            conf_match = re.search(rf"O/U \d+\.?\d* {stat_term} VALUABLE: (?:Under|Over) \((\d+)% conf", m_data['extra'])
            if conf_match:
                conf = float(conf_match.group(1))

            cur.execute("UPDATE daily_picks SET projected_margin = ?, sim_confidence = ? WHERE id = ?", (margin, conf, pick_id))
            total_updated += 1

conn.commit()
conn.close()
print(f"[+] Total Updated: {total_updated} rows.")
