"""
advanced_model/ml/evaluate_shadow.py
====================================
Evaluates the performance of ML Shadow picks vs Baseline picks.
Computes Hit Rate, Brier Score, and ROI.
Outputs a report to logs/ml_promotion_decision_YYYY-MM-DD.txt.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BET_DB = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')
NBA_DB = os.path.join(MODEL_DIR, 'database', 'nba_data.db')
LOGS_DIR = os.path.join(MODEL_DIR, 'logs')

def get_actuals(date_str, player, cat):
    conn = sqlite3.connect(NBA_DB)
    cur = conn.cursor()
    # Normalize category
    db_cat = cat.lower()
    if db_cat == 'points': db_cat = 'pts'
    elif db_cat == 'rebounds': db_cat = 'reb'
    elif db_cat == 'assists': db_cat = 'ast'
    
    cur.execute(f"SELECT {db_cat} FROM box_scores WHERE game_date=? AND player_name=?", (date_str, player))
    row = cur.fetchone()
    conn.close()
    if row:
        return row[0]
    return None

def compute_metrics(df):
    if len(df) == 0:
        return {'n': 0, 'hit_rate': 0.0, 'brier': 0.0, 'roi': 0.0}
        
    wins = df['is_win'].sum()
    n = len(df)
    
    # Brier: (prob - actual)^2
    # Probability is conf / 100
    prob = df['sim_confidence'] / 100.0
    actual_bin = df['is_win'].astype(float)
    brier = np.mean((prob - actual_bin)**2)
    
    # ROI: assume 1 unit bet, odds are not perfectly available in shadow_picks, but let's assume -110 (1.91)
    # Actually, shadow_picks doesn't store odds. We will assume standard 1.91 for all props.
    profit = wins * 0.91 - (n - wins)
    roi = (profit / n) * 100.0
    
    return {
        'n': n,
        'hit_rate': (wins / n) * 100.0,
        'brier': brier,
        'roi': roi
    }

def main():
    if not os.path.exists(BET_DB):
        print("Database not found.")
        return
        
    conn = sqlite3.connect(BET_DB)
    
    # 1. Fetch Baseline Picks (from daily_picks where ml_correction_applied = 'NONE')
    baseline = pd.read_sql_query("""
        SELECT d.date, d.player_or_team as player, d.stat_category, d.direction, d.line, 
               d.sim_confidence, d.odds, p.is_win
        FROM daily_picks d
        JOIN pick_results p ON d.id = p.pick_id
        WHERE d.bet_type = 'prop' AND d.ml_correction_applied = 'NONE'
    """, conn)
    
    # 2. Fetch ML Shadow Picks
    shadow = pd.read_sql_query("""
        SELECT game_date as date, player, stat_category, direction, line, sim_confidence
        FROM shadow_picks
        WHERE ban_reason = 'ML_SHADOW'
    """, conn)
    conn.close()
    
    # Resolve actuals for shadow picks
    actuals = []
    is_wins = []
    for _, row in shadow.iterrows():
        act = get_actuals(row['date'], row['player'], row['stat_category'])
        actuals.append(act)
        if act is not None:
            if row['direction'] == 'Over':
                is_wins.append(1 if act > row['line'] else 0)
            else:
                is_wins.append(1 if act < row['line'] else 0)
        else:
            is_wins.append(None)
            
    shadow['actual'] = actuals
    shadow['is_win'] = is_wins
    shadow = shadow.dropna(subset=['is_win'])
    
    base_metrics = compute_metrics(baseline)
    shad_metrics = compute_metrics(shadow)
    
    today = datetime.now().strftime("%Y-%m-%d")
    report = [
        "================================================",
        "      ML SHADOW PROMOTION DECISION REPORT       ",
        f"      Date: {today}                           ",
        "================================================",
        "",
        "BASELINE PIPELINE (ML OFF)",
        f"  Total Picks: {base_metrics['n']}",
        f"  Hit Rate:    {base_metrics['hit_rate']:.1f}%",
        f"  Brier Score: {base_metrics['brier']:.4f}",
        f"  Est. ROI:    {base_metrics['roi']:+.2f}%",
        "",
        "ML SHADOW PIPELINE (ML CALIBRATION ON)",
        f"  Total Picks: {shad_metrics['n']}",
        f"  Hit Rate:    {shad_metrics['hit_rate']:.1f}%",
        f"  Brier Score: {shad_metrics['brier']:.4f}",
        f"  Est. ROI:    {shad_metrics['roi']:+.2f}%",
        "",
        "================================================",
        "DECISION GUIDELINES:",
        "- If ML Brier Score is lower AND ROI is higher: PROMOTED",
        "- If N < 500: MORE DATA NEEDED",
        "================================================"
    ]
    
    report_text = "\n".join(report)
    print(report_text)
    
    os.makedirs(LOGS_DIR, exist_ok=True)
    out_path = os.path.join(LOGS_DIR, f"ml_promotion_decision_{today}.txt")
    with open(out_path, 'w') as f:
        f.write(report_text)
        
    print(f"\n[+] Saved report to {out_path}")

if __name__ == '__main__':
    main()
