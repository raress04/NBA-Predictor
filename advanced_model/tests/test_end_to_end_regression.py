"""
End-to-End Regression Test
Ensures refactoring does not alter core model behavior, pick generation,
or metric calculations. Compares current DB state against tests/baseline_metrics.json.
"""

import sqlite3
import json
import os
import sys

# --- Configuration & Tolerances ---
TOLERANCE_PCT = 1.0           # max 1% drift in hit rates
TOLERANCE_FLOAT = 0.02        # max 0.02 drift in average CLV
TOLERANCE_EDGE_PCT = 0.5      # max 0.5 drift in average edge (since edge is a percentage like 28.9)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT_DIR, 'database', 'bet_tracker.db')
BASELINE_PATH = os.path.join(ROOT_DIR, 'tests', 'baseline_metrics.json')

def get_current_metrics(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    current = {
        "table_counts": {},
        "summary_metrics": {
            "hit_rate_per_category": {},
            "avg_edge_per_category": {},
            "avg_clv_per_category": {},
            "calibration_bins": {}
        }
    }
    
    # 1. Exact Counts (excluding synthetic data)
    cur.execute("SELECT COUNT(*) FROM daily_picks WHERE synthetic_flag IS NULL OR synthetic_flag != 'ERA_INFLATED'")
    current["table_counts"]["daily_picks_live"] = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM pick_results")
    current["table_counts"]["pick_results"] = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM projection_outcomes WHERE synthetic_flag IS NULL")
    current["table_counts"]["projection_outcomes_clean"] = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM parlays")
    current["table_counts"]["parlays"] = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM parlay_legs")
    current["table_counts"]["parlay_legs"] = cur.fetchone()[0]
    
    # 2. Hit Rate per Category
    cur.execute("""
        SELECT dp.stat_category, ROUND(AVG(CASE WHEN pr.is_win=1 THEN 1.0 ELSE 0.0 END)*100, 2) as hr
        FROM pick_results pr
        JOIN daily_picks dp ON dp.id = pr.pick_id
        WHERE dp.synthetic_flag IS NULL OR dp.synthetic_flag != 'ERA_INFLATED'
        GROUP BY dp.stat_category
    """)
    for row in cur.fetchall():
        current["summary_metrics"]["hit_rate_per_category"][row['stat_category']] = row['hr']
        
    # 3. Avg Edge
    cur.execute("""
        SELECT stat_category, ROUND(AVG(edge_pct), 4) as avg_edge
        FROM daily_picks
        WHERE (synthetic_flag IS NULL OR synthetic_flag != 'ERA_INFLATED') AND edge_pct IS NOT NULL
        GROUP BY stat_category
    """)
    for row in cur.fetchall():
        current["summary_metrics"]["avg_edge_per_category"][row['stat_category']] = row['avg_edge']
        
    # 4. Avg CLV
    cur.execute("""
        SELECT stat_category, ROUND(AVG(clv), 4) as avg_clv
        FROM daily_picks
        WHERE (synthetic_flag IS NULL OR synthetic_flag != 'ERA_INFLATED') AND clv IS NOT NULL
        GROUP BY stat_category
    """)
    for row in cur.fetchall():
        current["summary_metrics"]["avg_clv_per_category"][row['stat_category']] = row['avg_clv']
        
    # 5. Calibration Bins
    cur.execute("""
        SELECT 
            CASE 
                WHEN sim_confidence < 0.55 THEN '0.50-0.55'
                WHEN sim_confidence < 0.60 THEN '0.55-0.60'
                WHEN sim_confidence < 0.65 THEN '0.60-0.65'
                ELSE '0.65+' 
            END as bin,
            COUNT(*) as n,
            ROUND(AVG(CASE WHEN pr.is_win=1 THEN 1.0 ELSE 0.0 END)*100, 2) as hr
        FROM daily_picks dp
        JOIN pick_results pr ON dp.id = pr.pick_id
        WHERE dp.synthetic_flag IS NULL OR dp.synthetic_flag != 'ERA_INFLATED'
        GROUP BY bin
    """)
    for row in cur.fetchall():
        current["summary_metrics"]["calibration_bins"][row['bin']] = {"n": row['n'], "hit_rate": row['hr']}
        
    conn.close()
    return current

def compare_metrics(baseline, current):
    errors = []
    
    # Check Counts (Allow growth, error on drop)
    for k, v in baseline["table_counts"].items():
        curr_v = current["table_counts"].get(k, 0)
        if curr_v < v:
            errors.append(f"[COUNT DROPPED] {k}: was {v}, now {curr_v} — data was deleted")

    # Check Hit Rates (Pct tolerance)
    for cat, hr in baseline["summary_metrics"]["hit_rate_per_category"].items():
        curr_hr = current["summary_metrics"]["hit_rate_per_category"].get(cat, 0)
        if abs(hr - curr_hr) > TOLERANCE_PCT:
            errors.append(f"[HIT RATE DRIFT] {cat}: expected {hr}%, got {curr_hr}%")

    # Check Edge (Float tolerance for percentages)
    for cat, edge in baseline["summary_metrics"]["avg_edge_per_category"].items():
        curr_edge = current["summary_metrics"]["avg_edge_per_category"].get(cat, 0)
        if abs(edge - curr_edge) > TOLERANCE_EDGE_PCT:
            errors.append(f"[EDGE DRIFT] {cat}: expected {edge}, got {curr_edge}")

    # Check CLV (Float tolerance)
    for cat, clv in baseline["summary_metrics"]["avg_clv_per_category"].items():
        curr_clv = current["summary_metrics"]["avg_clv_per_category"].get(cat, 0)
        if abs(clv - curr_clv) > TOLERANCE_FLOAT:
            errors.append(f"[CLV DRIFT] {cat}: expected {clv}, got {curr_clv}")

    if errors:
        print("❌ REGRESSION DETECTED! The refactoring changed model behavior.")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("✅ PASS: All metrics align within tolerance bounds. Refactor is safe.")

if __name__ == "__main__":
    if not os.path.exists(BASELINE_PATH):
        print("[-] Baseline not found. Run baseline generation first.")
        sys.exit(1)
        
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
        
    current = get_current_metrics(DB_PATH)
    compare_metrics(baseline, current)
