import sqlite3
import os
import pandas as pd
from datetime import datetime

DB_PATH = 'database/bet_tracker.db'
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"odds_coverage_audit_{datetime.now().strftime('%Y-%m-%d')}.txt")

def audit_coverage(out_path=None):
    if not os.path.exists(DB_PATH):
        print(f"Error: {DB_PATH} not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        stat_category, 
        MIN(date) AS earliest, 
        MAX(date) AS latest, 
        COUNT(*) AS n, 
        COUNT(DISTINCT date) AS game_days 
    FROM historical_odds_cache 
    GROUP BY stat_category 
    ORDER BY stat_category;
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()

    report = []
    report.append("============================================================")
    report.append("  HISTORICAL ODDS CACHE COVERAGE AUDIT")
    report.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("============================================================\n")
    
    if df.empty:
        report.append("NO DATA FOUND IN historical_odds_cache")
    else:
        report.append(df.to_string(index=False))
        
    report.append("\n\nVERDICT:")
    # Check for Oct-Dec 2025 gap
    earliest_date = df['earliest'].min() if not df.empty else "N/A"
    if earliest_date != "N/A" and earliest_date >= "20260101": # ESPN/AN uses YYYYMMDD
        report.append(f"• GAP CONFIRMED: Earliest data starts on {earliest_date}.")
        report.append("• Data for Oct-Dec 2025 (2025-10-01 to 2025-12-31) is COMPLETELY MISSING.")
    else:
        report.append(f"• Data found before 2026-01-01. Earliest is {earliest_date}.")

    final_report = "\n".join(report)
    
    target_out = out_path if out_path else os.path.join(LOG_DIR, f"odds_coverage_audit_{datetime.now().strftime('%Y-%m-%d')}.txt")
    os.makedirs(os.path.dirname(target_out), exist_ok=True)
    with open(target_out, 'w') as f:
        f.write(final_report)
    
    print(final_report)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    audit_coverage(args.out)

