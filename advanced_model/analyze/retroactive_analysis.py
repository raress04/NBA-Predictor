import sqlite3
import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime

# Ensure project root is in path
MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

# Use retrospective_db.db for new gen analysis
DB_PATH = os.path.join(MODEL_DIR, "database", "retrospective_db.db")

from config.settings import PROJECTION_TIERS

def run_analysis(month_filter=None):
    print(f"[+] Connecting to Database: {DB_PATH}")
    if not os.path.exists(DB_PATH):
        print(f"[!] Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # 0. Load Data
    query = "SELECT * FROM projection_outcomes WHERE source_mode = 'retroactive_live'"
    if month_filter:
        query += f" AND STRFTIME('%m', game_date) = '{month_filter}'"
        print(f"[*] Filtering for Month: {month_filter}")
    
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print("[!] No data found for the selected criteria.")
        conn.close()
        return

    # Derive direction if missing or NULL
    if 'direction' not in df.columns:
        df['direction'] = None
    
    def derive_direction(row):
        if pd.notnull(row['direction']):
            return row['direction']
        if row['category'] == 'SPREAD':
            return 'Cover'
        if row['projected_value'] > row['bookmaker_line']:
            return 'Over'
        return 'Under'
    
    df['direction'] = df.apply(derive_direction, axis=1)

    # Add Tier for analysis
    def get_tier(row):
        thresh = PROJECTION_TIERS.get(row['category'], 10)
        return "starter" if row['projected_value'] >= thresh else "bench"
    
    df['tier'] = df.apply(get_tier, axis=1)
    df['mae'] = (df['actual_value'] - df['projected_value']).abs()
    
    # Ensure game_date is datetime
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Report setup
    period_str = f"MONTH {month_filter}" if month_filter else "ALL TIME (JAN 1 - NOW)"
    report = []
    report.append(f"NBA MODEL - RETROACTIVE ANALYSIS - {period_str}")
    report.append("="*60)
    report.append(f"Analyzed {len(df)} projection rows.")
    report.append(f"Unique game dates covered: {df['game_date'].nunique()}")
    report.append("")

    # --- Question 1: Coverage & Sanity ---
    report.append("1. COVERAGE & INTEGRITY CHECK")
    report.append("-" * 40)
    
    coverage = df.groupby('category').agg(
        n=('category', 'count'),
        days=('game_date', 'nunique'),
        has_line=('bookmaker_line', lambda x: x.notnull().sum())
    ).reset_index()
    report.append("\nCoverage per Category:")
    report.append(coverage.to_string(index=False))
    report.append("")

    # --- Question 4: Win Rate vs Lines ---
    report.append("2. WIN RATE VS BOOKMAKER LINES (Jan Analysis)")
    report.append("-" * 40)
    results = []
    total_wins = 0
    total_n = 0
    
    for cat in ['POINTS', 'REBOUNDS', 'ASSISTS', 'TOTAL', 'SPREAD']:
        cat_df = df[df['category'] == cat].dropna(subset=['bookmaker_line'])
        if cat_df.empty: continue
        
        if cat == 'SPREAD':
            picks = cat_df
            # actual_value is now the point margin — cover = margin > line
            wins = (picks['actual_value'] > picks['bookmaker_line']).sum()
            n = len(picks)
            win_pct = (wins / n * 100) if n > 0 else 0
            results.append({'Category': cat, 'Picks': n, 'Wins': wins, 'Win%': f"{win_pct:.1f}%"})
            total_wins += wins
            total_n += n
        elif cat == 'TOTAL':
            # Handle Over/Under for totals
            overs = cat_df[cat_df['direction'] == 'Over']
            unders = cat_df[cat_df['direction'] == 'Under']
            o_wins = (overs['actual_value'] > overs['bookmaker_line']).sum()
            u_wins = (unders['actual_value'] < unders['bookmaker_line']).sum()
            
            n = len(cat_df)
            wins = o_wins + u_wins
            win_pct = (wins / n * 100) if n > 0 else 0
            results.append({'Category': cat, 'Picks': n, 'Wins': wins, 'Win%': f"{win_pct:.1f}%"})
            total_wins += wins
            total_n += n
        else:
            # Player Props
            overs = cat_df[cat_df['direction'] == 'Over']
            unders = cat_df[cat_df['direction'] == 'Under']
            
            o_wins = (overs['actual_value'] > overs['bookmaker_line']).sum()
            u_wins = (unders['actual_value'] < unders['bookmaker_line']).sum()
            
            n = len(cat_df)
            wins = o_wins + u_wins
            win_pct = (wins / n * 100) if n > 0 else 0
            results.append({'Category': cat, 'Picks': n, 'Wins': wins, 'Win%': f"{win_pct:.1f}%"})
            total_wins += wins
            total_n += n
        
    res_df = pd.DataFrame(results)
    report.append(res_df.to_string(index=False))
    
    overall_pct = (total_wins / total_n * 100) if total_n > 0 else 0
    report.append(f"\nOVERALL PERFORMANCE: {total_wins}/{total_n} ({overall_pct:.1f}%)")
    report.append("")

    # --- Question 5: Outliers ---
    report.append("3. TOP 20 ERRORS")
    report.append("-" * 40)
    outliers = df.sort_values('mae', ascending=False).head(20)[
        ['game_date', 'player', 'category', 'projected_value', 'actual_value', 'model_bias']
    ].copy()
    outliers['game_date'] = outliers['game_date'].dt.strftime('%m-%d')
    report.append(outliers.to_string(index=False))
    report.append("")

    # Finalize
    report_text = "\n".join(report)
    suffix = f"_{month_filter}" if month_filter else "_ALL"
    out_file = os.path.join(MODEL_DIR, "analyze", f"ANALYSIS{suffix}.txt")
    with open(out_file, "w") as f:
        f.write(report_text)
        
    print(f"\n[✓] Analysis Complete. Report saved to: {out_file}")
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", help="Month to filter (e.g. 01 for January, 02 for February)")
    args = parser.parse_args()
    
    run_analysis(month_filter=args.month)
