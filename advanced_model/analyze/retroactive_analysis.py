import sqlite3
import pandas as pd
import numpy as np
import os
import sys

# Ensure project root is in path
MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

DB_PATH = os.path.join(MODEL_DIR, "database", "bet_tracker.db")

from config.settings import PROJECTION_TIERS

def run_analysis():
    print("[+] Connecting to Database...")
    conn = sqlite3.connect(DB_PATH)
    
    # 0. Load Data
    query = "SELECT * FROM projection_outcomes WHERE source_mode = 'retroactive_live'"
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print("[!] No data found for 'retroactive_live' source mode.")
        conn.close()
        return

    # Add Tier for analysis
    def get_tier(row):
        thresh = PROJECTION_TIERS.get(row['category'], 10)
        return "starter" if row['projected_value'] >= thresh else "bench"
    
    df['tier'] = df.apply(get_tier, axis=1)
    df['mae'] = df['model_bias'].abs()
    # Ensure game_date is datetime
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Report setup
    report = []
    report.append("NBA MODEL - GENERAL RETROACTIVE ANALYSIS (JAN 1 - MAR 20)")
    report.append("="*60)
    report.append(f"Analyzed {len(df)} projection rows.")
    report.append(f"Unique game dates covered: {df['game_date'].nunique()}")
    report.append("")

    # --- Question 1: Coverage & Sanity ---
    report.append("1. COVERAGE & INTEGRITY CHECK")
    report.append("-" * 40)
    
    # Broken rows check
    broken = df[abs(df['model_bias'] - (df['actual_value'] - df['projected_value'])) > 0.01]
    report.append(f"Integrity check (actual - proj == bias): {'[✓] PASS' if len(broken) == 0 else '[!] FAIL'}")
    if len(broken) > 0:
        report.append(f"    - Found {len(broken)} rows with inconsistent bias calculations.")
    
    coverage = df.groupby('category').agg(
        n=('category', 'count'),
        days=('game_date', 'nunique'),
        has_line=('bookmaker_line', lambda x: x.notnull().sum())
    ).reset_index()
    report.append("\nCoverage per Category:")
    report.append(coverage.to_string(index=False))
    report.append("")

    # --- Question 2: Monthly Drift ---
    report.append("2. MONTHLY BIAS & MAE DRIFT")
    report.append("-" * 40)
    
    df['month'] = df['game_date'].dt.strftime('%m-%B')
    drift = df.groupby(['category', 'month']).agg(
        avg_bias=('model_bias', 'mean'),
        mae=('mae', 'mean'),
        sample_size=('model_bias', 'count')
    ).reset_index()
    
    report.append(drift.to_string(index=False))
    report.append("")

    # --- Question 3: Tiered Analysis (Bench vs Starter) ---
    report.append("3. TIERED BIAS BREAKDOWN (Starter >= threshold)")
    report.append("-" * 40)
    
    tiered = df.groupby(['category', 'tier']).agg(
        mean_bias=('model_bias', 'mean'),
        mae=('mae', 'mean'),
        samples=('model_bias', 'count')
    ).reset_index()
    
    report.append(tiered.to_string(index=False))
    report.append("")

    # --- Question 4: Win Rate vs Real Lines ---
    report.append("4. WIN RATE VS BOOKMAKER LINES (HOW OFTEN PREDICTION BEATS LINE)")
    report.append("-" * 40)
    
    results = []
    for cat in ['POINTS', 'REBOUNDS', 'ASSISTS', 'TOTAL']:
        cat_df = df[df['category'] == cat].dropna(subset=['bookmaker_line'])
        if cat_df.empty: continue
        
        # Over Predictions
        overs = cat_df[cat_df['projected_value'] > cat_df['bookmaker_line']]
        o_win = (overs['actual_value'] > overs['bookmaker_line']).mean() * 100 if not overs.empty else 0
        
        # Under Predictions
        unders = cat_df[cat_df['projected_value'] < cat_df['bookmaker_line']]
        u_win = (unders['actual_value'] < unders['bookmaker_line']).mean() * 100 if not unders.empty else 0
        
        results.append({
            'Category': cat,
            'N_Overs': len(overs),
            'Win%_Over': f"{o_win:.1f}%",
            'N_Unders': len(unders),
            'Win%_Under': f"{u_win:.1f}%"
        })
        
    res_df = pd.DataFrame(results)
    report.append(res_df.to_string(index=False))
    report.append("")

    # --- Question 5: Extreme Outliers ---
    report.append("5. TOP 20 OUTLIER ERRORS (ABS BIAS DESC)")
    report.append("-" * 40)
    
    outliers = df.sort_values('mae', ascending=False).head(20)[
        ['game_date', 'player', 'category', 'projected_value', 'actual_value', 'model_bias']
    ]
    # Format dates
    outliers['game_date'] = outliers['game_date'].dt.strftime('%m-%d')
    report.append(outliers.to_string(index=False))
    report.append("")

    # --- Question 6: Total Calibration Gap ---
    report.append("6. THE 'GAP' ANALYSIS (Line - Prediction)")
    report.append("-" * 30)
    totals = df[df['category'] == 'TOTAL'].dropna(subset=['bookmaker_line']).copy()
    if not totals.empty:
        totals['gap'] = totals['bookmaker_line'] - totals['projected_value']
        gap_mean = totals['gap'].mean()
        gap_median = totals['gap'].median()
        report.append(f"Mean Gap (Line - Proj): {gap_mean:.2f} pts")
        report.append(f"Median Gap: {gap_median:.2f} pts")
        report.append(f"Max Line Overestimation: {totals['gap'].max():.1f} pts")
        report.append(f"Max Prediction Overestimation: {totals['gap'].min():.1f} pts")
    else:
        report.append("No TOTAL outcomes with lines found.")

    # Finalize
    report_text = "\n".join(report)
    out_file = os.path.join(MODEL_DIR, "analyze", "GENERAL_ANALYSIS.txt")
    with open(out_file, "w") as f:
        f.write(report_text)
        
    print(f"\n[✓] Analysis Complete. Report saved to: {out_file}")
    conn.close()

if __name__ == "__main__":
    run_analysis()
