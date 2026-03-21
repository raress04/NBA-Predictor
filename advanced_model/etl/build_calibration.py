import sqlite3
import pandas as pd
import json
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression

# Ensure the advanced_model directory is in the path for relative imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

DB_PATH = os.path.join(ROOT_DIR, 'database', 'bet_tracker.db')
OUTPUT_PATH = os.path.join(ROOT_DIR, 'config', 'calibration_params.json')
MIN_CALIBRATION_SAMPLES = 200

def build_calibration():
    if not os.path.exists(DB_PATH):
        print(f"[-] DB not found at {DB_PATH}")
        return

    print(f"[+] Loading data from {DB_PATH}...")
    try:
        conn = sqlite3.connect(DB_PATH)
        # We use daily_picks and pick_results
        # sim_confidence is our raw probability (0-100)
        # is_win is our target (0 or 1)
        query = """
            SELECT dp.stat_category, dp.direction, dp.sim_confidence, pr.is_win
            FROM daily_picks dp
            JOIN pick_results pr ON pr.pick_id = dp.id
            WHERE pr.is_win IS NOT NULL
              AND dp.source_mode = 'live'
              AND dp.sim_confidence IS NOT NULL
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        print(f"[!] Error loading data: {e}")
        return

    if df.empty:
        print("[-] No resolved live picks found for calibration.")
        # Create empty file as per correct behaviour
        with open(OUTPUT_PATH, 'w') as f:
            json.dump({}, f)
        return

    calibration_params = {}

    # Group by category and direction
    groups = df.groupby(['stat_category', 'direction'])
    
    for (cat, direction), group in groups:
        n = len(group)
        if n < MIN_CALIBRATION_SAMPLES:
            print(f"[SKIP] {cat}_{direction}: only {n} samples, need {MIN_CALIBRATION_SAMPLES}")
            continue

        print(f"[+] Calibrating {cat}_{direction} ({n} samples)...")
        
        # X: raw/posterior prob (0-1), y: actual result (0/1)
        X = group['sim_confidence'].values.reshape(-1, 1) / 100.0
        y = group['is_win'].values
        
        try:
            lr = LogisticRegression()
            lr.fit(X, y)
            
            coef = float(lr.coef_[0][0])
            intercept = float(lr.intercept_[0])
            
            key = f"{cat.upper()}_{direction.upper()}"
            calibration_params[key] = {
                'coef': coef,
                'intercept': intercept,
                'n_samples': n
            }
        except Exception as e:
            print(f" [!] Error training model for {cat}_{direction}: {e}")

    # Write to JSON
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(calibration_params, f, indent=4)
    
    print(f"[+] Calibration parameters saved to {OUTPUT_PATH}")
    print(f"[+] Total categories calibrated: {len(calibration_params)}")

if __name__ == "__main__":
    build_calibration()
