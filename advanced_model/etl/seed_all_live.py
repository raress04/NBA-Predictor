import os
import subprocess
import glob
import sqlite3
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')
PREDICTIONS_DIR = os.path.join(MODEL_DIR, 'predictions')
MATCH_MONTHS = ["march", "april", "may"]  # extended to cover full live season

def get_date_from_filename(filename):
    """Extracts YYYY-MM-DD from predicts_YYYY-MM-DD.txt."""
    basename = os.path.basename(filename)
    # predicts_2026-03-04.txt
    parts = basename.replace(".txt", "").split("_")
    if len(parts) >= 2:
        return parts[1]
    return None

def is_already_seeded(date_str, filename):
    """Checks if the file has already been seeded into daily_picks."""
    if not os.path.exists(DB_PATH):
        return False
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM daily_picks WHERE date=? AND source_file=? LIMIT 1", 
                    (date_str, os.path.basename(filename)))
        exists = cur.fetchone() is not None
        conn.close()
        return exists
    except sqlite3.OperationalError:
        # Table might not exist yet
        return False

def main():
    print(f"[*] Scanning for live-mode reports in {PREDICTIONS_DIR}...")
    
    for month in MATCH_MONTHS:
        month_dir = os.path.join(PREDICTIONS_DIR, month)
        if not os.path.exists(month_dir):
            continue
        
        files = glob.glob(os.path.join(month_dir, "predicts_*.txt"))
        files.sort()
        
        for f in files:
            date_str = get_date_from_filename(f)
            if not date_str:
                continue
            
            if is_already_seeded(date_str, f):
                print(f"  [-] Skipping {date_str} (already seeded)")
                continue
            
            print(f"  [+] Seeding {date_str} from {os.path.basename(f)}...")
            # Call: python -m advanced_model.etl.bet_tracker seed --date <date_str>
            cmd = ["python3", "-m", "advanced_model.etl.db_manager", "seed", "--date", date_str]
            parent_dir = os.path.dirname(MODEL_DIR)
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=parent_dir)
            
            if result.returncode == 0:
                print(f"      Successfully seeded.")
            else:
                print(f"      [!] Error seeding {date_str}:")
                print(result.stderr)

if __name__ == "__main__":
    main()
