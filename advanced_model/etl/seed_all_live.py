"""Seed live predictions files into bet_tracker.daily_picks.

Each live predicts_YYYY-MM-DD.txt is parsed for its VALUABLE picks, which go into
daily_picks (source_mode='live'). The next morning, prepare_analyze grades them into
pick_results.

Until 2026-11-11 only the 2025-26 folders (predictions/march|april|may) were scanned,
so no 2026-27 pick ever reached daily_picks, and the monitoring report's win rates and
Kelly counts silently covered last season only.
"""
import os
import glob
import sqlite3

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')
PREDICTIONS_DIR = os.path.join(MODEL_DIR, 'predictions')
LEGACY_MONTHS = ["march", "april", "may"]   # 2025-26 layout: predictions/<month>/


def get_date_from_filename(filename):
    """Extracts YYYY-MM-DD from predicts_YYYY-MM-DD.txt."""
    parts = os.path.basename(filename).replace(".txt", "").split("_")
    return parts[1] if len(parts) >= 2 else None


def live_prediction_files(predictions_dir=PREDICTIONS_DIR):
    """All live predicts_*.txt files, legacy layout and season layout, oldest first.
    Backtest folders are never included."""
    files = []
    for month in LEGACY_MONTHS:
        files += glob.glob(os.path.join(predictions_dir, month, "predicts_*.txt"))
    # Season layout: predictions/<season>/live_predictions/<month>/predicts_*.txt
    files += glob.glob(os.path.join(predictions_dir, "*", "live_predictions", "*", "predicts_*.txt"))
    files = [f for f in files if get_date_from_filename(f)]
    return sorted(set(files), key=lambda f: get_date_from_filename(f))


def is_already_seeded(date_str, filename, db_path=DB_PATH):
    """Checks if the file has already been seeded into daily_picks."""
    if not os.path.exists(db_path):
        return False
    try:
        conn = sqlite3.connect(db_path)
        exists = conn.execute(
            "SELECT 1 FROM daily_picks WHERE date=? AND source_file=? LIMIT 1",
            (date_str, os.path.basename(filename))).fetchone() is not None
        conn.close()
        return exists
    except sqlite3.OperationalError:
        return False  # table might not exist yet


def _results_path(date_str):
    try:
        from utils.prediction_paths import get_results_filepath
        return get_results_filepath(date_str, mode='live', base_dir=MODEL_DIR)
    except Exception:
        return ''


def main():
    from etl.db_manager import seed_from_txt

    print(f"[*] Scanning for live-mode reports in {PREDICTIONS_DIR}...")
    for f in live_prediction_files():
        date_str = get_date_from_filename(f)
        if is_already_seeded(date_str, f):
            continue
        print(f"  [+] Seeding {date_str} from {os.path.relpath(f, MODEL_DIR)}...")
        try:
            seed_from_txt(f, _results_path(date_str), date_str)
        except Exception as e:
            print(f"      [!] Error seeding {date_str}: {e}")


if __name__ == "__main__":
    main()
