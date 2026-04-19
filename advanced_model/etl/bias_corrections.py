import sys
import os
import json

# Ensure the advanced_model directory is in the path for relative imports
MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

import sqlite3
import pandas as pd
from functools import lru_cache
from datetime import date
from config import settings

CORRECTIONS_METADATA_PATH = os.path.join(MODEL_DIR, 'config', 'bias_corrections_computed.json')


def should_recompute_corrections() -> bool:
    """
    Returns True if corrections are >30 days old or any tier has n<30.
    Used by run_pipeline.sh monthly scheduler to decide whether to auto-recompute.
    """
    if not os.path.exists(CORRECTIONS_METADATA_PATH):
        return True
    try:
        with open(CORRECTIONS_METADATA_PATH) as f:
            meta = json.load(f)
        last_computed_str = meta.get('computed_at', '2000-01-01')
        last_computed = date.fromisoformat(last_computed_str)
        if (date.today() - last_computed).days > 30:
            return True
        for key, val in meta.get('corrections', {}).items():
            if isinstance(val, dict) and val.get('n', 999) < 30:
                return True
        return False
    except Exception:
        return True  # recompute if the file is corrupt or unreadable

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'bet_tracker.db')

@lru_cache(maxsize=1)
def load_all_bias_corrections(db_path: str, min_samples: int = 30) -> dict:
    """
    Queries projection_outcomes for historical bias (actual - projected).
    Calculates mean bias per category and tier (bench vs starter).
    """
    if not os.path.exists(db_path):
        return {}

    # We need to compute the tier on the fly for each projection_outcome row
    # To do this efficiently, we pull the relevant data and tier it in Pandas
    query = """
        SELECT category, projected_value, model_bias
        FROM projection_outcomes
        WHERE source_mode = 'live'
    """
    
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        print(f"[!] Error loading bias corrections: {e}")
        return {}

    if df.empty:
        return {}

    def determine_tier(row):
        cat = row['category']
        val = row['projected_value']
        threshold = settings.PROJECTION_TIERS.get(cat)
        if threshold is None:
            return "neutral"
        return "bench" if val < threshold else "starter"

    df['tier'] = df.apply(determine_tier, axis=1)
    
    # Calculate group means and counts
    stats = df.groupby(['category', 'tier']).agg(
        mean_bias=('model_bias', 'mean'),
        count=('model_bias', 'count')
    ).reset_index()
    
    # Filter by min_samples and convert to dict
    corrections = {}
    for _, row in stats.iterrows():
        if row['count'] >= min_samples:
            corrections[(row['category'], row['tier'])] = row['mean_bias']
            
    return corrections

def get_tiered_bias(category: str, projection_value: float) -> float:
    """
    Returns the bias correction offset for a given category and projection level.
    """
    # Force upper case for keys
    cat = category.upper()
    
    # ISSUE 1 FIX (Stage 0 / Pre-Phase 6): TOTAL bias correction is DISABLED.
    # The correction was calibrated on historical-mode data (no injury matrix,
    # no synergy collapse) and was subtracting ~13.66 pts from totals that
    # live mode was ALREADY under-projecting by 8-22 pts.
    # Re-enable ONLY after recalibrating from retroactive_live data (Stage 3).
    if cat == 'TOTAL':
        return 0.0
    
    # Load (and cache) corrections
    corrections = load_all_bias_corrections(DB_PATH)
    
    # Determine tier
    threshold = settings.PROJECTION_TIERS.get(cat)
    if threshold is None:
        tier = "neutral"
    else:
        tier = "bench" if projection_value < threshold else "starter"
        
    # Lookup correction (fallback to 0.0)
    return corrections.get((cat, tier), 0.0)


if __name__ == "__main__":
    # Diagnostic run
    print("[+] Loading dynamic bias corrections...")
    bias_map = load_all_bias_corrections(DB_PATH, min_samples=1) # low threshold for test
    for (cat, tier), bias in bias_map.items():
        print(f"    {cat} [{tier}]: {bias:+.2f}")


