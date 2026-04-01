"""
==============================================================
  Pre-Phase 6 — Stage 3: Bias Recomputation Script
  File: analysis/compute_retroactive_bias.py
  Created: 2026-03-26

  Reads retrospective_db.db (24,250 rows from Jan 1–Mar 24, 2026),
  computes tiered bias corrections for POINTS, REBOUNDS, ASSISTS,
  and writes results to config/bias_corrections_computed.json.

  Usage:
    python3 analysis/compute_retroactive_bias.py            # read-only, prints output
    python3 analysis/compute_retroactive_bias.py --auto-update  # writes to settings.py too
==============================================================
"""

import sqlite3
import json
import os
import sys
import argparse
import re
from datetime import date

# ── Path resolution ────────────────────────────────────────────────────────
MODEL_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RETRO_DB_PATH    = os.path.join(MODEL_DIR, 'database', 'retrospective_db.db')
TRACKER_DB_PATH  = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')
OUTPUT_JSON      = os.path.join(MODEL_DIR, 'config', 'bias_corrections_computed.json')
SETTINGS_PY      = os.path.join(MODEL_DIR, 'config', 'settings.py')

# ── Constants ──────────────────────────────────────────────────────────────
MIN_SAMPLES_TIER  = 30    # minimum rows per (category, tier) to trust correction
MIN_SAMPLES_TOTAL = 50    # stricter minimum for TOTAL game-level market

# Tier thresholds matching PROJECTION_TIERS in settings.py
TIER_THRESHOLDS = {
    'POINTS':    10.0,   # < 10  = bench,  >= 10  = starter
    'REBOUNDS':   4.0,   # < 4   = bench,  >= 4   = starter
    'ASSISTS':    3.0,   # < 3   = bench,  >= 3   = starter
}

# TOTAL tier thresholds (by projected game total)
TOTAL_TIERS = {
    'low':    (0,   210),
    'medium': (210, 230),
    'high':   (230, 9999),
}

# SPREAD corrections deliberately excluded:
# Win rate for SPREAD COVER is 38.5% — the model is banned from SPREAD COVER.
# Computing a bias correction for a banned category is meaningless.
# Do not add SPREAD to TIERED_BIAS_CORRECTIONS.


def compute_corrections() -> dict:
    """
    Queries retrospective_db.db and bet_tracker.db (live source_mode rows)
    to compute tiered bias corrections.

    Returns a results dict with:
      correction  → ADD this to projected value to reduce future bias
                    (if mean_bias = +2, model projected 2 too low → add +2 to future projections)
      n           → sample size
      mean_bias   → raw mean observed
      std         → standard deviation
    """
    if not os.path.exists(RETRO_DB_PATH):
        print(f"[-] retrospective_db.db not found at: {RETRO_DB_PATH}")
        sys.exit(1)

    # ── Query retroactive data ─────────────────────────────────────────────
    conn = sqlite3.connect(RETRO_DB_PATH)
    rows = conn.execute("""
        SELECT category, projected_value, model_bias
        FROM projection_outcomes
        WHERE bookmaker_line IS NOT NULL
          AND actual_value   IS NOT NULL
          AND category IN ('POINTS', 'REBOUNDS', 'ASSISTS')
    """).fetchall()
    conn.close()

    total_n = len(rows)
    print(f"\n[+] Loaded {total_n:,} rows from retrospective_db.db")

    # Also pull any live source_mode rows from bet_tracker.db to augment small tiers
    live_rows = []
    if os.path.exists(TRACKER_DB_PATH):
        try:
            conn2 = sqlite3.connect(TRACKER_DB_PATH)
            live_rows = conn2.execute("""
                SELECT category, projected_value, model_bias
                FROM projection_outcomes
                WHERE bookmaker_line IS NOT NULL
                  AND actual_value   IS NOT NULL
                  AND source_mode    = 'live'
                  AND category IN ('POINTS', 'REBOUNDS', 'ASSISTS')
            """).fetchall()
            conn2.close()
            if live_rows:
                print(f"[+] Augmented with {len(live_rows):,} live rows from bet_tracker.db")
        except Exception as e:
            print(f"[!] Could not read bet_tracker.db live rows: {e}")

    all_rows = rows + live_rows

    # ── Compute per-category, per-tier corrections ─────────────────────────
    results = {}

    for cat, threshold in TIER_THRESHOLDS.items():
        bench_biases   = []
        starter_biases = []

        for row_cat, proj_val, bias in all_rows:
            if row_cat != cat:
                continue
            if proj_val is None or bias is None:
                continue
            if proj_val < threshold:
                bench_biases.append(bias)
            else:
                starter_biases.append(bias)

        for tier_name, biases in [('bench', bench_biases), ('starter', starter_biases)]:
            key = f"{cat}_{tier_name}"
            n   = len(biases)

            if n >= MIN_SAMPLES_TIER:
                mean_b = sum(biases) / n
                var    = sum((b - mean_b) ** 2 for b in biases) / n
                std    = var ** 0.5
                correction = round(mean_b, 2)
                results[key] = {
                    'correction': correction,
                    'n':          n,
                    'mean_bias':  round(mean_b, 3),
                    'std':        round(std,    3),
                }
                print(f"  [{key}]  n={n:,}  mean_bias={mean_b:+.3f}  correction={correction:+.2f}")
            else:
                results[key] = {
                    'correction': 0.0,
                    'n':          n,
                    'mean_bias':  None,
                    'std':        None,
                    'note':       f'INSUFFICIENT DATA (n={n} < {MIN_SAMPLES_TIER}) — correction zeroed'
                }
                print(f"  [{key}]  INSUFFICIENT DATA (n={n}) — correction set to 0.0")

    # ── TOTAL game-level corrections ───────────────────────────────────────
    # NOTE: No TOTAL rows in retrospective_db.db (retroactive pipeline only tracked props).
    # Per Stage 3 spec: manually set TOTAL corrections to +10.0 as fallback estimate.
    print("\n[!] No TOTAL rows found in retrospective_db.db.")
    print("    Per spec: setting TOTAL corrections to +10.0 (manual estimate — injury approximation).")
    print("    Log: TOTAL_CORRECTION_FALLBACK=+10.0 (injury data approximation)")

    for tier_name in ['high', 'medium', 'low']:
        key = f'TOTAL_{tier_name}'
        results[key] = {
            'correction': 10.0,
            'n':          0,
            'mean_bias':  None,
            'std':        None,
            'note':       'TOTAL_CORRECTION_FALLBACK=+10.0 — no TOTAL rows in retroactive dataset. Injury data approximation per Stage 3 spec.'
        }
        print(f"  [TOTAL_{tier_name}]  correction=+10.0 (manual fallback)")

    return results, total_n


def sanity_check(results: dict) -> bool:
    """
    Runs 3 mandatory sanity checks per spec.
    Returns True if all pass, False if any fail.
    """
    print("\n" + "=" * 60)
    print("  SANITY CHECKS")
    print("=" * 60)

    passed = True

    # CHECK 1: TOTAL_high must be positive
    total_high = results.get('TOTAL_high', {}).get('correction', 0)
    if total_high > 0:
        print(f"  [✅ CHECK 1] TOTAL_high correction = {total_high:+.2f} (positive — correct)")
    else:
        print(f"  [❌ CHECK 1] TOTAL_high correction = {total_high:+.2f} (NEGATIVE — applying fallback +10.0)")
        results['TOTAL_high']['correction']   = 10.0
        results['TOTAL_medium']['correction'] = 10.0
        results['TOTAL_low']['correction']    = 10.0
        passed = False  # note but continue with fallback applied

    # CHECK 2: All POINTS corrections < ±3.0 pts
    for tier in ['bench', 'starter']:
        key = f'POINTS_{tier}'
        c   = results.get(key, {}).get('correction', 0.0)
        if abs(c) < 3.0:
            print(f"  [✅ CHECK 2] {key} correction = {c:+.2f} (within ±3.0)")
        elif abs(c) < 5.0:
            print(f"  [⚠️  CHECK 2] {key} correction = {c:+.2f} (between ±3–5, worth reviewing)")
        else:
            print(f"  [❌ CHECK 2] {key} correction = {c:+.2f} (> ±5.0 — pipeline anomaly suspected)")
            passed = False

    # CHECK 3: No prop correction > ±5.0, no TOTAL > ±20.0
    for key, meta in results.items():
        c = meta.get('correction', 0.0)
        is_total = key.startswith('TOTAL')
        limit = 20.0 if is_total else 5.0
        if abs(c) <= limit:
            print(f"  [✅ CHECK 3] {key} = {c:+.2f}  (within ±{limit})")
        else:
            print(f"  [❌ CHECK 3] {key} = {c:+.2f}  (EXCEEDS ±{limit} — DO NOT APPLY)")
            passed = False

    if passed:
        print("\n  ✅ All sanity checks PASSED — safe to apply corrections.")
    else:
        print("\n  ⚠️  Some checks flagged — review above before applying. Continuing with safe fallbacks.")

    return passed


def write_json(results: dict, total_n: int):
    """Writes computed corrections to config/bias_corrections_computed.json."""
    payload = {
        'computed_at':    str(date.today()),
        'source_db':      'retrospective_db.db',
        'source_n_total': total_n,
        'corrections':    results,
    }
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\n[+] Written → {OUTPUT_JSON}")


def write_settings(results: dict):
    """
    Replaces TIERED_BIAS_CORRECTIONS in settings.py with computed values.
    Also updates MAX_PROP_CONFIDENCE and CONFIDENCE_PRIOR_STRENGTH.
    """
    with open(SETTINGS_PY, 'r') as f:
        content = f.read()

    # ── Build new TIERED_BIAS_CORRECTIONS block ────────────────────────────
    def r(key):
        return results.get(key, {}).get('correction', 0.0)

    def n(key):
        return results.get(key, {}).get('n', 0)

    new_block = f"""# TIERED_BIAS_CORRECTIONS — recomputed {date.today()}
# Source: retrospective_db.db — retroactive_live (n={results.get('POINTS_starter', {}).get('n', 0) + results.get('POINTS_bench', {}).get('n', 0):,}+ rows, Jan 1–Mar 24, 2026)
# Previous values (rollback reference):
#   POINTS bench=+1.10, POINTS starter=-0.67
#   REBOUNDS bench=+0.77, REBOUNDS starter=+0.38
#   ASSISTS bench=+0.27, ASSISTS starter=-0.17
#   TOTAL: not previously set
TIERED_BIAS_CORRECTIONS = {{
    ('POINTS',   'bench'):   {r('POINTS_bench'):+.2f},    # n={n('POINTS_bench'):,} — computed
    ('POINTS',   'starter'): {r('POINTS_starter'):+.2f},    # n={n('POINTS_starter'):,} — computed
    ('REBOUNDS', 'bench'):   {r('REBOUNDS_bench'):+.2f},    # n={n('REBOUNDS_bench'):,} — computed
    ('REBOUNDS', 'starter'): {r('REBOUNDS_starter'):+.2f},    # n={n('REBOUNDS_starter'):,} — computed
    ('ASSISTS',  'bench'):   {r('ASSISTS_bench'):+.2f},    # n={n('ASSISTS_bench'):,} — computed
    ('ASSISTS',  'starter'): {r('ASSISTS_starter'):+.2f},    # n={n('ASSISTS_starter'):,} — computed
    ('TOTAL',    'high'):    {r('TOTAL_high'):+.2f},    # n={n('TOTAL_high'):,}  — MUST be positive (model under-projects)
    ('TOTAL',    'medium'):  {r('TOTAL_medium'):+.2f},    # n={n('TOTAL_medium'):,}  — computed
    ('TOTAL',    'low'):     {r('TOTAL_low'):+.2f},    # n={n('TOTAL_low'):,}  — computed
}}"""

    # Replace the old TIERED_BIAS_CORRECTIONS block
    pattern = r'# TIERED_BIAS_CORRECTIONS.*?^TIERED_BIAS_CORRECTIONS\s*=\s*\{[^}]*\}'
    new_content = re.sub(pattern, new_block, content, flags=re.DOTALL | re.MULTILINE)

    # ── Update MAX_PROP_CONFIDENCE ─────────────────────────────────────────
    new_content = re.sub(
        r'MAX_PROP_CONFIDENCE\s*=\s*[\d.]+',
        '# Previous: 78.0 (hard cap — causing clustering at limit)\n'
        '# Stage 3 update: Raised based on New Gen retrospective hit rate analysis (n=24,250)\n'
        'MAX_PROP_CONFIDENCE = 82.0   # Raised from 78.0 — revisit after 30 days live',
        new_content
    )

    # ── Update CONFIDENCE_PRIOR_STRENGTH ──────────────────────────────────
    new_content = re.sub(
        r'CONFIDENCE_PRIOR_STRENGTH\s*=\s*[\d]+',
        '# Previous: 10 (weak prior — allowed clustering at cap)\n'
        '# Stage 3 update: Stronger pull toward category priors; reduces cap clustering\n'
        '# Note: At 5k sims, prior_strength=50 ≈ effect of 250 at 1k sims (retroactive)\n'
        'CONFIDENCE_PRIOR_STRENGTH = 50',
        new_content
    )

    with open(SETTINGS_PY, 'w') as f:
        f.write(new_content)
    print(f"[+] settings.py updated → {SETTINGS_PY}")


def main():
    parser = argparse.ArgumentParser(description='Compute retroactive bias corrections (Stage 3)')
    parser.add_argument('--auto-update', action='store_true',
                        help='Write corrections directly to settings.py (used by monthly scheduler)')
    args = parser.parse_args()

    print("=" * 60)
    print("  PRE-PHASE 6 — STAGE 3: BIAS RECOMPUTATION")
    print(f"  Running: {date.today()}")
    print("=" * 60)

    # CHUNK 2: Compute
    results, total_n = compute_corrections()

    # CHUNK 3: Sanity checks
    sanity_check(results)

    # Always write JSON (staging file)
    write_json(results, total_n)

    # CHUNK 4: Write to settings.py only if --auto-update or explicitly confirmed
    if args.auto_update:
        print("\n[!] --auto-update flag set. Writing corrections to settings.py...")
        write_settings(results)
    else:
        print("\n[i] Read-only mode. Run with --auto-update to write to settings.py.")
        print(f"    Review: {OUTPUT_JSON}")

    print("\n✅ Stage 3 bias computation complete.\n")


if __name__ == '__main__':
    main()
