"""
==============================================================
  NBA MODEL — Dual Retrospective Analysis (Consolidated)
  Writes: newGenPredictions.txt  &  oldGenPredictions.txt
  Includes support for POINTS, REBOUNDS, ASSISTS, TOTAL, SPREAD.
==============================================================
"""

import sqlite3
import os
import sys
import pandas as pd
from datetime import datetime

MODEL_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NEW_DB    = os.path.join(MODEL_DIR, 'database', 'retrospective_db.db')
OLD_DB    = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')
OUT_DIR   = os.path.join(MODEL_DIR, 'analyze')

NEW_OUT   = os.path.join(OUT_DIR, 'newGenPredictions.txt')
OLD_OUT   = os.path.join(OUT_DIR, 'oldGenPredictions.txt')
COMP_OUT  = os.path.join(OUT_DIR, 'model_comparison.txt')

def analyse(rows, label):
    """Return stats dict + formatted report string for a set of rows."""
    total = len(rows)
    if total == 0:
        return None, f"[-] No data found for {label}\n"

    # stats holders
    by_cat   = {
        'POINTS': [], 'REBOUNDS': [], 'ASSISTS': [], 
        'TOTAL': [], 'SPREAD': []
    }
    by_month = {}
    mae_sum = bias_sum = ev_sum = 0.0
    hits = 0

    for date, player, cat, proj, line, actual, ov_odds, un_odds in rows:
        if proj is None or line is None or actual is None:
            continue

        # Basic MAE/Bias
        error = abs(actual - proj)
        mae_sum  += error
        bias_sum += (actual - proj)

        # Hit Logic
        hit = 0
        if cat == 'SPREAD':
            # actual_value is now the point margin (positive = covered)
            # Cover = actual margin beats the spread line
            hit = int(actual > line)
        else:
            # For stats and TOTAL, use Over/Under logic
            proj_over = proj > line
            actual_over = actual > line
            if proj_over == actual_over:
                hit = 1
        
        hits += hit

        # EV calculation (assuming standard odds if missing)
        odds = ov_odds if (cat != 'SPREAD' and proj > line) else un_odds
        # If it's spread, we don't have direction easily from the SQL without more columns,
        # but let's assume we picked the side that the model chose.
        # For simplicity, 1.909 default odds
        if not odds or odds < 1:
            odds = 1.909
        
        ev = hit * (odds - 1) - (1 - hit)
        ev_sum += ev

        if cat in by_cat:
            by_cat[cat].append((error, hit, ev, actual - proj))

        month = str(date)[:7]
        if month not in by_month:
            by_month[month] = {'mae': 0.0, 'hits': 0, 'n': 0}
        by_month[month]['mae']  += error
        by_month[month]['hits'] += hit
        by_month[month]['n']    += 1

    # Overall Metrics
    final_n = sum(len(v) for v in by_cat.values())
    if final_n == 0: 
        return None, "[-] Logic error: 0 rows survived filtering."

    mae   = mae_sum  / final_n
    bias  = bias_sum / final_n
    hr    = hits     / final_n * 100
    roi   = ev_sum   / final_n * 100

    stats = dict(total=final_n, mae=mae, bias=bias, hr=hr,
                 ev=ev_sum, roi=roi, by_cat=by_cat, by_month=by_month)

    sep = "=" * 64
    lines = [
        sep,
        f"  {label}",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        sep,
        f"  Total projections  : {final_n:,}",
        f"  Overall MAE        : {mae:.3f} units",
        f"  Average Bias       : {bias:+.3f} (+ = under-projected)",
        f"  Hit Rate           : {hr:.1f}%  ({hits:,}/{final_n:,})",
        f"  Cumulative EV      : {ev_sum:+.2f} units",
        f"  ROI per pick       : {roi:+.2f}%",
        "",
        f"  {'CATEGORY':<12} {'N':>7} {'MAE':>7} {'BIAS':>7} {'HIT%':>7} {'EV/pick':>9}",
        "  " + "-" * 55,
    ]
    for cat in ['POINTS', 'REBOUNDS', 'ASSISTS', 'TOTAL', 'SPREAD']:
        data = by_cat[cat]
        if not data:
            continue
        n      = len(data)
        c_mae  = float(sum(d[0] for d in data)) / n
        c_hit  = float(sum(d[1] for d in data)) / n * 100
        c_ev   = float(sum(d[2] for d in data)) / n
        c_bias = float(sum(d[3] for d in data)) / n
        lines.append(
            f"  {cat:<12} {n:>7,} {c_mae:>7.3f} {c_bias:>+7.3f} {c_hit:>6.1f}% {c_ev:>+9.4f}"
        )

    lines += ["", f"  {'MONTH':<10} {'N':>7} {'MAE':>7} {'HIT%':>7}", "  " + "-" * 36]
    for month in sorted(by_month):
        d = by_month[month]
        n = d['n']
        lines.append(f"  {month:<10} {n:>7,} {d['mae']/n:>7.3f} {d['hits']/n*100:>6.1f}%")

    if hr >= 55:
        verdict = "✅  POSITIVE EDGE — beats 50/50 baseline"
    elif hr >= 52:
        verdict = "🟡  MARGINAL EDGE — above baseline"
    else:
        verdict = "❌  NO EDGE — at or below coin-flip"

    lines += ["", f"  Verdict: {verdict}", sep, ""]
    return stats, "\n".join(lines)


def compare(new_s, old_s):
    """Return a side-by-side comparison block."""
    sep = "=" * 64
    lines = [
        "",
        sep,
        "  MODEL COMPARISON  —  New Gen  vs.  Old Gen",
        sep,
        f"  {'Metric':<28} {'NEW GEN':>12} {'OLD GEN':>12} {'DELTA':>10}",
        "  " + "-" * 64,
    ]
    metrics = [
        ("Total Projections",    'total', "{:,.0f}",  False),
        ("MAE (lower better)",   'mae',   "{:.3f}",   True),
        ("Average Bias",         'bias',  "{:+.3f}",  False),
        ("Hit Rate (%)",         'hr',    "{:.1f}%",  False),
        ("ROI per Pick (%)",     'roi',   "{:+.2f}%", False),
        ("Cumulative EV (Units)",'ev',    "{:+.2f}",  False),
    ]
    for name, key, fmt, lower_better in metrics:
        nv = new_s[key]
        ov = old_s[key]
        delta = nv - ov
        d_str = f"{delta:+.3f}" if isinstance(delta, float) else f"{delta:+,}"
        winner = "NEW" if (delta < 0 if lower_better else delta > 0) else "OLD"
        lines.append(
            f"  {name:<28} {fmt.format(nv):>12} {fmt.format(ov):>12} {d_str:>10}  {winner}"
        )

    lines += [sep, ""]
    return "\n".join(lines)


def main():
    new_query = """
        SELECT game_date, player, category,
               projected_value, bookmaker_line, actual_value,
               COALESCE(over_odds, 1.909) AS over_odds,
               COALESCE(under_odds, 1.909) AS under_odds
        FROM projection_outcomes
        WHERE actual_value IS NOT NULL
          AND projected_value IS NOT NULL
          AND bookmaker_line IS NOT NULL
    """

    old_query = """
        SELECT game_date, player, category,
               projected_value, bookmaker_line, actual_value,
               1.909 AS over_odds,
               1.909 AS under_odds
        FROM projection_outcomes
        WHERE actual_value IS NOT NULL
          AND projected_value IS NOT NULL
          AND bookmaker_line IS NOT NULL
    """

    print("[*] Analysing NEW GEN model …")
    conn_new = sqlite3.connect(NEW_DB)
    try:
        new_rows = conn_new.execute(new_query).fetchall()
        new_stats, new_report = analyse(new_rows, "NEW GEN MODEL  (retrospective_db.db)")
    except Exception as e:
        print(f"    [!] Error reading NEW GEN DB: {e}")
        new_stats, new_report = None, None
    finally:
        conn_new.close()

    print("[*] Analysing OLD GEN model …")
    conn_old = sqlite3.connect(OLD_DB)
    try:
        old_rows = conn_old.execute(old_query).fetchall()
        old_stats, old_report = analyse(old_rows, "OLD GEN MODEL  (bet_tracker.db)")
    except Exception as e:
        print(f"    [!] Error reading OLD GEN DB: {e}")
        old_stats, old_report = None, None
    finally:
        conn_old.close()

    if new_report:
        with open(NEW_OUT, 'w', encoding='utf-8') as f: f.write(new_report)
        print(f"[✓] Written → {NEW_OUT}")

    if old_report:
        with open(OLD_OUT, 'w', encoding='utf-8') as f: f.write(old_report)
        print(f"[✓] Written → {OLD_OUT}")

    if new_stats and old_stats:
        comp = compare(new_stats, old_stats)
        with open(COMP_OUT, 'w', encoding='utf-8') as f:
            f.write(new_report + old_report + comp)
        print(f"[✓] Full comparison written → {COMP_OUT}")
        print(comp)

if __name__ == '__main__':
    main()
