"""
ml/evaluate_shadow.py
======================
P1-4: Evaluate XGB Shadow Picks vs Raw Sim.

Reads ml_shadow_picks from bet_tracker.db, computes Brier score and
calibration for both sim_confidence and xgb_confidence on resolved picks.
Outputs a promotion recommendation when >= MIN_PICKS resolved picks exist.

Usage:
    py -m ml.evaluate_shadow [--min-picks N] [--no-save]

Output:
    logs/shadow_evaluation_YYYY-MM-DD.json
    Printed report to stdout

Prerequisite: P1-3 (shadow_logger.py) must be accumulating data via
live_ingestion.py for ~4 weeks (~200 resolved picks needed).
"""

import os
import json
import sqlite3
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH   = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')
LOG_DIR   = os.path.join(MODEL_DIR, 'logs')

MIN_PICKS_DEFAULT = 200   # Platt gate: need ~200 resolved picks for stable eval
BRIER_GATE        = 0.005 # XGB must beat sim by this margin (absolute Brier) to promote


# ── Metric helpers ─────────────────────────────────────────────────────────────

def brier_score(y_true, y_prob):
    """Mean squared error between binary outcome and predicted probability."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    return float(np.mean((y_prob - y_true) ** 2))


def calibration_bias(y_true, y_prob, n_bins=5):
    """
    Mean signed deviation (predicted - actual) across quantile bins.
    Positive = overconfident, negative = underconfident.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    bins = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    bin_idx = np.digitize(y_prob, bins[1:-1])
    biases = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() < 5:
            continue
        biases.append(float(y_prob[mask].mean() - y_true[mask].mean()))
    return float(np.mean(biases)) if biases else 0.0


def roi_estimate(y_true, y_prob, odds=1.91):
    """
    Estimated flat-stake ROI at given decimal odds.
    roi = mean(y_true * (odds - 1)) - mean(1 - y_true)
    """
    y_true = np.asarray(y_true, dtype=float)
    return float(y_true.mean() * (odds - 1) - (1 - y_true.mean()))


# ── Main evaluation ────────────────────────────────────────────────────────────

def evaluate(min_picks: int = MIN_PICKS_DEFAULT, save: bool = True):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT game_date, player, stat_category, direction,
               line, projection, sim_confidence, xgb_confidence,
               edge_pct, actual_value, is_win
        FROM ml_shadow_picks
        WHERE is_win IS NOT NULL
          AND sim_confidence IS NOT NULL
          AND actual_value IS NOT NULL
          AND (cohort IS NULL OR cohort = 'selected')
    """, conn)
    conn.close()

    total_resolved = len(df)

    print("=" * 60)
    print("  ML SHADOW EVALUATION  (P1-4)")
    print("=" * 60)
    print(f"  Resolved picks: {total_resolved}  (need >= {min_picks} for full eval)")

    if total_resolved == 0:
        print("\n  [!] No resolved picks yet.")
        print(f"  [!] Run live ingestion daily — picks log automatically.")
        _print_eta(0, min_picks)
        return None

    if total_resolved < min_picks:
        pct = total_resolved / min_picks * 100
        print(f"\n  [!] {total_resolved}/{min_picks} picks resolved ({pct:.0f}%). "
              f"Not enough for promotion decision.")
        _print_eta(total_resolved, min_picks)
        _print_partial_metrics(df)
        return None

    # ── Full evaluation ──────────────────────────────────────────────────────
    y_true    = df['is_win'].values
    sim_probs = df['sim_confidence'].values / 100.0

    results = {
        'eval_date':      datetime.now().strftime('%Y-%m-%d'),
        'total_resolved': total_resolved,
        'overall':        {},
        'by_category':    {},
        'promotion':      None,
    }

    sim_brier = brier_score(y_true, sim_probs)
    sim_bias  = calibration_bias(y_true, sim_probs)
    sim_roi   = roi_estimate(y_true)

    print(f"\n  ── OVERALL ({total_resolved} picks) ──")
    print(f"  Hit rate (actual):    {y_true.mean()*100:.1f}%")
    print(f"  Sim Brier:            {sim_brier:.5f}  (bias: {sim_bias:+.4f})")
    print(f"  Flat-stake ROI est:   {sim_roi*100:+.2f}%")

    results['overall'].update({
        'sim_brier': sim_brier, 'sim_bias': sim_bias,
        'hit_rate': float(y_true.mean()), 'roi': sim_roi,
    })

    # XGB overall
    xgb_n = df['xgb_confidence'].notna().sum()
    if xgb_n >= 50:
        df_xgb   = df[df['xgb_confidence'].notna()].copy()
        xgb_prob = df_xgb['xgb_confidence'].values / 100.0
        yt_xgb   = df_xgb['is_win'].values
        sp_xgb   = df_xgb['sim_confidence'].values / 100.0

        xgb_brier    = brier_score(yt_xgb, xgb_prob)
        xgb_bias     = calibration_bias(yt_xgb, xgb_prob)
        sim_brier_x  = brier_score(yt_xgb, sp_xgb)
        improvement  = sim_brier_x - xgb_brier

        print(f"  XGB Brier:            {xgb_brier:.5f}  (bias: {xgb_bias:+.4f})")
        print(f"  Improvement vs sim:   {improvement:+.5f}  (gate >= +{BRIER_GATE:.3f})")

        results['overall'].update({
            'xgb_brier': xgb_brier, 'xgb_bias': xgb_bias,
            'improvement': improvement, 'xgb_n': int(xgb_n),
        })
    else:
        print(f"  XGB samples: {xgb_n} (< 50 — XGB models may not be saved yet)")
        improvement = 0.0
        results['overall']['xgb_n'] = int(xgb_n)

    # ── Per-category breakdown ────────────────────────────────────────────────
    print(f"\n  ── BY CATEGORY ──")
    cat_verdicts = {}

    for cat in ['POINTS', 'REBOUNDS', 'ASSISTS']:
        sub = df[df['stat_category'] == cat]
        n = len(sub)
        if n < 20:
            print(f"  {cat:<12} N={n:3d}  (< 20 — skip)")
            results['by_category'][cat] = {'n': n, 'verdict': 'INSUFFICIENT_DATA'}
            continue

        yt = sub['is_win'].values
        sp = sub['sim_confidence'].values / 100.0
        s_brier = brier_score(yt, sp)
        s_bias  = calibration_bias(yt, sp)

        sub_xgb = sub[sub['xgb_confidence'].notna()]
        nx = len(sub_xgb)

        if nx >= 20:
            xp   = sub_xgb['xgb_confidence'].values / 100.0
            yt_x = sub_xgb['is_win'].values
            sp_x = sub_xgb['sim_confidence'].values / 100.0
            x_brier = brier_score(yt_x, xp)
            imp     = brier_score(yt_x, sp_x) - x_brier
            gate    = 'PROMOTE' if imp >= BRIER_GATE else 'HOLD'
            cat_verdicts[cat] = gate

            print(f"  {cat:<12} N={n:3d}  "
                  f"sim={s_brier:.4f}  xgb={x_brier:.4f}  "
                  f"imp={imp:+.4f}  [{gate}]")
            results['by_category'][cat] = {
                'n': n, 'xgb_n': nx,
                'sim_brier': s_brier, 'sim_bias': s_bias,
                'xgb_brier': x_brier, 'improvement': imp, 'verdict': gate,
            }
        else:
            print(f"  {cat:<12} N={n:3d}  sim={s_brier:.4f}  bias={s_bias:+.4f}  "
                  f"(xgb_n={nx} < 20)")
            results['by_category'][cat] = {
                'n': n, 'xgb_n': nx,
                'sim_brier': s_brier, 'sim_bias': s_bias,
                'verdict': 'INSUFFICIENT_XGB',
            }

    # ── Promotion decision ────────────────────────────────────────────────────
    print(f"\n  ── PROMOTION DECISION ──")
    promotable = [c for c, v in cat_verdicts.items() if v == 'PROMOTE']
    holdback   = [c for c, v in cat_verdicts.items() if v == 'HOLD']

    if promotable:
        verdict = f"PROMOTE_PARTIAL ({', '.join(promotable)})"
        action  = (
            f"Set USE_ML_CALIBRATION=True in settings.py. "
            f"XGB beats sim by >= {BRIER_GATE} on: {', '.join(promotable)}. "
            f"Monitor holdback categories: {', '.join(holdback) or 'none'}."
        )
    elif cat_verdicts and all(v == 'HOLD' for v in cat_verdicts.values()):
        verdict = "HOLD"
        action  = (
            f"XGB does not improve Brier by >= {BRIER_GATE} on any category. "
            f"Keep USE_ML_CALIBRATION=False. Re-train with more data."
        )
    else:
        verdict = "INSUFFICIENT_DATA"
        action  = "Need >= 20 resolved picks with XGB values per category."

    results['promotion'] = {
        'verdict': verdict, 'action': action,
        'promotable': promotable, 'holdback': holdback,
        'brier_gate': BRIER_GATE,
    }

    print(f"  Verdict: {verdict}")
    print(f"  Action:  {action}")
    print("=" * 60)

    if save:
        os.makedirs(LOG_DIR, exist_ok=True)
        out_path = os.path.join(
            LOG_DIR, f"shadow_evaluation_{datetime.now().strftime('%Y-%m-%d')}.json"
        )
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\n[+] Saved: {out_path}")

    return results


def _print_eta(current: int, target: int):
    remaining = max(0, target - current)
    days = remaining / 10  # ~10 prop picks/day conservative
    print(f"\n  ETA at ~10 picks/day: ~{days:.0f} more days (~{days/7:.1f} weeks)")
    print(f"  Run: py -m ml.evaluate_shadow once {target}+ picks are resolved.")


def _print_partial_metrics(df: pd.DataFrame):
    """Interim metrics printed when below threshold."""
    y_true    = df['is_win'].values
    sim_probs = df['sim_confidence'].values / 100.0
    print(f"\n  ── INTERIM METRICS ({len(df)} picks) ──")
    print(f"  Hit rate:  {y_true.mean()*100:.1f}%")
    print(f"  Sim Brier: {brier_score(y_true, sim_probs):.5f}")
    for cat in ['POINTS', 'REBOUNDS', 'ASSISTS']:
        sub = df[df['stat_category'] == cat]
        if len(sub) > 0:
            yt = sub['is_win'].values
            sp = sub['sim_confidence'].values / 100.0
            print(f"  {cat:<12} N={len(sub):3d}  "
                  f"hit={yt.mean()*100:.1f}%  brier={brier_score(yt, sp):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ML shadow picks (P1-4)')
    parser.add_argument('--min-picks', type=int, default=MIN_PICKS_DEFAULT,
                        help=f'Min resolved picks for full eval (default: {MIN_PICKS_DEFAULT})')
    parser.add_argument('--no-save', action='store_true', help='Skip saving JSON log')
    args = parser.parse_args()
    evaluate(min_picks=args.min_picks, save=not args.no_save)
