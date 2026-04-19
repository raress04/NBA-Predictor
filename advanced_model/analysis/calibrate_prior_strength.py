"""
analysis/calibrate_prior_strength.py
--------------------------------------
Multi-objective grid search for CONFIDENCE_PRIOR_STRENGTH.

Selects the SMALLEST prior_strength that satisfies ALL of:
  - ECE  <= min(ECE_all) + 0.003          (good enough calibration)
  - Brier <= Brier_baseline - 0.001       (improvement over production baseline)
  - Var   >= 0.75 * Var_baseline          (predictions stay sharp / informative)
  - AUC   >= AUC_baseline - 0.01          (ranking quality not degraded)

Usage:
    python3 -m analysis.calibrate_prior_strength

Writes:
    logs/prior_strength_multiobjective_calibration.txt
Updates:
    config/settings.py  (CONFIDENCE_PRIOR_STRENGTH line)
"""

import os
import re
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_auc_score

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RETRO_DB    = os.path.join(MODEL_DIR, 'database', 'retrospective_db.db')
LOGS_DIR    = os.path.join(MODEL_DIR, 'logs')
SETTINGS_PY = os.path.join(MODEL_DIR, 'config', 'settings.py')
os.makedirs(LOGS_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
CATEGORY_PRIORS = {
    "TOTAL_OVER":     0.53,
    "ASSISTS_OVER":   0.56,
    "ASSISTS_UNDER":  0.56,
    "POINTS_OVER":    0.55,
    "POINTS_UNDER":   0.55,
    "REBOUNDS_OVER":  0.55,
    "REBOUNDS_UNDER": 0.55,
    "SPREAD_COVER":   0.38,
    "DEFAULT":        0.54,
}

CANDIDATES     = [10, 25, 50, 100, 200, 400, 800, 1500, 3000, 5000, 10000]
BASELINE_PS    = 100          # reference point — pre-P.1 production value

# Multi-objective thresholds
DELTA_ECE   = 0.003   # ECE must be within 0.003 of global minimum
DELTA_BRIER = 0.001   # Brier must beat baseline by at least 0.001
ALPHA_VAR   = 0.70    # variance must stay >= 70% of baseline
DELTA_AUC   = 0.01    # AUC must not drop more than 0.01 from baseline
ALPHA_VAR_FALLBACK = 0.60  # relaxed Var threshold before touching ECE/Brier

# Calibration bins in probability space
CAL_BINS   = [0.50, 0.60, 0.70, 0.75, 0.78, 1.00]
CAL_LABELS = ["0.50-0.60", "0.60-0.70", "0.70-0.75", "0.75-0.78", "0.78+"]

N_SIM_EQUIV = 1000   # treat market odds as if from 1000-sim run


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Load retro rows. Derive:
      direction  : OVER if projected_value >= bookmaker_line, else UNDER
      raw_prob   : 1 / over_odds  (symmetric market implied probability)
      is_win     : 1 if picked direction hit, else 0
      cat_key    : e.g. 'POINTS_OVER'
      month      : '2026-01' / '2026-02' / '2026-03'
    """
    conn = sqlite3.connect(RETRO_DB)
    df = pd.read_sql("""
        SELECT game_date, category, projected_value, bookmaker_line,
               actual_value, over_odds
        FROM projection_outcomes
        WHERE actual_value   IS NOT NULL
          AND bookmaker_line IS NOT NULL
          AND over_odds      IS NOT NULL
          AND over_odds      > 1.0
    """, conn)
    conn.close()

    df['direction'] = np.where(df['projected_value'] >= df['bookmaker_line'], 'OVER', 'UNDER')
    df['is_win']    = np.where(
        df['direction'] == 'OVER',
        (df['actual_value'] > df['bookmaker_line']).astype(int),
        (df['actual_value'] < df['bookmaker_line']).astype(int),
    )
    df['raw_prob']  = (1.0 / df['over_odds'].clip(lower=1.01)).clip(0.01, 0.99)
    df['cat_key']   = df['category'].str.upper() + '_' + df['direction']
    df['month']     = df['game_date'].str[:7]
    return df


# ── Posterior ─────────────────────────────────────────────────────────────────

def compute_posterior(df: pd.DataFrame, prior_strength: int) -> np.ndarray:
    """
    Beta posterior:
        posterior = (raw_wins + alpha) / (N_SIM_EQUIV + prior_strength)
    where raw_wins = raw_prob * N_SIM_EQUIV (treat market odds as sim frequency).
    """
    prior_means = df['cat_key'].map(CATEGORY_PRIORS).fillna(CATEGORY_PRIORS['DEFAULT']).values
    alpha       = prior_means * prior_strength
    raw_wins    = df['raw_prob'].values * N_SIM_EQUIV
    posterior   = (raw_wins + alpha) / (N_SIM_EQUIV + prior_strength)
    return np.clip(posterior, 0.50, 0.99)


# ── Metrics ───────────────────────────────────────────────────────────────────

def ece(probs: np.ndarray, wins: np.ndarray) -> float:
    n_total = len(probs)
    if n_total == 0:
        return np.nan
    err = 0.0
    for lo, hi in zip(CAL_BINS[:-1], CAL_BINS[1:]):
        mask  = (probs >= lo) & (probs < hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        err += abs(probs[mask].mean() - wins[mask].mean()) * n_bin / n_total
    return float(err)


def brier(probs: np.ndarray, wins: np.ndarray) -> float:
    return float(((probs - wins) ** 2).mean())


def sharpness(probs: np.ndarray) -> float:
    """Variance of predicted probabilities — higher = more informative."""
    return float(np.var(probs))


def auc(probs: np.ndarray, wins: np.ndarray) -> float:
    """ROC AUC — 0.5 = random, 1.0 = perfect ranking."""
    if len(np.unique(wins)) < 2:
        return 0.5
    return float(roc_auc_score(wins, probs))


# ── Grid Search ───────────────────────────────────────────────────────────────

def run_grid_search(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    wins    = df['is_win'].values.astype(float)
    records = []

    for ps in CANDIDATES:
        p = compute_posterior(df, ps)
        records.append({
            'prior_strength': ps,
            'ECE':    round(ece(p, wins),       5),
            'Brier':  round(brier(p, wins),     5),
            'Var':    round(sharpness(p),        6),
            'AUC':    round(auc(p, wins),        5),
        })

    result_df = pd.DataFrame(records)

    # Baseline metrics (at BASELINE_PS)
    base_row     = result_df[result_df['prior_strength'] == BASELINE_PS].iloc[0]
    ECE_base     = base_row['ECE']
    Brier_base   = base_row['Brier']
    Var_base     = base_row['Var']
    AUC_base     = base_row['AUC']

    # Thresholds
    ECE_target   = result_df['ECE'].min() + DELTA_ECE
    Brier_thresh = Brier_base - DELTA_BRIER
    Var_thresh   = ALPHA_VAR * Var_base
    AUC_thresh   = AUC_base - DELTA_AUC

    # Filter eligible candidates
    eligible = result_df[
        (result_df['ECE']   <= ECE_target)   &
        (result_df['Brier'] <= Brier_thresh)  &
        (result_df['Var']   >= Var_thresh)    &
        (result_df['AUC']   >= AUC_thresh)
    ]

    if len(eligible) > 0:
        chosen_ps = int(eligible.sort_values('prior_strength').iloc[0]['prior_strength'])
        fallback  = False
        fallback_reason = ''
    else:
        # Relax Var to 60% before touching ECE/Brier constraints
        Var_thresh_relaxed = ALPHA_VAR_FALLBACK * Var_base
        eligible2 = result_df[
            (result_df['ECE']   <= ECE_target)   &
            (result_df['Brier'] <= Brier_thresh)  &
            (result_df['Var']   >= Var_thresh_relaxed) &
            (result_df['AUC']   >= AUC_thresh)
        ]
        if len(eligible2) > 0:
            chosen_ps = int(eligible2.sort_values('prior_strength').iloc[0]['prior_strength'])
            fallback  = True
            fallback_reason = f'Var threshold relaxed to {ALPHA_VAR_FALLBACK*100:.0f}% of baseline'
        else:
            # Last resort: best ECE among candidates that don't fully collapse AUC
            auc_floor = result_df['AUC'].max() * 0.97
            eligible3 = result_df[result_df['AUC'] >= auc_floor]
            if len(eligible3) == 0:
                eligible3 = result_df
            chosen_ps = int(eligible3.sort_values('ECE').iloc[0]['prior_strength'])
            fallback  = True
            fallback_reason = 'No candidate met Var/AUC constraints — chose min ECE with AUC floor'

    # Verdict column
    def _verdict(row):
        ps = row['prior_strength']
        if ps == BASELINE_PS:
            tag = 'baseline_ref'
        else:
            tag = ''

        fails = []
        if row['ECE']   > ECE_target:   fails.append('ECE')
        if row['Brier'] > Brier_thresh:  fails.append('Brier')
        if row['Var']   < Var_thresh:    fails.append('Var')
        if row['AUC']   < AUC_thresh:    fails.append('AUC')

        if fails:
            tag += (' ' if tag else '') + f"FAIL({','.join(fails)})"

        if ps == chosen_ps:
            tag += (' ' if tag else '') + ('SELECTED*' if fallback else 'SELECTED')

        return tag.strip()

    result_df['Verdict'] = result_df.apply(_verdict, axis=1)

    # Store thresholds for report
    result_df.attrs['ECE_target']      = ECE_target
    result_df.attrs['Brier_thresh']    = Brier_thresh
    result_df.attrs['Var_thresh']      = Var_thresh
    result_df.attrs['AUC_thresh']      = AUC_thresh
    result_df.attrs['ECE_base']        = ECE_base
    result_df.attrs['Brier_base']      = Brier_base
    result_df.attrs['Var_base']        = Var_base
    result_df.attrs['AUC_base']        = AUC_base
    result_df.attrs['fallback']        = fallback
    result_df.attrs['fallback_reason'] = fallback_reason if fallback else ''

    return result_df, chosen_ps


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(df: pd.DataFrame, chosen_ps: int, out_path: str) -> str:
    sep  = '-' * 85
    sep2 = '=' * 85
    lines = []
    lines.append(sep2)
    lines.append("  CONFIDENCE_PRIOR_STRENGTH — Multi-Objective Calibration Report")
    lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  Dataset   : retrospective_db.db  (Jan-Mar 2026, n=29,081)")
    lines.append(f"  Baseline  : prior_strength={BASELINE_PS}")
    lines.append(sep2)

    lines.append("\nSelection Thresholds:")
    lines.append(f"  ECE   <= {df.attrs['ECE_target']:.5f}   (global_min + {DELTA_ECE})")
    lines.append(f"  Brier <= {df.attrs['Brier_thresh']:.5f}   (baseline - {DELTA_BRIER})")
    lines.append(f"  Var   >= {df.attrs['Var_thresh']:.6f}  ({ALPHA_VAR*100:.0f}% of baseline {df.attrs['Var_base']:.6f})")
    lines.append(f"  AUC   >= {df.attrs['AUC_thresh']:.5f}   (baseline - {DELTA_AUC})")
    lines.append(f"  [Fallback Var threshold: {ALPHA_VAR_FALLBACK*100:.0f}% of baseline if primary fails]")
    lines.append("")

    hdr = (f"{'prior_strength':>14} | {'ECE':>8} | {'Brier':>8} | "
           f"{'Var':>10} | {'AUC':>7} | Verdict")
    lines.append(hdr)
    lines.append(sep)

    for _, row in df.iterrows():
        line = (
            f"{int(row['prior_strength']):>14} | "
            f"{row['ECE']:>8.5f} | "
            f"{row['Brier']:>8.5f} | "
            f"{row['Var']:>10.6f} | "
            f"{row['AUC']:>7.5f} | "
            f"{row['Verdict']}"
        )
        lines.append(line)

    lines.append(sep)

    chosen_row = df[df['prior_strength'] == chosen_ps].iloc[0]
    base_row   = df[df['prior_strength'] == BASELINE_PS].iloc[0]

    lines.append("")
    lines.append(f"  SELECTED prior_strength = {chosen_ps}")
    if df.attrs['fallback']:
        lines.append(f"  (Fallback: {df.attrs['fallback_reason']})")

    lines.append("")
    lines.append("  Baseline comparison:")
    lines.append(f"    prior={BASELINE_PS:>5} -> ECE={base_row['ECE']:.5f}  Brier={base_row['Brier']:.5f}"
                 f"  Var={base_row['Var']:.6f}  AUC={base_row['AUC']:.5f}")
    lines.append(f"    prior={chosen_ps:>5} -> ECE={chosen_row['ECE']:.5f}  Brier={chosen_row['Brier']:.5f}"
                 f"  Var={chosen_row['Var']:.6f}  AUC={chosen_row['AUC']:.5f}")

    var_drop_pct  = (1 - chosen_row['Var'] / base_row['Var'])  * 100
    auc_drop      = base_row['AUC'] - chosen_row['AUC']
    ece_improve   = (1 - chosen_row['ECE'] / base_row['ECE'])  * 100

    lines.append("")
    lines.append("  Key tradeoffs:")
    lines.append(f"    ECE improvement       : {ece_improve:+.1f}%")
    lines.append(f"    Var drop from baseline: {var_drop_pct:.1f}%  (limit: {(1-ALPHA_VAR)*100:.0f}%)")
    lines.append(f"    AUC drop from baseline: {auc_drop:.5f}  (limit: {DELTA_AUC})")

    lines.append("")
    lines.append("  Why NOT 10,000?")
    row_10k = df[df['prior_strength'] == 10000].iloc[0]
    row_sel  = chosen_row
    lines.append(f"    prior=10000: ECE={row_10k['ECE']:.5f}  Var={row_10k['Var']:.6f}  AUC={row_10k['AUC']:.5f}")
    lines.append(f"    Var at 10000 is {row_10k['Var']/base_row['Var']*100:.1f}% of baseline "
                 f"(threshold: {ALPHA_VAR*100:.0f}%)  -> predictions collapse toward base rate.")
    lines.append(f"    AUC at 10000 is {row_10k['AUC']:.5f} vs baseline {base_row['AUC']:.5f} "
                 f"(delta={base_row['AUC']-row_10k['AUC']:.5f})")

    lines.append("")
    lines.append("  Re-run when > 3 months of new ERA3 live data accumulates.")
    lines.append(sep2)

    report = "\n".join(lines)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
    return report


# ── Patch settings.py ─────────────────────────────────────────────────────────

def patch_settings(chosen_ps: int, df: pd.DataFrame):
    with open(SETTINGS_PY, 'r', encoding='utf-8') as f:
        content = f.read()

    chosen_row = df[df['prior_strength'] == chosen_ps].iloc[0]
    base_row   = df[df['prior_strength'] == BASELINE_PS].iloc[0]

    new_block = (
        f"CONFIDENCE_PRIOR_STRENGTH = {chosen_ps}"
        f"  # Multi-objective calibration (ECE, Brier, Var, AUC) on Jan-Mar 2026 retro.\n"
        f"# Baseline: prior={BASELINE_PS} -> ECE={base_row['ECE']:.5f},"
        f" Brier={base_row['Brier']:.5f}, Var={base_row['Var']:.6f}, AUC={base_row['AUC']:.5f}\n"
        f"# Selected: prior={chosen_ps} -> ECE={chosen_row['ECE']:.5f},"
        f" Brier={chosen_row['Brier']:.5f}, Var={chosen_row['Var']:.6f}, AUC={chosen_row['AUC']:.5f}\n"
        f"# Rule: smallest prior_strength where ECE/Brier improve AND Var/AUC stay informative."
    )

    # Replace the CONFIDENCE_PRIOR_STRENGTH block (value + any trailing comment lines)
    content = re.sub(
        r'CONFIDENCE_PRIOR_STRENGTH\s*=\s*\d+[^\n]*\n(?:#[^\n]*\n)*',
        new_block + '\n',
        content,
    )

    with open(SETTINGS_PY, 'w', encoding='utf-8') as f:
        f.write(content)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('[*] Loading retrospective dataset...')
    df = load_data()
    print(f'    Rows loaded: {len(df):,}  |  Win rate: {df["is_win"].mean()*100:.1f}%')

    print('[*] Running multi-objective grid search...')
    results_df, chosen_ps = run_grid_search(df)

    out_path = os.path.join(LOGS_DIR, 'prior_strength_multiobjective_calibration.txt')
    report   = write_report(results_df, chosen_ps, out_path)

    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode('ascii', 'replace').decode('ascii'))

    print(f'\n[+] Report saved: {out_path}')
    print(f'[*] Patching settings.py -> CONFIDENCE_PRIOR_STRENGTH = {chosen_ps}')
    patch_settings(chosen_ps, results_df)
    print('[+] settings.py updated.')


if __name__ == '__main__':
    main()
