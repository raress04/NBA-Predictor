"""
analysis/compare_retro_live.py
--------------------------------
Task P.4 — Retro/Live Alignment Validation

Compares:
  - RETRO source : retrospective_db.projection_outcomes  (filtered to "valuable" picks)
  - LIVE source  : bet_tracker.daily_picks               (source_mode='live' or 'seed')

Window: February 2026 (2026-02-01 → 2026-02-28) — both DBs have full coverage.

Match key: (player, category, direction)  with ±0.5 line tolerance.
A pick is "valuable" in retro if:
  - projected in the dominant direction (OVER: proj > line, UNDER: proj < line)
  - |proj - line| >= LIVE_MIN_GAP threshold
  - category not in BANNED_COMBINATIONS

Usage:
    python3 -m analysis.compare_retro_live

Writes:
    logs/retro_live_alignment_YYYY-MM-DD.txt
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, date

MODEL_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RETRO_DB    = os.path.join(MODEL_DIR, 'database', 'retrospective_db.db')
LIVE_DB     = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')
LOGS_DIR    = os.path.join(MODEL_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

START_DATE = '2026-02-01'
END_DATE   = '2026-02-28'

# Must match config/settings.py
BANNED_CATS  = {('SPREAD', 'COVER')}
LIVE_MIN_GAP = {'POINTS': 2.0, 'REBOUNDS': 1.8, 'ASSISTS': 1.5, 'TOTAL': 5.0}
LINE_TOL     = 0.5   # ±0.5 tolerance for line matching


# ── Data Loaders ──────────────────────────────────────────────────────────────

def load_retro(start: str, end: str) -> pd.DataFrame:
    """
    Load retrospective picks for window. Apply same filters as live pipeline:
      - direction derived from projection vs line
      - passes min_gap
      - not banned
    """
    conn = sqlite3.connect(RETRO_DB)
    df = pd.read_sql(f"""
        SELECT game_date AS date,
               player,
               category,
               projected_value,
               bookmaker_line  AS line,
               actual_value
        FROM projection_outcomes
        WHERE game_date >= '{start}'
          AND game_date <= '{end}'
          AND bookmaker_line IS NOT NULL
    """, conn)
    conn.close()

    # Derive direction
    df['direction'] = np.where(df['projected_value'] >= df['line'], 'OVER', 'UNDER')
    df['gap']       = np.where(
        df['direction'] == 'OVER',
        df['projected_value'] - df['line'],
        df['line'] - df['projected_value'],
    )
    df['cat_up'] = df['category'].str.upper()
    df['dir_up'] = df['direction'].str.upper()

    # Apply min_gap filter
    df['min_gap'] = df['cat_up'].map(LIVE_MIN_GAP).fillna(1.5)
    df = df[df['gap'] >= df['min_gap']].copy()

    # Apply ban filter
    df = df[~df.apply(lambda r: (r['cat_up'], r['dir_up']) in BANNED_CATS, axis=1)].copy()

    df['key'] = df['player'].str.lower().str.strip() + '|' + df['cat_up'] + '|' + df['dir_up']
    return df.reset_index(drop=True)


def load_live(start: str, end: str) -> pd.DataFrame:
    """Load live picks from bet_tracker."""
    conn = sqlite3.connect(LIVE_DB)
    df = pd.read_sql(f"""
        SELECT date,
               player_or_team AS player,
               stat_category  AS category,
               direction,
               line,
               sim_confidence AS confidence,
               edge_pct
        FROM daily_picks
        WHERE date >= '{start}'
          AND date <= '{end}'
    """, conn)
    conn.close()

    df['cat_up'] = df['category'].str.upper()
    df['dir_up'] = df['direction'].str.upper()
    df['key']    = df['player'].str.lower().str.strip() + '|' + df['cat_up'] + '|' + df['dir_up']
    return df.reset_index(drop=True)


# ── Per-Day Comparison ────────────────────────────────────────────────────────

def compare_day(date_str: str, retro_day: pd.DataFrame, live_day: pd.DataFrame) -> dict:
    """
    For a single date:
      - retro → live overlap: retro picks that are also in live (by key + line tolerance)
      - live → retro overlap: live picks that are also in retro
    """
    def _find_match(row, target_df):
        """True if target_df has a matching (key, line ±tol) row."""
        candidates = target_df[target_df['key'] == row['key']]
        if candidates.empty:
            return False
        if 'line' in candidates.columns and 'line' in row.index:
            return any(abs(candidates['line'] - row['line']) <= LINE_TOL)
        return not candidates.empty

    if retro_day.empty or live_day.empty:
        return {
            'date': date_str,
            'retro_n': len(retro_day), 'live_n': len(live_day),
            'retro_in_live': 0, 'live_in_retro': 0,
            'retro_overlap_pct': 0.0 if not retro_day.empty else None,
            'live_overlap_pct':  0.0 if not live_day.empty  else None,
            'retro_only': retro_day['key'].tolist() if not retro_day.empty else [],
            'live_only':  live_day['key'].tolist()  if not live_day.empty  else [],
        }

    retro_matched = retro_day.apply(lambda r: _find_match(r, live_day), axis=1)
    live_matched  = live_day.apply(lambda r: _find_match(r, retro_day), axis=1)

    retro_in_live = int(retro_matched.sum())
    live_in_retro = int(live_matched.sum())

    return {
        'date':               date_str,
        'retro_n':            len(retro_day),
        'live_n':             len(live_day),
        'retro_in_live':      retro_in_live,
        'live_in_retro':      live_in_retro,
        'retro_overlap_pct':  round(retro_in_live / len(retro_day) * 100, 1) if len(retro_day) > 0 else None,
        'live_overlap_pct':   round(live_in_retro / len(live_day) * 100, 1)  if len(live_day) > 0  else None,
        'retro_only':         retro_day[~retro_matched]['key'].tolist(),
        'live_only':          live_day[~live_matched]['key'].tolist(),
    }


# ── Divergence Analysis ───────────────────────────────────────────────────────

def analyse_divergences(retro: pd.DataFrame, live: pd.DataFrame) -> list[str]:
    """
    Return bullet-point strings identifying systematic gaps between retro and live.
    """
    bullets = []

    # Categories present in retro but not at all in live
    retro_cats = set(retro['cat_up'].unique())
    live_cats  = set(live['cat_up'].unique())
    missing_in_live = retro_cats - live_cats
    if missing_in_live:
        bullets.append(f"• Categories in retro but absent from live: {', '.join(sorted(missing_in_live))}")

    extra_in_live = live_cats - retro_cats
    if extra_in_live:
        bullets.append(f"• Categories in live but absent from retro: {', '.join(sorted(extra_in_live))}")

    # Direction bias per category
    for cat in sorted(retro_cats & live_cats):
        r_dir = retro[retro['cat_up'] == cat]['dir_up'].value_counts(normalize=True)
        l_dir = live[live['cat_up'] == cat]['dir_up'].value_counts(normalize=True)
        for d in ['OVER', 'UNDER']:
            r_pct = r_dir.get(d, 0) * 100
            l_pct = l_dir.get(d, 0) * 100
            if abs(r_pct - l_pct) > 20:
                bullets.append(
                    f"• {cat} {d}: retro={r_pct:.0f}% vs live={l_pct:.0f}% — "
                    f"{'live skews more' if l_pct > r_pct else 'retro skews more'}"
                )

    # Volume mismatch
    ratio = len(live) / max(len(retro), 1)
    if ratio > 3.0:
        bullets.append(f"• Live volume {ratio:.1f}x higher than retro — live may include low-confidence picks absent from retro filter")
    elif ratio < 0.3:
        bullets.append(f"• Live volume is only {ratio:.1%} of retro — live pipeline is more selective")

    if not bullets:
        bullets.append("• No systematic divergences detected.")

    return bullets


# ── Report Writer ─────────────────────────────────────────────────────────────

def write_report(daily_results: list[dict], retro: pd.DataFrame, live: pd.DataFrame,
                 out_path: str) -> str:
    sep  = '=' * 68
    sep2 = '-' * 68
    lines = []
    lines.append(sep)
    lines.append("  RETRO / LIVE ALIGNMENT VALIDATION REPORT")
    lines.append(f"  Window    : {START_DATE} → {END_DATE}")
    lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  Retro DB  : retrospective_db.db  | Live DB: bet_tracker.db")
    lines.append(f"  Match key : (player, category, direction) + line ±{LINE_TOL}")
    lines.append(sep)

    lines.append("\n── PER-DAY BREAKDOWN ────────────────────────────────────────────")
    hdr = f"{'Date':<12} {'Retro':>6} {'Live':>6} {'R→L%':>7} {'L→R%':>7}"
    lines.append(hdr)
    lines.append(sep2)

    agg_retro_n = agg_live_n = agg_r_in_l = agg_l_in_r = 0
    for d in daily_results:
        r_pct = f"{d['retro_overlap_pct']:.1f}%" if d['retro_overlap_pct'] is not None else "  N/A"
        l_pct = f"{d['live_overlap_pct']:.1f}%"  if d['live_overlap_pct'] is not None  else "  N/A"
        lines.append(f"{d['date']:<12} {d['retro_n']:>6} {d['live_n']:>6} {r_pct:>7} {l_pct:>7}")
        agg_retro_n += d['retro_n']
        agg_live_n  += d['live_n']
        agg_r_in_l  += d['retro_in_live']
        agg_l_in_r  += d['live_in_retro']

    lines.append(sep2)
    overall_rl = round(agg_r_in_l / agg_retro_n * 100, 1) if agg_retro_n else 0
    overall_lr = round(agg_l_in_r / agg_live_n * 100, 1)  if agg_live_n  else 0
    lines.append(f"{'AGGREGATE':<12} {agg_retro_n:>6} {agg_live_n:>6} {overall_rl:>6.1f}% {overall_lr:>6.1f}%")

    lines.append("\n── AGGREGATE VERDICT ────────────────────────────────────────────")
    target_met = overall_rl >= 80.0
    lines.append(f"  Retro → Live overlap : {overall_rl:.1f}%  "
                 f"(target ≥80% — {'✓ MET' if target_met else '✗ NOT MET'})")
    lines.append(f"  Live → Retro overlap : {overall_lr:.1f}%")
    lines.append(f"  Total retro picks    : {agg_retro_n}")
    lines.append(f"  Total live picks     : {agg_live_n}")

    lines.append("\n── DIVERGENCE ANALYSIS ──────────────────────────────────────────")
    for b in analyse_divergences(retro, live):
        lines.append(f"  {b}")

    lines.append("\n── RETRO-ONLY PICKS (sample, max 20) ───────────────────────────")
    retro_only_all = [k for d in daily_results for k in d['retro_only']]
    from collections import Counter
    top_retro_only = Counter(retro_only_all).most_common(20)
    for key, cnt in top_retro_only:
        lines.append(f"  {key}  (missed on {cnt} day(s))")

    lines.append("\n── LIVE-ONLY PICKS (sample, max 20) ────────────────────────────")
    live_only_all = [k for d in daily_results for k in d['live_only']]
    top_live_only = Counter(live_only_all).most_common(20)
    for key, cnt in top_live_only:
        lines.append(f"  {key}  (extra on {cnt} day(s))")

    lines.append("\n" + sep)
    report = "\n".join(lines)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default=START_DATE)
    ap.add_argument('--end',   default=END_DATE)
    ap.add_argument('--out',   default=None, help='Override output file path')
    args = ap.parse_args()

    start_d  = args.start
    end_d    = args.end
    date_str = datetime.now().strftime('%Y-%m-%d')
    out_path = args.out if args.out else os.path.join(LOGS_DIR, f'retro_live_alignment_{date_str}.txt')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f'[*] Loading retro picks for {start_d} to {end_d}...')
    retro = load_retro(start_d, end_d)
    print(f'    {len(retro):,} valuable retro picks after min_gap + ban filters')

    print('[*] Loading live picks...')
    live  = load_live(start_d, end_d)
    print(f'    {len(live):,} live picks')

    all_dates = sorted(set(retro['date'].unique()) | set(live['date'].unique()))
    print(f'[*] Comparing {len(all_dates)} dates...')

    daily_results = []
    for d in all_dates:
        r_day = retro[retro['date'] == d]
        l_day = live[live['date'] == d]
        daily_results.append(compare_day(d, r_day, l_day))

    report = write_report(daily_results, retro, live, out_path)

if __name__ == '__main__':
    main()

