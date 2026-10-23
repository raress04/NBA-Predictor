"""
ml/daily_metrics.py
-------------------
Daily rolling metrics snapshot for the NBA Predictor model.

Computes hit rate, ROI, MAE, Brier Score, and CLV across three rolling
windows (7 / 30 / 90 days) per stat category and direction, using ONLY
real live picks (excludes synthetic, bulk_backtest, and preseason contamination).

Output:
  - JSON snapshot saved to logs/daily_metrics_YYYY-MM-DD.json
  - Console summary table

Usage:
    python -m ml.daily_metrics
    python -m ml.daily_metrics --date 2026-10-21
    python -m ml.daily_metrics --no-save
"""

import sys
import os
import json
import sqlite3
import argparse
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKER_DB = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')
LOGS_DIR   = os.path.join(MODEL_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOWS       = [7, 30, 90]
CATEGORIES    = ['POINTS', 'REBOUNDS', 'ASSISTS']
DIRECTIONS    = ['OVER', 'UNDER']
# Picks must pass ALL of these to be included in metrics
CLEAN_FILTER  = """
    dp.source_file NOT LIKE 'bulk_backtest%'
    AND dp.source_file NOT LIKE 'synthetic%'
    AND dp.source_mode IS NOT 'synthetic_circular'
"""
# Minimum resolved picks required to report a metric (avoid noise on tiny samples)
MIN_PICKS     = 5
# Assumed odds for ROI calculation when actual odds not stored (-110 American = 1.909 decimal)
DEFAULT_DECIMAL_ODDS = 1.909


# ── Core Query ────────────────────────────────────────────────────────────────

def _fetch_window(conn: sqlite3.Connection, category: str, direction: str,
                  days: int, as_of: str) -> list[dict]:
    """
    Fetch all resolved picks for a (category, direction, window) combination.
    Returns list of dicts with: sim_confidence, is_win, diff_val, clv, odds.
    """
    since = (datetime.strptime(as_of, '%Y-%m-%d') - timedelta(days=days)).strftime('%Y-%m-%d')
    query = f"""
        SELECT
            dp.sim_confidence,
            pr.is_win,
            pr.diff_val,
            dp.clv,
            COALESCE(dp.odds, {DEFAULT_DECIMAL_ODDS}) AS odds
        FROM daily_picks dp
        JOIN pick_results pr ON dp.id = pr.pick_id
        WHERE dp.stat_category = ?
          AND dp.direction      = ?
          AND dp.date           > ?
          AND dp.date          <= ?
          AND {CLEAN_FILTER}
    """
    rows = conn.execute(query, (category, direction, since, as_of)).fetchall()
    return [dict(zip(['sim_confidence', 'is_win', 'diff_val', 'clv', 'odds'], r))
            for r in rows]


# ── Metric Calculations ───────────────────────────────────────────────────────

def _hit_rate(picks: list[dict]) -> float | None:
    """Win rate as a percentage."""
    if len(picks) < MIN_PICKS:
        return None
    return round(sum(p['is_win'] for p in picks) / len(picks) * 100, 2)


def _roi(picks: list[dict]) -> float | None:
    """
    Flat-stake ROI assuming each pick stakes 1 unit at the stored decimal odds.
    ROI% = (profit / total_staked) * 100
    profit = sum(is_win * (odds - 1)) - sum(1 - is_win)
    """
    if len(picks) < MIN_PICKS:
        return None
    profit = sum(
        p['is_win'] * (p['odds'] - 1.0) - (1 - p['is_win'])
        for p in picks
    )
    total_staked = len(picks)
    return round(profit / total_staked * 100, 3)


def _brier(picks: list[dict]) -> float | None:
    """
    Brier Score = mean((pred_prob - actual_outcome)^2).
    Lower is better. Coin flip = 0.25. Target < 0.245.
    Only computed when sim_confidence is available.
    """
    valid = [p for p in picks if p['sim_confidence'] is not None]
    if len(valid) < MIN_PICKS:
        return None
    score = sum(
        ((p['sim_confidence'] / 100.0) - p['is_win']) ** 2
        for p in valid
    ) / len(valid)
    return round(score, 4)


def _mae(picks: list[dict]) -> float | None:
    """Mean Absolute Error of projection vs actual (from diff_val column)."""
    valid = [p for p in picks if p['diff_val'] is not None]
    if len(valid) < MIN_PICKS:
        return None
    return round(sum(abs(p['diff_val']) for p in valid) / len(valid), 4)


def _bias(picks: list[dict]) -> float | None:
    """Mean signed error (projection - actual). Positive = overestimating."""
    valid = [p for p in picks if p['diff_val'] is not None]
    if len(valid) < MIN_PICKS:
        return None
    return round(sum(p['diff_val'] for p in valid) / len(valid), 4)


def _clv(picks: list[dict]) -> dict | None:
    """
    Line-movement proxy (stored as 'clv' column).
    For SPREAD/TOTAL: measures how much the line moved in your favour between open and close.
    For player props: column is mostly NULL — prop markets are too thin for meaningful CLV.
    Returns avg and positive-rate for informational purposes only — NOT a gate metric.
    """
    valid = [p for p in picks if p['clv'] is not None]
    if len(valid) < MIN_PICKS:
        return None
    avg_clv  = round(sum(p['clv'] for p in valid) / len(valid), 4)
    pos_rate = round(sum(1 for p in valid if p['clv'] > 0) / len(valid) * 100, 1)
    return {'avg': avg_clv, 'positive_rate_pct': pos_rate, 'n': len(valid)}


# ── Rolling Window Snapshot ───────────────────────────────────────────────────

def compute_daily_metrics(as_of: str = None) -> dict:
    """
    Compute the full rolling metrics snapshot for all categories / directions
    across windows of 7, 30, and 90 days.

    Parameters
    ----------
    as_of : str
        Date string YYYY-MM-DD. Defaults to today.

    Returns
    -------
    dict with structure:
        {
          'as_of': '2026-10-22',
          'generated_at': '2026-10-22T13:00:00',
          'windows': {
            7:  { 'POINTS_Over': {...}, 'POINTS_Under': {...}, ... },
            30: { ... },
            90: { ... },
          },
          'totals': { 7: {...}, 30: {...}, 90: {...} },
          'platt_ready': [ 'POINTS_Over', ... ],   # categories >= 200 picks in 90d
          'alerts': [ ... ]
        }
    """
    if as_of is None:
        as_of = datetime.now().strftime('%Y-%m-%d')

    conn = sqlite3.connect(TRACKER_DB)
    conn.row_factory = sqlite3.Row

    result = {
        'as_of': as_of,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'windows': {w: {} for w in WINDOWS},
        'totals':  {w: {} for w in WINDOWS},
        'platt_ready': [],
        'alerts': [],
    }

    for days in WINDOWS:
        since = (datetime.strptime(as_of, '%Y-%m-%d') - timedelta(days=days)).strftime('%Y-%m-%d')

        for cat in CATEGORIES:
            for direction in DIRECTIONS:
                key   = f'{cat}_{direction}'
                picks = _fetch_window(conn, cat, direction, days, as_of)
                n     = len(picks)

                entry = {
                    'n':         n,
                    'hit_rate':  _hit_rate(picks),
                    'roi_pct':   _roi(picks),
                    'brier':     _brier(picks),
                    'mae':       _mae(picks),
                    'bias':      _bias(picks),
                    'clv':       _clv(picks),
                }
                result['windows'][days][key] = entry

                # Platt readiness check (90-day window, 200+ picks)
                if days == 90 and n >= 200:
                    result['platt_ready'].append(key)

        # Aggregate totals across all categories for this window
        all_picks_window = conn.execute(f"""
            SELECT pr.is_win,
                   pr.diff_val,
                   dp.clv,
                   COALESCE(dp.odds, {DEFAULT_DECIMAL_ODDS}) AS odds
            FROM daily_picks dp
            JOIN pick_results pr ON dp.id = pr.pick_id
            WHERE dp.stat_category IN ('POINTS','REBOUNDS','ASSISTS')
              AND dp.date > ?
              AND dp.date <= ?
              AND {CLEAN_FILTER}
        """, (since, as_of)).fetchall()
        all_picks = [dict(zip(['is_win', 'diff_val', 'clv', 'odds'], r))
                     for r in all_picks_window]

        n_total = len(all_picks)
        result['totals'][days] = {
            'n':        n_total,
            'hit_rate': _hit_rate(all_picks) if n_total >= MIN_PICKS else None,
            'roi_pct':  _roi(all_picks)      if n_total >= MIN_PICKS else None,
            'mae':      _mae(all_picks)      if n_total >= MIN_PICKS else None,
            'clv':      _clv(all_picks),
        }

    # ── Alerts ────────────────────────────────────────────────────────────────
    for cat in CATEGORIES:
        for direction in DIRECTIONS:
            key   = f'{cat}_{direction}'
            w30   = result['windows'][30].get(key, {})
            hr    = w30.get('hit_rate')
            brier = w30.get('brier')
            n     = w30.get('n', 0)

            if hr is not None and hr < 40.0 and n >= MIN_PICKS:
                result['alerts'].append(
                    f'CRITICAL: {key} hit rate {hr}% < 40% threshold (n={n}, last 30d)'
                )
            if brier is not None and brier > 0.25 and n >= MIN_PICKS:
                result['alerts'].append(
                    f'WARNING: {key} Brier {brier} > 0.25 — miscalibration (n={n}, last 30d)'
                )

    roi_90 = result['totals'][90].get('roi_pct')
    if roi_90 is not None and roi_90 < -2.0:
        result['alerts'].append(
            f'WARNING: Overall ROI {roi_90:+.2f}% in last 90 days — below -2% threshold'
        )

    conn.close()
    return result


# ── Pretty Print ──────────────────────────────────────────────────────────────

def _fmt(val, fmt='.2f', suffix='', na='n/a'):
    if val is None:
        return na
    return f'{val:{fmt}}{suffix}'


def format_report(metrics: dict) -> str:
    sep   = '=' * 68
    lines = []
    lines.append(sep)
    lines.append(f'  DAILY METRICS SNAPSHOT  |  as of: {metrics["as_of"]}')
    lines.append(f'  Generated: {metrics["generated_at"]}')
    lines.append(sep)

    # ── Per-category table ────────────────────────────────────────────────────
    for cat in CATEGORIES:
        lines.append(f'\n  {cat}')
        lines.append(f'  {"Direction":<9} {"Window":>6} {"N":>5} {"Hit%":>7} '
                     f'{"ROI%":>7} {"Brier":>7} {"MAE":>6} {"Bias":>7} {"LineMove":>9}')
        lines.append('  ' + '-' * 64)

        for direction in DIRECTIONS:
            key = f'{cat}_{direction}'
            for days in WINDOWS:
                e   = metrics['windows'][days].get(key, {})
                clv_avg = e.get('clv', {}).get('avg') if e.get('clv') else None
                lines.append(
                    f'  {direction:<9} {days:>5}d {_fmt(e.get("n"), ".0f"):>5} '
                    f'{_fmt(e.get("hit_rate")):>6}% '
                    f'{_fmt(e.get("roi_pct")):>6}% '
                    f'{_fmt(e.get("brier"), ".4f"):>7} '
                    f'{_fmt(e.get("mae"), ".3f"):>6} '
                    f'{_fmt(e.get("bias"), "+.3f"):>7} '
                    f'{_fmt(clv_avg, "+.3f"):>9}'
                )

    # ── Aggregate totals ──────────────────────────────────────────────────────
    lines.append(f'\n  {"AGGREGATE (all cats)":<20} {"Window":>6} {"N":>5} '
                 f'{"Hit%":>7} {"ROI%":>7} {"MAE":>6} {"LineMove":>9}')
    lines.append('  ' + '-' * 64)
    for days in WINDOWS:
        t       = metrics['totals'][days]
        clv_avg = t.get('clv', {}).get('avg') if t.get('clv') else None
        lines.append(
            f'  {"ALL":20} {days:>5}d {_fmt(t.get("n"), ".0f"):>5} '
            f'{_fmt(t.get("hit_rate")):>6}% '
            f'{_fmt(t.get("roi_pct")):>6}% '
            f'{_fmt(t.get("mae"), ".3f"):>6} '
            f'{_fmt(clv_avg, "+.3f"):>8}'
        )

    # ── Platt readiness ───────────────────────────────────────────────────────
    lines.append('\n  PLATT SCALING READINESS (>= 200 picks in 90 days):')
    if metrics['platt_ready']:
        lines.append('  READY: ' + ', '.join(metrics['platt_ready']))
        lines.append('  -> Run: python -m ml.train_calibration_model')
    else:
        lines.append('  PENDING — accumulating picks. Target: mid-November 2026.')

    # ── Line movement note ─────────────────────────────────────────────────────
    lines.append('\n  NOTE: "LineMove" = line-movement proxy (SPREAD/TOTAL only, informational).')
    lines.append('  Props: CLV excluded — thin markets, noisy closing lines. Not a gate metric.')

    # ── Alerts ────────────────────────────────────────────────────────────────
    lines.append('\n  ALERTS:')
    if metrics['alerts']:
        for a in metrics['alerts']:
            lines.append(f'  !! {a}')
    else:
        lines.append('  OK — no anomalies detected.')

    lines.append('\n' + sep)
    return '\n'.join(lines)


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='NBA Model — Daily Rolling Metrics')
    parser.add_argument('--date',    type=str, default=None,
                        help='Reference date YYYY-MM-DD (default: today)')
    parser.add_argument('--no-save', action='store_true',
                        help='Print only, do not save JSON snapshot')
    parser.add_argument('--json',    action='store_true',
                        help='Output raw JSON instead of formatted report')
    args = parser.parse_args()

    as_of   = args.date or datetime.now().strftime('%Y-%m-%d')
    metrics = compute_daily_metrics(as_of=as_of)
    report  = format_report(metrics)

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        try:
            print(report)
        except UnicodeEncodeError:
            print(report.encode('ascii', 'replace').decode('ascii'))

    if not args.no_save:
        out_path = os.path.join(LOGS_DIR, f'daily_metrics_{as_of}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f'\n[+] Metrics saved: {out_path}')


if __name__ == '__main__':
    main()
