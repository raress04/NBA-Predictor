"""
ml.check_promotion.py
----------------------------
Weekly promotion gate: evaluates whether the model meets all criteria
for Kelly criterion re-enablement and category unbanning.

This script is READ-ONLY — it reports readiness, makes no config changes.

Usage:
    python3 -m ml.check_promotion
    (or via scripts/weekly_checks.sh)
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKER_DB = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')
LOGS_DIR   = os.path.join(MODEL_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# ── Filters ───────────────────────────────────────────────────────────────────
LIVE_FILTER = """
    dp.source_mode = 'live'
    AND dp.synthetic_flag IS NULL
    AND dp.model_version = 'ERA3'
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(TRACKER_DB)
    conn.row_factory = sqlite3.Row
    return conn


# ── Check 1: Leg Volume ───────────────────────────────────────────────────────

def check_leg_volume(conn) -> dict:
    """Require at least 500 verified resolved legs."""
    row = conn.execute(f"""
        SELECT COUNT(*) as n
        FROM daily_picks dp
        JOIN pick_results pr ON dp.id = pr.pick_id
        WHERE {LIVE_FILTER}
    """).fetchone()
    n = row['n'] if row else 0
    return {
        'check': 'Leg Volume',
        'value': n,
        'threshold': 500,
        'pass': n >= 500,
        'detail': f"Verified legs: {n} (need >= 500)",
    }


# ── Check 2: Per-Category Win Rate ────────────────────────────────────────────

def check_category_win_rates(conn) -> dict:
    """Each (stat_category, direction) with n>=50 and edge_pct>0 must have WR >= 52.5%."""
    rows = conn.execute(f"""
        SELECT dp.stat_category, dp.direction,
               COUNT(*)           AS n,
               ROUND(AVG(pr.is_win) * 100, 2) AS win_rate
        FROM daily_picks dp
        JOIN pick_results pr ON dp.id = pr.pick_id
        WHERE {LIVE_FILTER}
          AND dp.edge_pct > 0
        GROUP BY dp.stat_category, dp.direction
        HAVING COUNT(*) >= 50
    """).fetchall()

    failures = []
    details = []
    for r in rows:
        wr = r['win_rate'] or 0.0
        label = 'PASS' if wr >= 52.5 else 'FAIL'
        details.append(f"  {r['stat_category']} {r['direction']}: {wr:.1f}% (n={r['n']}) [{label}]")
        if wr < 52.5:
            failures.append(f"{r['stat_category']} {r['direction']} @ {wr:.1f}%")

    passing = len(failures) == 0
    if not rows:
        passing = False
        details.append("  No categories with n>=50 and edge_pct>0 found.")

    return {
        'check': 'Per-Category Win Rate',
        'pass': passing,
        'failures': failures,
        'detail': '\n'.join(details) if details else '  No data.',
        'threshold': '52.5% per eligible category',
    }


# ── Check 3: Calibration ──────────────────────────────────────────────────────

def check_calibration(conn) -> dict:
    """Each confidence bin must have |predicted - actual| <= 0.08."""
    rows = conn.execute(f"""
        SELECT dp.sim_confidence / 100.0 AS pred, pr.is_win
        FROM daily_picks dp
        JOIN pick_results pr ON dp.id = pr.pick_id
        WHERE {LIVE_FILTER}
          AND dp.sim_confidence IS NOT NULL
    """).fetchall()

    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return {
            'check': 'Calibration',
            'pass': False,
            'detail': '  No data for calibration.',
            'threshold': '|predicted - actual| <= 0.08 per bin',
        }

    bins   = [0.55, 0.65, 0.75, 0.78, 1.0]
    labels = ['Bin 55-65%', 'Bin 65-75%', 'Bin 75-78%', 'Bin 78%+']
    df['bin'] = pd.cut(df['pred'], bins=bins, labels=labels, right=False)

    result = df.groupby('bin', observed=False)['is_win'].agg(['mean', 'count']).reset_index()

    failures = []
    details  = []
    for _, row in result.iterrows():
        n = int(row['count'])
        if n < 10:
            details.append(f"  {row['bin']}: insufficient data (n={n}) [SKIP]")
            continue
        bin_mid_map = {'Bin 55-65%': 0.60, 'Bin 65-75%': 0.70, 'Bin 75-78%': 0.765, 'Bin 78%+': 0.82}
        predicted  = bin_mid_map.get(str(row['bin']), 0.70)
        actual     = float(row['mean'])
        diff       = abs(predicted - actual)
        label      = 'PASS' if diff <= 0.08 else 'FAIL'
        details.append(
            f"  {row['bin']}: predicted={predicted*100:.1f}% | actual={actual*100:.1f}% | "
            f"diff={diff*100:.1f}pp [n={n}] [{label}]"
        )
        if diff > 0.08:
            failures.append(str(row['bin']))

    passing = len(failures) == 0 and len(result) > 0

    return {
        'check': 'Calibration',
        'pass': passing,
        'failures': failures,
        'detail': '\n'.join(details),
        'threshold': '|predicted - actual| <= 8pp per bin',
    }


# ── Check 4: CLV ──────────────────────────────────────────────────────────────

def check_clv(conn) -> dict:
    """Average CLV over last 30 days must be >= 0.0."""
    row = conn.execute(f"""
        SELECT ROUND(AVG(clv), 3) AS avg_clv, COUNT(*) AS n
        FROM daily_picks dp
        WHERE {LIVE_FILTER}
          AND clv IS NOT NULL
          AND dp.date >= date('now', '-30 days')
    """).fetchone()
    avg_clv = row['avg_clv'] if row and row['avg_clv'] is not None else None
    n       = row['n'] if row else 0

    passing = avg_clv is not None and avg_clv >= 0.0
    if avg_clv is None:
        detail = "  No CLV data found (need closing-line data)."
    else:
        detail = f"  Average CLV (last 30d): {avg_clv:+.3f} (n={n})"

    return {
        'check': 'CLV Average',
        'value': avg_clv,
        'threshold': '>= 0.0',
        'pass': passing,
        'detail': detail,
    }


# ── Check 5: Parlay Performance ───────────────────────────────────────────────

def check_parlay_performance(conn) -> dict:
    """At least 30 resolved parlays in last 30d; hit rate >= 20%."""
    row = conn.execute("""
        SELECT COUNT(*) AS n,
               SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) AS wins
        FROM parlays
        WHERE result IS NOT NULL
          AND game_date >= date('now', '-30 days')
    """).fetchone()
    n    = row['n']    if row else 0
    wins = row['wins'] if row and row['wins'] else 0
    hit  = (wins / n * 100) if n > 0 else 0.0

    volume_pass = n >= 30
    rate_pass   = hit >= 20.0
    passing     = volume_pass and rate_pass

    detail = f"  Parlays (last 30d): {wins}/{n} resolved | Hit rate: {hit:.1f}%"
    if not volume_pass:
        detail += f" [FAIL — need >= 30, have {n}]"
    elif not rate_pass:
        detail += f" [FAIL — need >= 20%, have {hit:.1f}%]"
    else:
        detail += " [PASS]"

    return {
        'check': 'Parlay Performance',
        'n': n,
        'hit_rate': hit,
        'pass': passing,
        'detail': detail,
        'threshold': 'n>=30 and hit_rate>=20%',
    }


# ── Report Builder ────────────────────────────────────────────────────────────

def run_promotion_check(date_str: str = None) -> str:
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')

    conn   = _get_conn()
    sep    = '=' * 60
    lines  = []

    lines.append(sep)
    lines.append(f"  PROMOTION GATE REPORT  {date_str}")
    lines.append(f"  Filters: source_mode=live | synthetic_flag IS NULL | model_version=ERA3")
    lines.append(sep)

    checks = [
        check_leg_volume(conn),
        check_category_win_rates(conn),
        check_calibration(conn),
        check_clv(conn),
        check_parlay_performance(conn),
    ]

    all_pass = True
    for c in checks:
        status = 'PASS' if c['pass'] else 'FAIL'
        if not c['pass']:
            all_pass = False
        lines.append(f"\n[{status}] {c['check']}  (threshold: {c.get('threshold', 'see detail')})")
        lines.append(c['detail'])

    lines.append('\n' + sep)
    if all_pass:
        verdict = 'PROMOTION APPROVED -- KELLY CRITERION MAY BE RE-ENABLED'
    else:
        verdict = 'NOT READY -- KELLY REMAINS DISABLED (see FAILs above)'
    lines.append(f"  OVERALL VERDICT: {verdict}")
    lines.append(sep)

    conn.close()
    return '\n'.join(lines)


def main():
    date_str = datetime.now().strftime('%Y-%m-%d')
    report   = run_promotion_check(date_str)

    log_path = os.path.join(LOGS_DIR, f'promotion_check_{date_str}.log')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(report)

    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode('ascii', 'replace').decode('ascii'))

    print(f"\n[+] Log saved: {log_path}")


if __name__ == '__main__':
    main()
