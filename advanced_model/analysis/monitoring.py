"""
analysis/monitoring.py
----------------------
Daily monitoring report. Run after fetch_range_box_scores.py resolves outcomes.

Usage:
    python3 -m analysis.monitoring
    python3 advanced_model/analysis/monitoring.py
"""

import sqlite3
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKER_DB = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')
LOGS_DIR = os.path.join(MODEL_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)


# ── 7.1 Calibration Curves ────────────────────────────────────────────────────

def compute_calibration_curve(category: str, direction: str, conn) -> list | None:
    """Bins predictions by confidence and computes actual win rate per bin."""
    query = """
        SELECT dp.sim_confidence, pr.is_win
        FROM daily_picks dp
        JOIN pick_results pr ON dp.id = pr.pick_id
        WHERE dp.stat_category = ? AND dp.direction = ?
          AND dp.date >= date('now', '-90 days')
          AND dp.source_file NOT LIKE 'bulk_backtest%'
          AND dp.source_file NOT LIKE 'synthetic%'
    """
    df = pd.read_sql(query, conn, params=[category, direction])
    if len(df) < 30:
        return None

    bins = [50, 60, 65, 70, 75, 78, 100]
    df['bin'] = pd.cut(df['sim_confidence'], bins=bins)

    result = df.groupby('bin', observed=True)['is_win'].agg(['mean', 'count']).reset_index()
    result.columns = ['confidence_bin', 'actual_win_rate', 'n']

    result['miscalibrated'] = abs(
        result['confidence_bin'].apply(lambda x: x.mid / 100) -
        result['actual_win_rate']
    ) > 0.08

    return result.to_dict('records')


# ── 7.2 Brier Score ───────────────────────────────────────────────────────────

def compute_brier_score(category: str, direction: str, conn) -> float | None:
    """
    Brier Score = mean((predicted_prob - actual_outcome)^2)
    Range: 0.0 (perfect) to 1.0 (worst). Coin flip = 0.25. Target: < 0.20.
    """
    query = """
        SELECT dp.sim_confidence / 100.0 as pred_prob, pr.is_win
        FROM daily_picks dp
        JOIN pick_results pr ON dp.id = pr.pick_id
        WHERE dp.stat_category = ? AND dp.direction = ?
          AND dp.date >= date('now', '-90 days')
          AND dp.source_file NOT LIKE 'bulk_backtest%'
          AND dp.source_file NOT LIKE 'synthetic%'
    """
    df = pd.read_sql(query, conn, params=[category, direction])
    if len(df) < 30:
        return None

    brier = ((df['pred_prob'] - df['is_win']) ** 2).mean()
    return round(float(brier), 4)


# ── 7.3 CLV Stats ─────────────────────────────────────────────────────────────

def compute_clv_stats(conn) -> list:
    query = """
        SELECT stat_category, direction,
               ROUND(AVG(clv), 3) as avg_clv,
               SUM(CASE WHEN clv > 0 THEN 1 ELSE 0 END) as positive_clv_n,
               COUNT(*) as n
        FROM daily_picks
        WHERE clv IS NOT NULL
          AND date >= date('now', '-30 days')
        GROUP BY stat_category, direction
        HAVING COUNT(*) >= 10
    """
    return conn.execute(query).fetchall()


# ── 7.4 Platt Scaling Readiness ───────────────────────────────────────────────

def check_calibration_readiness(conn) -> list:
    """Returns list of (category, direction) pairs ready for Platt scaling (200+ picks)."""
    query = """
        SELECT dp.stat_category, dp.direction, COUNT(*) as n
        FROM daily_picks dp
        JOIN pick_results pr ON dp.id = pr.pick_id
        WHERE dp.date >= date('now', '-90 days')
          AND dp.source_file NOT LIKE 'bulk_backtest%'
          AND dp.source_file NOT LIKE 'synthetic%'
        GROUP BY dp.stat_category, dp.direction
        HAVING COUNT(*) >= 200
    """
    rows = conn.execute(query).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


# ── 7.5 Kelly Promotion Criteria ─────────────────────────────────────────────

def check_kelly_criteria(conn) -> dict:
    """Check all 6 Kelly re-enable criteria."""

    # 1. Total resolved picks
    total_picks = conn.execute("""
        SELECT COUNT(*) FROM daily_picks dp
        JOIN pick_results pr ON dp.id = pr.pick_id
        WHERE dp.source_file NOT LIKE 'bulk_backtest%'
          AND dp.source_file NOT LIKE 'synthetic%'
    """).fetchone()[0]

    # 2. Platt scaling active categories
    platt_ready = check_calibration_readiness(conn)

    # 3. Brier < 0.20 for all categories with 30+ picks
    cats = conn.execute("""
        SELECT DISTINCT stat_category, direction FROM daily_picks dp
        JOIN pick_results pr ON dp.id = pr.pick_id
        WHERE dp.date >= date('now', '-90 days')
          AND dp.source_file NOT LIKE 'bulk_backtest%'
          AND dp.source_file NOT LIKE 'synthetic%'
        GROUP BY stat_category, direction HAVING COUNT(*) >= 30
    """).fetchall()

    brier_pass = True
    for cat, direction in cats:
        b = compute_brier_score(cat, direction, conn)
        if b is not None and b >= 0.20:
            brier_pass = False
            break

    # 4. CLV positive rate
    clv_rows = compute_clv_stats(conn)
    clv_pass = False
    if clv_rows:
        pos_rates = [r[3] / r[4] * 100 for r in clv_rows if r[4] > 0]
        clv_pass = all(r >= 60 for r in pos_rates)

    # 5. Parlay hit rate
    parlay_row = conn.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins
        FROM parlays
        WHERE result IS NOT NULL
          AND game_date >= date('now', '-30 days')
    """).fetchone()
    parlay_n = parlay_row[0]
    parlay_wins = parlay_row[1] or 0
    parlay_hit_rate = (parlay_wins / parlay_n * 100) if parlay_n >= 30 else 0
    parlay_pass = parlay_hit_rate >= 20

    # 6. No category < 40% win rate in last 30 days
    low_cats = conn.execute("""
        SELECT dp.stat_category, dp.direction, ROUND(AVG(pr.is_win)*100,1) as wr
        FROM daily_picks dp
        JOIN pick_results pr ON dp.id = pr.pick_id
        WHERE dp.date >= date('now', '-30 days')
          AND dp.source_file NOT LIKE 'bulk_backtest%'
          AND dp.source_file NOT LIKE 'synthetic%'
        GROUP BY dp.stat_category, dp.direction
        HAVING COUNT(*) >= 5 AND AVG(pr.is_win) < 0.40
    """).fetchall()
    no_low_cat_pass = len(low_cats) == 0

    return {
        'total_picks': total_picks,
        'picks_pass': total_picks >= 500,
        'platt_count': len(platt_ready),
        'platt_pass': len(platt_ready) >= 2,
        'brier_pass': brier_pass,
        'clv_pass': clv_pass,
        'parlay_hit_rate': parlay_hit_rate,
        'parlay_n': parlay_n,
        'parlay_pass': parlay_pass,
        'low_cats': low_cats,
        'no_low_cat_pass': no_low_cat_pass,
        'all_pass': all([
            total_picks >= 500,
            len(platt_ready) >= 2,
            brier_pass,
            clv_pass,
            parlay_pass,
            no_low_cat_pass,
        ])
    }


# ── 7.6 Daily Report ─────────────────────────────────────────────────────────

def generate_monitoring_report(date_str: str = None) -> str:
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')

    conn = sqlite3.connect(TRACKER_DB)
    conn.row_factory = sqlite3.Row
    lines = []
    sep = '=' * 55

    lines.append(sep)
    lines.append(f"  MONITORING REPORT  {date_str}")
    lines.append(sep)

    # Pick Volume (last 30 days, real only)
    vol = conn.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN pr.pick_id IS NOT NULL THEN 1 ELSE 0 END) as resolved
        FROM daily_picks dp
        LEFT JOIN pick_results pr ON dp.id = pr.pick_id
        WHERE dp.date >= date('now', '-30 days')
          AND dp.source_file NOT LIKE 'bulk_backtest%'
          AND dp.source_file NOT LIKE 'synthetic%'
    """).fetchone()
    total_picks = vol[0] or 0
    resolved_picks = vol[1] or 0
    pending = total_picks - resolved_picks

    lines.append("\nPICK VOLUME (last 30 days):")
    lines.append(f"  Total: {total_picks}  |  Resolved: {resolved_picks}  |  Pending: {pending}")

    # Win rates by category
    lines.append("\nWIN RATES BY CATEGORY:")
    cat_rows = conn.execute("""
        SELECT dp.stat_category, dp.direction, COUNT(*) as n, ROUND(AVG(pr.is_win)*100,1) as wr
        FROM daily_picks dp
        JOIN pick_results pr ON dp.id = pr.pick_id
        WHERE dp.date >= date('now', '-30 days')
          AND dp.source_file NOT LIKE 'bulk_backtest%'
          AND dp.source_file NOT LIKE 'synthetic%'
        GROUP BY dp.stat_category, dp.direction
        ORDER BY dp.stat_category
    """).fetchall()

    for cat, direction, n, wr in cat_rows:
        brier = compute_brier_score(cat, direction, conn)
        brier_str = f"Brier: {brier:.3f}" if brier is not None else "Brier: n/a"
        flag = " ⚠️" if (wr is not None and wr < 40) else ""
        lines.append(f"  {cat} {direction}:  {wr}% (n={n}) [{brier_str}]{flag}")

    # Parlay performance
    parlay_row = conn.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
               ROUND(SUM(CASE WHEN result='WIN' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) as hit_rate,
               ROUND(SUM(roi) * 100, 2) as total_roi_pct
        FROM parlays
        WHERE result IS NOT NULL
          AND game_date >= date('now', '-30 days')
    """).fetchone()

    lines.append("\nPARLAY PERFORMANCE:")
    if parlay_row and parlay_row[0]:
        lines.append(f"  Hit rate: {parlay_row[2]}% ({parlay_row[1]}/{parlay_row[0]})  |  ROI: {parlay_row[3]}%")
    else:
        lines.append("  No resolved parlays in last 30 days.")

    # CLV
    lines.append("\nCLV (last 30 days):")
    clv_rows = compute_clv_stats(conn)
    if clv_rows:
        for cat, direction, avg_clv, pos_n, n in clv_rows:
            pos_rate = (pos_n / n * 100) if n > 0 else 0
            status = "EDGE CONFIRMED" if pos_rate >= 65 else "REVIEW"
            lines.append(f"  {cat} {direction}: Avg CLV={avg_clv:+.3f}  |  Positive rate: {pos_rate:.0f}%  |  [{status}]")
    else:
        lines.append("  No CLV data (needs closing line tracking).")

    # Calibration status
    platt_ready = check_calibration_readiness(conn)
    cal_params_path = os.path.join(MODEL_DIR, 'config', 'calibration_params.json')
    platt_active = os.path.exists(cal_params_path)

    lines.append("\nCALIBRATION STATUS:")
    if platt_ready:
        labels = [f"{r[0]}_{r[1]} (n={r[2]})" for r in platt_ready]
        lines.append(f"  Platt scaling: READY for {', '.join(labels)}")
        lines.append("  ▶ Run: python3 -m etl.build_calibration")
    else:
        lines.append("  Platt scaling: PENDING — requires 200+ picks per category")

    lines.append(f"  Platt params file: {'EXISTS' if platt_active else 'NOT FOUND'}")

    # Kelly status
    kelly_data = check_kelly_criteria(conn)
    lines.append("\nKELLY CRITERION:")
    lines.append(f"  [{'✅' if kelly_data['picks_pass'] else '❌'}] 500+ resolved picks: {kelly_data['total_picks']}")
    lines.append(f"  [{'✅' if kelly_data['platt_pass'] else '❌'}] 2+ Platt categories: {kelly_data['platt_count']}")
    lines.append(f"  [{'✅' if kelly_data['brier_pass'] else '❌'}] Brier < 0.20 all cats")
    lines.append(f"  [{'✅' if kelly_data['clv_pass'] else '❌'}] CLV positive rate > 60%")
    lines.append(f"  [{'✅' if kelly_data['parlay_pass'] else '❌'}] Parlay hit rate > 20%: {kelly_data['parlay_hit_rate']:.1f}% (n={kelly_data['parlay_n']})")
    lines.append(f"  [{'✅' if kelly_data['no_low_cat_pass'] else '❌'}] No category < 40% WR")
    lines.append(f"  STATUS: {'🟢 KELLY ENABLED' if kelly_data['all_pass'] else '🔴 KELLY DISABLED — USE FLAT 0.2% STAKE'}")

    # Alerts summary
    lines.append("\nALERTS:")
    alerts = []
    for cat, direction, wr in (kelly_data['low_cats'] or []):
        alerts.append(f"  🔴 {cat} {direction} win rate {wr}% — below 40% threshold")
    if not alerts:
        lines.append("  🟢 All systems nominal")
    else:
        lines.extend(alerts)

    lines.append("\n" + sep)

    conn.close()
    return '\n'.join(lines)


# ── Season Summary Report ─────────────────────────────────────────────────────

def end_of_season_report(start_date: str, end_date: str, output_path: str) -> str:
    """
    Produce a full-season summary report for an explicit date window.
    Writes to output_path and returns the report string.

    Parameters
    ----------
    start_date : str   e.g. '2026-01-01'
    end_date   : str   e.g. '2026-04-13'
    output_path: str   absolute path for the .txt output file
    """
    conn = sqlite3.connect(TRACKER_DB)
    conn.row_factory = sqlite3.Row

    sep   = '=' * 62
    lines = []

    lines.append(sep)
    lines.append("  2025-26 NBA SEASON — FINAL PERFORMANCE REPORT")
    lines.append(f"  Period : {start_date}  →  {end_date}")
    lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(sep)

    date_filter = "dp.date >= ? AND dp.date <= ? AND dp.source_mode != 'synthetic_circular'"

    # ── Block 1 · Calibration Curves ─────────────────────────────────────────
    lines.append("\n── CALIBRATION CURVES ──────────────────────────────────────")

    cal_rows = conn.execute(f"""
        SELECT dp.stat_category, dp.direction,
               dp.sim_confidence / 100.0 AS pred,
               pr.is_win
        FROM daily_picks dp
        JOIN pick_results pr ON dp.id = pr.pick_id
        WHERE {date_filter}
          AND dp.sim_confidence IS NOT NULL
    """, (start_date, end_date)).fetchall()

    import pandas as pd
    df_cal = pd.DataFrame([dict(r) for r in cal_rows])

    if df_cal.empty:
        lines.append("  No calibration data.")
    else:
        bins   = [0.50, 0.60, 0.70, 0.80, 1.0]
        labels = ["Bin 50-60%", "Bin 60-70%", "Bin 70-80%", "Bin 80%+"]
        mids   = {"Bin 50-60%": 55.0, "Bin 60-70%": 65.0, "Bin 70-80%": 75.0, "Bin 80%+": 85.0}
        df_cal['bin'] = pd.cut(df_cal['pred'], bins=bins, labels=labels, right=False)

        for cat_dir, grp in df_cal.groupby(['stat_category', 'direction'], observed=True):
            cat, direction = cat_dir
            lines.append(f"{cat} {direction}:")
            result = grp.groupby('bin', observed=False)['is_win'].agg(['mean', 'count'])
            for bin_lbl, row in result.iterrows():
                n_bin = int(row['count'])
                if n_bin == 0:
                    continue
                pred_pct = mids[str(bin_lbl)]
                actual_pct = row['mean'] * 100
                lines.append(
                    f"  {bin_lbl}: predicted={pred_pct:.1f}%"
                    f" | actual={actual_pct:.1f}%"
                    f" | n={n_bin}"
                )

    # ── Block 2 · Hit Rates, MAE & Bias by Category ───────────────────────────
    lines.append("\n── HIT RATES, MAE & BIAS BY CATEGORY ───────────────────────")

    bias_rows = conn.execute(f"""
        SELECT dp.stat_category,
               dp.direction,
               COUNT(*)                     AS n,
               ROUND(AVG(pr.is_win)*100, 1) AS win_rate,
               ROUND(AVG(pr.diff_val), 3)   AS mean_bias,
               ROUND(AVG(ABS(pr.diff_val)), 3) AS mae
        FROM daily_picks dp
        JOIN pick_results pr ON dp.id = pr.pick_id
        WHERE {date_filter}
        GROUP BY dp.stat_category, dp.direction
        ORDER BY dp.stat_category, dp.direction
    """, (start_date, end_date)).fetchall()

    if not bias_rows:
        lines.append("  No resolved pick data.")
    else:
        hdr = f"  {'CATEGORY':<12} {'DIR':<6} {'N':>5}  {'WR%':>6}  {'BIAS':>8}  {'MAE':>6}"
        lines.append(hdr)
        lines.append("  " + "-" * 50)
        for r in bias_rows:
            wr   = r['win_rate'] if r['win_rate'] is not None else 0.0
            bias = r['mean_bias'] if r['mean_bias'] is not None else 0.0
            mae  = r['mae'] if r['mae'] is not None else 0.0
            flag = "  ⚠️" if wr < 40.0 else ""
            lines.append(
                f"  {r['stat_category']:<12} {r['direction']:<6} {r['n']:>5}"
                f"  {wr:>5.1f}%  {bias:>+8.3f}  {mae:>6.3f}{flag}"
            )

    # ── Block 3 · Singles & Parlay ROI ───────────────────────────────────────
    lines.append("\n── ROI ANALYSIS ─────────────────────────────────────────────")

    s_row = conn.execute(f"""
        SELECT COUNT(*) AS n, SUM(pr.is_win) AS wins
        FROM daily_picks dp
        JOIN pick_results pr ON dp.id = pr.pick_id
        WHERE {date_filter}
    """, (start_date, end_date)).fetchone()

    n_s   = s_row['n']    if s_row else 0
    wins_s= s_row['wins'] if s_row and s_row['wins'] else 0
    roi_s = ((wins_s * 0.909) - (n_s - wins_s)) / n_s * 100 if n_s > 0 else 0.0
    lines.append(f"  Singles — flat -110 stake  : {roi_s:+.2f}%  [n={n_s}]")

    p_row = conn.execute("""
        SELECT COUNT(*) AS n,
               SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) AS wins,
               ROUND(SUM(CASE WHEN roi IS NOT NULL THEN roi ELSE 0 END)*100, 2) AS total_roi
        FROM parlays
        WHERE game_date >= ? AND game_date <= ?
          AND result IS NOT NULL
    """, (start_date, end_date)).fetchone()

    n_p    = p_row['n']         if p_row else 0
    wins_p = p_row['wins']      if p_row and p_row['wins'] else 0
    roi_p  = p_row['total_roi'] if p_row and p_row['total_roi'] else 0.0
    hr_p   = wins_p / n_p * 100 if n_p > 0 else 0.0
    lines.append(f"  Parlays — hit rate {hr_p:.1f}%       : {roi_p:+.2f}%  [n={n_p}]")

    # ── Block 4 · CLV ─────────────────────────────────────────────────────────
    lines.append("\n── CLV AVERAGES & POSITIVE RATE ─────────────────────────────")

    clv_rows = conn.execute(f"""
        SELECT dp.stat_category, dp.direction,
               ROUND(AVG(dp.clv), 3)  AS avg_clv,
               SUM(CASE WHEN dp.clv > 0 THEN 1 ELSE 0 END) AS pos_n,
               COUNT(*) AS n
        FROM daily_picks dp
        WHERE {date_filter}
          AND dp.clv IS NOT NULL
        GROUP BY dp.stat_category, dp.direction
        HAVING COUNT(*) >= 5
        ORDER BY dp.stat_category, dp.direction
    """, (start_date, end_date)).fetchall()

    if not clv_rows:
        lines.append("  No CLV data for this window.")
    else:
        for r in clv_rows:
            avg   = r['avg_clv'] if r['avg_clv'] is not None else 0.0
            pos_r = r['pos_n'] / r['n'] * 100 if r['n'] > 0 else 0.0
            status = "EDGE CONFIRMED" if pos_r >= 65 else ("REVIEW" if pos_r >= 50 else "NO EDGE")
            lines.append(
                f"  {r['stat_category']} {r['direction']:<6}"
                f"  Avg CLV={avg:+.3f}  Pos%={pos_r:.0f}%  (n={r['n']})  [{status}]"
            )

    lines.append("\n" + sep)

    conn.close()
    report = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate daily monitoring report')
    parser.add_argument('--date',   type=str, default=None,  help='Date YYYY-MM-DD (default: today)')
    parser.add_argument('--season', action='store_true',     help='Generate 2025-26 season summary instead')
    parser.add_argument('--no-save',action='store_true',     help='Print only, do not save to file')
    parser.add_argument('--out',    type=str, default=None,  help='Override output file path')
    args = parser.parse_args()

    if args.season:
        out_path = os.path.join(LOGS_DIR, 'season_2025_26_final_report.txt')
        report   = end_of_season_report('2026-01-01', '2026-04-13', out_path)
        try:
            print(report)
        except UnicodeEncodeError:
            print(report.encode('ascii', 'replace').decode('ascii'))
        print(f"\n[+] Season report saved: {out_path}")
        return

    report = generate_monitoring_report(args.date)
    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode('ascii', 'replace').decode('ascii'))

    if not args.no_save:
        date_str = args.date or datetime.now().strftime('%Y-%m-%d')
        out_path = args.out if args.out else os.path.join(LOGS_DIR, f'monitoring_{date_str}.txt')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n[+] Report saved to: {out_path}")



if __name__ == '__main__':
    main()
