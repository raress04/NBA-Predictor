"""
scripts/repair_measurement_20261106.py
=====================================
One-off, idempotent data repair for Step 1 of Plans/CRITICAL_BUGS_FOUND_06.11.md
("fix the measurement"). Dry-run by default: works on in-memory copies and only
prints what would change. Pass --apply to write.

    python3 -m scripts.repair_measurement_20261106            # dry run
    python3 -m scripts.repair_measurement_20261106 --apply    # write

What it does
------------
A. nba_data.box_scores: remove exact duplicate (game_id, player_id) rows (keeps the
   lowest id) and add UNIQUE(game_id, player_id) so writers can't duplicate again.
B. bet_tracker: add candidate stage columns to shadow_picks / ml_shadow_picks.
C. shadow_picks: re-date rows written by after-midnight runs before the Oct 30
   2.5 h fix (game_date = next calendar day). Only batches whose players match the
   US-Eastern run date in box scores (>= 80%) and beat the stored date by >= 50 pts.
D. shadow_picks: mark rows from an earlier run of the same slate as superseded=1.
E. shadow_picks: back-fill conf_stage / raw_sim_freq / posterior_conf / blended_conf
   for 2026-27 rows (legacy sim_confidence held raw P(OVER) for early gates).
F. shadow_picks: re-grade every 2026-27 prop row against real box scores.
G. ml_shadow_picks: replace sim_confidence with the direction-specific raw sim
   frequency (from the matching shadow_picks row) and re-grade every row against
   real box scores (they had been graded against placeholder zeros).

Backups are the caller's job (database/archive/bet_tracker_backup_20261106_pre_step1.db).
"""

import argparse
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(HERE)
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from utils.box_actuals import ActualsIndex, actual_for, grade  # noqa: E402
from database.bet_tracker_schema import SHADOW_STAGE_COLUMNS   # noqa: E402

PROP_CATS = ('POINTS', 'REBOUNDS', 'ASSISTS')

RAW_OVER_PREFIXES = ('MIN_GAP', 'BLOWOUT_RISK', 'MINUTES_RESTRICTION', 'MINUTES_RISK',
                     'LOW_TRIGGER', 'LOW_SAFE_CONF', 'LOW_LINE_VOLATILITY', 'MISSING_DATA',
                     'NEAR_MISS', 'NOT_VALUABLE')
POSTERIOR_PREFIXES = ('HIGH_VARIANCE', 'PARLAY_GATE_CLOSED')
BLENDED_PREFIXES = ('LOW_CONFIDENCE', 'DAILY_CAP')


def classify_stage(reason: str, stat_category: str) -> str:
    """Which value the legacy sim_confidence column holds for a 2026-27 row."""
    r = (reason or '').strip()
    cat = (stat_category or '').upper()
    if cat in ('SPREAD', 'TOTAL') and r.startswith(('BANNED_CATEGORY', 'MARKET_INVERSION')):
        return 'none'
    if r.startswith('LOW_EDGE'):
        return 'blended' if 'min=' in r else 'raw_over'
    if r.startswith('LOW_CONF(') or r.startswith('LOW_CONF_'):
        return 'raw_over'      # compute_prop_edge gate (raw frequency)
    if r.startswith(BLENDED_PREFIXES):
        return 'blended'
    if r.startswith(POSTERIOR_PREFIXES):
        return 'posterior'
    if r.startswith(RAW_OVER_PREFIXES) or r.startswith('BANNED_CATEGORY'):
        return 'raw_over'
    return 'unknown'


# ── helpers ──────────────────────────────────────────────────────────────────

def open_db(path: str, apply: bool) -> sqlite3.Connection:
    """Apply -> the file itself. Dry run -> an in-memory copy."""
    src = sqlite3.connect(path)
    if apply:
        return src
    mem = sqlite3.connect(':memory:')
    src.backup(mem)
    src.close()
    return mem


def load_actuals_cache(nba: sqlite3.Connection):
    """date -> ActualsIndex, from the (deduplicated) box_scores table."""
    by_date = defaultdict(dict)
    seen = set()
    for game_id, pid, name, minutes, pts, reb, ast, gdate in nba.execute(
            "SELECT game_id, player_id, player_name, minutes, pts, reb, ast, "
            "substr(game_date,1,10) FROM box_scores WHERE game_date >= '2026-10-01'"):
        if (game_id, pid) in seen or not name:
            continue
        seen.add((game_id, pid))
        try:
            m = float(minutes) if minutes not in (None, '') and ':' not in str(minutes) else None
            if minutes and ':' in str(minutes):
                mm, ss = str(minutes).split(':', 1)
                m = float(mm) + float(ss) / 60
        except ValueError:
            m = None
        by_date[gdate][name] = {'POINTS': pts, 'REBOUNDS': reb, 'ASSISTS': ast, 'minutes': m}
    return {d: ActualsIndex(a) for d, a in by_date.items()}


def win_rates(conn, table, flag_col, where=""):
    q = (f"SELECT direction, COUNT(*), SUM({flag_col} IS NOT NULL), "
         f"ROUND(AVG({flag_col})*100,1) FROM {table} "
         f"WHERE stat_category IN ('POINTS','REBOUNDS','ASSISTS') {where} GROUP BY direction")
    return conn.execute(q).fetchall()


# ── steps ────────────────────────────────────────────────────────────────────

def step_a_dedupe_box_scores(nba, log):
    before = nba.execute("SELECT COUNT(*) FROM box_scores").fetchone()[0]
    by_season = nba.execute("""
        SELECT season, SUM(c - 1) FROM (
            SELECT season, COUNT(*) c FROM box_scores GROUP BY game_id, player_id HAVING c > 1
        ) GROUP BY season""").fetchall()
    nba.execute("""
        DELETE FROM box_scores
        WHERE id NOT IN (SELECT MIN(id) FROM box_scores GROUP BY game_id, player_id)""")
    after = nba.execute("SELECT COUNT(*) FROM box_scores").fetchone()[0]
    nba.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_box_scores_game_player "
                "ON box_scores(game_id, player_id)")
    log(f"[A] box_scores: removed {before - after} duplicate rows "
        f"(by season: {dict(by_season)}); UNIQUE(game_id, player_id) index in place.")


def step_b_columns(bet, log):
    sp = {r[1] for r in bet.execute("PRAGMA table_info(shadow_picks)")}
    added = []
    for col, ctype in SHADOW_STAGE_COLUMNS.items():
        if col not in sp:
            bet.execute(f"ALTER TABLE shadow_picks ADD COLUMN {col} {ctype}")
            added.append(col)
    msp = {r[1] for r in bet.execute("PRAGMA table_info(ml_shadow_picks)")}
    if 'conf_stage' not in msp:
        bet.execute("ALTER TABLE ml_shadow_picks ADD COLUMN conf_stage TEXT")
        added.append('ml_shadow_picks.conf_stage')
    log(f"[B] columns added: {added or 'none (already present)'}")


def step_c_redate(bet, actuals, season_start, log):
    batches = bet.execute("""
        SELECT game_date, date(created_at, '-5 hours') AS et_date, COUNT(*)
        FROM shadow_picks
        WHERE game_date >= ? AND created_at IS NOT NULL
        GROUP BY 1, 2 HAVING game_date != et_date""", (season_start,)).fetchall()
    total = 0
    for label_date, et_date, n in batches:
        players = {p for (p,) in bet.execute("""
            SELECT DISTINCT player FROM shadow_picks
            WHERE game_date = ? AND date(created_at, '-5 hours') = ?
              AND stat_category IN ('POINTS','REBOUNDS','ASSISTS')""", (label_date, et_date))}
        if not players:
            log(f"[C] {label_date} (run ET date {et_date}): {n} rows, no props — skipped")
            continue
        on_et = sum(1 for p in players if actuals.get(et_date) and actuals[et_date].get(p))
        on_label = sum(1 for p in players if actuals.get(label_date) and actuals[label_date].get(p))
        cov_et, cov_label = on_et / len(players), on_label / len(players)
        # Players on back-to-backs appear on both dates, so require a clear margin
        # rather than near-zero coverage on the stored date.
        if cov_et >= 0.8 and (cov_et - cov_label) >= 0.5:
            bet.execute("""
                UPDATE shadow_picks SET game_date = ?, actual_value = NULL,
                       actual_result = NULL, resolved_at = NULL
                WHERE game_date = ? AND date(created_at, '-5 hours') = ?""",
                        (et_date, label_date, et_date))
            total += n
            log(f"[C] re-dated {n} rows {label_date} -> {et_date} "
                f"(players in box scores: {cov_et:.0%} on {et_date}, {cov_label:.0%} on {label_date})")
        else:
            log(f"[C] {label_date} (run ET date {et_date}): {n} rows NOT re-dated — evidence "
                f"inconclusive ({cov_et:.0%} vs {cov_label:.0%})")
    log(f"[C] total re-dated: {total}")


def step_d_superseded(bet, season_start, log):
    bet.execute("UPDATE shadow_picks SET superseded = 0 WHERE game_date >= ?", (season_start,))
    bet.execute("""
        UPDATE shadow_picks SET superseded = 1
        WHERE game_date >= ? AND stat_category IN ('POINTS','REBOUNDS','ASSISTS')
          AND (julianday((SELECT MAX(s2.created_at) FROM shadow_picks s2
                           WHERE s2.game_date = shadow_picks.game_date
                             AND s2.player = shadow_picks.player
                             AND s2.stat_category = shadow_picks.stat_category))
               - julianday(created_at)) * 86400 >= 600""", (season_start,))
    n = bet.execute("SELECT COUNT(*) FROM shadow_picks WHERE superseded = 1 AND game_date >= ?",
                    (season_start,)).fetchone()[0]
    log(f"[D] rows marked superseded (earlier run of the same slate): {n}")


def step_e_backfill_stages(bet, season_start, log):
    rows = bet.execute("""
        SELECT id, ban_reason, stat_category, direction, sim_confidence
        FROM shadow_picks WHERE game_date >= ? AND conf_stage IS NULL""", (season_start,)).fetchall()
    counts = defaultdict(int)
    for rid, reason, cat, direction, sc in rows:
        stage = classify_stage(reason, cat)
        counts[stage] += 1
        raw = post = blend = None
        if sc is not None and stage == 'raw_over' and (cat or '').upper() in PROP_CATS:
            raw = sc if (direction or '').upper() == 'OVER' else 100.0 - sc
        elif stage == 'posterior':
            post = sc
        elif stage == 'blended':
            blend = sc
        bet.execute("""UPDATE shadow_picks SET conf_stage = ?, raw_sim_freq = COALESCE(raw_sim_freq, ?),
                       posterior_conf = COALESCE(posterior_conf, ?), blended_conf = COALESCE(blended_conf, ?)
                       WHERE id = ?""", (stage, raw, post, blend, rid))
    log(f"[E] stage back-fill: {dict(counts)}")


def step_f_regrade_shadow(bet, actuals, season_start, log):
    before = bet.execute("""SELECT COUNT(*), SUM(actual_result IS NOT NULL) FROM shadow_picks
        WHERE game_date >= ? AND stat_category IN ('POINTS','REBOUNDS','ASSISTS')""",
                         (season_start,)).fetchone()
    rows = bet.execute("""
        SELECT id, game_date, player, stat_category, direction, line FROM shadow_picks
        WHERE game_date >= ? AND stat_category IN ('POINTS','REBOUNDS','ASSISTS')""",
                       (season_start,)).fetchall()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    graded = pushes = missing = 0
    for rid, gdate, player, cat, direction, line in rows:
        idx = actuals.get(gdate)
        actual = actual_for(idx, player, cat) if idx else None
        if actual is None or line is None:
            bet.execute("UPDATE shadow_picks SET actual_value=NULL, actual_result=NULL, resolved_at=NULL "
                        "WHERE id=?", (rid,))
            missing += 1
            continue
        result = grade(direction, actual, line)
        pushes += result is None
        bet.execute("UPDATE shadow_picks SET actual_value=?, actual_result=?, resolved_at=? WHERE id=?",
                    (actual, result, now, rid))
        graded += 1
    log(f"[F] shadow_picks props: before {before[1]}/{before[0]} graded -> after {graded}/{len(rows)} "
        f"graded ({pushes} pushes, {missing} no box score / DNP)")


def step_g_ml_shadow(bet, actuals, log):
    before = win_rates(bet, 'ml_shadow_picks', 'is_win')
    rows = bet.execute("""SELECT id, game_date, player, stat_category, direction, line, created_at
                          FROM ml_shadow_picks WHERE conf_stage IS NULL""").fetchall()
    matched = 0
    for rid, gdate, player, cat, direction, line, created in rows:
        hit = bet.execute("""
            SELECT raw_sim_freq FROM shadow_picks
            WHERE game_date = ? AND player = ? AND stat_category = ? AND direction = ?
              AND ABS(COALESCE(line, -1) - COALESCE(?, -1)) < 1e-9 AND raw_sim_freq IS NOT NULL
            ORDER BY ABS(julianday(created_at) - julianday(?)) LIMIT 1""",
                          (gdate, player, cat, direction, line, created)).fetchone()
        if hit:
            bet.execute("UPDATE ml_shadow_picks SET sim_confidence = ?, conf_stage = 'raw_dir' WHERE id = ?",
                        (round(hit[0], 4), rid))
            matched += 1
        else:
            bet.execute("UPDATE ml_shadow_picks SET conf_stage = 'legacy_unknown' WHERE id = ?", (rid,))
    log(f"[G] ml_shadow sim_confidence -> direction-specific raw freq: {matched}/{len(rows)} matched")

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    graded = missing = 0
    for rid, gdate, player, cat, direction, line in bet.execute(
            "SELECT id, game_date, player, stat_category, direction, line FROM ml_shadow_picks").fetchall():
        idx = actuals.get(gdate)
        actual = actual_for(idx, player, cat) if idx else None
        if actual is None or line is None:
            bet.execute("UPDATE ml_shadow_picks SET actual_value=NULL, is_win=NULL, resolved_at=NULL WHERE id=?",
                        (rid,))
            missing += 1
            continue
        bet.execute("UPDATE ml_shadow_picks SET actual_value=?, is_win=?, resolved_at=? WHERE id=?",
                    (actual, grade(direction, actual, line), now, rid))
        graded += 1
    after = win_rates(bet, 'ml_shadow_picks', 'is_win')
    log(f"[G] ml_shadow re-graded from box scores: {graded} graded, {missing} unresolved")
    log(f"[G] win rate by direction BEFORE (direction, n, resolved, win%): {before}")
    log(f"[G] win rate by direction AFTER  (direction, n, resolved, win%): {after}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bet-db', default=os.path.join(MODEL_DIR, 'database', 'bet_tracker.db'))
    ap.add_argument('--nba-db', default=os.path.join(MODEL_DIR, 'database', 'nba_data.db'))
    ap.add_argument('--season-start', default='2026-10-20')
    ap.add_argument('--apply', action='store_true', help='write changes (default: dry run)')
    ap.add_argument('--report', default=None, help='also write the log to this file')
    args = ap.parse_args(argv)

    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    log(f"repair_measurement_20261106 — {'APPLY' if args.apply else 'DRY RUN (nothing written)'} "
        f"— {datetime.now():%Y-%m-%d %H:%M}")
    nba = open_db(args.nba_db, args.apply)
    bet = open_db(args.bet_db, args.apply)
    try:
        step_a_dedupe_box_scores(nba, log)
        nba.commit()
        actuals = load_actuals_cache(nba)
        step_b_columns(bet, log)
        step_c_redate(bet, actuals, args.season_start, log)
        step_d_superseded(bet, args.season_start, log)
        step_e_backfill_stages(bet, args.season_start, log)
        step_f_regrade_shadow(bet, actuals, args.season_start, log)
        step_g_ml_shadow(bet, actuals, log)
        bet.commit()
    finally:
        nba.close()
        bet.close()

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
