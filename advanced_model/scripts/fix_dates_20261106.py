"""
scripts/fix_dates_20261106.py
=============================
Database part of the date fixes approved on 06.11.2026 (fixes 1, 3 and 4 in
Plans/CRITICAL_BUGS_FOUND_06.11.md, Section 15). Dry run by default.

    python3 -m scripts.fix_dates_20261106 --model-dir <advanced_model copy> [--apply]

Rule for every re-date
----------------------
The true slate date is the US-Eastern calendar date of the run that produced the
rows (shadow_picks.created_at, or the mtime of the prediction file named in
source_file). A batch is only moved when box scores agree: >= 80% of its prop
players played on the new date and that is >= 30 points more than on the stored
date. When the new date is beyond the box-score coverage (playoffs after May 10)
the batch is moved on run-time evidence only and its grades are cleared.
Batches touching a date with a known data gap (2026-03-15) are left alone.

Steps
-----
 1. shadow_picks         re-date mismatched runs (2025-26 + 2026-27), re-grade, re-mark superseded
 2. daily_picks           re-date live batches; re-grade pick_results from box scores, quarantine
                          grades that cannot be recomputed (pick_results_quarantine_20261106)
 3. projection_outcomes   re-date live / live_pending batches (incl. the mixed Oct 26, 2026 batch)
 4. projection_outcomes_raw re-date live batches
 5. source_file           follow the prediction-file renames done by the file step
 6. YYYYMMDD -> YYYY-MM-DD in projection_outcomes
 7. game_results          rebuilt from nba_data.db games + box scores for every date it covers
"""
import argparse
import datetime as dt
import glob
import os
import sqlite3
import sys
from collections import defaultdict

PROPS = ('POINTS', 'REBOUNDS', 'ASSISTS')
KNOWN_GAPS = {'2026-03-15'}          # games missing from nba_data.db (needs backfill)
SEASON_START = '2026-03-01'          # live tracking (shadow / live picks) begins in March 2026


def iso(s):
    s = str(s)[:10]
    if len(s) == 8 and s.isdigit():
        s = f'{s[:4]}-{s[4:6]}-{s[6:8]}'
    return s


def shift(d, days):
    return (dt.date.fromisoformat(d) + dt.timedelta(days=days)).isoformat()


class Box:
    """Box-score lookups (deduplicated) from nba_data.db."""

    def __init__(self, nba):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.box_actuals import ActualsIndex
        by_date = defaultdict(dict)
        seen = set()
        for gid, pid, name, mins, pts, reb, ast, d in nba.execute(
                "SELECT game_id, player_id, player_name, minutes, pts, reb, ast, substr(game_date,1,10) "
                "FROM box_scores WHERE game_date >= '2025-10-01'"):
            if (gid, pid) in seen or not name:
                continue
            seen.add((gid, pid))
            try:
                m = float(mins) if mins not in (None, '') and ':' not in str(mins) else None
            except ValueError:
                m = None
            by_date[d][name] = {'POINTS': pts, 'REBOUNDS': reb, 'ASSISTS': ast, 'minutes': m}
        self.idx = {d: ActualsIndex(a) for d, a in by_date.items()}
        self.max_date = max(by_date)
        self.dates = set(by_date)

    def played(self, player, d):
        i = self.idx.get(d)
        s = i.get(player) if i else None
        return bool(s) and (s.get('minutes') is None or s['minutes'] > 0)

    def actual(self, player, cat, d):
        i = self.idx.get(d)
        s = i.get(player) if i else None
        if not s or (s.get('minutes') is not None and s['minutes'] <= 0):
            return None
        v = s.get(cat.upper())
        return None if v is None else float(v)

    def coverage(self, players, d):
        players = [p for p in players if p]
        return sum(self.played(p, d) for p in players) / len(players) if players else 0.0

    def covered(self, d):
        return d <= self.max_date


def decide(box, players, stored, target):
    """Return (ok, reason) for moving a batch stored -> target."""
    if (stored in KNOWN_GAPS and stored not in box.dates) or (target in KNOWN_GAPS and target not in box.dates):
        return False, 'known data gap — left for backfill'   # lifts itself once the games are downloaded
    if not box.covered(target):
        return (True, 'run-time evidence only (no box scores yet)') if not box.covered(stored) else \
            (False, 'target beyond box coverage but stored date covered')
    c_t, c_s = box.coverage(players, target), box.coverage(players, stored)
    if c_t >= 0.8 and c_t - c_s >= 0.3:
        return True, f'box evidence {c_t:.0%} vs {c_s:.0%}'
    return False, f'evidence too weak ({c_t:.0%} vs {c_s:.0%})'


def grade(direction, actual, line):
    if actual is None or line is None or actual == line:
        return None
    d = str(direction).upper()
    if d == 'OVER':
        return 1 if actual > line else 0
    if d == 'UNDER':
        return 1 if actual < line else 0
    return None


def file_run_et(model_dir, source_file):
    """US-Eastern date of the run that wrote a prediction file (file mtime)."""
    if not source_file:
        return None
    name = os.path.basename(str(source_file))
    hits = glob.glob(os.path.join(model_dir, 'predictions', '*', '*', '*', name)) + \
        glob.glob(os.path.join(model_dir, 'predictions', '*', '*', name))
    hits = [h for h in hits if 'first_gen_prediction' not in h and 'backtest' not in h]
    if not hits:
        return None
    mt = dt.datetime.utcfromtimestamp(os.path.getmtime(hits[0]))
    et = (mt - dt.timedelta(hours=4)).date().isoformat()
    label = name[len('predicts_'):len('predicts_') + 10] if name.startswith('predicts_') else None
    try:
        if label and abs((dt.date.fromisoformat(et) - dt.date.fromisoformat(label)).days) > 2:
            return None                   # file re-written much later — mtime is not the run time
    except ValueError:
        pass
    return et


# ── steps ────────────────────────────────────────────────────────────────────

def step1_shadow(bet, box, log):
    # one batch = one run (rows written within the same 10-minute bucket)
    batches = bet.execute(f"""
        SELECT game_date, substr(created_at, 1, 15) run, date(created_at, '-5 hours') et, COUNT(*)
        FROM shadow_picks
        WHERE game_date >= '{SEASON_START}' AND created_at IS NOT NULL
        GROUP BY 1, 2 HAVING game_date != et ORDER BY 1, 2""").fetchall()
    moved = 0
    for stored, run, target, n in batches:
        players = [p for (p,) in bet.execute(
            "SELECT DISTINCT player FROM shadow_picks WHERE game_date=? AND substr(created_at,1,15)=? "
            "AND stat_category IN ('POINTS','REBOUNDS','ASSISTS')", (stored, run))]
        ok, why = decide(box, players, stored, target)
        log(f"[1] shadow {stored} run {run}0Z -> {target}: {n} rows — {'MOVED' if ok else 'kept'} ({why})")
        if ok:
            bet.execute("""UPDATE shadow_picks SET game_date=?, actual_value=NULL, actual_result=NULL,
                           resolved_at=NULL WHERE game_date=? AND substr(created_at,1,15)=?""",
                        (target, stored, run))
            moved += n
    # re-grade every prop row since March 2026 from box scores
    now = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    g = 0
    for rid, d, p, c, di, line in bet.execute(f"""SELECT id, game_date, player, stat_category, direction, line
            FROM shadow_picks WHERE game_date >= '{SEASON_START}'
            AND stat_category IN ('POINTS','REBOUNDS','ASSISTS')""").fetchall():
        a = box.actual(p, c, d)
        if a is None:
            bet.execute("UPDATE shadow_picks SET actual_value=NULL, actual_result=NULL, resolved_at=NULL WHERE id=?", (rid,))
            continue
        bet.execute("UPDATE shadow_picks SET actual_value=?, actual_result=?, resolved_at=? WHERE id=?",
                    (a, grade(di, a, line), now, rid))
        g += 1
    bet.execute(f"UPDATE shadow_picks SET superseded = 0 WHERE game_date >= '{SEASON_START}'")
    bet.execute(f"""UPDATE shadow_picks SET superseded = 1
        WHERE game_date >= '{SEASON_START}' AND stat_category IN ('POINTS','REBOUNDS','ASSISTS')
          AND (julianday((SELECT MAX(s2.created_at) FROM shadow_picks s2 WHERE s2.game_date=shadow_picks.game_date
                AND s2.player=shadow_picks.player AND s2.stat_category=shadow_picks.stat_category))
               - julianday(created_at)) * 86400 >= 600""")
    sup = bet.execute(f"SELECT COUNT(*) FROM shadow_picks WHERE superseded=1 AND game_date>='{SEASON_START}'").fetchone()[0]
    log(f"[1] shadow: moved {moved} rows; re-graded {g} prop rows since {SEASON_START}; superseded={sup}")


def step2_daily_picks(bet, box, model_dir, log):
    bet.execute("""CREATE TABLE IF NOT EXISTS pick_results_quarantine_20261106 AS
                   SELECT pr.*, '' AS reason FROM pick_results pr WHERE 0""")
    groups = bet.execute("""SELECT date, source_file, COUNT(*) FROM daily_picks
                            WHERE source_mode = 'live' AND date >= ? GROUP BY 1, 2 ORDER BY 1""",
                         (SEASON_START,)).fetchall()
    moved_ids = []
    for stored, sf, n in groups:
        props = bet.execute("""SELECT dp.player_or_team, dp.stat_category, pr.actual_value FROM daily_picks dp
                               LEFT JOIN pick_results pr ON pr.pick_id = dp.id
                               WHERE dp.date=? AND dp.source_file IS ? AND dp.stat_category IN ('POINTS','REBOUNDS','ASSISTS')""",
                            (stored, sf)).fetchall()
        players = [p for p, _, _ in props]
        cov_s = box.coverage(players, stored) if box.covered(stored) else None
        if cov_s is not None and cov_s >= 0.6:
            continue                                   # stored date is consistent
        target = file_run_et(model_dir, sf)
        basis = 'run time'
        if target is None or target == stored:
            # no usable run time: try +-1 day where the stored actuals already match the box scores
            best = None
            for off in (-1, 1):
                t = shift(stored, off)
                graded = [(p, c, a) for p, c, a in props if a is not None]
                match = sum(1 for p, c, a in graded if box.actual(p, c, t) == a)
                if graded and match / len(graded) >= 0.8:
                    best = t
            target, basis = best, 'stored actuals match box scores'
        if not target:
            log(f"[2] daily_picks {stored} ({sf}): {n} rows inconsistent, no target found — kept")
            continue
        ok, why = decide(box, players, stored, target)
        log(f"[2] daily_picks {stored} -> {target} ({sf}, {n} rows, {basis}): {'MOVED' if ok else 'kept'} ({why})")
        if ok:
            ids = [i for (i,) in bet.execute("SELECT id FROM daily_picks WHERE date=? AND source_file IS ? AND source_mode='live'",
                                             (stored, sf))]
            bet.executemany("UPDATE daily_picks SET date=? WHERE id=?", [(target, i) for i in ids])
            moved_ids += ids
    # re-grade pick_results of moved picks
    regraded = quarantined = 0
    now = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for pid in moved_ids:
        d, p, c, di, line = bet.execute("SELECT date, player_or_team, stat_category, direction, line FROM daily_picks WHERE id=?",
                                        (pid,)).fetchone()
        a = box.actual(p, c, d) if c in PROPS else None
        w = grade(di, a, line)
        if w is None:
            bet.execute("""INSERT INTO pick_results_quarantine_20261106 SELECT pr.*,
                           'cannot re-grade after re-date (no box score / push / team bet)' FROM pick_results pr
                           WHERE pr.pick_id=?""", (pid,))
            quarantined += bet.execute("DELETE FROM pick_results WHERE pick_id=?", (pid,)).rowcount
            continue
        diff = a - line
        cur = bet.execute("SELECT id FROM pick_results WHERE pick_id=?", (pid,)).fetchone()
        if cur:
            bet.execute("UPDATE pick_results SET actual_value=?, is_win=?, diff_val=?, diff_pct=?, resolved_at=? WHERE pick_id=?",
                        (a, w, diff, diff / line * 100 if line else None, now, pid))
        else:
            bet.execute("INSERT INTO pick_results (pick_id, actual_value, is_win, diff_val, diff_pct, resolved_at) VALUES (?,?,?,?,?,?)",
                        (pid, a, w, diff, diff / line * 100 if line else None, now))
        regraded += 1
    log(f"[2] daily_picks: moved {len(moved_ids)} picks; re-graded {regraded}; quarantined {quarantined} results")


def step3_projection_outcomes(bet, box, model_dir, table, modes, log):
    groups = bet.execute(f"""SELECT game_date, source_file, COUNT(*) FROM {table}
                             WHERE source_mode IN ({','.join('?' * len(modes))}) AND game_date >= ?
                             GROUP BY 1, 2 ORDER BY 1""", (*modes, SEASON_START)).fetchall()
    total = 0
    for stored, sf, n in groups:
        rows = bet.execute(f"""SELECT id, player FROM {table} WHERE game_date=? AND source_file IS ?
                               AND source_mode IN ({','.join('?' * len(modes))}) AND category IN ('POINTS','REBOUNDS','ASSISTS')
                               ORDER BY id""", (stored, sf, *modes)).fetchall()
        players = [p for _, p in rows]
        prev = shift(stored, -1)
        only_prev = sorted(i for i, p in rows if box.played(p, prev) and not box.played(p, stored))
        only_same = sorted(i for i, p in rows if box.played(p, stored) and not box.played(p, prev))
        if len(only_prev) >= 10 and len(only_same) >= 10 and not (stored in KNOWN_GAPS and stored not in box.dates):
            # two runs mixed in one batch: split by insertion order
            ids = None
            if sum(i < only_same[0] for i in only_prev) / len(only_prev) >= 0.85:
                ids = [i for (i,) in bet.execute(f"SELECT id FROM {table} WHERE game_date=? AND source_file IS ? AND id < ?",
                                                 (stored, sf, only_same[0]))]
                where = f'ids < {only_same[0]}'
            elif sum(i > only_same[-1] for i in only_prev) / len(only_prev) >= 0.85:
                ids = [i for (i,) in bet.execute(f"SELECT id FROM {table} WHERE game_date=? AND source_file IS ? AND id > ?",
                                                 (stored, sf, only_same[-1]))]
                where = f'ids > {only_same[-1]}'
            if ids:
                bet.executemany(f"UPDATE {table} SET game_date=? WHERE id=?", [(prev, i) for i in ids])
                total += len(ids)
                log(f"[3] {table} {stored} ({sf}): mixed batch — run for {prev} split off ({where}): moved {len(ids)} rows")
            else:
                log(f"[3] {table} {stored} ({sf}): mixed batch, runs not separable by insertion order — kept")
            continue
        if box.covered(stored) and box.coverage(players, stored) >= 0.6:
            continue
        target = file_run_et(model_dir, sf)
        if not target or target == stored:
            log(f"[3] {table} {stored} ({sf}): {n} rows inconsistent, run time unknown — kept")
            continue
        ok, why = decide(box, players, stored, target)
        log(f"[3] {table} {stored} -> {target} ({sf}, {n} rows): {'MOVED' if ok else 'kept'} ({why})")
        if ok:
            cur = bet.execute(f"UPDATE {table} SET game_date=? WHERE game_date=? AND source_file IS ? "
                              f"AND source_mode IN ({','.join('?' * len(modes))})", (target, stored, sf, *modes))
            total += cur.rowcount
    log(f"[3] {table}: moved {total} rows")


def step5_source_files(bet, renames, log):
    n = 0
    for old, new in renames.items():
        for table, col in (('daily_picks', 'source_file'), ('projection_outcomes', 'source_file'),
                           ('projection_outcomes_raw', 'source_file')):
            # live rows only — backtest rows can reference a backtest file with the same name
            # ...and only rows that now sit on the file's new date (a batch can hold two runs)
            new_date = new[len('predicts_'):len('predicts_') + 10]
            n += bet.execute(f"UPDATE {table} SET {col} = REPLACE({col}, ?, ?) WHERE {col} LIKE ? "
                             f"AND source_mode IN ('live', 'live_pending') AND {'date' if table == 'daily_picks' else 'game_date'} = ?",
                             (old, new, f'%{old}', new_date)).rowcount
    log(f"[5] source_file references updated for {len(renames)} renamed prediction files: {n} rows")


def step6_formats(bet, log):
    n = bet.execute("""UPDATE projection_outcomes
                       SET game_date = substr(game_date,1,4)||'-'||substr(game_date,5,2)||'-'||substr(game_date,7,2)
                       WHERE length(game_date) = 8 AND game_date GLOB '[0-9]*'""").rowcount
    log(f"[6] projection_outcomes: {n} YYYYMMDD dates normalised to YYYY-MM-DD")


def step7_game_results(bet, nba, log):
    dates = [d for (d,) in bet.execute("SELECT DISTINCT date FROM game_results ORDER BY 1")]
    teams = {t: a for t, a in nba.execute("SELECT team_id, abbreviation FROM teams")}
    before = bet.execute("SELECT COUNT(*) FROM game_results").fetchone()[0]
    bet.execute("CREATE TABLE IF NOT EXISTS game_results_backup_20261106 AS SELECT id, date, home_team, away_team, "
                "home_score, away_score FROM game_results")
    bet.execute("DELETE FROM game_results")
    rows = 0
    for d in dates:
        for gid, matchup in nba.execute("SELECT game_id, matchup FROM games WHERE substr(game_date,1,10)=?", (d,)):
            pts = defaultdict(int)
            seen = set()
            for tid, pid, p in nba.execute("SELECT team_id, player_id, pts FROM box_scores WHERE game_id=?", (gid,)):
                if pid in seen:
                    continue
                seen.add(pid)
                pts[teams.get(tid)] += p or 0
            if ' vs. ' in matchup:
                home, away = [x.strip() for x in matchup.split(' vs. ')]
            elif ' @ ' in matchup:
                away, home = [x.strip() for x in matchup.split(' @ ')]
            else:
                continue
            if home not in pts or away not in pts:
                continue
            bet.execute("INSERT OR IGNORE INTO game_results (date, home_team, away_team, home_score, away_score) "
                        "VALUES (?,?,?,?,?)", (d, home, away, pts[home], pts[away]))
            rows += 1
    log(f"[7] game_results rebuilt from games/box scores for {len(dates)} dates: {before} -> {rows} rows "
        f"(old rows kept in game_results_backup_20261106)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--renames', default=None, help='file with "old_name new_name" lines (prediction file renames)')
    ap.add_argument('--report', default=None)
    a = ap.parse_args(argv)
    lines = []

    def log(m):
        print(m)
        lines.append(m)

    log(f"fix_dates_20261106 — {'APPLY' if a.apply else 'DRY RUN'} — {dt.datetime.now():%Y-%m-%d %H:%M}")
    bet_path = os.path.join(a.model_dir, 'database', 'bet_tracker.db')
    nba = sqlite3.connect(f"file:{os.path.join(a.model_dir, 'database', 'nba_data.db')}?mode=ro", uri=True)
    bet = sqlite3.connect(bet_path) if a.apply else sqlite3.connect(':memory:')
    if not a.apply:
        src = sqlite3.connect(bet_path)
        src.backup(bet)
        src.close()
    box = Box(nba)
    log(f"box scores cover up to {box.max_date}; known gaps: {sorted(KNOWN_GAPS)}")
    renames = {}
    if a.renames and os.path.exists(a.renames):
        for line in open(a.renames, encoding='utf-8'):
            parts = line.split()
            if len(parts) == 2:
                renames[parts[0]] = parts[1]
    step1_shadow(bet, box, log)
    step2_daily_picks(bet, box, a.model_dir, log)
    step3_projection_outcomes(bet, box, a.model_dir, 'projection_outcomes', ('live', 'live_pending'), log)
    step3_projection_outcomes(bet, box, a.model_dir, 'projection_outcomes_raw', ('live',), log)
    step5_source_files(bet, renames, log)
    step6_formats(bet, log)
    step7_game_results(bet, nba, log)
    bet.commit()
    bet.close()
    if a.report:
        with open(a.report, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
