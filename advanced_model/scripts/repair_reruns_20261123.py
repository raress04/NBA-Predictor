"""
scripts/repair_reruns_20261123.py
=================================
W1 (Plans/W1_RERUN_AUDIT.md): flag the rows that earlier runs of a slate left behind,
using the same rule the code now applies at write time: the latest run is the record,
game by game. Rows are flagged superseded = 1; nothing is deleted.

DRY RUN BY DEFAULT: works on an in-memory copy of bet_tracker.db and prints what it would
change. With --apply it first copies the database to database/archive/ and then writes.

How runs are told apart
  shadow_picks, ml_shadow_picks, parlays: by insert time. Within one slate, a gap of more
    than RUN_GAP_S seconds between consecutive rows (by id) starts a new run.
  projection_outcomes has no timestamp: for a (game_date, player, category) the highest id
    wins. Old TOTAL rows there have no game id, so they are left alone (reported).
Keys: (player, category); old TOTAL rows in shadow_picks have no matchup, so the line
stands in for the game.

Run from advanced_model/:
  python3 -m scripts.repair_reruns_20261123            # dry run
  python3 -m scripts.repair_reruns_20261123 --apply    # backup, then write
"""
import argparse
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from database.bet_tracker_schema import ensure_rerun_columns  # noqa: E402

DB = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')
SEASON_START = '2026-10-20'
RUN_GAP_S = 5


def _ts(s):
    return datetime.fromisoformat(str(s).replace('T', ' ')[:19])


def _runs(rows):
    """rows: [(id, created_at, ...)] of one slate sorted by id -> run number per id."""
    run, last, out = 0, None, {}
    for r in rows:
        t = _ts(r[1]) if r[1] else None
        if last is not None and t is not None and (t - last).total_seconds() > RUN_GAP_S:
            run += 1
        if t is not None:
            last = t
        out[r[0]] = run
    return out


def _flag_by_runs(conn, table, cat_col, where, key_fn, log, label):
    """Within each slate, flag a row when a LATER run has the same key."""
    rows = conn.execute(f"""SELECT id, created_at, game_date, player, {cat_col}, line
                            FROM {table} WHERE COALESCE(superseded,0)=0 {where}
                            ORDER BY game_date, id""").fetchall()
    by_date = defaultdict(list)
    for r in rows:
        by_date[r[2]].append(r)
    flag, per_date = [], {}
    for d, rs in by_date.items():
        run_of = _runs(rs)
        if max(run_of.values()) == 0:
            continue
        latest_run_of_key = {}
        for r in rs:
            k = key_fn(r)
            latest_run_of_key[k] = max(latest_run_of_key.get(k, -1), run_of[r[0]])
        ids = [r[0] for r in rs if run_of[r[0]] < latest_run_of_key[key_fn(r)]]
        if ids:
            per_date[d] = (len(ids), max(run_of.values()) + 1)
            flag += ids
    for d, (n, runs) in sorted(per_date.items()):
        log(f"  {label} {d}: {runs} runs, {n} rows from earlier runs")
    for i in range(0, len(flag), 500):
        chunk = flag[i:i + 500]
        conn.execute(f"UPDATE {table} SET superseded = 1 WHERE id IN ({','.join('?' * len(chunk))})", chunk)
    log(f"[{label}] flagged {len(flag)} rows")
    return len(flag)


def repair(conn, log):
    added = ensure_rerun_columns(conn)
    log(f"[schema] added columns: {', '.join(added) or 'none'}")

    def shadow_key(r):
        cat = str(r[4] or '').upper()
        return (r[3], cat, r[5] if cat == 'TOTAL' else None)   # line stands in for an old TOTAL's game

    n_sp = _flag_by_runs(conn, 'shadow_picks', 'stat_category', f"AND game_date >= '{SEASON_START}'",
                         shadow_key, log, 'shadow_picks')
    n_ml = _flag_by_runs(conn, 'ml_shadow_picks', 'stat_category', '',
                         lambda r: (r[3], str(r[4]).upper()), log, 'ml_shadow_picks')

    # projection_outcomes: highest id wins per (game_date, player, category); TOTAL left alone
    rows = conn.execute("""SELECT id FROM projection_outcomes p
        WHERE source_mode = 'live_pending' AND UPPER(category) <> 'TOTAL' AND COALESCE(superseded,0) = 0
          AND id < (SELECT MAX(id) FROM projection_outcomes q
                    WHERE q.source_mode = 'live_pending' AND q.game_date = p.game_date
                      AND q.player = p.player AND q.category = p.category)""").fetchall()
    ids = [r[0] for r in rows]
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        conn.execute(f"UPDATE projection_outcomes SET superseded = 1 WHERE id IN ({','.join('?' * len(chunk))})", chunk)
    per_date = conn.execute(f"""SELECT game_date, COUNT(*) FROM projection_outcomes
        WHERE id IN ({','.join('?' * len(ids)) or 'NULL'}) GROUP BY 1 ORDER BY 1""", ids).fetchall() if ids else []
    for d, n in per_date:
        log(f"  projection_outcomes {d}: {n} older rows")
    tot = conn.execute("""SELECT COUNT(*) FROM (SELECT game_date FROM projection_outcomes
        WHERE source_mode='live_pending' AND UPPER(category)='TOTAL' GROUP BY game_date, bookmaker_line
        HAVING COUNT(*) > 1)""").fetchone()[0]
    log(f"[projection_outcomes] flagged {len(ids)} rows; {tot} old TOTAL groups left alone (no game id)")

    # parlays: every parlay of an earlier run of the slate
    rows = conn.execute("""SELECT id, created_at, game_date FROM parlays
                           WHERE COALESCE(superseded,0)=0 ORDER BY game_date, id""").fetchall()
    by_date = defaultdict(list)
    for r in rows:
        by_date[r[2]].append(r)
    pids = []
    for d, rs in sorted(by_date.items()):
        run_of = _runs(rs)
        last = max(run_of.values())
        old = [r[0] for r in rs if run_of[r[0]] < last]
        if old:
            res = conn.execute(f"SELECT COUNT(*) FROM parlays WHERE id IN ({','.join('?' * len(old))}) "
                               "AND result IS NOT NULL", old).fetchone()[0]
            log(f"  parlays {d}: {last + 1} runs, {len(old)} parlays from earlier runs ({res} already settled)")
            pids += old
    if pids:
        conn.execute(f"UPDATE parlays SET superseded = 1 WHERE id IN ({','.join('?' * len(pids))})", pids)
    log(f"[parlays] flagged {len(pids)} rows")
    return {'shadow_picks': n_sp, 'ml_shadow_picks': n_ml, 'projection_outcomes': len(ids), 'parlays': len(pids)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true', help='back up, then write (default: dry run)')
    ap.add_argument('--db', default=DB)
    ap.add_argument('--log', default=os.path.join(MODEL_DIR, 'logs', 'november', 'repair_reruns_2026-11-23.txt'))
    args = ap.parse_args()

    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    log(f"repair_reruns_20261123 — {'APPLY' if args.apply else 'DRY RUN'} — {datetime.now():%Y-%m-%d %H:%M:%S}")
    src = sqlite3.connect(args.db)
    if args.apply:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = os.path.join(os.path.dirname(args.db), 'archive', f'bet_tracker_pre_rerun_repair_{stamp}.db')
        os.makedirs(os.path.dirname(backup), exist_ok=True)
        dst = sqlite3.connect(backup)
        src.backup(dst)
        ok = dst.execute("PRAGMA integrity_check").fetchone()[0]
        dst.close()
        log(f"[backup] {backup} (integrity_check = {ok})")
        if ok != 'ok':
            raise SystemExit("backup failed its integrity check; nothing written")
        conn = src
    else:
        conn = sqlite3.connect(':memory:')
        src.backup(conn)
        src.close()
    counts = repair(conn, log)
    if args.apply:
        conn.commit()
        log("[done] written")
    else:
        log("[dry run] nothing written; re-run with --apply to write")
    conn.close()
    log(f"totals: {counts}")
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    suffix = '' if args.apply else '_dryrun'
    path = args.log.replace('.txt', f'{suffix}.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"log: {path}")


if __name__ == '__main__':
    main()
