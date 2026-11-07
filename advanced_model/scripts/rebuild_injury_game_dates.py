"""
T3b — re-key historical_injuries by the game each row is about
(Plans/T2_T3_IMPLEMENTATION_AUDIT.md, T3b).

Until 2026-11-07 the scraper stored every row under the day the report was PUBLISHED.
A report published on a day without games (e.g. 2026-11-03) is entirely about the next
day's games, and game-day reports sometimes list next-day games too.

This script re-fetches the 5 PM report for every date stored with the old keying, then:
  1. moves those old rows to historical_injuries_superseded (quarantine, nothing dropped)
  2. stores the corrected rows: game_date = the report's 'Game Date', report_date kept,
     one snapshot per game date (latest report wins; older-report rows also quarantined).
Dates whose report cannot be fetched keep their old rows untouched.

Dry run (default): the whole procedure runs on an in-memory copy of the DB; the real DB is
opened read-only. --apply: backup to database/archive/, then the same procedure, one transaction.
Fetched reports are cached to CSV so --apply stores exactly what the dry run showed.

Run from advanced_model/:
  python3 -m scripts.rebuild_injury_game_dates --batch-size 1 --delay 2      # dry run
  python3 -m scripts.rebuild_injury_game_dates --apply                       # uses the cache
"""
import argparse
import asyncio
import csv
import os
import sqlite3
import sys
from datetime import datetime

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODEL_DIR)
from etl import scrape_historical_injuries as s  # noqa: E402

NBA_DB = os.path.join(MODEL_DIR, 'database', 'nba_data.db')
FIELDS = ['game_date', 'report_date', 'player', 'team', 'status', 'reason']
NOTE = 'T3b rekey: stored under report date'


def default_cache():
    now = datetime.now()
    return os.path.join(MODEL_DIR, 'logs', now.strftime('%B').lower(), 'injuries',
                        f"rekey_fetch_{now:%Y-%m-%d}.csv")


def legacy_dates(conn) -> list:
    """Dates stored with the old keying (= report dates)."""
    where = "WHERE report_date IS NULL" if s._has_report_col(conn) else ""
    return [r[0] for r in conn.execute(f"SELECT DISTINCT game_date FROM historical_injuries {where} ORDER BY 1")]


def load_or_fetch(dates: list, cache: str, batch_size: int, delay: float) -> list:
    if os.path.exists(cache):
        with open(cache, encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        print(f"[cache] {len(rows):,} rows from {cache}")
        return rows
    print(f"[fetch] {len(dates)} report dates (batch {batch_size}, delay {delay}s) ...", flush=True)
    rows = asyncio.run(s.fetch_dates([datetime.strptime(d, '%Y-%m-%d') for d in dates], batch_size, delay))
    rows = [r for r in rows if r['player'] and r['team']]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k, '') for k in FIELDS} for r in rows)
    print(f"[cache] wrote {len(rows):,} rows to {cache}")
    return rows


def rekey(conn, rows: list, candidates: list) -> dict:
    """The whole re-key on `conn` (caller commits). Returns counts."""
    s.ensure_schema(conn)
    fetched = sorted({r['report_date'] for r in rows} & set(candidates))
    failed = sorted(set(candidates) - set(fetched))
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _rekey_dates (d TEXT PRIMARY KEY)")
    conn.execute("DELETE FROM _rekey_dates")
    conn.executemany("INSERT INTO _rekey_dates VALUES (?)", [(d,) for d in fetched])
    now = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    cond = "report_date IS NULL AND game_date IN (SELECT d FROM _rekey_dates)"
    conn.execute(f"""INSERT INTO historical_injuries_superseded
                     SELECT game_date, player, team, status, reason, report_date, ?, ?
                     FROM historical_injuries WHERE {cond}""", (now, NOTE))
    moved = conn.execute(f"DELETE FROM historical_injuries WHERE {cond}").rowcount
    plan = s.plan_store(conn, rows)
    inserted = s.apply_store(conn, plan, note='T3b rekey: older report for this game date')
    return {'fetched_dates': len(fetched), 'failed_dates': failed, 'moved': moved, 'inserted': inserted,
            'superseded_older': len(plan['supersede']),
            'ignored_older': plan['ignored_older'] + plan['older_in_fetch']}


def snapshot(conn) -> dict:
    game_days = {r[0] for r in conn.execute(
        "SELECT DISTINCT substr(game_date,1,10) FROM games UNION SELECT DISTINCT substr(game_date,1,10) FROM box_scores")}
    per_date = dict(conn.execute("SELECT game_date, COUNT(*) FROM historical_injuries GROUP BY 1").fetchall())
    in_season = {d: n for d, n in per_date.items() if d >= '2025-10-21'}
    off = {d: n for d, n in in_season.items() if d not in game_days}
    shifted = 0
    if s._has_report_col(conn):
        shifted = conn.execute("SELECT COUNT(*) FROM historical_injuries "
                               "WHERE report_date IS NOT NULL AND report_date <> game_date").fetchone()[0]
    return {'rows': sum(per_date.values()), 'dates': len(per_date), 'per_date': per_date,
            'non_game_days': off, 'shifted': shifted}


def report(before: dict, after: dict, stats: dict):
    print(f"\nRows            : {before['rows']:,} -> {after['rows']:,}")
    print(f"Dates           : {before['dates']} -> {after['dates']}")
    print(f"Report dates re-fetched: {stats['fetched_dates']} | failed (old rows kept): {len(stats['failed_dates'])}"
          + (f" {stats['failed_dates'][:20]}" if stats['failed_dates'] else ''))
    print(f"Old rows -> quarantine : {stats['moved']:,}")
    print(f"Corrected rows inserted: {stats['inserted']:,}")
    print(f"Older-report rows ignored/quarantined: {stats['ignored_older']:,} / {stats['superseded_older']:,}")
    print(f"Rows whose game date differs from the report date: {after['shifted']:,}")
    nb, na = before['non_game_days'], after['non_game_days']
    print(f"Injury dates with NO games (2025-10-21+): {len(nb)} dates / {sum(nb.values()):,} rows"
          f" -> {len(na)} dates / {sum(na.values()):,} rows")
    for d in sorted(na)[:20]:
        print(f"    still on a no-game day: {d} ({na[d]} rows)")
    changed = sorted(d for d in set(before['per_date']) | set(after['per_date'])
                     if before['per_date'].get(d, 0) != after['per_date'].get(d, 0))
    print(f"\nDates whose row count changes: {len(changed)}")
    for d in changed[:60]:
        print(f"    {d}: {before['per_date'].get(d, 0):4} -> {after['per_date'].get(d, 0):4}")
    if len(changed) > 60:
        print(f"    ... {len(changed) - 60} more")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--cache', default=default_cache())
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--delay', type=float, default=2.0)
    ap.add_argument('--db', default=NBA_DB)
    args = ap.parse_args()

    ro = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    candidates = legacy_dates(ro)
    print(f"[rekey] {len(candidates)} dates stored with the old keying "
          f"({candidates[0] if candidates else '-'} .. {candidates[-1] if candidates else '-'})")
    if not candidates:
        print("[rekey] nothing to do.")
        return
    rows = load_or_fetch(candidates, args.cache, args.batch_size, args.delay)

    if not args.apply:
        mem = sqlite3.connect(':memory:')
        ro.backup(mem)
        ro.close()
        before = snapshot(mem)
        stats = rekey(mem, rows, candidates)
        report(before, snapshot(mem), stats)
        print("\n[dry-run] ran on an in-memory copy. Nothing written.")
        return
    ro.close()

    arch = os.path.join(os.path.dirname(args.db), 'archive')
    os.makedirs(arch, exist_ok=True)
    backup = os.path.join(arch, f"nba_data_pre_injury_rekey_{datetime.now():%Y%m%d_%H%M%S}.db")
    conn = sqlite3.connect(args.db)
    b = sqlite3.connect(backup)
    conn.backup(b)
    b.close()
    print(f"[backup] {backup}")
    before = snapshot(conn)
    try:
        conn.execute("BEGIN")
        stats = rekey(conn, rows, candidates)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    report(before, snapshot(conn), stats)
    print(f"\nintegrity_check: {conn.execute('PRAGMA integrity_check').fetchone()[0]}")
    conn.close()


if __name__ == '__main__':
    main()
