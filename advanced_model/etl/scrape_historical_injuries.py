"""
advanced_model/etl/scrape_historical_injuries.py
=================================================
Scrapes NBA official injury reports for the 2025-26 season
using the nbainjuries package and stores them in nba_data.db.

Originally run once (2025-26). Since 2026-11-06 (T3) run_pipeline.sh calls it every
night with --previous-slate, so the official report is stored every day.
T3b (2026-11-07): rows are keyed by the report's own 'Game Date' (not the publish date),
with report_date stored. One snapshot per game date: a newer report moves the older rows
to historical_injuries_superseded (quarantine); an older report is ignored; re-running
the same report inserts nothing. Existing rows: scripts/rebuild_injury_game_dates.py.

Usage:
    python3 -m etl.scrape_historical_injuries
    python3 -m etl.scrape_historical_injuries --start 2026-01-01 --end 2026-03-24
    python3 -m etl.scrape_historical_injuries --start 2026-04-07 --end 2026-11-05 --dry-run
    python3 -m etl.scrape_historical_injuries --previous-slate          # nightly
"""
import asyncio
import sqlite3
import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd
import aiohttp
from nbainjuries import injury_asy

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NBA_DB    = os.path.join(MODEL_DIR, 'database', 'nba_data.db')

DEFAULT_START = '2026-01-01'
DEFAULT_END   = '2026-03-24'
REPORT_HOUR   = 17   # 5 PM ET — final pre-game report
REPORT_MIN    = 0

STATUS_MAP = {
    'Out':          'Out',
    'Questionable': 'Questionable',
    'Doubtful':     'Doubtful',
    'Available':    'Available',
    'GTD':          'Questionable',
}


def date_range(start: str, end: str):
    cur = datetime.strptime(start, '%Y-%m-%d')
    end_dt = datetime.strptime(end, '%Y-%m-%d')
    while cur <= end_dt:
        yield cur
        cur += timedelta(days=1)


def parse_game_date(value, fallback: str) -> str:
    """Report 'Game Date' ('MM/DD/YYYY') -> 'YYYY-MM-DD'; the report date when missing."""
    try:
        if value is None or pd.isna(value):
            return fallback
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.strftime('%Y-%m-%d')
        s = str(value).strip()
        for fmt in ('%m/%d/%Y', '%Y-%m-%d'):
            try:
                return datetime.strptime(s, fmt).strftime('%Y-%m-%d')
            except ValueError:
                pass
    except (TypeError, ValueError):
        pass
    return fallback


# ── T3b storage: one snapshot per game date (latest report wins) ──────────────
SUPERSEDED_DDL = """CREATE TABLE IF NOT EXISTS historical_injuries_superseded (
        game_date TEXT, player TEXT, team TEXT, status TEXT, reason TEXT,
        report_date TEXT, superseded_at TEXT, note TEXT)"""


def _report_of(row: dict) -> str:
    return row.get('report_date') or row['game_date']


def _has_report_col(conn) -> bool:
    return any(r[1] == 'report_date' for r in conn.execute("PRAGMA table_info(historical_injuries)"))


def ensure_schema(conn):
    """Adds report_date (NULL on legacy rows = stored under the report date) and the quarantine table."""
    if not _has_report_col(conn):
        conn.execute("ALTER TABLE historical_injuries ADD COLUMN report_date TEXT")
    conn.execute(SUPERSEDED_DDL)


def latest_snapshot(rows: list):
    """Per game date keep only rows from the latest report covering it. Returns (kept, n_older)."""
    latest = {}
    for r in rows:
        g, rep = r['game_date'], _report_of(r)
        if rep > latest.get(g, ''):
            latest[g] = rep
    kept = [r for r in rows if _report_of(r) == latest[r['game_date']]]
    return kept, len(rows) - len(kept)


def plan_store(conn, rows: list) -> dict:
    """
    Snapshot rule per game date G (read-only):
      stored report newer than this one -> ignore the new rows;
      stored rows from an older report   -> quarantine them, insert the new snapshot;
      same report                        -> insert rows not yet stored.
    Legacy rows (report_date NULL) count as published on their game_date.
    """
    kept, n_older_fetch = latest_snapshot(rows)
    rep_expr = "COALESCE(report_date, game_date)" if _has_report_col(conn) else "game_date"
    by_g = {}
    for r in kept:
        by_g.setdefault(r['game_date'], []).append(r)
    plan = {'insert': [], 'supersede': [], 'ignored_older': 0, 'older_in_fetch': n_older_fetch}
    for g, new in by_g.items():
        rep = _report_of(new[0])
        stored = conn.execute(f"SELECT player, team, {rep_expr} FROM historical_injuries WHERE game_date=?",
                              (g,)).fetchall()
        if any(s[2] > rep for s in stored):
            plan['ignored_older'] += len(new)
            continue
        old = [(g, s[0], s[1]) for s in stored if s[2] < rep]
        plan['supersede'].extend(old)
        present = {(s[0], s[1]) for s in stored if s[2] == rep}
        seen = set()
        for r in new:
            k = (r['player'], r['team'])
            if k not in present and k not in seen:
                seen.add(k)
                plan['insert'].append(r)
    return plan


def apply_store(conn, plan: dict, note: str = 'newer report'):
    """Moves superseded rows to historical_injuries_superseded, then inserts. Caller commits."""
    ensure_schema(conn)
    now = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    for g, p, t in plan['supersede']:
        conn.execute("""INSERT INTO historical_injuries_superseded
                        SELECT game_date, player, team, status, reason, report_date, ?, ?
                        FROM historical_injuries WHERE game_date=? AND player=? AND team=?""",
                     (now, note, g, p, t))
        conn.execute("DELETE FROM historical_injuries WHERE game_date=? AND player=? AND team=?", (g, p, t))
    n = 0
    for r in plan['insert']:
        n += conn.execute("INSERT OR IGNORE INTO historical_injuries "
                          "(game_date, player, team, status, reason, report_date) VALUES (?,?,?,?,?,?)",
                          (r['game_date'], r['player'], r['team'], r['status'], r['reason'],
                           _report_of(r))).rowcount
    return n


async def fetch_one_day(session, dt: datetime):
    """Fetch injury report for a single date. Returns list of row dicts."""
    ts = datetime(dt.year, dt.month, dt.day, REPORT_HOUR, REPORT_MIN)
    try:
        valid = await injury_asy.check_reportvalid(ts, session=session)
        if not valid:
            return []
        df = await injury_asy.get_reportdata(ts, session=session, return_df=True)
        if df is None or len(df) == 0:
            return []

        rows = []
        report_date = dt.strftime('%Y-%m-%d')
        for _, rec in df.iterrows():
            status_raw = str(rec.get('Current Status', 'Available')).strip()
            # Convert 'Last, First' → 'First Last'
            raw_name = str(rec.get('Player Name', '')).strip()
            if ',' in raw_name:
                parts = raw_name.split(',', 1)
                player_name = f"{parts[1].strip()} {parts[0].strip()}"
            else:
                player_name = raw_name
            rows.append({
                # T3b: the game the row is about, not the day the report was published
                'game_date':   parse_game_date(rec.get('Game Date'), report_date),
                'report_date': report_date,
                'player':    player_name,
                'team':      str(rec.get('Team', '')).strip(),
                'status':    STATUS_MAP.get(status_raw, 'Available'),
                'reason':    str(rec.get('Reason', '')).strip(),
            })
        return rows
    except Exception as e:
        print(f"    [!] {dt.strftime('%Y-%m-%d')}: {e}")
        return []


async def fetch_all(start: str, end: str, batch_size: int = 10, delay: float = 0.5):
    """Fetch injury data for all dates in range concurrently (batched)."""
    return await fetch_dates(list(date_range(start, end)), batch_size, delay)


async def fetch_dates(dates: list, batch_size: int = 10, delay: float = 0.5):
    """Fetch the 5 PM report for each datetime in `dates` (batched)."""
    all_rows = []
    BATCH_SIZE = batch_size  # avoid hammering the server (the NBA site throttles bursts)

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(dates), BATCH_SIZE):
            batch = dates[i:i + BATCH_SIZE]
            tasks = [fetch_one_day(session, dt) for dt in batch]
            results = await asyncio.gather(*tasks)
            for rows in results:
                all_rows.extend(rows)
            print(f"    Fetched {min(i + BATCH_SIZE, len(dates))}/{len(dates)} days...")
            await asyncio.sleep(delay)  # polite delay between batches

    return all_rows


def summarize(rows: list) -> str:
    """Counts per date and status (for dry runs and logs)."""
    by = {}
    for r in rows:
        by.setdefault(r['game_date'], {}).setdefault(r['status'], 0)
        by[r['game_date']][r['status']] += 1
    lines = [f"    {d}: {sum(s.values()):4} rows  " +
             ' '.join(f"{k}={v}" for k, v in sorted(s.items())) for d, s in sorted(by.items())]
    return '\n'.join(lines)


def run(start_date: str = DEFAULT_START, end_date: str = DEFAULT_END, dry_run: bool = False,
        allow_empty: bool = False, db_path: str = None, batch_size: int = 10, delay: float = 0.5):
    """
    dry_run     : fetch and print counts per date/status; write nothing (T3, 2026-11-06).
    allow_empty : an empty result is normal (e.g. no games that day) -> exit quietly.
    """
    print(f"[+] Fetching injury data for {start_date} → {end_date}...")
    print(f"    Using {REPORT_HOUR}:00 ET reports")

    all_rows = asyncio.run(fetch_all(start_date, end_date, batch_size, delay))
    print(f"[+] Total rows fetched: {len(all_rows)}")

    if not all_rows:
        if allow_empty:
            print("[i] No injury report rows (no games, or report not published). Nothing to store.")
            return
        print("[-] No data returned. Check nbainjuries package or date range.")
        sys.exit(1)

    # Filter out blank players
    all_rows = [r for r in all_rows if r['player'] and r['team']]
    print(summarize(all_rows))

    def plan_text(plan):
        return (f"{len(plan['insert']):,} of {len(all_rows):,} rows are new and would be inserted | "
                f"{len(plan['supersede']):,} stored rows from an older report -> quarantine | "
                f"{plan['ignored_older'] + plan['older_in_fetch']:,} rows ignored (a newer report exists)")

    if dry_run:
        conn = sqlite3.connect(f"file:{db_path or NBA_DB}?mode=ro", uri=True)
        plan = plan_store(conn, all_rows)
        conn.close()
        print(f"\n[dry-run] {plan_text(plan)}. Nothing written.")
        return

    conn = sqlite3.connect(db_path or NBA_DB)
    plan = plan_store(conn, all_rows)
    inserted = apply_store(conn, plan)
    skipped = len(all_rows) - inserted
    conn.commit()
    print(f"[+] {plan_text(plan)}")

    # B.4 gate verification
    days = conn.execute(
        "SELECT COUNT(DISTINCT game_date) FROM historical_injuries "
        "WHERE game_date BETWEEN ? AND ?", (start_date, end_date)
    ).fetchone()[0]
    total = conn.execute(
        "SELECT COUNT(*) FROM historical_injuries "
        "WHERE game_date BETWEEN ? AND ?", (start_date, end_date)
    ).fetchone()[0]

    print(f"\n[+] Done. Inserted: {inserted:,} | Skipped (duplicates): {skipped:,}")
    print(f"[+] Coverage: {days} distinct game days | {total:,} total rows in DB")

    span_days = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days + 1
    if span_days <= 60:
        # B.4 gates belong to the original one-off 2025-26 backfill; meaningless for short ranges
        conn.close()
        return

    if days < 70:
        print(f"[!] WARNING: Only {days} days covered — gate requires >= 70")
    else:
        print(f"[✅] B.4 Gate 1 PASSED: {days} days >= 70")

    if total < 1500:
        print(f"[!] WARNING: Only {total} total rows — gate requires >= 1,500")
    else:
        print(f"[✅] B.4 Gate 2 PASSED: {total} rows >= 1,500")

    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrape historical NBA injury reports')
    parser.add_argument('--start', default=DEFAULT_START, help='Start date YYYY-MM-DD')
    parser.add_argument('--end',   default=DEFAULT_END,   help='End date YYYY-MM-DD')
    parser.add_argument('--dry-run', action='store_true', help='Fetch and print counts; write nothing.')
    parser.add_argument('--batch-size', type=int, default=10, help='Reports fetched concurrently (1 = one at a time).')
    parser.add_argument('--delay', type=float, default=0.5, help='Seconds between batches.')
    parser.add_argument('--previous-slate', action='store_true',
                        help="Nightly mode: fetch the previous slate date's 5 PM ET report "
                             "(US-Eastern). No report (no games) is not an error.")
    args = parser.parse_args()
    if args.previous_slate:
        sys.path.insert(0, MODEL_DIR)
        from utils.slate_date import previous_slate_date
        d = previous_slate_date()
        run(d, d, dry_run=args.dry_run, allow_empty=True)
    else:
        run(args.start, args.end, dry_run=args.dry_run, batch_size=args.batch_size, delay=args.delay)
