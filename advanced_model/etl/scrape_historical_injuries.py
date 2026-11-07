"""
advanced_model/etl/scrape_historical_injuries.py
=================================================
Scrapes NBA official injury reports for the 2025-26 season
using the nbainjuries package and stores them in nba_data.db.

Originally run once (2025-26). Since 2026-11-06 (T3) run_pipeline.sh calls it every
night with --previous-slate, so the official report is stored every day.
Idempotent: INSERT OR IGNORE prevents duplicates.

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
                'game_date': dt.strftime('%Y-%m-%d'),
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
    all_rows = []
    dates = list(date_range(start, end))
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

    if dry_run:
        conn = sqlite3.connect(f"file:{db_path or NBA_DB}?mode=ro", uri=True)
        new = sum(1 for r in all_rows if not conn.execute(
            "SELECT 1 FROM historical_injuries WHERE game_date=? AND player=? AND team=?",
            (r['game_date'], r['player'], r['team'])).fetchone())
        conn.close()
        print(f"\n[dry-run] {new:,} of {len(all_rows):,} rows are new and would be inserted. Nothing written.")
        return

    conn = sqlite3.connect(db_path or NBA_DB)
    inserted = 0
    skipped  = 0
    for row in all_rows:
        try:
            cur = conn.execute(
                "INSERT OR IGNORE INTO historical_injuries "
                "(game_date, player, team, status, reason) VALUES (?,?,?,?,?)",
                (row['game_date'], row['player'], row['team'],
                 row['status'], row['reason'])
            )
            if cur.rowcount > 0:
                inserted += 1
            else:
                skipped += 1
        except Exception as ex:
            print(f"    [!] DB error: {ex} — {row}")
            skipped += 1

    conn.commit()

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
