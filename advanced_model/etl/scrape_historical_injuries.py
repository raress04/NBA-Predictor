"""
advanced_model/etl/scrape_historical_injuries.py
=================================================
Scrapes NBA official injury reports for the 2025-26 season
using the nbainjuries package and stores them in nba_data.db.

Run once before the retroactive pipeline re-run.
Idempotent: INSERT OR IGNORE prevents duplicates.

Usage:
    python3 -m etl.scrape_historical_injuries
    python3 -m etl.scrape_historical_injuries --start 2026-01-01 --end 2026-03-24
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


async def fetch_all(start: str, end: str):
    """Fetch injury data for all dates in range concurrently (batched)."""
    all_rows = []
    dates = list(date_range(start, end))
    BATCH_SIZE = 10  # avoid hammering the server

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(dates), BATCH_SIZE):
            batch = dates[i:i + BATCH_SIZE]
            tasks = [fetch_one_day(session, dt) for dt in batch]
            results = await asyncio.gather(*tasks)
            for rows in results:
                all_rows.extend(rows)
            print(f"    Fetched {min(i + BATCH_SIZE, len(dates))}/{len(dates)} days...")
            await asyncio.sleep(0.5)  # polite delay between batches

    return all_rows


def run(start_date: str = DEFAULT_START, end_date: str = DEFAULT_END):
    print(f"[+] Fetching injury data for {start_date} → {end_date}...")
    print(f"    Using {REPORT_HOUR}:00 ET reports")

    all_rows = asyncio.run(fetch_all(start_date, end_date))
    print(f"[+] Total rows fetched: {len(all_rows)}")

    if not all_rows:
        print("[-] No data returned. Check nbainjuries package or date range.")
        sys.exit(1)

    # Filter out blank players
    all_rows = [r for r in all_rows if r['player'] and r['team']]

    conn = sqlite3.connect(NBA_DB)
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
    args = parser.parse_args()
    run(args.start, args.end)
