"""
utils/slate_date.py
===================
Single source of truth for the live slate date.

Why this exists (CRITICAL_BUGS_FOUND_06.11, §15-16, handoff "Prevention"):
    The live run labelled its output with *local time - 2.5 h*
    (live_ingestion.get_effective_date), while the games it simulated came from
    ESPN for the *US-Eastern* date computed with a fixed -4 h offset
    (espn_odds.fetch_espn_odds). From Europe, and in winter when Eastern time is
    UTC-5, the two disagree, which mislabelled whole slates by one day.
    persist_parlay() and the ML layer used datetime.now() - a third rule.

The rule everything now follows (same rule the 06.11 re-dating used):
    slate date = the calendar date in America/New_York at the moment of the run.

NBA game dates (nba_api, ESPN) are US-Eastern dates, so this is also the date the
box scores will carry.

Project-wide time conventions:
  * Anything that selects or labels a game day (today's slate, "yesterday's"
    results, rolling N-day windows, report "as of" dates, nightly log names)
    uses slate_date() / previous_slate_date() — never date.today(),
    datetime.now() or SQLite date('now').
  * Timestamps stored in the databases (created_at, resolved_at) are UTC text
    'YYYY-MM-DD HH:MM:SS', the same as SQLite CURRENT_TIMESTAMP / datetime('now').
  * "Generated at" lines in human-readable reports stay in local wall-clock time.

CLI:  python3 -m utils.slate_date            -> prints YYYY-MM-DD
      python3 -m utils.slate_date --compact  -> prints YYYYMMDD
"""

import sys
from datetime import datetime, timedelta, timezone

try:  # Python 3.9+; needs the IANA tz database (present on Linux/WSL)
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo('America/New_York')
except Exception:  # e.g. Windows Python without the `tzdata` package
    _ET = None


def _us_eastern_offset(utc_dt: datetime) -> timedelta:
    """UTC offset of US-Eastern time at `utc_dt` (fallback when zoneinfo is unavailable).
    DST: second Sunday of March 02:00 local -> first Sunday of November 02:00 local."""
    y = utc_dt.year
    march1 = datetime(y, 3, 1, tzinfo=timezone.utc)
    second_sun_march = march1 + timedelta(days=(6 - march1.weekday()) % 7 + 7)
    nov1 = datetime(y, 11, 1, tzinfo=timezone.utc)
    first_sun_nov = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
    dst_start_utc = second_sun_march + timedelta(hours=7)   # 02:00 EST = 07:00 UTC
    dst_end_utc = first_sun_nov + timedelta(hours=6)        # 02:00 EDT = 06:00 UTC
    return timedelta(hours=-4) if dst_start_utc <= utc_dt < dst_end_utc else timedelta(hours=-5)


def to_eastern(dt: datetime = None) -> datetime:
    """Convert `dt` (default: now) to US-Eastern time.
    A naive datetime is taken as this machine's local time (what datetime.now() returns)."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.astimezone()  # attach the local timezone
    if _ET is not None:
        return dt.astimezone(_ET)
    utc = dt.astimezone(timezone.utc)
    off = _us_eastern_offset(utc)
    return (utc + off).replace(tzinfo=timezone(off))


def slate_date(dt: datetime = None) -> str:
    """Slate date 'YYYY-MM-DD' = US-Eastern calendar date of `dt` (default: now)."""
    return to_eastern(dt).strftime('%Y-%m-%d')


def slate_date_compact(dt: datetime = None) -> str:
    """Slate date as 'YYYYMMDD' (ESPN scoreboard `dates=` format)."""
    return to_eastern(dt).strftime('%Y%m%d')


def previous_slate_date(dt: datetime = None) -> str:
    """The slate before the current one ('yesterday' in US-Eastern terms) — the default
    date whose results are prepared/graded by the nightly pipeline."""
    return (to_eastern(dt) - timedelta(days=1)).strftime('%Y-%m-%d')


def eastern_date_of_utc(ts: str):
    """US-Eastern date 'YYYY-MM-DD' of a UTC timestamp text as SQLite writes it
    (CURRENT_TIMESTAMP / datetime('now'): 'YYYY-MM-DD HH:MM:SS'). None if unparsable."""
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(str(ts).strip().replace('Z', '').replace('T', ' ')[:19])
    except ValueError:
        return None
    return slate_date(t.replace(tzinfo=timezone.utc))


def register_sqlite(conn) -> None:
    """Expose the same rule inside SQL on this connection:
         slate_date()              -> current slate date, e.g. date(slate_date(), '-30 days')
         eastern_date(created_at)  -> US-Eastern date of a stored UTC timestamp
    Use instead of date('now'), which is the UTC date (already 'tomorrow' after 8 pm ET)."""
    conn.create_function('slate_date', 0, slate_date)
    conn.create_function('eastern_date', 1, eastern_date_of_utc)


if __name__ == '__main__':
    print(slate_date_compact() if '--compact' in sys.argv[1:] else slate_date())
