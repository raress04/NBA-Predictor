"""
utils/box_scores_io.py
======================
Idempotent writes to nba_data.db.box_scores.

Why this exists (CRITICAL_BUGS_FOUND_06.11, Step 1):
    box_scores had no UNIQUE constraint. `fetch_range_box_scores.py` used
    INSERT OR REPLACE (which never "replaces" without a unique key) and
    `prepare_analyze.py` / `incremental_backfill.py` / `fast_backfiller.py`
    used pandas `to_sql(if_exists='append')`. Re-running any of them for a
    date that was already ingested duplicated every row: 1,337 duplicated
    (game_id, player_id) pairs in 2026-27 (Oct 26 -> Nov 4) and 203 in
    Apr 2026. Duplicates double-count games in EWMA baselines, minutes
    stats, games-played thresholds and DB hit rates.

All writers must go through `insert_box_scores()`.
"""

import sqlite3
from typing import Iterable, List

UNIQUE_INDEX = "ux_box_scores_game_player"


def has_unique_index(conn: sqlite3.Connection) -> bool:
    rows = conn.execute("PRAGMA index_list(box_scores)").fetchall()
    return any(r[1] == UNIQUE_INDEX for r in rows)


def ensure_unique_index(conn: sqlite3.Connection) -> bool:
    """Create the UNIQUE(game_id, player_id) index if the table is already clean.
    Returns True if the index exists afterwards. Never deletes data."""
    if has_unique_index(conn):
        return True
    dup = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT 1 FROM box_scores GROUP BY game_id, player_id HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    if dup:
        return False  # run scripts/repair_measurement_20261106.py first
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {UNIQUE_INDEX} "
                 f"ON box_scores(game_id, player_id)")
    return True


def insert_box_scores(conn: sqlite3.Connection, df, cols: List[str],
                      replace: bool = False) -> int:
    """
    Insert box-score rows without ever creating duplicates.

    Args:
        df:      pandas DataFrame with at least `cols`, including game_id and player_id.
        replace: True  -> refresh existing (game_id, player_id) rows with the new values
                 False -> keep existing rows, insert only new ones
    Returns the number of rows written.
    """
    if df is None or len(df) == 0:
        return 0
    if 'game_id' not in cols or 'player_id' not in cols:
        raise ValueError("insert_box_scores requires game_id and player_id columns")

    work = df[cols].copy()
    work['game_id'] = work['game_id'].astype(str)
    work = work.drop_duplicates(subset=['game_id', 'player_id'], keep='last')

    game_ids = sorted(set(work['game_id']))
    existing = set()
    for i in range(0, len(game_ids), 500):
        chunk = game_ids[i:i + 500]
        q = (f"SELECT game_id, player_id FROM box_scores "
             f"WHERE game_id IN ({','.join('?' * len(chunk))})")
        existing.update((str(g), int(p)) for g, p in conn.execute(q, chunk) if p is not None)

    placeholders = ','.join('?' * len(cols))
    insert_sql = f"INSERT OR IGNORE INTO box_scores ({','.join(cols)}) VALUES ({placeholders})"
    written = 0
    for row in work.itertuples(index=False, name=None):
        rec = dict(zip(cols, row))
        key = (str(rec['game_id']), int(rec['player_id']))
        values = tuple(_py(v) for v in row)
        if key in existing:
            if not replace:
                continue
            conn.execute("DELETE FROM box_scores WHERE game_id = ? AND player_id = ?", key)
        conn.execute(insert_sql, values)
        existing.add(key)
        written += 1
    return written


def _py(v):
    """numpy scalars / NaN -> plain Python for sqlite3."""
    try:
        import math
        if hasattr(v, 'item'):
            v = v.item()
        if isinstance(v, float) and math.isnan(v):
            return None
    except Exception:
        pass
    return v
