"""
utils/rerun.py
==============
W1 (Plans/W1_RERUN_AUDIT.md): a re-run of a slate replaces its rows instead of adding a
second set.

Rule: the latest run of a slate is the record, game by game. Before a run writes its rows,
rows from an earlier run with the same key are flagged `superseded = 1` (never deleted).
A game the re-run did not simulate (it had already started) keeps its earlier rows.

Key of a candidate: (player, category). A player plays one game a night, so the player
identifies the game. TOTAL rows are all named 'Game Total', so they also need the matchup;
a TOTAL row without a matchup is never superseded (its game cannot be told apart).
"""
from typing import Iterable, Optional, Tuple

Key = Tuple[str, str, Optional[str]]


def current_only(conn, table: str, alias: str = '') -> str:
    """SQL fragment ' AND COALESCE(<alias.>superseded, 0) = 0' for readers, or '' when the
    table has no superseded column yet (old databases, read-only connections, test schemas)."""
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if 'superseded' not in cols:
        return ''
    return f" AND COALESCE({alias + '.' if alias else ''}superseded, 0) = 0"


def candidate_key(player, category, matchup=None) -> Key:
    cat = str(category or '').upper()
    return (str(player or ''), cat, (matchup or None) if cat == 'TOTAL' else None)


def supersede_keys(conn, table: str, game_date: str, keys: Iterable[Key],
                   category_col: str = 'stat_category', extra_where: str = '',
                   extra_params: tuple = ()) -> int:
    """Flag the current rows of `table` on `game_date` that have one of `keys`.
    Call it BEFORE inserting the new run's rows. Returns the number of rows flagged.
    The caller commits."""
    flagged = 0
    for player, cat, matchup in set(keys):
        if not player or not cat:
            continue
        sql = (f"UPDATE {table} SET superseded = 1 "
               f"WHERE game_date = ? AND player = ? AND UPPER({category_col}) = ? "
               f"AND COALESCE(superseded, 0) = 0 {extra_where}")
        params = [game_date, player, cat]
        if cat == 'TOTAL':
            if not matchup:
                continue
            sql += " AND matchup = ?"
            params.append(matchup)
        flagged += conn.execute(sql, params + list(extra_params)).rowcount
    return flagged


def supersede_parlays(conn, game_date: str, keep_ids: Iterable[int]) -> int:
    """After a live run has saved its parlays, flag every other current parlay of the slate.
    A run that built no parlay still supersedes the earlier ones: its report says
    "no parlay" and the latest run is the record. Returns the number flagged."""
    keep = [int(i) for i in keep_ids if i is not None and int(i) > 0]
    sql = "UPDATE parlays SET superseded = 1 WHERE game_date = ? AND COALESCE(superseded, 0) = 0"
    params = [game_date]
    if keep:
        sql += f" AND id NOT IN ({','.join('?' * len(keep))})"
        params += keep
    return conn.execute(sql, params).rowcount
