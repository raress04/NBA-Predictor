"""
utils/box_actuals.py
====================
Single source of truth for grading logged candidates against REAL box scores.

Why this exists (CRITICAL_BUGS_FOUND_06.11 — C4):
    ml_shadow_picks used to be resolved by joining bet_tracker.projection_outcomes,
    whose 2026-27 rows are 'live_pending' placeholders with actual_value = 0.0.
    Every OVER was graded a loss and every UNDER a win. Grading must only ever
    use nba_data.db.box_scores.

Rules:
    * A player with no box-score row on the game date (DNP / not in data yet) is
      left unresolved (None) — never graded as a loss.
    * A player listed with 0 minutes is a DNP -> unresolved.
    * actual == line is a PUSH -> is_win None (resolved, but excluded from win rates).
    * Duplicate box-score rows (same game_id + player_id) are collapsed.
"""

import difflib
import os
import re
import sqlite3
import unicodedata
from typing import Dict, Optional

NBA_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'database', 'nba_data.db')

STAT_COLUMNS = {'POINTS': 'pts', 'REBOUNDS': 'reb', 'ASSISTS': 'ast'}
_SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}


def normalize_name(name: str) -> str:
    """Lower-case, strip accents/punctuation and generational suffixes."""
    if not name:
        return ''
    s = unicodedata.normalize('NFKD', str(name))
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-zA-Z0-9 ]", ' ', s).lower()
    parts = [p for p in s.split() if p not in _SUFFIXES]
    return ' '.join(parts)


def _minutes_to_float(value) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        if ':' in s:
            mm, ss = s.split(':', 1)
            return float(mm) + float(ss) / 60.0
        return float(s)
    except ValueError:
        return None


def load_box_actuals(game_date: str, nba_db: Optional[str] = None) -> Dict[str, dict]:
    """
    Return {player_name: {'POINTS', 'REBOUNDS', 'ASSISTS', 'minutes'}} for one date.
    Duplicate rows for the same (game_id, player_id) are collapsed.
    """
    path = nba_db or NBA_DB
    if not os.path.exists(path):
        return {}
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            """
            SELECT game_id, player_id, player_name, minutes, pts, reb, ast
            FROM box_scores
            WHERE substr(game_date, 1, 10) = ?
            """,
            (game_date[:10],),
        ).fetchall()
    finally:
        conn.close()

    seen = set()
    actuals: Dict[str, dict] = {}
    for game_id, player_id, name, minutes, pts, reb, ast in rows:
        key = (game_id, player_id)
        if key in seen:
            continue
        seen.add(key)
        if not name:
            continue
        actuals[name] = {
            'POINTS': pts,
            'REBOUNDS': reb,
            'ASSISTS': ast,
            'minutes': _minutes_to_float(minutes),
        }
    return actuals


class ActualsIndex:
    """Name lookup over one date's actuals: exact -> normalized -> fuzzy (0.85)."""

    def __init__(self, actuals: Dict[str, dict]):
        self.actuals = actuals
        self._by_norm = {}
        for name, stats in actuals.items():
            self._by_norm.setdefault(normalize_name(name), stats)
        self._norm_keys = list(self._by_norm.keys())

    def get(self, player: str) -> Optional[dict]:
        if not player:
            return None
        if player in self.actuals:
            return self.actuals[player]
        norm = normalize_name(player)
        if norm in self._by_norm:
            return self._by_norm[norm]
        close = difflib.get_close_matches(norm, self._norm_keys, n=1, cutoff=0.85)
        return self._by_norm[close[0]] if close else None


def actual_for(index: ActualsIndex, player: str, stat_category: str) -> Optional[float]:
    """Actual stat value, or None if the player did not play / is unknown."""
    stats = index.get(player)
    if not stats:
        return None
    minutes = stats.get('minutes')
    if minutes is not None and minutes <= 0:
        return None  # DNP
    value = stats.get(str(stat_category).upper())
    return None if value is None else float(value)


def grade(direction: str, actual: Optional[float], line: Optional[float]) -> Optional[int]:
    """1 = win, 0 = loss, None = push or ungradeable."""
    if actual is None or line is None:
        return None
    d = str(direction).upper()
    if actual == line:
        return None
    if d == 'OVER':
        return 1 if actual > line else 0
    if d == 'UNDER':
        return 1 if actual < line else 0
    return None
