import sqlite3
import os
from typing import List

# Paths relative to project root
NBA_DB = 'database/nba_data.db'

def get_historical_injury_state(iso_date: str, team_id: int) -> List[str]:
    """
    Returns list of player_names ABSENT on iso_date for team_id.
    Absent = appeared in 2025-26 season for this team but NOT in this game.

    Args:
        iso_date: 'YYYY-MM-DD' format (matched with LIKE 'YYYY-MM-DD%')
        team_id:  Full NBA integer ID (e.g. 1610612757)

    Returns: list of absent player name strings. Returns [] on any data gap (safe default).
    """
    if not os.path.exists(NBA_DB):
        # Maybe running from inside utils/
        db_path = '../database/nba_data.db'
    else:
        db_path = NBA_DB
        
    conn = sqlite3.connect(db_path)

    # SEASON ROSTER: date-bounded to prevent post-trade players appearing in past games
    # Uses season='2025-26' (not game_date LIKE '2026%') so Oct-Dec appearances included
    season_players = conn.execute("""
        SELECT DISTINCT player_name FROM box_scores
        WHERE team_id = ?
          AND season = '2025-26'
          AND game_date < ?
    """, [team_id, iso_date]).fetchall()
    season_names = {r[0] for r in season_players}

    if not season_names:
        conn.close()
        return []  # no history yet (very early season games) — safe default

    # WHO ACTUALLY PLAYED on this date
    played = conn.execute("""
        SELECT DISTINCT player_name FROM box_scores
        WHERE team_id = ?
          AND game_date LIKE ?
    """, [team_id, f'{iso_date}%']).fetchall()
    played_names = {r[0] for r in played}

    conn.close()

    if not played_names:
        return []  # no box score for this date/team — return [] not entire roster

    return list(season_names - played_names)


def validate_injury_reconstruction():
    """
    Validation test using Portland on March 10, 2026.
    Expected absent: Blake Wesley, Rayan Rupert, Shaedon Sharpe, Caleb Love, Duop Reath
    Pass criteria: >= 3 of 5 found.
    """
    db_path = NBA_DB if os.path.exists(NBA_DB) else '../database/nba_data.db'
    conn = sqlite3.connect(db_path)
    por_id = conn.execute(
        "SELECT team_id FROM teams WHERE abbreviation = 'POR'"
    ).fetchone()[0]
    conn.close()

    absent = get_historical_injury_state('2026-03-10', por_id)
    print(f'Portland absent on 2026-03-10 (team_id={por_id}):')
    print(f'  {sorted(absent)}')

    expected = {'Blake Wesley', 'Rayan Rupert', 'Shaedon Sharpe', 'Caleb Love', 'Duop Reath'}
    matched = expected & set(absent)
    print(f'  Matched {len(matched)}/5 expected: {matched}')

    if len(matched) >= 3:
        print('  STATUS: PASS — proceeding to full pipeline')
        return True
    else:
        print('  STATUS: FAIL — check box_scores data before running pipeline')
        return False


if __name__ == '__main__':
    validate_injury_reconstruction()
