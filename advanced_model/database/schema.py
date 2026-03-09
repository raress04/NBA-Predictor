import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'nba_data.db')

def init_db():
    '''
    Initialize the SQLite database and create necessary tables.
    '''
    # Start fresh for v3 transition
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table for Teams
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY,
            abbreviation TEXT NOT NULL,
            city TEXT NOT NULL,
            nickname TEXT NOT NULL
        )
    ''')

    # Table for Games
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            season TEXT NOT NULL,
            game_date TEXT NOT NULL,
            home_team_id INTEGER NOT NULL,
            away_team_id INTEGER NOT NULL,
            home_points INTEGER,
            away_points INTEGER,
            FOREIGN KEY(home_team_id) REFERENCES teams(team_id),
            FOREIGN KEY(away_team_id) REFERENCES teams(team_id)
        )
    ''')

    # Table for Player Box Scores (1 entry per player per game)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS box_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            team_id INTEGER NOT NULL,
            minutes TEXT,
            fgm INTEGER,
            fga INTEGER,
            fg_pct REAL,
            fg3m INTEGER,
            fg3a INTEGER,
            fg3_pct REAL,
            ftm INTEGER,
            fta INTEGER,
            ft_pct REAL,
            oreb INTEGER,
            dreb INTEGER,
            reb INTEGER,
            ast INTEGER,
            stl INTEGER,
            blk INTEGER,
            tov INTEGER,
            pf INTEGER,
            pts INTEGER,
            plus_minus INTEGER,
            FOREIGN KEY(game_id) REFERENCES games(game_id),
            FOREIGN KEY(team_id) REFERENCES teams(team_id)
        )
    ''')
    
    # Optional performance indices on frequently joined columns
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_box_game ON box_scores(game_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_box_player ON box_scores(player_id)')

    # Table for Play-by-Play Events
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS play_by_play (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            action_number INTEGER NOT NULL,
            clock TEXT,
            period INTEGER,
            team_id INTEGER,
            person_id INTEGER,
            player_name TEXT,
            x_legacy INTEGER,
            y_legacy INTEGER,
            shot_distance INTEGER,
            shot_result TEXT,
            is_field_goal INTEGER,
            score_home TEXT,
            score_away TEXT,
            points_total INTEGER,
            location TEXT,
            description TEXT,
            action_type TEXT,
            sub_type TEXT,
            shot_value INTEGER,
            action_id INTEGER,
            FOREIGN KEY(game_id) REFERENCES games(game_id)
        )
    ''')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_pbp_game ON play_by_play(game_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_pbp_player1 ON play_by_play(person_id)')

    conn.commit()
    conn.close()
    print(f"[+] Initialized target database at {DB_PATH}")

if __name__ == '__main__':
    init_db()
