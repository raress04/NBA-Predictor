"""
Bet Tracker DB Schema
---------------------
Creates and initialises bet_tracker.db — a persistent SQLite database
that records every pick the AI Bet Tool generates, along with the
resolved outcomes, so we can track real win rates over time.

Tables
------
daily_picks   — one row per pick generated per day
pick_results  — resolved outcome for each pick (joined by pick_id)
game_results  — actual game scores per day
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'bet_tracker.db')


def get_connection() -> sqlite3.Connection:
    """Return a connection to the bet tracker DB (creating it if needed)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create all tables if they do not already exist (idempotent)."""
    conn = get_connection()
    cur = conn.cursor()

    # ── daily_picks ──────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_picks (
            id                INTEGER  PRIMARY KEY AUTOINCREMENT,
            date              TEXT     NOT NULL,           -- YYYY-MM-DD
            bet_type          TEXT     NOT NULL,           -- 'prop' | 'spread' | 'total'
            stat_category     TEXT     NOT NULL,           -- POINTS | REBOUNDS | ASSISTS | SPREAD | TOTAL
            player_or_team    TEXT     NOT NULL,
            game              TEXT     NOT NULL,           -- "Away @ Home"
            direction         TEXT     NOT NULL,           -- OVER | UNDER | COVER
            line              REAL,                        -- the bookmaker's line
            odds              REAL,                        -- decimal odds
            sim_confidence    REAL,                        -- raw sim confidence (%)
            blended_confidence REAL,                       -- blended sim+DB confidence (%)
            edge_pct          REAL,                        -- edge above 50% flip
            -- Spread context fields (NULL for non-spread picks)
            projected_margin  REAL,
            safety_margin     REAL,
            market_spread     REAL,
            -- source info
            source_file       TEXT                         -- which predicts_report.tex generated this
        )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_dp_date ON daily_picks(date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_dp_type ON daily_picks(bet_type, stat_category)")

    # ── pick_results ─────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pick_results (
            id            INTEGER  PRIMARY KEY AUTOINCREMENT,
            pick_id       INTEGER  NOT NULL REFERENCES daily_picks(id),
            actual_value  REAL,                            -- actual stat or margin
            is_win        INTEGER  NOT NULL DEFAULT 0,     -- 1 = WIN, 0 = LOSS
            diff_val      REAL,                            -- actual - line
            diff_pct      REAL,                            -- (actual - line) / line * 100
            resolved_at   TEXT                             -- YYYY-MM-DD when result was recorded
        )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_pr_pick ON pick_results(pick_id)")

    # ── game_results ─────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS game_results (
            id             INTEGER  PRIMARY KEY AUTOINCREMENT,
            date           TEXT     NOT NULL,
            home_team      TEXT     NOT NULL,
            away_team      TEXT     NOT NULL,
            home_score     INTEGER,
            away_score     INTEGER,
            total          INTEGER  GENERATED ALWAYS AS (home_score + away_score) VIRTUAL,
            actual_spread  REAL     GENERATED ALWAYS AS (home_score - away_score)  VIRTUAL,
            UNIQUE(date, home_team, away_team)
        )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_gr_date ON game_results(date)")

    conn.commit()
    conn.close()
    print(f"[+] bet_tracker.db initialised at: {DB_PATH}")


if __name__ == '__main__':
    init_db()
