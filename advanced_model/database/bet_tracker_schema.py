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
            raw_sim_freq      REAL,                        -- raw sim frequency (0-1)
            blended_confidence REAL,                       -- blended sim+DB confidence (%)
            edge_pct          REAL,                        -- edge above 50% flip
            -- Spread context fields (NULL for non-spread picks)
            projected_margin  REAL,
            safety_margin     REAL,
            market_spread     REAL,
            -- source info
            source_file       TEXT,                        -- which predicts_report.tex generated this
            source_mode       TEXT DEFAULT 'live',         -- 'live' | 'synthetic_circular'
            clv               REAL
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

    # ── projection_outcomes ──────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS projection_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date TEXT NOT NULL,
            player TEXT NOT NULL,
            category TEXT NOT NULL,          -- 'POINTS', 'REBOUNDS', 'ASSISTS', 'TOTAL', 'SPREAD'
            projected_value REAL NOT NULL,   -- from model
            bookmaker_line REAL,             -- NULL for pure historical-mode where no line existed
            actual_value REAL NOT NULL,      -- from box score
            model_bias REAL NOT NULL,        -- actual_value - projected_value
            source_mode TEXT NOT NULL,       -- 'live' or 'historical'
            source_file TEXT NOT NULL
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_po_date ON projection_outcomes(game_date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_po_player ON projection_outcomes(player)")
    
    # ── shadow_picks ─────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS shadow_picks (
            id                INTEGER  PRIMARY KEY AUTOINCREMENT,
            game_date         TEXT,                        -- YYYY-MM-DD
            player            TEXT,
            stat_category     TEXT,                        -- POINTS | REBOUNDS | ASSISTS | TOTAL | SPREAD
            direction         TEXT,                        -- OVER | UNDER | COVER
            line              REAL,
            projection        REAL,
            sim_confidence    REAL,
            edge_pct          REAL,
            ban_reason        TEXT,                        -- 'BANNED_CATEGORY', 'SYNERGY_COLLAPSE', 'MARKET_INVERSION', etc.
            actual_value      REAL,                        -- the actual stat or margin
            actual_result     INTEGER,                     -- 1 = WIN, 0 = LOSS
            clv               REAL,                        -- closing line value/odds diff if applicable
            created_at        TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sp_date ON shadow_picks(game_date)")

    # ── parlays ───────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS parlays (
            id             INTEGER  PRIMARY KEY AUTOINCREMENT,
            game_date      TEXT     NOT NULL,               -- YYYY-MM-DD
            model_version  TEXT     DEFAULT 'ERA_3',
            combined_odds  REAL,                            -- product of all leg decimal odds
            stake          REAL,                            -- fractional bankroll (e.g. 0.002)
            payout         REAL,                            -- NULL until resolved
            result         TEXT,                            -- 'WIN' | 'LOSS' | NULL
            roi            REAL,                            -- (payout-stake)/stake, NULL until resolved
            created_at     TEXT     DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_par_date ON parlays(game_date)")

    # ── parlay_legs ───────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS parlay_legs (
            id             INTEGER  PRIMARY KEY AUTOINCREMENT,
            parlay_id      INTEGER  NOT NULL REFERENCES parlays(id),
            leg_number     INTEGER  NOT NULL,
            pick_id        INTEGER,                         -- REFERENCES daily_picks(id), nullable
            player         TEXT,
            stat_category  TEXT,
            direction      TEXT,
            line           REAL,
            odds           REAL,
            sim_confidence REAL,
            edge_pct       REAL
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pl_parlay ON parlay_legs(parlay_id)")

    # ── model_version columns (idempotent) ────────────────────────────────────
    for table in ('daily_picks', 'shadow_picks'):
        try:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN model_version TEXT DEFAULT 'ERA_3'")
        except Exception:
            pass  # column already exists

    conn.commit()
    conn.close()
    print(f"[+] bet_tracker.db initialised at: {DB_PATH}")


if __name__ == '__main__':
    init_db()
