"""
advanced_model/etl/parse_all_predictions.py
============================================
Parses ALL /predictions/*.txt monthly archives into projection_outcomes_raw table
in bet_tracker.db. Extends and replaces etl/parse_historical_reports.py.

Extracts for every player in every game:
  - projected pts / reb / ast
  - bookmaker line per stat (from O/U markers)
  - confidence and edge (where VALUABLE markers exist)
  - is_valuable flag

Uses INSERT OR IGNORE — fully idempotent.

Usage:
    python3 -m etl.parse_all_predictions
"""

import os, re, sqlite3, pandas as pd
from pathlib import Path

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDS_ROOT = os.path.join(MODEL_DIR, 'predictions')
DB_PATH    = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')
LOG_PATH   = os.path.join(MODEL_DIR, 'logs', 'ml3b_parse_predictions.txt')

MONTH_DIRS = ['january', 'february', 'march', 'april', 'may', 'june',
              'october', 'november', 'december']

# ── Regex patterns ────────────────────────────────────────────────────────────

# Date from filename: predicts_2026-01-15.txt or predicts_2026-01-15.txt
DATE_RE      = re.compile(r'predicts[_-](\d{4}-\d{2}-\d{2})\.txt$', re.IGNORECASE)

# Matchup header line
MATCHUP_RE   = re.compile(r'MATCHUP:\s*(.+)', re.IGNORECASE)

# Player projection line (core):
#   LeBron James         -> 24 PTS, 8 REB, 9 AST
#   LeBron James         -> 24 PTS, 8 REB, 9 AST (O/U 23.5 points ignore, ...)
PLAYER_RE    = re.compile(
    r"^\s{0,6}([A-Za-z\xC0-\xD6\xD8-\xF6\xF8-\xFF][A-Za-z\xC0-\xD6\xD8-\xF6\xF8-\xFF\s'\.\-]+?)"
    r"\s*->\s*(\d+(?:\.\d+)?)\s+PTS?,\s*(\d+(?:\.\d+)?)\s+REB?,\s*(\d+(?:\.\d+)?)\s+AST?"
    r"([^\n]*)",
    re.MULTILINE
)

# Extract per-stat bookmaker line from the trailing part of a player line
# e.g. "(O/U 23.5 points VALUABLE: Over (82% conf ...))"
#  or  "(O/U 23.5 points ignore)"
STAT_LINE_RE = {
    'POINTS':   re.compile(r'O/U\s*([\d.]+)\s+point', re.IGNORECASE),
    'REBOUNDS': re.compile(r'O/U\s*([\d.]+)\s+rebound', re.IGNORECASE),
    'ASSISTS':  re.compile(r'O/U\s*([\d.]+)\s+assist', re.IGNORECASE),
}

# Confidence extraction: "82% conf" or "conf (raw: 82%)"
CONF_RE      = re.compile(r'(\d+(?:\.\d+)?)\s*%\s*conf(?:idence)?', re.IGNORECASE)
EDGE_RE      = re.compile(r'([\d.]+)%\s*edge', re.IGNORECASE)
VALUABLE_RE  = re.compile(r'VALUABLE', re.IGNORECASE)

# Over/Under direction
OVER_RE      = re.compile(r'VALUABLE:\s*Over', re.IGNORECASE)
UNDER_RE     = re.compile(r'VALUABLE:\s*Under', re.IGNORECASE)


# ── Schema setup ──────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS projection_outcomes_raw (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_date       TEXT,
    matchup         TEXT,
    player          TEXT,
    category        TEXT,
    projected_value REAL,
    bookmaker_line  REAL,
    actual_value    REAL,
    model_bias      REAL,
    direction       TEXT,
    confidence      REAL,
    edge_pct        REAL,
    is_valuable     INTEGER DEFAULT 0,
    source_file     TEXT,
    source_mode     TEXT
);
"""

CREATE_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_unique
ON projection_outcomes_raw (game_date, player, category, source_file);
"""


# ── File parser ───────────────────────────────────────────────────────────────

def parse_file(filepath: str) -> list:
    fname = os.path.basename(filepath)
    date_m = DATE_RE.search(fname)
    if not date_m:
        return []
    game_date = date_m.group(1)

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        print(f"    [!] Cannot read {filepath}: {e}")
        return []

    has_valuable = bool(VALUABLE_RE.search(content))
    source_mode  = 'live' if has_valuable else 'historical'
    rel_path     = os.path.relpath(filepath, MODEL_DIR).replace('\\', '/')

    rows = []
    current_matchup = 'UNKNOWN'

    for match in PLAYER_RE.finditer(content):
        player   = match.group(1).strip().rstrip('-').strip()
        # Skip known non-player lines
        if len(player) < 4 or player.upper() in ('HOME TEAM', 'AWAY TEAM', 'BIAS CORRECTION'):
            continue
        # Skip if player name looks like a header (all caps or contains '[')
        if '[' in player or player.isupper():
            continue

        proj_pts = float(match.group(2))
        proj_reb = float(match.group(3))
        proj_ast = float(match.group(4))
        trailing = match.group(5) or ''

        # Update matchup context from content up to this match
        preceding = content[:match.start()]
        mm = list(MATCHUP_RE.finditer(preceding))
        if mm:
            current_matchup = mm[-1].group(1).strip()

        for cat, proj_val, stat_re in [
            ('POINTS',   proj_pts, STAT_LINE_RE['POINTS']),
            ('REBOUNDS', proj_reb, STAT_LINE_RE['REBOUNDS']),
            ('ASSISTS',  proj_ast, STAT_LINE_RE['ASSISTS']),
        ]:
            # Extract bookmaker line for this stat
            line_m  = stat_re.search(trailing)
            bk_line = float(line_m.group(1)) if line_m else None

            # Extract confidence and edge (only present near VALUABLE lines)
            conf_m  = CONF_RE.search(trailing)
            edge_m  = EDGE_RE.search(trailing)
            conf    = float(conf_m.group(1)) if conf_m else None
            edge    = float(edge_m.group(1)) if edge_m else None

            # Direction
            if OVER_RE.search(trailing):
                direction = 'OVER'
            elif UNDER_RE.search(trailing):
                direction = 'UNDER'
            else:
                direction = None

            # is_valuable: only if VALUABLE present for this specific stat
            is_val  = 1 if VALUABLE_RE.search(trailing) and line_m else 0

            rows.append({
                'game_date':       game_date,
                'matchup':         current_matchup,
                'player':          player,
                'category':        cat,
                'projected_value': proj_val,
                'bookmaker_line':  bk_line,
                'actual_value':    None,
                'model_bias':      None,
                'direction':       direction,
                'confidence':      conf,
                'edge_pct':        edge,
                'is_valuable':     is_val,
                'source_file':     rel_path,
                'source_mode':     source_mode,
            })

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PARSE ALL PREDICTIONS → projection_outcomes_raw")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(CREATE_TABLE_SQL)
    conn.execute(CREATE_INDEX_SQL)
    conn.commit()

    # Already-parsed source files (idempotency)
    existing = set(
        r[0] for r in conn.execute(
            "SELECT DISTINCT source_file FROM projection_outcomes_raw"
        )
    )
    print(f"[*] Already parsed files in DB: {len(existing)}")

    log_lines = []
    total_new = 0

    for month in MONTH_DIRS:
        month_dir = os.path.join(PREDS_ROOT, month)
        if not os.path.isdir(month_dir):
            continue

        txt_files = sorted(Path(month_dir).glob('predicts_*.txt'))
        if not txt_files:
            txt_files = sorted(Path(month_dir).glob('predicts*.txt'))

        month_new = 0
        for fpath in txt_files:
            rel = os.path.relpath(str(fpath), MODEL_DIR).replace('\\', '/')
            if rel in existing:
                continue

            rows = parse_file(str(fpath))
            if not rows:
                print(f"  [!] No rows parsed from {fpath.name}")
                continue

            df = pd.DataFrame(rows)
            # INSERT OR IGNORE via to_sql with method=multi and duplicates handled by index
            inserted = 0
            for _, r in df.iterrows():
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO projection_outcomes_raw
                        (game_date, matchup, player, category, projected_value,
                         bookmaker_line, actual_value, model_bias, direction,
                         confidence, edge_pct, is_valuable, source_file, source_mode)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (r.game_date, r.matchup, r.player, r.category,
                          r.projected_value, r.bookmaker_line, r.actual_value,
                          r.model_bias, r.direction, r.confidence,
                          r.edge_pct, int(r.is_valuable), r.source_file, r.source_mode))
                    inserted += conn.execute("SELECT changes()").fetchone()[0]
                except Exception:
                    pass
            conn.commit()

            month_new += inserted
            total_new += inserted
            log_lines.append(f"  {fpath.name:<45} → {inserted:>4} rows inserted")

        if month_new:
            print(f"[{month.upper():<10}] {month_new:,} new rows inserted")

    # Summary
    total_raw = conn.execute("SELECT COUNT(*) FROM projection_outcomes_raw").fetchone()[0]
    days_raw  = conn.execute("SELECT COUNT(DISTINCT game_date) FROM projection_outcomes_raw").fetchone()[0]
    players_raw = conn.execute("SELECT COUNT(DISTINCT player) FROM projection_outcomes_raw").fetchone()[0]

    summary = [
        "",
        "=" * 60,
        f"  TOTAL NEW ROWS INSERTED : {total_new:,}",
        f"  TOTAL IN TABLE          : {total_raw:,}",
        f"  DISTINCT GAME DATES     : {days_raw}",
        f"  DISTINCT PLAYERS        : {players_raw:,}",
        "=" * 60,
    ]
    for l in summary:
        print(l)

    conn.close()

    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(log_lines + summary))
    print(f"\n[+] Log → {LOG_PATH}")
    print("[✅] ML.3b-1 Complete.")


if __name__ == '__main__':
    main()
