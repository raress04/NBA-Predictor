"""
Bet Tracker ETL
---------------
Reads predicts_report.tex (picks) and analyze.tex (outcomes) and
persists them into bet_tracker.db for long-term performance tracking.

Usage
-----
# Seed yesterday's data (run manually after each day's ml.
python -m advanced_model.etl.bet_tracker seed --date 2026-03-04

# Query summary stats
python -m advanced_model.etl.bet_tracker summary

# Query confidence band breakdown
python -m advanced_model.etl.bet_tracker bands
"""

import re
import os
import sqlite3
import argparse
from datetime import date as date_type, datetime

# Add etl directory to path for imports if needed, but bet_tracker is already in etl
try:
    from advanced_model.etl.espn_odds import fetch_espn_odds, match_espn_odds_to_game
except ImportError:
    from etl.espn_odds import fetch_espn_odds, match_espn_odds_to_game

def get_month_folder(date_str):
    """Maps YYYY-MM-DD to lowercase month name (e.g. 'march')."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%B").lower()
    except:
        return "march" # Fallback

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR      = os.path.join(MODEL_DIR, 'database')
DB_PATH     = os.path.join(DB_DIR, 'bet_tracker.db')

DEFAULT_PREDICTS  = os.path.join(MODEL_DIR, 'predicts_report.tex')
DEFAULT_ANALYZE   = os.path.join(MODEL_DIR, 'analyze.tex')
PREDICTS_MAR3     = os.path.join(MODEL_DIR, 'predicts_2026-03-03.txt')


# ── DB Connection ─────────────────────────────────────────────────────────────
def _get_conn() -> sqlite3.Connection:
    """Open bet_tracker.db, auto-initialising if missing."""
    if not os.path.exists(DB_PATH):
        from database.bet_tracker_schema import init_db
        init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_analyze_tex_full(tex_path: str) -> list[dict]:
    """
    Parse analyze.tex as the PRIMARY source for BOTH picks and outcomes.
    Returns combined pick+outcome dicts.
    """
    if not os.path.exists(tex_path):
        print(f"  [-] File not found: {tex_path}")
        return []

    with open(tex_path, 'r', encoding='utf-8') as fh:
        content = fh.read()

    records = []
    stat_map = {
        'REB': 'REBOUNDS', 'POI': 'POINTS', 'AST': 'ASSISTS', 'ASS': 'ASSISTS',
        'TOT': 'TOTAL', 'SPR': 'SPREAD',
        'POINTS': 'POINTS', 'REBOUNDS': 'REBOUNDS', 'ASSISTS': 'ASSISTS',
        'TOTAL': 'TOTAL', 'SPREAD': 'SPREAD',
    }
    bet_type_map = {
        'POINTS': 'prop', 'REBOUNDS': 'prop', 'ASSISTS': 'prop',
        'TOTAL': 'total', 'SPREAD': 'spread',
    }

    row_re = re.compile(
        r'^([A-Za-z\s\.\'\-]+?)\s*&\s*(\d+)\\%\s*&\s*(OVER|UNDER|COVER)\s+([+\-]?[\d\.]+)\s+(\w+)'
        r'\s*&\s*([\-\d\.]+)\s*&\s*([+\-]?[\d\.]+)\s*&\s*([+\-]?[\d\.]+)\\%\s*&\s*\\textcolor\{[^}]+\}\{(WON|LOST)',
        re.MULTILINE | re.IGNORECASE
    )

    for m in row_re.finditer(content):
        player, conf, direction, line, stat_raw, actual, diff_val, diff_pct, outcome = m.groups()
        player = player.strip()
        stat = stat_map.get(stat_raw.upper(), stat_raw.upper())
        bet_type = bet_type_map.get(stat, 'prop')

        records.append({
            'player_or_team': player,
            'stat_category': stat,
            'bet_type': bet_type,
            'direction': direction.upper(),
            'line': float(line),
            'blended_confidence': float(conf),
            'sim_confidence': None, # To be enriched from predicts.tex if available
            'odds': None,
            'edge_pct': None,
            'game': '',
            'projected_margin': None,
            'safety_margin': None,
            'market_spread': None,
            'actual_value': float(actual),
            'is_win': 1 if 'WON' in outcome.upper() else 0,
            'diff_val': float(diff_val),
            'diff_pct': float(diff_pct),
        })

    return records


def _enrich_from_predicts_tex(records: list[dict], tex_path: str) -> list[dict]:
    """Enrich records with odds, edge_pct, and game from predicts_report.tex."""
    if not os.path.exists(tex_path) or not records:
        return records

    with open(tex_path, 'r', encoding='utf-8') as fh:
        content = fh.read()

    row_re = re.compile(
        r'(\d+)\s*&\s*\\small\{(.+?)\}\s*&\s*\\small\{(.+?)\}\s*&\s*([\d\.]+)x\s*&\s*\+?([\d\.]+)\\%\s*&\s*\\cellcolor\{\w+\}(\d+)\\%',
        re.DOTALL
    )
    for m in row_re.finditer(content):
        _, pick_text, game_text, odds, edge, conf = m.groups()
        pick_text = pick_text.strip()
        game_text = game_text.strip()
        for r in records:
            if r.get('game'):
                continue
            if r['player_or_team'] in pick_text:
                r['game'] = game_text
                r['odds'] = float(odds)
                r['edge_pct'] = float(edge)
                r['blended_confidence'] = float(conf)
                # Note: We do NOT overwrite r['sim_confidence'] here if it was already set from analyze.tex/TXT
                break
            if 'Over' in pick_text and r['bet_type'] == 'total' and str(r['line']) in pick_text:
                r['game'] = game_text
                r['odds'] = float(odds)
                r['edge_pct'] = float(edge)
                r['blended_confidence'] = float(conf)
                break

    spread_re = re.compile(
        r'\\small\{([A-Za-z\s]+?)\s+([+-][\d\.]+)\s+\$\\rightarrow\$',
        re.IGNORECASE
    )
    for m in spread_re.finditer(content):
        team, market_spread = m.groups()
        team = team.strip()
        for r in records:
            if r['stat_category'] == 'SPREAD' and r['player_or_team'] in team:
                r['market_spread'] = float(market_spread)
                break

    return records


def _enrich_clv(records: list[dict], game_date: str) -> list[dict]:
    """Enrich records with CLV (Closing Line Value) from ESPN open/close odds data."""
    # We only fetch live odds if it's today's run
    today = datetime.now().strftime("%Y-%m-%d")
    if game_date != today:
        return records

    try:
        all_odds = fetch_espn_odds()
    except Exception as e:
        print(f"    [-] Could not fetch ESPN odds for CLV: {e}")
        return records

    if not all_odds:
        return records

    for r in records:
        game_label = r.get('game', '')
        if ' @ ' not in game_label:
            continue
            
        away_team, home_team = game_label.split(' @ ')
        match = match_espn_odds_to_game(all_odds, home_team.strip(), away_team.strip())
        if not match:
            continue
            
        _, odds_data = match
        stat = r.get('stat_category')
        direction = r.get('direction')
        
        clv = None
        if stat == 'SPREAD':
            open_val = odds_data.get('open_home_spread')
            close_val = odds_data.get('spreads', {}).get('home_spread')
            if open_val is not None and close_val is not None:
                # For SPREAD, 'line' is from the perspective of the bet side (r['line'])
                # So Opening - Closing works for both fav and dog
                clv = r.get('line', 0) - close_val if r.get('player_or_team') == odds_data['home_team'] else r.get('line', 0) - odds_data.get('spreads', {}).get('away_spread', -close_val)
        elif stat == 'TOTAL':
            open_val = odds_data.get('open_total')
            close_val = odds_data.get('totals', {}).get('total')
            if open_val is not None and close_val is not None:
                if direction == 'OVER':
                    clv = close_val - open_val
                elif direction == 'UNDER':
                    clv = open_val - close_val
        
        if clv is not None:
            r['clv'] = round(clv, 2)
            
    return records


def _parse_txt_valuable_picks(txt_path: str) -> list[dict]:
    """Parse VALUABLE picks (props, spreads, totals) from a raw predicts_YYYY-MM-DD.txt file."""
    if not os.path.exists(txt_path):
        print(f"  [-] File not found: {txt_path}")
        return []
        
    records = []
    with open(txt_path, 'r', encoding='utf-8') as fh:
        content = fh.read()
        
    # Split by matchup to help context
    matchups = re.split(r'={10,}', content)
    
    stat_map = {'points': 'POINTS', 'rebounds': 'REBOUNDS', 'assists': 'ASSISTS'}

    for section in matchups:
        # Try to find matchup name
        game_label = ""
        m_match = re.search(r'MATCHUP:\s*(.+)', section)
        if m_match:
            game_label = m_match.group(1).strip()

        # 1. Player Props
        # Example: Rudy Gobert -> 10 PTS (O/U 12.5 rebounds VALUABLE: Under (94% conf, 43.5% edge))
        prop_container_re = re.compile(r'^\s*([A-Za-z\s\.\'-]+?)\s*->\s*\d+\s+PTS.*?(\(O/U.*\))', re.MULTILINE)
        for m in prop_container_re.finditer(section):
            player = m.group(1).strip()
            props_str = m.group(2)
            
            prop_re = re.compile(r'O/U\s+([\d\.]+)\s+(\w+)\s+VALUABLE:\s+(Over|Under)\s+\((\d+)%\s+conf(?:\s+\(raw:\s+([\d\.]+)%\))?(?:,\s+([\d\.]+)%\s+edge)?\)')
            for pm in prop_re.finditer(props_str):
                line, stat_raw, direction, conf, raw_conf, edge = pm.groups()
                stat = stat_map.get(stat_raw.lower(), stat_raw.upper())
                records.append({
                    'player_or_team': player,
                    'stat_category': stat,
                    'bet_type': 'prop',
                    'direction': direction.upper(),
                    'line': float(line),
                    'sim_confidence': float(conf),
                    'raw_sim_freq': float(raw_conf)/100.0 if raw_conf else None,
                    'blended_confidence': float(conf),
                    'edge_pct': float(edge) if edge else None,
                    'game': game_label,
                })

        # 2. Spreads
        # -> HOME SPREAD: LAL -11.5 VALUABLE: Cover (95% conf, 45.3% edge)
        spread_re = re.compile(r'->\s+(?:HOME|AWAY)\s+SPREAD:\s+([A-Za-z\s]+)\s+([+-]?[\d\.]+)\s+VALUABLE:\s+(Cover)\s+\((\d+)%\s+conf(?:\s+\(raw:\s+([\d\.]+)%\))?(?:,\s+([\d\.]+)%\s+edge)?\)', re.IGNORECASE)
        for m in spread_re.finditer(section):
            team, line, status, conf, raw_conf, edge = m.groups()
            records.append({
                'player_or_team': team.strip(),
                'stat_category': 'SPREAD',
                'bet_type': 'spread',
                'direction': 'COVER',
                'line': float(line),
                'sim_confidence': float(conf),
                'raw_sim_freq': float(raw_conf)/100.0 if raw_conf else None,
                'blended_confidence': float(conf),
                'edge_pct': float(edge) if edge else None,
                'game': game_label,
            })

        # 3. Totals
        # -> GAME TOTAL: O/U 222.5 VALUABLE: Under (79% conf, 28.7% edge)
        total_re = re.compile(r'->\s+GAME\s+TOTAL:\s+O/U\s+([\d\.]+)\s+VALUABLE:\s+(Over|Under)\s+\((\d+)%\s+conf(?:\s+\(raw:\s+([\d\.]+)%\))?(?:,\s+([\d\.]+)%\s+edge)?\)', re.IGNORECASE)
        for m in total_re.finditer(section):
            line, direction, conf, raw_conf, edge = m.groups()
            records.append({
                'player_or_team': 'Game Total',
                'stat_category': 'TOTAL',
                'bet_type': 'total',
                'direction': direction.upper(),
                'line': float(line),
                'sim_confidence': float(conf),
                'raw_sim_freq': float(raw_conf)/100.0 if raw_conf else None,
                'blended_confidence': float(conf),
                'edge_pct': float(edge) if edge else None,
                'game': game_label,
            })
            
    return records


def _parse_results_txt(results_path: str) -> list[dict]:
    """
    Parse yesterday_results.txt into game_results.
    Returns list of {home_team, away_team, home_score, away_score}.
    """
    if not os.path.exists(results_path):
        return []

    game_outcomes = []
    with open(results_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            # MATCHUP: BOS vs. CHA    89 - 118
            m = re.match(r'MATCHUP:\s+(\w+)\s+vs\.\s+(\w+)\s+(\d+)\s+-\s+(\d+)', line)
            if m:
                t1, t2, s1, s2 = m.groups()
                game_outcomes.append({
                    'home_team': t1,
                    'away_team': t2,
                    'home_score': int(s1),
                    'away_score': int(s2),
                })
    return game_outcomes


# ── DB Writers ────────────────────────────────────────────────────────────────

def ingest_picks(picks: list[dict], game_date: str, source_file: str):
    """Insert picks into daily_picks. Skips duplicates (same date+player+stat+direction+line)."""
    if not picks:
        return 0

    conn = _get_conn()
    cur = conn.cursor()
    inserted = 0
    for p in picks:
        cur.execute("""
            SELECT id FROM daily_picks
            WHERE date=? AND player_or_team=? AND stat_category=? AND direction=? AND line=?
        """, (game_date, p['player_or_team'], p['stat_category'], p['direction'], p['line']))
        if cur.fetchone():
            continue  # already seeded

        cur.execute("""
            INSERT INTO daily_picks
                (date, bet_type, stat_category, player_or_team, game,
                 direction, line, odds, sim_confidence, raw_sim_freq, blended_confidence, edge_pct,
                 projected_margin, safety_margin, market_spread, source_file, clv, source_mode, ml_correction_applied)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            game_date,
            p['bet_type'],
            p['stat_category'],
            p['player_or_team'],
            p.get('game', ''),
            p['direction'],
            p.get('line'),
            p.get('odds'),
            p.get('sim_confidence'),
            p.get('raw_sim_freq'),
            p.get('blended_confidence'),
            p.get('edge_pct'),
            p.get('projected_margin'),
            p.get('safety_margin'),
            p.get('market_spread'),
            source_file,
            p.get('clv'),
            'synthetic_circular' if 'backtest' in source_file.lower() else 'live',
            p.get('ml_correction_applied', 'NONE')
        ))
        inserted += 1

    conn.commit()
    conn.close()
    return inserted


def ingest_game_results(game_outcomes: list[dict], game_date: str):
    """Insert game scores into game_results. Skips duplicates."""
    if not game_outcomes:
        return 0
    conn = _get_conn()
    cur = conn.cursor()
    inserted = 0
    for g in game_outcomes:
        cur.execute("""
            INSERT OR IGNORE INTO game_results (date, home_team, away_team, home_score, away_score)
            VALUES (?,?,?,?,?)
        """, (game_date, g['home_team'], g['away_team'], g['home_score'], g['away_score']))
        inserted += cur.rowcount
    conn.commit()
    conn.close()
    return inserted


def ingest_results(pick_outcomes: list[dict], game_date: str):
    """
    Match pick_outcomes against daily_picks for game_date and upsert pick_results.
    Returns number of picks resolved.
    """
    if not pick_outcomes:
        return 0
    conn = _get_conn()
    cur = conn.cursor()
    resolved = 0
    for r in pick_outcomes:
        cur.execute("""
            SELECT id FROM daily_picks
            WHERE date=? AND player_or_team=? AND stat_category=? AND direction=? AND line=?
        """, (game_date, r['player_or_team'], r['stat_category'], r['direction'], r['line']))
        row = cur.fetchone()
        if not row:
            continue
        pick_id = row[0]

        cur.execute("SELECT id FROM pick_results WHERE pick_id=?", (pick_id,))
        if cur.fetchone():
            continue  # already resolved

        cur.execute("""
            INSERT INTO pick_results (pick_id, actual_value, is_win, diff_val, diff_pct, resolved_at)
            VALUES (?,?,?,?,?,?)
        """, (pick_id, r['actual_value'], r['is_win'], r['diff_val'], r['diff_pct'], game_date))
        resolved += 1

    conn.commit()
    conn.close()
    return resolved


def ingest_shadow_picks(picks: list[dict]):
    """Insert a list of shadow picks into the shadow_picks table."""
    if not picks:
        return 0
        
    conn = _get_conn()
    cur = conn.cursor()
    inserted = 0
    for p in picks:
        cur.execute("""
            INSERT INTO shadow_picks 
                (game_date, player, stat_category, direction, line, projection, sim_confidence, edge_pct, ban_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            p.get('game_date'),
            p.get('player'),
            p.get('stat_category'),
            p.get('direction'),
            p.get('line'),
            p.get('projection'),
            p.get('sim_confidence'),
            p.get('edge_pct'),
            p.get('ban_reason')
        ))
        inserted += 1
    conn.commit()
    conn.close()
    return inserted

def ingest_shadow_pick(p: dict):
    """Convenience helper to insert a single shadow pick."""
    return ingest_shadow_picks([p])


# ── High-level Seed Functions ──────────────────────────────────────────────────
def seed_from_tex(predicts_tex, analyze_tex, results_txt, game_date):
    """Seed data from analyze.tex + predicts.tex (full enrichment mode)."""
    print(f"\n[+] Seeding {game_date} from TeX reports...")
    records = _parse_analyze_tex_full(analyze_tex)
    records = _enrich_from_predicts_tex(records, predicts_tex)
    records = _enrich_clv(records, game_date)

    n_inserted = ingest_picks(records, game_date, os.path.basename(predicts_tex))
    print(f"    Inserted {n_inserted} new picks into daily_picks (odds enriched)")

    n_resolved = ingest_results(records, game_date)
    print(f"    Resolved {n_resolved} pick outcomes")

    game_outcomes = _parse_results_txt(results_txt)
    n_games = ingest_game_results(game_outcomes, game_date)
    print(f"    Inserted {n_games} game results")


def seed_from_txt(predicts_txt, results_txt, game_date):
    """Seed data from a .txt prediction file (historical mode / simplified live)."""
    print(f"\n[+] Seeding {game_date} from .txt report...")
    records = _parse_txt_valuable_picks(predicts_txt)
    records = _enrich_clv(records, game_date)
    
    n_inserted = ingest_picks(records, game_date, os.path.basename(predicts_txt))
    print(f"    Inserted {n_inserted} new picks into daily_picks")

    if os.path.exists(results_txt):
        game_outcomes = _parse_results_txt(results_txt)
        n_games = ingest_game_results(game_outcomes, game_date)
        print(f"    Inserted {n_games} game results")


def seed_march3():
    """
    Seed March 3 data. March 3 was historical-mode only (no live odds),
    so we seed picks from the simulation txt but without odds/conf data.
    """
    print("\n[+] Seeding March 3 (simulation-only mode — no live odds)")
    picks_path = PREDICTS_MAR3
    if not os.path.exists(picks_path):
        print(f"  [-] {picks_path} not found, skipping March 3.")
        return

    records = _parse_txt_valuable_picks(picks_path)
    print(f"    Parsed {len(records)} valuable picks from {os.path.basename(picks_path)}")
    
    n_inserted = ingest_picks(records, '2026-03-03', os.path.basename(picks_path))
    print(f"    Inserted {n_inserted} simulation picks")

    results_path = os.path.join(MODEL_DIR, 'yesterday_results.txt') # We'll just map this to whatever result file if present
    # actually use the same date suffix for results
    results_path = os.path.join(MODEL_DIR, 'results_2026-03-03.txt')
    if os.path.exists(results_path):
        game_outcomes = _parse_results_txt(results_path)
        n_games = ingest_game_results(game_outcomes, '2026-03-03')
        print(f"    Inserted {n_games} game results for 2026-03-03")
    else:
        # Fallback to general yesterday_results.txt if it contains it
        results_path = os.path.join(MODEL_DIR, 'yesterday_results.txt')
        game_outcomes = _parse_results_txt(results_path)
        n_games = ingest_game_results(game_outcomes, '2026-03-03')
        print(f"    Inserted {n_games} game results extracted from {os.path.basename(results_path)}")


def seed_march4():
    """Seed March 4 (first full day with live odds + analyze results)."""
    predicts_tex = DEFAULT_PREDICTS
    analyze_tex  = DEFAULT_ANALYZE
    results_txt  = os.path.join(MODEL_DIR, 'yesterday_results.txt')
    seed_from_tex(predicts_tex, analyze_tex, results_txt, '2026-03-04')


def seed_all():
    """Seed all historical data starting from March 3."""
    seed_march3()
    seed_march4()


# ── Query Functions ───────────────────────────────────────────────────────────

def query_summary(days: int = None, bet_type: str = None, stat_category: str = None, verified_only: bool = True) -> dict:
    """
    Return aggregated win rate stats.

    Args:
        days:          Last N days (None = all time)
        bet_type:      Filter by 'prop' | 'spread' | 'total' (None = all)
        stat_category: Filter by 'POINTS' | 'REBOUNDS' etc. (None = all)
        verified_only: If True, exclude synthetic backtest projections (where line=0.0 or from bulk_backtest)
    """
    conn = _get_conn()
    cur = conn.cursor()

    where_clauses = ["pr.is_win IS NOT NULL"]
    params = []

    if verified_only:
        where_clauses.append("(dp.line != 0.0 AND dp.source_mode != 'synthetic_circular')")

    if days is not None:
        where_clauses.append("dp.date >= date('now', ?)")
        params.append(f'-{days} days')
    if bet_type:
        where_clauses.append("dp.bet_type = ?")
        params.append(bet_type)
    if stat_category:
        where_clauses.append("dp.stat_category = ?")
        params.append(stat_category.upper())

    where = " AND ".join(where_clauses)
    cur.execute(f"""
        SELECT
            COUNT(*)              AS total,
            SUM(pr.is_win)        AS wins,
            AVG(pr.is_win)*100    AS win_rate,
            AVG(NULLIF(dp.blended_confidence, 0)) AS avg_conf
        FROM daily_picks dp
        JOIN pick_results pr ON pr.pick_id = dp.id
        WHERE {where}
    """, params)
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else {}


def query_by_type(verified_only: bool = True) -> list[dict]:
    """Return win rates broken down by stat_category."""
    conn = _get_conn()
    cur = conn.cursor()
    where = "WHERE (dp.line != 0.0 AND dp.source_mode != 'synthetic_circular')" if verified_only else ""
    cur.execute(f"""
        SELECT
            dp.stat_category,
            COUNT(*)           AS total,
            SUM(pr.is_win)     AS wins,
            AVG(pr.is_win)*100 AS win_rate
        FROM daily_picks dp
        JOIN pick_results pr ON pr.pick_id = dp.id
        {where}
        GROUP BY dp.stat_category
        ORDER BY win_rate DESC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_db_date_range():
    """Returns (min_date, max_date) from the daily_picks table."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT MIN(date), MAX(date) FROM daily_picks")
    row = cur.fetchone()
    conn.close()
    return row if row else (None, None)


def query_confidence_bands() -> list[dict]:
    """Return win rate grouped by confidence band (deciles)."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            CASE
                WHEN dp.blended_confidence >= 90 THEN '≥90%'
                WHEN dp.blended_confidence >= 80 THEN '80-89%'
                WHEN dp.blended_confidence >= 70 THEN '70-79%'
                ELSE '<70%'
            END AS band,
            COUNT(*)           AS total,
            SUM(pr.is_win)     AS wins,
            AVG(pr.is_win)*100 AS win_rate
        FROM daily_picks dp
        JOIN pick_results pr ON pr.pick_id = dp.id
        WHERE dp.blended_confidence IS NOT NULL
        GROUP BY band
        ORDER BY dp.blended_confidence DESC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def print_summary(verified_only: bool = True):
    """Pretty-print the current performance summary."""
    overall = query_summary(verified_only=verified_only)
    d_min, d_max = get_db_date_range()

    if not overall or not overall.get('total'):
        label = "verified" if verified_only else "any"
        print(f"\n[!] No {label} picks found in range {d_min} to {d_max}.")
        return

    print("\n" + "="*55)
    print("  📊  AI BET TOOL — HISTORICAL PERFORMANCE TRACKER")
    if verified_only:
        print("      (VERIFIED PLAYER PROPS & REAL-TIME REPORTS)")
    else:
        print("      (FULL HISTORY: VERIFIED + SYNTHETIC BACKTESTS)")
    if d_min and d_max:
        print(f"      Period: {d_min} to {d_max}")
    print("="*55)
    
    label_picks = "Verified" if verified_only else "Total"
    print(f"  {label_picks} Picks Tracked: {int(overall['total'])}  |  "
          f"Wins: {int(overall['wins'])}  |  "
          f"Win Rate: {overall['win_rate']:.1f}%")
    avg_conf = overall['avg_conf'] if overall['avg_conf'] is not None else 0.0
    conf_label = "Avg Confidence (Verified)" if verified_only else "Avg Confidence (Overall)"
    print(f"  {conf_label}: {avg_conf:.1f}%")
    print()

    print(f"  By Category ({'Verified Only' if verified_only else 'Overall'}):")
    by_type = query_by_type(verified_only=verified_only)
    for row in by_type:
        bar = "█" * int(row['win_rate'] / 5)
        print(f"    {row['stat_category']:<12}  {row['wins']:.0f}/{row['total']}  "
              f"({row['win_rate']:.1f}%)  {bar}")

    print()
    print("  Confidence Band Accuracy:")
    bands = query_confidence_bands()
    for b in bands:
        bar = "█" * int(b['win_rate'] / 5)
        print(f"    {b['band']:<8}  {b['wins']:.0f}/{b['total']}  "
              f"({b['win_rate']:.1f}%)  {bar}")
    print("\n  [!] Projections listed as 'synthetic' are excluded from Win Rate.")
    print("      Run 'python3 -m etl.bet_tracker lines' for full bias ml.")
    print()


def query_lines_ml() -> list[dict]:
    """Return aggregated bias metrics (Actual - Projected) per category and direction."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            dp.stat_category,
            dp.direction,
            COUNT(*)                    AS total,
            SUM(
                CASE 
                    WHEN dp.stat_category = 'SPREAD' AND (pr.actual_value + dp.line = 0) THEN 1
                    WHEN dp.stat_category != 'SPREAD' AND (pr.actual_value = dp.line) THEN 1
                    ELSE 0
                END
            )                           AS pushes,
            SUM(
                CASE 
                    WHEN dp.stat_category = 'SPREAD' THEN (pr.actual_value + dp.line > 0)
                    WHEN dp.direction = 'OVER' OR dp.direction = 'COVER' THEN (pr.actual_value > dp.line)
                    WHEN dp.direction = 'UNDER' THEN (pr.actual_value < dp.line)
                    ELSE pr.is_win
                END
            )                           AS wins,
            CAST(SUM(
                CASE 
                    WHEN dp.stat_category = 'SPREAD' THEN (pr.actual_value + dp.line > 0)
                    WHEN dp.direction = 'OVER' OR dp.direction = 'COVER' THEN (pr.actual_value > dp.line)
                    WHEN dp.direction = 'UNDER' THEN (pr.actual_value < dp.line)
                    ELSE pr.is_win
                END
            ) AS FLOAT) * 100.0 / NULLIF(COUNT(*) - SUM(
                CASE 
                    WHEN dp.stat_category = 'SPREAD' AND (pr.actual_value + dp.line = 0) THEN 1
                    WHEN dp.stat_category != 'SPREAD' AND (pr.actual_value = dp.line) THEN 1
                    ELSE 0
                END
            ), 0)                       AS win_rate,
            AVG(pr.diff_val)            AS mean_bias,
            AVG(ABS(pr.diff_val))       AS mae
        FROM daily_picks dp
        JOIN pick_results pr ON pr.pick_id = dp.id
        WHERE pr.is_win IS NOT NULL
          AND dp.source_mode != 'synthetic_circular'
        GROUP BY dp.stat_category, dp.direction
        ORDER BY dp.stat_category, dp.direction
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def print_lines_summary():
    """Print the 'Super Analysis' report showing systematic model bias."""
    rows = query_lines_ml()
    if not rows:
        print("[!] No resolved data found for ml.")
        return

    print("\n" + "="*85)
    print("  🎯  AI BET TOOL — AGGREGATE HISTORICAL BIAS ANALYSIS ('LINES')")
    print("="*85)
    print(f"{'CATEGORY':<12} | {'DIR':<6} | {'N':<5} | {'P':<3} | {'WIN %':<8} | {'MEAN BIAS':<12} | {'MAE':<6}")
    print("-" * 85)

    # We'll group by category for cleaner output
    current_cat = None
    for r in rows:
        if r['stat_category'] != current_cat:
            if current_cat is not None:
                print("-" * 85)
            current_cat = r['stat_category']
            
        mb = r['mean_bias']
        bias_desc = "(PESSIMISTIC)" if mb > 0 else "(OPTIMISTIC)"
        if abs(mb) < 0.1: bias_desc = "(BALANCED)"
        
        # SPREAD bias is trickier to label as pessimistic/optimistic without side context,
        # but the MB still tells us which way the error leans.
        if r['stat_category'] in ['SPREAD', 'TOTAL']:
            bias_desc = ""

        print(f"{r['stat_category']:<12} | {r['direction']:<6} | {r['total']:<5} | {r['pushes']:<3} | "
              f"{r['win_rate']:>6.1f}% | {mb:>+10.2f} {bias_desc:<13} | {r['mae']:>5.1f}")
    
    print("="*85)
    print("  GUIDE:")
    print("  - MEAN BIAS > 0: Model is UNDER-PROJECTING (Reality > Model).")
    print("  - MEAN BIAS < 0: Model is OVER-PROJECTING (Model > Reality).")
    print("  - MAE: Mean Absolute Error (overall distance from reality).")
    print("="*85 + "\n")


def check_db():
    """Check record counts in DB by month and stat."""
    conn = _get_conn()
    cursor = conn.cursor()
    
    print("--- Daily Picks Counts by Month ---")
    try:
        cursor.execute("SELECT substr(date, 1, 7) as month, COUNT(*) FROM daily_picks GROUP BY month")
        for row in cursor.fetchall():
            print(f"{row[0]}: {row[1]}")
    except Exception as e:
        print(f"Error querying daily_picks: {e}")
    
    print("\n--- Summary of Stat Categories ---")
    try:
        cursor.execute("SELECT stat_category, COUNT(*) FROM daily_picks GROUP BY stat_category")
        for row in cursor.fetchall():
            print(f"{row[0]}: {row[1]}")
    except Exception as e:
        print(f"Error querying stat_category: {e}")
        
    conn.close()

def compare_bias():
    """Compare bias dynamically split by March and earlier."""
    conn = _get_conn()
    cur = conn.cursor()
    
    q = """
    SELECT 
        dp.stat_category, 
        dp.direction, 
        COUNT(*), 
        AVG(pr.is_win)*100, 
        AVG(pr.diff_val) 
    FROM daily_picks dp 
    JOIN pick_results pr ON pr.pick_id = dp.id 
    WHERE %s 
    GROUP BY 1, 2
    """
    
    print("=== MARCH BIAS ANALYSIS ===")
    cur.execute(q % "dp.date >= '2026-03-01'")
    for r in cur.fetchall():
        print(f"{r[0]:<10} | {r[1]:<6} | C: {r[2]:<4} | W%: {r[3]:>5.1f}% | MB: {r[4]:>+5.2f}")
        
    print("\n=== JAN/FEB BIAS ANALYSIS ===")
    cur.execute(q % "dp.date < '2026-03-01'")
    for r in cur.fetchall():
        print(f"{r[0]:<10} | {r[1]:<6} | C: {r[2]:<4} | W%: {r[3]:>5.1f}% | MB: {r[4]:>+5.2f}")
        
    conn.close()

# ── CLI Entry Point ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI Bet Tool — Tracker CLI')
    sub = parser.add_subparsers(dest='cmd')

    # seed
    seed_parser = sub.add_parser('seed', help='Seed historical data into the tracker')
    seed_parser.add_argument('--date', type=str, default=None,
                             help='Date to seed (YYYY-MM-DD). Use "all" to seed all history.')
    seed_parser.add_argument('--predicts', type=str, default=DEFAULT_PREDICTS)
    seed_parser.add_argument('--analyze', type=str, default=DEFAULT_ANALYZE)
    seed_parser.add_argument('--results', type=str,
                             default=os.path.join(MODEL_DIR, 'yesterday_results.txt'))

    # summary
    summary_parser = sub.add_parser('summary', help='Print performance summary')
    summary_parser.add_argument('--all', action='store_true', help='Include synthetic backtest picks in summary')
    
    sub.add_parser('bands',   help='Print confidence band accuracy')
    sub.add_parser('lines',   help='Print aggregate bias ml.(combined daily format)')
    sub.add_parser('check',   help='Check record counts in DB by month and stat')
    sub.add_parser('bias',    help='Print older bias comparison (Jan/Feb vs March)')

    args = parser.parse_args()

    if args.cmd == 'seed':
        if args.date == 'all' or args.date is None:
            seed_all()
        elif args.date == '2026-03-03':
            seed_march3()
        elif args.date == '2026-03-04':
            seed_march4()
        else:
            # Dynamic filename detection for historical dates
            month = get_month_folder(args.date)
            pred_path = args.predicts
            if pred_path == DEFAULT_PREDICTS:
                # Check root, then check predictions/{month}/
                specific_options = [
                    os.path.join(MODEL_DIR, f'predicts_report_{args.date}.tex'),
                    os.path.join(MODEL_DIR, 'predictions', month, f'predicts_report_{args.date}.tex'),
                    os.path.join(MODEL_DIR, 'predictions', month, f'predicts_{args.date}.txt') # fallback to txt if needed
                ]
                for opt in specific_options:
                    if os.path.exists(opt):
                        pred_path = opt
                        break
                    
            analyze_path = args.analyze
            if analyze_path == DEFAULT_ANALYZE:
                # Check root, then check analyze/{month}/
                specific_options = [
                    os.path.join(MODEL_DIR, f'analyze_{args.date}.tex'),
                    os.path.join(MODEL_DIR, 'analyze', month, f'analyze_{args.date}.tex')
                ]
                for opt in specific_options:
                    if os.path.exists(opt):
                        analyze_path = opt
                        break
                    
            res_path = args.results
            if os.path.basename(res_path) == 'yesterday_results.txt':
                # Check root, then check results/{month}/
                specific_options = [
                    os.path.join(MODEL_DIR, f'results_{args.date}.txt'),
                    os.path.join(MODEL_DIR, f'yesterday_results_{args.date}.txt'),
                    os.path.join(MODEL_DIR, 'results', month, f'yesterday_results_{args.date}.txt')
                ]
                for opt in specific_options:
                    if os.path.exists(opt):
                        res_path = opt
                        break
                    
            if pred_path.endswith('.txt'):
                seed_from_txt(pred_path, res_path, args.date)
            else:
                seed_from_tex(pred_path, analyze_path, res_path, args.date)
    elif args.cmd == 'summary':
        print_summary(verified_only=(not args.all))
    elif args.cmd == 'lines':
        print_lines_summary()
    elif args.cmd == 'bands':
        bands = query_confidence_bands()
        for b in bands:
            print(f"  {b['band']}: {b['wins']:.0f}/{b['total']} ({b['win_rate']:.1f}%)")
    elif args.cmd == 'check':
        check_db()
    elif args.cmd == 'bias':
        compare_bias()
    else:
        parser.print_help()
