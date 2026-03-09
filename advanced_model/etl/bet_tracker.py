"""
Bet Tracker ETL
---------------
Reads predicts_report.tex (picks) and analyze.tex (outcomes) and
persists them into bet_tracker.db for long-term performance tracking.

Usage
-----
# Seed yesterday's data (run manually after each day's analysis)
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
from datetime import date as date_type

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
        'REB': 'REBOUNDS', 'POI': 'POINTS', 'AST': 'ASSISTS',
        'TOT': 'TOTAL', 'SPR': 'SPREAD',
        'POINTS': 'POINTS', 'REBOUNDS': 'REBOUNDS', 'ASSISTS': 'ASSISTS',
        'TOTAL': 'TOTAL', 'SPREAD': 'SPREAD',
    }
    bet_type_map = {
        'POINTS': 'prop', 'REBOUNDS': 'prop', 'ASSISTS': 'prop',
        'TOTAL': 'total', 'SPREAD': 'spread',
    }

    row_re = re.compile(
        r'^([A-Za-z\s\.\'\-]+?)\s*&\s*(\d+)\\%\s*&\s*(OVER|UNDER|COVER)\s+([\d\.]+)\s+(\w+)'
        r'\s*&\s*([\-\d\.]+)\s*&\s*([+\-][\d\.]+)\s*&\s*([+\-][\d\.]+)\\%\s*&\s*\\textcolor\{[^}]+\}\{(WON|LOST)',
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
            'sim_confidence': float(conf),
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


def _parse_txt_valuable_picks(txt_path: str) -> list[dict]:
    """Parse VALUABLE picks from a raw predicts_YYYY-MM-DD.txt file (historical mode)."""
    if not os.path.exists(txt_path):
        print(f"  [-] File not found: {txt_path}")
        return []
        
    records = []
    with open(txt_path, 'r', encoding='utf-8') as fh:
        content = fh.read()
        
    # Example line:
    # Rudy Gobert          -> 10 PTS, 8 REB, 1 AST (O/U 11.5 points ignore, O/U 12.5 rebounds VALUABLE: Under (94% conf, 43.5% edge))
    line_re = re.compile(r'^\s*([A-Za-z\s\.\'-]+?)\s*->\s*\d+ PTS.*?(\(O/U.*\))', re.MULTILINE)
    stat_map = {'points': 'POINTS', 'rebounds': 'REBOUNDS', 'assists': 'ASSISTS'}
    
    for m in line_re.finditer(content):
        player = m.group(1).strip()
        props_str = m.group(2)
        
        prop_re = re.compile(r'O/U\s+([\d\.]+)\s+(\w+)\s+VALUABLE:\s+(Over|Under)\s+\((\d+)%\s+conf')
        for pm in prop_re.finditer(props_str):
            line, stat_raw, direction, conf = pm.groups()
            stat = stat_map.get(stat_raw.lower(), stat_raw.upper())
            records.append({
                'player_or_team': player,
                'stat_category': stat,
                'bet_type': 'prop',
                'direction': direction.upper(),
                'line': float(line),
                'sim_confidence': float(conf),
                'blended_confidence': float(conf),
                'odds': None,
                'edge_pct': None,
                'game': '',
                'projected_margin': None,
                'safety_margin': None,
                'market_spread': None,
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
                 direction, line, odds, sim_confidence, blended_confidence, edge_pct,
                 projected_margin, safety_margin, market_spread, source_file)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            p.get('blended_confidence'),
            p.get('edge_pct'),
            p.get('projected_margin'),
            p.get('safety_margin'),
            p.get('market_spread'),
            source_file,
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


# ── High-level Seed Functions ──────────────────────────────────────────────────


    n_inserted = ingest_picks(records, game_date, os.path.basename(predicts_tex))
    print(f"    Inserted {n_inserted} new picks into daily_picks (odds enriched)")

    n_resolved = ingest_results(records, game_date)
    print(f"    Resolved {n_resolved} pick outcomes")

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

def query_summary(days: int = None, bet_type: str = None, stat_category: str = None) -> dict:
    """
    Return aggregated win rate stats.

    Args:
        days:          Last N days (None = all time)
        bet_type:      Filter by 'prop' | 'spread' | 'total' (None = all)
        stat_category: Filter by 'POINTS' | 'REBOUNDS' etc. (None = all)
    """
    conn = _get_conn()
    cur = conn.cursor()

    where_clauses = ["pr.is_win IS NOT NULL"]
    params = []

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
            AVG(dp.blended_confidence) AS avg_conf
        FROM daily_picks dp
        JOIN pick_results pr ON pr.pick_id = dp.id
        WHERE {where}
    """, params)
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else {}


def query_by_type() -> list[dict]:
    """Return win rates broken down by stat_category."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            dp.stat_category,
            COUNT(*)           AS total,
            SUM(pr.is_win)     AS wins,
            AVG(pr.is_win)*100 AS win_rate
        FROM daily_picks dp
        JOIN pick_results pr ON pr.pick_id = dp.id
        GROUP BY dp.stat_category
        ORDER BY win_rate DESC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


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


def print_summary():
    """Pretty-print the current performance summary."""
    overall = query_summary()
    if not overall or not overall.get('total'):
        print("[!] No resolved picks in the tracker yet.")
        return

    print("\n" + "="*55)
    print("  📊  AI BET TOOL — HISTORICAL PERFORMANCE TRACKER")
    print("="*55)
    print(f"  Total Picks Tracked: {int(overall['total'])}  |  "
          f"Wins: {int(overall['wins'])}  |  "
          f"Win Rate: {overall['win_rate']:.1f}%")
    avg_conf = overall['avg_conf'] if overall['avg_conf'] is not None else 0.0
    print(f"  Avg Confidence Declared: {avg_conf:.1f}%")
    print()

    print("  By Category:")
    by_type = query_by_type()
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
    print()


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
    sub.add_parser('summary', help='Print performance summary')
    sub.add_parser('bands',   help='Print confidence band accuracy')

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
            pred_path = args.predicts
            if pred_path == DEFAULT_PREDICTS:
                specific_pred = os.path.join(MODEL_DIR, f'predicts_report_{args.date}.tex')
                if os.path.exists(specific_pred):
                    pred_path = specific_pred
                    
            analyze_path = args.analyze
            if analyze_path == DEFAULT_ANALYZE:
                specific_analyze = os.path.join(MODEL_DIR, f'analyze_{args.date}.tex')
                if os.path.exists(specific_analyze):
                    analyze_path = specific_analyze
                    
            res_path = args.results
            if os.path.basename(res_path) == 'yesterday_results.txt':
                specific_res = os.path.join(MODEL_DIR, f'results_{args.date}.txt')
                if os.path.exists(specific_res):
                    res_path = specific_res
                    
            seed_from_tex(pred_path, analyze_path, res_path, args.date)
    elif args.cmd == 'summary':
        print_summary()
    elif args.cmd == 'bands':
        bands = query_confidence_bands()
        for b in bands:
            print(f"  {b['band']}: {b['wins']:.0f}/{b['total']} ({b['win_rate']:.1f}%)")
    else:
        parser.print_help()
