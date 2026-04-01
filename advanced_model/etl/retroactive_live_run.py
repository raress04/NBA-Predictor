import pandas as pd
import numpy as np
import sqlite3
import os
import sys
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from simulator.parlay_builder import compute_prop_edge, MAX_PROP_CONFIDENCE, EDGE_THRESHOLDS

warnings.filterwarnings("ignore")

from simulator.matrix_builder import get_player_stats_matrix, get_team_pace
from simulator.markov_engine import MarkovSimulator, compute_posterior_confidence
from simulator.synergy_tracker import compute_matchup_multiplier
from simulator.parlay_builder import (
    EDGE_THRESHOLDS, compute_prop_edge,
    MAX_PROP_CONFIDENCE, BANNED_COMBINATIONS
)
from config.settings import PROJECTION_TIERS, TIERED_BIAS_CORRECTIONS, get_prior

MODEL_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NBA_DB     = os.path.join(MODEL_DIR, 'database', 'nba_data.db')
ODDS_DB    = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')     # READ-ONLY
TRACKER_DB = os.path.join(MODEL_DIR, 'database', 'retrospective_db.db') # WRITE

TEAM_ID_TO_NAME = {
    1610612737: "Atlanta Hawks",        1610612738: "Boston Celtics",
    1610612739: "Cleveland Cavaliers",  1610612740: "New Orleans Pelicans",
    1610612741: "Chicago Bulls",        1610612742: "Dallas Mavericks",
    1610612743: "Denver Nuggets",       1610612744: "Golden State Warriors",
    1610612745: "Houston Rockets",      1610612746: "LA Clippers",
    1610612747: "Los Angeles Lakers",   1610612748: "Miami Heat",
    1610612749: "Milwaukee Bucks",      1610612750: "Minnesota Timberwolves",
    1610612751: "Brooklyn Nets",        1610612752: "New York Knicks",
    1610612753: "Orlando Magic",        1610612754: "Indiana Pacers",
    1610612755: "Philadelphia 76ers",   1610612756: "Phoenix Suns",
    1610612757: "Portland Trail Blazers", 1610612758: "Sacramento Kings",
    1610612759: "San Antonio Spurs",    1610612760: "Oklahoma City Thunder",
    1610612761: "Toronto Raptors",      1610612762: "Utah Jazz",
    1610612763: "Memphis Grizzlies",    1610612764: "Washington Wizards",
    1610612765: "Detroit Pistons",      1610612766: "Charlotte Hornets",
}

# ── Injury Matrix Constants (C.1 — mirrors live_scraper.py) ───────────────────
INJURY_TIER_PPG_STAR      = 20.0   # avg PPG >= 20 → Tier 1 Star
INJURY_TIER_MIN_STARTER   = 28.0   # avg MIN >= 28 → Tier 2 Starter
INJURY_TIER_MIN_ROTATION  = 15.0   # avg MIN >= 15 → Tier 3 Rotation

INJURY_PENALTIES = {
    1: 0.12,   # Tier 1 Star missing: 12% shooting penalty
    2: 0.07,   # Tier 2 Starter missing: 7% penalty
    3: 0.03,   # Tier 3 Rotation missing: 3% penalty
}
SYNERGY_COLLAPSE_THRESHOLD   = 3      # 3+ core (Tier 1 or 2) players missing
SYNERGY_COLLAPSE_MULTIPLIER  = 0.85   # 0.85x fg% when collapse fires
MAX_CUMULATIVE_PENALTY       = 0.30   # Cap: never reduce fg% by more than 30%

# ── Blowout Constants (D.1) ───────────────────────────────────────────────────
BLOWOUT_THRESHOLD = 11.5   # spread magnitude triggering blowout shift


def get_historical_lines(date_str):
    """Returns { (player, CAT): {'line': X, 'over_odds': Y, 'under_odds': Z} }"""
    conn = sqlite3.connect(ODDS_DB)
    try:
        df = pd.read_sql_query("""
            SELECT player_name, stat_category, line,
                   COALESCE(over_odds,  1.909) AS over_odds,
                   COALESCE(under_odds, 1.909) AS under_odds
            FROM historical_odds_cache
            WHERE date = ?
        """, conn, params=(date_str,))
    except Exception:
        conn.close()
        return {}
    conn.close()

    lines = {}
    for _, row in df.iterrows():
        cat = row['stat_category'].upper()
        if cat in ('POINTS', 'REBOUNDS', 'ASSISTS'):
            lines[(row['player_name'], cat)] = {
                'line':       float(row['line']),
                'over_odds':  float(row['over_odds']),
                'under_odds': float(row['under_odds']),
            }
    return lines

# ── Injury Matrix Helpers (C.2 / C.3 / C.4) ──────────────────────────────────

def classify_injury_tier(player_name: str, season_stats: dict) -> int:
    """
    1 = Star (avg_pts >= INJURY_TIER_PPG_STAR)
    2 = Starter (avg_min >= INJURY_TIER_MIN_STARTER)
    3 = Rotation (avg_min >= INJURY_TIER_MIN_ROTATION)
    0 = Bench/irrelevant — no penalty
    Default if unknown: 2 (conservative Starter)
    """
    key = player_name.strip().lower()
    if key not in season_stats:
        return 2
    stats = season_stats[key]
    if stats.get('avg_pts', 0.0) >= INJURY_TIER_PPG_STAR:
        return 1
    elif stats.get('avg_min', 0.0) >= INJURY_TIER_MIN_STARTER:
        return 2
    elif stats.get('avg_min', 0.0) >= INJURY_TIER_MIN_ROTATION:
        return 3
    return 0


def load_historical_injuries(date_str: str, conn_nba, season_stats: dict) -> dict:
    """
    Returns {team_name: [{'player': str, 'tier': int}, ...]}
    Only 'Out' players trigger penalty — matches live_scraper.py behavior.
    """
    try:
        rows = conn_nba.execute(
            "SELECT player, team, status FROM historical_injuries WHERE game_date = ?",
            (date_str,)
        ).fetchall()
    except Exception:
        return {}   # Table may not exist yet — soft fail

    by_team = {}
    for player, team, status in rows:
        if status != 'Out':
            continue
        tier = classify_injury_tier(player, season_stats)
        if tier == 0:
            continue
        by_team.setdefault(team, []).append({'player': player, 'tier': tier})
    return by_team


def compute_team_injury_penalty(injured_players: list) -> tuple:
    """
    Returns (penalty_fraction: float, synergy_collapse: bool)
    penalty_fraction: scale team fg% by (1 - penalty_fraction)
    """
    if not injured_players:
        return 0.0, False
    core_missing = sum(1 for p in injured_players if p['tier'] in (1, 2))
    if core_missing >= SYNERGY_COLLAPSE_THRESHOLD:
        return (1.0 - SYNERGY_COLLAPSE_MULTIPLIER), True
    total_penalty = sum(INJURY_PENALTIES.get(p['tier'], 0.0) for p in injured_players)
    return min(total_penalty, MAX_CUMULATIVE_PENALTY), False


# ── Blowout Shift Helper (D.1) ────────────────────────────────────────────────

def apply_blowout_shift(roster: dict, shift_pct: float = 0.15):
    """
    Shifts avg_fga + avg_reb from starters (avg_min>=28) to bench (8<=avg_min<28).
    Modifies roster dict in-place. Called ONLY on the underdog team.
    """
    starters = [p for p, m in roster.items() if m.get('avg_min', 0) >= INJURY_TIER_MIN_STARTER]
    bench    = [p for p, m in roster.items()
                if 8.0 <= m.get('avg_min', 0) < INJURY_TIER_MIN_STARTER]
    if not starters or not bench:
        return

    pool_fga = sum(roster[p].get('avg_fga', 0) * shift_pct for p in starters)
    pool_reb = sum(roster[p].get('avg_reb', 0) * shift_pct for p in starters)

    for p in starters:
        roster[p]['avg_fga'] = round(roster[p].get('avg_fga', 0) * (1 - shift_pct), 3)
        roster[p]['avg_reb'] = round(roster[p].get('avg_reb', 0) * (1 - shift_pct), 3)

    n_bench = len(bench)
    for p in bench:
        roster[p]['avg_fga'] = round(roster[p].get('avg_fga', 0) + pool_fga / n_bench, 3)
        roster[p]['avg_reb'] = round(roster[p].get('avg_reb', 0) + pool_reb / n_bench, 3)



# ── DB Init ───────────────────────────────────────────────────────────

def init_retro_db():
    conn = sqlite3.connect(TRACKER_DB)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS projection_outcomes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date       TEXT,
            player          TEXT,
            category        TEXT,
            projected_value REAL,
            bookmaker_line  REAL,
            actual_value    REAL,
            over_odds       REAL,
            under_odds      REAL,
            model_bias      REAL,
            source_mode     TEXT DEFAULT 'retroactive_live',
            source_file     TEXT,
            UNIQUE(game_date, player, category, source_file)
        )
    """)
    conn.commit()
    conn.close()


def insert_projection_outcome(conn_bet, game_date, player, category,
                              projected_value, bookmaker_line, actual_value,
                              over_odds, under_odds, reality_date=None, original_iso=None):
    if actual_value is None or bookmaker_line is None:
        return
    model_bias = round(actual_value - projected_value, 4)
    
    # Use reality_date for storage if provided (Fixes 1-day shift)
    target_date = reality_date if reality_date else game_date
    target_iso  = original_iso if original_iso else game_date

    conn_bet.execute("""
        INSERT OR IGNORE INTO projection_outcomes (
            game_date, player, category, projected_value,
            bookmaker_line, actual_value, over_odds, under_odds, model_bias, source_file
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (target_iso, player, category, projected_value, 
          bookmaker_line, actual_value, over_odds, under_odds,
          model_bias, f"predicts_{target_date}.txt"))
    # Commits are now handled in batch by the orchestrator for stability.


def true_edge_pct(conf_pct: float, decimal_odds: float) -> float:
    """(model_prob - implied_prob) * 100"""
    model_prob    = conf_pct / 100.0
    implied_prob  = 1.0 / decimal_odds if decimal_odds > 1 else 0.5
    return (model_prob - implied_prob) * 100.0


def build_retro_parlays(all_picks: list) -> list:
    """
    Given a list of pick dicts (from all games that day), return
    formatted parlay section lines identical to the live output.

    Pick dict shape:
        player, cat, direction, line, proj, conf, edge_pct,
        over_odds, under_odds, matchup
    """
    if not all_picks:
        return []

    # Filter banned combinations
    eligible = [
        p for p in all_picks
        if (p['cat'].upper(), p['direction'].upper())
        not in BANNED_COMBINATIONS
    ]

    if not eligible:
        return []

    # Score = true_edge (using real odds) × confidence
    def score(p):
        odds     = p['over_odds'] if p['direction'] == 'Over' else p['under_odds']
        t_edge   = true_edge_pct(p['conf'], odds)
        return t_edge * p['conf'] / 100.0

    ranked = sorted(eligible, key=score, reverse=True)

    def fmt_pick(i, p):
        odds_dec = p['over_odds'] if p['direction'] == 'Over' else p['under_odds']
        t_edge   = true_edge_pct(p['conf'], odds_dec)
        return (f"  {i}. {p['player']} {p['direction']} {p['line']} {p['cat']} "
                f"({p['conf']:.0f}% conf, {t_edge:.1f}% edge) | {p['matchup']}")

    lines = []

    # ── TOP 3-LEG PARLAY ──
    top3 = ranked[:3]
    if len(top3) >= 2:
        lines.append("")
        lines.append("=== TOP 3-LEG PARLAY ===")
        for i, p in enumerate(top3, 1):
            lines.append(fmt_pick(i, p))

    # ── SYSTEM PARLAY (4 picks, play any 3/4) ──
    top4 = ranked[:4]
    if len(top4) >= 3:
        lines.append("")
        lines.append(f"=== SYSTEM PARLAY ({len(top4)} PICKS, 3/{len(top4)}) ===")
        for i, p in enumerate(top4, 1):
            lines.append(fmt_pick(i, p))

    return lines


def save_report(date_str, report_lines, total_valuable):
    month      = date_str[4:6]
    month_name = datetime.strptime(month, "%m").strftime("%B").lower()
    out_dir    = os.path.join(MODEL_DIR, 'new_gen_prediction', month_name)
    os.makedirs(out_dir, exist_ok=True)
    filename   = os.path.join(out_dir,
                    f"predicts_{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}.txt")
    with open(filename, 'w') as f:
        f.write("\n".join(report_lines))
        f.write(f"\n\nTotal VALUABLE picks today: {total_valuable}\n")
    print(f"  [+] Saved: {filename} ({total_valuable} valuable picks)")


# ── Parallel Worker ───────────────────────────────────────────────────

def process_retroactive_game(gid, iso_date, date_str, t1_id, t1_name,
                              t2_id, t2_name, rosters, lines, g_data,
                              injuries_by_team=None, spread_map=None):
    """Worker — runs 5000 Markov sims + injury matrix + blowout, returns picks + DB rows."""
    import io, contextlib
    from simulator.matrix_builder import get_team_pace
    from simulator.markov_engine import MarkovSimulator, compute_posterior_confidence
    from simulator.synergy_tracker import compute_matchup_multiplier
    from simulator.parlay_builder import compute_prop_edge, MAX_PROP_CONFIDENCE
    from config.settings import PROJECTION_TIERS, TIERED_BIAS_CORRECTIONS, get_prior, is_allowed

    if injuries_by_team is None:
        injuries_by_team = {}
    if spread_map is None:
        spread_map = {}

    buf = io.StringIO()
    payload = {'logs': '', 'rows': [], 'report': [], 'val_count': 0, 'picks': []}

    with contextlib.redirect_stdout(buf):
        print(f"  [+] {t1_name} vs {t2_name}...")

        # ── C.6: Apply Injury Matrix penalties before synergy/simulation ─────────
        synergy_collapse = False
        for tid, tname in [(t1_id, t1_name), (t2_id, t2_name)]:
            team_injured = injuries_by_team.get(tname, [])
            penalty_frac, is_collapse = compute_team_injury_penalty(team_injured)
            if is_collapse:
                synergy_collapse = True
                print(f"    [SYNERGY COLLAPSE] {tname}: {len(team_injured)} core players Out")
            if penalty_frac > 0:
                for mx in rosters[tid].values():
                    mx['fg2_pct'] = mx.get('fg2_pct', 0.45) * (1.0 - penalty_frac)
                    mx['fg3_pct'] = mx.get('fg3_pct', 0.35) * (1.0 - penalty_frac)

        # ── D.2: Apply Blowout shift to underdog roster ───────────────────────
        spread_line = spread_map.get((t1_id, t2_id), spread_map.get((t2_id, t1_id), 0.0))
        if abs(spread_line) >= BLOWOUT_THRESHOLD:
            underdog_id = t1_id if spread_line > 0 else t2_id
            uname = t1_name if spread_line > 0 else t2_name
            apply_blowout_shift(rosters[underdog_id], shift_pct=0.15)
            print(f"    [BLOWOUT SHIFT] spread={spread_line:+.1f}, underdog={uname}")

        for p, m in rosters[t1_id].items():
            syn = compute_matchup_multiplier(p, t2_name)
            m['fg2_pct'] = m.get('fg2_pct', 0.45) * syn
            m['fg3_pct'] = m.get('fg3_pct', 0.35) * syn
        for p, m in rosters[t2_id].items():
            syn = compute_matchup_multiplier(p, t1_name)
            m['fg2_pct'] = m.get('fg2_pct', 0.45) * syn
            m['fg3_pct'] = m.get('fg3_pct', 0.35) * syn

        pace = (get_team_pace(t1_id, before_date=iso_date) +
                get_team_pace(t2_id, before_date=iso_date)) / 2

        sim     = MarkovSimulator(rosters[t1_id], rosters[t2_id])
        N_SIMS  = 5000  # A.2: aligned with live_scraper.py (was 1000)
        agg     = {}
        for _ in range(N_SIMS):
            r = sim.run_full_game(pace=pace)
            for pn, ps in r['player_stats'].items():
                if pn not in agg:
                    agg[pn] = {'POINTS': [], 'REBOUNDS': [], 'ASSISTS': []}
                agg[pn]['POINTS'].append(ps['PTS'])
                agg[pn]['REBOUNDS'].append(ps['REB'])
                agg[pn]['ASSISTS'].append(ps['AST'])

        payload['report'].append(f"\nMATCHUP: {t1_name} vs. {t2_name}\n" + "-" * 30)
        cat_col  = {'POINTS': 'pts', 'REBOUNDS': 'reb', 'ASSISTS': 'ast'}
        stat_key = {'POINTS': 'points', 'REBOUNDS': 'rebounds', 'ASSISTS': 'assists'}
        matchup  = f"{t1_name} vs. {t2_name}"

        for tid in [t1_id, t2_id]:
            for _, row in g_data[g_data['team_id'] == tid].iterrows():
                pn = row['player_name']
                if pn not in agg:
                    continue
                for cat in ['POINTS', 'REBOUNDS', 'ASSISTS']:
                    dist  = sorted(agg[pn][cat])
                    raw_p = float(np.mean(dist))
                    tier  = "starter" if raw_p >= PROJECTION_TIERS.get(cat, 10) else "bench"
                    corr_p = raw_p + TIERED_BIAS_CORRECTIONS.get((cat, tier), 0.0)

                    line_data = lines.get((pn, cat))
                    line_val  = line_data['line'] if line_data else None
                    act       = row[cat_col[cat]] if cat_col[cat] in row else None
                    ov_odds   = line_data.get('over_odds') if line_data else None
                    un_odds   = line_data.get('under_odds') if line_data else None

                    payload['rows'].append((iso_date, pn, cat, raw_p, line_val, act, ov_odds, un_odds))

                    if line_data is None:
                        continue

                    edge_data = compute_prop_edge(dist, line_data['line'], stat_key[cat])
                    if not edge_data['valuable']:
                        continue

                    direction = edge_data['direction']

                    # ── Gate 1: Banned combinations ──────────
                    if not is_allowed(cat, direction):
                        continue

                    # ── Gate 2: Retro-Calibration (Aggressive) ──────────────
                    # Goldilocks Zone: 2.0 gap (Points), 1.8 gap (Rebounds).
                    # This ensures we get high-signal picks without empty days.
                    RETRO_THRESHOLDS = {
                        'points':   {'min_prob': 72.0, 'min_edge': 18.0, 'min_gap': 2.0},
                        'rebounds': {'min_prob': 74.0, 'min_edge': 22.0, 'min_gap': 1.8},
                        'assists':  {'min_prob': 75.0, 'min_edge': 20.0, 'min_gap': 1.5},
                    }
                    rt = RETRO_THRESHOLDS.get(stat_key[cat], RETRO_THRESHOLDS['points'])

                    proj_gap = abs(corr_p - line_data['line'])
                    if proj_gap < rt['min_gap']:
                        continue

                    p_prior = get_prior(stat_key[cat], direction)
                    target  = (edge_data['over_count']
                               if direction == 'Over'
                               else N_SIMS - edge_data['over_count'])
                    
                    # A.1: prior_strength=10 matches live_scraper.py (was 300 — caused 30x divergence)
                    p_post, _ = compute_posterior_confidence(
                        target, N_SIMS, prior_mean=p_prior, prior_strength=10)
                    
                    conf     = min(p_post * 100.0, MAX_PROP_CONFIDENCE)
                    edge_pct = abs(edge_data['edge_pct'])

                    if conf < rt['min_prob']:
                        continue

                    payload['report'].append(
                        f"  {pn:20} | {cat:8} | Line {line_data['line']:4.1f} "
                        f"| Proj {corr_p:4.1f} | VALUABLE: {direction} "
                        f"({conf:.0f}% conf, {edge_pct:.1f}% edge)"
                    )
                    payload['val_count'] += 1
                    payload['picks'].append({
                        'player':     pn,
                        'cat':        cat,
                        'direction':  direction,
                        'line':       line_data['line'],
                        'proj':       corr_p,
                        'conf':       conf,
                        'edge_pct':   edge_pct,
                        'over_odds':  line_data['over_odds'],
                        'under_odds': line_data['under_odds'],
                        'matchup':    matchup,
                    })

    payload['logs'] = buf.getvalue()
    return payload


# ── Daily Orchestrator ────────────────────────────────────────────────

def run_retroactive_day(date_str):
    """
    date_str: Reality date (YYYYMMDD). Matches database exactly.
    """
    iso_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    print(f"[>>>] ANALYSING: {date_str} (ISO: {iso_date})")

    conn_nba = sqlite3.connect(NBA_DB)
    actual   = pd.read_sql_query(
        "SELECT game_id, team_id, player_name, pts, reb, ast "
        "FROM box_scores WHERE game_date LIKE ?",
        conn_nba, params=(f"{iso_date}%",))
    lines = get_historical_lines(date_str)

    if actual.empty or not lines:
        print(f"  [!] No valid data in DB for {date_str}. Skipping.")
        conn_nba.close()
        return

    # Cache seasonal players for this day once to avoid 20+ redundant SQL calls
    season_players = {r[0] for r in conn_nba.execute(
        "SELECT DISTINCT player_name FROM box_scores "
        "WHERE season='2025-26' AND game_date < ?", [iso_date]).fetchall()}

    # ── C.5: Load season stats for tier classification (once per day) ─────────
    season_stats = {}
    for player, avg_pts, avg_min in conn_nba.execute("""
        SELECT LOWER(player_name), AVG(pts), AVG(minutes)
        FROM box_scores
        WHERE game_date < ? GROUP BY LOWER(player_name)
    """, (iso_date,)).fetchall():
        season_stats[player] = {
            'avg_pts': float(avg_pts or 0.0),
            'avg_min': float(avg_min or 0.0),
        }

    # C.5: Load injuries for this date (uses ISO YYYY-MM-DD format)
    injuries_by_team = load_historical_injuries(iso_date, conn_nba, season_stats)
    injured_teams = len(injuries_by_team)
    print(f"  [INJURY MATRIX] {iso_date}: {injured_teams} teams with Out players")

    # ── D.2: Load spread data for blowout detection ───────────────────────────
    # Build a {(t1_id, t2_id): spread_for_t1} map using historical_odds_cache
    spread_map = {}
    try:
        spread_rows = conn_nba.execute("""
            SELECT home_team_id, away_team_id, home_spread
            FROM historical_odds_cache
            WHERE date = ? AND market = 'spreads'
        """, (date_str,)).fetchall()
        for home_id, away_id, home_spread in spread_rows:
            if home_id and away_id and home_spread is not None:
                spread_map[(int(home_id), int(away_id))] = float(home_spread)
    except Exception:
        pass  # Soft fail — spreads not available for every day

    game_tasks = []
    for gid in actual['game_id'].unique():
        g      = actual[actual['game_id'] == gid]
        t_ids  = g['team_id'].unique()
        if len(t_ids) < 2:
            continue
        t1, t2   = int(t_ids[0]), int(t_ids[1])
        t1n, t2n = TEAM_ID_TO_NAME.get(t1, f"T{t1}"), TEAM_ID_TO_NAME.get(t2, f"T{t2}")

        rosters = {t1: {}, t2: {}}
        skip    = False
        for tid in [t1, t2]:
            t_played = actual[actual['team_id'] == tid]['player_name'].tolist()
            for pn in t_played:
                if pn in season_players:
                    mx = get_player_stats_matrix(pn, before_date=date_str)
                    if mx and mx.get('avg_fga', 0) > 0:
                        rosters[tid][pn] = mx
            if len(rosters[tid]) < 5:
                skip = True

        if not skip:
            game_tasks.append((gid, iso_date, date_str, t1, t1n, t2, t2n,
                               rosters, lines, g, injuries_by_team, spread_map))

    conn_nba.close()
    if not game_tasks:
        print(f"  [!] No valid game tasks for {date_str}.")
        return

    print(f"  [*] Simulating {len(game_tasks)} games in parallel...")

    total_val   = 0
    full_report = [f"NBA MODEL PREDICTIONS - {iso_date}", "=" * 60, ""]
    all_picks   = []

    conn_tracker = sqlite3.connect(TRACKER_DB, timeout=30)
    conn_tracker.execute("PRAGMA journal_mode=WAL")

    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
        futures = {executor.submit(process_retroactive_game, *t): t
                   for t in game_tasks}
        for future in as_completed(futures):
            try:
                res = future.result()
            except Exception as e:
                print(f"  [!] Worker error: {e}")
                continue
            sys.stdout.write(res['logs'])
            full_report.extend(res['report'])
            total_val += res['val_count']
            all_picks.extend(res['picks'])
            for row in res['rows']:
                insert_projection_outcome(conn_tracker, *row, reality_date=date_str, original_iso=iso_date)

    conn_tracker.commit()
    conn_tracker.close()

    # ── Parlay Section ────────────────────────────────────────────────
    parlay_lines = build_retro_parlays(all_picks)
    if parlay_lines:
        full_report.append("")
        full_report.append("=" * 60)
        full_report.extend(parlay_lines)
        full_report.append("")

    save_report(date_str, full_report, total_val)


# ── Main ──────────────────────────────────────────────────────────────

def main(start_date, end_date):
    current = datetime.strptime(start_date, "%Y%m%d")
    end     = datetime.strptime(end_date,   "%Y%m%d")
    idx     = 0

    while current <= end:
        ds = current.strftime("%Y%m%d")
        run_retroactive_day(ds)

        if (idx + 1) % 10 == 0:
            conn = sqlite3.connect(TRACKER_DB)
            n    = conn.execute(
                "SELECT COUNT(*) FROM projection_outcomes "
                "WHERE source_mode='retroactive_live'"
            ).fetchone()[0]
            conn.close()
            print(f"[PROGRESS] {idx+1} dates done — {n:,} DB rows")

        current += timedelta(days=1)
        idx     += 1


if __name__ == "__main__":
    if len(sys.argv) > 2:
        init_retro_db()
        main(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python3 advanced_model/etl/retroactive_live_run.py YYYYMMDD YYYYMMDD")
