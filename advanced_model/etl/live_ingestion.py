import sys
import os
import io
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import sqlite3
import time
import unicodedata
import pandas as pd
from datetime import datetime, timedelta
from typing import List

# Ensure the parent directory is in the path
MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(MODEL_DIR)

from nba_api.live.nba.endpoints import scoreboard
from simulator.matrix_builder import get_player_stats_matrix, get_team_pace, get_team_efficiency, get_team_rebound_efficiency, get_player_historical_hit_rates
from simulator.markov_engine import MarkovSimulator
from etl.injury_scraper import get_injured_players, is_player_injured
from simulator.synergy_tracker import compute_matchup_multiplier

from etl.espn_odds import fetch_espn_odds, match_espn_odds_to_game
from etl.odds_fetcher import fetch_player_props, fetch_event_ids, match_event_id_to_game
from config.settings import (
    CATEGORY_PRIORS, get_prior, PROJECTION_TIERS, is_allowed,
    MAX_PROP_CONFIDENCE
)
from utils.bias import get_tiered_bias
from simulator.parlay_builder import (
    build_parlays, format_parlay_output, EDGE_THRESHOLDS,
    compute_spread_edge, compute_total_edge, compute_prop_edge
)
from simulator.markov_engine import compute_posterior_confidence
from etl.bet_tracker import ingest_shadow_picks

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'nba_data.db')

def get_connection():
    return sqlite3.connect(DB_PATH)

BET_TRACKER_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'bet_tracker.db')

def log_live_projection(player: str, category: str, projected_value: float, bookmaker_line: float,
                        over_odds: float = 1.91, under_odds: float = 1.91):
    """
    Log today's live projection to bet_tracker.db/projection_outcomes for Step 3.4 bias verification.
    actual_value and model_bias are set to 0.0 as placeholders — game not yet resolved.
    source_mode='live_pending' distinguishes these from retroactive resolved picks.
    """
    from datetime import datetime
    game_date = datetime.now().strftime('%Y-%m-%d')
    try:
        conn = sqlite3.connect(BET_TRACKER_DB)
        conn.execute("""
            INSERT OR IGNORE INTO projection_outcomes
                (game_date, player, category, projected_value, bookmaker_line,
                 actual_value, model_bias, source_mode, source_file)
            VALUES (?, ?, ?, ?, ?, 0.0, 0.0, 'live_pending', ?)
        """, (game_date, player, category, round(projected_value, 2),
              bookmaker_line, f"predicts_{game_date}.txt"))
        conn.commit()
        conn.close()
    except Exception:
        pass  # Non-critical — don't crash the scraper over logging

def get_latest_season(conn) -> str:
    '''Auto-detect the latest season available in the database.'''
    try:
        c = conn.cursor()
        c.execute("SELECT MAX(season) FROM box_scores")
        row = c.fetchone()
        if row and row[0]:
            return row[0]
    except:
        pass
    return '2025-26'  # fallback

ROSTER_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'rosters.csv')

# NBA team_id → full team name mapping
TEAM_ID_TO_NAME = {
    1610612737: 'Atlanta Hawks', 1610612738: 'Boston Celtics',
    1610612751: 'Brooklyn Nets', 1610612766: 'Charlotte Hornets',
    1610612741: 'Chicago Bulls', 1610612739: 'Cleveland Cavaliers',
    1610612742: 'Dallas Mavericks', 1610612743: 'Denver Nuggets',
    1610612765: 'Detroit Pistons', 1610612744: 'Golden State Warriors',
    1610612745: 'Houston Rockets', 1610612754: 'Indiana Pacers',
    1610612746: 'LA Clippers', 1610612747: 'Los Angeles Lakers',
    1610612763: 'Memphis Grizzlies', 1610612748: 'Miami Heat',
    1610612749: 'Milwaukee Bucks', 1610612750: 'Minnesota Timberwolves',
    1610612740: 'New Orleans Pelicans', 1610612752: 'New York Knicks',
    1610612760: 'Oklahoma City Thunder', 1610612753: 'Orlando Magic',
    1610612755: 'Philadelphia 76ers', 1610612756: 'Phoenix Suns',
    1610612757: 'Portland Trail Blazers', 1610612758: 'Sacramento Kings',
    1610612759: 'San Antonio Spurs', 1610612761: 'Toronto Raptors',
    1610612762: 'Utah Jazz', 1610612764: 'Washington Wizards',
}

def normalize_name(name: str) -> str:
    '''Strip diacriticals and lowercase for robust matching.'''
    if not name: return ""
    nfkd = unicodedata.normalize('NFKD', name)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).strip().lower()

# Cache: maps normalized_name -> db_name (built once per run)
_DB_NAME_MAP = {}

def _build_db_name_map():
    '''Build a lookup of normalize_name(db_name) -> db_name for all players in the DB.'''
    global _DB_NAME_MAP
    if _DB_NAME_MAP:
        return  # already built
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT DISTINCT player_name FROM box_scores")
        for (name,) in c.fetchall():
            _DB_NAME_MAP[normalize_name(name)] = name
        conn.close()
    except Exception as e:
        print(f"    [!] Error building DB name map: {e}")


def csv_name_to_db_name(csv_name: str) -> str:
    '''Convert a CSV roster name to the matching DB name, handling diacriticals.'''
    _build_db_name_map()
    normalized = normalize_name(csv_name)
    return _DB_NAME_MAP.get(normalized, csv_name)  # fallback to CSV name if no match


def get_active_roster(team_id: int) -> set:
    '''
    Reads the active roster from the local rosters.csv file.
    CSV format: Team, Player, Position
    Maps numeric team_id to team name for lookup.
    Returns DB-formatted names (with diacriticals) for downstream matching.
    '''
    if not os.path.isfile(ROSTER_CSV):
        print(f"[-] Roster file not found: {ROSTER_CSV}")
        print(f"    Please create database/rosters.csv with columns: Team,Player,Position")
        return set()

    team_name = TEAM_ID_TO_NAME.get(team_id)
    if not team_name:
        print(f"[-] Unknown team_id: {team_id}")
        return set()

    try:
        df = pd.read_csv(ROSTER_CSV)
        df.columns = [c.strip().lower() for c in df.columns]

        if 'team' not in df.columns or 'player' not in df.columns:
            print(f"[-] rosters.csv must have 'Team' and 'Player' columns.")
            print(f"    Found columns: {list(df.columns)}")
            return set()

        team_df = df[df['team'].str.strip() == team_name]
        csv_names = team_df['player'].dropna().str.strip().tolist()
        # Convert CSV names to DB names (handles diacriticals)
        db_names = set(csv_name_to_db_name(n) for n in csv_names)
        return db_names
    except Exception as e:
        print(f"[-] Error reading roster CSV for team {team_id} ({team_name}): {e}")
        return set()


def get_expected_lineup(team_id: int, conn, injured_normalized: dict = None) -> List[str]:
    '''
    Retrieves the projected starting 5 + 6th man by calculating the players
    with the highest average minutes played for the team this season in the DB,
    filtered by the actual current active roster to handle trades properly.
    Uses fuzzy name matching for injuries to handle Jr./III suffix mismatches.
    '''
    if injured_normalized is None:
        injured_normalized = {}
        
    active_roster = get_active_roster(team_id)
    if not active_roster:
        print(f"[-] Could not get active roster for team {team_id}")
        return []
        
    healthy_roster = [
        p for p in active_roster
        if not is_player_injured(p, injured_normalized)
    ]
    if not healthy_roster:
        return []

    # Get the average minutes for these healthy players across their most recent games
    latest_season = get_latest_season(conn)
    placeholders = ','.join(['?'] * len(healthy_roster))
    query = f"SELECT player_name, minutes FROM box_scores WHERE player_name IN ({placeholders}) AND season = ?"
    
    df = pd.read_sql_query(query, conn, params=healthy_roster + [latest_season])
    
    def parse_minutes(m):
        if pd.isna(m): return 0.0
        if isinstance(m, str) and ':' in m:
            parts = m.split(':')
            return float(parts[0]) + float(parts[1])/60.0
        try:
            return float(m)
        except:
            return 0.0

    if df.empty:
        return healthy_roster[:10]

    df['min_float'] = df['minutes'].apply(parse_minutes)
    
    avg_mins = df.groupby('player_name')['min_float'].mean().sort_values(ascending=False)
    
    sorted_players = avg_mins.index.tolist()
    
    # Some players might have 0 games this season if they are returning from long injury
    # Ensure they are added at the end
    missing = set(healthy_roster) - set(sorted_players)
    sorted_players.extend(list(missing))
    
    return sorted_players[:10]


def get_actual_lineup_from_boxscores(team_id: int, target_date: str, conn) -> List[str]:
    """
    For historical dates: return the players who actually logged minutes for a
    team on the given date, sorted by minutes played (desc).
    This avoids today's roster/injury guessing for past games.
    target_date: 'YYYY-MM-DD'
    """
    query = '''
        SELECT bs.player_name, bs.minutes
        FROM box_scores bs
        JOIN games g ON bs.game_id = g.game_id
        WHERE bs.team_id = ?
          AND g.game_date LIKE ?
          AND bs.minutes IS NOT NULL
          AND bs.minutes != '0:00'
          AND bs.minutes != '0'
    '''
    df = pd.read_sql_query(query, conn, params=(team_id, f"{target_date}%"))

    if df.empty:
        return []

    def parse_minutes(m):
        if pd.isna(m): return 0.0
        if isinstance(m, str) and ':' in m:
            parts = m.split(':')
            return float(parts[0]) + float(parts[1]) / 60.0
        try:
            return float(m)
        except:
            return 0.0

    df['min_float'] = df['minutes'].apply(parse_minutes)
    df = df[df['min_float'] >= 5.0]  # ignore garbage time DNPs / <5min
    df = df.sort_values('min_float', ascending=False)
    return df['player_name'].tolist()[:10]



def evaluate_returning_player(player_name: str, team_id: int, conn) -> str:
    '''
    Checks the local database to see how many consecutive games a player has missed
    leading up to today. Returns classification:
    RETURNING_EXTENDED, RETURNING_MODERATE, RETURNING_MINOR, or None.
    '''
    # Get team's last 15 games
    team_query = '''
        SELECT game_id, game_date
        FROM box_scores
        WHERE team_id = ? AND season = (SELECT MAX(season) FROM box_scores)
        GROUP BY game_id
        ORDER BY game_date DESC
        LIMIT 15
    '''
    team_games = pd.read_sql_query(team_query, conn, params=(team_id,))
    if team_games.empty:
        return None

    # Get player's games in the same period
    player_query = '''
        SELECT game_id
        FROM box_scores
        WHERE player_name = ? AND minutes IS NOT NULL AND minutes != '0:00'
    '''
    player_games = pd.read_sql_query(player_query, conn, params=(player_name,))
    played_game_ids = set(player_games['game_id'])

    # Count consecutive misses from most recent game backwards
    missed_consecutive = 0
    for gid in team_games['game_id']:
        if gid not in played_game_ids:
            missed_consecutive += 1
        else:
            break

    if missed_consecutive == 0:
        return None

    if missed_consecutive >= 10:
        return 'RETURNING_EXTENDED'
    elif missed_consecutive >= 5:
        return 'RETURNING_MODERATE'
    else:
        return 'RETURNING_MINOR'


def apply_returning_restrictions(matrix: dict, classification: str, is_b2b: bool = False, is_new_team: bool = False, is_surgical: bool = False) -> dict:
    '''
    Enforces hard minute ceilings and efficiency rust multipliers on returning players.
    Reduces the player's matrix statistical shares accordingly.
    '''
    if classification == 'RETURNING_EXTENDED':
        capped_mins = 18.0
        if is_surgical: capped_mins = min(capped_mins, 15.0)
        if is_b2b: capped_mins -= 3.0
        if is_new_team: capped_mins -= 3.0
        rust = 0.82
    elif classification == 'RETURNING_MODERATE':
        capped_mins = 24.0
        if is_b2b: capped_mins -= 3.0
        rust = 0.91
    elif classification == 'RETURNING_MINOR':
        capped_mins = 30.0
        rust = 0.97
    else:
        return matrix
        
    capped_mins = max(capped_mins, 5.0)
    season_mins = matrix.get('min_mean', 30.0)
    if season_mins <= 0: season_mins = capped_mins
    
    # Do not increase minutes if they normally play less than the cap
    actual_mins = min(capped_mins, season_mins)
    multiplier = (actual_mins / season_mins) * rust
    
    matrix['avg_fga'] *= multiplier
    matrix['avg_reb'] *= multiplier
    matrix['avg_ast'] *= multiplier
    matrix['avg_stl'] *= multiplier
    matrix['avg_blk'] *= multiplier
    matrix['avg_pts'] *= multiplier
    
    if is_new_team:
        matrix['avg_ast'] *= 0.88
        matrix['avg_fga'] *= 0.88
        
    matrix['minutes_restriction_flag'] = classification
    return matrix


# ── B2B Fatigue Detection ────────────────────────────────────────
B2B_SHOOTING_PENALTY = 0.96   # 4% drop in FG%
B2B_TOV_BOOST = 1.02          # 2% increase in turnovers

def is_back_to_back(team_id: int, conn) -> bool:
    '''
    Checks if the given team played a game yesterday.
    Returns True if the team is on the second night of a back-to-back.
    '''
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    cursor = conn.cursor()
    cursor.execute(
        'SELECT COUNT(*) FROM box_scores WHERE team_id = ? AND game_date = ?',
        (team_id, yesterday)
    )
    count = cursor.fetchone()[0]
    return count > 0


def apply_b2b_fatigue(player_matrix: dict) -> dict:
    '''Applies fatigue penalty to a single player's matrix.'''
    player_matrix['fg2_pct'] *= B2B_SHOOTING_PENALTY
    player_matrix['fg3_pct'] *= B2B_SHOOTING_PENALTY
    player_matrix['tov_rate'] *= B2B_TOV_BOOST
    return player_matrix


# ── Dynamic Injury Matrix ──────────────────────────────────────────
def get_team_player_roles(team_id: int, conn) -> dict:
    query = '''
    SELECT player_name, pts, fga, minutes
    FROM box_scores
    WHERE team_id = ? AND minutes IS NOT NULL AND minutes != '0:00'
    AND season = (SELECT MAX(season) FROM box_scores)
    AND game_id IN (
        SELECT DISTINCT game_id FROM box_scores WHERE team_id = ?
        ORDER BY game_date DESC LIMIT 15
    )
    '''
    df = pd.read_sql_query(query, conn, params=(team_id, team_id))
    if df.empty: return {}
    
    def parse_minutes(m):
        if pd.isna(m): return 0.0
        if isinstance(m, str) and ':' in m:
            parts = m.split(':')
            return float(parts[0]) + float(parts[1])/60.0
        try: return float(m)
        except: return 0.0
        
    df['min_float'] = df['minutes'].apply(parse_minutes)
    
    roles = {}
    for player, group in df.groupby('player_name'):
        avg_pts = group['pts'].mean()
        avg_fga = group['fga'].mean()
        min_mean = group['min_float'].mean()
        roles[player] = {'avg_pts': avg_pts, 'avg_fga': avg_fga, 'min_mean': min_mean, 'games': len(group)}
        
    return roles


def apply_dynamic_injury_matrix(team_matrix: dict, team_id: int, team_tricode: str, conn):
    '''
    Calculates the Current Roster Rating penalties based on active roster.
    If Stars/Starters are missing, penalizes the team's efficiency natively in the matrix.
    If 3+ Starters are missing, triggers Synergy Collapse.
    '''
    roles = get_team_player_roles(team_id, conn)
    if not roles: return False
    
    active_names = set(team_matrix.keys())
    missing_names = [p for p in roles.keys() if p not in active_names and roles[p]['games'] >= 4]
    
    sorted_by_pts = sorted(roles.keys(), key=lambda x: roles[x]['avg_pts'], reverse=True)
    top_2_scorers = set(sorted_by_pts[:2])
    
    total_fga = sum(r['avg_fga'] for r in roles.values())
    
    penalties = []
    missing_starters_count = 0
    
    for p in missing_names:
        role = roles[p]
        usage = role['avg_fga'] / total_fga if total_fga > 0 else 0
        is_star = (p in top_2_scorers) or (usage > 0.25)
        is_starter = role['min_mean'] >= 25.0 and not is_star
        is_role_player = role['min_mean'] >= 12.0 and role['min_mean'] < 25.0 and not is_star
        
        if is_star:
            penalties.append(0.94)
            missing_starters_count += 1
            print(f"  🚨 [INJURY MATRIX] {team_tricode} missing STAR: {p} (Tier 1 Penalty)")
        elif is_starter:
            penalties.append(0.97)
            missing_starters_count += 1
            print(f"  🚨 [INJURY MATRIX] {team_tricode} missing STARTER: {p} (Tier 2 Penalty)")
        elif is_role_player:
            penalties.append(0.99)
            print(f"  🚨 [INJURY MATRIX] {team_tricode} missing ROTATION: {p} (Tier 3 Penalty)")
            
    penalties.sort()
    final_penalty = 1.0
    for pen in penalties[:3]:
        final_penalty *= pen
        
    synergy_collapse = (missing_starters_count >= 3)
    if synergy_collapse:
        final_penalty *= 0.85
        print(f"  ⚠️  [SYNERGY COLLAPSE] {team_tricode} missing 3+ Core Players! Global 0.85x penalty applied to active roster.")
        
    if final_penalty < 1.0:
        print(f"  📉 [CURRENT ROSTER RATING] {team_tricode} Offense collectively scaled to {final_penalty*100:.1f}% capacity.")
        for p in team_matrix:
            team_matrix[p]['fg2_pct'] *= final_penalty
            team_matrix[p]['fg3_pct'] *= final_penalty
            
    return synergy_collapse


# ── Usage Rate Redistribution ────────────────────────────────────
def get_team_historical_usage(team_id: int, conn, limit: int = 15) -> dict:
    '''
    Returns {player_name: avg_fga_per_game} for the CURRENT roster over the last N games.
    Only includes players on the active roster (from rosters.csv) to avoid
    counting traded players from old seasons.
    '''
    # Get current roster to filter
    current_roster = get_active_roster(team_id)
    
    query = '''
    SELECT player_name, AVG(fga) as avg_fga
    FROM box_scores
    WHERE team_id = ? AND minutes IS NOT NULL AND minutes != '0:00'
    AND season = (SELECT MAX(season) FROM box_scores)
    AND game_id IN (
        SELECT DISTINCT game_id FROM box_scores WHERE team_id = ?
        ORDER BY game_date DESC LIMIT ?
    )
    GROUP BY player_name
    '''
    df = pd.read_sql_query(query, conn, params=(team_id, team_id, limit))
    usage = dict(zip(df['player_name'], df['avg_fga']))
    
    # Filter to only current roster players
    if current_roster:
        usage = {p: fga for p, fga in usage.items() if p in current_roster}
    
    return usage


def redistribute_usage(team_matrix: dict, full_roster_usage: dict, team_tricode: str) -> dict:
    '''
    When key players are missing, redistribute their FGA proportionally
    to the active players based on existing usage shares.
    '''
    active_names = set(team_matrix.keys())
    all_names = set(full_roster_usage.keys())
    missing_names = all_names - active_names

    # Total FGA missing from injured/resting players
    missing_fga = sum(full_roster_usage.get(p, 0) for p in missing_names)

    if missing_fga < 1.0:
        return team_matrix  # No significant usage to redistribute

    # Active players' current usage
    active_usage = {p: team_matrix[p].get('avg_fga', 1) for p in active_names}
    total_active_usage = sum(active_usage.values())

    if total_active_usage == 0:
        return team_matrix

    # Redistribute proportionally
    top_missing = sorted(missing_names, key=lambda p: full_roster_usage.get(p, 0), reverse=True)[:3]
    missing_names_str = ', '.join(top_missing)
    print(f"  📈 [USAGE BOOST] {team_tricode}: {missing_fga:.1f} extra FGA redistributed (missing: {missing_names_str})")

    for player in active_names:
        share = active_usage[player] / total_active_usage
        extra_fga = missing_fga * share
        old_fga = team_matrix[player].get('avg_fga', 0)
        team_matrix[player]['avg_fga'] = old_fga + extra_fga

        # Diminishing-returns penalty: extra volume reduces shooting efficiency.
        # For every 25% increase in FGA, reduce FG% by 3% (real-world bench players
        # can't maintain efficiency on inflated usage).
        if old_fga > 0:
            fga_increase_pct = extra_fga / old_fga
            # Flag as boosted if receiving > 1.5 extra FGA or > 15% usage increase
            if extra_fga >= 1.5 or fga_increase_pct >= 0.15:
                team_matrix[player]['usage_boosted'] = True
                
            efficiency_penalty = 1.0 - (fga_increase_pct * 0.12)  # 0.12 = 3% per 25%
            efficiency_penalty = max(efficiency_penalty, 0.88)      # Floor at 12% penalty
            team_matrix[player]['fg2_pct'] *= efficiency_penalty
            team_matrix[player]['fg3_pct'] *= efficiency_penalty

    return team_matrix



def process_hist_game(game, target_date, injured_normalized):
    import io, contextlib, time
    from simulator.markov_engine import MarkovSimulator
    from simulator.matrix_builder import get_player_stats_matrix, get_team_pace, get_team_efficiency, get_team_rebound_efficiency
    from simulator.synergy_tracker import compute_matchup_multiplier
    
    output_buffer = io.StringIO()
    payload = {'game_logs': ''}
    
    with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
        conn = get_connection()
        try:
            home_team = game['homeTeam']
            away_team = game['awayTeam']
            
            matchup_str = f"{away_team['teamCity']} {away_team['teamName']} @ {home_team['teamCity']} {home_team['teamName']}"
            print(f"\n{'='*60}")
            print(f" MATCHUP: {matchup_str}")
            print(f"{'='*60}")

            home_team_full = f"{home_team['teamCity']} {home_team['teamName']}"
            away_team_full = f"{away_team['teamCity']} {away_team['teamName']}"

            # 1. Get actual lineup from box scores (who really played that night)
            # This is 100% accurate — no live roster or injury guessing needed.
            home_lineup = get_actual_lineup_from_boxscores(home_team['teamId'], target_date, conn)
            away_lineup = get_actual_lineup_from_boxscores(away_team['teamId'], target_date, conn)

            if not home_lineup or not away_lineup:
                print("[-] No box score data found for this game. Run 'py etl/fetch_range_box_scores.py' first.")
                return payload

            # 2. Build the Matrices
            home_matrix = {}
            away_matrix = {}

            home_b2b = is_back_to_back(home_team['teamId'], conn)
            away_b2b = is_back_to_back(away_team['teamId'], conn)
            if home_b2b:
                print(f"  ⚠️  [B2B FATIGUE] {home_team['teamTricode']} is on a BACK-TO-BACK")
            if away_b2b:
                print(f"  ⚠️  [B2B FATIGUE] {away_team['teamTricode']} is on a BACK-TO-BACK")

            for hp in home_lineup:
                try:
                    matrix = get_player_stats_matrix(hp, limit=15, current_team_id=home_team['teamId'])
                    modifier = compute_matchup_multiplier(hp, away_team_full)
                    if modifier != 1.0:
                        print(f"  [Synergy] {hp} vs {away_team_full}: {modifier:.3f}x Offense Modifier")
                    matrix['fg2_pct'] *= modifier
                    matrix['fg3_pct'] *= modifier
                    ret_class = evaluate_returning_player(hp, home_team['teamId'], conn)
                    if ret_class:
                        matrix = apply_returning_restrictions(matrix, classification=ret_class, is_b2b=home_b2b)
                        print(f"  ⚠️  [MINUTES RESTRICTION] {hp} classified as {ret_class}.")
                    if home_b2b:
                        matrix = apply_b2b_fatigue(matrix)
                    home_matrix[hp] = matrix
                except ValueError:
                    pass
            
            for ap in away_lineup:
                try:
                    matrix = get_player_stats_matrix(ap, limit=15, current_team_id=away_team['teamId'])
                    modifier = compute_matchup_multiplier(ap, home_team_full)
                    if modifier != 1.0:
                        print(f"  [Synergy] {ap} vs {home_team_full}: {modifier:.3f}x Offense Modifier")
                    matrix['fg2_pct'] *= modifier
                    matrix['fg3_pct'] *= modifier
                    ret_class = evaluate_returning_player(ap, away_team['teamId'], conn)
                    if ret_class:
                        matrix = apply_returning_restrictions(matrix, classification=ret_class, is_b2b=away_b2b)
                        print(f"  ⚠️  [MINUTES RESTRICTION] {ap} classified as {ret_class}.")
                    if away_b2b:
                        matrix = apply_b2b_fatigue(matrix)
                    away_matrix[ap] = matrix
                except ValueError:
                    pass

            if not home_matrix or not away_matrix:
                print("[-] Could not build player matrices from the database.")
                return payload

            # Usage Redistribution
            home_usage = get_team_historical_usage(home_team['teamId'], conn)
            away_usage = get_team_historical_usage(away_team['teamId'], conn)
            home_matrix = redistribute_usage(home_matrix, home_usage, home_team['teamTricode'])
            away_matrix = redistribute_usage(away_matrix, away_usage, away_team['teamTricode'])

            # Team Efficiency
            home_eff = get_team_efficiency(home_team['teamId'])
            away_eff = get_team_efficiency(away_team['teamId'])
            home_reb_eff = get_team_rebound_efficiency(home_team['teamId'])
            away_reb_eff = get_team_rebound_efficiency(away_team['teamId'])
            
            # Dynamic Injury Matrix
            home_collapse = apply_dynamic_injury_matrix(home_matrix, home_team['teamId'], home_team['teamTricode'], conn)
            away_collapse = apply_dynamic_injury_matrix(away_matrix, away_team['teamId'], away_team['teamTricode'], conn)
            
            print(f"  ⚡ [TEAM RATINGS] {home_team['teamTricode']}: OffRtg {home_eff.get('off_rtg', 112):.1f} | DefRtg {home_eff.get('def_rtg', 112):.1f}")
            print(f"  ⚡ [TEAM RATINGS] {away_team['teamTricode']}: OffRtg {away_eff.get('off_rtg', 112):.1f} | DefRtg {away_eff.get('def_rtg', 112):.1f}")

            # Apply defensive efficiency
            for player in home_matrix:
                home_matrix[player]['fg2_pct'] *= away_eff['def_multiplier']
                home_matrix[player]['fg3_pct'] *= away_eff['def_multiplier']
                if 'avg_reb' in home_matrix[player]:
                    home_matrix[player]['avg_reb'] *= away_reb_eff['rebound_modifier']
            for player in away_matrix:
                away_matrix[player]['fg2_pct'] *= home_eff['def_multiplier']
                away_matrix[player]['fg3_pct'] *= home_eff['def_multiplier']
                if 'avg_reb' in away_matrix[player]:
                    away_matrix[player]['avg_reb'] *= home_reb_eff['rebound_modifier']

            # Pace
            home_pace = get_team_pace(home_team['teamId'])
            away_pace = get_team_pace(away_team['teamId'])
            expected_pace = int((home_pace + away_pace) / 2)
            print(f"  [Calculated Game Pace: ~{expected_pace} possessions per team]")
            is_fast_pace = expected_pace > 100.5

            # Run Monte Carlo Simulations
            engine = MarkovSimulator(home_matrix, away_matrix, is_fast_pace=is_fast_pace)
            simulations = 5000
            aggregate_home_score = 0
            aggregate_away_score = 0
            
            prop_tracker = {player: {'PTS': [], 'REB': [], 'AST': [], 'FG3M': []} 
                            for player in list(home_matrix.keys()) + list(away_matrix.keys())}
            
            for i in range(simulations):
                result = engine.run_full_game(pace=expected_pace)
                aggregate_home_score += result['home_score']
                aggregate_away_score += result['away_score']
                
                for player, stats in result['player_stats'].items():
                    if player in prop_tracker:
                        prop_tracker[player]['PTS'].append(stats['PTS'])
                        prop_tracker[player]['REB'].append(stats['REB'])
                        prop_tracker[player]['AST'].append(stats['AST'])
                        prop_tracker[player]['FG3M'].append(stats['FG3M'])

            # Output Projections
            avg_home = aggregate_home_score / simulations
            avg_away = aggregate_away_score / simulations
            
            print(f"PROJECTED TOTAL: {avg_home + avg_away:.1f}")
            print(f"PROJECTED SCORE: {home_team['teamTricode']} {avg_home:.1f} - {away_team['teamTricode']} {avg_away:.1f}")

            print("-" * 40)
            print("   TOP PLAYER PROP PROJECTIONS (MEDIAN)")
            print("-" * 40)

            # ── Load stale cached prop lines from when this date was originally run ──
            import json as _json
            game_props = {}
            _cache_events_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.cache', 'odds_api_events.json')
            _cached_event_ids = {}
            if os.path.exists(_cache_events_path):
                try:
                    with open(_cache_events_path, 'r', encoding='utf-8') as _f:
                        _cached_event_ids = _json.load(_f)
                except Exception:
                    pass

            if _cached_event_ids:
                _eid = match_event_id_to_game(_cached_event_ids, home_team_full, away_team_full)
                if _eid:
                    try:
                        game_props = fetch_player_props(_eid, force_cache=True)
                    except Exception as _e:
                        print(f"    [!] Could not load cached props: {_e}")

            for team, lineup in [("HOME", home_lineup), ("AWAY", away_lineup)]:
                print(f"[{team} TEAM]")
                for p in lineup:
                    if p in prop_tracker:
                        pts = sorted(prop_tracker[p]['PTS'])
                        reb = sorted(prop_tracker[p]['REB'])
                        ast = sorted(prop_tracker[p]['AST'])
                        median_pts = pts[len(pts)//2]
                        median_reb = reb[len(reb)//2]
                        median_ast = ast[len(ast)//2]

                        props_str = ""
                        if game_props and p in game_props:
                            p_props = game_props[p]
                            prop_details = []
                            base_stats = [
                                ("points", "PTS", "points"),
                                ("rebounds", "REB", "rebounds"),
                                ("assists", "AST", "assists"),
                            ]
                            for stat_name, dict_key, stat_key in base_stats:
                                if stat_name in p_props:
                                    line = p_props[stat_name]['line']
                                    dist = sorted(prop_tracker[p][dict_key])
                                    edge_data = compute_prop_edge(dist, line, stat_key)
                                    if edge_data['valuable']:
                                        raw_conf = edge_data['confidence'] if edge_data['direction'] == "Over" else (100 - edge_data['confidence'])
                                        best_conf = min(raw_conf, MAX_PROP_CONFIDENCE)
                                        prop_details.append(f"O/U {line} {stat_name} VALUABLE: {edge_data['direction']} ({best_conf:.0f}% conf, {abs(edge_data['edge_pct']):.1f}% edge)")
                                    else:
                                        prop_details.append(f"O/U {line} {stat_name} ignore")
                            if prop_details:
                                props_str = " (" + ", ".join(prop_details) + ")"

                        # Check if this player has high trade uncertainty
                        _matrix_dict = home_matrix if team == "HOME" else away_matrix
                        _uncertain = _matrix_dict.get(p, {}).get('_high_uncertainty', False)
                        _uncertain_tag = " [HIGH_UNCERTAINTY]" if _uncertain else ""

                        print(f"  {p:20} -> {median_pts} PTS, {median_reb} REB, {median_ast} AST{props_str}{_uncertain_tag}")
                print("")


        finally:
            conn.close()
            payload['game_logs'] = output_buffer.getvalue()
    return payload

def generate_historical_projections(target_date: str):
    """
    Run the simulation pipeline for a historical date (YYYY-MM-DD).
    Sources matchups from the database instead of ESPN odds.
    Runs simulations without live odds/player props (projection-only mode).
    """
    from nba_api.stats.static import teams
    nba_teams = teams.get_teams()

    print(f"[+] HISTORICAL MODE: Generating projections for {target_date}")
    print(f"    (No live odds/props — simulation-only)")
    
    # Pre-fetch live injuries
    injured_raw, injured_normalized = get_injured_players()

    conn = get_connection()
    
    # Get games from the database for this date
    # The game_date column stores dates as 'YYYY-MM-DDT00:00:00' format
    cursor = conn.cursor()
    cursor.execute(
        "SELECT game_id, matchup FROM games WHERE game_date LIKE ?",
        (f"{target_date}%",)
    )
    db_games = cursor.fetchall()
    
    if not db_games:
        print(f"[!] No games found in database for {target_date}.")
        print(f"    Run 'py etl/fetch_range_box_scores.py' first to backfill the data.")
        conn.close()
        return

    print(f"    [+] Found {len(db_games)} games in database for {target_date}")
    
    # Parse matchups into home/away teams
    # DB format: "HOME vs. AWAY" (home perspective) or "AWAY @ HOME" (away perspective)
    # We only need one perspective per game, so deduplicate by game_id
    seen_game_ids = set()
    games = []
    
    # NBA team abbreviation → team data lookup
    abbrev_to_team = {t['abbreviation']: t for t in nba_teams}
    
    for game_id, matchup in db_games:
        if game_id in seen_game_ids:
            continue
        seen_game_ids.add(game_id)
        
        if ' vs. ' in matchup:
            home_abbr, away_abbr = matchup.split(' vs. ')
        elif ' @ ' in matchup:
            away_abbr, home_abbr = matchup.split(' @ ')
        else:
            continue
            
        home_abbr = home_abbr.strip()
        away_abbr = away_abbr.strip()
        
        h_data = abbrev_to_team.get(home_abbr)
        a_data = abbrev_to_team.get(away_abbr)
        
        if h_data and a_data:
            games.append({
                'homeTeam': {
                    'teamId': h_data['id'],
                    'teamName': h_data['nickname'],
                    'teamCity': h_data['city'],
                    'teamTricode': h_data['abbreviation']
                },
                'awayTeam': {
                    'teamId': a_data['id'],
                    'teamName': a_data['nickname'],
                    'teamCity': a_data['city'],
                    'teamTricode': a_data['abbreviation']
                }
            })
    
    if not games:
        print("[!] Could not parse any matchups from database.")
        conn.close()
        return

    print(f"    [+] Parsed {len(games)} unique matchups:")
    for g in games:
        print(f"        {g['awayTeam']['teamTricode']} @ {g['homeTeam']['teamTricode']}")


    print("\n[+] Commencing Parallel Historical Simulations...")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_hist_game, game, target_date, injured_normalized) for game in games]
        
        for future in as_completed(futures):
            try:
                res = future.result()
                sys.stdout.write(res['game_logs'])
            except Exception as e:
                print(f"[-] Error processing hist game: {e}")

    conn.close()
    print(f"\n[+] Historical projections for {target_date} complete.")
    print(f"    Compare these against actual results in yesterday_results.txt")



def process_live_game(game, odds_available, all_game_odds, odds_api_events, injured_normalized):
    import io, contextlib, time
    from simulator.markov_engine import MarkovSimulator
    from simulator.matrix_builder import get_player_stats_matrix, get_team_pace, get_team_efficiency, get_team_rebound_efficiency
    from simulator.synergy_tracker import compute_matchup_multiplier
    from etl.injury_scraper import is_player_injured
    
    # We will buffer output to avoid parallel print collisions
    output_buffer = io.StringIO()
    payload = {
        'game_logs': '',
        'sim_results': None,
        'cached_props': {},
        'score_dists': None,
        'prop_dists': {},
        'player_minutes': {},
        'event_id': None,
        'home_team_full': None,
        'away_team_full': None,
        'excluded': False
    }

    with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
        conn = get_connection()
        try:
            home_team = game['homeTeam']
            away_team = game['awayTeam']
            
            matchup_str = f"{away_team['teamCity']} {away_team['teamName']} @ {home_team['teamCity']} {home_team['teamName']}"
            print(f"\n{'='*60}")
            print(f" MATCHUP: {matchup_str}")
            print(f"{'='*60}")

            home_team_full = f"{home_team['teamCity']} {home_team['teamName']}"
            away_team_full = f"{away_team['teamCity']} {away_team['teamName']}"

            # ── Show sportsbook odds if available ──────────────────
            matched_event = match_espn_odds_to_game(all_game_odds, home_team_full, away_team_full) if odds_available else None
            event_id = matched_event[0] if matched_event else None
            game_odds = matched_event[1] if matched_event else None
            
            payload['event_id'] = event_id
            payload['home_team_full'] = home_team_full
            payload['away_team_full'] = away_team_full

            if game_odds:
                print(f"  📊 SPORTSBOOK ODDS:")
                if game_odds.get('h2h'):
                    h = game_odds['h2h']
                    print(f"     H2H:    {home_team['teamTricode']} {h['home_odds']:.2f} | {away_team['teamTricode']} {h['away_odds']:.2f}  ({h['book']})")
                if game_odds.get('spreads'):
                    s = game_odds['spreads']
                    print(f"     Spread: {home_team['teamTricode']} {s['home_spread']:+.1f} @ {s['home_odds']:.2f}  ({s['book']})")
                if game_odds.get('totals'):
                    t = game_odds['totals']
                    print(f"     Total:  O/U {t['total']} — Over {t['over_odds']:.2f} / Under {t['under_odds']:.2f}  ({t['book']})")

            # 1. Project Lineups dynamically based on season averages
            home_lineup = get_expected_lineup(home_team['teamId'], conn, injured_normalized)
            away_lineup = get_expected_lineup(away_team['teamId'], conn, injured_normalized)

            if not home_lineup or not away_lineup:
                print("[-] Insufficient historical data in local DB for these teams yet.")
                return payload  # Exited early

            # 2. Build the Matrices and apply Matchup Defensive Multipliers
            home_matrix = {}
            away_matrix = {}

            # ── B2B Fatigue Detection ──────────────────────────
            home_b2b = is_back_to_back(home_team['teamId'], conn)
            away_b2b = is_back_to_back(away_team['teamId'], conn)
            if home_b2b:
                print(f"  ⚠️  [B2B FATIGUE] {home_team['teamTricode']} is on a BACK-TO-BACK (4% shooting penalty applied)")
            if away_b2b:
                print(f"  ⚠️  [B2B FATIGUE] {away_team['teamTricode']} is on a BACK-TO-BACK (4% shooting penalty applied)")

            for hp in home_lineup:
                try:
                    matrix = get_player_stats_matrix(hp, limit=15, current_team_id=home_team['teamId'])
                    modifier = compute_matchup_multiplier(hp, away_team_full)
                    if modifier != 1.0:
                        print(f"  [Synergy] {hp} vs {away_team_full}: {modifier:.3f}x Offense Modifier")
                    matrix['fg2_pct'] *= modifier
                    matrix['fg3_pct'] *= modifier
                    
                    # Check Returning Player Restrictions
                    ret_class = evaluate_returning_player(hp, home_team['teamId'], conn)
                    if ret_class:
                        matrix = apply_returning_restrictions(matrix, classification=ret_class, is_b2b=home_b2b)
                        print(f"  ⚠️  [MINUTES RESTRICTION] {hp} classified as {ret_class}. Stats auto-scaled.")

                    if home_b2b:
                        matrix = apply_b2b_fatigue(matrix)
                    home_matrix[hp] = matrix
                except ValueError:
                    pass
            
            for ap in away_lineup:
                try:
                    matrix = get_player_stats_matrix(ap, limit=15, current_team_id=away_team['teamId'])
                    modifier = compute_matchup_multiplier(ap, home_team_full)
                    if modifier != 1.0:
                        print(f"  [Synergy] {ap} vs {home_team_full}: {modifier:.3f}x Offense Modifier")
                    matrix['fg2_pct'] *= modifier
                    matrix['fg3_pct'] *= modifier
                    
                    # Check Returning Player Restrictions
                    ret_class = evaluate_returning_player(ap, away_team['teamId'], conn)
                    if ret_class:
                        matrix = apply_returning_restrictions(matrix, classification=ret_class, is_b2b=away_b2b)
                        print(f"  ⚠️  [MINUTES RESTRICTION] {ap} classified as {ret_class}. Stats auto-scaled.")

                    if away_b2b:
                        matrix = apply_b2b_fatigue(matrix)
                    away_matrix[ap] = matrix
                except ValueError:
                    pass

            if not home_matrix or not away_matrix:
                print("[-] Could not build player matrices from the database.")
                return payload  # Exited early

            # ── Usage Redistribution (when stars are out) ──────────
            home_usage = get_team_historical_usage(home_team['teamId'], conn)
            away_usage = get_team_historical_usage(away_team['teamId'], conn)
            home_matrix = redistribute_usage(home_matrix, home_usage, home_team['teamTricode'])
            away_matrix = redistribute_usage(away_matrix, away_usage, away_team['teamTricode'])

            # ── Team Efficiency Ratings ──────────────────────────
            home_eff = get_team_efficiency(home_team['teamId'])
            away_eff = get_team_efficiency(away_team['teamId'])
            
            home_reb_eff = get_team_rebound_efficiency(home_team['teamId'])
            away_reb_eff = get_team_rebound_efficiency(away_team['teamId'])
            
            # ── Module 2: Dynamic Injury Matrix ──────────────────
            home_collapse = apply_dynamic_injury_matrix(home_matrix, home_team['teamId'], home_team['teamTricode'], conn)
            away_collapse = apply_dynamic_injury_matrix(away_matrix, away_team['teamId'], away_team['teamTricode'], conn)
            
            print(f"  ⚡ [TEAM RATINGS] {home_team['teamTricode']}: OffRtg {home_eff.get('off_rtg', 112):.1f} | DefRtg {home_eff.get('def_rtg', 112):.1f} | OREB% {home_reb_eff['oreb_pct']*100:.1f} DREB% {home_reb_eff['dreb_pct']*100:.1f}")
            print(f"  ⚡ [TEAM RATINGS] {away_team['teamTricode']}: OffRtg {away_eff.get('off_rtg', 112):.1f} | DefRtg {away_eff.get('def_rtg', 112):.1f} | OREB% {away_reb_eff['oreb_pct']*100:.1f} DREB% {away_reb_eff['dreb_pct']*100:.1f}")

            # Apply opponent's defensive efficiency to each team's shooting
            # If opponent has def_multiplier 0.94 (elite defense), our FG% gets scaled down 6%
            for player in home_matrix:
                home_matrix[player]['fg2_pct'] *= away_eff['def_multiplier']
                home_matrix[player]['fg3_pct'] *= away_eff['def_multiplier']
                if 'avg_reb' in home_matrix[player]:
                    home_matrix[player]['avg_reb'] *= away_reb_eff['rebound_modifier']
            for player in away_matrix:
                away_matrix[player]['fg2_pct'] *= home_eff['def_multiplier']
                away_matrix[player]['fg3_pct'] *= home_eff['def_multiplier']
                if 'avg_reb' in away_matrix[player]:
                    away_matrix[player]['avg_reb'] *= home_reb_eff['rebound_modifier']

            # 3. Calculate dynamic Game Pace
            home_pace = get_team_pace(home_team['teamId'])
            away_pace = get_team_pace(away_team['teamId'])
            expected_pace = int((home_pace + away_pace) / 2)
            print(f"  [Calculated Game Pace: ~{expected_pace} possessions per team]")
            is_fast_pace = expected_pace > 100.5
            
            # ── Module 1: Garbage Time Rebound Shift (Blowout Protocol) ──
            # ISSUE 3 FIX (Stage 0): Apply ONLY to the UNDERDOG team.
            # Favorite starters dominate Q1-Q3 and accumulate stats normally.
            # Underdog starters sit in garbage time -> bench gets usage.
            if game_odds and game_odds.get('spreads'):
                market_spread = abs(float(game_odds['spreads']['home_spread']))
                if market_spread > 9.5:
                    home_spread_val = float(game_odds['spreads']['home_spread'])
                    # Negative home_spread = home team is favorite -> underdog is away
                    underdog_matrix = away_matrix if home_spread_val < 0 else home_matrix
                    underdog_name = away_team.get('teamName', 'Away') if home_spread_val < 0 else home_team.get('teamName', 'Home')
                    print(f"  \U0001f6a8 [BLOWOUT SHIFT] Spread={market_spread:.1f}. Shifting usage/rebounds to {underdog_name} bench only (underdog)...")
                    sorted_players = sorted(underdog_matrix.keys(), key=lambda p: underdog_matrix[p].get('avg_fga', 0), reverse=True)
                    if len(sorted_players) >= 6:
                        starters = sorted_players[:3]
                        bench = sorted_players[-3:]
                        fga_pool = 0.0
                        reb_pool = 0.0
                        
                        for s in starters:
                            if 'avg_fga' in underdog_matrix[s]:
                                reduction = underdog_matrix[s]['avg_fga'] * 0.15
                                underdog_matrix[s]['avg_fga'] -= reduction
                                fga_pool += reduction
                            if 'avg_reb' in underdog_matrix[s]:
                                reduction = underdog_matrix[s]['avg_reb'] * 0.15
                                underdog_matrix[s]['avg_reb'] -= reduction
                                reb_pool += reduction
                                
                        for b in bench:
                            if 'avg_fga' in underdog_matrix[b]: underdog_matrix[b]['avg_fga'] += fga_pool / len(bench)
                            if 'avg_reb' in underdog_matrix[b]: underdog_matrix[b]['avg_reb'] += reb_pool / len(bench)


            # 4. Feed the expected rosters into the Markov Simulator
            engine = MarkovSimulator(home_matrix, away_matrix, is_fast_pace=is_fast_pace)
            # ── Start Monte Carlo Simulations ──────────────
            simulations = 5000
            aggregate_home_score = 0
            aggregate_away_score = 0
            
            # Track props for all active players (full distributions for parlay builder)
            prop_tracker = {player: {'PTS': [], 'REB': [], 'AST': [], 'FG3M': []} 
                            for player in list(home_matrix.keys()) + list(away_matrix.keys())}
            
            # Track score distributions for spread/total edge calculation
            home_scores_list = []
            away_scores_list = []
            
            for i in range(simulations):
                result = engine.run_full_game(pace=expected_pace)
                aggregate_home_score += result['home_score']
                aggregate_away_score += result['away_score']
                home_scores_list.append(result['home_score'])
                away_scores_list.append(result['away_score'])
                
                for player, stats in result['player_stats'].items():
                    if player in prop_tracker:
                        prop_tracker[player]['PTS'].append(stats['PTS'])
                        prop_tracker[player]['REB'].append(stats['REB'])
                        prop_tracker[player]['AST'].append(stats['AST'])
                        prop_tracker[player]['FG3M'].append(stats['FG3M'])
                        
            # ── Fetch player props via Odds API (props only) ─────────
            game_props = {}
            if event_id and odds_api_events:
                odds_api_eid = match_event_id_to_game(odds_api_events, home_team_full, away_team_full)
                if odds_api_eid:
                    try:
                        game_props = fetch_player_props(odds_api_eid)
                        if game_props:
                            payload['cached_props'] = game_props
                        time.sleep(0.5)
                    except Exception as e:
                        print(f"    [!] Could not fetch player props: {e}")

            # 5. Output Projections
            avg_home = aggregate_home_score / simulations
            avg_away = aggregate_away_score / simulations
            
            print(f"PROJECTED TOTAL: {avg_home + avg_away:.1f}")
            print(f"PROJECTED SCORE: {home_team['teamTricode']} {avg_home:.1f} - {away_team['teamTricode']} {avg_away:.1f}")
            
            # ── Print Game Spread & Total Evaluation ──
            if event_id and event_id in all_game_odds:
                g_odds = all_game_odds[event_id]
                
                if g_odds.get('totals'):
                    t_data = g_odds['totals']
                    log_live_projection('Game Total', 'TOTAL', avg_home + avg_away, t_data['total'],
                                        over_odds=t_data.get('over_odds', 1.91),
                                        under_odds=t_data.get('under_odds', 1.91))
                    t_edge = compute_total_edge(home_scores_list, away_scores_list, t_data['total'])
                    if t_edge['valuable']:
                        raw_conf = t_edge['confidence_over'] if t_edge['direction'] == 'Over' else t_edge['confidence_under']
                        t_prior = get_prior('TOTAL', t_edge['direction'])
                        t_target_count = t_edge['over_count'] if t_edge['direction'] == 'Over' else t_edge['under_count']
                        t_post, _ = compute_posterior_confidence(t_target_count, t_edge['n_sim'], prior_mean=t_prior, prior_strength=10)
                        best_conf = min(t_post * 100.0, MAX_PROP_CONFIDENCE)
                        if not is_allowed('TOTAL', t_edge['direction']):
                            print(f"  -> GAME TOTAL: O/U {t_data['total']} ⛔ BANNED_CATEGORY")
                        else:
                            print(f"  -> GAME TOTAL: O/U {t_data['total']} VALUABLE: {t_edge['direction']} ({best_conf:.0f}% conf (raw: {raw_conf:.0f}%), {abs(t_edge['edge_pct']):.1f}% edge)")
                    else:
                        print(f"  -> GAME TOTAL: O/U {t_data['total']} ignore")
                
                if g_odds.get('spreads'):
                    s_data = g_odds['spreads']
                    s_thresh = EDGE_THRESHOLDS['spreads']
                    
                    h_edge = compute_spread_edge(home_scores_list, away_scores_list, s_data['home_spread'], is_home=True)
                    h_prior = get_prior('SPREAD', 'COVER')
                    h_post, _ = compute_posterior_confidence(h_edge['over_count'], h_edge['n_sim'], prior_mean=h_prior, prior_strength=10)
                    h_conf = min(h_post * 100.0, MAX_PROP_CONFIDENCE)
                    if h_conf >= s_thresh['min_prob'] and h_edge['edge_pct'] >= s_thresh['min_edge']:
                        if not is_allowed('SPREAD', 'COVER'):
                            print(f"  -> HOME SPREAD: {home_team['teamTricode']} {s_data['home_spread']:+.1f} ⛔ BANNED_CATEGORY")
                        else:
                            print(f"  -> HOME SPREAD: {home_team['teamTricode']} {s_data['home_spread']:+.1f} VALUABLE: Cover ({h_conf:.0f}% conf (raw: {h_edge['confidence']:.0f}%), {h_edge['edge_pct']:.1f}% edge)")
                    else:
                        print(f"  -> HOME SPREAD: {home_team['teamTricode']} {s_data['home_spread']:+.1f} ignore")
                        
                    a_edge = compute_spread_edge(home_scores_list, away_scores_list, s_data['away_spread'], is_home=False)
                    a_prior = get_prior('SPREAD', 'COVER')
                    a_post, _ = compute_posterior_confidence(a_edge['over_count'], a_edge['n_sim'], prior_mean=a_prior, prior_strength=10)
                    a_conf = min(a_post * 100.0, MAX_PROP_CONFIDENCE)
                    if a_conf >= s_thresh['min_prob'] and a_edge['edge_pct'] >= s_thresh['min_edge']:
                        if not is_allowed('SPREAD', 'COVER'):
                            print(f"  -> AWAY SPREAD: {away_team['teamTricode']} {s_data['away_spread']:+.1f} ⛔ BANNED_CATEGORY")
                        else:
                            print(f"  -> AWAY SPREAD: {away_team['teamTricode']} {s_data['away_spread']:+.1f} VALUABLE: Cover ({a_conf:.0f}% conf (raw: {a_edge['confidence']:.0f}%), {a_edge['edge_pct']:.1f}% edge)")
                    else:
                        print(f"  -> AWAY SPREAD: {away_team['teamTricode']} {s_data['away_spread']:+.1f} ignore")

            print("-" * 40)
            print("   TOP PLAYER PROP PROJECTIONS (MEDIAN)")
            print("-" * 40)
            
            # Sort and print median projection for all players
            for team, lineup in [("HOME", home_lineup), ("AWAY", away_lineup)]:
                print(f"[{team} TEAM]")
                for p in lineup: 
                    if p in prop_tracker:
                        pts = sorted(prop_tracker[p]['PTS'])
                        reb = sorted(prop_tracker[p]['REB'])
                        ast = sorted(prop_tracker[p]['AST'])
                        median_pts = pts[len(pts)//2]
                        median_reb = reb[len(reb)//2]
                        median_ast = ast[len(ast)//2]

                        props_str = ""
                        if game_props and p in game_props:
                            p_props = game_props[p]
                            prop_details = []
                            
                            # 1. Base Stats
                            base_stats = [
                                ("points", "PTS", "points"), 
                                ("rebounds", "REB", "rebounds"), 
                                ("assists", "AST", "assists")
                            ]
                            
                            for stat_name, dict_key, stat_key in base_stats:
                                if stat_name in p_props:
                                    line = p_props[stat_name]['line']
                                    dist = sorted(prop_tracker[p][dict_key])
                                    median_proj = dist[len(dist)//2]
                                    is_returning = bool(home_matrix.get(p, {}).get('minutes_restriction_flag', False) or away_matrix.get(p, {}).get('minutes_restriction_flag', False))
                                    edge_data = compute_prop_edge(dist, line, stat_key, is_returning=is_returning)
                                    
                                    # Log for Step 3.4 verification (POINTS/REBOUNDS/ASSISTS vs bookmaker line)
                                    log_live_projection(p, stat_key.upper(), median_proj, line)
                                    
                                    if edge_data.get('returning_ban'):
                                        prop_details.append(f"O/U {line} {stat_name} \u26d4 DISQUALIFIED: MINUTES_RESTRICTION")
                                    elif edge_data['valuable']:
                                        raw_conf = edge_data['confidence'] if edge_data['direction'] == "Over" else (100 - edge_data['confidence'])
                                        p_prior = get_prior(stat_key, edge_data['direction'])
                                        target_count = edge_data['over_count'] if edge_data['direction'] == "Over" else (edge_data['n_sim'] - edge_data['over_count'])
                                        p_post, _ = compute_posterior_confidence(target_count, edge_data['n_sim'], prior_mean=p_prior, prior_strength=10)
                                        best_conf = min(p_post * 100.0, MAX_PROP_CONFIDENCE)
                                        prop_details.append(f"O/U {line} {stat_name} VALUABLE: {edge_data['direction']} ({best_conf:.0f}% conf (raw: {raw_conf:.0f}%), {abs(edge_data['edge_pct']):.1f}% edge)")
                                    else:
                                        prop_details.append(f"O/U {line} {stat_name} ignore")

                                        
                            # 2. Combo Stats
                            combo_stats = [
                                ("points_rebounds", ["PTS", "REB"], "PR"),
                                ("points_assists", ["PTS", "AST"], "PA"),
                                ("rebounds_assists", ["REB", "AST"], "RA"),
                                ("points_rebounds_assists", ["PTS", "REB", "AST"], "PRA")
                            ]
                            
                            for combo_name, components, short_name in combo_stats:
                                if combo_name in p_props:
                                    line = p_props[combo_name]['line']
                                    # Synthesize the combo distribution by summing elements
                                    dist = [sum(prop_tracker[p][c][i] for c in components) for i in range(len(prop_tracker[p]['PTS']))]
                                    is_returning = bool(home_matrix.get(p, {}).get('minutes_restriction_flag', False) or away_matrix.get(p, {}).get('minutes_restriction_flag', False))
                                    edge_data = compute_prop_edge(dist, line, 'combo', is_returning=is_returning)
                                    
                                    if edge_data.get('returning_ban'):
                                        prop_details.append(f"O/U {line} {short_name} \u26d4 DISQUALIFIED: MINUTES_RESTRICTION")
                                    elif edge_data['valuable']:
                                        raw_conf = edge_data['confidence'] if edge_data['direction'] == "Over" else (100 - edge_data['confidence'])
                                        best_conf = min(raw_conf, MAX_PROP_CONFIDENCE)
                                        prop_details.append(f"O/U {line} {short_name} VALUABLE: {edge_data['direction']} ({best_conf:.0f}% conf (raw: {raw_conf:.0f}%), {abs(edge_data['edge_pct']):.1f}% edge)")
                                    else:
                                        prop_details.append(f"O/U {line} {short_name} ignore")
                                        
                            if prop_details:
                                props_str = " (" + ", ".join(prop_details) + ")"
                                
                        print(f"  {p:20} -> {median_pts} PTS, {median_reb} REB, {median_ast} AST{props_str}")
                print("")

            # ── Issue 3 Fix: Skip unstable games from parlay pool ──────
            # If either team has a synergy collapse (3+ missing starters), the game is too
            # unpredictable for parlay inclusion. Projections are still displayed above.
            if home_collapse or away_collapse:
                collapse_team = home_team['teamTricode'] if home_collapse else away_team['teamTricode']
                print(f"  ⛔ [PARLAY EXCLUDED] {matchup_str} — {collapse_team} has Synergy Collapse. Too volatile for parlay picks.")
                payload['sim_results'] = {
                    'home_team': home_team_full,
                    'away_team': away_team_full,
                    'avg_home': avg_home,
                    'avg_away': avg_away,
                    'event_id': event_id,
                }
                payload['excluded'] = True
                return payload  # Exited early

            # ── Collect data for parlay builder ────────────────────
            if event_id:
                payload['score_dists'] = {
                    'home': home_scores_list,
                    'away': away_scores_list,
                }

            # Merge player distributions into global tracker
            for player, dists in prop_tracker.items():
                if player not in payload['prop_dists']:
                    payload['prop_dists'][player] = dists
                
                # Store minutes risk metrics
                if player in home_matrix:
                    payload['player_minutes'][player] = {
                        'team': home_team_full,
                        'mean': home_matrix[player].get('min_mean', 0), 
                        'std': home_matrix[player].get('min_std', 0),
                        'restriction': home_matrix[player].get('minutes_restriction_flag', None),
                        'usage': home_matrix[player].get('avg_fga', 0),
                        'usage_boosted': home_matrix[player].get('usage_boosted', False)
                    }
                elif player in away_matrix:
                    payload['player_minutes'][player] = {
                        'team': away_team_full,
                        'mean': away_matrix[player].get('min_mean', 0), 
                        'std': away_matrix[player].get('min_std', 0),
                        'restriction': away_matrix[player].get('minutes_restriction_flag', None),
                        'usage': away_matrix[player].get('avg_fga', 0),
                        'usage_boosted': away_matrix[player].get('usage_boosted', False)
                    }

            payload['sim_results'] = {
                'home_team': home_team_full,
                'away_team': away_team_full,
                'avg_home': avg_home,
                'avg_away': avg_away,
                'event_id': event_id,
            }

        finally:
            conn.close()
            payload['game_logs'] = output_buffer.getvalue()
            
    return payload

def generate_todays_projections():
    print("[+] Fetching Live Scoreboard/Odds for Today's NBA Games from ESPN...")
    
    # Pre-fetch live injuries — returns (raw_set, normalized_dict)
    injured_raw, injured_normalized = get_injured_players()

    # ── STEP 0: Fetch live odds from ESPN. THIS IS NOW OUR SOURCE OF TRUTH FOR MATCHUPS ──
    all_game_odds = {}
    odds_available = False
    try:
        all_game_odds = fetch_espn_odds()
        odds_available = len(all_game_odds) > 0
        if odds_available:
            print(f"    [+] ESPN odds loaded for {len(all_game_odds)} events:")
            for eid, ev in all_game_odds.items():
                print(f"        {ev['away_team']} @ {ev['home_team']}")
        else:
            print("    [!] ESPN returned 0 upcoming events (games may already be in progress).")
            return
    except Exception as e:
        print(f"[!] Could not fetch ESPN odds: {e}")
        return

    # Mock the `games` list format using ESPN odds so the rest of the script is unaffected
    from nba_api.stats.static import teams
    nba_teams = teams.get_teams()
    
    # ESPN sometimes uses shortened city names like "LA Clippers" / "LA Lakers"
    # which don't match the NBA API full names. This explicit map fixes those.
    ESPN_NAME_OVERRIDES = {
        'la clippers': 'Los Angeles Clippers',
        'la lakers': 'Los Angeles Lakers',
        'golden state': 'Golden State Warriors',
        'oklahoma city': 'Oklahoma City Thunder',
        'new orleans': 'New Orleans Pelicans',
        'new york': 'New York Knicks',
        'san antonio': 'San Antonio Spurs',
        'portland': 'Portland Trail Blazers',
        'utah': 'Utah Jazz',
        'memphis': 'Memphis Grizzlies',
        'charlotte': 'Charlotte Hornets',
    }
    
    def find_team_by_name(name):
        name_lower = name.lower().strip()
        
        # Step 0: Check explicit overrides for ESPN's non-standard names
        if name_lower in ESPN_NAME_OVERRIDES:
            canonical = ESPN_NAME_OVERRIDES[name_lower].lower()
            for t in nba_teams:
                if t['full_name'].lower() == canonical:
                    return t
        
        # Step 1: Exact full name or nickname match
        for t in nba_teams:
            if t['full_name'].lower() == name_lower or t['nickname'].lower() == name_lower:
                return t
        
        # Step 2: Whole-word match only (avoids "la" matching "atlanta")
        # Split both into word sets and require the NICKNAME to match exactly
        name_words = set(name_lower.split())
        for t in nba_teams:
            full_words = set(t['full_name'].lower().split())
            # Must match the team's NICKNAME word exactly (e.g. "clippers", "hawks")
            if t['nickname'].lower() in name_words:
                return t
            # Or the team's CITY word exactly in the query
            city_words = set(t['city'].lower().split())
            if city_words and city_words.issubset(name_words):
                return t
        
        return None

    games = []
    for odds_dict in all_game_odds.values():
        h_data = find_team_by_name(odds_dict['home_team'])
        a_data = find_team_by_name(odds_dict['away_team'])
        if h_data and a_data:
            games.append({
                'homeTeam': {
                    'teamId': h_data['id'],
                    'teamName': h_data['nickname'],
                    'teamCity': h_data['city'],
                    'teamTricode': h_data['abbreviation']
                },
                'awayTeam': {
                    'teamId': a_data['id'],
                    'teamName': a_data['nickname'],
                    'teamCity': a_data['city'],
                    'teamTricode': a_data['abbreviation']
                }
            })
            
    if not games:
        print("[!] No matched NBA games found between ESPN and local DB.")
        return

    # ── STEP 0b: Fetch Odds API event IDs for player prop lookups ────
    odds_api_events = {}
    try:
        odds_api_events = fetch_event_ids()
    except Exception as e:
        print(f"[!] Could not fetch Odds API event list: {e}")
        print("    Player props will not be available.")

    conn = get_connection()

    # Accumulators for parlay builder (collected across ALL games)
    all_sim_results = []
    all_player_props = {}          # event_id -> player props
    all_prop_distributions = {}    # player_name -> {'PTS': [...], 'REB': [...], 'AST': [...]}
    all_score_distributions = {}   # event_id -> {'home': [...], 'away': [...]}
    all_player_minutes = {}        # player_name -> {'mean': X, 'std': Y}
    event_id_map = {}              # event_id -> game info for display


    # Run in parallel
    print("\n[+] Commencing Parallel Matchup Simulations...")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_live_game, game, odds_available, all_game_odds, odds_api_events, injured_normalized) for game in games]
        
        for future in as_completed(futures):
            try:
                res = future.result()
                # Print the buffered output for this game
                sys.stdout.write(res['game_logs'])
                
                # Merge accumulators
                if res['sim_results']:
                    all_sim_results.append(res['sim_results'])
                
                if res['event_id']:
                    event_id_map[res['event_id']] = {
                        'home_team': res['home_team_full'],
                        'away_team': res['away_team_full']
                    }
                    if res.get('score_dists'):
                        all_score_distributions[res['event_id']] = res['score_dists']
                    if res.get('cached_props'):
                        all_player_props[res['event_id']] = res['cached_props']
                        
                for p, dist in res['prop_dists'].items():
                    all_prop_distributions[p] = dist
                for p, min_data in res['player_minutes'].items():
                    all_player_minutes[p] = min_data
            except Exception as e:
                print(f"[-] Error parsing game from thread: {e}")
                import traceback
                traceback.print_exc()


    conn.close()

    # ══════════════════════════════════════════════════════════════
    #  PARLAY GENERATION — After ALL simulations are complete
    # ══════════════════════════════════════════════════════════════
    if odds_available and all_game_odds and all_score_distributions:
        print("\n" + "=" * 60)
        print("  🎯 GENERATING OPTIMAL PARLAYS...")
        print("=" * 60)

        try:
            # Compute historical hit rates for all players with prop lines
            all_historical_rates = {}
            for eid, props in all_player_props.items():
                for player_name, player_data in props.items():
                    if player_name in all_historical_rates:
                        continue
                    prop_lines = {}
                    stat_map = {'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST'}
                    for stat_key, sim_key in stat_map.items():
                        if stat_key in player_data:
                            prop_lines[sim_key] = player_data[stat_key]['line']
                    if prop_lines:
                        all_historical_rates[player_name] = get_player_historical_hit_rates(
                            player_name, prop_lines, limit=20
                        )

            parlays = build_parlays(
                all_sim_results=all_sim_results,
                all_game_odds=all_game_odds,
                all_player_props=all_player_props,
                all_prop_distributions=all_prop_distributions,
                all_score_distributions=all_score_distributions,
                all_historical_rates=all_historical_rates,
                all_player_minutes=all_player_minutes,
            )

            # Ingest shadow picks for research (Gate 4.C)
            shadow_data = parlays.get('shadow_picks', [])
            if shadow_data:
                game_date = datetime.now().strftime("%Y-%m-%d")
                for sp in shadow_data:
                    sp['game_date'] = game_date
                n_shadow = ingest_shadow_picks(shadow_data)
                print(f"\n[+] Logged {n_shadow} filtered picks (REBOUNDS, Banned, Under-EV) to shadow_picks table.")

            if getattr(args, 'shadow', False):
                game_date = datetime.now().strftime("%Y-%m-%d")
                ml_shadows = []
                for p in parlays.get('all_picks', []):
                    if p.get('type') == 'prop':
                        ml_shadows.append({
                            'game_date': game_date,
                            'player': p.get('player'),
                            'stat_category': p.get('stat_category'),
                            'direction': p.get('direction'),
                            'line': p.get('line'),
                            'projection': p.get('projection'),
                            'sim_confidence': p.get('confidence'),
                            'edge_pct': p.get('edge_pct'),
                            'ban_reason': 'ML_SHADOW'
                        })
                if ml_shadows:
                    n_ml = ingest_shadow_picks(ml_shadows)
                    print(f"\n[+] Logged {n_ml} ML-modified accepted picks as ML_SHADOW.")

            output = format_parlay_output(parlays)
            print(output)

        except Exception as e:
            print(f"\n[!] Error generating parlays: {e}")
            import traceback
            traceback.print_exc()
    elif not odds_available:
        if not all_game_odds:
            print("\n[!] Skipping parlay generation — Odds API returned no upcoming NBA events.")
            print("    This can happen when games are in progress or the key has no remaining credits.")
            print("    If you haven't set your key: create advanced_model/.env with ODDS_API_KEY=your_key")
        else:
            print("\n[!] Skipping parlay generation — no score distributions collected.")
    else:
        print("\n[!] No matched events found for parlay generation.")
        print(f"    Odds events: {list(all_game_odds.keys())[:3]}")
        print(f"    Sims collected for: {list(all_score_distributions.keys())[:3]}")

class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        self.buffer = ""
        self.skip_keywords = [
            "[Synergy]",
            "[MINUTES RESTRICTION]",
            "[B2B FATIGUE]",
            "[TEAM RATINGS]",
            "[Calculated Game Pace:",
            "[BLOWOUT SHIFT]",
            "[CURRENT ROSTER RATING]",
            "[USAGE BOOST]",
            "[INJURY MATRIX]",
            "[SYNERGY COLLAPSE]",
            "[PARLAY EXCLUDED]"
        ]

    def write(self, message):
        self.terminal.write(message)
        self.buffer += message
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            line_with_nl = line + '\n'
            if any(kw in line_with_nl for kw in self.skip_keywords):
                continue
            self.log.write(line_with_nl)

    def flush(self):
        self.terminal.flush()
        if self.buffer:
            if not any(kw in self.buffer for kw in self.skip_keywords):
                self.log.write(self.buffer)
            self.buffer = ""
        self.log.flush()

if __name__ == '__main__':
    import argparse
    import config.settings as cfg_settings
    
    parser = argparse.ArgumentParser(description='NBA Projection Engine')
    parser.add_argument('--date', type=str, help='Historical date to simulate (YYYY-MM-DD). Omit for today\'s live projections.')
    parser.add_argument('--shadow', action='store_true', help='Run in ML shadow mode, modifying confidences and saving as shadow picks.')
    args = parser.parse_args()

    if args.shadow:
        cfg_settings.USE_ML_CALIBRATION = True
        cfg_settings.USE_ML_RESIDUAL_CORRECTION = False  # Residuals failed the gate. Calibration passed.
        print("\n[!] RUNNING IN SHADOW MODE: ML Calibration ENABLED.")

    if args.date:
        date_obj = datetime.strptime(args.date, '%Y-%m-%d')
        month_name = date_obj.strftime('%B').lower()
        folder_path = os.path.join(MODEL_DIR, 'predictions', month_name)
        os.makedirs(folder_path, exist_ok=True)
        log_path = os.path.join(folder_path, f'predicts_{args.date}.txt')
    else:
        log_name = 'shadow_predicts.txt' if getattr(args, 'shadow', False) else 'predicts.txt'
        log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), log_name)
        
    sys.stdout = TeeLogger(log_path)    
    if args.date:
        generate_historical_projections(args.date)
        
        # Automatically generate the PDF/TeX report for the historical date
        print(f"\n[+] Generating historical PDF report for {args.date}...")
        try:
            import subprocess
            # call predicts_to_pdf.py --input predicts_{args.date}.txt --output predicts_report_{args.date}
            subprocess.run([
                sys.executable, 
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predicts_to_pdf.py'),
                '--input', log_path, # Use the absolute path since we saved it in folder_path
                '--output', f'predicts_report_{args.date}'
            ], check=True)
            print(f"    [+] Successfully generated predicts_report_{args.date}.tex")
        except Exception as e:
            print(f"    [-] Failed to generate PDF report: {e}")
            
    else:
        generate_todays_projections()
        
        # Archive it automatically
        import shutil
        today = datetime.now()
        month_name = today.strftime('%B').lower()
        today_str = today.strftime('%Y-%m-%d')
        
        archive_dir = os.path.join(MODEL_DIR, 'predictions', month_name)
        os.makedirs(archive_dir, exist_ok=True)
        archive_path = os.path.join(archive_dir, f'predicts_{today_str}.txt')
        
        try:
            shutil.copy(log_path, archive_path)
            print(f"\n[+] Archived live projections to: {archive_path}")
        except Exception as e:
            print(f"\n[-] Failed to archive predictions: {e}")

