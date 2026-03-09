"""
Backtester — Runs the simulation engine on historical games from the database
and compares projected outcomes to actual results.

Measures:
  1. Score accuracy: Mean Absolute Error of projected vs actual scores
  2. Spread calibration: % of games where projected spread direction matched reality
  3. Confidence bucket calibration: do 70% confidence picks actually hit 70%?

Usage:
  python3 -m simulator.backtester --games 50
"""

import argparse
import sqlite3
import os
import sys
import random
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.matrix_builder import get_player_stats_matrix, get_team_pace, get_team_efficiency, get_connection
from simulator.markov_engine import MarkovSimulator

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'database', 'nba_data.db')


def get_historical_games(n_games: int = 50) -> List[dict]:
    """
    Pulls the most recent N completed games from the database.
    Returns list of dicts with actual scores and team/player info.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get unique games with actual scores
    cursor.execute('''
        SELECT g.game_id, g.game_date, g.matchup,
               bs_home.team_id as home_team_id,
               SUM(CASE WHEN bs_home.team_id = bs_home.team_id THEN bs_home.pts ELSE 0 END) as home_pts,
               bs_away.team_id as away_team_id
        FROM games g
        JOIN box_scores bs_home ON g.game_id = bs_home.game_id
        JOIN box_scores bs_away ON g.game_id = bs_away.game_id
        WHERE g.game_id IS NOT NULL
        GROUP BY g.game_id
        ORDER BY g.game_date DESC
        LIMIT ?
    ''', (n_games * 2,))  # Get extra to account for filtering

    # Simpler approach: get distinct games, then compute team scores
    cursor.execute('''
        SELECT game_id, game_date, matchup FROM games
        ORDER BY game_date DESC
        LIMIT ?
    ''', (n_games,))
    
    games = []
    for row in cursor.fetchall():
        game_id, game_date, matchup = row
        
        # Get team scores for this game
        cursor.execute('''
            SELECT team_id, SUM(pts) as total_pts
            FROM box_scores
            WHERE game_id = ? AND minutes IS NOT NULL AND minutes != '0:00'
            GROUP BY team_id
        ''', (game_id,))
        
        teams = cursor.fetchall()
        if len(teams) != 2:
            continue
        
        # Get top players by avg minutes for each team
        team1_id, team1_pts = teams[0]
        team2_id, team2_pts = teams[1]
        
        # Determine home/away from matchup string (contains '@' or 'vs.')
        if matchup and '@' in matchup:
            # Away team is before @
            games.append({
                'game_id': game_id,
                'game_date': game_date,
                'matchup': matchup,
                'home_team_id': team2_id,
                'away_team_id': team1_id,
                'actual_home_pts': team2_pts,
                'actual_away_pts': team1_pts,
            })
        else:
            games.append({
                'game_id': game_id,
                'game_date': game_date,
                'matchup': matchup,
                'home_team_id': team1_id,
                'away_team_id': team2_id,
                'actual_home_pts': team1_pts,
                'actual_away_pts': team2_pts,
            })
    
    conn.close()
    return games[:n_games]


def get_top_players_for_team(team_id: int, game_id: str) -> List[str]:
    """Get top 8 players by minutes for a specific team BEFORE this game."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT player_name, AVG(CAST(
            CASE WHEN minutes LIKE '%:%' THEN 
                CAST(SUBSTR(minutes, 1, INSTR(minutes, ':') - 1) AS FLOAT)
            ELSE 
                CAST(minutes AS FLOAT)
            END AS FLOAT)) as avg_min
        FROM box_scores
        WHERE team_id = ? AND game_id < ? AND minutes IS NOT NULL AND minutes != '0:00'
        GROUP BY player_name
        HAVING COUNT(*) >= 3
        ORDER BY avg_min DESC
        LIMIT 8
    ''', (team_id, game_id))
    
    players = [row[0] for row in cursor.fetchall()]
    conn.close()
    return players


def simulate_game(home_players: List[str], away_players: List[str],
                  home_team_id: int = None, away_team_id: int = None,
                  pace: int = 100, sims: int = 500) -> dict:
    """
    Run Monte Carlo simulation for a game and return projected scores.
    Applies team efficiency ratings for realistic score differentiation.
    """
    home_matrix = {}
    away_matrix = {}
    
    for hp in home_players:
        try:
            home_matrix[hp] = get_player_stats_matrix(hp, limit=15)
        except (ValueError, Exception):
            pass
    
    for ap in away_players:
        try:
            away_matrix[ap] = get_player_stats_matrix(ap, limit=15)
        except (ValueError, Exception):
            pass
    
    if not home_matrix or not away_matrix:
        return None
    
    # Apply team efficiency ratings
    if home_team_id and away_team_id:
        home_eff = get_team_efficiency(home_team_id)
        away_eff = get_team_efficiency(away_team_id)
        # Opponent's defensive rating scales our shooting
        for player in home_matrix:
            home_matrix[player]['fg2_pct'] *= away_eff['def_multiplier']
            home_matrix[player]['fg3_pct'] *= away_eff['def_multiplier']
        for player in away_matrix:
            away_matrix[player]['fg2_pct'] *= home_eff['def_multiplier']
            away_matrix[player]['fg3_pct'] *= home_eff['def_multiplier']
    
    engine = MarkovSimulator(home_matrix, away_matrix)
    
    total_home = 0
    total_away = 0
    for _ in range(sims):
        result = engine.run_full_game(pace=pace)
        total_home += result['home_score']
        total_away += result['away_score']
    
    return {
        'proj_home': total_home / sims,
        'proj_away': total_away / sims,
    }


def run_backtest(n_games: int = 50, sims_per_game: int = 100):
    """Main backtesting function."""
    print(f"\n{'='*60}")
    print(f"  📊 BACKTESTING ENGINE — Testing on {n_games} historical games")
    print(f"{'='*60}\n")
    
    games = get_historical_games(n_games)
    if not games:
        print("[-] No historical games found in database.")
        return
    
    print(f"[+] Loaded {len(games)} games from database.\n")
    
    results = []
    errors_home = []
    errors_away = []
    errors_total = []
    spread_correct = 0
    spread_total = 0
    
    # Confidence buckets: track how often our confidence level matches reality
    conf_buckets = {
        '50-60': {'predicted': 0, 'actual': 0},
        '60-70': {'predicted': 0, 'actual': 0},
        '70-80': {'predicted': 0, 'actual': 0},
        '80+':   {'predicted': 0, 'actual': 0},
    }
    
    for i, game in enumerate(games):
        game_id = game['game_id']
        
        # Get players that were active BEFORE this game
        home_players = get_top_players_for_team(game['home_team_id'], game_id)
        away_players = get_top_players_for_team(game['away_team_id'], game_id)
        
        if len(home_players) < 3 or len(away_players) < 3:
            continue
        
        try:
            # Get team pace
            home_pace = get_team_pace(game['home_team_id'])
            away_pace = get_team_pace(game['away_team_id'])
            pace = int((home_pace + away_pace) / 2)
        except Exception:
            pace = 100
        
        sim = simulate_game(home_players, away_players,
                           home_team_id=game['home_team_id'],
                           away_team_id=game['away_team_id'],
                           pace=pace, sims=sims_per_game)
        if sim is None:
            continue
        
        actual_home = game['actual_home_pts']
        actual_away = game['actual_away_pts']
        
        err_home = abs(sim['proj_home'] - actual_home)
        err_away = abs(sim['proj_away'] - actual_away)
        err_total = abs((sim['proj_home'] + sim['proj_away']) - (actual_home + actual_away))
        
        errors_home.append(err_home)
        errors_away.append(err_away)
        errors_total.append(err_total)
        
        # Spread direction check
        proj_spread = sim['proj_home'] - sim['proj_away']
        actual_spread = actual_home - actual_away
        if (proj_spread > 0 and actual_spread > 0) or (proj_spread < 0 and actual_spread < 0):
            spread_correct += 1
        spread_total += 1
        
        # Confidence tracking (how confident we were in the projected winner)
        proj_winner_margin = abs(proj_spread)
        if proj_winner_margin > 10:
            bucket = '80+'
        elif proj_winner_margin > 5:
            bucket = '70-80'
        elif proj_winner_margin > 2:
            bucket = '60-70'
        else:
            bucket = '50-60'
        
        conf_buckets[bucket]['predicted'] += 1
        if (proj_spread > 0 and actual_spread > 0) or (proj_spread < 0 and actual_spread < 0):
            conf_buckets[bucket]['actual'] += 1
        
        status = "✅" if (proj_spread > 0) == (actual_spread > 0) else "❌"
        print(f"  [{i+1}/{len(games)}] {game['matchup'][:35]:35s} "
              f"Proj: {sim['proj_home']:.0f}-{sim['proj_away']:.0f}  "
              f"Actual: {actual_home:.0f}-{actual_away:.0f}  "
              f"Err: {err_total:.1f}pts  {status}")
    
    # Summary
    if not errors_total:
        print("\n[-] No games could be simulated. Is the database populated?")
        return
    
    print(f"\n{'='*60}")
    print(f"  📈 BACKTESTING RESULTS ({spread_total} games simulated)")
    print(f"{'='*60}")
    print(f"\n  Score Accuracy:")
    print(f"    Mean Absolute Error (Total)   : {sum(errors_total)/len(errors_total):.1f} pts")
    print(f"    Mean Absolute Error (Home)    : {sum(errors_home)/len(errors_home):.1f} pts")
    print(f"    Mean Absolute Error (Away)    : {sum(errors_away)/len(errors_away):.1f} pts")
    print(f"\n  Spread Direction Accuracy:")
    pct = spread_correct / spread_total * 100 if spread_total > 0 else 0
    print(f"    Correctly predicted winner    : {spread_correct}/{spread_total} ({pct:.1f}%)")
    print(f"    (Random baseline = 50%)")
    
    print(f"\n  Confidence Calibration:")
    print(f"    {'Confidence':12s} {'Predicted':>10s} {'Correct':>10s} {'Hit Rate':>10s}")
    print(f"    {'-'*44}")
    for bucket, data in conf_buckets.items():
        if data['predicted'] > 0:
            hit = data['actual'] / data['predicted'] * 100
            print(f"    {bucket:12s} {data['predicted']:10d} {data['actual']:10d} {hit:9.1f}%")
    
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NBA Simulation Backtester')
    parser.add_argument('--games', type=int, default=50, help='Number of historical games to test')
    parser.add_argument('--sims', type=int, default=500, help='Simulations per game (default 500)')
    args = parser.parse_args()
    
    run_backtest(n_games=args.games, sims_per_game=args.sims)
