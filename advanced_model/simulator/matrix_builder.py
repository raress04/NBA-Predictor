import sqlite3
import pandas as pd
import numpy as np
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'nba_data.db')

# ── League-Average Priors (2024-25 season averages) ──────────────
# Used for Bayesian regression — small-sample player stats are pulled
# toward these values to reduce noise from hot/cold streaks.
LEAGUE_PRIORS = {
    'fg2_pct': 0.525,      # league-avg 2pt FG%
    'fg3_pct': 0.145,      # league-avg 3PM per total FGA (≈ 0.363 3PT% × 0.40 freq)
                           # NOTE: Markov engine stores fg3_pct as fg3m/fga, then divides
                           # by p_shoot_3 internally to get per-attempt probability
    'ft_pct':  0.782,      # league-avg FT%
    'tov_rate': 0.130,     # league-avg TOV rate per possession
    'p_shoot_3': 0.40,     # league-avg fraction of FGA that are 3s
    'fta_per_fga': 0.27,   # league-avg FTA / FGA ratio
}

# Prior strength: how many "virtual games" the league average represents.
# Lower = trust the player's own data more. With 15 games and prior=5,
# the player's own stats carry 75% weight vs 25% league average.
PRIOR_STRENGTH = 5


def _bayesian_shrink(player_value: float, league_mean: float, n_games: int) -> float:
    """
    Bayesian shrinkage: pulls player_value toward league_mean.
    With few games, it leans heavily on the prior.
    With many games, the player's own data dominates.
    """
    return (player_value * n_games + league_mean * PRIOR_STRENGTH) / (n_games + PRIOR_STRENGTH)


def get_connection():
    return sqlite3.connect(DB_PATH)


def get_player_stats_matrix(player_name: str, limit: int = 82, before_date: str = None,
                             current_team_id: int = None) -> dict:
    '''
    Retrieves the last N games for a specific player and calculates their
    percentage weights for the Markov Simulation Engine.
    '''
    conn = get_connection()
    
    date_filter = "AND game_date < ?" if before_date else ""
    params = [player_name]
    if before_date: params.append(before_date)
    params.append(limit)

    # Phase 1: Try current + previous season (2024-25 onwards) for fresh data
    query = f'''
    SELECT 
        fgm, fga, fg3m, fg3a, ftm, fta, 
        oreb, dreb, ast, stl, blk, tov, pts, minutes,
        team_id, game_date
    FROM box_scores
    WHERE player_name = ? AND minutes IS NOT NULL AND minutes != '0:00'
    AND season >= '2024-25' {date_filter}
    ORDER BY game_date DESC
    LIMIT ?
    '''
    df = pd.read_sql_query(query, conn, params=params)
    
    # Phase 2: If insufficient data (<10 games), extend to 2023-24
    if len(df) < 10:
        query_fallback = f'''
        SELECT 
            fgm, fga, fg3m, fg3a, ftm, fta, 
            oreb, dreb, ast, stl, blk, tov, pts, minutes,
            team_id, game_date
        FROM box_scores
        WHERE player_name = ? AND minutes IS NOT NULL AND minutes != '0:00'
        AND season >= '2023-24' {date_filter}
        ORDER BY game_date DESC
        LIMIT ?
        '''
        df = pd.read_sql_query(query_fallback, conn, params=params)
    
    conn.close()

    if df.empty:
        raise ValueError(f"No stats found for {player_name} in the local database. Ensure backfiller has run.")

    from config.settings import TRADE_CONTEXT_MIN_GAMES, TRADE_CONTEXT_BLEND_GAMES, EWMA_SPAN

    # Sort ascending so EWMA sees oldest → newest
    df = df.sort_values('game_date').reset_index(drop=True)

    # ── Trade Context ─────────────────────────────────────────────
    # If current_team_id is provided, filter to post-trade games when possible.
    high_uncertainty = False
    if current_team_id is not None and 'team_id' in df.columns:
        post_trade = df[df['team_id'] == current_team_id]
        n_post = len(post_trade)
        if n_post >= TRADE_CONTEXT_MIN_GAMES:
            df = post_trade.reset_index(drop=True)
            high_uncertainty = (n_post < TRADE_CONTEXT_BLEND_GAMES)
        else:
            # Insufficient post-trade data — use full log but flag it
            high_uncertainty = True

    n_games = len(df)

    # ── EWMA Recency Weighting ────────────────────────────────────
    # Exponential weights replace the old linear 2.0→1.0 scheme.
    # span=EWMA_SPAN means the last ~10 games carry ~63% of total weight.
    # Values are per-game means from the EWMA tail (iloc[-1]).
    ewma_kw = dict(span=EWMA_SPAN, adjust=False)
    ewma_fgm  = df['fgm'].ewm(**ewma_kw).mean().iloc[-1]
    ewma_fga  = df['fga'].ewm(**ewma_kw).mean().iloc[-1]
    ewma_fg3m = df['fg3m'].ewm(**ewma_kw).mean().iloc[-1]
    ewma_fg3a = df['fg3a'].ewm(**ewma_kw).mean().iloc[-1]
    ewma_ftm  = df['ftm'].ewm(**ewma_kw).mean().iloc[-1]
    ewma_fta  = df['fta'].ewm(**ewma_kw).mean().iloc[-1]
    ewma_oreb = df['oreb'].ewm(**ewma_kw).mean().iloc[-1]
    ewma_dreb = df['dreb'].ewm(**ewma_kw).mean().iloc[-1]
    ewma_ast  = df['ast'].ewm(**ewma_kw).mean().iloc[-1]
    ewma_stl  = df['stl'].ewm(**ewma_kw).mean().iloc[-1]
    ewma_blk  = df['blk'].ewm(**ewma_kw).mean().iloc[-1]
    ewma_tov  = df['tov'].ewm(**ewma_kw).mean().iloc[-1]
    ewma_pts  = df['pts'].ewm(**ewma_kw).mean().iloc[-1]

    # Reconstruct totals dict scaled to n_games for Bayesian shrinkage downstream
    # (shrinkage uses n_games as the effective count, so totals = mean * n_games)
    totals = {
        'fgm':  ewma_fgm  * n_games,
        'fga':  ewma_fga  * n_games,
        'fg3m': ewma_fg3m * n_games,
        'fg3a': ewma_fg3a * n_games,
        'ftm':  ewma_ftm  * n_games,
        'fta':  ewma_fta  * n_games,
        'oreb': ewma_oreb * n_games,
        'dreb': ewma_dreb * n_games,
        'ast':  ewma_ast  * n_games,
        'stl':  ewma_stl  * n_games,
        'blk':  ewma_blk  * n_games,
        'tov':  ewma_tov  * n_games,
        'pts':  ewma_pts  * n_games,
    }
    weight_sum = n_games  # used downstream for avg_* division


    # Base Metrics (avoid divide by zero)
    total_fga = totals['fga'] if totals['fga'] > 0 else 1
    total_fg3a = totals['fg3a'] if totals['fg3a'] > 0 else 1
    total_fta = totals['fta'] if totals['fta'] > 0 else 1

    matrix = {}

    # 1. Shot Distribution
    raw_p_shoot_3 = totals['fg3a'] / total_fga
    matrix['p_shoot_3'] = _bayesian_shrink(raw_p_shoot_3, LEAGUE_PRIORS['p_shoot_3'], n_games)
    matrix['p_shoot_2'] = 1.0 - matrix['p_shoot_3']

    # 2. Shooting Percentages (Bayesian-regressed)
    raw_fg2 = (totals['fgm'] - totals['fg3m']) / (total_fga - totals['fg3a']) if (total_fga - totals['fg3a']) > 0 else 0.0
    # fg3_pct stored as fg3m/total_fga (NOT fg3m/fg3a) — this is what the Markov engine expects.
    # The engine divides by p_shoot_3 internally to get the per-attempt make probability.
    raw_fg3 = totals['fg3m'] / total_fga
    raw_ft  = totals['ftm'] / total_fta

    matrix['fg2_pct'] = _bayesian_shrink(raw_fg2, LEAGUE_PRIORS['fg2_pct'], n_games)
    matrix['fg3_pct'] = _bayesian_shrink(raw_fg3, LEAGUE_PRIORS['fg3_pct'], n_games)
    matrix['ft_pct']  = _bayesian_shrink(raw_ft,  LEAGUE_PRIORS['ft_pct'],  n_games)

    # 3. Aggressiveness (Bayesian-regressed)
    raw_fta_per_fga = total_fta / total_fga
    matrix['fta_per_fga'] = _bayesian_shrink(raw_fta_per_fga, LEAGUE_PRIORS['fta_per_fga'], n_games)

    # 4. Ball Security & Playmaking Rates
    possessions_used = total_fga + (0.44 * total_fta) + totals['tov']
    if possessions_used == 0:
        possessions_used = 1

    raw_tov_rate = totals['tov'] / possessions_used
    matrix['tov_rate'] = _bayesian_shrink(raw_tov_rate, LEAGUE_PRIORS['tov_rate'], n_games)

    # 5. Per-game averages from EWMA (directly, not divided from totals)
    matrix['avg_reb'] = ewma_oreb + ewma_dreb
    matrix['avg_ast'] = ewma_ast
    matrix['avg_stl'] = ewma_stl
    matrix['avg_blk'] = ewma_blk
    matrix['avg_pts'] = ewma_pts
    matrix['avg_fga'] = ewma_fga

    # 6. Minutes variance
    def parse_minutes(m):
        if pd.isna(m): return 0.0
        if isinstance(m, str) and ':' in m:
            parts = m.split(':')
            return float(parts[0]) + float(parts[1])/60.0
        try:
            return float(m)
        except:
            return 0.0

    df['min_float'] = df['minutes'].apply(parse_minutes)
    matrix['min_std'] = df['min_float'].std() if n_games > 1 else 0.0
    matrix['min_mean'] = df['min_float'].mean()

    # Store metadata for downstream usage rate adjustment
    matrix['_n_games']           = n_games
    matrix['_high_uncertainty']  = high_uncertainty

    return matrix

def get_team_pace(team_id: int, limit: int = 15, before_date: str = None) -> float:
    '''
    Calculates the average number of offensive possessions a team uses per game.
    Basic Formula: FGA + 0.44 * FTA + TOV - OREB
    '''
    conn = get_connection()
    date_filter = "AND game_date < ?" if before_date else ""
    params = [team_id]
    if before_date: params.append(before_date)
    params.append(limit)

    query = f'''
    SELECT 
        game_id,
        SUM(fga) as fga,
        SUM(fta) as fta,
        SUM(tov) as tov,
        SUM(oreb) as oreb
    FROM box_scores
    WHERE team_id = ? {date_filter}
    GROUP BY game_id
    ORDER BY game_date DESC
    LIMIT ?
    '''
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if df.empty:
        return 100.0 # Default fallback
        
    df['possessions'] = df['fga'] + (0.44 * df['fta']) + df['tov'] - df['oreb']
    return df['possessions'].mean()


def get_team_efficiency(team_id: int, limit: int = 15, before_date: str = None) -> dict:
    conn = get_connection()
    df_filter = "AND game_date < ?" if before_date else ""
    params = [team_id]
    if before_date: params.append(before_date)
    params.append(limit)
    
    # Get team's points scored and possessions per game
    off_query = f'''
    SELECT 
        b.game_id,
        SUM(b.pts) as pts_scored,
        SUM(b.fga) as fga,
        SUM(b.fta) as fta,
        SUM(b.tov) as tov,
        SUM(b.oreb) as oreb
    FROM box_scores b
    WHERE b.team_id = ? AND b.minutes IS NOT NULL AND b.minutes != '0:00'
    {df_filter}
    GROUP BY b.game_id
    ORDER BY b.game_date DESC
    LIMIT ?
    '''
    off_df = pd.read_sql_query(off_query, conn, params=params)
    
    # Get opponent's points scored against this team
    def_query = f'''
    SELECT 
        b.game_id,
        SUM(b.pts) as pts_allowed,
        SUM(b.fga) as fga,
        SUM(b.fta) as fta,
        SUM(b.tov) as tov,
        SUM(b.oreb) as oreb
    FROM box_scores b
    WHERE b.game_id IN (
        SELECT DISTINCT game_id FROM box_scores WHERE team_id = ?
        {df_filter}
        ORDER BY game_date DESC LIMIT ?
    )
    AND b.team_id != ?
    AND b.minutes IS NOT NULL AND b.minutes != '0:00'
    GROUP BY b.game_id
    ORDER BY b.game_date DESC
    '''
    def_df = pd.read_sql_query(def_query, conn, params=(team_id, before_date, limit, team_id) if before_date else (team_id, limit, team_id))
    conn.close()
    
    result = {'off_multiplier': 1.0, 'def_multiplier': 1.0}
    
    if off_df.empty or def_df.empty:
        return result
    
    # Offensive rating: points per 100 possessions
    off_df['poss'] = off_df['fga'] + (0.44 * off_df['fta']) + off_df['tov'] - off_df['oreb']
    off_df['off_rtg'] = (off_df['pts_scored'] / off_df['poss']) * 100
    team_off_rtg = off_df['off_rtg'].mean()
    
    # Defensive rating: opponent points per 100 possessions
    def_df['poss'] = def_df['fga'] + (0.44 * def_df['fta']) + def_df['tov'] - def_df['oreb']
    def_df['def_rtg'] = (def_df['pts_allowed'] / def_df['poss']) * 100
    team_def_rtg = def_df['def_rtg'].mean()
    
    # League average is approximately 112 pts per 100 possessions
    LEAGUE_AVG_RTG = 112.0
    
    raw_off = team_off_rtg / LEAGUE_AVG_RTG if LEAGUE_AVG_RTG > 0 else 1.0
    raw_def = team_def_rtg / LEAGUE_AVG_RTG if LEAGUE_AVG_RTG > 0 else 1.0
    
    # Clamp multipliers to prevent extreme inflation/deflation from outlier defensive ratings.
    # Without capping, a 119 DefRtg team yields 1.063x which compounds with pace to over-inflate totals.
    result['off_multiplier'] = max(0.93, min(1.07, raw_off))
    result['def_multiplier'] = max(0.93, min(1.07, raw_def))
    result['off_rtg'] = team_off_rtg
    result['def_rtg'] = team_def_rtg
    
    return result


def get_team_rebound_efficiency(team_id: int, limit: int = 15, before_date: str = None) -> dict:
    conn = get_connection()
    df_filter = "AND game_date < ?" if before_date else ""
    
    query = f'''
    SELECT 
        b.team_id,
        SUM(b.dreb) * 1.0 / (SUM(b.dreb) + SUM(o.oreb)) as dreb_pct,
        SUM(b.oreb) * 1.0 / (SUM(b.oreb) + SUM(o.dreb)) as oreb_pct
    FROM box_scores b
    JOIN box_scores o ON b.game_id = o.game_id AND b.team_id != o.team_id
    WHERE b.game_id IN (
        SELECT DISTINCT game_id FROM box_scores WHERE team_id = ?
        {df_filter}
        ORDER BY game_date DESC LIMIT ?
    )
    GROUP BY b.team_id
    '''
    params = (team_id, before_date, limit) if before_date else (team_id, limit)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if df.empty:
        return {'dreb_pct': 0.72, 'oreb_pct': 0.28, 'rebound_modifier': 1.0}
        
    target_row = df[df['team_id'] == team_id]
    if target_row.empty:
        return {'dreb_pct': 0.72, 'oreb_pct': 0.28, 'rebound_modifier': 1.0}
        
    team_dreb = target_row.iloc[0]['dreb_pct']
    team_oreb = target_row.iloc[0]['oreb_pct']
    
    # Calculate league averages roughly from current distribution
    league_dreb = df['dreb_pct'].mean() if len(df) > 1 else 0.72
    
    # Dynamic Scale: If the opponent grabs 76% of defensive rebounds (elite vs avg 72%),
    # the multiplier for our players becomes (1.0 - (0.76 - 0.72) * 2.5) ~ 0.90x
    # If the opponent grabs 68% of defensive rebounds (terrible),
    # the multiplier becomes (1.0 - (0.68 - 0.72) * 2.5) ~ 1.10x
    diff = team_dreb - league_dreb
    modifier = 1.0 - (diff * 2.5)
    
    # Clip extreme values to keep simulation stable
    modifier = float(np.clip(modifier, 0.85, 1.15))
    
    return {
        'dreb_pct': team_dreb,
        'oreb_pct': team_oreb,
        'rebound_modifier': modifier
    }


def get_player_historical_hit_rates(player_name: str, lines: dict, limit: int = 20) -> dict:
    '''
    Given a player name and a dict of prop lines like {'PTS': 25.5, 'REB': 7.5, 'AST': 5.5},
    query the last N games from the CURRENT SEASON and return split-window hit rates.
    
    Returns per stat:
        hit:        hits over last 20 games
        total:      total games in window
        pct:        hit% over last 20 games (extended)
        recent_hit: hits over last 10 games
        recent_total: games in recent window
        recent_pct: hit% over last 10 games
    Falls back to include previous season if current season has <10 games.
    '''
    conn = get_connection()

    def _fetch(lim):
        q = '''
        SELECT pts, reb, ast, oreb + dreb as total_reb
        FROM box_scores
        WHERE player_name = ? AND minutes IS NOT NULL AND minutes != '0:00'
        AND season = (SELECT MAX(season) FROM box_scores)
        ORDER BY game_date DESC
        LIMIT ?
        '''
        result = pd.read_sql_query(q, conn, params=(player_name, lim))
        if len(result) < 5:
            # Fallback to include previous season
            q_fb = '''
            SELECT pts, reb, ast, oreb + dreb as total_reb
            FROM box_scores
            WHERE player_name = ? AND minutes IS NOT NULL AND minutes != '0:00'
            AND season >= '2023-24'
            ORDER BY game_date DESC
            LIMIT ?
            '''
            result = pd.read_sql_query(q_fb, conn, params=(player_name, lim))
        return result

    df_extended = _fetch(limit)          # last 20 games
    df_recent   = _fetch(min(10, limit)) # last 10 games
    conn.close()

    if df_extended.empty:
        return {}

    stat_col_map = {'PTS': 'pts', 'REB': 'total_reb', 'AST': 'ast'}
    results = {}

    for stat_key, line_val in lines.items():
        col = stat_col_map.get(stat_key)
        if col is None or col not in df_extended.columns:
            continue

        # Extended window (last 20)
        total_ext   = len(df_extended)
        hit_ext     = int((df_extended[col] > line_val).sum())
        pct_ext     = (hit_ext / total_ext * 100) if total_ext > 0 else 0.0

        # Recent window (last 10)
        total_rec   = len(df_recent)
        hit_rec     = int((df_recent[col] > line_val).sum()) if not df_recent.empty else 0
        pct_rec     = (hit_rec / total_rec * 100) if total_rec > 0 else pct_ext

        results[stat_key] = {
            'hit':          hit_ext,
            'total':        total_ext,
            'pct':          pct_ext,       # extended (last 20)
            'recent_hit':   hit_rec,
            'recent_total': total_rec,
            'recent_pct':   pct_rec,       # recent (last 10)
        }

    return results


if __name__ == '__main__':
    try:
        jayson_tatum = get_player_stats_matrix('Jayson Tatum', limit=15)
        print("--- Jayson Tatum Matrix (Last 15 Games) ---")
        for k, v in jayson_tatum.items():
            print(f"{k}: {v:.3f}")
    except ValueError as e:
        print(f"Test failed: {e}")


