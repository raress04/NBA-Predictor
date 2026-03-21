import random
from typing import Dict, Tuple, Optional
from scipy.stats import beta as beta_dist

class MarkovSimulator:
    '''
    A possession-level Monte Carlo engine for NBA games.
    '''
    def __init__(self, home_roster_matrices: Dict[str, dict], away_roster_matrices: Dict[str, dict], is_fast_pace: bool = False):
        self.home_roster = home_roster_matrices
        self.away_roster = away_roster_matrices
        self.is_fast_pace = is_fast_pace
        
        # We calculate the proportional share of various events for all 10 players
        self.home_usage_weights = self._calculate_weights(self.home_roster, 'avg_fga')
        self.away_usage_weights = self._calculate_weights(self.away_roster, 'avg_fga')
        
        self.home_ast_weights = self._calculate_weights(self.home_roster, 'avg_ast')
        self.away_ast_weights = self._calculate_weights(self.away_roster, 'avg_ast')
        
        self.home_reb_weights = self._calculate_weights(self.home_roster, 'avg_reb', apply_variance=self.is_fast_pace)
        self.away_reb_weights = self._calculate_weights(self.away_roster, 'avg_reb', apply_variance=self.is_fast_pace)
        
        self.home_stl_weights = self._calculate_weights(self.home_roster, 'avg_stl')
        self.away_stl_weights = self._calculate_weights(self.away_roster, 'avg_stl')

    def _calculate_weights(self, roster: Dict[str, dict], key: str, apply_variance: bool = False) -> dict:
        weights = {}
        for name, p in roster.items():
            val = p.get(key, 0)
            if apply_variance and key == 'avg_reb' and val > 0:
                # Compress the distribution to increase variance (more random/loose balls)
                val = val ** 0.85
            weights[name] = val
            
        total = sum(weights.values())
        if total == 0:
            total = 1
        return {name: (v / total) for name, v in weights.items()}

    def _get_ball_handler(self, is_home: bool) -> str:
        weights = self.home_usage_weights if is_home else self.away_usage_weights
        players = list(weights.keys())
        probs = list(weights.values())
        return random.choices(players, weights=probs, k=1)[0]
        
    def simulate_possession(self, is_home: bool, stats_tracker: dict):
        roster = self.home_roster if is_home else self.away_roster
        player_name = self._get_ball_handler(is_home)
        matrix = roster[player_name]
        
        # Node 1: Does the possession end in a Turnover?
        if random.random() < matrix['tov_rate']:
            stats_tracker[player_name]['TOV'] += 1
            # Weighted Steal
            opp_weights = self.away_stl_weights if is_home else self.home_stl_weights
            stealer = random.choices(list(opp_weights.keys()), weights=list(opp_weights.values()), k=1)[0]
            stats_tracker[stealer]['STL'] += 1
            return 0 # 0 points scored
            
        # Node 2: The player chooses to shoot. Is it a 3PT or 2PT?
        is_three_pointer = random.random() < matrix['p_shoot_3']
        points_scored = 0
        
        if is_three_pointer:
            stats_tracker[player_name]['FG3A'] += 1
            if random.random() < matrix['fg3_pct'] / (matrix['p_shoot_3'] if matrix['p_shoot_3'] > 0 else 1):
                stats_tracker[player_name]['FG3M'] += 1
                stats_tracker[player_name]['PTS'] += 3
                points_scored += 3
        else:
            if random.random() < matrix['fg2_pct']:
                stats_tracker[player_name]['PTS'] += 2
                points_scored += 2

        # Node 3: Free Throw & Rebound Logic
        if points_scored > 0:
            # AND-1: ~5% of made baskets draw a foul → 1 bonus FT
            if random.random() < 0.05:
                stats_tracker[player_name]['FTA'] += 1
                if random.random() < matrix['ft_pct']:
                    stats_tracker[player_name]['FTM'] += 1
                    stats_tracker[player_name]['PTS'] += 1
                    points_scored += 1

            # Assist credit on made baskets (~69% of NBA FGs are assisted vs old 64% - Fixes low PG projections)
            if random.random() < 0.69:
                ast_weights = self.home_ast_weights if is_home else self.away_ast_weights
                passers = {k: v for k, v in ast_weights.items() if k != player_name}
                total_passer_weight = sum(passers.values())
                if total_passer_weight > 0:
                    probs = [v / total_passer_weight for v in passers.values()]
                    assister = random.choices(list(passers.keys()), weights=probs, k=1)[0]
                    stats_tracker[assister]['AST'] += 1
        else:
            # Missed shot — always generates a rebound
            shot_missed = True

            # SHOOTING FOUL on a miss: fta_per_fga chance → 2 FTs (or 3 for 3PT foul)
            if random.random() < matrix['fta_per_fga']:
                n_fts = 3 if is_three_pointer else 2
                stats_tracker[player_name]['FTA'] += n_fts
                made_fts = 0
                last_ft_made = True
                for i in range(n_fts):
                    if random.random() < matrix['ft_pct']:
                        stats_tracker[player_name]['FTM'] += 1
                        stats_tracker[player_name]['PTS'] += 1
                        points_scored += 1
                        made_fts += 1
                        last_ft_made = True
                    else:
                        last_ft_made = False

                # Rebound on missed LAST free throw
                if not last_ft_made:
                    if random.random() < 0.15:  # ~15% OREB on missed FT
                        reb_weights = self.home_reb_weights if is_home else self.away_reb_weights
                        rebounder = random.choices(list(reb_weights.keys()), weights=list(reb_weights.values()), k=1)[0]
                        stats_tracker[rebounder]['REB'] += 1
                        stats_tracker[rebounder]['OREB'] += 1
                    else:
                        opp_reb_weights = self.away_reb_weights if is_home else self.home_reb_weights
                        rebounder = random.choices(list(opp_reb_weights.keys()), weights=list(opp_reb_weights.values()), k=1)[0]
                        stats_tracker[rebounder]['REB'] += 1
                        stats_tracker[rebounder]['DREB'] += 1
            else:
                # No foul — rebound on the missed field goal
                if random.random() < 0.25:
                    # 25% Offensive Rebound
                    reb_weights = self.home_reb_weights if is_home else self.away_reb_weights
                    rebounder = random.choices(list(reb_weights.keys()), weights=list(reb_weights.values()), k=1)[0]
                    stats_tracker[rebounder]['REB'] += 1
                    stats_tracker[rebounder]['OREB'] += 1
                else:
                    # 75% Defensive Rebound
                    opp_reb_weights = self.away_reb_weights if is_home else self.home_reb_weights
                    rebounder = random.choices(list(opp_reb_weights.keys()), weights=list(opp_reb_weights.values()), k=1)[0]
                    stats_tracker[rebounder]['REB'] += 1
                    stats_tracker[rebounder]['DREB'] += 1
                
        return points_scored

    def run_full_game(self, pace: int = 100) -> dict:
        '''
        Runs exactly ~Pace possessions per team to recreate a full NBA box score.
        '''
        
        # Initialize an empty Box Score 0-tracker for every player
        stats_tracker = {}
        for player in list(self.home_roster.keys()) + list(self.away_roster.keys()):
            stats_tracker[player] = {'PTS': 0, 'REB': 0, 'OREB':0, 'DREB':0, 'AST': 0, 'STL': 0, 'BLK': 0, 'TOV': 0, 'FG3M': 0, 'FG3A': 0, 'FTA': 0, 'FTM': 0}
            
        home_score = 0
        away_score = 0
        
        # Apply a small atmospheric modifier to pace to account for second-chance points
        # and transition variance not fully captured by the possession model.
        # NOTE: Previously 1.08 which caused ~17pt total inflation. Reduced to 1.02.
        adjusted_pace = int(pace * 1.02)
        
        for _ in range(adjusted_pace):
            # Home Possession
            home_score += self.simulate_possession(is_home=True, stats_tracker=stats_tracker)
            # Away Possession
            away_score += self.simulate_possession(is_home=False, stats_tracker=stats_tracker)
            
        return {
            'home_score': home_score,
            'away_score': away_score,
            'player_stats': stats_tracker
        }

def compute_posterior_confidence(over_count: int, n_sim: int, prior_mean: float = 0.54, prior_strength: int = 10):
    """
    Return (posterior_mean, ci_width) for a Beta posterior.
    prior_mean: prior probability (e.g., 54% or per-category lookup).
    prior_strength: pseudo-counts (alpha+beta).
    """
    alpha_prior = prior_mean * prior_strength
    beta_prior = (1 - prior_mean) * prior_strength

    alpha_post = alpha_prior + over_count
    beta_post  = beta_prior + (n_sim - over_count)

    posterior_mean = alpha_post / (alpha_post + beta_post)
    
    # Calculate 90% confidence interval width
    ci_low, ci_high = beta_dist.interval(0.90, alpha_post, beta_post)
    ci_width = ci_high - ci_low
    
    return posterior_mean, ci_width
