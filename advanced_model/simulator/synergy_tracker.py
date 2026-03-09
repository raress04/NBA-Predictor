from nba_api.stats.endpoints import synergyplaytypes
from functools import lru_cache
import pandas as pd
import numpy as np
import time

# Cache defensive and league average stats to avoid slow redundant API calls during simulations
@lru_cache(maxsize=1)
def get_league_defensive_data(season='2025-26'):
    play_types = ['Isolation', 'PRBallHandler', 'Spotup', 'Transition', 'Handoff']
    team_defense = {}
    league_averages = {}
    
    print("[+] Caching NBA Synergy Defensive Play Types...")
    for pt in play_types:
        try:
            df = synergyplaytypes.SynergyPlayTypes(
                player_or_team_abbreviation='T',
                play_type_nullable=pt,
                type_grouping_nullable='defensive',
                season=season,
                per_mode_simple='PerGame',
                timeout=60
            ).get_data_frames()[0]
            time.sleep(1) # Be nice to API
            
            # PPP (Points Per Possession) is our primary defensive metric
            df = df.dropna(subset=['PPP'])
            
            league_avg_ppp = df['PPP'].mean()
            league_averages[pt] = league_avg_ppp
            
            # Map Team Name -> PPP for easy lookup
            for _, row in df.iterrows():
                team_name = row['TEAM_NAME']
                if team_name not in team_defense:
                    team_defense[team_name] = {}
                team_defense[team_name][pt] = row['PPP']
                
        except Exception as e:
            print(f"[-] Error tracking Synergy for {pt}: {e}")
            league_averages[pt] = 0.95 # Generic baseline if API fails
            
    return team_defense, league_averages

@lru_cache(maxsize=1)
def get_league_offensive_data(season='2025-26'):
    play_types = ['Isolation', 'PRBallHandler', 'Spotup', 'Transition', 'Handoff']
    league_offense = {}
    
    print("[+] Caching NBA Synergy Offensive Play Types (League-wide)...")
    for pt in play_types:
        try:
            df = synergyplaytypes.SynergyPlayTypes(
                player_or_team_abbreviation='P',
                play_type_nullable=pt,
                type_grouping_nullable='offensive',
                season=season,
                per_mode_simple='PerGame',
                timeout=60
            ).get_data_frames()[0]
            league_offense[pt] = df
            time.sleep(1) # Be nice to API
        except Exception as e:
            print(f"[-] Error caching {pt} offense: {e}")
            league_offense[pt] = pd.DataFrame()
            
    return league_offense


# ── Enhancement 7: Individual Defender Tracking ──────────────────
@lru_cache(maxsize=1)
def get_player_defensive_data(season='2025-26'):
    """
    Fetches individual player defensive PPP per play type.
    Returns: {player_name: {play_type: PPP}}
    This allows matching specific defenders (e.g., Kawhi on Luka)
    instead of just team-level averages.
    """
    play_types = ['Isolation', 'PRBallHandler', 'Spotup']  # Focus on most impactful
    player_defense = {}

    print("[+] Caching NBA Synergy Individual Defensive Data...")
    for pt in play_types:
        try:
            df = synergyplaytypes.SynergyPlayTypes(
                player_or_team_abbreviation='P',
                play_type_nullable=pt,
                type_grouping_nullable='defensive',
                season=season,
                per_mode_simple='PerGame',
                timeout=60
            ).get_data_frames()[0]
            time.sleep(1)

            df = df.dropna(subset=['PPP'])
            # Only keep players with enough possessions to be meaningful
            if 'POSS' in df.columns:
                df = df[df['POSS'] >= 20]

            for _, row in df.iterrows():
                pname = row.get('PLAYER_NAME', '')
                tname = row.get('TEAM_NAME', '')
                if not pname:
                    continue
                key = f"{pname}|{tname}"
                if key not in player_defense:
                    player_defense[key] = {}
                player_defense[key][pt] = row['PPP']

        except Exception as e:
            print(f"[-] Error fetching individual defensive synergy for {pt}: {e}")

    print(f"    Loaded individual defensive data for {len(player_defense)} player-team combos.")
    return player_defense


def _get_best_defender_ppp(opponent_team_name, play_type, player_defense):
    """
    Find the best individual defender on the opponent team for a specific play type.
    Returns their PPP if found, else None.
    """
    best_ppp = None
    for key, pt_data in player_defense.items():
        pname, tname = key.split('|', 1)
        if tname.lower() == opponent_team_name.lower() and play_type in pt_data:
            ppp = pt_data[play_type]
            if best_ppp is None or ppp < best_ppp:
                best_ppp = ppp
    return best_ppp


TEAM_WEIGHT = 0.70   # Weight for team-level defense
PLAYER_WEIGHT = 0.30  # Weight for individual defender

@lru_cache(maxsize=500)
def compute_matchup_multiplier(player_name, opponent_team_name, season='2025-26'):
    '''
    Returns a float multiplier (e.g., 0.92 for a tough defensive matchup, 1.08 for a weak one)
    Calculated by matching the player's offensive play-type payload against the opponent's 
    specific defensive PPP in those identical play sets.

    Enhancement 7: Also checks individual player defensive data for the opponent team.
    When a specific elite defender is identified, blends 70% team + 30% individual.
    '''
    team_defense, league_averages = get_league_defensive_data(season)
    league_offense = get_league_offensive_data(season)
    player_defense = get_player_defensive_data(season)
    
    if opponent_team_name not in team_defense:
        return 1.0 # Cannot calculate, neutral multiplier
        
    play_types = ['Isolation', 'PRBallHandler', 'Spotup', 'Transition', 'Handoff']
    player_payload = {}
    
    try:
        # Fetch the player's offensive volume in these play types from cached league data
        for pt in play_types:
            res = league_offense.get(pt, pd.DataFrame())
            
            if res.empty:
                player_payload[pt] = 0.0
                continue
                
            target_player = res[res['PLAYER_NAME'].str.lower() == player_name.lower()]
            if not target_player.empty:
                # Store the percentage of their offensive possessions this play type makes up
                poss_pct = target_player.iloc[0]['POSS_PCT']
                player_payload[pt] = poss_pct
            else:
                player_payload[pt] = 0.0
                
        if sum(player_payload.values()) == 0:
            return 1.0
            
    except Exception as e:
        print(f"[-] Error fetching Offensive Synergy for {player_name}: {e}")
        return 1.0

    # Calculate the Weighted Multiplier
    total_multiplier = 0.0
    total_tracked_freq = 0.0
    
    for pt, freq in player_payload.items():
        if freq > 0 and pt in team_defense[opponent_team_name] and pt in league_averages:
            team_ppp = team_defense[opponent_team_name][pt]
            avg_ppp = league_averages[pt]
            
            # Team-level multiplier
            team_mult = team_ppp / avg_ppp if avg_ppp > 0 else 1.0

            # Individual defender multiplier (Enhancement 7)
            defender_ppp = _get_best_defender_ppp(opponent_team_name, pt, player_defense)
            if defender_ppp is not None and avg_ppp > 0:
                player_mult = defender_ppp / avg_ppp
                # Blend team + individual
                pt_multiplier = (team_mult * TEAM_WEIGHT) + (player_mult * PLAYER_WEIGHT)
            else:
                pt_multiplier = team_mult
            
            total_multiplier += (pt_multiplier * freq)
            total_tracked_freq += freq
            
    # Normalize back up to 1.0 for untracked play types (putbacks, cuts, etc.)
    if total_tracked_freq > 0:
        weighted_modifier = total_multiplier / total_tracked_freq
        # Cap the modifier to prevent extreme edge cases from breaking math
        return float(np.clip(weighted_modifier, 0.80, 1.25))
    else:
        return 1.0

if __name__ == '__main__':
    # Test Jayson Tatum vs a good defense (Thunder) vs bad defense
    print(f"Jayson Tatum vs Oklahoma City Thunder: {compute_matchup_multiplier('Jayson Tatum', 'Oklahoma City Thunder'):.3f}x Metric")
    print(f"Jayson Tatum vs Charlotte Hornets: {compute_matchup_multiplier('Jayson Tatum', 'Charlotte Hornets'):.3f}x Metric")

