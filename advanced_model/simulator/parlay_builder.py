"""
Parlay Builder — Compares simulation projections against live sportsbook lines,
ranks edges, applies Masterclass rules, and constructs 2-3 pick and 4-pick parlays.
"""

import math
from typing import Dict, List, Tuple, Optional

# Stat-specific weights for blending Simulation Confidence and Player History DB Confidence.
# Assists are noisy, so we trust historical hit rate equal to the sim.
# Points and Totals are more reliable in Sims, so we weight Sim higher.
SIM_DB_WEIGHTS = {
    'assists': (0.50, 0.50),
    'rebounds': (0.60, 0.40),
    'points': (0.80, 0.20),
    'totals': (0.80, 0.20),
    'combo': (0.70, 0.30)
}

# Optional DB tracking for live calibration
try:
    from etl.bet_tracker import query_by_type
    _HAS_TRACKER = True
except ImportError:
    _HAS_TRACKER = False
    
_CALIBRATION_CACHE = None
def get_category_calibration(stat_category: str) -> float:
    """
    Returns a multiplier (<= 1.0) to scale down confidence if a category is underperforming.
    Needs at least 15 picks in the DB to trigger.
    """
    global _CALIBRATION_CACHE
    if not _HAS_TRACKER: return 1.0
    
    if _CALIBRATION_CACHE is None:
        try:
            stats = query_by_type()
            _CALIBRATION_CACHE = {row['stat_category']: row for row in stats}
        except Exception:
            _CALIBRATION_CACHE = {}
            
    cat_stats = _CALIBRATION_CACHE.get(stat_category.upper())
    if not cat_stats or cat_stats['total'] < 15:
        return 1.0
        
    win_rate = cat_stats['win_rate']
    # If a category is hitting < 50%, heavily penalize its confidence scores
    if win_rate < 50.0:
        return 0.85
    # If hitting 50-65%, slightly penalize
    elif win_rate < 65.0:
        return 0.95
    return 1.0


# ── Blowout Protocol ──────────────────────────────────────────
# If a team's market spread is > this threshold, ban player OVER props on the favorite
BLOWOUT_SPREAD_THRESHOLD = 12.0

# ── Safety Margin ───────────────────────────────────────────
# Reduce spreads by this many points (the Masterclass 2-3 pt reduction)
SPREAD_SAFETY_MARGIN = 3.0

# ── Minimum thresholds & Safety Margins per Category ──
EDGE_THRESHOLDS = {
    'totals': {'min_prob': 75.0, 'min_edge': 20.9, 'trigger': 4.5, 'safety': 2.5},
    'points': {'min_prob': 75.0, 'min_edge': 20.9, 'trigger': 2.5, 'safety': 1.75},
    'rebounds': {'min_prob': 75.0, 'min_edge': 13.9, 'trigger': 0, 'safety': 0}, # safety is dynamic (std dev)
    'assists': {'min_prob': 82.0, 'min_edge': 22.0, 'trigger': 0, 'safety': 0},
    'combo': {'min_prob': 70.0, 'min_edge': 15.9, 'trigger': 0, 'safety': 1.5},
    'spreads': {'min_prob': 75.0, 'min_edge': 10.0} # Raised from 60.0 to 75.0
}

# ── Line Movement ───────────────────────────────────────────
LINE_MOVEMENT_BONUS = 5.0   # Max +/-5% confidence adjustment

# ── Kelly Criterion & Calibration ───────────────────────────
KELLY_FRACTION = 0.25        # Fractional Kelly (Quarter Kelly)
BANKROLL_HARD_CAP = 0.025    # Max 2.5% of bankroll per parlay
CONFIDENCE_PRIOR_STRENGTH = 20 # Number of sims for Bayesian shrinkage
CONFIDENCE_PRIOR_P = 0.54    # Conservative prior for a coin-flip + vig
MAX_PROP_CONFIDENCE = 88.0   # Hard cap for single-leg player props


def calibrate_confidence(raw_prob: float, n_sims: int = 1000) -> float:
    """
    Applies Bayesian shrinkage and hard caps to prevent simulation hyperinflation.
    Pulls raw % toward a conservative prior (54%).
    """
    # Convert to 0.0-1.0 scale
    p = raw_prob / 100.0
    
    # Bayesian Shrinkage: p_calib = (p*n + prior_p*k) / (n + k)
    p_calibrated = (p * n_sims + CONFIDENCE_PRIOR_P * CONFIDENCE_PRIOR_STRENGTH) / (n_sims + CONFIDENCE_PRIOR_STRENGTH)
    
    # Convert back to 0-100
    calib_prob = p_calibrated * 100.0
    
    return min(calib_prob, 99.0)


def compute_kelly_stake(decimal_odds: float, probability: float) -> float:
    """
    Computes the optimal bankroll percentage to wager using the Kelly Criterion:
    f* = (bp - q) / b
    where:
      b = net decimal odds (decimal odds - 1)
      p = probability of winning (0.0 - 1.0)
      q = probability of losing (1 - p)
    """
    if decimal_odds <= 1.0 or probability <= 0:
        return 0.0
        
    b = decimal_odds - 1.0
    p = probability / 100.0
    q = 1.0 - p
    
    # Standard Kelly formula
    f_star = (b * p - q) / b
    
    # Apply Fractional Kelly and Hard Cap
    suggested_stake = max(0.0, f_star * KELLY_FRACTION)
    return min(suggested_stake, BANKROLL_HARD_CAP)


def compute_line_movement_bonus(game_odds: dict, pick_type: str, pick_direction: str) -> float:
    """
    Compare the sharp Pinnacle line vs. the consensus average across all books.
    If sharp money has moved TOWARD our pick direction, boost confidence.
    If it moved AGAINST, penalize.

    Returns: float between -LINE_MOVEMENT_BONUS and +LINE_MOVEMENT_BONUS
    """
    if pick_type == 'spread':
        all_lines = game_odds.get('all_spreads', [])
        sharp_line = game_odds.get('sharp_spread')
        if not all_lines or sharp_line is None or len(all_lines) < 2:
            return 0.0

        consensus = sum(l['spread'] for l in all_lines) / len(all_lines)
        # Positive diff = sharp book has a MORE negative spread (favoring home more)
        diff = consensus - sharp_line

        # If we're picking the home team's spread and sharp moved toward home, boost
        if pick_direction == 'home' and diff > 0.5:
            return min(diff * 2, LINE_MOVEMENT_BONUS)
        elif pick_direction == 'away' and diff < -0.5:
            return min(abs(diff) * 2, LINE_MOVEMENT_BONUS)
        elif abs(diff) > 0.5:
            return -min(abs(diff) * 2, LINE_MOVEMENT_BONUS)  # Penalize

    elif pick_type == 'total':
        all_lines = game_odds.get('all_totals', [])
        sharp_total = game_odds.get('sharp_total')
        if not all_lines or sharp_total is None or len(all_lines) < 2:
            return 0.0

        consensus = sum(l['total'] for l in all_lines) / len(all_lines)
        diff = sharp_total - consensus

        # Sharp book has higher total than consensus → market moving toward Over
        if pick_direction == 'over' and diff > 0.5:
            return min(diff, LINE_MOVEMENT_BONUS)
        elif pick_direction == 'under' and diff < -0.5:
            return min(abs(diff), LINE_MOVEMENT_BONUS)
        elif abs(diff) > 0.5:
            return -min(abs(diff), LINE_MOVEMENT_BONUS)

    return 0.0


def compute_prop_edge(sim_distribution: list, book_line: float, stat_key: str = 'points', is_returning: bool = False) -> dict:
    """
    Given a list of simulated values and a bookmaker's Over/Under line,
    compute the edge and confidence applying category-specific safety margins.
    stat_key can be 'points', 'rebounds', 'assists', or 'combo'.
    """
    import statistics
    
    if not sim_distribution or book_line is None:
        return {'median': 0, 'mean': 0, 'confidence': 0, 'edge_pct': 0, 'valuable': False}

    sorted_vals = sorted(sim_distribution)
    n = len(sorted_vals)
    median = sorted_vals[n // 2]
    mean = sum(sorted_vals) / n
    std_dev = statistics.stdev(sorted_vals) if n > 1 else 0

    # Determine thresholds
    thresh = EDGE_THRESHOLDS.get(stat_key, EDGE_THRESHOLDS['points'])

    # Confidence = % of simulations that cleared the line
    over_count = sum(1 for v in sorted_vals if v > book_line)
    confidence = (over_count / n) * 100

    # Edge = % points above a coin flip
    edge_pct = confidence - 50.0
    direction = "Over" if edge_pct > 0 else "Under"
    
    # Absolute confidence and edge for threshold checks
    abs_conf = confidence if direction == "Over" else (100 - confidence)
    abs_edge = abs(edge_pct)
    
    valuable = False
    
    # Apply category-specific "Value Edge Trigger" and "Safety Margin" rules
    if stat_key == 'points':
        # Value trigger: Line move >= 2.5 pts
        trigger_met = abs(median - book_line) >= thresh['trigger']
        
        # Recalculate safe confidence by artificially padding the book line against us
        safe_line = book_line + thresh['safety'] if direction == "Over" else book_line - thresh['safety']
        safe_over_count = sum(1 for v in sorted_vals if v > safe_line)
        safe_conf = (safe_over_count / n) * 100 if direction == "Over" else ((n - safe_over_count) / n) * 100
        
        if trigger_met and safe_conf >= thresh['min_prob'] and abs_edge >= thresh['min_edge']:
            valuable = True

    elif stat_key in ['rebounds', 'assists']:
        # Poisson OVER > 68%, Safety: lambda +- std dev
        safe_line = book_line + std_dev if direction == "Over" else book_line - std_dev
        safe_over_count = sum(1 for v in sorted_vals if v > safe_line)
        safe_conf = (safe_over_count / n) * 100 if direction == "Over" else ((n - safe_over_count) / n) * 100
        
        if abs_conf >= thresh['min_prob'] and abs_edge >= thresh['min_edge']:
            # If standard deviation is insanely tight, it passes, else check safe conf
            valuable = True 
            
    elif stat_key == 'combo':
        safe_line = book_line + thresh['safety'] if direction == "Over" else book_line - thresh['safety']
        safe_over_count = sum(1 for v in sorted_vals if v > safe_line)
        safe_conf = (safe_over_count / n) * 100 if direction == "Over" else ((n - safe_over_count) / n) * 100
        
        if abs_conf >= thresh['min_prob'] and abs_edge >= thresh['min_edge']:
            valuable = True

    # High Volatility Filter: Never include small lines (< 5.5)
    if book_line < 5.5:
        valuable = False

    returning_ban = False
    if is_returning and median < book_line:
        valuable = False
        returning_ban = True

    return {
        'median': median,
        'mean': mean,
        'confidence': confidence,
        'edge_pct': edge_pct,
        'valuable': valuable,
        'direction': direction,
        'returning_ban': returning_ban
    }


def compute_spread_edge(sim_home_scores: list, sim_away_scores: list,
                         book_spread: float, is_home: bool) -> dict:
    """
    Compute edge for a spread bet.
    book_spread is from the home team's perspective (e.g., -5.5 means home favored by 5.5)
    Edge is expressed as (confidence - 50) so it represents how many % points above
    a coin flip our model gives this pick. Avoids division-by-small-spread blowup.
    """
    if not sim_home_scores or not sim_away_scores:
        return {'projected_margin': 0, 'safe_spread': 0, 'confidence': 0, 'edge_pct': 0}

    n = len(sim_home_scores)
    margins = [sim_home_scores[i] - sim_away_scores[i] for i in range(n)]
    margins.sort()

    median_margin = margins[n // 2]

    # Apply safety margin (for UI calculation only, not confidence probability)
    safe_spread = book_spread + SPREAD_SAFETY_MARGIN if book_spread < 0 else book_spread - SPREAD_SAFETY_MARGIN

    # How many sims cover the ACTUAL book spread? (This fixes Issue 6: overlapping math)
    if is_home:
        # Issue 5 Fix: Reject contradictory spread picks instantly
        if median_margin < 0: 
            return {'projected_margin': median_margin, 'safe_spread': safe_spread, 'confidence': 0, 'edge_pct': 0}
        cover_count = sum(1 for m in margins if m > book_spread)
    else:
        # Issue 5 Fix: Reject contradictory spread picks instantly
        if median_margin > 0:
            return {'projected_margin': -median_margin, 'safe_spread': safe_spread, 'confidence': 0, 'edge_pct': 0}
        cover_count = sum(1 for m in margins if -m > -book_spread)

    confidence = (cover_count / n) * 100
    
    # ── Calibration ──
    calib = get_category_calibration('SPREAD')
    confidence *= calib

    # Edge = how many % points above a coin flip (50%) our model gives this pick.
    # A spread that 68% of sims cover has an edge of +18% over random.
    edge_pct = confidence - 50.0

    return {
        'projected_margin': median_margin if is_home else -median_margin,
        'safe_spread': safe_spread,
        'confidence': confidence,
        'edge_pct': edge_pct,
    }


def compute_total_edge(sim_home_scores: list, sim_away_scores: list, book_total: float) -> dict:
    """
    Compute edge for an over/under total bet.
    """
    if not sim_home_scores or not sim_away_scores:
        return {'projected_total': 0, 'confidence_over': 0, 'confidence_under': 0, 'edge_pct': 0, 'valuable': False}

    n = len(sim_home_scores)
    totals = [sim_home_scores[i] + sim_away_scores[i] for i in range(n)]
    totals.sort()

    median_total = totals[n // 2]
    
    # Determine value based on new thresholds
    thresh = EDGE_THRESHOLDS['totals']
    
    # Evaluate Trigger (+4.5 pts against market)
    trigger_met = abs(median_total - book_total) >= thresh['trigger']
    
    # Calculate confidence applying safety margin (+-2.5 pts)
    over_count = sum(1 for t in totals if t > book_total)
    under_count = sum(1 for t in totals if t < book_total)
    
    confidence_over = (over_count / n) * 100
    confidence_under = (under_count / n) * 100
    
    best_confidence = max(confidence_over, confidence_under)
    
    # ── Calibration ──
    calib = get_category_calibration('TOTAL')
    best_confidence *= calib
    if confidence_over >= confidence_under:
        confidence_over = best_confidence
    else:
        confidence_under = best_confidence
        
    edge_pct = best_confidence - 50.0
    
    direction = "Over" if confidence_over > confidence_under else "Under"
    valuable = False
    
    # Apply safety logic just for evaluation
    safe_line = book_total + thresh['safety'] if direction == "Over" else book_total - thresh['safety']
    safe_over_count = sum(1 for t in totals if t > safe_line)
    safe_conf = (safe_over_count / n) * 100 if direction == "Over" else ((n - safe_over_count) / n) * 100
    
    if trigger_met and best_confidence >= thresh['min_prob'] and edge_pct >= thresh['min_edge']:
        valuable = True

    return {
        'projected_total': median_total,
        'confidence_over': confidence_over,
        'confidence_under': confidence_under,
        'edge_pct': edge_pct if direction == "Over" else -edge_pct,
        'valuable': valuable,
        'direction': direction
    }


def _is_blowout_risk(game_odds: dict) -> bool:
    """Check if the game has high blowout risk based on the spread."""
    if not game_odds or not game_odds.get('spreads'):
        return False
    spread = abs(game_odds['spreads']['home_spread'])
    return spread > BLOWOUT_SPREAD_THRESHOLD


def build_parlays(
    all_sim_results: List[dict],
    all_game_odds: Dict,
    all_player_props: Dict,
    all_prop_distributions: Dict,
    all_score_distributions: Dict,
    all_historical_rates: Dict = None,
    all_player_minutes: Dict = None,
) -> dict:
    """
    Main parlay construction function.
    
    Args:
        all_sim_results: List of per-game sim summaries (from live_scraper)
            Each: {'home_team': str, 'away_team': str, 'avg_home': float, 'avg_away': float, ...}
        all_game_odds: Dict from odds_fetcher.fetch_game_odds()
            Keyed by event_id
        all_player_props: Dict keyed by event_id -> player_props from odds_fetcher
        all_prop_distributions: Dict keyed by player_name -> {'PTS': [list], 'REB': [list], 'AST': [list]}
        all_score_distributions: Dict keyed by event_id -> {'home': [list], 'away': [list]}
        all_historical_rates: Dict keyed by player_name -> {'PTS': {'hit': N, 'total': M, 'pct': X}, ...}
    
    Returns:
        {
            'parlay_2_3': [list of pick dicts],  # Best 2-3 picks
            'parlay_4':   [list of pick dicts],  # Best 4 picks (3/4 system)
        }
    """
    all_picks = []

    for event_id, game_odds in all_game_odds.items():
        is_blowout = _is_blowout_risk(game_odds)
        home_name = game_odds['home_team']
        away_name = game_odds['away_team']
        
        # ── Module 4: Blowout Risk Protocol (Top 3 Favorite Bans) ──
        top_3_fav = set()
        if is_blowout and game_odds.get('spreads') and all_player_minutes:
            fav_team = home_name if game_odds['spreads']['home_spread'] < 0 else away_name
            fav_players = [p for p, data in all_player_minutes.items() if data.get('team') == fav_team]
            fav_players.sort(key=lambda p: all_player_minutes[p].get('usage', 0), reverse=True)
            top_3_fav = set(fav_players[:3])
            print(f"  \u26a0\ufe0f  [BLOWOUT RISK] {away_name} @ {home_name}: Banning OVERs for {', '.join(top_3_fav)}")
        
        market_inversion = False
        if event_id in all_score_distributions:
            scores = all_score_distributions[event_id]
            n_sims = len(scores['home'])
            margins = [scores['home'][i] - scores['away'][i] for i in range(n_sims)]
            margins.sort()
            expected_margin = margins[n_sims // 2]
            
            if game_odds.get('spreads'):
                market_margin = -game_odds['spreads']['home_spread']
                if abs(expected_margin - market_margin) > 6.0:
                    print(f"  \u26a0\ufe0f  [MARKET DISCREPANCY] {away_name} @ {home_name}: Proj Margin {expected_margin:+.1f}, Market Margin {market_margin:+.1f}")
                
                if (expected_margin > 0 and market_margin < 0) or (expected_margin < 0 and market_margin > 0):
                    print(f"  \u26d4  [FATAL: MARKET_INVERSION_DETECTED] {away_name} @ {home_name}. Proj {expected_margin:+.1f} vs Market {market_margin:+.1f}. Spreads skip to avoid trap.")
                    market_inversion = True

        # ── Spread Picks ──────────────────────
        if not market_inversion and game_odds.get('spreads') and event_id in all_score_distributions:
            scores = all_score_distributions[event_id]
            spread_data = game_odds['spreads']
            
            # Module 3.3 Rule Check: "Ensure spread picks ALIGN with projected winner"
            # This is naturally handled by compute_spread_edge returning a low confidence if we pick against our own median!
            
            # Compare Home vs Away spread and pick ONLY the best one (Fixing Issue 1)
            best_spread_pick = None
            
            # Home spread check
            home_edge = compute_spread_edge(
                scores['home'], scores['away'],
                spread_data['home_spread'], is_home=True
            )
            
            # ── CALIBRATION ──
            raw_conf = home_edge['confidence']
            calib_conf = calibrate_confidence(raw_conf)
            
            spread_thresh = EDGE_THRESHOLDS['spreads']
            if calib_conf >= spread_thresh['min_prob'] and home_edge['edge_pct'] >= spread_thresh['min_edge']:
                lm_bonus = compute_line_movement_bonus(game_odds, 'spread', 'home')
                adj_confidence = min(99, calib_conf + lm_bonus)
                lm_str = f" [Sharp: {'+' if lm_bonus>0 else ''}{lm_bonus:.0f}%]" if lm_bonus != 0 else ""
                best_spread_pick = {
                    'type': 'spread',
                    'event_id': event_id,
                    'bet_team': home_name,
                    'spread_val': spread_data['home_spread'],
                    'game': f"{away_name} @ {home_name}",
                    'pick': f"{home_name} {spread_data['home_spread']:+.1f} \u2192 {home_edge['safe_spread']:+.1f} (reduced {SPREAD_SAFETY_MARGIN}pts) | Proj margin: {home_edge['projected_margin']:+.1f}pts{lm_str}",
                    'odds': spread_data['home_odds'],
                    'book': spread_data['book'],
                    'edge_pct': home_edge['edge_pct'],
                    'confidence': adj_confidence,
                    'score': home_edge['edge_pct'] * adj_confidence / 100,
                }

            # Away spread check
            away_edge = compute_spread_edge(
                scores['home'], scores['away'],
                spread_data['away_spread'], is_home=False
            )
            
            # ── CALIBRATION ──
            raw_away_conf = away_edge['confidence']
            calib_away_conf = calibrate_confidence(raw_away_conf)
            
            if calib_away_conf >= spread_thresh['min_prob'] and away_edge['edge_pct'] >= spread_thresh['min_edge']:
                lm_bonus = compute_line_movement_bonus(game_odds, 'spread', 'away')
                adj_confidence = min(99, calib_away_conf + lm_bonus)
                away_score = away_edge['edge_pct'] * adj_confidence / 100
                
                # If away is better OR home didn't pass, assign to away
                if not best_spread_pick or away_score > best_spread_pick['score']:
                    lm_str = f" [Sharp: {'+' if lm_bonus>0 else ''}{lm_bonus:.0f}%]" if lm_bonus != 0 else ""
                    best_spread_pick = {
                        'type': 'spread',
                        'event_id': event_id,
                        'bet_team': away_name,
                        'spread_val': spread_data['away_spread'],
                        'game': f"{away_name} @ {home_name}",
                        'pick': f"{away_name} {spread_data['away_spread']:+.1f} \u2192 {away_edge['safe_spread']:+.1f} (reduced {SPREAD_SAFETY_MARGIN}pts) | Proj margin: {away_edge['projected_margin']:+.1f}pts{lm_str}",
                        'odds': spread_data['away_odds'],
                        'book': spread_data['book'],
                        'edge_pct': away_edge['edge_pct'],
                        'confidence': adj_confidence,
                        'score': away_score,
                    }
            
            if best_spread_pick:
                all_picks.append(best_spread_pick)

        # ── Total Picks ───────────────────────
        if game_odds.get('totals') and event_id in all_score_distributions:
            scores = all_score_distributions[event_id]
            totals_data = game_odds['totals']
            total_edge = compute_total_edge(
                scores['home'], scores['away'],
                totals_data['total']
            )

            # Valuable flag internally sets strict thresholds
            if total_edge['valuable']:
                # Over
                if total_edge['direction'] == "Over":
                    raw_conf = total_edge['confidence_over']
                    calib_conf = calibrate_confidence(raw_conf)
                    
                    if calib_conf < thresh['min_prob']:
                        continue
                        
                    all_picks.append({
                        'type': 'total',
                        'event_id': event_id,
                        'game': f"{away_name} @ {home_name}",
                        'pick': f"Over {totals_data['total']} (Projected: {total_edge['projected_total']:.1f})",
                        'odds': totals_data['over_odds'],
                        'book': totals_data['book'],
                        'edge_pct': abs(total_edge['edge_pct']),
                        'confidence': calib_conf,
                        'score': abs(total_edge['edge_pct']) * calib_conf / 100,
                    })
                # Under
                elif total_edge['direction'] == "Under":
                    raw_conf = total_edge['confidence_under']
                    calib_conf = calibrate_confidence(raw_conf)
                    
                    if calib_conf < thresh['min_prob']:
                        continue
                        
                    all_picks.append({
                        'type': 'total',
                        'event_id': event_id,
                        'game': f"{away_name} @ {home_name}",
                        'pick': f"Under {totals_data['total']} (Projected: {total_edge['projected_total']:.1f})",
                        'odds': totals_data['under_odds'],
                        'book': totals_data['book'],
                        'edge_pct': abs(total_edge['edge_pct']),
                        'confidence': calib_conf,
                        'score': abs(total_edge['edge_pct']) * calib_conf / 100,
                    })

        # ── Player Prop Picks ─────────────────
        if event_id not in all_player_props:
            continue

        props = all_player_props[event_id]
        stat_map = {'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST'}

        for player_name, player_data in props.items():
            # Check if we have sim data for this player
            if player_name not in all_prop_distributions:
                continue

            for stat_key, sim_key in stat_map.items():
                if stat_key not in player_data:
                    continue
                if sim_key not in all_prop_distributions[player_name]:
                    continue

                prop_info = player_data[stat_key]
                sim_dist = all_prop_distributions[player_name][sim_key]
                line = prop_info['line']

                # Determine internal stat key mapping for model evaluation
                internal_stat_key = 'points'
                if stat_key == 'rebounds':
                    internal_stat_key = 'rebounds'
                elif stat_key == 'assists':
                    internal_stat_key = 'assists'
                elif '_' in stat_key: # combo like points_rebounds
                    internal_stat_key = 'combo'

                is_returning = False
                if all_player_minutes and player_name in all_player_minutes:
                    is_returning = bool(all_player_minutes[player_name].get('restriction', False))

                edge = compute_prop_edge(sim_dist, line, internal_stat_key, is_returning=is_returning)
                # BLOWOUT PROTOCOL: ban OVER props for the fav team's top 3 players
                if is_blowout and edge['direction'] == "Over" and player_name in top_3_fav:
                    continue
                    
                if edge.get('returning_ban'):
                    continue

                # MINUTES RISK PROTOCOL
                # Relaxed: was mean<24 which excluded valuable role players (e.g. Clingan at 22 min)
                if all_player_minutes and player_name in all_player_minutes:
                    p_min = all_player_minutes[player_name]
                    if p_min.get('std', 0) > 6.5 or p_min.get('mean', 30) < 18.0:
                        continue

                # Internal evaluation returns True if mathematically sound
                if edge['valuable']:
                    direction = edge['direction']
                    conf = edge['confidence'] if direction == "Over" else (100 - edge['confidence'])
                    
                    # ── CALIBRATION ──
                    calib_conf = calibrate_confidence(conf)
                    # For props, apply a harder cap
                    calib_conf = min(calib_conf, MAX_PROP_CONFIDENCE)

                    # ── Module 5: DB Historical Hit Rate Validation ──
                    hist_str = ""
                    hist_pct = None
                    if all_historical_rates and player_name in all_historical_rates:
                        hr = all_historical_rates[player_name].get(sim_key, {})
                        if hr and hr.get('total', 0) > 0:
                            # Rule 5.1: Invalid Data Triggers
                            if is_returning or hr['total'] < 5:
                                pass # Ignore DB completely
                            else:
                                # Split-window: 65% weight on last 10, 35% on last 20
                                pct_ext = hr['pct']          # last 20 games
                                pct_rec = hr.get('recent_pct', pct_ext)  # last 10 games
                                
                                # Flip percentages for Under bets
                                if direction == 'Under':
                                    pct_ext = 100.0 - pct_ext
                                    pct_rec = 100.0 - pct_rec

                                # Rule 5.2: Contextual Hit Rate
                                if is_blowout:
                                    pct_ext = max(0.0, pct_ext - 5.0)
                                    pct_rec = max(0.0, pct_rec - 5.0)

                                # Weighted DB confidence: recent form matters more
                                hist_pct = (pct_rec * 0.65) + (pct_ext * 0.35)

                                rec_total = hr.get('recent_total', hr['total'])
                                rec_hit   = hr.get('recent_hit', hr['hit'])
                                hist_str  = (
                                    f" | DB: {pct_rec:.0f}%({rec_hit}/{rec_total}) "
                                    f"/ {pct_ext:.0f}%({hr['hit']}/{hr['total']}) "
                                    f"→ {hist_pct:.0f}%w"
                                )

                    # Compute blended confidence weighted by category reliability
                    blended_conf = calib_conf
                    if hist_pct is not None:
                        sim_w, db_w = SIM_DB_WEIGHTS.get(internal_stat_key, (0.7, 0.3))
                        
                        # ── Dynamic Blending for Role Changes ──
                        # If the player is flagged as having a usage boost (due to missing teammates),
                        # trust the Simulation MUCH more (90/10 split) than historical data.
                        is_boosted = all_player_minutes.get(player_name, {}).get('usage_boosted', False)
                        if is_boosted:
                            sim_w, db_w = 0.90, 0.10
                        
                        blended_conf = (calib_conf * sim_w) + (hist_pct * db_w)
                        
                    # ── Live Category Calibration ──
                    calib = get_category_calibration(internal_stat_key)
                    blended_conf *= calib

                    # Determine inclusion threshold
                    thresh = EDGE_THRESHOLDS.get(internal_stat_key, EDGE_THRESHOLDS['combo'])
                    min_prob = thresh['min_prob']
                    
                    # ── Agreement Exception ("Trust ✅") ──
                    # If BOTH the Sim and the DB history agree they are high-prob (>75%),
                    # we lower the specific qualification bar (e.g. bypass the 82% assist rule).
                    if hist_pct is not None and calib_conf >= 75.0 and hist_pct >= 75.0:
                        min_prob = 75.0
                    
                    if hist_pct is None:
                        # 100% Sim Confidence relied upon, raise the bar to 75%
                        min_prob = max(min_prob, 75.0)
                        
                    if blended_conf < min_prob:
                        continue

                    # Attempt to resolve human-readable stat string
                    stat_display = stat_key.upper()
                    if '_' in stat_key:
                        acronym = ''.join([part[0].upper() for part in stat_key.split('_')])
                        stat_display = acronym
                        
                    pick_label = f"{player_name} {direction} {line} {stat_display}"

                    # Trust level: HIGH if both agree (within 10%), LOW if they diverge
                    trust = ""
                    if hist_pct is not None:
                        if abs(calib_conf - hist_pct) < 10:
                            trust = " ✅"
                        elif abs(calib_conf - hist_pct) > 25:
                            trust = " ⚠️"

                    pick_display = f"{pick_label} (Sim: {calib_conf:.0f}%{hist_str}){trust}"

                    all_picks.append({
                        'type': 'prop',
                        'event_id': event_id,
                        'team': all_player_minutes.get(player_name, {}).get('team'),
                        'direction': direction,
                        'game': f"{away_name} @ {home_name}",
                        'pick': pick_display,
                        'odds': prop_info['over_odds'] if direction == "Over" else prop_info.get('under_odds', 1.90),
                        'book': prop_info['book'],
                        'edge_pct': abs(edge['edge_pct']),
                        'confidence': blended_conf,
                        'score': abs(edge['edge_pct']) * blended_conf / 100,
                    })

    # Sort all picks primarily by CONFIDENCE % to ensure 90%+ picks are prioritized (Issue 4 Fix).
    # Secondary sort by 'score' (which factors in the edge amount)
    all_picks.sort(key=lambda p: (p['confidence'], p['score']), reverse=True)

    # ── Build Parlay #1: Best 2-3 Picks ──────
    parlay_short = _select_diverse_picks(all_picks, target_count=3, max_per_game=2)
    if len(parlay_short) < 2:
        parlay_short = all_picks[:3] if len(all_picks) >= 2 else all_picks

    # ── Build Parlay #2: Best 4 Picks (3/4 System) ──
    parlay_system = _select_diverse_picks(all_picks, target_count=4, max_per_game=2)
    if len(parlay_system) < 4:
        parlay_system = all_picks[:4]

    return {
        'parlay_2_3': parlay_short[:3],
        'parlay_4': parlay_system[:4],
        'all_picks': all_picks,
    }


def _select_diverse_picks(picks: list, target_count: int, max_per_game: int) -> list:
    """
    Select picks ensuring diversity across games and enforcing Module 6 Correlation rules.
    """
    selected = []
    game_count = {}

    for pick in picks:
        eid = pick['event_id']
        if game_count.get(eid, 0) >= max_per_game:
            continue
            
        conflict = False
        for existing in selected:
            if existing['event_id'] != eid: continue
            
            p1, p2 = pick, existing
            spread_pick, prop_pick = None, None
            
            if p1['type'] == 'spread' and p2['type'] == 'prop':
                spread_pick, prop_pick = p1, p2
            elif p2['type'] == 'spread' and p1['type'] == 'prop':
                spread_pick, prop_pick = p2, p1
                
            if spread_pick and prop_pick:
                s_val = spread_pick.get('spread_val', 0)
                # If we are betting a heavy favorite spread (-7.5 or worse)
                if s_val <= -7.5:
                    fav_team = spread_pick['bet_team']
                    # Ban Underdog OVER props in the same parlay
                    if prop_pick.get('direction') == "Over" and prop_pick.get('team') != fav_team:
                        conflict = True
                        break

        if conflict:
            continue

        selected.append(pick)
        game_count[eid] = game_count.get(eid, 0) + 1
        if len(selected) >= target_count:
            break

    return selected


def format_parlay_output(parlays: dict) -> str:
    """
    Pretty-print the parlay recommendations.
    """
    lines = []

    lines.append("")
    lines.append("=" * 60)
    lines.append("  🏀 PARLAY RECOMMENDATION #1 (2-3 PICKS)")
    lines.append("=" * 60)
    
    combined_odds = 1.0
    for i, pick in enumerate(parlays['parlay_2_3'], 1):
        odds_str = f"@ {pick['odds']:.2f}" if pick['odds'] else ""
        lines.append(f"  Pick {i}: [{pick['game']}] {pick['pick']} {odds_str}")
        lines.append(f"          Edge: {pick['edge_pct']:+.1f}%  |  Confidence: {pick['confidence']:.0f}%  |  Book: {pick['book']}")
        if pick['odds']:
            combined_odds *= pick['odds']
    
    lines.append(f"  ─────────────────────────────────────────")
    lines.append(f"  Combined Odds: {combined_odds:.2f}x")
    
    # Kelly Stake Recommendation
    # For a parlay we treat it as a single bet with combined odds and joint probability.
    # Joint Prob = Product of individual calibrated confidences.
    joint_prob = 100.0
    for p in parlays['parlay_2_3']:
        joint_prob *= (p['confidence'] / 100.0)
        
    stake = compute_kelly_stake(combined_odds, joint_prob)
    lines.append(f"  🔥 RECOMMENDED STAKE: {stake*100:.2f}% of Bankroll")
    lines.append("")

    lines.append("=" * 60)
    lines.append("  🏀 PARLAY RECOMMENDATION #2 (4 PICKS — 3/4 SYSTEM)")
    lines.append("=" * 60)

    combined_odds_4 = 1.0
    for i, pick in enumerate(parlays['parlay_4'], 1):
        odds_str = f"@ {pick['odds']:.2f}" if pick['odds'] else ""
        lines.append(f"  Pick {i}: [{pick['game']}] {pick['pick']} {odds_str}")
        lines.append(f"          Edge: {pick['edge_pct']:+.1f}%  |  Confidence: {pick['confidence']:.0f}%  |  Book: {pick['book']}")
        if pick['odds']:
            combined_odds_4 *= pick['odds']
    
    # Calculate 3/4 system odds approximation
    # In a 3/4 system, you need 3 out of 4 to win
    individual_odds = [p['odds'] for p in parlays['parlay_4'] if p.get('odds')]
    if len(individual_odds) == 4:
        # 3/4 system = sum of (product of each 3-leg combo) / 4
        from itertools import combinations
        combos_3 = list(combinations(individual_odds, 3))
        system_payout = sum(math.prod(c) for c in combos_3) / len(combos_3)
        
        # Approximate win prob for 3/4 system: P(at least 3 win)
        probs = [p['confidence']/100.0 for p in parlays['parlay_4']]
        p1, p2, p3, p4 = probs
        prob_3_of_4 = (p1*p2*p3*(1-p4)) + (p1*p2*(1-p3)*p4) + (p1*(1-p2)*p3*p4) + ((1-p1)*p2*p3*p4) + (p1*p2*p3*p4)
        
        stake = compute_kelly_stake(system_payout, prob_3_of_4 * 100.0)
        
        lines.append(f"  ─────────────────────────────────────────")
        lines.append(f"  Combined Odds (all 4): {combined_odds_4:.2f}x")
        lines.append(f"  System 3/4 Avg Payout: {system_payout:.2f}x")
        lines.append(f"  🔥 RECOMMENDED STAKE: {stake*100:.2f}% of Bankroll")
    else:
        lines.append(f"  ─────────────────────────────────────────")
        lines.append(f"  Combined Odds: {combined_odds_4:.2f}x")

    lines.append("=" * 60)

    # --- Honorable Mentions ---
    all_picks = parlays.get('all_picks', [])
    parlay_4_picks = parlays.get('parlay_4', [])
    
    # Exclude picks that already made it into the main parlays
    used_picks = {f"[{p['game']}] {p['pick']}" for p in parlay_4_picks}
    remaining_picks = [p for p in all_picks if f"[{p['game']}] {p['pick']}" not in used_picks]
    
    if remaining_picks:
        lines.append("")
        lines.append("  🔥 HONORABLE MENTIONS (Next Best Picks)")
        lines.append("  ─────────────────────────────────────────")
        
        # Take the next 4 best picks
        for i, pick in enumerate(remaining_picks[:4], 1):
            odds_str = f"@ {pick['odds']:.2f}" if pick['odds'] else ""
            lines.append(f"  [{pick['game']}] {pick['pick']} {odds_str}")
            lines.append(f"      Edge: {pick['edge_pct']:+.1f}%  |  Conf: {pick['confidence']:.0f}%  |  Book: {pick['book']}")
            if i < min(4, len(remaining_picks)):
                lines.append("")

    # Show total available picks for transparency
    total = len(all_picks)
    if total > 4:
        lines.append(f"\n  📊 Analyzed {total} potential picks across all games.")
        lines.append(f"  Showing only the highest-confidence selections.\n")

    return "\n".join(lines)


if __name__ == '__main__':
    # Quick test with fake data
    fake_dist = list(range(15, 35)) * 25  # 500 values averaging ~25
    edge = compute_prop_edge(fake_dist, 24.5)
    print(f"Test Edge: median={edge['median']}, conf={edge['confidence']:.1f}%, edge={edge['edge_pct']:.1f}%")
