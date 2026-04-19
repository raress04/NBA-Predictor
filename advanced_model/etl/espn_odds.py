"""
ESPN Odds Scraper — Free, unlimited NBA odds from ESPN's public API.
Source: DraftKings lines via ESPN scoreboard endpoint.
Provides: spreads, totals, moneyline, and open/close line movement.
No API key required. Cached for 60 minutes between runs.
"""

import json
import os
import time
import requests
from typing import Dict, Optional, Tuple

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.cache')
ESPN_CACHE_FILE = os.path.join(CACHE_DIR, 'espn_odds.json')
CACHE_TTL = 3600  # 60 minutes


def _load_cache(path: str) -> dict:
    """Load cached data if file exists and is within TTL."""
    if not os.path.isfile(path):
        return None
    try:
        age = time.time() - os.path.getmtime(path)
        if age > CACHE_TTL:
            return None
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        mins_left = int((CACHE_TTL - age) / 60)
        print(f"    [CACHE HIT] Using cached odds ({mins_left}min remaining)")
        return data
    except:
        return None


def _save_cache(path: str, data: dict):
    """Save data to cache file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    except:
        pass


def _american_to_decimal(american_str: str) -> float:
    """Convert American odds string (e.g. '-110', '+150') to decimal odds."""
    try:
        odds = int(american_str)
        if odds > 0:
            return round(1 + (odds / 100), 3)
        else:
            return round(1 + (100 / abs(odds)), 3)
    except (ValueError, ZeroDivisionError):
        return 1.91  # default vig line


from datetime import datetime

def fetch_espn_odds() -> Dict:
    """
    Fetch moneyline, spreads, and totals for all today's NBA games from ESPN.
    Uses 60-minute JSON cache to avoid redundant API calls.
    """
    # Check cache first
    cached = _load_cache(ESPN_CACHE_FILE)
    if cached is not None:
        return cached
    
    try:
        from datetime import timezone, timedelta
        # ESPN/NBA schedules use US Eastern Time (UTC-4 EDT / UTC-5 EST)
        # Always use ET date regardless of local timezone
        et_offset = timedelta(hours=-4)  # EDT (Mar-Nov); adjust to -5 for EST if needed
        et_now = datetime.now(timezone.utc) + et_offset
        today_date = et_now.strftime('%Y%m%d')
        url = f"{ESPN_SCOREBOARD_URL}?dates={today_date}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[!] ESPN odds fetch failed: {e}")
        return {}

    all_odds = {}

    for event in data.get('events', []):
        event_id = event['id']
        competition = event['competitions'][0]
        
        # Identify home and away teams
        home_team = None
        away_team = None
        for comp in competition['competitors']:
            if comp['homeAway'] == 'home':
                home_team = comp['team']['displayName']
            else:
                away_team = comp['team']['displayName']

        if not home_team or not away_team:
            continue

        # Only process scheduled (pre-game) events
        status = competition.get('status', {}).get('type', {}).get('state', '')
        if status != 'pre':
            continue

        # Extract odds from ESPN's DraftKings integration
        odds_list = competition.get('odds', [])
        if not odds_list:
            continue

        odds_data = odds_list[0]  # Primary provider (DraftKings)
        provider_name = odds_data.get('provider', {}).get('name', 'ESPN BET')

        game_odds = {
            'home_team': home_team,
            'away_team': away_team,
            'commence_time': event.get('date', ''),
            'h2h': None,
            'spreads': None,
            'totals': None,
            'all_spreads': [],
            'all_totals': [],
            'sharp_spread': None,
            'sharp_total': None,
        }

        # ── Moneyline (H2H) ──────────────────────────
        ml = odds_data.get('moneyline', {})
        if ml:
            home_ml = ml.get('home', {}).get('close', {}).get('odds', '')
            away_ml = ml.get('away', {}).get('close', {}).get('odds', '')
            if home_ml and away_ml:
                game_odds['h2h'] = {
                    'home_odds': _american_to_decimal(home_ml),
                    'away_odds': _american_to_decimal(away_ml),
                    'book': provider_name,
                }

        # ── Spread ────────────────────────────────────
        ps = odds_data.get('pointSpread', {})
        if ps:
            home_line = ps.get('home', {}).get('close', {}).get('line', '')
            away_line = ps.get('away', {}).get('close', {}).get('line', '')
            home_odds_str = ps.get('home', {}).get('close', {}).get('odds', '-110')
            away_odds_str = ps.get('away', {}).get('close', {}).get('odds', '-110')
            
            # Open lines for line movement tracking
            home_open = ps.get('home', {}).get('open', {}).get('line', '')
            
            if home_line and away_line:
                home_spread = float(home_line)
                away_spread = float(away_line)
                
                game_odds['spreads'] = {
                    'home_spread': home_spread,
                    'away_spread': away_spread,
                    'home_odds': _american_to_decimal(home_odds_str),
                    'away_odds': _american_to_decimal(away_odds_str),
                    'book': provider_name,
                }
                
                spread_entry = {
                    'book': provider_name,
                    'home_spread': home_spread,
                    'away_spread': away_spread,
                }
                game_odds['all_spreads'].append(spread_entry)
                
                # Use DraftKings as our "sharp" reference
                game_odds['sharp_spread'] = home_spread
                
                # Track open line for movement
                if home_open:
                    game_odds['open_home_spread'] = float(home_open)

        # ── Totals ────────────────────────────────────
        total_data = odds_data.get('total', {})
        ou_value = odds_data.get('overUnder')
        
        if ou_value is not None:
            over_odds_str = total_data.get('over', {}).get('close', {}).get('odds', '-110')
            under_odds_str = total_data.get('under', {}).get('close', {}).get('odds', '-110')
            
            # Open total for line movement
            open_total_str = total_data.get('over', {}).get('open', {}).get('line', '')
            
            game_odds['totals'] = {
                'total': float(ou_value),
                'over_odds': _american_to_decimal(over_odds_str),
                'under_odds': _american_to_decimal(under_odds_str),
                'book': provider_name,
            }
            
            total_entry = {
                'book': provider_name,
                'total': float(ou_value),
            }
            game_odds['all_totals'].append(total_entry)
            game_odds['sharp_total'] = float(ou_value)
            
            if open_total_str:
                # Strip 'o' or 'u' prefix (ESPN formats as "o230.5")
                clean = open_total_str.replace('o', '').replace('u', '')
                try:
                    game_odds['open_total'] = float(clean)
                except ValueError:
                    pass

        all_odds[event_id] = game_odds

    print(f"[+] ESPN: Fetched odds for {len(all_odds)} upcoming NBA games (source: {provider_name if all_odds else 'N/A'})")
    return all_odds


def match_espn_odds_to_game(all_odds: Dict, home_team_name: str, away_team_name: str) -> Optional[Tuple[str, dict]]:
    """
    Match ESPN odds event to our NBA API game by fuzzy-matching team names.
    Returns (event_id, odds_dict) or None.
    """
    home_lower = home_team_name.lower()
    away_lower = away_team_name.lower()

    for event_id, odds in all_odds.items():
        api_home = odds['home_team'].lower()
        api_away = odds['away_team'].lower()

        home_match = any(w in api_home for w in home_lower.split()) or any(w in home_lower for w in api_home.split())
        away_match = any(w in api_away for w in away_lower.split()) or any(w in away_lower for w in api_away.split())

        if home_match and away_match:
            return (event_id, odds)

    return None


if __name__ == '__main__':
    odds = fetch_espn_odds()
    for eid, g in odds.items():
        print(f"\n{g['away_team']} @ {g['home_team']}")
        if g['h2h']:
            print(f"  H2H:    Home {g['h2h']['home_odds']} | Away {g['h2h']['away_odds']}  ({g['h2h']['book']})")
        if g['spreads']:
            s = g['spreads']
            print(f"  Spread: Home {s['home_spread']:+.1f} @ {s['home_odds']}  ({s['book']})")
        if g['totals']:
            t = g['totals']
            print(f"  Total:  O/U {t['total']} — Over {t['over_odds']} / Under {t['under_odds']}  ({t['book']})")
