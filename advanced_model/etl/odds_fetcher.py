"""
Odds Fetcher — Pulls live NBA odds from The Odds API v4
Supports: spreads, totals, h2h (moneyline), and player props (pts/reb/ast)
Primary bookmaker: Unibet (EU/UK), fallback: Pinnacle for sharp lines
Cached for 60 minutes between runs to save API credits.
"""

import json
import os
import requests
import time
from typing import Dict, List, Optional, Tuple

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.cache')
CACHE_TTL = 7200  # 120 minutes

def _get_cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f'odds_api_{name}.json')

def _load_cache(path: str, force: bool = False) -> dict:
    if not os.path.isfile(path):
        return None
    try:
        age = time.time() - os.path.getmtime(path)
        if not force and age > CACHE_TTL:
            return None
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if force and age > CACHE_TTL:
            mins_old = int(age / 60)
            print(f"    [CACHE HIT (historical)] Loaded from {os.path.basename(path)} ({mins_old}min old)")
        else:
            mins_left = int((CACHE_TTL - age) / 60)
            print(f"    [CACHE HIT] Loaded from {os.path.basename(path)} ({mins_left}min remaining)")
        return data
    except:
        return None

def _save_cache(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"    [CACHE WRITE ERROR] -> {e}")

def _load_env_file():
    """Manually parse a .env file — no python-dotenv required."""
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _CANDIDATES = [
        os.path.join(os.path.dirname(_THIS_DIR), '.env'),   # advanced_model/.env  (expected)
        os.path.join(_THIS_DIR, '.env'),                     # advanced_model/etl/.env
        os.path.join(os.getcwd(), '.env'),                   # wherever you ran the command from
        os.path.join(os.path.dirname(os.getcwd()), '.env'),  # one level above cwd
    ]
    for path in _CANDIDATES:
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    key, _, val = line.partition('=')
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:  # don't override existing shell exports
                        os.environ[key] = val
            return  # stop at first found file

_load_env_file()

API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

# Preferred bookmakers in order — Unibet first, then Pinnacle for sharp reference
PREFERRED_BOOKS = ['unibet', 'unibet_eu', 'pinnacle', 'betsson', 'onexbet']

def _get_api_key() -> str:
    key = os.environ.get('ODDS_API_KEY', '').strip()
    if not key:
        # Show exactly where we looked so user can debug
        _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        expected = os.path.join(os.path.dirname(_THIS_DIR), '.env')
        raise ValueError(
            f"ODDS_API_KEY not set.\n"
            f"  Expected .env location: {expected}\n"
            f"  File exists: {os.path.isfile(expected)}\n"
            f"  Make sure the file contains exactly: ODDS_API_KEY=your_key_here (no quotes)\n"
            f"  Get a free key at https://the-odds-api.com"
        )
    return key


def fetch_event_ids(api_key: str = None) -> Dict:
    """
    Lightweight fetch: just get Odds API event IDs mapped to team names.
    Costs only 1 API request. Cached for 60 min.
    Returns: {
        event_id: {'home_team': str, 'away_team': str}
    }
    """
    cache_path = _get_cache_path('events')
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached

    if api_key is None:
        api_key = _get_api_key()

    url = f"{API_BASE}/sports/{SPORT}/events"
    params = {
        'apiKey': api_key,
        'dateFormat': 'iso',
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    events = resp.json()
    remaining = resp.headers.get('x-requests-remaining', '?')
    print(f"    [+] Odds API: Found {len(events)} events. Credits remaining: {remaining}")

    result = {}
    for ev in events:
        result[ev['id']] = {
            'home_team': ev.get('home_team', ''),
            'away_team': ev.get('away_team', ''),
        }
    
    _save_cache(cache_path, result)
    return result


def match_event_id_to_game(event_ids: Dict, home_team: str, away_team: str) -> str:
    """Find the Odds API event_id matching a game by fuzzy team name matching."""
    home_lower = home_team.lower()
    away_lower = away_team.lower()
    for eid, teams in event_ids.items():
        api_home = teams['home_team'].lower()
        api_away = teams['away_team'].lower()
        home_match = any(w in api_home for w in home_lower.split()) or any(w in home_lower for w in api_home.split())
        away_match = any(w in api_away for w in away_lower.split()) or any(w in away_lower for w in api_away.split())
        if home_match and away_match:
            return eid
    return None


def fetch_game_odds(api_key: str = None, regions: str = 'uk,eu') -> Dict:
    """
    Fetch moneyline, spreads, and totals for all upcoming NBA games.
    Cached for 60 min.
    Returns: {
        event_id: {
            'home_team': str,
            'away_team': str,
            'commence_time': str,
            'h2h': {'home': decimal, 'away': decimal, 'book': str},
            'spreads': {'home_spread': float, 'home_odds': decimal, 'away_odds': decimal, 'book': str},
            'totals': {'total': float, 'over_odds': decimal, 'under_odds': decimal, 'book': str},
        }
    }
    """
    cache_path = _get_cache_path('game_odds')
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached

    if api_key is None:
        api_key = _get_api_key()

    url = f"{API_BASE}/sports/{SPORT}/odds"
    params = {
        'apiKey': api_key,
        'regions': regions,
        'markets': 'h2h,spreads,totals',
        'oddsFormat': 'decimal',
        'dateFormat': 'iso',
    }

    print("[+] Fetching NBA game odds (spreads, totals, h2h)...")
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    
    data = resp.json()
    remaining = resp.headers.get('x-requests-remaining', '?')
    print(f"    API credits remaining: {remaining}")

    results = {}
    for event in data:
        event_id = event['id']
        entry = {
            'home_team': event['home_team'],
            'away_team': event['away_team'],
            'commence_time': event.get('commence_time', ''),
            'h2h': None,
            'spreads': None,
            'totals': None,
            'all_spreads': [],   # all bookmaker home spreads, for line movement detection
            'all_totals': [],    # all bookmaker totals
            'sharp_spread': None,  # Pinnacle spread (sharpest line)
            'sharp_total': None,   # Pinnacle total
        }

        for bookmaker in event.get('bookmakers', []):
            book_key = bookmaker['key']
            # Only pick from preferred bookmakers, but accept any if preferred not found
            for market in bookmaker.get('markets', []):
                mk = market['key']
                outcomes = market['outcomes']

                if mk == 'h2h' and entry['h2h'] is None:
                    home_odds = next((o['price'] for o in outcomes if o['name'] == event['home_team']), None)
                    away_odds = next((o['price'] for o in outcomes if o['name'] == event['away_team']), None)
                    if home_odds and away_odds:
                        entry['h2h'] = {'home': home_odds, 'away': away_odds, 'book': book_key}

                elif mk == 'spreads' and entry['spreads'] is None:
                    home_out = next((o for o in outcomes if o['name'] == event['home_team']), None)
                    away_out = next((o for o in outcomes if o['name'] == event['away_team']), None)
                    if home_out and away_out:
                        entry['spreads'] = {
                            'home_spread': home_out['point'],
                            'home_odds': home_out['price'],
                            'away_spread': away_out['point'],
                            'away_odds': away_out['price'],
                            'book': book_key,
                        }

                elif mk == 'totals' and entry['totals'] is None:
                    over = next((o for o in outcomes if o['name'] == 'Over'), None)
                    under = next((o for o in outcomes if o['name'] == 'Under'), None)
                    if over and under:
                        entry['totals'] = {
                            'total': over['point'],
                            'over_odds': over['price'],
                            'under_odds': under['price'],
                            'book': book_key,
                        }

                # Collect ALL spreads/totals for line movement proxy
                if mk == 'spreads':
                    home_out = next((o for o in outcomes if o['name'] == event['home_team']), None)
                    if home_out:
                        entry['all_spreads'].append({'spread': home_out['point'], 'book': book_key})
                        if book_key == 'pinnacle':
                            entry['sharp_spread'] = home_out['point']
                if mk == 'totals':
                    over_out = next((o for o in outcomes if o['name'] == 'Over'), None)
                    if over_out:
                        entry['all_totals'].append({'total': over_out['point'], 'book': book_key})
                        if book_key == 'pinnacle':
                            entry['sharp_total'] = over_out['point']

        # Now try overriding with preferred bookmaker if available
        for bookmaker in event.get('bookmakers', []):
            book_key = bookmaker['key']
            if book_key not in PREFERRED_BOOKS:
                continue
            for market in bookmaker.get('markets', []):
                mk = market['key']
                outcomes = market['outcomes']

                if mk == 'h2h':
                    home_odds = next((o['price'] for o in outcomes if o['name'] == event['home_team']), None)
                    away_odds = next((o['price'] for o in outcomes if o['name'] == event['away_team']), None)
                    if home_odds and away_odds:
                        entry['h2h'] = {'home': home_odds, 'away': away_odds, 'book': book_key}

                elif mk == 'spreads':
                    home_out = next((o for o in outcomes if o['name'] == event['home_team']), None)
                    away_out = next((o for o in outcomes if o['name'] == event['away_team']), None)
                    if home_out and away_out:
                        entry['spreads'] = {
                            'home_spread': home_out['point'],
                            'home_odds': home_out['price'],
                            'away_spread': away_out['point'],
                            'away_odds': away_out['price'],
                            'book': book_key,
                        }

                elif mk == 'totals':
                    over = next((o for o in outcomes if o['name'] == 'Over'), None)
                    under = next((o for o in outcomes if o['name'] == 'Under'), None)
                    if over and under:
                        entry['totals'] = {
                            'total': over['point'],
                            'over_odds': over['price'],
                            'under_odds': under['price'],
                            'book': book_key,
                        }

        results[event_id] = entry

    print(f"    Found odds for {len(results)} NBA events.")
    _save_cache(cache_path, results)
    return results

def fetch_player_props(event_id: str, api_key: str = None, regions: str = 'uk,eu', force_cache: bool = False) -> Dict:
    """
    Fetch player prop lines (points, rebounds, assists) for a single NBA event.
    Cached for 60 min. Pass force_cache=True to load stale cache (for historical mode).
    ...
    """
    cache_path = _get_cache_path(f'props_{event_id}')
    cached = _load_cache(cache_path, force=force_cache)
    if cached is not None:
        return cached

    if api_key is None:
        api_key = _get_api_key()

    url = f"{API_BASE}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        'apiKey': api_key,
        'regions': regions,
        'markets': 'player_points,player_rebounds,player_assists',
        'oddsFormat': 'decimal',
        'dateFormat': 'iso',
    }

    print(f"    [+] Fetching player props for event {event_id[:12]}...")
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()

    data = resp.json()
    remaining = resp.headers.get('x-requests-remaining', '?')
    
    PROP_MAP = {
        'player_points': 'points',
        'player_rebounds': 'rebounds',
        'player_assists': 'assists',
    }

    players = {}

    # First pass: grab any bookmaker's data
    for bookmaker in data.get('bookmakers', []):
        book_key = bookmaker['key']
        for market in bookmaker.get('markets', []):
            mk = market['key']
            prop_type = PROP_MAP.get(mk)
            if not prop_type:
                continue

            for outcome in market.get('outcomes', []):
                pname = outcome.get('description', '')
                if not pname:
                    continue

                if pname not in players:
                    players[pname] = {}

                if prop_type not in players[pname]:
                    if outcome['name'] == 'Over':
                        players[pname][prop_type] = {
                            'line': outcome.get('point', 0),
                            'over_odds': outcome['price'],
                            'under_odds': None,
                            'book': book_key,
                        }
                    elif outcome['name'] == 'Under':
                        if prop_type in players[pname]:
                            players[pname][prop_type]['under_odds'] = outcome['price']
                        else:
                            players[pname][prop_type] = {
                                'line': outcome.get('point', 0),
                                'over_odds': None,
                                'under_odds': outcome['price'],
                                'book': book_key,
                            }
                else:
                    # Fill in the missing side
                    if outcome['name'] == 'Over' and players[pname][prop_type]['over_odds'] is None:
                        players[pname][prop_type]['over_odds'] = outcome['price']
                    elif outcome['name'] == 'Under' and players[pname][prop_type]['under_odds'] is None:
                        players[pname][prop_type]['under_odds'] = outcome['price']

    # Second pass: prefer data from preferred bookmakers
    for bookmaker in data.get('bookmakers', []):
        book_key = bookmaker['key']
        if book_key not in PREFERRED_BOOKS:
            continue
        for market in bookmaker.get('markets', []):
            mk = market['key']
            prop_type = PROP_MAP.get(mk)
            if not prop_type:
                continue

            # Group Over/Under by player
            player_outcomes = {}
            for outcome in market.get('outcomes', []):
                pname = outcome.get('description', '')
                if not pname:
                    continue
                if pname not in player_outcomes:
                    player_outcomes[pname] = {}
                player_outcomes[pname][outcome['name']] = outcome

            for pname, sides in player_outcomes.items():
                over = sides.get('Over')
                under = sides.get('Under')
                if over:
                    if pname not in players:
                        players[pname] = {}
                    players[pname][prop_type] = {
                        'line': over.get('point', 0),
                        'over_odds': over['price'],
                        'under_odds': under['price'] if under else None,
                        'book': book_key,
                    }

    print(f"        Found props for {len(players)} players. Credits remaining: {remaining}")
    _save_cache(cache_path, players)
    return players

def match_odds_to_game(all_odds: Dict, home_team_name: str, away_team_name: str) -> Optional[Tuple[str, dict]]:
    """
    Matches The Odds API event to our NBA API game by fuzzy-matching team names.
    Returns (event_id, odds_dict) or None.
    """
    home_lower = home_team_name.lower()
    away_lower = away_team_name.lower()

    for event_id, odds in all_odds.items():
        api_home = odds['home_team'].lower()
        api_away = odds['away_team'].lower()

        # Check if the team city+name matches (e.g. "Boston Celtics" contains "celtics")
        home_match = any(w in api_home for w in home_lower.split()) or any(w in home_lower for w in api_home.split())
        away_match = any(w in api_away for w in away_lower.split()) or any(w in away_lower for w in api_away.split())

        if home_match and away_match:
            return (event_id, odds)

    return None


if __name__ == '__main__':
    try:
        odds = fetch_game_odds()
        for eid, g in odds.items():
            print(f"\n{g['away_team']} @ {g['home_team']}")
            if g['h2h']:
                print(f"  H2H: Home {g['h2h']['home']} | Away {g['h2h']['away']} ({g['h2h']['book']})")
            if g['spreads']:
                print(f"  Spread: Home {g['spreads']['home_spread']} @ {g['spreads']['home_odds']} ({g['spreads']['book']})")
            if g['totals']:
                print(f"  Total: {g['totals']['total']} O{g['totals']['over_odds']}/U{g['totals']['under_odds']} ({g['totals']['book']})")

            # Test player props for first event
            props = fetch_player_props(eid)
            for pname, pdata in list(props.items())[:3]:
                print(f"  {pname}:")
                for stat, info in pdata.items():
                    print(f"    {stat}: O{info['line']} @ {info['over_odds']} ({info['book']})")
            break  # Only test first event
    except Exception as e:
        print(f"Error: {e}")
