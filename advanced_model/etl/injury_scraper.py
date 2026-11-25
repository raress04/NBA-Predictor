import requests
import unicodedata
import re
from bs4 import BeautifulSoup
from typing import Set

def _alert(level: str, source: str, message: str) -> None:
    """W2: one [ALERT ...] line (utils/run_alerts.py); falls back to a plain print."""
    try:
        from utils.run_alerts import alert
    except ImportError:
        print(f"[ALERT {level}] {source}: {message}")
        return
    alert(level, source, message)

# Suffixes that appear in one source but not the other
_SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}

def _normalize(name: str) -> str:
    """
    Lowercase, strip diacritics, remove punctuation and known suffixes
    so 'Jimmy Butler III' and 'Jimmy Butler' both become 'jimmy butler'.
    """
    # Remove diacritics
    name = unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode('utf-8')
    # Lowercase and strip
    name = name.lower().strip()
    # Remove punctuation (periods, apostrophes, hyphens)
    name = re.sub(r"[.\'\-]", '', name)
    # Remove known suffixes as standalone words at end
    parts = name.split()
    while parts and parts[-1] in _SUFFIXES:
        parts.pop()
    return ' '.join(parts)


def get_injured_players() -> Set[str]:
    '''
    Scrapes the CBS Sports NBA Injury Report to return a set of currently injured NBA player names.
    Returns both the raw names AND a normalized set so matching is robust
    regardless of suffixes (Jr., III, etc.) or diacritics.
    '''
    url = "https://www.cbssports.com/nba/injuries/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    injured_set = set()       # Raw names from CBS
    injured_normalized = {}   # normalized_name -> raw_name (for debugging)

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # CBS Sports structure: the names are in anchors within a specific cell class
        # The 5th column contains the injury status details
        rows = soup.select('tr.TableBase-bodyTr')
        
        for row in rows:
            name_node = row.select_one('.CellPlayerName--long a')
            if not name_node:
                continue
                
            name = name_node.text.strip()
            cells = row.select('td')
            
            if len(cells) >= 5:
                details = cells[4].text.strip().lower()
                # If they are GTD or Day-to-Day, assume they might play so we don't exclude them
                if 'game time decision' in details or 'day-to-day' in details:
                    continue
                    
            injured_set.add(name)
            injured_normalized[_normalize(name)] = name

        if not rows:
            # W2 (2026-11-25, owner decision): the page loaded but the injury table was not
            # found, so CBS changed its layout. The old fallback marked every player linked
            # anywhere on the page as out (headlines, side panels), which could bench healthy
            # players. Removed: say so loudly and treat everyone as healthy.
            _alert('ERROR', 'cbs_injuries', "page loaded but the injury table was not found "
                                            "(layout changed?) -> all players treated as healthy tonight")
            return injured_set, injured_normalized

        print(f"[+] Loaded live Injury Report: {len(injured_set)} players marked Out/GTD.")

    except Exception as e:
        print(f"[-] Failed to fetch CBS Injury Report: {e}")
        _alert('ERROR', 'cbs_injuries', f"fetch failed ({e}) -> all players treated as healthy tonight")

    return injured_set, injured_normalized


def is_player_injured(player_name: str, injured_normalized: dict) -> bool:
    """
    Fuzzy-match a player name against the injured list.
    Handles suffixes (Jr./III), diacritics, and minor name variations.
    """
    return _normalize(player_name) in injured_normalized


if __name__ == '__main__':
    raw, normalized = get_injured_players()
    print("Sample raw names:", list(raw)[:10])
    print("Sample normalized:", list(normalized.keys())[:10])
    # Test specific known player
    print("Jimmy Butler injured?", is_player_injured("Jimmy Butler III", normalized))
    print("Jimmy Butler injured?", is_player_injured("Jimmy Butler", normalized))
