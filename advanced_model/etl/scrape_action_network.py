import requests
import sqlite3
import time
import json
from datetime import datetime, timedelta
import os
import sys

DB_PATH = "database/bet_tracker.db"

# Action Network Props Mapping
PROP_IDS = {
    "POINTS": "core_bet_type_27_points",
    "REBOUNDS": "core_bet_type_23_rebounds",
    "ASSISTS": "core_bet_type_26_assists"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

def fetch_props(date_str, stat_name, market_key, results):
    """
    Fetch prop markets (Points, Rebounds, Assists).
    results: { (eid, name): {stat: line} }
    """
    url = f"https://api.actionnetwork.com/web/v2/scoreboard/nba/markets?customPickTypes={market_key}&date={date_str}"
    print(f"    [+] Fetching {stat_name}...")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200: return
        data = resp.json()
        players_map = {p['id']: p['full_name'] for p in data.get('players', [])}
        all_groups = data.get('markets', {})
        
        # Temp: { (eid, pid): [lines] }
        temp = {}
        for book_id, info in all_groups.items():
            for ev in info.get('event', {}).get(market_key, []):
                val, eid, pid = ev.get('value'), ev.get('event_id'), ev.get('player_id')
                if val is not None and eid and pid:
                    k = (eid, pid)
                    if k not in temp: temp[k] = []
                    temp[k].append(val)
        
        for (eid, pid), lines in temp.items():
            name = players_map.get(pid)
            if not name: continue
            consensus = max(set(lines), key=lines.count)
            res_key = (eid, name)
            if res_key not in results: results[res_key] = {}
            results[res_key][stat_name] = consensus
    except Exception as e: print(f"      [!] Error: {e}")

def fetch_game_odds(date_str, results):
    """
    Fetch Spreads and Totals from unified scoreboard.
    results: { (eid, name): {stat: line} }
    """
    url = f"https://api.actionnetwork.com/web/v2/scoreboard/nba?date={date_str}&periods=event"
    print(f"    [+] Fetching Games...")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200: return
        data = resp.json()
        for game in data.get('games', []):
            eid = game.get('id')
            teams = {t['id']: t['full_name'] for t in game.get('teams', [])}
            # { (stat, name): [lines] }
            local_lines = {}
            for book_id, info in game.get('markets', {}).items():
                ev = info.get('event', {})
                # Total
                for m in ev.get('total', []):
                    if m.get('value') is not None:
                        k = ("TOTAL", "Game Total")
                        if k not in local_lines: local_lines[k] = []
                        local_lines[k].append(float(m['value']))
                # Spread
                for m in ev.get('spread', []):
                    tname = teams.get(m.get('team_id'))
                    if m.get('value') is not None and tname:
                        k = ("SPREAD", tname)
                        if k not in local_lines: local_lines[k] = []
                        local_lines[k].append(float(m['value']))
            # Resolve
            for (stat, name), lines in local_lines.items():
                consensus = max(set(lines), key=lines.count)
                res_key = (eid, name)
                if res_key not in results: results[res_key] = {}
                results[res_key][stat] = consensus
    except Exception as e: print(f"      [!] Error: {e}")

def store_day(date_str, data):
    """ data: { (eid, name): {stat: line} } """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    upserts = 0
    for (eid, name), stats in data.items():
        for stat, line in stats.items():
            cursor.execute("""
                INSERT INTO historical_odds_cache (event_id, date, player_name, stat_category, line, source)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id, player_name, stat_category) DO UPDATE SET line = excluded.line, date = excluded.date
            """, (f"AN_{eid}", date_str, name, stat, line, "ActionNetwork"))
            upserts += 1
    conn.commit()
    conn.close()
    return upserts

def main(start_date, end_date):
    if len(start_date) == 8 and not end_date: end_date = start_date
    current = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    
    while current <= end:
        ds = current.strftime("%Y%m%d")
        print(f"[*] Processing {ds}...")
        
        day_results = {}
        # Get Game Odds first (faster)
        fetch_game_odds(ds, day_results)
        # Get Props
        for stat, mkey in PROP_IDS.items():
            fetch_props(ds, stat, mkey, day_results)
            time.sleep(1.0) # Throttling
            
        if day_results:
            n = store_day(ds, day_results)
            print(f"  [✓] Stored {n} lines.")
        else:
            print(f"  [!] No data found.")
            
        current += timedelta(days=1)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        main(sys.argv[1], None)
    else:
        print("Usage: python3 etl/scrape_action_network.py YYYYMMDD [YYYYMMDD]")
