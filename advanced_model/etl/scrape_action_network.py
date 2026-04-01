import requests
import sqlite3
import time
import json
from datetime import datetime, timedelta
import os
import sys

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'database', 'bet_tracker.db')

# Action Network Props Mapping
PROP_IDS = {
    "POINTS":   "core_bet_type_27_points",
    "REBOUNDS": "core_bet_type_23_rebounds",
    "ASSISTS":  "core_bet_type_26_assists"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

def american_to_decimal(american: int) -> float:
    """Convert American odds (e.g. -110) to decimal odds (e.g. 1.909)."""
    if american is None:
        return 1.909  # default -110
    try:
        american = int(american)
        if american > 0:
            return round(1 + american / 100, 4)
        else:
            return round(1 + 100 / abs(american), 4)
    except Exception:
        return 1.909

def fetch_props(date_str, stat_name, market_key, results):
    """
    Fetch prop markets (Points, Rebounds, Assists).
    results: { (eid, name): {stat: {'line': X, 'over_odds': Y, 'under_odds': Z}} }
    """
    url = f"https://api.actionnetwork.com/web/v2/scoreboard/nba/markets?customPickTypes={market_key}&date={date_str}"
    print(f"    [+] Fetching {stat_name}...")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            print(f"      [!] HTTP {resp.status_code}")
            return
        data = resp.json()
        players_map = {p['id']: p['full_name'] for p in data.get('players', [])}
        all_groups = data.get('markets', {})

        # Temp: { (eid, pid): {'lines': [], 'over_odds': [], 'under_odds': []} }
        temp = {}
        for book_id, info in all_groups.items():
            for ev in info.get('event', {}).get(market_key, []):
                val  = ev.get('value')
                eid  = ev.get('event_id')
                pid  = ev.get('player_id')
                # Action Network stores over/under as separate entries with 'type'
                # type: 'over' or 'under', price / ml / odds depending on version
                otype = ev.get('type', '').lower()
                price = ev.get('ml') or ev.get('odds') or ev.get('price')  # American odds

                if val is None or not eid or not pid:
                    continue

                k = (eid, pid)
                if k not in temp:
                    temp[k] = {'lines': [], 'over_odds': [], 'under_odds': []}

                temp[k]['lines'].append(float(val))
                if otype == 'over' and price is not None:
                    temp[k]['over_odds'].append(int(price))
                elif otype == 'under' and price is not None:
                    temp[k]['under_odds'].append(int(price))
                elif price is not None:
                    # No type tag — treat as symmetric (both sides same odds)
                    temp[k]['over_odds'].append(int(price))
                    temp[k]['under_odds'].append(int(price))

        for (eid, pid), d in temp.items():
            name = players_map.get(pid)
            if not name:
                continue

            consensus_line = max(set(d['lines']), key=d['lines'].count)

            # Consensus odds: most common American price, then convert to decimal
            if d['over_odds']:
                consensus_over_am  = max(set(d['over_odds']),  key=d['over_odds'].count)
                consensus_over_dec = american_to_decimal(consensus_over_am)
            else:
                consensus_over_dec = 1.909  # -110 default

            if d['under_odds']:
                consensus_under_am  = max(set(d['under_odds']), key=d['under_odds'].count)
                consensus_under_dec = american_to_decimal(consensus_under_am)
            else:
                consensus_under_dec = 1.909  # -110 default

            res_key = (eid, name)
            if res_key not in results:
                results[res_key] = {}
            results[res_key][stat_name] = {
                'line':       consensus_line,
                'over_odds':  consensus_over_dec,
                'under_odds': consensus_under_dec,
            }

    except Exception as e:
        print(f"      [!] Error: {e}")


def fetch_game_odds(date_str, results):
    """
    Fetch Spreads and Totals from unified scoreboard.
    results: { (eid, name): {stat: {'line': X, 'over_odds': Y, 'under_odds': Z}} }
    """
    url = f"https://api.actionnetwork.com/web/v2/scoreboard/nba?date={date_str}&periods=event"
    print(f"    [+] Fetching Games (Spreads/Totals)...")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return
        data = resp.json()
        for game in data.get('games', []):
            eid   = game.get('id')
            teams = {t['id']: t['full_name'] for t in game.get('teams', [])}
            local = {}  # { (stat, name): {'lines': [], 'over_odds': [], 'under_odds': []} }

            for book_id, info in game.get('markets', {}).items():
                ev = info.get('event', {})
                # ── Totals ──
                for m in ev.get('total', []):
                    val   = m.get('value')
                    price = m.get('ml') or m.get('odds') or m.get('price')
                    otype = m.get('type', '').lower()
                    if val is None:
                        continue
                    k = ("TOTAL", "Game Total")
                    if k not in local:
                        local[k] = {'lines': [], 'over_odds': [], 'under_odds': []}
                    local[k]['lines'].append(float(val))
                    if otype == 'over' and price:
                        local[k]['over_odds'].append(int(price))
                    elif otype == 'under' and price:
                        local[k]['under_odds'].append(int(price))

                # ── Spreads ──
                for m in ev.get('spread', []):
                    tname = teams.get(m.get('team_id'))
                    val   = m.get('value')
                    price = m.get('ml') or m.get('odds') or m.get('price')
                    if val is None or not tname:
                        continue
                    k = ("SPREAD", tname)
                    if k not in local:
                        local[k] = {'lines': [], 'over_odds': [], 'under_odds': []}
                    local[k]['lines'].append(float(val))
                    if price:
                        local[k]['over_odds'].append(int(price))

            for (stat, name), d in local.items():
                if not d['lines']:
                    continue
                consensus_line = max(set(d['lines']), key=d['lines'].count)
                over_dec  = american_to_decimal(max(set(d['over_odds']),  key=d['over_odds'].count))  if d['over_odds']  else 1.909
                under_dec = american_to_decimal(max(set(d['under_odds']), key=d['under_odds'].count)) if d['under_odds'] else 1.909

                res_key = (eid, name)
                if res_key not in results:
                    results[res_key] = {}
                results[res_key][stat] = {
                    'line':       consensus_line,
                    'over_odds':  over_dec,
                    'under_odds': under_dec,
                }

    except Exception as e:
        print(f"      [!] Error: {e}")


def store_day(date_str, data):
    """
    data: { (eid, name): {stat: {'line': X, 'over_odds': Y, 'under_odds': Z}} }
    Backwards-compatible: also accepts old {stat: line} float format.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    upserts = 0

    for (eid, name), stats in data.items():
        for stat, payload in stats.items():
            # Support both old (float) and new (dict) format
            if isinstance(payload, dict):
                line       = payload['line']
                over_odds  = payload.get('over_odds',  1.909)
                under_odds = payload.get('under_odds', 1.909)
            else:
                line       = float(payload)
                over_odds  = 1.909
                under_odds = 1.909

            cursor.execute("""
                INSERT INTO historical_odds_cache
                    (event_id, date, player_name, stat_category, line, over_odds, under_odds, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id, player_name, stat_category)
                DO UPDATE SET
                    line       = excluded.line,
                    over_odds  = excluded.over_odds,
                    under_odds = excluded.under_odds,
                    date       = excluded.date
            """, (f"AN_{eid}", date_str, name, stat, line, over_odds, under_odds, "ActionNetwork"))
            upserts += 1

    conn.commit()
    conn.close()
    return upserts


def main(start_date, end_date=None):
    if not end_date:
        end_date = start_date
    current = datetime.strptime(start_date, "%Y%m%d")
    end     = datetime.strptime(end_date,   "%Y%m%d")

    while current <= end:
        ds = current.strftime("%Y%m%d")
        print(f"[*] Processing {ds}...")

        day_results = {}
        fetch_game_odds(ds, day_results)
        for stat, mkey in PROP_IDS.items():
            fetch_props(ds, stat, mkey, day_results)
            time.sleep(1.0)   # polite throttling

        if day_results:
            n = store_day(ds, day_results)
            print(f"  [✓] Stored {n} lines (with odds).")
        else:
            print(f"  [!] No data found for {ds}.")

        current += timedelta(days=1)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Usage: python3 etl/scrape_action_network.py YYYYMMDD [YYYYMMDD]")
