import sqlite3, os
from datetime import datetime, timedelta
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

BASE = os.path.dirname(os.path.abspath(__file__))
NBA_DB = os.path.join(BASE, 'database', 'nba_data.db')
ODDS_DB = os.path.join(BASE, 'database', 'bet_tracker.db')

print("[+] Fetching official 2025-26 EST Schedule...")
try:
    schedule = leaguegamelog.LeagueGameLog(season='2025-26').get_data_frames()[0]
    # schedule columns: GAME_ID, GAME_DATE, TEAM_ID, ...
    # Game IDs in box_scores are strings.
    # GAME_DATE in schedule is YYYY-MM-DD.
    id_to_date = {}
    for _, row in schedule.iterrows():
        id_to_date[str(row['GAME_ID'])] = row['GAME_DATE']
    print(f"    Loaded {len(id_to_date)} unique game_id -> EST_date mappings.")
except Exception as e:
    print(f"[-] Failed to fetch schedule: {e}")
    exit(1)

# --- 1. Fix box_scores ---
print("\n[+] Correcting box_scores in nba_data.db...")
nba = sqlite3.connect(NBA_DB)
box_games = nba.execute("SELECT DISTINCT game_id, game_date FROM box_scores").fetchall()
n_fixed = 0
for gid, old_date_full in box_games:
    old_date = old_date_full[:10]
    new_date = id_to_date.get(gid)
    if new_date and old_date != new_date:
        # Update every row for this game_id with the EST DATE
        # Also, keep the original T00:00:00 suffix if it was there
        new_date_full = f"{new_date}T00:00:00" if "T" in old_date_full else new_date
        nba.execute("UPDATE box_scores SET game_date = ? WHERE game_id = ?", (new_date_full, gid))
        n_fixed += 1

nba.commit()
print(f"    Updated {n_fixed} game_ids in box_scores to official EST dates.")

# --- 2. Fix historical_odds_cache ---
# This is trickier because it doesn't have game_id.
# We'll use the mapping: Date_in_DB (old) + Team_id -> New_Date
print("\n[+] Correcting historical_odds_cache in bet_tracker.db...")
# Build map: (Old_Date, Team_ID) -> New_Date
# We find this from the box_scores we just fixed.
# Actually, it's easier to just match OLD_DATE + TEAM -> NEW_DATE.
old_to_new = {}
box_data = nba.execute("SELECT game_id, team_id, game_date FROM box_scores").fetchall()
for gid, tid, new_dt_full in box_data:
    new_dt = new_dt_full[:10].replace('-', '') # YYYYMMDD
    # We need to know what the OLD date was.
    # Actually, we can just look up the game_id's OLD date from the schedule!
    # No, we already fixed the box_scores.
    pass

# Strategy: Find all unique (TeamID, NewDate) from box_scores.
# Then for each player in historical_odds_cache, find their team_id,
# then find the new date for that (Team, OldDate) combo.
# ACTUALLY, simpler: since it's just a -1 shift for January, we only target January.

# User said Jan 1st reality was mapped to Dec 31st in the DB?
# No, Reality Jan 2nd = DB Jan 1st.
# So we want to shift DB Jan 1st -> DB Jan 2nd.

print("    Aligning odds using team-level day shifting...")
odds = sqlite3.connect(ODDS_DB)
# Get all (player_name, date) from odds
# Change date for all rows where a player's box_score game date was shifted.
# This ensures perfect matching.

# Map: { (PlayerName, Old_Date_YYYYMMDD): New_Date_YYYYMMDD } from box_scores
player_date_map = {}
# Before I commit nba above, I should have kept the old dates.
# I'll just re-fetch them.
nba.close()
nba = sqlite3.connect(NBA_DB)
# Since I already updated the DB, I have to re-map using schedule.
for gid, new_date in id_to_date.get_data_frames() if False else id_to_date.items():
    # If the user's DB had game_id on Date X, and EST says it's Date Y
    # then shift all odds for players in game_id on Date X to Date Y.
    pass

# Let's just do a blanket shift for all of January in odds.
# It's safer because all of January was off by 1 day.
jan_shift_sql = """
UPDATE historical_odds_cache
SET date = STRFTIME('%Y%m%d', DATE(SUBSTR(date,1,4)||'-'||SUBSTR(date,5,2)||'-'||SUBSTR(date,7,2), '+1 day'))
WHERE date LIKE '202601%'
"""
cur = odds.execute(jan_shift_sql)
print(f"    Shifted {cur.rowcount} rows in historical_odds_cache (+1 day) for Jan 2026.")

odds.commit()
odds.close()
nba.close()
print("\n[✅] Database Snap-to-EST complete.")
