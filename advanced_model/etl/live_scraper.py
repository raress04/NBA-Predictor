# etl/live_scraper.py — SHIM
# DEPRECATED: Canonical module is etl/live_ingestion.py
# This shim exists so that existing imports and scripts (e.g. run_pipeline.sh) keep working.

from etl.live_ingestion import (
    main, is_player_banned, parse_roster, match_team_name, get_opposing_team,
    fetch_daily_box_scores, parse_daily_box_scores, process_games
)

if __name__ == '__main__':
    # Forward CLI executions to live_ingestion
    import sys
    import runpy
    sys.argv[0] = 'etl.live_ingestion'
    runpy.run_module('etl.live_ingestion', run_name='__main__')
