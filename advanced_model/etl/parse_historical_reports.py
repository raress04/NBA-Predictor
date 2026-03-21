import os
import re
import glob
import sqlite3
import argparse
from datetime import datetime

def parse_historical_file(filepath):
    """
    Return list of dicts with keys:
    game_date, matchup, player, category, projected_value, source_file, mode='historical'.
    """
    basename = os.path.basename(filepath)
    # predicts_2026-01-01.txt
    date_str = basename.replace(".txt", "").split("_")[1]
    
    with open(filepath, 'r', encoding='utf-8') as fh:
        content = fh.read()

    # Find all matchup blocks. Each block starts with ================= (line 1) MATCHUP: ... (line 2) ================= (line 3)
    # We can split by the MATCHUP line itself if we're careful.
    
    # Better yet: Find all MATCHUP lines and the text that follows until the next MATCHUP
    pattern = re.compile(r'MATCHUP:\s*(.+?)\n(.*?)(?=MATCHUP:|$)', re.DOTALL)
    
    records = []
    for m_block in pattern.finditer(content):
        matchup = m_block.group(1).strip()
        section = m_block.group(2)

        # Projected Total
        m_total = re.search(r'PROJECTED TOTAL:\s*([\d\.]+)', section)
        if m_total:
            records.append({
                'game_date': date_str,
                'matchup': matchup,
                'player': 'Game Total',
                'category': 'TOTAL',
                'projected_value': float(m_total.group(1)),
                'source_file': basename,
                'mode': 'historical'
            })

        # Projected Spread
        # PROJECTED SCORE: LAC 122.9 - UTA 113.1
        m_score = re.search(r'PROJECTED SCORE:\s*([A-Z]{2,3})\s*([\d\.]+)\s*-\s*([A-Z]{2,3})\s*([\d\.]+)', section)
        if m_score:
            t1, s1, t2, s2 = m_score.groups()
            spread = float(s1) - float(s2) # t1 margin
            records.append({
                'game_date': date_str,
                'matchup': matchup,
                'player': t1, # The first team in the score line is our reference for spread
                'category': 'SPREAD',
                'projected_value': spread,
                'source_file': basename,
                'mode': 'historical'
            })

        # Player Projections
        # Kawhi Leonard        -> 32 PTS, 6 REB, 5 AST
        player_re = re.compile(r'^\s*([A-Za-z\s\.\'-]+?)\s*->\s*(\d+)\s+PTS,\s+(\d+)\s+REB,\s+(\d+)\s+AST', re.MULTILINE)
        for m in player_re.finditer(section):
            player = m.group(1).strip()
            pts, reb, ast = m.groups()[1:]
            
            # Add separate records for each category
            categories = {
                'POINTS': float(pts),
                'REBOUNDS': float(reb),
                'ASSISTS': float(ast)
            }
            for cat, val in categories.items():
                records.append({
                    'game_date': date_str,
                    'matchup': matchup,
                    'player': player,
                    'category': cat,
                    'projected_value': val,
                    'source_file': basename,
                    'mode': 'historical'
                })

    return records

def main():
    parser = argparse.ArgumentParser(description='Parse historical NBA reports')
    parser.add_argument('directory', help='Directory containing historical reports (e.g. predictions/january)')
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.directory, "predicts_*.txt"))
    all_records = []
    for f in files:
        print(f"[*] Parsing {f}...")
        records = parse_historical_file(f)
        all_records.extend(records)
        print(f"    Found {len(records)} projections")

    print(f"\n[+] Total historical projections parsed: {len(all_records)}")
    
    # In Phase 1.C, we will write these to the projection_outcomes table.
    # For now, we just verify the parsing.

if __name__ == "__main__":
    main()
