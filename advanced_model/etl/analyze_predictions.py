import re
import os
import numpy as np
import datetime
from collections import defaultdict

import sqlite3

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTS_PATH = os.path.join(MODEL_DIR, 'predicts.txt')
RESULTS_PATH = os.path.join(MODEL_DIR, 'yesterday_results.txt')

def write_db_results(target_date, props_results, medians_results, actual_games, actual_props):
    db_path = os.path.join(MODEL_DIR, "database", "bet_tracker.db")
    if not os.path.exists(db_path):
        print(f"[-] DB not found at {db_path}, skipping write.")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. Update pick_results for VALUABLE picks
    if props_results:
        inserted_picks = 0
        for r in props_results:
            team_or_player = r['player']
            stat = r['stat']
            direction = r['direction']
            is_win = 1 if 'WON' in r['status'] else 0

            # Find pick_id
            cur.execute('''
                SELECT id FROM daily_picks 
                WHERE date = ? AND UPPER(player_or_team) = ? AND UPPER(stat_category) = ? AND UPPER(direction) = ?
            ''', (target_date, team_or_player.upper(), stat.upper(), direction.upper()))
            row = cur.fetchone()
            if row:
                pick_id = row[0]
                # Check exist
                cur.execute("SELECT id FROM pick_results WHERE pick_id = ?", (pick_id,))
                if not cur.fetchone():
                    cur.execute('''
                        INSERT INTO pick_results (pick_id, actual_value, is_win, diff_val, diff_pct, resolved_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (pick_id, r['actual'], is_win, r['diff_val'], r['diff_pct'], target_date))
                    inserted_picks += 1
        print(f"    -> Logged {inserted_picks} evaluated results to pick_results.")

    # 2. Write medians to projection_outcomes (raw accuracy base)
    if medians_results:
        inserted_po = 0
        for m in medians_results:
            player = m['player']
            for stat_short, stat_db in [('pts', 'POINTS'), ('reb', 'REBOUNDS'), ('ast', 'ASSISTS')]:
                pred = m[f'pred_{stat_short}']
                act = m[f'act_{stat_short}']
                bias = act - pred # model_bias = actual - projected
                
                cur.execute("SELECT id FROM projection_outcomes WHERE game_date = ? AND player = ? AND category = ?", 
                            (target_date, player, stat_db))
                if not cur.fetchone():
                    cur.execute('''
                        INSERT INTO projection_outcomes 
                        (game_date, player, category, projected_value, actual_value, model_bias, source_mode, source_file)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (target_date, player, stat_db, pred, act, bias, 'live', 'analyze_predictions.py'))
                    inserted_po += 1
        print(f"    -> Logged {inserted_po} raw player median records to projection_outcomes.")

    # 3. Evaluate shadow_picks for this date that are not yet resolved
    cur.execute('''
        SELECT id, player, stat_category, direction, line 
        FROM shadow_picks 
        WHERE game_date = ? AND actual_result IS NULL
    ''', (target_date,))
    shadows = cur.fetchall()
    
    shadow_updates = 0
    for sid, player, stat_cat, direction, line in shadows:
        actual_val = None
        is_win = 0
        stat_key = stat_cat.upper()
        direction = direction.upper()
        
        if stat_key in ['POINTS', 'REBOUNDS', 'ASSISTS']:
            if player in actual_props:
                actual_val = actual_props[player][stat_key]
                if direction == 'OVER' and actual_val > line:
                    is_win = 1
                elif direction == 'UNDER' and actual_val < line:
                    is_win = 1
                    
        elif stat_key == 'SPREAD':
            for matchup, scores in actual_games.items():
                if player in scores: # player is the team name here
                    other_team = [t for t in scores.keys() if t != player][0]
                    margin = scores[player] - scores[other_team]
                    actual_val = margin
                    # Cover check logic
                    diff_val = margin + line
                    is_win = 1 if diff_val > 0 else 0
                    break
                    
        elif stat_key == 'TOTAL':
            for matchup, scores in actual_games.items():
                if "TOTAL" in player.upper() or True: # fallback since "Game Total" doesn't have matchup context clearly
                    # For total, we can only roughly try to map if we had the teams, but shadow pick might just be Game Total.
                    # We skip for now unless it's strictly mapped
                    pass

        if actual_val is not None:
            cur.execute('''
                UPDATE shadow_picks 
                SET actual_value = ?, actual_result = ?
                WHERE id = ?
            ''', (actual_val, is_win, sid))
            shadow_updates += 1

    conn.commit()
    conn.close()
    print(f"    -> Evaluated and updated {shadow_updates} shadow_picks actual outcomes.")

def get_month_folder(date_str):
    """Maps YYYY-MM-DD to lowercase month name (e.g. 'march')."""
    try:
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%B").lower()
    except:
        return datetime.datetime.now().strftime("%B").lower()

def parse_predicts(filepath):
    """Parses predicts.txt for game projections and valuable props."""
    if not os.path.exists(filepath):
        print(f"[-] {filepath} not found.")
        return {}, []

    games = {}
    props = []
    medians = []
    
    current_game = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        
        # Match Game Header: MATCHUP: Away Team @ Home Team
        matchup_match = re.search(r'MATCHUP:\s+(.*?)\s+@\s+(.*)', line)
        if matchup_match:
            away = matchup_match.group(1).strip()
            home = matchup_match.group(2).strip()
            current_game = f"{away} @ {home}"
            games[current_game] = {'market_spread': None, 'proj_margin': None}
            continue

        # Match Proj Score: PROJECTED SCORE: WAS 108.2 - HOU 122.0
        score_match = re.search(r'PROJECTED SCORE:\s+([A-Z]+)\s+([\d\.]+)\s+-\s+([A-Z]+)\s+([\d\.]+)', line)
        if score_match and current_game:
            team1, score1, team2, score2 = score_match.groups()
            games[current_game]['tricodes'] = (team1, team2)
            games[current_game]['proj_scores'] = {team1: float(score1), team2: float(score2)}
            continue

        # Match Total: -> GAME TOTAL: O/U 224.5 VALUABLE: Over (56% conf, 10.3% edge)
        total_match = re.search(r'GAME TOTAL:\s+O/U\s+([\d\.]+)\s+VALUABLE:\s+(Over|Under)(?:\s+\((\d+)%\s+conf)?', line)
        if total_match and current_game:
            line_val, direction, conf = total_match.groups()
            props.append({
                'game': current_game,
                'player': 'GAME TOTAL',
                'stat': 'TOTAL',
                'line': float(line_val),
                'direction': direction.upper(),
                'conf': int(conf) if conf else None
            })

        # Match Spread: -> HOME SPREAD: BOS -6.5 VALUABLE: Cover (62% conf, 12.0% edge)
        spread_match = re.search(r'(HOME|AWAY) SPREAD:\s+([A-Z]+)\s+([+-][\d\.]+)\s+VALUABLE:\s+Cover(?:\s+\((\d+)%\s+conf)?', line)
        if spread_match and current_game:
            _, team_tricode, spread_val, conf = spread_match.groups()
            props.append({
                'game': current_game,
                'player': f'{team_tricode} SPREAD',
                'stat': 'SPREAD',
                'line': float(spread_val),
                'direction': 'COVER',
                'team_tricode': team_tricode,
                'conf': int(conf) if conf else None
            })

        # Match inline VALUABLE props: Player -> 18 PTS, 4 REB, 2 AST (O/U 12.5 points VALUABLE: Under (80% conf, 30.3% edge))
        if 'VALUABLE:' in line and current_game:
            # We want to extract the player and the specific valuable bets
            player_match = re.search(r'^\s*([A-Za-z\s\.\'-]+)\s*->', line)
            
            # Fallback for the old standalone format (no '->' on the same line)
            if not player_match:
                player_match = re.search(r'^\s*([A-Za-z\s\.\'-]+)\s*\|', line)
                
            if player_match:
                player_name = player_match.group(1).strip()
                
                # Find all (O/U X.X stat VALUABLE: Dir ... ) or O/U X.X stat VALUABLE: Dir
                valuable_matches = re.findall(r'O/U\s+([\d\.]+)\s+([a-zA-Z]+)\s+VALUABLE:\s+(Over|Under)(?:\s+\((\d+)%\s+conf)?', line)
                for val_match in valuable_matches:
                    line_val, stat, direction, conf = val_match
                    props.append({
                        'game': current_game,
                        'player': player_name,
                        'stat': stat.upper(),
                        'line': float(line_val),
                        'direction': direction.upper(),
                        'conf': int(conf) if conf else None
                    })
        # Match Median Stats: Anthony Edwards      -> 29 PTS, 7 REB, 6 AST
        median_match = re.search(r'^\s*([A-Za-z\s\.\'-]+)\s*->\s*(\d+)\s*PTS,\s*(\d+)\s*REB,\s*(\d+)\s*AST', line)
        if median_match and current_game:
            player_name, pts, reb, ast = median_match.groups()
            medians.append({
                'game': current_game,
                'player': player_name.strip(),
                'PTS': int(pts),
                'REB': int(reb),
                'AST': int(ast)
            })

    return games, props, medians

def parse_results(filepath):
    """Parses yesterday_results.txt for actual game scores and player stats."""
    if not os.path.exists(filepath):
        print(f"[-] {filepath} not found.")
        return {}, {}

    actual_games = {}
    actual_props = defaultdict(dict)
    
    current_matchup = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        
        # Matchup Headers in results: MATCHUP: WAS vs. HOU    118 - 123
        matchup_match = re.search(r'MATCHUP:\s+(.*?)\s+vs\.\s+(.*?)\s+(\d+)\s+-\s+(\d+)', line)
        if matchup_match:
            team1, team2, score1, score2 = matchup_match.groups()
            current_matchup = (team1.strip(), team2.strip())
            actual_games[current_matchup] = {team1.strip(): int(score1), team2.strip(): int(score2)}
            
            # Also register the reverse for easy lookup
            actual_games[(team2.strip(), team1.strip())] = {team1.strip(): int(score1), team2.strip(): int(score2)}
            continue
            
        # Match Player Stats: Alperen Sengun         | 32 PTS, 13 REB,  2 AST,  0 3PM
        stat_match = re.search(r'^([A-Za-z\s\.\'-]+)\s*\|\s*(\d+)\s*PTS,\s*(\d+)\s*REB,\s*(\d+)\s*AST', line)
        if stat_match:
            player_name = stat_match.group(1).strip()
            pts, reb, ast = map(int, stat_match.groups()[1:])
            actual_props[player_name] = {'POINTS': pts, 'REBOUNDS': reb, 'ASSISTS': ast}
            
    return actual_games, actual_props

def generate_analysis(predicts_path=None, results_path=None, output_dir=None, write_db=False):
    # Auto-resolve paths if only filename provided or specific date needed
    if predicts_path and not os.path.exists(predicts_path):
        # Check if it matches predicts_YYYY-MM-DD.txt
        match = re.search(r'predicts_(\d{4}-\d{2}-\d{2})\.txt', predicts_path)
        if match:
            date_str = match.group(1)
            month = get_month_folder(date_str)
            predicts_path = os.path.join(MODEL_DIR, 'predictions', month, predicts_path)
            
    if results_path and not os.path.exists(results_path):
        match = re.search(r'yesterday_results_(\d{4}-\d{2}-\d{2})\.txt', results_path)
        if match:
            date_str = match.group(1)
            month = get_month_folder(date_str)
            results_path = os.path.join(MODEL_DIR, 'results', month, results_path)

    predicts_path = predicts_path or PREDICTS_PATH
    results_path = results_path or RESULTS_PATH
    
    # Attempt to extract target date from files, fallback to yesterday
    date_str = None
    path_match = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(predicts_path) + os.path.basename(results_path))
    if path_match:
        date_str = path_match.group(1)
    else:
        date_str = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    # Force output directory to analyze/month
    month = get_month_folder(date_str)
    output_dir = os.path.join(MODEL_DIR, 'analyze', month)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[+] Running automated prediction analysis for {date_str}...")
    print(f"    -> Output directory: {output_dir}")
    
    games, props, medians = parse_predicts(predicts_path)
    actual_games, actual_props = parse_results(results_path)
    
    if not props and not medians:
        print(f"[-] No valuable props or medians found in {os.path.basename(predicts_path)} to analyze.")
        return
        
    if not actual_props:
        print(f"[-] No actual player results found in {os.path.basename(results_path)}")
        return

    # Clean up previous analyze txt file if it exists to prevent appending endlessly
    base_name = f'analyze_{date_str}'
    txt_out_path = os.path.join(output_dir, f'{base_name}.txt')
    if os.path.exists(txt_out_path):
        os.remove(txt_out_path)
    
    # Always run Game Results Analysis (Team stats) first at the top
    print(f"    -> Running Game Accuracy Analysis.")
    analyze_games(games, actual_games, output_dir=output_dir, date_str=date_str)

    props_results = []
    # Run props analysis (VALUABLE bets WIN/LOSS) whenever prop data exists
    if props:
        print(f"    -> Running Pick Analysis ({len(props)} valuable props).")
        props_results = analyze_props(props, games, actual_games, actual_props, is_only_props=(not medians), output_dir=output_dir, date_str=date_str)
        
    # Always run median accuracy analysis if medians exist (runs alongside props if both available)
    medians_results = []
    if medians:
        print(f"    -> Running Median Accuracy Analysis ({len(medians)} player projections).")
        medians_results = analyze_medians(medians, actual_props, is_only_medians=(not props), output_dir=output_dir, date_str=date_str)
        
    if write_db:
        # We need the target date, try to extract from paths or use yesterday
        date_str = None
        match = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(predicts_path))
        if match:
            date_str = match.group(1)
        else:
            date_str = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        
        print(f"\n[+] Writing evaluated outcomes to DB for {date_str}...")
        write_db_results(date_str, props_results, medians_results, actual_games, actual_props)

def analyze_games(games, actual_games, output_dir=None, date_str=None):
    output_dir = output_dir or MODEL_DIR
    base_name = f'analyze_{date_str}' if date_str else 'analyze'
    
    # ── Game Result Analysis ──
    game_results = []
    team_errors = []
    total_errors = []
    spread_errors = []
    ml_correct = 0
    ml_total = 0
    
    for game_label, game_data in games.items():
        tricodes = game_data.get('tricodes')
        proj_scores = game_data.get('proj_scores')
        if not tricodes or not proj_scores or tricodes not in actual_games:
            continue
            
        act_scores = actual_games[tricodes]
        t1, t2 = tricodes
        
        p1, p2 = proj_scores[t1], proj_scores[t2]
        a1, a2 = act_scores[t1], act_scores[t2]
        
        # Errors
        err1 = abs(p1 - a1)
        err2 = abs(p2 - a2)
        team_errors.extend([err1, err2])
        
        proj_total = p1 + p2
        act_total = a1 + a2
        total_errors.append(abs(proj_total - act_total))
        
        proj_spread = p1 - p2 # Margin for team 1
        act_spread = a1 - a2
        spread_errors.append(abs(proj_spread - act_spread))
        
        # Moneyline Accuracy
        if (p1 > p2 and a1 > a2) or (p1 < p2 and a1 < a2):
            ml_correct += 1
        ml_total += 1
        
        game_results.append({
            'matchup': game_label,
            'proj': f"{t1} {p1:.1f} - {t2} {p2:.1f}",
            'actual': f"{t1} {a1} - {t2} {a2}",
            'err_total': abs(proj_total - act_total),
            'status': "✅" if (p1 > p2) == (a1 > a2) else "❌"
        })
        
    game_mae_team = np.mean(team_errors) if team_errors else 0
    game_mae_total = np.mean(total_errors) if total_errors else 0
    game_mae_spread = np.mean(spread_errors) if spread_errors else 0
    ml_pct = (ml_correct / ml_total * 100) if ml_total > 0 else 0
    
    txt_out = os.path.join(output_dir, f'{base_name}.txt')
    # Mode 'a' if file exists, but we want it at the top, so we handle creation in generate_analysis
    with open(txt_out, 'a', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(" 🎯 AUTOMATED PREDICTION ANALYSIS 🎯\n")
        f.write("="*60 + "\n\n")
        
        if ml_total > 0:
            f.write("="*60 + "\n")
            f.write(" 🏀 GAME RESULT ACCURACY (TEAM SCORES) 🏀\n")
            f.write("="*60 + "\n")
            f.write(f"MAE - Team Score: {game_mae_team:.1f} | Total: {game_mae_total:.1f} | Spread (Margin): {game_mae_spread:.1f}\n")
            f.write(f"Moneyline Accuracy: {ml_correct}/{ml_total} ({ml_pct:.1f}%)\n\n")
            
            f.write(f"{'MATCHUP':<35} | {'PROJECTED':<20} | {'ACTUAL':<15} | {'STATUS'}\n")
            f.write("-" * 88 + "\n")
            for gr in game_results:
                f.write(f"{gr['matchup']:<35} | {gr['proj']:<20} | {gr['actual']:<15} | {gr['status']}\n")
            f.write("\n")
        else:
            f.write("[-] No game result data matched for analysis.\n\n")

def analyze_props(props, games, actual_games, actual_props, is_only_props=True, output_dir=None, date_str=None):
    output_dir = output_dir or MODEL_DIR
    results = []
    wins = 0
    losses = 0
    
    # Determine base filename
    base_name = f'analyze_{date_str}' if date_str else 'analyze'

    for p in props:
        player = p['player']
        stat_key = p['stat']
        direction = p['direction']
        pred_line = p['line']
        
        is_win = False
        diff_pct = 0
            
        if stat_key == 'TOTAL':
            tricodes = games.get(p['game'], {}).get('tricodes')
            if not tricodes or tricodes not in actual_games:
                continue
            scores = actual_games[tricodes]
            actual_val = scores[tricodes[0]] + scores[tricodes[1]]
            
            diff_val = actual_val - pred_line
            diff_pct = (diff_val / pred_line) * 100 if pred_line > 0 else 0
            
            if direction == 'OVER' and actual_val > pred_line:
                is_win = True
            elif direction == 'UNDER' and actual_val < pred_line:
                is_win = True
                
        elif stat_key == 'SPREAD':
            tricodes = games.get(p['game'], {}).get('tricodes')
            if not tricodes or tricodes not in actual_games:
                continue
            scores = actual_games[tricodes]
            team_tricode = p['team_tricode']
            other_team = tricodes[0] if tricodes[1] == team_tricode else tricodes[1]
            
            margin = scores[team_tricode] - scores[other_team]
            actual_val = margin
            
            # To cover, margin + pred_line must be > 0.
            # E.g. WAS spread is +14.5. Margin was WAS (-5). -5 + 14.5 = 9.5 > 0 (Covered)
            diff_val = margin + pred_line
            is_win = diff_val > 0
            
        elif stat_key in ['POINTS', 'REBOUNDS', 'ASSISTS']:
            if player not in actual_props:
                continue
            actual_val = actual_props[player][stat_key]
            
            diff_val = actual_val - pred_line
            diff_pct = (diff_val / pred_line) * 100 if pred_line > 0 else 0
            
            if direction == 'OVER' and actual_val > pred_line:
                is_win = True
            elif direction == 'UNDER' and actual_val < pred_line:
                is_win = True
        else:
            continue
            
        if is_win:
            wins += 1
            status = 'WON \u2705'
        else:
            losses += 1
            status = 'LOST \u274c'
            
        results.append({
            'player': player,
            'stat': stat_key,
            'direction': direction,
            'line': pred_line,
            'actual': actual_val,
            'diff_val': diff_val,
            'diff_pct': diff_pct,
            'status': status,
            'conf': p.get('conf')
        })
        
    # Sort results by confidence descending
    results.sort(key=lambda x: x['conf'] if x['conf'] is not None else -1, reverse=True)
        
    total = wins + losses
    win_rate = (wins / total * 100) if total > 0 else 0
    
    # ── Game Result Analysis ──
    game_results = []
    team_errors = []
    total_errors = []
    spread_errors = []
    ml_correct = 0
    ml_total = 0
    
    for game_label, game_data in games.items():
        tricodes = game_data.get('tricodes')
        proj_scores = game_data.get('proj_scores')
        if not tricodes or not proj_scores or tricodes not in actual_games:
            continue
            
        act_scores = actual_games[tricodes]
        t1, t2 = tricodes
        
        p1, p2 = proj_scores[t1], proj_scores[t2]
        a1, a2 = act_scores[t1], act_scores[t2]
        
        # Errors
        err1 = abs(p1 - a1)
        err2 = abs(p2 - a2)
        team_errors.extend([err1, err2])
        
        proj_total = p1 + p2
        act_total = a1 + a2
        total_errors.append(abs(proj_total - act_total))
        
        proj_spread = p1 - p2 # Margin for team 1
        act_spread = a1 - a2
        spread_errors.append(abs(proj_spread - act_spread))
        
        # Moneyline Accuracy
        if (p1 > p2 and a1 > a2) or (p1 < p2 and a1 < a2):
            ml_correct += 1
        ml_total += 1
        
        game_results.append({
            'matchup': game_label,
            'proj': f"{t1} {p1:.1f} - {t2} {p2:.1f}",
            'actual': f"{t1} {a1} - {t2} {a2}",
            'err_total': abs(proj_total - act_total),
            'status': "✅" if (p1 > p2) == (a1 > a2) else "❌"
        })
        
    game_mae_team = np.mean(team_errors) if team_errors else 0
    game_mae_total = np.mean(total_errors) if total_errors else 0
    game_mae_spread = np.mean(spread_errors) if spread_errors else 0
    ml_pct = (ml_correct / ml_total * 100) if ml_total > 0 else 0
            
    # Text Generation
    txt_out = os.path.join(output_dir, f'{base_name}.txt')
    with open(txt_out, 'a', encoding='utf-8') as f:
        f.write(f"Total Props Evaluated: {total}\n")
        f.write(f"Wins: {wins} | Losses: {losses} | Win Rate: {win_rate:.1f}%\n\n")

        if ml_total > 0:
            f.write("="*60 + "\n")
            f.write(" 🏀 GAME RESULT ACCURACY (TEAM SCORES) 🏀\n")
            f.write("="*60 + "\n")
            f.write(f"MAE - Team Score: {game_mae_team:.1f} | Total: {game_mae_total:.1f} | Spread: {game_mae_spread:.1f}\n")
            f.write(f"Moneyline Accuracy: {ml_correct}/{ml_total} ({ml_pct:.1f}%)\n\n")
            
            f.write(f"{'MATCHUP':<35} | {'PROJECTED':<20} | {'ACTUAL':<15} | {'STATUS'}\n")
            f.write("-" * 88 + "\n")
            for gr in game_results:
                f.write(f"{gr['matchup']:<35} | {gr['proj']:<20} | {gr['actual']:<15} | {gr['status']}\n")
            f.write("\n")
        
        f.write(f"{'PLAYER':<22} | {'CONF':<4} | {'BET':<15} | {'ACTUAL':<8} | {'DIFF':<7} | {'DIFF %':<8} | {'STATUS'}\n")
        f.write("-" * 88 + "\n")
        
        for r in results:
            bet_str = f"{r['direction']} {r['line']} {r['stat'][:3]}"
            diff_str = f"{r['diff_val']:+.1f}"
            diff_pct_str = f"{r['diff_pct']:+.1f}%"
            conf_str = f"{r['conf']}%" if r['conf'] else "N/A"
            f.write(f"{r['player']:<22} | {conf_str:<4} | {bet_str:<15} | {r['actual']:<8} | {diff_str:<7} | {diff_pct_str:<8} | {r['status']}\n")
            
    # Add date suffix to files if we are in a month folder
    print(f"[+] Text report generated at: {txt_out}")

    return results

def analyze_medians(medians, actual_props, is_only_medians=True, output_dir=None, date_str=None):
    output_dir = output_dir or MODEL_DIR
    results = []
    
    # Determine base filename
    base_name = f'analyze_{date_str}' if date_str else 'analyze'

    pts_err, reb_err, ast_err = 0, 0, 0
    pts_wins, reb_wins, ast_wins = 0, 0, 0  # direction accuracy
    count = 0
    
    for m in medians:
        player = m['player']
        if player not in actual_props:
            continue
            
        actual = actual_props[player]
        pts_diff = m['PTS'] - actual['POINTS']
        reb_diff = m['REB'] - actual['REBOUNDS']
        ast_diff = m['AST'] - actual['ASSISTS']
        
        count += 1

        # We flag WIN logically for the summary, but text will show HIGH/LOW
        pts_win_log = abs(pts_diff) <= 4
        reb_win_log = abs(reb_diff) <= 2
        ast_win_log = abs(ast_diff) <= 2

        def format_hl(diff, is_win):
            txt = 'EXACT' if diff == 0 else ('LOW' if diff < 0 else 'HIGH')
            return txt

        pts_win = format_hl(pts_diff, pts_win_log)
        reb_win = format_hl(reb_diff, reb_win_log)
        ast_win = format_hl(ast_diff, ast_win_log)
        
        results.append({
            'player': player,
            'pred_pts': m['PTS'], 'act_pts': actual['POINTS'], 'diff_pts': pts_diff, 'pts_win': pts_win, 'pts_win_log': pts_win_log,
            'pred_reb': m['REB'], 'act_reb': actual['REBOUNDS'], 'diff_reb': reb_diff, 'reb_win': reb_win, 'reb_win_log': reb_win_log,
            'pred_ast': m['AST'], 'act_ast': actual['ASSISTS'], 'diff_ast': ast_diff, 'ast_win': ast_win, 'ast_win_log': ast_win_log,
        })
        
    if count == 0:
        print("[-] No matching players found between predictions and actual results.")
        return results
        
    # Helper to calculate filtered metrics using IQR
    def get_filtered_metrics(diffs_list, tol):
        if not diffs_list:
            return 0, 0, 0.0, 0.0
            
        arr = np.array(diffs_list)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        filtered = arr[(arr >= lower_bound) & (arr <= upper_bound)]
        valid_count = len(filtered)
        if valid_count == 0:
            return 0, 0, 0.0, 0.0
            
        wins = np.sum(np.abs(filtered) <= tol)
        mae = np.mean(np.abs(filtered))
        pct = (wins / valid_count) * 100
        
        return int(wins), int(valid_count), float(mae), float(pct)

    # Filtered Metrics
    pts_wins_f, pts_count_f, pts_mae_f, pts_pct_f = get_filtered_metrics([r['diff_pts'] for r in results], 4)
    reb_wins_f, reb_count_f, reb_mae_f, reb_pct_f = get_filtered_metrics([r['diff_reb'] for r in results], 2)
    ast_wins_f, ast_count_f, ast_mae_f, ast_pct_f = get_filtered_metrics([r['diff_ast'] for r in results], 2)
        
    # Sort by PTS diff magnitude
    results.sort(key=lambda x: abs(x['diff_pts']), reverse=True)
    
    txt_out = os.path.join(output_dir, f'{base_name}.txt')
    mode = 'a'
    with open(txt_out, mode, encoding='utf-8') as f:
        f.write("\n" + "="*60 + "\n")
        f.write(" 🎯 MEDIAN PROJECTION ACCURACY (Outliers Filtered IQR) 🎯\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total Players Evaluated: {count}\n")
        f.write(f"MAE (Filtered) - PTS: {pts_mae_f:.1f} | REB: {reb_mae_f:.1f} | AST: {ast_mae_f:.1f}\n")
        f.write(f"Direction Accuracy (Filtered) - PTS: {pts_wins_f}/{pts_count_f} ({pts_pct_f:.1f}%) | REB: {reb_wins_f}/{reb_count_f} ({reb_pct_f:.1f}%) | AST: {ast_wins_f}/{ast_count_f} ({ast_pct_f:.1f}%)\n\n")
        
        f.write(f"{'PLAYER':<22} | {'PRED':>5} {'ACT':>4} {'D':>4} {'R':>5} | {'PRED':>5} {'ACT':>4} {'D':>4} {'R':>5} | {'PRED':>5} {'ACT':>4} {'D':>4} {'R':>5}\n")
        f.write(f"{'':22}   {'--- PTS ---':^21}   {'--- REB ---':^21}   {'--- AST ---':^21}\n")
        f.write("-" * 100 + "\n")
        
        for r in results:
            f.write(f"{r['player']:<22} | {r['pred_pts']:>5} {r['act_pts']:>4} {r['diff_pts']:>+4} {r['pts_win']:>6} | {r['pred_reb']:>5} {r['act_reb']:>4} {r['diff_reb']:>+4} {r['reb_win']:>6} | {r['pred_ast']:>5} {r['act_ast']:>4} {r['diff_ast']:>+4} {r['ast_win']:>6}\n")
            
    print(f"[+] Text report appended at: {txt_out}")

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze prediction accuracy')
    parser.add_argument('--input', type=str, help='Path to predicts file (default: predicts.txt)')
    parser.add_argument('--actuals', type=str, help='Path to actual results file (default: yesterday_results.txt)')
    parser.add_argument('--output-dir', type=str, help='Directory to output analyze.txt and analyze.tex')
    parser.add_argument('--write-db', action='store_true', help='Write evaluations to bet_tracker.db')
    args = parser.parse_args()
    generate_analysis(predicts_path=args.input, results_path=args.actuals, output_dir=args.output_dir, write_db=args.write_db)
