import re
import os
from collections import defaultdict

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTS_PATH = os.path.join(MODEL_DIR, 'predicts.txt')
RESULTS_PATH = os.path.join(MODEL_DIR, 'yesterday_results.txt')

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

def generate_analysis(predicts_path=None):
    predicts_path = predicts_path or PREDICTS_PATH
    print(f"[+] Running automated prediction analysis on {os.path.basename(predicts_path)}...")
    games, props, medians = parse_predicts(predicts_path)
    actual_games, actual_props = parse_results(RESULTS_PATH)
    
    if not props and not medians:
        print(f"[-] No valuable props or medians found in {os.path.basename(predicts_path)} to analyze.")
        return
        
    if not actual_props:
        print("[-] No actual player results found in yesterday_results.txt")
        return

    # Initialize a fresh analyze.tex file empty
    tex_out = os.path.join(MODEL_DIR, 'analyze.tex')
    open(tex_out, 'w').close()
    
    # Run props analysis (VALUABLE bets WIN/LOSS) whenever prop data exists
    if props:
        print(f"    -> Running Pick Analysis ({len(props)} valuable props).")
        analyze_props(props, games, actual_games, actual_props, is_only_props=(not medians))
    # Always run median accuracy analysis if medians exist (runs alongside props if both available)
    if medians:
        print(f"    -> Running Median Accuracy Analysis ({len(medians)} player projections).")
        analyze_medians(medians, actual_props, is_only_medians=(not props))

def analyze_props(props, games, actual_games, actual_props, is_only_props=True):
    results = []
    wins = 0
    losses = 0
    
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
    
    # Text Generation
    txt_out = os.path.join(MODEL_DIR, 'analyze.txt')
    with open(txt_out, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(" 🎯 AUTOMATED PREDICTION ANALYSIS 🎯\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total Props Evaluated: {total}\n")
        f.write(f"Wins: {wins} | Losses: {losses} | Win Rate: {win_rate:.1f}%\n\n")
        
        f.write(f"{'PLAYER':<22} | {'CONF':<4} | {'BET':<15} | {'ACTUAL':<8} | {'DIFF':<7} | {'DIFF %':<8} | {'STATUS'}\n")
        f.write("-" * 88 + "\n")
        
        for r in results:
            bet_str = f"{r['direction']} {r['line']} {r['stat'][:3]}"
            diff_str = f"{r['diff_val']:+.1f}"
            diff_pct_str = f"{r['diff_pct']:+.1f}%"
            conf_str = f"{r['conf']}%" if r['conf'] else "N/A"
            f.write(f"{r['player']:<22} | {conf_str:<4} | {bet_str:<15} | {r['actual']:<8} | {diff_str:<7} | {diff_pct_str:<8} | {r['status']}\n")
            
    print(f"[+] Text report generated at: {txt_out}")

    # TeX Generation
    tex_out = os.path.join(MODEL_DIR, 'analyze.tex')
    with open(tex_out, 'a', encoding='utf-8') as f:
        f.write(r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{geometry}
\usepackage{xcolor}
\usepackage{amssymb}
\geometry{a4paper, margin=1in}
\title{Automated Prediction Analysis}
\author{AI Bet Tool}
\date{\today}
\begin{document}
\maketitle

\section*{Pick Analysis Summary}
\textbf{Total Props Evaluated:} """ + str(total) + r""" \\
\textbf{Wins:} """ + str(wins) + r""" \quad \textbf{Losses:} """ + str(losses) + r""" \quad \textbf{Win Rate:} """ + f"{win_rate:.1f}\\%" + r"""

\section*{Valuable Picks Breakdown}
\begin{table}[h]
\centering
\begin{tabular}{llccccc}
\toprule
\textbf{Player} & \textbf{Conf} & \textbf{Bet} & \textbf{Actual} & \textbf{Diff} & \textbf{Diff \%} & \textbf{Status} \\
\midrule
""")
        for r in results:
            bet_str = f"{r['direction']} {r['line']} {r['stat'][:3]}"
            diff_str = f"{r['diff_val']:+.1f}"
            diff_pct_str = f"{r['diff_pct']:+.1f}\\%"
            status_color = r"\textcolor{green!70!black}{WON $\checkmark$}" if "WON" in r['status'] else r"\textcolor{red}{LOST $\times$}"
            conf_str = f"{r['conf']}\\%" if r['conf'] else "N/A"
            
            # Escape underscores in player names for LaTeX
            safe_player = r['player'].replace('_', r'\_')
            f.write(f"{safe_player} & {conf_str} & {bet_str} & {r['actual']} & {diff_str} & {diff_pct_str} & {status_color} \\\\\n")
            
        f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
        if is_only_props:
            f.write(r"\end{document}" + "\n")
            
    print(f"[+] TeX template updated at: {tex_out}")

def analyze_medians(medians, actual_props, is_only_medians=True):
    results = []
    
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
        
        pts_err += abs(pts_diff)
        reb_err += abs(reb_diff)
        ast_err += abs(ast_diff)
        count += 1

        # Direction accuracy: did the model project above or below actual?
        # WIN = model over-projected AND actual > median (should have bet OVER)
        # OR model under-projected AND actual < median (should have bet UNDER)
        # We define WIN as: model's projected median was in the correct direction vs midpoint
        # Simpler: if model projected 29 and actual is 31, model was LOW → OVER was right
        #          if model projected 29 and actual is 24, model was HIGH → UNDER was right
        # We flag WIN if |diff| <= 5 (acceptable) OR model direction matches a 3pt+ margin
        pts_win = 'WIN ✅' if abs(pts_diff) <= 4 else ('LOW ↑' if pts_diff < 0 else 'HIGH ↓')
        reb_win = 'WIN ✅' if abs(reb_diff) <= 2 else ('LOW ↑' if reb_diff < 0 else 'HIGH ↓')
        ast_win = 'WIN ✅' if abs(ast_diff) <= 2 else ('LOW ↑' if ast_diff < 0 else 'HIGH ↓')

        if 'WIN' in pts_win: pts_wins += 1
        if 'WIN' in reb_win: reb_wins += 1
        if 'WIN' in ast_win: ast_wins += 1
        
        results.append({
            'player': player,
            'pred_pts': m['PTS'], 'act_pts': actual['POINTS'], 'diff_pts': pts_diff, 'pts_win': pts_win,
            'pred_reb': m['REB'], 'act_reb': actual['REBOUNDS'], 'diff_reb': reb_diff, 'reb_win': reb_win,
            'pred_ast': m['AST'], 'act_ast': actual['ASSISTS'], 'diff_ast': ast_diff, 'ast_win': ast_win,
        })
        
    if count == 0:
        print("[-] No matching players found between predictions and actual results.")
        return
        
    # Sort by PTS diff magnitude
    results.sort(key=lambda x: abs(x['diff_pts']), reverse=True)
    
    txt_out = os.path.join(MODEL_DIR, 'analyze.txt')
    mode = 'a' if os.path.exists(txt_out) else 'w'
    with open(txt_out, mode, encoding='utf-8') as f:
        f.write("\n" + "="*60 + "\n")
        f.write(" 🎯 MEDIAN PROJECTION ACCURACY 🎯\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Players Evaluated: {count}\n")
        f.write(f"MAE - PTS: {pts_err/count:.1f} | REB: {reb_err/count:.1f} | AST: {ast_err/count:.1f}\n")
        f.write(f"Direction Accuracy (within tolerance) - PTS: {pts_wins}/{count} | REB: {reb_wins}/{count} | AST: {ast_wins}/{count}\n\n")
        
        f.write(f"{'PLAYER':<22} | {'PRED':>5} {'ACT':>4} {'D':>4} {'R':>5} | {'PRED':>5} {'ACT':>4} {'D':>4} {'R':>5} | {'PRED':>5} {'ACT':>4} {'D':>4} {'R':>5}\n")
        f.write(f"{'':22}   {'--- PTS ---':^21}   {'--- REB ---':^21}   {'--- AST ---':^21}\n")
        f.write("-" * 100 + "\n")
        
        for r in results:
            f.write(f"{r['player']:<22} | {r['pred_pts']:>5} {r['act_pts']:>4} {r['diff_pts']:>+4} {r['pts_win']:>6} | {r['pred_reb']:>5} {r['act_reb']:>4} {r['diff_reb']:>+4} {r['reb_win']:>6} | {r['pred_ast']:>5} {r['act_ast']:>4} {r['diff_ast']:>+4} {r['ast_win']:>6}\n")
            
    print(f"[+] Text report appended at: {txt_out}")

    tex_out = os.path.join(MODEL_DIR, 'analyze.tex')
    with open(tex_out, 'a', encoding='utf-8') as f:
        if is_only_medians:
            f.write(r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{geometry}
\usepackage{xcolor}
\usepackage{amssymb}
\geometry{a4paper, margin=1in}
\title{Historical ACCURACY - Medians}
\author{AI Bet Tool}
\date{\today}
\begin{document}
\maketitle
""")

        f.write(r"""
\section*{Median Projection Accuracy Summary}
\textbf{Players Evaluated:} """ + str(count) + r""" \\
\textbf{MAE - PTS:} """ + f"{pts_err/count:.1f}" + r""" \quad \textbf{REB:} """ + f"{reb_err/count:.1f}" + r""" \quad \textbf{AST:} """ + f"{ast_err/count:.1f}" + r""" \\
\textbf{Direction Accuracy (within tol) - PTS:} """ + f"{pts_wins}/{count}" + r""" \quad \textbf{REB:} """ + f"{reb_wins}/{count}" + r""" \quad \textbf{AST:} """ + f"{ast_wins}/{count}" + r"""

\section*{Median Largest Deviations}
\begin{table}[h]
\centering
\begin{tabular}{llclclc}
\toprule
\textbf{Player} & \textbf{PTS (P/A/D)} & \textbf{W/L} & \textbf{REB (P/A/D)} & \textbf{W/L} & \textbf{AST (P/A/D)} & \textbf{W/L} \\
\midrule
""")
        for i, r in enumerate(results[:35]): # top 35 deviations
            def fmt_res(kind):
                pred, act, diff, win = r[f'pred_{kind}'], r[f'act_{kind}'], r[f'diff_{kind}'], r[f'{kind}_win']
                res_str = f"{pred}/{act}/{diff:+}"
                if 'WIN' in win:
                    win_tex = r"\textcolor{green!70!black}{WIN $\checkmark$}"
                elif 'HIGH' in win:
                    win_tex = r"\textcolor{red}{HIGH $\downarrow$}"
                else:
                    win_tex = r"\textcolor{blue}{LOW $\uparrow$}"
                return res_str, win_tex

            pts_str, pts_win = fmt_res('pts')
            reb_str, reb_win = fmt_res('reb')
            ast_str, ast_win = fmt_res('ast')
            
            safe_player = r['player'].replace('_', r'\_')
            f.write(f"{safe_player} & {pts_str} & {pts_win} & {reb_str} & {reb_win} & {ast_str} & {ast_win} \\\\\n")
            
        f.write(r"""\bottomrule
\end{tabular}
\end{table}
\end{document}
""")
    print(f"[+] TeX template updated at: {tex_out}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze prediction accuracy')
    parser.add_argument('--input', type=str, help='Path to predicts file (default: predicts.txt)')
    args = parser.parse_args()
    generate_analysis(predicts_path=args.input)
