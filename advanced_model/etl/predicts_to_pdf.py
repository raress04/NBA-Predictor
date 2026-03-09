#!/usr/bin/env python3
"""
predicts_to_pdf.py
------------------
Parses advanced_model/predicts.txt and generates a beautiful LaTeX PDF report
summarising the day's game projections, valuable picks, player props, and parlay recommendations.

Usage:
    python3 predicts_to_pdf.py [--input predicts.txt] [--output predicts_report]
    pdflatex predicts_report.tex
"""

import re
import os
import sys
import argparse
from datetime import date
from typing import List, Dict, Optional

# Optional: bet_tracker for historical performance section
try:
    from etl.bet_tracker import query_summary, query_by_type, query_confidence_bands
    _HAS_TRACKER = True
except ImportError:
    _HAS_TRACKER = False

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Helpers ───────────────────────────────────────────────────────────────────

def tex_escape(text: str) -> str:
    """Escape special LaTeX characters in a string."""
    replacements = [
        ('\\', r'\textbackslash{}'),
        ('&', r'\&'),
        ('%', r'\%'),
        ('$', r'\$'),
        ('#', r'\#'),
        ('^', r'\^{}'),
        ('_', r'\_'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('~', r'\textasciitilde{}'),
        ('→', r'$\rightarrow$'),
        ('⚠️', r'\textbf{!}'),
        ('📊', ''),
        ('🏀', ''),
        ('🎯', ''),
        ('🔥', ''),
        ('⚡', ''),
        ('📈', ''),
        ('📉', ''),
        ('⛔', ''),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def conf_color(conf: int) -> str:
    """Return a LaTeX color command based on confidence level."""
    if conf >= 90:
        return r'\cellcolor{highconf}'
    elif conf >= 80:
        return r'\cellcolor{medconf}'
    else:
        return r'\cellcolor{lowconf}'


# ── Parser ────────────────────────────────────────────────────────────────────

class PredictsParser:
    def __init__(self, text: str):
        self.text = text
        self.games: List[Dict] = []
        self.parlays: List[Dict] = []
        self.honorable_mentions: List[Dict] = []
        self.run_date = date.today().strftime('%B %d, %Y')
        self._parse()

    def _parse(self):
        lines = self.text.splitlines()

        # ── Game blocks ──
        game_blocks = re.split(r'={50,}', self.text)
        current_game = None

        for block in game_blocks:
            block = block.strip()
            if not block:
                continue

            # Detect matchup header
            matchup_m = re.search(r'MATCHUP:\s+(.+)', block)
            if matchup_m:
                current_game = {
                    'matchup': matchup_m.group(1).strip(),
                    'away': '',
                    'home': '',
                    'odds_h2h': '',
                    'odds_spread': '',
                    'odds_total': '',
                    'proj_total': '',
                    'proj_score': '',
                    'game_total': '',
                    'spread': '',
                    'home_players': [],
                    'away_players': [],
                    'off_rtg_home': '',
                    'def_rtg_home': '',
                    'off_rtg_away': '',
                    'def_rtg_away': '',
                    'pace': '',
                }
                parts = matchup_m.group(1).strip().split(' @ ')
                if len(parts) == 2:
                    current_game['away'] = parts[0].strip()
                    current_game['home'] = parts[1].strip()

                # Sportsbook odds
                h2h_m = re.search(r'H2H:\s+(.+)', block)
                if h2h_m:
                    current_game['odds_h2h'] = h2h_m.group(1).strip()

                spread_m = re.search(r'Spread:\s+(.+)', block)
                if spread_m:
                    current_game['odds_spread'] = spread_m.group(1).strip()

                total_m = re.search(r'Total:\s+(.+)', block)
                if total_m:
                    current_game['odds_total'] = total_m.group(1).strip()

                # Team ratings
                home_rtg_m = re.search(r'\[TEAM RATINGS\] (\w+): OffRtg ([\d.]+) \| DefRtg ([\d.]+)', block)
                if home_rtg_m:
                    current_game['off_rtg_home'] = home_rtg_m.group(2)
                    current_game['def_rtg_home'] = home_rtg_m.group(3)
                    # try to get both
                matches = re.findall(r'\[TEAM RATINGS\] (\w+): OffRtg ([\d.]+) \| DefRtg ([\d.]+)', block)
                for m in matches:
                    tcode, off, dfn = m
                    # Associate by home/away tricode order
                    if not current_game.get('_rtg1'):
                        current_game['_rtg1'] = (tcode, off, dfn)
                    else:
                        current_game['_rtg2'] = (tcode, off, dfn)

                # Pace
                pace_m = re.search(r'Calculated Game Pace: ~(\d+)', block)
                if pace_m:
                    current_game['pace'] = pace_m.group(1)

                # Projected totals
                pt_m = re.search(r'PROJECTED TOTAL: ([\d.]+)', block)
                if pt_m:
                    current_game['proj_total'] = pt_m.group(1)

                ps_m = re.search(r'PROJECTED SCORE: (\w+ [\d.]+ - \w+ [\d.]+)', block)
                if ps_m:
                    current_game['proj_score'] = ps_m.group(1)

                # Game total evaluation
                gt_m = re.search(r'-> GAME TOTAL: (.+)', block)
                if gt_m:
                    current_game['game_total'] = gt_m.group(1).strip()

                # Spread evaluation — pick the VALUABLE one if any
                spread_evals = re.findall(r'-> (?:HOME|AWAY) SPREAD: (.+)', block)
                valuable_spreads = [s for s in spread_evals if 'VALUABLE' in s]
                if valuable_spreads:
                    current_game['spread'] = valuable_spreads[0]
                elif spread_evals:
                    current_game['spread'] = spread_evals[0]

                # Player props
                home_section = re.search(r'\[HOME TEAM\](.*?)(?:\[AWAY TEAM\]|$)', block, re.DOTALL)
                away_section = re.search(r'\[AWAY TEAM\](.*?)$', block, re.DOTALL)

                for section, key in [(home_section, 'home_players'), (away_section, 'away_players')]:
                    if not section:
                        continue
                    player_lines = re.findall(
                        r'(\w[\w\s\'.]+?)\s+->\s+(\d+ PTS, \d+ REB, \d+ AST)(.*?)$',
                        section.group(1),
                        re.MULTILINE
                    )
                    for pname, stats, props_raw in player_lines:
                        player = {
                            'name': pname.strip(),
                            'stats': stats.strip(),
                            'props': [],
                        }
                        # Extract prop evaluations
                        prop_matches = re.findall(
                            r'O/U ([\d.]+) (\w+) (VALUABLE: (\w+) \((\d+)% conf, ([\d.]+)% edge\)|ignore)',
                            props_raw
                        )
                        for pm in prop_matches:
                            line_val, stat_name, full, direction, conf, edge = pm
                            if 'VALUABLE' in full and direction:
                                player['props'].append({
                                    'line': line_val,
                                    'stat': stat_name,
                                    'direction': direction,
                                    'conf': int(conf),
                                    'edge': edge,
                                })
                        current_game[key].append(player)

                self.games.append(current_game)
                continue

            # ── Parlays & Honorable Mentions block ──
            # (Already handled after main loop below)
            continue

        # ── Parse Parlays and Honorable Mentions from full text ──
        self._parse_parlays()

    def _parse_parlays(self):
        text = self.text
        pick_re = re.compile(
            r'Pick\s+\d+:\s+\[(.+?)\]\s+(.+?)\s+@\s+([\d.]+)\s*\n'
            r'\s+Edge:\s+\+?([\d.]+)%\s+\|.*?Confidence:\s+(\d+)%\s+\|.*?Book:\s+(.+)',
            re.MULTILINE
        )

        p1_start = text.find('PARLAY RECOMMENDATION #1')
        p2_start = text.find('PARLAY RECOMMENDATION #2')
        hm_start = text.find('HONORABLE MENTIONS')

        for m in pick_re.finditer(text):
            pos = m.start()
            game, pick_desc, odds, edge, conf, book = m.groups()
            pick = {
                'game': game.strip(),
                'pick': tex_escape(pick_desc.strip()),
                'odds': odds,
                'edge': edge,
                'conf': int(conf),
                'book': book.strip(),
            }
            if p1_start != -1 and p2_start != -1:
                if p1_start < pos < p2_start:
                    self.parlays.append(('p1', pick))
                elif pos > p2_start and (hm_start == -1 or pos < hm_start):
                    self.parlays.append(('p2', pick))
            elif p1_start != -1 and pos > p1_start:
                self.parlays.append(('p1', pick))

        # Honorable Mentions
        if hm_start != -1:
            hm_text = text[hm_start:]
            hm_re = re.compile(
                r'\[(.+?)\]\s+(.+?)\s+@\s+([\d.]+)\s*\n'
                r'\s+Edge:\s+\+?([\d.]+)%\s+\|.*?Conf:\s+(\d+)%',
                re.MULTILINE
            )
            for m in hm_re.finditer(hm_text):
                game, pick_desc, odds, edge, conf = m.groups()
                self.honorable_mentions.append({
                    'game': game.strip(),
                    'pick': tex_escape(pick_desc.strip()),
                    'odds': odds,
                    'edge': edge,
                    'conf': int(conf),
                })


# ── LaTeX Generator ───────────────────────────────────────────────────────────

def generate_latex(parsed: PredictsParser, output_stem: str):
    lines = []

    def w(s=''):
        lines.append(s)

    # ── Preamble ──
    w(r'\documentclass[10pt,a4paper]{article}')
    w(r'\usepackage[utf8]{inputenc}')
    w(r'\usepackage[T1]{fontenc}')
    w(r'\usepackage[margin=1.5cm]{geometry}')
    w(r'\usepackage{booktabs}')
    w(r'\usepackage{xcolor}')
    w(r'\usepackage{colortbl}')
    w(r'\usepackage{tabularx}')
    w(r'\usepackage{array}')
    w(r'\usepackage{amssymb}')
    w(r'\usepackage{amsmath}')
    w(r'\usepackage{titlesec}')
    w(r'\usepackage{fancyhdr}')
    w(r'\usepackage{mdframed}')
    w(r'\usepackage{multicol}')
    w(r'\usepackage{microtype}')
    w(r'\usepackage{lmodern}')
    w(r'\usepackage{parskip}')
    w()
    # Color palette
    w(r'% ── Color Palette ──')
    w(r'\definecolor{darkbg}{HTML}{0D1117}')
    w(r'\definecolor{accent}{HTML}{58A6FF}')
    w(r'\definecolor{accentgold}{HTML}{E3B341}')
    w(r'\definecolor{lightgray}{HTML}{F6F8FA}')
    w(r'\definecolor{midgray}{HTML}{CCCCCC}')
    w(r'\definecolor{tablehead}{HTML}{21262D}')
    w(r'\definecolor{highconf}{HTML}{D4EDDA}')  # green tint
    w(r'\definecolor{medconf}{HTML}{FFF3CD}')   # amber tint
    w(r'\definecolor{lowconf}{HTML}{F8D7DA}')   # red tint
    w(r'\definecolor{valuablebg}{HTML}{E8F5E9}')
    w(r'\definecolor{matchupbar}{HTML}{1F6FEB}')
    w()
    # Section formatting
    w(r'\titleformat{\section}{\large\bfseries\color{accent}}{}{0em}{}[\color{accent}\titlerule]')
    w(r'\titleformat{\subsection}{\normalsize\bfseries}{}{0em}{}')
    w()
    # Page style
    w(r'\pagestyle{fancy}')
    w(r'\fancyhf{}')
    w(r'\fancyhead[L]{\textbf{\textcolor{accent}{AI Bet Tool}} \textcolor{midgray}{| NBA Projection Report}}')
    w(r'\fancyhead[R]{\textcolor{midgray}{' + parsed.run_date + r'}}')
    w(r'\fancyfoot[C]{\textcolor{midgray}{\thepage}}')
    w(r'\renewcommand{\headrulewidth}{0.4pt}')
    w(r'\renewcommand{\headrule}{\hbox to\headwidth{\color{accent}\leaders\hrule height \headrulewidth\hfill}}')
    w()
    w(r'\setlength{\parindent}{0pt}')
    w(r'\setlength{\parskip}{4pt}')
    w()
    w(r'\begin{document}')
    w()

    # ── Title ──
    w(r'\begin{center}')
    w(r'{\Huge\bfseries\textcolor{accent}{NBA Projection Report}}\\[4pt]')
    w(r'{\large\textcolor{accentgold}{' + parsed.run_date + r'}}\\[2pt]')
    w(r'{\small\textcolor{midgray}{Generated by AI Bet Tool — Markov Monte Carlo Engine}}')
    w(r'\end{center}')
    w(r'\vspace{4pt}')
    w(r'\noindent\textcolor{accent}{\rule{\linewidth}{1.2pt}}')
    w(r'\vspace{6pt}')
    w()

    # ── Historical Performance Section (from DB) ──
    if _HAS_TRACKER:
        overall = query_summary()
        total_picks = int(overall.get('total') or 0)
        if total_picks >= 10:
            wins = int(overall.get('wins') or 0)
            win_rate = overall.get('win_rate') or 0.0
            avg_conf = overall.get('avg_conf') or 0.0
            by_type = query_by_type()
            bands    = query_confidence_bands()

            w(r'\begin{mdframed}[linecolor=accent,linewidth=1pt,backgroundcolor=lightgray,')
            w(r'  innertopmargin=6pt,innerbottommargin=6pt,innerrightmargin=5pt,innerleftmargin=5pt]')
            w(r'{\bfseries\textcolor{accent}{Historical Performance (All Time)}}\\[3pt]')
            w(r'\begin{tabular}{@{}p{0.48\linewidth}p{0.48\linewidth}@{}}')
            # Left column: overall + by type
            left_lines = []
            left_lines.append(f'\\textbf{{All Picks:}} {wins}/{total_picks} ({win_rate:.1f}\\%) '
                              f'\\textcolor{{midgray}}{{— Avg Conf declared: {avg_conf:.0f}\\%}}\\\\')
            left_lines.append(r'\vspace{2pt}')
            left_lines.append(r'\textbf{By Category:}\\')
            for row in by_type:
                bar_len = max(0, int(row['win_rate'] / 10))
                bar = r'\textcolor{green!60!black}{' + ('$\\blacksquare$' * bar_len) + '}'
                left_lines.append(
                    f"  {row['stat_category']:<12} "
                    f"{int(row['wins'])}/{int(row['total'])} "
                    f"({row['win_rate']:.1f}\\%) {bar}\\\\"
                )
            w('\n'.join(left_lines))
            w(r'&')
            # Right column: confidence bands
            right_lines = [r'\textbf{Confidence Band Accuracy:}\\']
            for b in bands:
                bar_len = max(0, int(b['win_rate'] / 10))
                bar = r'\textcolor{green!60!black}{' + ('$\\blacksquare$' * bar_len) + '}'
                right_lines.append(
                    f"  {b['band']:<8} "
                    f"{int(b['wins'])}/{int(b['total'])} "
                    f"({b['win_rate']:.1f}\\%) {bar}\\\\"
                )
            w('\n'.join(right_lines))
            w(r'\end{tabular}')
            w(r'\end{mdframed}')
            w(r'\vspace{6pt}')
            w()

    # ── Parlay Summary Box (top of page) ──
    p1_picks = [p for tag, p in parsed.parlays if tag == 'p1']
    p2_picks = [p for tag, p in parsed.parlays if tag == 'p2']

    if p1_picks:
        w(r'\begin{mdframed}[linecolor=accentgold,linewidth=1.5pt,backgroundcolor=lightgray,')
        w(r'  innertopmargin=8pt,innerbottommargin=8pt,innerrightmargin=8pt,innerleftmargin=8pt]')
        w(r'{\large\bfseries\textcolor{accentgold}{$\bigstar$ Parlay Recommendation \#1 — Best 2-3 Picks}}\\[4pt]')
        w(r'\begin{tabular}{@{}p{0.05\linewidth}p{0.35\linewidth}p{0.32\linewidth}p{0.06\linewidth}p{0.07\linewidth}p{0.06\linewidth}@{}}')
        w(r'\toprule')
        w(r'\textbf{\#} & \textbf{Pick} & \textbf{Game} & \textbf{Odds} & \textbf{Edge} & \textbf{Conf} \\')
        w(r'\midrule')
        for i, p in enumerate(p1_picks, 1):
            cc = conf_color(p['conf'])
            w(f"  {i} & {p['pick']} & \\small{{{tex_escape(p['game'])}}} & {p['odds']}x & +{p['edge']}\\% & {cc}{p['conf']}\\% \\\\")
        w(r'\bottomrule')
        w(r'\end{tabular}')

        if p1_picks:
            combined = 1.0
            for p in p1_picks:
                try:
                    combined *= float(p['odds'])
                except:
                    pass
            w(f"\\\\[4pt]{{\\textbf{{Combined Odds: \\textcolor{{accentgold}}{{{combined:.2f}x}}}}}}")
        w(r'\end{mdframed}')
        w(r'\vspace{6pt}')
        w()

    # ── Parlay 2 (compact) ──
    if p2_picks:
        w(r'\begin{mdframed}[linecolor=accent,linewidth=1pt,backgroundcolor=lightgray,')
        w(r'  innertopmargin=6pt,innerbottommargin=6pt,innerrightmargin=6pt,innerleftmargin=6pt]')
        w(r'{\bfseries\textcolor{accent}{Parlay \#2 — 3/4 System (4 Picks)}}\\[3pt]')
        w(r'\begin{tabular}{@{}p{0.04\linewidth}p{0.38\linewidth}p{0.32\linewidth}p{0.06\linewidth}p{0.07\linewidth}p{0.06\linewidth}@{}}')
        w(r'\toprule')
        w(r'\# & \textbf{Pick} & \textbf{Game} & \textbf{Odds} & \textbf{Edge} & \textbf{Conf} \\')
        w(r'\midrule')
        for i, p in enumerate(p2_picks, 1):
            cc = conf_color(p['conf'])
            w(f"  {i} & \\small{{{p['pick']}}} & \\small{{{tex_escape(p['game'])}}} & {p['odds']}x & +{p['edge']}\\% & {cc}{p['conf']}\\% \\\\")
        w(r'\bottomrule')
        w(r'\end{tabular}')
        w(r'\end{mdframed}')
        w(r'\vspace{6pt}')
        w()

    # ── Honorable Mentions ──
    if parsed.honorable_mentions:
        w(r'\begin{mdframed}[linecolor=midgray,linewidth=0.7pt,innertopmargin=5pt,innerbottommargin=5pt,')
        w(r'  innerrightmargin=5pt,innerleftmargin=5pt]')
        w(r'{\bfseries Honorable Mentions}\\[2pt]')
        w(r'\begin{tabular}{@{}p{0.36\linewidth}p{0.32\linewidth}p{0.06\linewidth}p{0.08\linewidth}p{0.07\linewidth}@{}}')
        w(r'\toprule')
        w(r'\textbf{Pick} & \textbf{Game} & \textbf{Odds} & \textbf{Edge} & \textbf{Conf} \\')
        w(r'\midrule')
        for hm in parsed.honorable_mentions:
            cc = conf_color(hm['conf'])
            w(f"  \\small{{{hm['pick']}}} & \\small{{{tex_escape(hm['game'])}}} & {hm['odds']}x & +{hm['edge']}\\% & {cc}{hm['conf']}\\% \\\\")
        w(r'\bottomrule')
        w(r'\end{tabular}')
        w(r'\end{mdframed}')
        w()

    w(r'\newpage')
    w()

    # ── Per-Game Sections ──
    w(r'\section*{Game-by-Game Projections}')
    w()

    for game in parsed.games:
        matchup = tex_escape(game['matchup'])
        home = tex_escape(game['home'])
        away = tex_escape(game['away'])

        # Game header bar
        w(r'\noindent\colorbox{matchupbar}{\parbox{\dimexpr\linewidth-2\fboxsep}{%')
        w(r'  \vspace{3pt}\hspace{4pt}{\bfseries\large\textcolor{white}{' + matchup + r'}}')
        w(r'  \hfill{\small\textcolor{white}{Pace: ' + game.get('pace', '?') + r' poss/team}}\vspace{3pt}')
        w(r'}}')
        w(r'\vspace{4pt}')
        w()

        # Odds + Projection row
        w(r'\begin{tabular}{@{}p{0.32\linewidth}p{0.32\linewidth}p{0.33\linewidth}@{}}')
        w(r'\small')
        # Column 1: odds
        odds_h2h = tex_escape(game.get('odds_h2h', '—'))
        odds_spread = tex_escape(game.get('odds_spread', '—'))
        odds_total = tex_escape(game.get('odds_total', '—'))
        w(r'\textbf{Sportsbook Odds:} \\')
        if odds_h2h:
            w(r'H2H: ' + odds_h2h + r'\\')
        if odds_spread:
            w(r'Spread: ' + odds_spread + r'\\')
        if odds_total:
            w(r'Total: ' + odds_total + r'\\')
        w(r'&')
        # Column 2: projections
        proj_score = tex_escape(game.get('proj_score', ''))
        proj_total = game.get('proj_total', '')
        w(r'\textbf{Model Projection:} \\')
        w(r'Score: \textbf{' + proj_score + r'}\\')
        w(r'Total: \textbf{' + proj_total + r'} pts\\')
        w(r'&')
        # Column 3: evaluations  
        gt = game.get('game_total', '')
        sp = game.get('spread', '')
        w(r'\textbf{Model Evaluation:} \\')
        # color code VALUABLE vs ignore
        if 'VALUABLE' in gt:
            w(r'\textcolor{green!60!black}{\textbf{TOTAL: ' + tex_escape(gt) + r'}}\\')
        elif gt:
            w(r'\textcolor{gray}{Total: ignore}\\')
        if sp and 'VALUABLE' in sp:
            w(r'\textcolor{green!60!black}{\textbf{SPREAD: ' + tex_escape(sp) + r'}}\\')
        elif sp:
            w(r'\textcolor{gray}{Spread: ignore}\\')
        w(r'\\')
        w(r'\end{tabular}')
        w()

        # Player tables — both teams side by side
        for team_label, team_name, players in [
            ('Home', home, game['home_players']),
            ('Away', away, game['away_players']),
        ]:
            if not players:
                continue

            # Check if any player has VALUABLE props
            has_valuable = any(p['props'] for p in players)

            w(f"\\noindent{{\\small\\textbf{{{team_label}: {team_name}}}}}")
            w()
            w(r'\begin{tabular}{@{}p{0.25\linewidth}p{0.08\linewidth}p{0.08\linewidth}p{0.08\linewidth}p{0.47\linewidth}@{}}')
            w(r'\toprule')
            w(r'{\bfseries Player} & {\bfseries PTS} & {\bfseries REB} & {\bfseries AST} & {\bfseries Valuable Props} \\')
            w(r'\midrule')

            for player in players:
                name = tex_escape(player['name'])
                # Parse stats
                pts_m = re.search(r'(\d+) PTS', player['stats'])
                reb_m = re.search(r'(\d+) REB', player['stats'])
                ast_m = re.search(r'(\d+) AST', player['stats'])
                pts = pts_m.group(1) if pts_m else '—'
                reb = reb_m.group(1) if reb_m else '—'
                ast = ast_m.group(1) if ast_m else '—'

                # Build props string
                prop_parts = []
                for prop in player['props']:
                    cc = conf_color(prop['conf'])
                    direction_sym = r'$\downarrow$' if prop['direction'] == 'Under' else r'$\uparrow$'
                    prop_str = (
                        f"{cc}\\textbf{{{direction_sym} {prop['direction']} "
                        f"{prop['line']} {prop['stat'].upper()}}} "
                        f"({prop['conf']}\\% conf, +{prop['edge']}\\% edge)"
                    )
                    prop_parts.append(prop_str)

                props_cell = r' \newline '.join(prop_parts) if prop_parts else r'\textcolor{gray}{\small—}'

                # Highlight row if player has valuable props
                row_prefix = r'\rowcolor{valuablebg}' if player['props'] else ''
                w(f"  {row_prefix}{name} & {pts} & {reb} & {ast} & \\small{{{props_cell}}} \\\\")

            w(r'\bottomrule')
            w(r'\end{tabular}')
            w(r'\vspace{3pt}')
            w()

        w(r'\vspace{8pt}')
        w()

    w(r'\end{document}')
    w()

    return '\n'.join(lines)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Convert predicts.txt to a LaTeX PDF report')
    parser.add_argument('--input', default='predicts.txt', help='Input predicts file (default: predicts.txt)')
    parser.add_argument('--output', default='predicts_report', help='Output filename stem (without extension)')
    args = parser.parse_args()

    input_path = args.input if os.path.isabs(args.input) else os.path.join(MODEL_DIR, args.input)
    if not os.path.isfile(input_path):
        print(f'[!] File not found: {input_path}')
        sys.exit(1)

    with open(input_path, 'r', encoding='utf-8') as f:
        raw = f.read()

    print(f'[+] Parsing {os.path.basename(input_path)}...')
    parsed = PredictsParser(raw)
    print(f'    Found {len(parsed.games)} games, {len(parsed.parlays)} parlay picks, {len(parsed.honorable_mentions)} honorable mentions.')

    latex = generate_latex(parsed, args.output)

    output_tex = os.path.join(MODEL_DIR, args.output + '.tex')
    with open(output_tex, 'w', encoding='utf-8') as f:
        f.write(latex)

    print(f'[+] LaTeX written to: {output_tex}')
    print(f'    Run: pdflatex {args.output}.tex')
    print(f'    Run twice for correct page references.')


if __name__ == '__main__':
    main()
