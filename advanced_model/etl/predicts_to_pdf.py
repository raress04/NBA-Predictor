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
from datetime import date, datetime
from typing import List, Dict, Optional


def _slate_date() -> str:
    """US-Eastern slate date (utils.slate_date); this file also runs as a plain script."""
    try:
        from utils.slate_date import slate_date
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.slate_date import slate_date
    return slate_date()

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
        ('≥', r'$\ge$'),
        ('≤', r'$\le$'),
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
    if conf >= 80:
        return r'\cellcolor{highconf}'
    elif conf >= 75:
        return r'\cellcolor{medconf}'
    else:
        return r'\cellcolor{lowconf}'


# ── Parser ────────────────────────────────────────────────────────────────────

class PredictsParser:
    def __init__(self, text: str, game_date: Optional[str] = None, generated_at: Optional[str] = None):
        self.text = text
        self.games: List[Dict] = []
        self.parlays: List[Dict] = []
        self.honorable_mentions: List[Dict] = []

        # Parse generation timestamp (day and time when report is generated)
        gen_dt = None
        if generated_at:
            try:
                gen_dt = datetime.strptime(generated_at.strip(), '%Y-%m-%d %H:%M:%S')
            except Exception:
                try:
                    gen_dt = datetime.fromisoformat(generated_at.strip())
                except Exception:
                    pass
        if not gen_dt:
            gen_dt = datetime.now()
        self.generated_at = gen_dt.strftime('%B %d, %Y at %H:%M')

        parsed_dt = None
        if game_date:
            try:
                parsed_dt = datetime.strptime(game_date.strip(), '%Y-%m-%d')
            except Exception:
                pass

        if not parsed_dt:
            # Check for patterns like "Generating projections for 2026-10-20" or "predicts_2026-10-20"
            m = re.search(r'projections\s+for\s+(\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
            if not m:
                m = re.search(r'predicts_(\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
            if not m:
                m = re.search(r'(\d{4}-\d{2}-\d{2})', text)
            if m:
                try:
                    parsed_dt = datetime.strptime(m.group(1), '%Y-%m-%d')
                except Exception:
                    pass

        if parsed_dt:
            self.run_date = parsed_dt.strftime('%B %d, %Y')
        else:
            self.run_date = datetime.strptime(_slate_date(), '%Y-%m-%d').strftime('%B %d, %Y')

        self._parse()

    def _parse_game_block(self, block: str, game: dict):
        import re as _re
        for pat, key in [
            (r'H2H:\s+(.+)', 'odds_h2h'),
            (r'Spread:\s+(.+)', 'odds_spread'),
            (r'Total:\s+(.+)', 'odds_total'),
        ]:
            m = _re.search(pat, block)
            if m:
                game[key] = m.group(1).strip()

        for tcode, off, dfn in _re.findall(r'\[TEAM RATINGS\] (\w+): OffRtg ([\d.]+) \| DefRtg ([\d.]+)', block):
            if not game.get('_rtg1'):
                game['_rtg1'] = (tcode, off, dfn)
            else:
                game['_rtg2'] = (tcode, off, dfn)

        m = _re.search(r'Calculated Game Pace: ~(\d+)', block)
        if m:
            game['pace'] = m.group(1)

        m = _re.search(r'PROJECTED TOTAL: ([\d.]+)', block)
        if m:
            game['proj_total'] = m.group(1)

        m = _re.search(r'PROJECTED SCORE: (\w+ [\d.]+ - \w+ [\d.]+)', block)
        if m:
            game['proj_score'] = m.group(1)

        m = _re.search(r'-> GAME TOTAL: (.+)', block)
        if m:
            game['game_total'] = m.group(1).strip()

        spread_evals = _re.findall(r'-> (?:HOME|AWAY) SPREAD: (.+)', block)
        valuable = [s for s in spread_evals if 'VALUABLE' in s]
        if valuable:
            game['spread'] = valuable[0]
        elif spread_evals:
            game['spread'] = spread_evals[0]

        home_sec = _re.search(r'\[HOME TEAM\](.*?)(?:\[AWAY TEAM\]|$)', block, _re.DOTALL)
        away_sec = _re.search(r'\[AWAY TEAM\](.*?)$', block, _re.DOTALL)

        for section, key in [(home_sec, 'home_players'), (away_sec, 'away_players')]:
            if not section:
                continue
            player_lines = _re.findall(
                r"^\s{2}([A-Z][A-Za-z'.\-][\w\s'.\-]+?)\s+->\s+(\d+ PTS, \d+ REB, \d+ AST)(.*?)$",
                section.group(1),
                _re.MULTILINE,
            )
            for pname, stats, props_raw in player_lines:
                is_uncertain = '[HIGH_UNCERTAINTY]' in props_raw
                player = {
                    'name': pname.strip(),
                    'stats': stats.strip(),
                    'props': [],
                    'high_uncertainty': is_uncertain
                }
                prop_matches = _re.findall(
                    r'O/U ([\d.]+) (\w+) (VALUABLE: (\w+) \((\d+)% conf(?:\s*\(raw:\s*[\d.]+\%\))?, ([\d.]+)% edge\)|ignore)',
                    props_raw,
                )
                for line_val, stat_name, full, direction, conf, edge in prop_matches:
                    if 'VALUABLE' in full and direction:
                        player['props'].append({
                            'line': line_val,
                            'stat': stat_name,
                            'direction': direction,
                            'conf': int(conf),
                            'edge': edge,
                        })
                game[key].append(player)

    def _parse(self):
        import re as _re
        # The txt wraps each MATCHUP header between its own pair of ====... separator lines.
        # Splitting yields: [junk, MATCHUP-only, data-block, MATCHUP-only, data-block, ...]
        # When we see a MATCHUP block we create the game record; the following non-empty,
        # non-MATCHUP block contains all the player data.
        raw_blocks = _re.split(r'={50,}', self.text)
        current_game = None

        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue

            matchup_m = _re.search(r'MATCHUP:\s+(.+)', block)
            if matchup_m:
                current_game = {
                    'matchup': matchup_m.group(1).strip(),
                    'away': '', 'home': '',
                    'odds_h2h': '', 'odds_spread': '', 'odds_total': '',
                    'proj_total': '', 'proj_score': '',
                    'game_total': '', 'spread': '',
                    'home_players': [], 'away_players': [],
                    'off_rtg_home': '', 'def_rtg_home': '',
                    'off_rtg_away': '', 'def_rtg_away': '',
                    'pace': '',
                }
                parts = matchup_m.group(1).strip().split(' @ ')
                if len(parts) == 2:
                    current_game['away'] = parts[0].strip()
                    current_game['home'] = parts[1].strip()
                # Parse anything that may already be in this same block
                self._parse_game_block(block, current_game)
                self.games.append(current_game)
            elif current_game is not None:
                # Data block immediately following a MATCHUP header block
                self._parse_game_block(block, current_game)
                current_game = None  # consumed

        self._parse_parlays()

    def _parse_parlays(self):
        text = self.text
        pick_re = re.compile(
            r'Pick\s+\d+:\s+\[(.+?)\]\s+(.+?)\s+@\s+([\d.]+)\s*\n'
            r'\s+Edge:\s+\+?([\d.]+)%\s+\|.*?Confidence:\s+(\d+)%(?:\s*\(raw:\s*[\d.]+\%\))?\s+\|.*?Book:\s+(.+)',
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
    w(r'{\small\textcolor{midgray}{Generated on ' + parsed.generated_at + r' by AI Bet Tool — Markov Monte Carlo Engine}}')
    w(r'\end{center}')
    w(r'\vspace{4pt}')
    w(r'\noindent\textcolor{accent}{\rule{\linewidth}{1.2pt}}')
    w(r'\vspace{6pt}')
    w()

    has_frontpage_content = False

    # ──     # -- Top 10 by Confidence + Top 10 by Edge --
    # Collect all evaluated picks from parlays + honorable mentions + inline props
    _all_picks = []
    for _tag, _p in parsed.parlays:
        _all_picks.append(dict(_p))
    for _p in parsed.honorable_mentions:
        _all_picks.append(dict(_p))
    for _g in parsed.games:
        for _pl in _g.get('home_players', []) + _g.get('away_players', []):
            for _pr in _pl.get('props', []):
                _all_picks.append({
                    'game': _g['matchup'],
                    'pick': tex_escape(f"{_pl['name']} {_pr['direction']} {_pr['line']} {_pr['stat']}"),
                    'conf': _pr['conf'],
                    'edge': _pr['edge'],
                    'odds': '',
                })

    _seen = set()
    _deduped = []
    for _p in _all_picks:
        _key = (_p.get('pick', ''), _p.get('game', ''))
        if _key not in _seen:
            _seen.add(_key)
            _deduped.append(_p)

    _top_conf = sorted(_deduped, key=lambda x: int(x.get('conf', 0)), reverse=True)[:10]
    _top_edge = sorted(_deduped, key=lambda x: float(x.get('edge', 0)), reverse=True)[:10]

    def _top10_table(picks, sort_label, primary_col, secondary_col):
        if not picks:
            return
        w(r'\begin{mdframed}[linecolor=accent,linewidth=1pt,backgroundcolor=lightgray,')
        w(r'  innertopmargin=5pt,innerbottommargin=5pt,innerrightmargin=5pt,innerleftmargin=5pt]')
        w(r'{\bfseries\textcolor{accent}{Top 10 Picks \textemdash{} ' + sort_label + r'}}\\[3pt]')
        w(r'\begin{tabular}{@{}p{0.04\linewidth}p{0.44\linewidth}p{0.24\linewidth}p{0.1\linewidth}p{0.1\linewidth}@{}}')
        w(r'\toprule')
        w(r'\# & \textbf{Pick} & \textbf{Game} & \textbf{' + primary_col + r'} & \textbf{' + secondary_col + r'} \\')
        w(r'\midrule')
        for _i, _p in enumerate(picks, 1):
            _conf = int(_p.get('conf', 0))
            _edge = _p.get('edge', '?')
            _cc = conf_color(_conf)
            _pick_str = str(_p.get('pick', ''))[:55]
            _game_str = tex_escape(str(_p.get('game', '')))[:32]
            if primary_col == 'Conf':
                _pv = f'{_cc}{_conf}\\%'
                _sv = f'+{_edge}\\%'
            else:
                _pv = f'+{_edge}\\%'
                _sv = f'{_cc}{_conf}\\%'
            w(f'  {_i} & {_pick_str} & {{\\small {_game_str}}} & {_pv} & {_sv} \\\\')
        w(r'\bottomrule')
        w(r'\end{tabular}')
        w(r'\end{mdframed}')
        w(r'\vspace{4pt}')
        w()

    if _top_conf or _top_edge:
        has_frontpage_content = True
        _top10_table(_top_conf, 'Confidence', 'Conf', 'Edge')
        _top10_table(_top_edge, 'Edge', 'Edge', 'Conf')

    # ── Parlay Summary Box (top of page) ──
    p1_picks = [p for tag, p in parsed.parlays if tag == 'p1']
    p2_picks = [p for tag, p in parsed.parlays if tag == 'p2']

    if p1_picks:
        has_frontpage_content = True
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
        has_frontpage_content = True
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
        has_frontpage_content = True
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

    if has_frontpage_content:
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
                if player.get('high_uncertainty'):
                    name += r' \textcolor{accentgold}{\scriptsize\textbf{[UNCERTAIN]}}'
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
    parser.add_argument('--date', default=None, help='Game date in YYYY-MM-DD format')
    parser.add_argument('--generated-at', default=None, help='Report generation timestamp (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--compile', action='store_true', help='Compile LaTeX to PDF using pdflatex')
    parser.add_argument('--clean', action='store_true', help='Clean up auxiliary files (.aux, .log, .tex) after compiling PDF')
    args = parser.parse_args()

    input_path = args.input if os.path.isabs(args.input) else os.path.join(MODEL_DIR, args.input)
    if not os.path.isfile(input_path):
        print(f'[!] File not found: {input_path}')
        sys.exit(1)

    with open(input_path, 'r', encoding='utf-8') as f:
        raw = f.read()

    print(f'[+] Parsing {os.path.basename(input_path)}...')

    # Infer date from --date, filename, or raw content
    game_date = args.date
    if not game_date:
        m = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(input_path))
        if m:
            game_date = m.group(1)

    parsed = PredictsParser(raw, game_date=game_date, generated_at=args.generated_at)
    print(f'    Report Date: {parsed.run_date}')
    print(f'    Generated At: {parsed.generated_at}')
    print(f'    Found {len(parsed.games)} games, {len(parsed.parlays)} parlay picks, {len(parsed.honorable_mentions)} honorable mentions.')

    latex = generate_latex(parsed, args.output)

    output_tex = args.output + '.tex' if os.path.isabs(args.output) else os.path.join(MODEL_DIR, args.output + '.tex')
    with open(output_tex, 'w', encoding='utf-8') as f:
        f.write(latex)

    print(f'[+] LaTeX written to: {output_tex}')

    if args.compile:
        from etl.live_ingestion import compile_latex_to_pdf
        out_dir = os.path.dirname(os.path.abspath(output_tex))
        compile_latex_to_pdf(output_tex, out_dir)


if __name__ == '__main__':
    main()
