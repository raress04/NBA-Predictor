#!/usr/bin/env python3
"""
analyze_to_pdf.py
-----------------
Parses analyze_<date>.txt files and generates a LaTeX PDF report
summarizing game score accuracy, valuable props win/loss outcomes,
and player median projection accuracy.

Usage:
    python3 -m etl.analyze_to_pdf --input advanced_model/analyze/.../analyze_2026-10-22.txt
    python3 -m etl.analyze_to_pdf --dir advanced_model/analyze/2026-2027/live_predictions/october
"""

import os
import re
import sys
import shutil
import argparse
import subprocess
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── LaTeX Helpers ─────────────────────────────────────────────────────────────

def tex_escape(text: str) -> str:
    """Escape special LaTeX characters in a string."""
    if text is None:
        return ''
    text = str(text)
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
        ('🏀', ''),
        ('🎯', ''),
        ('🔥', ''),
        ('⚡', ''),
        ('📈', ''),
        ('📉', ''),
        ('⛔', ''),
        ('✅', ''),
        ('❌', ''),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def conf_color(conf_val) -> str:
    """Return cell color command based on confidence integer."""
    try:
        val = int(str(conf_val).replace('%', '').strip())
        if val >= 80:
            return r'\cellcolor{highconf}'
        elif val >= 70:
            return r'\cellcolor{medconf}'
        else:
            return r'\cellcolor{lowconf}'
    except Exception:
        return ''


# ── Parser ────────────────────────────────────────────────────────────────────

class AnalyzeParser:
    def __init__(self, text: str, filepath: Optional[str] = None, game_date: Optional[str] = None):
        self.text = text
        self.filepath = filepath
        self.date_str = game_date
        self.formatted_date = ""

        # Section 1: Game Result Accuracy
        self.game_mae_team: Optional[float] = None
        self.game_mae_total: Optional[float] = None
        self.game_mae_spread: Optional[float] = None
        self.ml_correct: int = 0
        self.ml_total: int = 0
        self.ml_pct: float = 0.0
        self.game_results: List[Dict] = []

        # Section 2: Valuable Props
        self.props_total: int = 0
        self.props_wins: int = 0
        self.props_losses: int = 0
        self.props_win_rate: float = 0.0
        self.props_results: List[Dict] = []

        # Section 3: Median Projections
        self.medians_total: int = 0
        self.medians_mae: Dict[str, float] = {}
        self.medians_dir_acc: Dict[str, Tuple[int, int, float]] = {}
        self.medians_results: List[Dict] = []

        self._resolve_date()
        self._parse()

    def _resolve_date(self):
        if not self.date_str and self.filepath:
            m = re.search(r'analyze_(\d{4}-\d{2}-\d{2})', os.path.basename(self.filepath))
            if m:
                self.date_str = m.group(1)
        if not self.date_str:
            m = re.search(r'(\d{4}-\d{2}-\d{2})', self.text)
            if m:
                self.date_str = m.group(1)
        if not self.date_str:
            self.date_str = date.today().strftime('%Y-%m-%d')

        try:
            dt = datetime.strptime(self.date_str, '%Y-%m-%d')
            self.formatted_date = dt.strftime('%B %d, %Y')
        except Exception:
            self.formatted_date = self.date_str

    def _parse(self):
        lines = self.text.splitlines()
        i = 0
        n = len(lines)

        while i < n:
            line = lines[i].strip()

            # 1. Game result MAE & ML Accuracy
            if "MAE - Team Score:" in line:
                m_mae = re.search(r'Team Score:\s*([\d\.]+)\s*\|\s*Total:\s*([\d\.]+)\s*\|\s*Spread(?:\s*\(Margin\))?:\s*([\d\.]+)', line)
                if m_mae:
                    self.game_mae_team = float(m_mae.group(1))
                    self.game_mae_total = float(m_mae.group(2))
                    self.game_mae_spread = float(m_mae.group(3))

            if "Moneyline Accuracy:" in line:
                m_ml = re.search(r'Moneyline Accuracy:\s*(\d+)/(\d+)\s*\(([\d\.]+)%\)', line)
                if m_ml:
                    self.ml_correct = int(m_ml.group(1))
                    self.ml_total = int(m_ml.group(2))
                    self.ml_pct = float(m_ml.group(3))

            # Game table header: MATCHUP | PROJECTED | ACTUAL | STATUS
            if re.match(r'^MATCHUP\s*\|\s*PROJECTED\s*\|\s*ACTUAL\s*\|\s*STATUS', line, re.IGNORECASE):
                i += 1
                if i < n and '---' in lines[i]:
                    i += 1
                while i < n:
                    row_line = lines[i].strip()
                    if not row_line or row_line.startswith('===') or 'Total Props' in row_line or 'PLAYER' in row_line:
                        break
                    parts = [p.strip() for p in row_line.split('|')]
                    if len(parts) >= 4:
                        matchup, proj, actual, status = parts[0], parts[1], parts[2], parts[3]
                        # Avoid duplicates if section was printed multiple times in raw txt
                        if not any(g['matchup'] == matchup for g in self.game_results):
                            is_correct = '✅' in status or 'WON' in status or 'CORRECT' in status
                            self.game_results.append({
                                'matchup': matchup,
                                'projected': proj,
                                'actual': actual,
                                'status': 'CORRECT' if is_correct else 'WRONG',
                                'is_correct': is_correct
                            })
                    i += 1
                continue

            # 2. Props Summary
            if "Total Props Evaluated:" in line:
                m_p = re.search(r'Total Props Evaluated:\s*(\d+)', line)
                if m_p:
                    self.props_total = int(m_p.group(1))

            if "Wins:" in line and "Losses:" in line and "Win Rate:" in line:
                m_wl = re.search(r'Wins:\s*(\d+)\s*\|\s*Losses:\s*(\d+)\s*\|\s*Win Rate:\s*([\d\.]+)%', line)
                if m_wl:
                    self.props_wins = int(m_wl.group(1))
                    self.props_losses = int(m_wl.group(2))
                    self.props_win_rate = float(m_wl.group(3))

            # Props table header: PLAYER | CONF | BET | ACTUAL | DIFF | DIFF % | STATUS
            if re.match(r'^PLAYER\s*\|\s*CONF\s*\|\s*BET\s*\|\s*ACTUAL\s*\|\s*DIFF', line, re.IGNORECASE):
                i += 1
                if i < n and '---' in lines[i]:
                    i += 1
                while i < n:
                    row_line = lines[i].strip()
                    if not row_line or row_line.startswith('===') or 'Total Players' in row_line:
                        break
                    parts = [p.strip() for p in row_line.split('|')]
                    if len(parts) >= 7:
                        player, conf, bet, actual, diff, diff_pct, status = parts[:7]
                        is_win = 'WON' in status or '✅' in status
                        self.props_results.append({
                            'player': player,
                            'conf': conf.replace('%', '').strip(),
                            'bet': bet,
                            'actual': actual,
                            'diff': diff,
                            'diff_pct': diff_pct,
                            'status': 'WON' if is_win else 'LOST',
                            'is_win': is_win
                        })
                    i += 1
                continue

            # 3. Median Projections Summary
            if "Total Players Evaluated:" in line:
                m_tp = re.search(r'Total Players Evaluated:\s*(\d+)', line)
                if m_tp:
                    self.medians_total = int(m_tp.group(1))

            if "MAE (Filtered) - PTS:" in line:
                m_fmae = re.search(r'PTS:\s*([\d\.]+)\s*\|\s*REB:\s*([\d\.]+)\s*\|\s*AST:\s*([\d\.]+)', line)
                if m_fmae:
                    self.medians_mae = {
                        'PTS': float(m_fmae.group(1)),
                        'REB': float(m_fmae.group(2)),
                        'AST': float(m_fmae.group(3)),
                    }

            if "Direction Accuracy (Filtered) - PTS:" in line:
                m_dir = re.search(r'PTS:\s*(\d+)/(\d+)\s*\(([\d\.]+)%\)\s*\|\s*REB:\s*(\d+)/(\d+)\s*\(([\d\.]+)%\)\s*\|\s*AST:\s*(\d+)/(\d+)\s*\(([\d\.]+)%\)', line)
                if m_dir:
                    self.medians_dir_acc = {
                        'PTS': (int(m_dir.group(1)), int(m_dir.group(2)), float(m_dir.group(3))),
                        'REB': (int(m_dir.group(4)), int(m_dir.group(5)), float(m_dir.group(6))),
                        'AST': (int(m_dir.group(7)), int(m_dir.group(8)), float(m_dir.group(9))),
                    }

            # Medians table header
            if '--- PTS ---' in line and '--- REB ---' in line and '--- AST ---' in line:
                i += 1
                if i < n and '---' in lines[i]:
                    i += 1
                while i < n:
                    row_line = lines[i].strip()
                    if not row_line or row_line.startswith('==='):
                        break
                    parts = [p.strip() for p in row_line.split('|')]
                    if len(parts) >= 4:
                        pname = parts[0]
                        pts_tokens = parts[1].split()
                        reb_tokens = parts[2].split()
                        ast_tokens = parts[3].split()
                        if len(pts_tokens) >= 4 and len(reb_tokens) >= 4 and len(ast_tokens) >= 4:
                            self.medians_results.append({
                                'player': pname,
                                'pts_pred': pts_tokens[0], 'pts_act': pts_tokens[1], 'pts_diff': pts_tokens[2], 'pts_res': pts_tokens[3],
                                'reb_pred': reb_tokens[0], 'reb_act': reb_tokens[1], 'reb_diff': reb_tokens[2], 'reb_res': reb_tokens[3],
                                'ast_pred': ast_tokens[0], 'ast_act': ast_tokens[1], 'ast_diff': ast_tokens[2], 'ast_res': ast_tokens[3],
                            })
                    i += 1
                continue

            i += 1


# ── LaTeX Generator ───────────────────────────────────────────────────────────

def generate_latex(parsed: AnalyzeParser) -> str:
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
    w(r'\usepackage{longtable}')
    w(r'\usepackage{array}')
    w(r'\usepackage{amssymb}')
    w(r'\usepackage{amsmath}')
    w(r'\usepackage{pifont}')
    w(r'\usepackage{titlesec}')
    w(r'\usepackage{fancyhdr}')
    w(r'\usepackage{mdframed}')
    w(r'\usepackage{microtype}')
    w(r'\usepackage{lmodern}')
    w(r'\usepackage{parskip}')
    w()
    # ── Color Palette ──
    w(r'% Color Palette matching AI Bet Tool Standard')
    w(r'\definecolor{darkbg}{HTML}{0D1117}')
    w(r'\definecolor{accent}{HTML}{58A6FF}')
    w(r'\definecolor{accentgold}{HTML}{E3B341}')
    w(r'\definecolor{lightgray}{HTML}{F6F8FA}')
    w(r'\definecolor{midgray}{HTML}{CCCCCC}')
    w(r'\definecolor{tablehead}{HTML}{21262D}')
    w(r'\definecolor{highconf}{HTML}{D4EDDA}')
    w(r'\definecolor{medconf}{HTML}{FFF3CD}')
    w(r'\definecolor{lowconf}{HTML}{F8D7DA}')
    w(r'\definecolor{winbg}{HTML}{D4EDDA}')
    w(r'\definecolor{wintext}{HTML}{155724}')
    w(r'\definecolor{lossbg}{HTML}{F8D7DA}')
    w(r'\definecolor{losstext}{HTML}{721C24}')
    w(r'\definecolor{neutralbg}{HTML}{E9ECEF}')
    w(r'\definecolor{neutraltext}{HTML}{495057}')
    w(r'\definecolor{matchupbar}{HTML}{1F6FEB}')
    w()
    # ── Section & Page Formatting ──
    w(r'\titleformat{\section}{\large\bfseries\color{accent}}{}{0em}{}[\color{accent}\titlerule]')
    w(r'\titleformat{\subsection}{\normalsize\bfseries}{}{0em}{}')
    w()
    w(r'\pagestyle{fancy}')
    w(r'\fancyhf{}')
    w(r'\fancyhead[L]{\textbf{\textcolor{accent}{AI Bet Tool}} \textcolor{midgray}{| NBA Outcome Analysis}}')
    w(r'\fancyhead[R]{\textcolor{midgray}{' + parsed.formatted_date + r'}}')
    w(r'\fancyfoot[C]{\textcolor{midgray}{\thepage}}')
    w(r'\renewcommand{\headrulewidth}{0.4pt}')
    w(r'\renewcommand{\headrule}{\hbox to\headwidth{\color{accent}\leaders\hrule height \headrulewidth\hfill}}')
    w()
    w(r'\setlength{\parindent}{0pt}')
    w(r'\setlength{\parskip}{4pt}')
    w()
    w(r'\begin{document}')
    w()

    # ── Header Title Block ──
    w(r'\begin{center}')
    w(r'{\Huge\bfseries\textcolor{accent}{NBA Prediction Analysis}}\\[4pt]')
    w(r'{\large\textcolor{accentgold}{' + parsed.formatted_date + r'}}\\[2pt]')
    w(r'{\small\textcolor{midgray}{Automated Performance Verification \& WIN/LOSS Evaluation}}')
    w(r'\end{center}')
    w(r'\vspace{3pt}')
    w(r'\noindent\textcolor{accent}{\rule{\linewidth}{1.2pt}}')
    w(r'\vspace{6pt}')
    w()

    # ── Executive Performance Scorecard ──
    w(r'\begin{mdframed}[linecolor=accentgold,linewidth=1.3pt,backgroundcolor=lightgray,')
    w(r'  innertopmargin=8pt,innerbottommargin=8pt,innerrightmargin=8pt,innerleftmargin=8pt]')
    w(r'{\large\bfseries\textcolor{accentgold}{$\bigstar$ Daily Executive Scorecard}}\\[5pt]')
    w(r'\begin{tabular}{@{}p{0.32\linewidth}p{0.34\linewidth}p{0.30\linewidth}@{}}')

    # Card 1: Props Win Rate
    card1 = []
    card1.append(r'\textbf{\textcolor{accent}{Valuable Picks:}}\\')
    if parsed.props_total > 0:
        rate = parsed.props_win_rate
        if rate >= 55.0:
            badge_color = 'winbg'
            text_color = 'wintext'
        elif rate >= 50.0:
            badge_color = 'medconf'
            text_color = 'black'
        else:
            badge_color = 'lossbg'
            text_color = 'losstext'
        card1.append(f'\\textbf{{Record:}} {parsed.props_wins}W - {parsed.props_losses}L ({parsed.props_total} Total)\\\\')
        card1.append(f'\\textbf{{Win Rate:}} \\colorbox{{{badge_color}}}{{\\textbf{{\\textcolor{{{text_color}}}{{{rate:.1f}\\%}}}}}}\\\\')
    else:
        card1.append(r'\textit{No valuable props on date}\\')

    # Card 2: Moneyline & Team Scores
    card2 = []
    card2.append(r'\textbf{\textcolor{accent}{Game Result Accuracy:}}\\')
    if parsed.ml_total > 0:
        ml_badge = 'winbg' if parsed.ml_pct >= 60.0 else ('medconf' if parsed.ml_pct >= 50.0 else 'lossbg')
        ml_txt = 'wintext' if parsed.ml_pct >= 60.0 else ('black' if parsed.ml_pct >= 50.0 else 'losstext')
        card2.append(f'\\textbf{{Moneyline:}} {parsed.ml_correct}/{parsed.ml_total} '
                     f'\\colorbox{{{ml_badge}}}{{\\textbf{{\\textcolor{{{ml_txt}}}{{{parsed.ml_pct:.1f}\\%}}}}}}\\\\')
    if parsed.game_mae_team is not None:
        card2.append(f'\\textbf{{Team Score MAE:}} {parsed.game_mae_team:.1f} pts\\\\')
        card2.append(f'\\textbf{{Spread Margin MAE:}} {parsed.game_mae_spread:.1f} pts\\\\')
    else:
        card2.append(r'\textit{No game score data}\\')

    # Card 3: Median Projections
    card3 = []
    card3.append(r'\textbf{\textcolor{accent}{Median Projections:}}\\')
    if parsed.medians_total > 0:
        card3.append(f'\\textbf{{Players Evaluated:}} {parsed.medians_total}\\\\')
        if parsed.medians_mae:
            card3.append(f"\\textbf{{MAE (IQR):}} {parsed.medians_mae.get('PTS', 0):.1f}p | "
                         f"{parsed.medians_mae.get('REB', 0):.1f}r | {parsed.medians_mae.get('AST', 0):.1f}a\\\\")
        if parsed.medians_dir_acc:
            p_pct = parsed.medians_dir_acc.get('PTS', (0, 0, 0))[2]
            r_pct = parsed.medians_dir_acc.get('REB', (0, 0, 0))[2]
            a_pct = parsed.medians_dir_acc.get('AST', (0, 0, 0))[2]
            card3.append(f"\\textbf{{Dir. Acc:}} {p_pct:.0f}\\% | {r_pct:.0f}\\% | {a_pct:.0f}\\%\\\\")
    else:
        card3.append(r'\textit{No median projections}\\')

    w('\n'.join(card1) + ' & ' + '\n'.join(card2) + ' & ' + '\n'.join(card3) + r' \\')
    w(r'\end{tabular}')
    w(r'\end{mdframed}')
    w(r'\vspace{6pt}')
    w()

    # ── Section 1: Valuable Picks WIN / LOSS ──
    if parsed.props_results:
        w(r'\section*{Valuable Picks Outcome (WIN / LOSS)}')
        w(r'{\small Complete settlement of Markov-evaluated betting propositions, sorted by confidence rank.}\\[4pt]')
        w()
        w(r'\begin{longtable}{@{}p{0.28\linewidth}p{0.08\linewidth}p{0.22\linewidth}p{0.10\linewidth}p{0.10\linewidth}p{0.14\linewidth}@{}}')
        w(r'\toprule')
        w(r'\textbf{Player / Pick} & \textbf{Conf} & \textbf{Bet Proposition} & \textbf{Actual} & \textbf{Diff} & \textbf{Outcome} \\')
        w(r'\midrule')
        w(r'\endhead')
        w(r'\bottomrule')
        w(r'\endfoot')

        for p in parsed.props_results:
            pname = tex_escape(p['player'])
            conf_str = p['conf'] + r'\%' if p['conf'] and p['conf'] != 'N/A' else 'N/A'
            cc = conf_color(p['conf'])
            bet_str = tex_escape(p['bet'])
            actual_str = tex_escape(p['actual'])
            diff_str = tex_escape(p['diff'])
            if p['is_win']:
                status_tex = r'\colorbox{winbg}{\textbf{\textcolor{wintext}{WON \ding{51}}}}'
            else:
                status_tex = r'\colorbox{lossbg}{\textbf{\textcolor{losstext}{LOST \ding{55}}}}'

            w(f"  {pname} & {cc}{conf_str} & {bet_str} & {actual_str} & {diff_str} & {status_tex} \\\\")

        w(r'\end{longtable}')
        w(r'\vspace{8pt}')
        w()

    # ── Section 2: Game Projections & Score Accuracy ──
    if parsed.game_results:
        w(r'\section*{Game Projections \& Moneyline Accuracy}')
        w(r'\begin{longtable}{@{}p{0.38\linewidth}p{0.24\linewidth}p{0.20\linewidth}p{0.12\linewidth}@{}}')
        w(r'\toprule')
        w(r'\textbf{Matchup} & \textbf{Projected Score} & \textbf{Actual Score} & \textbf{Moneyline} \\')
        w(r'\midrule')
        w(r'\endhead')
        w(r'\bottomrule')
        w(r'\endfoot')

        for g in parsed.game_results:
            matchup = tex_escape(g['matchup'])
            proj = tex_escape(g['projected'])
            act = tex_escape(g['actual'])
            if g['is_correct']:
                stat_tex = r'\colorbox{winbg}{\textbf{\textcolor{wintext}{CORRECT \ding{51}}}}'
            else:
                stat_tex = r'\colorbox{lossbg}{\textbf{\textcolor{losstext}{WRONG \ding{55}}}}'
            w(f"  {matchup} & {proj} & {act} & {stat_tex} \\\\")

        w(r'\end{longtable}')
        w(r'\vspace{8pt}')
        w()

    # ── Section 3: Median Projection Accuracy (Filtered IQR) ──
    if parsed.medians_results:
        w(r'\section*{Player Median Projections Accuracy (Filtered IQR)}')
        if parsed.medians_mae and parsed.medians_dir_acc:
            pts_dir = parsed.medians_dir_acc.get('PTS', (0, 0, 0))
            reb_dir = parsed.medians_dir_acc.get('REB', (0, 0, 0))
            ast_dir = parsed.medians_dir_acc.get('AST', (0, 0, 0))
            w(r'\begin{mdframed}[linecolor=accent,linewidth=0.8pt,backgroundcolor=lightgray,')
            w(r'  innertopmargin=4pt,innerbottommargin=4pt,innerrightmargin=6pt,innerleftmargin=6pt]')
            w(f"{{\\small \\textbf{{Filtered MAE:}} PTS: {parsed.medians_mae.get('PTS', 0):.1f} | REB: {parsed.medians_mae.get('REB', 0):.1f} | AST: {parsed.medians_mae.get('AST', 0):.1f} "
              f"\\hfill \\textbf{{Direction Accuracy:}} PTS: {pts_dir[0]}/{pts_dir[1]} ({pts_dir[2]:.1f}\\%) | REB: {reb_dir[0]}/{reb_dir[1]} ({reb_dir[2]:.1f}\\%) | AST: {ast_dir[0]}/{ast_dir[1]} ({ast_dir[2]:.1f}\\%)}}")
            w(r'\end{mdframed}')
            w(r'\vspace{4pt}')

        w(r'\begin{longtable}{@{}p{0.25\linewidth}p{0.23\linewidth}p{0.23\linewidth}p{0.23\linewidth}@{}}')
        w(r'\toprule')
        w(r'\textbf{Player} & \textbf{Points (P / A / Diff)} & \textbf{Rebounds (P / A / Diff)} & \textbf{Assists (P / A / Diff)} \\')
        w(r'\midrule')
        w(r'\endhead')
        w(r'\bottomrule')
        w(r'\endfoot')

        def format_dir_tex(res_str):
            if res_str == 'EXACT':
                return r'\textbf{\textcolor{wintext}{EXACT}}'
            elif res_str == 'HIGH':
                return r'\textcolor{neutraltext}{HIGH}'
            elif res_str == 'LOW':
                return r'\textcolor{neutraltext}{LOW}'
            return tex_escape(res_str)

        for m in parsed.medians_results:
            pname = tex_escape(m['player'])
            pts_info = f"{m['pts_pred']} / {m['pts_act']} ({m['pts_diff']}) {format_dir_tex(m['pts_res'])}"
            reb_info = f"{m['reb_pred']} / {m['reb_act']} ({m['reb_diff']}) {format_dir_tex(m['reb_res'])}"
            ast_info = f"{m['ast_pred']} / {m['ast_act']} ({m['ast_diff']}) {format_dir_tex(m['ast_res'])}"
            w(f"  {pname} & \\small{{{pts_info}}} & \\small{{{reb_info}}} & \\small{{{ast_info}}} \\\\")

        w(r'\end{longtable}')
        w()

    w(r'\end{document}')
    return '\n'.join(lines)


# ── Compiler & File Management ────────────────────────────────────────────────

def compile_latex_to_pdf(tex_path: str, out_dir: str) -> bool:
    """Compile tex to pdf using pdflatex on Windows or WSL, then clean auxiliary files."""
    tex_path = os.path.abspath(tex_path)
    out_dir = os.path.abspath(out_dir)
    stem = os.path.splitext(os.path.basename(tex_path))[0]
    pdf_path = os.path.join(out_dir, stem + '.pdf')

    cmd = None
    if shutil.which('pdflatex') is not None:
        cmd = ['pdflatex', '-interaction=nonstopmode', f'-output-directory={out_dir}', tex_path]
    else:
        try:
            wsl_check = subprocess.run(['wsl', 'which', 'pdflatex'], capture_output=True, text=True)
            if wsl_check.returncode == 0 and '/pdflatex' in wsl_check.stdout:
                def win_to_wsl(p):
                    drive, rest = os.path.splitdrive(p)
                    drive_letter = drive.replace(':', '').lower()
                    return f"/mnt/{drive_letter}" + rest.replace('\\', '/')
                cmd = ['wsl', 'pdflatex', '-interaction=nonstopmode',
                       f'-output-directory={win_to_wsl(out_dir)}', win_to_wsl(tex_path)]
        except Exception:
            pass

    if not cmd:
        print("    [!] pdflatex not found on Windows or WSL.")
        return False

    # 2 passes for cross references and longtable geometry
    for _ in range(2):
        subprocess.run(cmd, check=False, capture_output=True)

    if os.path.exists(pdf_path):
        print(f"    [+] PDF compiled successfully: {pdf_path}")
        # Clean auxiliary files
        for ext in ['.aux', '.log', '.tex', '.out']:
            clean_f = os.path.join(out_dir, stem + ext)
            if os.path.exists(clean_f):
                try:
                    os.remove(clean_f)
                except OSError:
                    pass
        print(f"    [+] Cleaned up auxiliary files ({stem}.aux, .log, .tex, .out)")
        return True
    else:
        print(f"    [-] pdflatex ran but PDF not found: {pdf_path}")
        return False


def convert_txt_to_pdf(
    input_txt: str,
    output_pdf: Optional[str] = None,
    game_date: Optional[str] = None,
    clean_txt: bool = True
) -> Optional[str]:
    """
    Parses an analyze_<date>.txt file, compiles it to analyze_report_<date>.pdf,
    cleans up auxiliary LaTeX files (.tex, .aux, .log, .out), and optionally deletes .txt.
    """
    if not os.path.isfile(input_txt):
        print(f"[-] Input file not found: {input_txt}")
        return None

    with open(input_txt, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    parsed = AnalyzeParser(raw_text, filepath=input_txt, game_date=game_date)
    out_dir = os.path.dirname(os.path.abspath(input_txt))

    if output_pdf:
        target_stem = os.path.splitext(output_pdf)[0]
    else:
        target_stem = os.path.join(out_dir, f"analyze_report_{parsed.date_str}")

    tex_path = target_stem + '.tex'
    pdf_path = target_stem + '.pdf'

    latex_code = generate_latex(parsed)
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)

    success = compile_latex_to_pdf(tex_path, os.path.dirname(tex_path))
    if success and os.path.exists(pdf_path):
        if clean_txt and os.path.exists(input_txt):
            try:
                os.remove(input_txt)
                print(f"    [+] Removed raw text analysis file: {input_txt}")
            except OSError as e:
                print(f"    [!] Failed to remove raw text file: {e}")
        return pdf_path

    return None


def main():
    parser = argparse.ArgumentParser(description='Convert analyze_<date>.txt to a styled PDF report')
    parser.add_argument('--input', type=str, help='Path to analyze_<date>.txt file')
    parser.add_argument('--dir', type=str, help='Directory containing analyze_*.txt files to batch convert')
    parser.add_argument('--output', type=str, help='Output PDF path (stem or .pdf)')
    parser.add_argument('--date', type=str, help='Game date in YYYY-MM-DD format')
    parser.add_argument('--keep-txt', action='store_true', help='Keep the raw .txt file after PDF generation')
    args = parser.parse_args()

    clean_txt = not args.keep_txt

    if args.input:
        res = convert_txt_to_pdf(args.input, output_pdf=args.output, game_date=args.date, clean_txt=clean_txt)
        if not res:
            sys.exit(1)
    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"[-] Directory not found: {args.dir}")
            sys.exit(1)
        txt_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.startswith('analyze_') and f.endswith('.txt')]
        print(f"[+] Found {len(txt_files)} analyze text files in {args.dir}")
        for tf in txt_files:
            print(f"\nProcessing {os.path.basename(tf)}...")
            convert_txt_to_pdf(tf, clean_txt=clean_txt)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
