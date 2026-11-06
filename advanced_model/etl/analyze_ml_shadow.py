"""
etl/analyze_ml_shadow.py
========================
Evaluates how valuable picks suggested by the ML Shadow Layer performed
against actual NBA box scores, and compiles a comprehensive PDF report.

Features:
- Loads predictions (predicts_YYYY-MM-DD.txt) and actual box scores (yesterday_results_YYYY-MM-DD.txt).
- Evaluates XGBoost calibration models to compute ML-calibrated win probability for each prop.
- Compares baseline model vs ML Shadow layer:
  * Win rates, Brier scores, edge calibration
  * Filtered false positives (props baseline took that ML correctly rejected)
  * Upgraded picks (props ML promoted)
- Compiles an executive PDF report into:
  analyze/<season>/ml_shadow_layer/<month>/analyze_ml_shadow_YYYY-MM-DD.pdf
- Automatically cleans up auxiliary LaTeX pollution (.aux, .log, .tex, .out).

Usage:
    py -m etl.analyze_ml_shadow --date 2026-10-22
    py -m etl.analyze_ml_shadow --date 2026-10-23 --mode live
    py -m etl.analyze_ml_shadow --start-date 2026-10-22 --end-date 2026-10-25
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import re
import argparse
import sqlite3
import subprocess
import shutil
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from utils.prediction_paths import (
    get_nba_season,
    resolve_prediction_file,
    resolve_results_file,
)
from etl.analyze_predictions import parse_predicts, parse_results
from ml.shadow_logger import _xgb_predict, _load_models
from etl.live_ingestion import compile_latex_to_pdf

NBA_DB = os.path.join(MODEL_DIR, 'database', 'nba_data.db')
TRACKER_DB = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')


def tex_escape(text: str) -> str:
    """Escapes special LaTeX characters and symbols."""
    if not text:
        return ""
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
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def get_ml_shadow_dir(date_val: str, base_dir: Optional[str] = None) -> str:
    """
    Returns path:
    <base_dir>/analyze/<season>/ml_shadow_layer/<month>/
    """
    base = base_dir or MODEL_DIR
    dt = datetime.strptime(date_val[:10], '%Y-%m-%d')
    season = get_nba_season(dt)
    month = dt.strftime('%B').lower()
    target_dir = os.path.join(base, 'analyze', season, 'ml_shadow_layer', month)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def evaluate_date_ml_shadow(date_str: str, mode: str = 'live', base_dir: Optional[str] = None) -> Optional[Dict]:
    """
    Evaluates ML layer predictions vs actual box scores for a given date.
    Returns dictionary with evaluation metrics and detailed pick results.
    """
    base = base_dir or MODEL_DIR
    predicts_file = resolve_prediction_file(date_str, mode=mode, base_dir=base, allow_legacy=True)
    if not predicts_file:
        alt_mode = 'backtest' if mode == 'live' else 'live'
        predicts_file = resolve_prediction_file(date_str, mode=alt_mode, base_dir=base, allow_legacy=True)

    results_file = resolve_results_file(date_str, mode=mode, base_dir=base, allow_legacy=True)

    if not predicts_file or not os.path.exists(predicts_file):
        print(f"[-] No prediction file found for {date_str} (mode={mode})")
        return None

    if not results_file or not os.path.exists(results_file):
        print(f"[-] No results file found for {date_str} (mode={mode})")
        return None

    games, baseline_props, medians = parse_predicts(predicts_file)
    actual_games, actual_props = parse_results(results_file)

    if not actual_props:
        print(f"[-] No actual player results found in {os.path.basename(results_file)}")
        return None

    # Load XGBoost models
    _load_models()

    evaluated_picks = []

    # Map player projected medians for context
    median_lookup = {}
    for m in medians:
        median_lookup[(m['player'], 'POINTS')] = m.get('PTS')
        median_lookup[(m['player'], 'REBOUNDS')] = m.get('REB')
        median_lookup[(m['player'], 'ASSISTS')] = m.get('AST')

    for p in baseline_props:
        cat = p['stat']
        if cat not in ('POINTS', 'REBOUNDS', 'ASSISTS'):
            continue

        player = p['player']
        line = float(p['line'])
        direction = p['direction'].upper()
        base_conf = float(p.get('conf') or 65.0)

        projected = median_lookup.get((player, cat))
        if projected is None:
            projected = line + (2.0 if direction == 'OVER' else -2.0)

        # Build feature vector for XGB calibration
        vec = {
            'projected_value': float(projected),
            'bookmaker_line': line,
            'edge_pct': abs(float(projected) - line) / max(line, 1.0) * 100.0,
            'conf': base_conf,
            'gap': float(projected) - line,
            'over_implied': 0.5,
            'under_implied': 0.5,
            'vig': 0.0,
            'odds_skew': 0.0,
            'direction_bin': 1.0 if direction == 'OVER' else 0.0
        }

        prob_over = _xgb_predict(cat, vec)
        if prob_over is not None:
            if direction == 'UNDER':
                ml_conf = (1.0 - prob_over) * 100.0
            else:
                ml_conf = prob_over * 100.0
        else:
            ml_conf = base_conf

        # Match against actual result
        actual_val = actual_props.get(player, {}).get(cat, None)
        if actual_val is None:
            continue

        if direction == 'OVER':
            is_win = 1 if actual_val > line else (0 if actual_val < line else None)
        else:
            is_win = 1 if actual_val < line else (0 if actual_val > line else None)

        if is_win is None:
            status = 'PUSH'
        elif is_win == 1:
            status = 'WON'
        else:
            status = 'LOST'

        # Classification under ML Shadow Layer
        # ML considers valuable if ml_conf >= 65.0%
        is_ml_valuable = ml_conf >= 65.0
        is_base_valuable = base_conf >= 65.0

        evaluated_picks.append({
            'player': player,
            'stat': cat,
            'line': line,
            'direction': direction,
            'base_conf': round(base_conf, 1),
            'ml_conf': round(ml_conf, 1),
            'projected': round(projected, 1),
            'actual': actual_val,
            'diff': round(actual_val - line, 1),
            'status': status,
            'is_win': is_win,
            'is_base_valuable': is_base_valuable,
            'is_ml_valuable': is_ml_valuable,
        })

    if not evaluated_picks:
        print(f"[-] No evaluatable picks found for {date_str}.")
        return None

    # Calculate summary statistics
    # 1. Baseline stats
    base_picks = [p for p in evaluated_picks if p['is_win'] is not None and p['is_base_valuable']]
    base_n = len(base_picks)
    base_wins = sum(1 for p in base_picks if p['is_win'] == 1)
    base_win_rate = (base_wins / base_n * 100.0) if base_n > 0 else 0.0

    # 2. ML Shadow Layer stats
    ml_picks = [p for p in evaluated_picks if p['is_win'] is not None and p['is_ml_valuable']]
    ml_n = len(ml_picks)
    ml_wins = sum(1 for p in ml_picks if p['is_win'] == 1)
    ml_win_rate = (ml_wins / ml_n * 100.0) if ml_n > 0 else 0.0

    # 3. Filtered false positives: props baseline took that lost, which ML correctly lowered < 65%
    avoided_losses = [
        p for p in evaluated_picks 
        if p['is_win'] == 0 and p['is_base_valuable'] and not p['is_ml_valuable']
    ]

    # Category breakdown
    cat_summary = {}
    for cat in ('POINTS', 'REBOUNDS', 'ASSISTS'):
        c_base = [p for p in base_picks if p['stat'] == cat]
        c_ml = [p for p in ml_picks if p['stat'] == cat]

        c_base_wr = (sum(1 for p in c_base if p['is_win'] == 1) / len(c_base) * 100.0) if c_base else 0.0
        c_ml_wr = (sum(1 for p in c_ml if p['is_win'] == 1) / len(c_ml) * 100.0) if c_ml else 0.0

        cat_summary[cat] = {
            'base_n': len(c_base),
            'base_wr': c_base_wr,
            'ml_n': len(c_ml),
            'ml_wr': c_ml_wr,
        }

    return {
        'date': date_str,
        'mode': mode,
        'predicts_file': predicts_file,
        'results_file': results_file,
        'base_n': base_n,
        'base_wins': base_wins,
        'base_win_rate': base_win_rate,
        'ml_n': ml_n,
        'ml_wins': ml_wins,
        'ml_win_rate': ml_win_rate,
        'avoided_losses_count': len(avoided_losses),
        'cat_summary': cat_summary,
        'picks': evaluated_picks,
    }


def generate_ml_shadow_latex(data: Dict) -> str:
    """Generates clean, professional LaTeX markup for the ML Shadow Layer Report."""
    date_str = data['date']
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    formatted_date = dt.strftime('%B %d, %Y')

    lines = []
    def w(s=''):
        lines.append(s)

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
    w(r'\usepackage{fancyhdr}')
    w(r'\usepackage{mdframed}')
    w(r'\usepackage{microtype}')
    w(r'\usepackage{lmodern}')
    w(r'\usepackage{parskip}')
    w()

    # Colors
    w(r'\definecolor{darkbg}{HTML}{0D1117}')
    w(r'\definecolor{accent}{HTML}{1F6FEB}')
    w(r'\definecolor{accentgreen}{HTML}{2EA043}')
    w(r'\definecolor{accentred}{HTML}{DA3633}')
    w(r'\definecolor{winbg}{HTML}{D4EDDA}')
    w(r'\definecolor{lossbg}{HTML}{F8D7DA}')
    w(r'\definecolor{wintext}{HTML}{155724}')
    w(r'\definecolor{losstext}{HTML}{721C24}')
    w(r'\definecolor{cardbg}{HTML}{F6F8FA}')
    w(r'\definecolor{midgray}{HTML}{6E7681}')
    w()

    # Page setup
    w(r'\pagestyle{fancy}')
    w(r'\fancyhf{}')
    w(r'\fancyhead[L]{\small\textcolor{midgray}{AI Bet Tool $|$ ML Shadow Layer Analysis}}')
    w(f'\\fancyhead[R]{{\\small\\textcolor{{midgray}}{{{formatted_date}}}}}')
    w(r'\fancyfoot[C]{\small\textcolor{midgray}{\thepage}}')
    w(r'\renewcommand{\headrulewidth}{0.4pt}')
    w(r'\renewcommand{\footrulewidth}{0pt}')
    w()

    w(r'\begin{document}')
    w()

    # Header
    w(r'\begin{center}')
    w(r'{\LARGE\bfseries\textcolor{accent}{ML Shadow Layer Performance Report}}\\[4pt]')
    w(f'{{\\large\\bfseries {formatted_date}}} \\\\[2pt]')
    w(r'{\small\textcolor{midgray}{Benchmarking XGBoost Probability Calibration vs Baseline Markov Simulation Projections}}')
    w(r'\end{center}')
    w(r'\vspace{6pt}')

    # KPI Summary Cards
    delta_wr = data['ml_win_rate'] - data['base_win_rate']
    delta_color = 'accentgreen' if delta_wr >= 0 else 'accentred'
    delta_sign = '+' if delta_wr >= 0 else ''

    w(r'\begin{mdframed}[linecolor=accent,linewidth=1.2pt,backgroundcolor=cardbg,innertopmargin=8pt,innerbottommargin=8pt]')
    w(r'\begin{tabularx}{\textwidth}{XXXX}')
    w(r'\textbf{Baseline Pick Slate} & \textbf{ML Shadow Slate} & \textbf{Win Rate Impact} & \textbf{False Positives Filtered} \\')
    w(f"{data['base_wins']}/{data['base_n']} ({data['base_win_rate']:.1f}\\%) & "
      f"{data['ml_wins']}/{data['ml_n']} ({data['ml_win_rate']:.1f}\\%) & "
      f"\\textcolor{{{delta_color}}}{{\\textbf{{{delta_sign}{delta_wr:.1f}\\%}}}} & "
      f"\\textbf{{{data['avoided_losses_count']}}} losing bets avoided \\\\")
    w(r'\end{tabularx}')
    w(r'\end{mdframed}')
    w(r'\vspace{8pt}')

    # Category Breakdown Table
    w(r'{\bfseries\large\textcolor{darkbg}{1. Category Breakdown (Baseline vs ML Calibrated)}}\\[4pt]')
    w(r'\begin{tabularx}{\textwidth}{lcccc}')
    w(r'\toprule')
    w(r'\textbf{Prop Category} & \textbf{Baseline Volume} & \textbf{Baseline Win Rate} & \textbf{ML Shadow Volume} & \textbf{ML Shadow Win Rate} \\')
    w(r'\midrule')
    for cat in ('POINTS', 'REBOUNDS', 'ASSISTS'):
        c = data['cat_summary'][cat]
        w(f"{cat} & {c['base_n']} picks & {c['base_wr']:.1f}\\% & {c['ml_n']} picks & {c['ml_wr']:.1f}\\% \\\\")
    w(r'\bottomrule')
    w(r'\end{tabularx}')
    w(r'\vspace{12pt}')

    # Detailed Pick Audit Table
    w(r'{\bfseries\large\textcolor{darkbg}{2. Detailed Pick-by-Pick Audit}}\\[4pt]')
    w(r'\begin{small}')
    w(r'\begin{longtable}{p{4.2cm}lrrrccrr}')
    w(r'\toprule')
    w(r'\textbf{Player} & \textbf{Stat} & \textbf{Line} & \textbf{Base \%} & \textbf{ML \%} & \textbf{Actual} & \textbf{Result} & \textbf{Diff} & \textbf{ML Action} \\')
    w(r'\midrule')
    w(r'\endhead')

    for p in data['picks']:
        p_name = tex_escape(p['player'])
        cat = p['stat'][:3]
        line_val = f"{p['direction']} {p['line']:.1f}"
        base_c = f"{p['base_conf']:.0f}\\%"
        ml_c = f"{p['ml_conf']:.0f}\\%"
        act_val = f"{p['actual']}"
        diff_val = f"{p['diff']:+0.1f}"

        if p['status'] == 'WON':
            res_cell = r'\cellcolor{winbg}\textcolor{wintext}{\textbf{WON}}'
        elif p['status'] == 'LOST':
            res_cell = r'\cellcolor{lossbg}\textcolor{losstext}{\textbf{LOST}}'
        else:
            res_cell = r'\textcolor{midgray}{PUSH}'

        # Action tag
        if p['is_base_valuable'] and p['is_ml_valuable']:
            action = r'\textcolor{accent}{\textbf{Accepted}}'
        elif p['is_base_valuable'] and not p['is_ml_valuable']:
            if p['status'] == 'LOST':
                action = r'\textcolor{accentgreen}{\textbf{Avoided Loss}}'
            else:
                action = r'\textcolor{accentred}{Downgraded}'
        elif not p['is_base_valuable'] and p['is_ml_valuable']:
            action = r'\textcolor{accentgreen}{\textbf{Upgraded}}'
        else:
            action = r'\textcolor{midgray}{Ignored}'

        w(f"{p_name} & {cat} & {line_val} & {base_c} & {ml_c} & {act_val} & {res_cell} & {diff_val} & {action} \\\\")

    w(r'\bottomrule')
    w(r'\end{longtable}')
    w(r'\end{small}')

    w(r'\end{document}')
    return '\n'.join(lines)


def run_ml_shadow_analysis(target_date: str, mode: str = 'live', base_dir: Optional[str] = None) -> Optional[str]:
    """
    Main runner: evaluates ML shadow picks, writes LaTeX, compiles PDF, cleans auxiliary files.
    Returns path of generated PDF report.
    """
    base = base_dir or MODEL_DIR
    data = evaluate_date_ml_shadow(target_date, mode=mode, base_dir=base)
    if not data:
        return None

    out_dir = get_ml_shadow_dir(target_date, base_dir=base)
    stem = f"analyze_ml_shadow_{target_date}"
    tex_path = os.path.join(out_dir, f"{stem}.tex")
    pdf_path = os.path.join(out_dir, f"{stem}.pdf")

    latex_code = generate_ml_shadow_latex(data)
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)

    print(f"[+] LaTeX generated at: {tex_path}")
    print(f"[*] Compiling ML Shadow PDF report...")

    success = compile_latex_to_pdf(tex_path, out_dir)

    # Clean auxiliary files gracefully (.aux, .log, .tex, .out, .fls, .fdb_latexmk)
    for ext in ('.aux', '.log', '.tex', '.out', '.fls', '.fdb_latexmk', 'texput.log'):
        aux_f = os.path.join(out_dir, f"{stem}{ext}")
        if os.path.exists(aux_f):
            try:
                os.remove(aux_f)
            except OSError:
                pass

    if success and os.path.exists(pdf_path):
        print(f"[+] ML Shadow PDF report compiled successfully:")
        print(f"    -> {pdf_path}")
        return pdf_path
    else:
        print(f"[-] Compilation failed or PDF not found at {pdf_path}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate ML Shadow Layer predictions vs box scores and generate PDF report.")
    parser.add_argument('--date', type=str, help="Target game date (YYYY-MM-DD)")
    parser.add_argument('--start-date', type=str, help="Start date for range evaluation")
    parser.add_argument('--end-date', type=str, help="End date for range evaluation")
    parser.add_argument('--mode', choices=['live', 'backtest'], default='live', help="Execution mode (default: live)")
    args = parser.parse_args()

    if args.date:
        dates = [args.date]
    elif args.start_date and args.end_date:
        start_dt = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(args.end_date, '%Y-%m-%d')
        dates = []
        cur = start_dt
        while cur <= end_dt:
            dates.append(cur.strftime('%Y-%m-%d'))
            cur += timedelta(days=1)
    else:
        # Default: the previous US-Eastern slate
        from utils.slate_date import previous_slate_date
        yesterday = previous_slate_date()
        dates = [yesterday]

    for d in dates:
        print(f"\n==================================================")
        print(f"🎯 Running ML Shadow Layer Analysis for {d}...")
        print(f"==================================================")
        run_ml_shadow_analysis(d, mode=args.mode)


if __name__ == '__main__':
    main()
