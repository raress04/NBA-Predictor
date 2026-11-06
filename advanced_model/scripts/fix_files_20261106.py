"""
scripts/fix_files_20261106.py
=============================
File part of the date fixes approved on 06.11.2026 (Plans/CRITICAL_BUGS_FOUND_06.11.md, §15).
Works on one advanced_model directory and records every operation in a manifest
(JSON) so the exact same changes can be replayed on another copy.

    python3 scripts/fix_files_20261106.py --model-dir DIR [--apply] --manifest ops.json

Nothing is ever deleted: invalid files are moved to `_invalid_20261106/` folders,
superseded folders to `_archive_*_20261106/`.
"""
import argparse
import datetime as dt
import json
import os
import re
import shutil
import sqlite3
import sys
from collections import defaultdict

MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
          'september', 'october', 'november', 'december']


def month_dir(d):
    return MONTHS[int(d[5:7]) - 1]


class Ops:
    def __init__(self, root, apply):
        self.root, self.apply, self.items = root, apply, []
        self.gone, self.made = set(), set()     # virtual view of moves during a dry run

    def exists(self, rel):
        rel = os.path.normpath(rel)
        if rel in self.made:
            return True
        if any(rel == g or rel.startswith(g + os.sep) for g in self.gone):
            return False
        return os.path.exists(self._p(rel))

    def _p(self, rel):
        return os.path.join(self.root, rel)

    def move(self, src, dst, why):
        if not self.exists(src):
            return False
        if self.exists(dst):
            raise SystemExit(f'refusing to overwrite {dst}')
        self.gone.add(os.path.normpath(src))
        self.made.add(os.path.normpath(dst))
        self.items.append({'op': 'move', 'src': src, 'dst': dst, 'why': why})
        print(f'  MOVE {src} -> {dst}   ({why})')
        if self.apply:
            os.makedirs(os.path.dirname(self._p(dst)), exist_ok=True)
            shutil.move(self._p(src), self._p(dst))
        return True

    def write(self, rel, text, why):
        self.items.append({'op': 'write', 'dst': rel, 'why': why})
        print(f'  WRITE {rel}   ({why})')
        if self.apply:
            os.makedirs(os.path.dirname(self._p(rel)), exist_ok=True)
            with open(self._p(rel), 'w', encoding='utf-8') as f:
                f.write(text)

    def produced(self, rel, why):
        """A file produced by an external generator (pdflatex / analysis)."""
        self.items.append({'op': 'write', 'dst': rel, 'why': why})
        print(f'  GENERATED {rel}   ({why})')


# ── ground truth ─────────────────────────────────────────────────────────────
class Truth:
    def __init__(self, root):
        nba = sqlite3.connect(f"file:{os.path.join(root, 'database', 'nba_data.db')}?mode=ro", uri=True)
        self.nba = nba
        self.tid = {t: a for t, a in nba.execute("SELECT team_id, abbreviation FROM teams")}
        self.nick = {a: n for _, a, _, n in nba.execute("SELECT team_id, abbreviation, city, nickname FROM teams")}
        self.pair_dates = defaultdict(set)
        teams = defaultdict(set)
        for gid, tid in nba.execute("SELECT DISTINCT game_id, team_id FROM box_scores"):
            if tid in self.tid:
                teams[gid].add(self.tid[tid])
        for gid, d in nba.execute("SELECT game_id, substr(game_date,1,10) FROM games"):
            if len(teams[gid]) == 2:
                self.pair_dates[frozenset(teams[gid])].add(d)
        self.max_date = nba.execute("SELECT MAX(substr(game_date,1,10)) FROM games").fetchone()[0]
        self.alias = re.compile(r'Draft\s*Kings|DraftKings', re.I)

    def pairs(self, text):
        out = set()
        for line in text.splitlines():
            if not re.search(r'(MATCHUP|\s@\s|\svs\.?\s)', line):
                continue
            core = self.alias.sub(' ', line.split('|')[0] if '|' in line and 'MATCHUP' not in line else line)
            found = {a for a, nk in self.nick.items() if re.search(r'\b' + re.escape(nk) + r'\b', core)}
            if len(found) == 2:
                out.add(frozenset(found))
        return out

    def results_text(self, d):
        """Exactly the format written by etl/prepare_analyze.generate_results_file()."""
        rows = self.nba.execute("""
            SELECT b.game_id, b.player_id, b.player_name, t.abbreviation, b.pts, b.reb, b.ast, g.matchup
            FROM box_scores b LEFT JOIN teams t ON b.team_id = t.team_id
            LEFT JOIN games g ON b.game_id = g.game_id
            WHERE b.game_date LIKE ? ORDER BY b.game_id""", (d + '%',)).fetchall()
        games = defaultdict(list)
        seen = set()
        for gid, pid, name, ab, pts, reb, ast, matchup in rows:
            if (gid, pid) in seen:
                continue
            seen.add((gid, pid))
            games[gid].append((name, ab, int(pts or 0), int(reb or 0), int(ast or 0), matchup or ''))
        out = []
        for gid in sorted(games):
            g = games[gid]
            m = g[0][5]
            if ' @ ' in m:
                away, home = [x.strip() for x in m.split(' @ ')]
            elif ' vs. ' in m:
                home, away = [x.strip() for x in m.split(' vs. ')]
            else:
                continue
            ap = sum(p[2] for p in g if p[1] == away)
            hp = sum(p[2] for p in g if p[1] == home)
            out.append(f"MATCHUP: {away} vs. {home}    {ap} - {hp}\n")
            for name, _, pts, reb, ast, _ in sorted(g, key=lambda p: -p[2]):
                out.append(f"{name:<22} | {pts} PTS, {reb} REB,  {ast} AST\n")
            out.append("\n")
        return ''.join(out)


def run_et(path):
    mt = dt.datetime.utcfromtimestamp(os.path.getmtime(path))
    return (mt - dt.timedelta(hours=4)).date().isoformat()


# ── steps ────────────────────────────────────────────────────────────────────
def step_rename_predictions(ops, truth, renames):
    print('\n[A] Live prediction files whose content belongs to another date')
    bases = ['predictions/2026-2027/live_predictions', 'predictions/2025-2026/live_new_gen_prediction']
    cands = []
    for base in bases:
        for dp, _, fs in os.walk(os.path.join(ops.root, base)):
            for fn in fs:
                m = re.fullmatch(r'predicts_(\d{4}-\d\d-\d\d)\.txt', fn)
                if m:
                    cands.append((m[1], os.path.relpath(os.path.join(dp, fn), ops.root)))
    for label, rel in sorted(cands):
        path = os.path.join(ops.root, rel)
        target = run_et(path)
        if target == label:
            continue
        text = open(path, encoding='utf-8', errors='replace').read()
        pairs = truth.pairs(text)
        if not pairs:
            continue
        on_label = sum(1 for p in pairs if label in truth.pair_dates.get(p, ()))
        on_target = sum(1 for p in pairs if target in truth.pair_dates.get(p, ()))
        if abs((dt.date.fromisoformat(target) - dt.date.fromisoformat(label)).days) > 2:
            continue                      # file re-written much later — mtime is not the run time
        if target > truth.max_date:
            ok = on_label == 0            # run-time evidence only (box scores not ingested yet)
        else:
            ok = on_target == len(pairs) and on_label < len(pairs)
        if not ok:
            print(f'  keep {rel}: run date {target}, games on target {on_target}/{len(pairs)}, on label {on_label}')
            continue
        new_rel = os.path.join(os.path.dirname(os.path.dirname(rel)), month_dir(target), f'predicts_{target}.txt')
        if ops.move(rel, new_rel, f'run on {target} (US Eastern); games played {target}'):
            renames[f'predicts_{label}.txt'] = f'predicts_{target}.txt'
            # header lines carry the wrong date as well
            if ops.apply:
                lines = open(os.path.join(ops.root, new_rel), encoding='utf-8', errors='replace').read().split('\n')
                for i in range(min(8, len(lines))):
                    lines[i] = lines[i].replace(label, target)
                open(os.path.join(ops.root, new_rel), 'w', encoding='utf-8').write('\n'.join(lines))
            pdf = os.path.join(os.path.dirname(rel), f'predicts_report_{label}.pdf')
            if ops.exists(pdf):
                ops.move(pdf, os.path.join('predictions/_invalid_20261106', os.path.relpath(pdf, 'predictions')),
                         f'title says {label}; regenerated as predicts_report_{target}.pdf')
                renames[f'__pdf__{label}'] = new_rel


def regenerate_pdf(ops, new_rel, target):
    print(f'\n[A2] Re-compiling PDF for {new_rel}')
    if not ops.apply:
        print('  (dry run)')
        return
    sys.path.insert(0, ops.root)
    import subprocess
    out_base = os.path.join(ops.root, os.path.dirname(new_rel), f'predicts_report_{target}')
    cmd = [sys.executable, '-m', 'etl.predicts_to_pdf', '--input', os.path.join(ops.root, new_rel),
           '--output', out_base, '--date', target, '--compile', '--clean']
    r = subprocess.run(cmd, cwd=ops.root, capture_output=True, text=True,
                       env={**os.environ, 'PYTHONPATH': os.environ.get('PYTHONPATH', '')})
    tex = out_base + '.tex'
    if os.path.exists(tex) and not os.path.exists(out_base + '.pdf'):
        for _ in range(2):
            subprocess.run(['pdflatex', '-interaction=nonstopmode', '-output-directory',
                            os.path.dirname(tex), tex], capture_output=True, text=True)
        for ext in ('.aux', '.log', '.out', '.tex'):
            if os.path.exists(out_base + ext):
                os.remove(out_base + ext)
    if os.path.exists(out_base + '.pdf'):
        ops.produced(os.path.relpath(out_base + '.pdf', ops.root), 'recompiled from corrected txt')
    else:
        print('  !! PDF not produced:', r.stdout[-600:], r.stderr[-600:])


def step_results(ops, truth):
    print('\n[B] Results files')
    # 2026-27: bad (doubled) and missing season-tree results
    for d, mode in (('2026-10-29', 'live'), ('2026-10-26', 'live'), ('2026-10-27', 'live'),
                    ('2026-10-28', 'live'), ('2026-11-02', 'backtest')):
        rel = f'results/2026-2027/{mode}_results/{month_dir(d)}/yesterday_results_{d}.txt'
        if ops.exists(rel):
            ops.move(rel, f'results/_invalid_20261106/{os.path.relpath(rel, "results")}',
                     'built from duplicated box-score rows (players twice, scores doubled)')
        ops.write(rel, truth.results_text(d), 'generated from de-duplicated box scores')
    # legacy flat folders (second writer)
    for m in ('october', 'november'):
        ops.move(f'results/{m}', f'results/_archive_legacy_flat_20261106/{m}',
                 'legacy writer (fetch_range_box_scores) — season tree is canonical')
    # 2025-26
    ops.move('results/2025-2026/april/yesterday_results_2026-04-20.txt',
             'results/_invalid_20261106/2025-2026/april/yesterday_results_2026-04-20.txt',
             'players twice, scores doubled')
    ops.write('results/2025-2026/april/yesterday_results_2026-04-20.txt', truth.results_text('2026-04-20'),
              'generated from de-duplicated box scores')
    for d in ('2026-02-14', '2026-02-16', '2026-02-17', '2026-02-18'):
        ops.move(f'results/2025-2026/february/yesterday_results_{d}.txt',
                 f'results/_invalid_20261106/2025-2026/february/yesterday_results_{d}.txt',
                 'no NBA games that day (All-Star break); file was a copy of Feb 1')
    ops.move('results/2025-2026/march/yesterday_results_2026-03-15.txt',
             'results/_invalid_20261106/2025-2026/march/yesterday_results_2026-03-15.txt',
             'contains the Mar 14 games; real Mar 15 games missing from nba_data.db (backfill, then regenerate)')


def step_analyses(ops, renames):
    print('\n[C] Analyses graded against the wrong day')
    for old in [k for k in renames if not k.startswith('__pdf__')]:
        label = old[len('predicts_'):-len('.txt')]
        for rel in (f'analyze/2026-2027/live_predictions/{month_dir(label)}/analyze_report_{label}.pdf',
                    f'analyze/2025-2026/{month_dir(label)}/analyze_{label}.txt'):
            if ops.exists(rel):
                ops.move(rel, f'analyze/_invalid_20261106/{os.path.relpath(rel, "analyze")}',
                         f'predictions in it were for a different day than the results used')


def regenerate_analyses(ops, dates):
    print('\n[C2] Regenerating live analyses (DB not written: write_db=False)')
    if not ops.apply:
        print('  (dry run)', dates)
        return
    sys.path.insert(0, ops.root)
    os.chdir(ops.root)
    from etl.analyze_predictions import generate_evaluate
    from utils.prediction_paths import get_analyze_dir
    for d in dates:
        p = os.path.join(ops.root, 'predictions/2026-2027/live_predictions', month_dir(d), f'predicts_{d}.txt')
        r = os.path.join(ops.root, 'results/2026-2027/live_results', month_dir(d), f'yesterday_results_{d}.txt')
        if not (os.path.exists(p) and os.path.exists(r)):
            print('  skip', d, 'missing', 'pred' if not os.path.exists(p) else 'results')
            continue
        out = get_analyze_dir(d, mode='live', base_dir=ops.root)
        generate_evaluate(predicts_path=p, results_path=r, output_dir=out, write_db=False,
                          generate_pdf=True, clean_txt=True)
        pdf = os.path.join(out, f'analyze_report_{d}.pdf')
        if os.path.exists(pdf):
            ops.produced(os.path.relpath(pdf, ops.root), 'analysis of live predictions vs same-day results')
        else:
            print('  !! no PDF for', d, os.listdir(out))


def step_archive_duplicate(ops):
    print('\n[D] Duplicate folder')
    ops.move('predictions/2025-2026/first_gen_prediction',
             'predictions/_archive_duplicate_20261106/2025-2026/first_gen_prediction',
             '95 files byte-identical to live_new_gen_prediction/')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--renames-out', required=True)
    a = ap.parse_args()
    root = os.path.abspath(a.model_dir)
    ops = Ops(root, a.apply)
    truth = Truth(root)
    renames = {}
    step_rename_predictions(ops, truth, renames)
    step_results(ops, truth)
    step_analyses(ops, renames)
    step_archive_duplicate(ops)
    for k, new_rel in list(renames.items()):
        if k.startswith('__pdf__'):
            regenerate_pdf(ops, new_rel, k[len('__pdf__'):] and new_rel[-14:-4])
    live_dates = sorted({v[len('predicts_'):-4] for k, v in renames.items()
                         if not k.startswith('__pdf__') and v.startswith('predicts_2026-1')}
                        | {'2026-10-26', '2026-10-27', '2026-10-28'})
    regenerate_analyses(ops, live_dates)
    with open(a.manifest, 'w') as f:
        json.dump(ops.items, f, indent=1)
    with open(a.renames_out, 'w') as f:
        for k, v in renames.items():
            if not k.startswith('__pdf__'):
                f.write(f'{k} {v}\n')
    print(f'\n{len(ops.items)} operations recorded in {a.manifest}')


if __name__ == '__main__':
    main()
