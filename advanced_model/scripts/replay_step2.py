"""
Step 2 replay — old vs new, same inputs (Plans/STEP2_IMPLEMENTATION_AUDIT.md §8).
--compare t2: legacy vs fitted player shrinkage (Plans/T2_T3_IMPLEMENTATION_AUDIT.md 2.5).

For each date with games and prop lines:
  1. Simulate every game twice with the backtest path (process_hist_game): once with the
     LEGACY profile matrices, once with STEP2 matrices. Same random seed per game (common
     random numbers), same lineups (real box-score lineups = known backtest hindsight).
  2. Feed the distributions into the real build_parlays gates, with saving disabled:
        OLD = legacy sims + legacy gate     (pre-Step-2 model)
        AB  = legacy sims + step2 gate      (only A+B: bias off, one confidence scale)
        NEW = step2 sims  + step2 gate      (A+B+C+D)
  3. Grade picks against the actual box scores (pushes and DNPs excluded) and measure
     projection error (sim median vs actual) per group: all, B2B 2nd night, returning.

Read-only: databases are only read (no function in this path writes; persist_parlay is
patched out). Output: logs/november/step2_replay/ (report + per-pick CSV + projection CSV).

Line source per date: historical_odds_cache (scraped, has odds) when present, else the
lines the live run logged in shadow_picks (odds not stored there -> 1.909 default). The
same lines/odds are used for every variant.

Run from advanced_model/ (WSL):
    python3 -m scripts.replay_step2 --start 2026-10-20 --end 2026-11-04
"""

import argparse
import math
import os
import sqlite3
import sys
import time
import unittest.mock
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

import pandas as pd  # noqa: E402

import config.settings as settings  # noqa: E402

NBA_DB = os.path.join(MODEL_DIR, 'database', 'nba_data.db')
BET_DB = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')
OUT_DIR = os.path.join(MODEL_DIR, 'logs', 'november', 'step2_replay')

_STEP2 = dict(USE_RUNTIME_BIAS=False, CONFIDENCE_GATE='raw', B2B_MODEL='fitted',
              RETURNING_MODEL='fitted', SHADOW_MODEL_VERSION='ERA_3_STEP2')
COMPARISONS = {
    # Step 2: pre-Step-2 model vs Step 2 (both with the legacy player shrinkage, as at that time)
    'step2': {
        'profiles': {'legacy': dict(USE_RUNTIME_BIAS=True, CONFIDENCE_GATE='legacy', B2B_MODEL='off',
                                    RETURNING_MODEL='legacy', SHADOW_MODEL_VERSION='ERA_3',
                                    SHRINKAGE_MODEL='legacy', ENGINE_CALIBRATION='legacy'),
                     'step2': dict(_STEP2, SHRINKAGE_MODEL='legacy', ENGINE_CALIBRATION='legacy')},
        'variants': [('OLD', 'legacy', 'legacy'), ('AB', 'legacy', 'step2'), ('NEW', 'step2', 'step2')],
    },
    # T2: Step 2 model with legacy (league-average) vs fitted player shrinkage; same gates
    't2': {
        'profiles': {'legacy': dict(_STEP2, SHRINKAGE_MODEL='legacy', ENGINE_CALIBRATION='legacy'),
                     'step2': dict(_STEP2, SHRINKAGE_MODEL='fitted', ENGINE_CALIBRATION='v2')},
        'variants': [('OLD', 'legacy', 'step2'), ('NEW', 'step2', 'step2')],
    },
}
PROFILES = COMPARISONS['step2']['profiles']
VARIANTS = COMPARISONS['step2']['variants']
CAT = {'points': 'POINTS', 'rebounds': 'REBOUNDS', 'assists': 'ASSISTS'}
SIM_KEY = {'POINTS': 'PTS', 'REBOUNDS': 'REB', 'ASSISTS': 'AST'}


def apply_profile(name: str, compare: str = None):
    profiles = COMPARISONS[compare]['profiles'] if compare else PROFILES
    for k, v in profiles[name].items():
        setattr(settings, k, v)


def _ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


# ── Inputs ─────────────────────────────────────────────────────────────────
def games_for(d: str) -> list:
    from nba_api.stats.static import teams
    abbrev = {t['abbreviation']: t for t in teams.get_teams()}
    conn = _ro(NBA_DB)
    rows = conn.execute("SELECT game_id, matchup FROM games WHERE game_date LIKE ? "
                        "AND substr(game_id,1,3) <> '001'", (f"{d}%",)).fetchall()
    conn.close()
    seen, out = set(), []
    for gid, matchup in rows:
        if gid in seen:
            continue
        seen.add(gid)
        if ' vs. ' in matchup:
            h, a = matchup.split(' vs. ')
        elif ' @ ' in matchup:
            a, h = matchup.split(' @ ')
        else:
            continue
        hd, ad = abbrev.get(h.strip()), abbrev.get(a.strip())
        if not hd or not ad:
            continue
        mk = lambda t: {'teamId': t['id'], 'teamName': t['nickname'], 'teamCity': t['city'],
                        'teamTricode': t['abbreviation']}
        out.append({'game_id': gid, 'homeTeam': mk(hd), 'awayTeam': mk(ad)})
    return out


def lines_for(d: str):
    """{player: {'points': {...}, ...}}, source label."""
    conn = _ro(BET_DB)
    rows = conn.execute("SELECT player_name, stat_category, line, over_odds, under_odds "
                        "FROM historical_odds_cache WHERE date = ? "
                        "AND stat_category IN ('POINTS','REBOUNDS','ASSISTS')", (d,)).fetchall()
    source = 'historical_odds_cache'
    if not rows:
        rows = conn.execute("""
            SELECT player, stat_category, line, over_odds, under_odds FROM shadow_picks s
            WHERE game_date = ? AND stat_category IN ('POINTS','REBOUNDS','ASSISTS')
              AND COALESCE(superseded,0) = 0
              AND id = (SELECT MAX(id) FROM shadow_picks t WHERE t.game_date = s.game_date
                        AND t.player = s.player AND t.stat_category = s.stat_category
                        AND COALESCE(t.superseded,0) = 0)""", (d,)).fetchall()
        source = 'shadow_picks (live lines; odds default 1.909)'
    conn.close()
    inv = {v: k for k, v in CAT.items()}
    out = defaultdict(dict)
    for player, cat, line, over, under in rows:
        out[player][inv[cat]] = {'line': float(line), 'over_odds': float(over or 1.909),
                                 'under_odds': float(under or 1.909), 'book': 'replay'}
    return dict(out), (source if rows else None)


def actuals_for(d: str) -> dict:
    conn = _ro(NBA_DB)
    rows = conn.execute("SELECT player_name, pts, reb, ast FROM box_scores WHERE substr(game_date,1,10) = ? "
                        "AND CAST(minutes AS REAL) > 0", (d,)).fetchall()
    conn.close()
    return {p: {'POINTS': pts, 'REBOUNDS': reb, 'ASSISTS': ast} for p, pts, reb, ast in rows}


# ── Worker ─────────────────────────────────────────────────────────────────
def sim_task(task):
    profile, d, game, n_sims, compare = task
    apply_profile(profile, compare)
    from simulator import parlay_builder as pb
    pb.bias_logger.disabled = True     # do not append replay lines to logs/bias_corrections.log
    from etl.live_ingestion import process_hist_game
    t0 = time.time()
    p = process_hist_game(game, d, {}, n_sims=n_sims, seed=f"{d}|{game['game_id']}", return_dists=True)
    p.pop('game_logs', None)
    p['secs'] = round(time.time() - t0, 1)
    return profile, d, game['game_id'], p


# ── Gates ──────────────────────────────────────────────────────────────────
def run_gates(d, sims_by_game, lines, gate_profile, hist_rates):
    from simulator import parlay_builder as pb
    pb.bias_logger.disabled = True
    apply_profile(gate_profile)
    odds, props, dists, minutes = {}, {}, {}, {}
    for gid, p in sims_by_game.items():
        if 'prop_dists' not in p:
            continue
        odds[gid] = {'home_team': p['home_team_full'], 'away_team': p['away_team_full']}
        props[gid] = {pl: lines[pl] for pl in p['prop_dists'] if pl in lines}
        dists.update(p['prop_dists'])
        minutes.update(p['player_minutes'])
    with unittest.mock.patch.object(pb, 'persist_parlay', return_value=None), \
         unittest.mock.patch.object(pb, 'get_category_calibration', return_value=1.0), \
         unittest.mock.patch.object(settings, 'MAX_VALUABLE_PICKS_PER_DAY', 10 ** 6), \
         unittest.mock.patch('builtins.print'):
        out = pb.build_parlays([], odds, props, dists, {}, all_historical_rates=hist_rates,
                               all_player_minutes=minutes, game_date=d)
    return out


def grade(direction, line, actual):
    if actual is None or actual == line:
        return None
    return int(actual > line) if direction.upper() == 'OVER' else int(actual < line)


def wilson(k, n):
    if n == 0:
        return (float('nan'), float('nan'))
    p, z = k / n, 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (c - h), 100 * (c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2026-10-20')
    ap.add_argument('--end', default='2026-11-04')
    ap.add_argument('--sims', type=int, default=5000)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--daily-cap', type=int, default=settings.MAX_VALUABLE_PICKS_PER_DAY)
    ap.add_argument('--compare', choices=sorted(COMPARISONS), default='step2',
                    help="step2: pre-Step-2 vs Step 2 | t2: legacy vs fitted player shrinkage")
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()
    global PROFILES, VARIANTS, OUT_DIR
    PROFILES = COMPARISONS[args.compare]['profiles']
    VARIANTS = COMPARISONS[args.compare]['variants']
    if args.out_dir:
        OUT_DIR = args.out_dir
    elif args.compare != 'step2':
        OUT_DIR = os.path.join(MODEL_DIR, 'logs', 'november', f'{args.compare}_replay')
    os.makedirs(OUT_DIR, exist_ok=True)

    d0, d1 = date.fromisoformat(args.start), date.fromisoformat(args.end)
    dates, plan, skipped = [], {}, []
    for i in range((d1 - d0).days + 1):
        d = (d0 + timedelta(days=i)).isoformat()
        games = games_for(d)
        lines, src = lines_for(d)
        if not games or not lines:
            skipped.append((d, len(games), src))
            continue
        dates.append(d)
        plan[d] = (games, lines, src)
    print(f"[replay] dates: {len(dates)} | skipped: {skipped}", flush=True)

    tasks = [(prof, d, g, args.sims, args.compare) for d in dates for g in plan[d][0] for prof in ('legacy', 'step2')]
    print(f"[replay] {len(tasks)} game simulations x {args.sims} sims on {args.workers} workers", flush=True)
    sims = {d: {'legacy': {}, 'step2': {}} for d in dates}
    remaining = {d: 2 * len(plan[d][0]) for d in dates}
    pick_rows, proj_rows, team_rows, funnel = [], [], [], defaultdict(lambda: defaultdict(int))
    t_start = time.time()

    from simulator.context_builder import get_player_historical_hit_rates

    def finish_date(d):
        games, lines, src = plan[d]
        act = actuals_for(d)
        # date-aware DB hit rates (same for every variant)
        hist = {}
        for pl, pdata in lines.items():
            pl_lines = {SIM_KEY[CAT[k]]: v['line'] for k, v in pdata.items()}
            try:
                hist[pl] = get_player_historical_hit_rates(pl, pl_lines, limit=20, before_date=d)
            except Exception:
                pass
        for name, sim_prof, gate_prof in VARIANTS:
            out = run_gates(d, sims[d][sim_prof], lines, gate_prof, hist)
            for r in out['shadow_picks']:
                funnel[name][r['ban_reason'].split('(')[0].strip()] += 1
            for p in out['all_picks']:
                cat = CAT.get(p['stat_category'], p['stat_category'].upper())
                actual = act.get(p['player'], {}).get(cat)
                pick_rows.append(dict(variant=name, date=d, player=p['player'], cat=cat,
                                      direction=p['direction'].upper(), line=p['line'],
                                      projection=p['projection'], confidence=p['confidence'],
                                      raw=p.get('raw_sim_freq_pct'), posterior=p.get('posterior_conf'),
                                      edge=p['edge_pct'], score=p['score'], odds=p['odds'],
                                      actual=actual, win=grade(p['direction'], p['line'], actual),
                                      line_source=src))
        # team score accuracy (sim mean vs actual team points)
        nconn = _ro(NBA_DB)
        for g in games:
            pts = dict(nconn.execute("SELECT team_id, SUM(pts) FROM box_scores WHERE game_id = ? "
                                     "GROUP BY team_id", (g['game_id'],)).fetchall())
            for prof in ('legacy', 'step2'):
                p = sims[d][prof].get(g['game_id'], {})
                if 'score_dists' not in p:
                    continue
                for side, team in (('home', g['homeTeam']), ('away', g['awayTeam'])):
                    if team['teamId'] in pts:
                        sc = p['score_dists'][side]
                        team_rows.append(dict(profile=prof, date=d, game=g['game_id'], side=side,
                                              sim=sum(sc) / len(sc), actual=pts[team['teamId']]))
        nconn.close()
        # projection accuracy (all simulated players with an actual line-up appearance)
        for prof in ('legacy', 'step2'):
            for gid, p in sims[d][prof].items():
                if 'prop_dists' not in p:
                    continue
                b2b_teams = {p['home_team_full'] if p['b2b']['home'] else None,
                             p['away_team_full'] if p['b2b']['away'] else None}
                for pl, dist in p['prop_dists'].items():
                    if pl not in act:
                        continue
                    pm = p['player_minutes'].get(pl, {})
                    ri = sims[d]['step2'].get(gid, {}).get('player_minutes', {}).get(pl, {}).get('return_info')
                    for cat, key in SIM_KEY.items():
                        v = sorted(dist[key])
                        proj_rows.append(dict(profile=prof, date=d, game=gid, player=pl, cat=cat,
                                              median=v[len(v) // 2], mean=sum(v) / len(v),
                                              actual=act[pl][cat], b2b=pm.get('team') in b2b_teams,
                                              returning=bool(ri), return_game=(ri or {}).get('return_game'),
                                              games_missed=(ri or {}).get('games_missed'),
                                              has_line=pl in lines and cat.lower() in lines[pl]))
        sims[d] = None
        print(f"[replay] {d} graded ({len(games)} games, lines: {src}) "
              f"[{(time.time() - t_start) / 60:.1f} min]", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(sim_task, t) for t in tasks]
        for f in as_completed(futs):
            prof, d, gid, payload = f.result()
            sims[d][prof][gid] = payload
            remaining[d] -= 1
            if remaining[d] == 0:
                finish_date(d)

    picks = pd.DataFrame(pick_rows)
    proj = pd.DataFrame(proj_rows)
    picks.to_csv(os.path.join(OUT_DIR, 'replay_picks.csv'), index=False)
    proj.to_csv(os.path.join(OUT_DIR, 'replay_projections.csv'), index=False)
    teams = pd.DataFrame(team_rows)
    teams.to_csv(os.path.join(OUT_DIR, 'replay_team_scores.csv'), index=False)
    report = build_report(picks, proj, funnel, dates, skipped, plan, args, teams)
    with open(os.path.join(OUT_DIR, 'replay_report.txt'), 'w', encoding='utf-8') as fh:
        fh.write(report)
    print(report)


def build_report(picks, proj, funnel, dates, skipped, plan, args, teams=None):
    L = []
    w = L.append
    w("STEP 2 REPLAY — old vs new on identical inputs")
    w(f"window {args.start}..{args.end} | dates replayed {len(dates)} | sims/game {args.sims} | "
      f"daily cap for 'published' = {args.daily_cap}")
    w(f"skipped dates (games, line source): {skipped}")
    w("line sources: " + ', '.join(f"{d}: {plan[d][2].split(' ')[0]}" for d in dates))
    w("Variants: OLD = legacy sims + legacy gate | AB = legacy sims + step2 gate (A+B only) | "
      "NEW = step2 sims + step2 gate (A+B+C+D)")
    w("Grading: actual box score vs line; pushes and DNP excluded. Break-even at 1.909 = 52.4%.\n")

    w("1. PICKS THAT PASS ALL GATES (before the daily cap)")
    if picks.empty:
        w("   no picks in any variant")
    else:
        rows = []
        for name, _, _ in VARIANTS:
            g = picks[picks.variant == name]
            graded = g[g.win.notna()]
            k, n = int(graded.win.sum()), len(graded)
            lo, hi = wilson(k, n)
            over = (g.direction == 'OVER').mean() * 100 if len(g) else float('nan')
            rows.append([name, len(g), f"{len(g) / len(dates):.1f}", n,
                         f"{100 * k / n:.1f}%" if n else '-', f"[{lo:.1f}, {hi:.1f}]" if n else '-',
                         f"{over:.0f}%" if len(g) else '-',
                         ' '.join(f"{c[:3]}={int((g.cat == c).sum())}" for c in ('POINTS', 'REBOUNDS', 'ASSISTS'))])
        w(pd.DataFrame(rows, columns=['variant', 'picks', 'per day', 'graded', 'win rate', '95% CI',
                                      'OVER share', 'by category']).to_string(index=False))
        w("\n   by variant x category x direction (win rate, n graded):")
        t = (picks[picks.win.notna()].groupby(['variant', 'cat', 'direction']).win
             .agg(['mean', 'count']).reset_index())
        t['win'] = (100 * t['mean']).round(1).astype(str) + '% (' + t['count'].astype(str) + ')'
        w(t.pivot_table(index=['cat', 'direction'], columns='variant', values='win', aggfunc='first')
          .reindex(columns=[v[0] for v in VARIANTS]).fillna('-').to_string())

        w(f"\n2. 'PUBLISHED' — top {args.daily_cap}/day by (confidence, score), as the live cap does")
        rows = []
        for name, _, _ in VARIANTS:
            g = picks[picks.variant == name].sort_values(['date', 'confidence', 'score'],
                                                         ascending=[True, False, False])
            g = g.groupby('date').head(args.daily_cap)
            graded = g[g.win.notna()]
            k, n = int(graded.win.sum()), len(graded)
            lo, hi = wilson(k, n)
            rows.append([name, len(g), n, f"{100 * k / n:.1f}%" if n else '-',
                         f"[{lo:.1f}, {hi:.1f}]" if n else '-',
                         f"{(g.direction == 'OVER').mean() * 100:.0f}%" if len(g) else '-'])
        w(pd.DataFrame(rows, columns=['variant', 'picks', 'graded', 'win rate', '95% CI', 'OVER share'])
          .to_string(index=False))

    w("\n3. GATE FUNNEL (rejection reasons, all candidates)")
    reasons = sorted({r for f in funnel.values() for r in f})
    w(pd.DataFrame({name: [funnel[name].get(r, 0) for r in reasons] for name, _, _ in VARIANTS},
                   index=reasons).to_string())

    w("\n4. PROJECTION ACCURACY — sim median vs actual (players who played); MAE and bias (actual - median)")
    if not proj.empty:
        proj = proj.copy()
        proj['err'] = proj.actual - proj['median']
        groups = [('all players', proj.index == proj.index),
                  ('players with a prop line', proj.has_line),
                  ('B2B 2nd night', proj.b2b),
                  ('returning (games 1-3)', proj.returning),
                  ('returning game 1', proj.return_game == 1),
                  ('returning, missed >= 5', proj.games_missed.fillna(0) >= 5)]
        rows = []
        for label, m in groups:
            for cat in ('POINTS', 'REBOUNDS', 'ASSISTS'):
                cell = [label, cat]
                for prof in ('legacy', 'step2'):
                    s = proj[m & (proj.cat == cat) & (proj.profile == prof)]
                    cell += [f"{s.err.abs().mean():.2f}" if len(s) else '-',
                             f"{s.err.mean():+.2f}" if len(s) else '-']
                cell.append(int((m & (proj.cat == cat) & (proj.profile == 'step2')).sum()))
                rows.append(cell)
        w(pd.DataFrame(rows, columns=['group', 'cat', 'MAE old', 'bias old', 'MAE new', 'bias new', 'n'])
          .to_string(index=False))
    if teams is not None and not teams.empty:
        w("\n5. TEAM SCORES — sim mean vs actual team points; MAE and bias (actual - sim)")
        for prof, label in (('legacy', 'old'), ('step2', 'new')):
            t = teams[teams.profile == prof]
            e = t.actual - t.sim
            tot = (t.groupby('game').actual.sum() - t.groupby('game').sim.sum()).abs().mean()
            w(f"   {label}: n={len(t)} team-games | MAE {e.abs().mean():.2f} | bias {e.mean():+.2f} | "
              f"game total MAE {tot:.2f}")
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
