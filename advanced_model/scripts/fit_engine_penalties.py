"""
T2b — fit the shooting penalties the engine applies when players are missing
(Plans/T2_T3_IMPLEMENTATION_AUDIT.md, T2b).

Hard-coded today (etl/live_ingestion.py):
  apply_dynamic_injury_matrix : every active player's fg2/fg3 x 0.94 per missing star,
                                0.97 per starter, 0.99 per rotation player (worst 3),
                                x 0.85 more if 3+ core players missing ("synergy collapse")
  redistribute_usage          : fg2/fg3 x (1 - 0.12 * extra_fga / own_fga), floor 0.88
Measured on 29 games: together they cut shooting by ~11% (2P% 54.5 -> 48.6).

Fit: one player-level weighted regression on past regular seasons, using the SAME
definitions as the runtime (roles from the team's previous 15 games; "missing" = in that
window but not playing today; redistributed FGA = missing FGA x active player's share):

  eFG_today - eFG_baseline = c + a_star*n_star + a_starter*n_starter + a_rot*n_rot
                             + a_collapse*collapse + b*fga_increase + e      (weight = FGA today)

Multipliers = 1 + coefficient / mean baseline eFG. Coefficients whose 95% CI contains 0
are written as 1.0 (no effect). Writes config/engine_penalties.json.

Run from advanced_model/:  python3 -m scripts.fit_engine_penalties [--dry-run]
"""
import argparse
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NBA_DB = os.path.join(MODEL_DIR, 'database', 'nba_data.db')
OUT_PATH = os.path.join(MODEL_DIR, 'config', 'engine_penalties.json')
WINDOW = 15
MIN_ROLE_GAMES = 4          # runtime: roles[p]['games'] >= 4
MIN_PLAYER_GAMES = 5
INC_CAP = 1.5


def load(last_season):
    conn = sqlite3.connect(f"file:{NBA_DB}?mode=ro", uri=True)
    bs = pd.read_sql("""SELECT season, game_id, substr(game_date,1,10) d, team_id, player_name p, minutes,
                               pts, fga, fgm, fg3m FROM box_scores
                        WHERE game_id LIKE '002%' AND season >= '2019-20' AND season <= ?""",
                     conn, params=(last_season,))
    conn.close()
    bs['min'] = pd.to_numeric(bs.minutes, errors='coerce')
    bs = bs[bs['min'] > 0].copy()
    for c in ('pts', 'fga', 'fgm', 'fg3m'):
        bs[c] = pd.to_numeric(bs[c], errors='coerce').fillna(0.0)
    return bs


def build_rows(bs):
    rows = []
    for (season, team), g in bs.groupby(['season', 'team_id'], sort=False):
        games = g[['game_id', 'd']].drop_duplicates().sort_values(['d', 'game_id']).game_id.tolist()
        by_game = {gid: grp.set_index('p') for gid, grp in g.groupby('game_id')}
        for i in range(WINDOW, len(games)):
            window = pd.concat([by_game[x] for x in games[i - WINDOW:i]])
            agg = window.groupby(level=0).agg(games=('min', 'size'), pts=('pts', 'mean'), fga_m=('fga', 'mean'),
                                              min=('min', 'mean'), fga=('fga', 'sum'), fgm=('fgm', 'sum'),
                                              fg3m=('fg3m', 'sum'))
            today = by_game[games[i]]
            active = set(today.index)
            # injury-matrix roles (runtime logic)
            total_fga = agg.fga_m.sum()
            top2 = set(agg.pts.sort_values(ascending=False).index[:2])
            n_star = n_starter = n_rot = 0
            for p, r in agg[(agg.games >= MIN_ROLE_GAMES) & (~agg.index.isin(active))].iterrows():
                usage = r.fga_m / total_fga if total_fga > 0 else 0
                if p in top2 or usage > 0.25:
                    n_star += 1
                elif r['min'] >= 25.0:
                    n_starter += 1
                elif r['min'] >= 12.0:
                    n_rot += 1
            collapse = int(n_star + n_starter >= 3)
            # usage redistribution (runtime logic)
            missing_fga = agg.fga_m[~agg.index.isin(active)].sum()
            act = agg[agg.index.isin(active)]
            act_total = act.fga_m.sum()
            for p, r in act.iterrows():
                if r.games < MIN_PLAYER_GAMES or r.fga <= 0 or r.fga_m <= 0:
                    continue
                t = today.loc[p]
                if isinstance(t, pd.DataFrame):
                    t = t.iloc[0]
                if t.fga <= 0:
                    continue
                extra = missing_fga * (r.fga_m / act_total) if act_total > 0 and missing_fga >= 1.0 else 0.0
                rows.append(dict(season=season, efg_today=(t.fgm + 0.5 * t.fg3m) / t.fga,
                                 efg_base=(r.fgm + 0.5 * r.fg3m) / r.fga, w=t.fga,
                                 n_star=min(n_star, 3), n_starter=min(n_starter, 3), n_rot=min(n_rot, 3),
                                 collapse=collapse, inc=min(extra / r.fga_m, INC_CAP)))
    return pd.DataFrame(rows)


def fit_pace_factor(last_season, first_season='2023-24'):
    """
    Engine possessions = int(pace * factor) per team; every possession is one FGA or one
    TOV. Legacy factor 1.02 (a hack for missing second-chance shots). Fit: factor =
    sum(actual FGA + TOV) / sum(expected pace), expected pace = runtime get_team_pace
    (mean FGA + 0.44*FTA + TOV - OREB over each team's previous 15 games), averaged over
    both teams and truncated to int as the runtime does.
    """
    conn = sqlite3.connect(f"file:{NBA_DB}?mode=ro", uri=True)
    t = pd.read_sql("""SELECT season, game_id, MIN(substr(game_date,1,10)) d, team_id,
                              SUM(fga) fga, SUM(fta) fta, SUM(tov) tov, SUM(oreb) oreb
                       FROM box_scores WHERE game_id LIKE '002%' AND season >= ? AND season <= ?
                       GROUP BY season, game_id, team_id""", conn, params=(first_season, last_season))
    conn.close()
    for c in ('fga', 'fta', 'tov', 'oreb'):
        t[c] = pd.to_numeric(t[c], errors='coerce').fillna(0.0)
    t = t.sort_values(['team_id', 'd', 'game_id'])
    t['poss'] = t.fga + 0.44 * t.fta + t.tov - t.oreb
    t['pace15'] = t.groupby('team_id').poss.transform(lambda s: s.shift(1).rolling(15, min_periods=15).mean())
    g = t.groupby('game_id').agg(n=('team_id', 'size'), pace=('pace15', 'mean'), events=('fga', 'sum'),
                                 tov=('tov', 'sum'))
    g = g[(g.n == 2) & g.pace.notna()]
    expected = np.floor(g.pace) * 2            # int() per game, both teams
    actual = g.events + g.tov                  # FGA + TOV, both teams
    factor = float(actual.sum() / expected.sum())
    return {'factor': round(factor, 4), 'legacy': 1.02, 'games': int(len(g)),
            'seasons': [first_season, last_season],
            'note': 'engine possessions per team = int(pace * factor); matches real FGA + TOV'}


def wls(df, cols):
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in cols])
    y = (df.efg_today - df.efg_base).to_numpy()
    w = df.w.to_numpy()
    W = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
    resid = y - X @ beta
    sigma2 = np.sum(w * resid ** 2) / (w.sum() * (len(y) - X.shape[1]) / len(y))
    cov = sigma2 * np.linalg.inv((X * w[:, None]).T @ X)
    return dict(zip(['const'] + cols, beta)), dict(zip(['const'] + cols, np.sqrt(np.diag(cov))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--last-season', default='2025-26')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    bs = load(args.last_season)
    print(f"[fit] {len(bs):,} player-games; building rows (runtime definitions) ...", flush=True)
    df = build_rows(bs)
    cols = ['n_star', 'n_starter', 'n_rot', 'collapse', 'inc']
    coef, se = wls(df, cols)
    efg = float(np.average(df.efg_base, weights=df.w))
    print(f"[fit] {len(df):,} player-games | mean baseline eFG {efg:.4f}")
    print(f"      share with >=1 missing star {100*(df.n_star>0).mean():.0f}% | starter {100*(df.n_starter>0).mean():.0f}% "
          f"| rotation {100*(df.n_rot>0).mean():.0f}% | collapse {100*df.collapse.mean():.1f}% | inc>0 {100*(df.inc>0).mean():.0f}%")
    out = {}
    legacy = {'n_star': 0.94, 'n_starter': 0.97, 'n_rot': 0.99, 'collapse': 0.85, 'inc': '1 - 0.12*inc (floor 0.88)'}
    print(f"\n{'term':10} {'coef':>9} {'±95%':>8} {'multiplier':>11}  legacy")
    for c in cols:
        sig = abs(coef[c]) > 1.96 * se[c]
        mult = 1.0 + coef[c] / efg
        out[c] = {'coef': round(float(coef[c]), 5), 'ci95': round(float(1.96 * se[c]), 5),
                  'multiplier_per_unit': round(float(mult if sig else 1.0), 4),
                  'measured_multiplier': round(float(mult), 4), 'significant': bool(sig)}
        print(f"{c:10} {coef[c]:+9.4f} {1.96*se[c]:8.4f} {mult:11.4f}{'' if sig else ' (ns->1.0)'}  {legacy[c]}")
    print(f"{'const':10} {coef['const']:+9.4f} {1.96*se['const']:8.4f}  (regression to the mean / drift, not applied)")
    params = {'computed_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
              'source': 'scripts/fit_engine_penalties.py (T2b)',
              'data': {'seasons': ['2019-20', args.last_season], 'rows': int(len(df)), 'window_games': WINDOW},
              'mean_baseline_efg': round(efg, 4),
              'injury': {'star': out['n_star'], 'starter': out['n_starter'], 'rotation': out['n_rot'],
                         'collapse': out['collapse']},
              'usage_boost': {'per_unit_fga_increase': out['inc'], 'inc_cap': INC_CAP,
                              'note': 'shooting x (1 + coef/eFG * fga_increase), fga_increase = extra/own FGA'},
              'pace': fit_pace_factor(args.last_season)}
    print(f"\npace factor: {params['pace']['factor']} (legacy 1.02) over {params['pace']['games']:,} games")
    if args.dry_run:
        print("\n[dry-run] nothing written.")
        return
    if os.path.exists(OUT_PATH):
        arch = os.path.join(MODEL_DIR, 'config', 'archive')
        os.makedirs(arch, exist_ok=True)
        shutil.copy2(OUT_PATH, os.path.join(arch, f"engine_penalties_{datetime.now():%Y%m%d_%H%M%S}.json"))
    with open(OUT_PATH, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(params, f, indent=2)
        f.write('\n')
    print(f"\n[fit] wrote {OUT_PATH}")


if __name__ == '__main__':
    main()
