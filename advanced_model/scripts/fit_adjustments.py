"""
Fit the data-driven back-to-back (B2B) and returning-player adjustments.

Step 2 — Plans/STEP2_IMPLEMENTATION_AUDIT.md §4.3 and §5.3.

Reads  : database/nba_data.db (read-only: box_scores regular season, historical_injuries)
Writes : config/adjustment_params.json
         (an existing file is first copied to config/archive/adjustment_params_<timestamp>.json)

Run from advanced_model/:
    python3 -m scripts.fit_adjustments                 # fit on seasons <= 2025-26, write JSON
    python3 -m scripts.fit_adjustments --dry-run       # print only, write nothing
    python3 -m scripts.fit_adjustments --last-season 2026-27

Re-run once a season (after the regular season) to refresh the numbers.

Method (same as the read-only analysis in Season-Updates entry 39):
  B2B       — within player-season: each player's B2B 2nd-night games vs his other
              games; player-seasons with >= 3 B2B and >= 10 other games; weighted by
              B2B games; 95% CI. An effect whose CI contains the null is stored as the
              null value (significant=false).
  Returning — a return = player played >= 5 games for the team this season, then missed
              k >= 1 consecutive team games, then played again. Return game n (1..3) =
              n-th consecutive played game after the absence. Minutes ratio per
              role x absence bucket x n = sum(minutes) / sum(baseline), baseline = mean
              minutes over his last <= 10 played games before the absence. Shrunk toward
              the next-shorter absence bucket (m = 30). Per-minute FGA/REB/AST ratios per
              bucket x n, pooled over roles, shrunk to 1.0 (m = 30).
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
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

NBA_DB = os.path.join(MODEL_DIR, 'database', 'nba_data.db')
OUT_PATH = os.path.join(MODEL_DIR, 'config', 'adjustment_params.json')

# Role / bucket definitions — stored in the JSON so runtime and fit always agree.
B2B_ROLE_BOUNDS = [20.0, 28.0]                  # bench < 20 <= rotation < 28 <= starter
B2B_ROLES = ['bench', 'rotation', 'starter']
RETURN_ROLE_BOUNDS = [18.0, 24.0, 30.0]         # bench < 18 <= rotation < 24 <= starter < 30 <= star
RETURN_ROLES = ['bench', 'rotation', 'starter', 'star']
ABSENCE_BUCKETS = [(1, 2, '1-2'), (3, 4, '3-4'), (5, 9, '5-9'), (10, 19, '10-19'), (20, 10_000, '20+')]
RETURN_GAMES = [1, 2, 3]
SHRINK_M = 30.0
MIN_PRIOR_GAMES = 5
BASELINE_GAMES = 10


def _bucket(k: int) -> str:
    for lo, hi, label in ABSENCE_BUCKETS:
        if lo <= k <= hi:
            return label
    raise ValueError(k)


def _role(value: float, bounds: list, roles: list) -> str:
    for b, r in zip(bounds, roles):
        if value < b:
            return r
    return roles[-1]


def load_box_scores(last_season: str, first_season: str = '2019-20') -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{NBA_DB}?mode=ro", uri=True)
    bs = pd.read_sql(
        """SELECT season, game_id, substr(game_date,1,10) AS d, team_id, player_id, player_name, minutes,
                  fgm, fga, fg3m, fg3a, tov, pts, reb, ast
           FROM box_scores
           WHERE game_id LIKE '002%' AND season >= ? AND season <= ?""",
        conn, params=(first_season, last_season))
    conn.close()
    bs['min'] = pd.to_numeric(bs.minutes, errors='coerce')
    bs = bs[bs['min'] > 0].copy()          # played = minutes > 0 (same as runtime)
    bs['date'] = pd.to_datetime(bs.d)
    return bs


def team_games(bs: pd.DataFrame) -> pd.DataFrame:
    tg = (bs[['season', 'team_id', 'game_id', 'date']].drop_duplicates()
          .sort_values(['season', 'team_id', 'date']).reset_index(drop=True))
    tg['prev'] = tg.groupby(['season', 'team_id']).date.shift()
    tg['b2b'] = (tg.date - tg.prev).dt.days == 1
    return tg


# ════════════════════════════════════════════════════════════════════════════
#  B2B
# ════════════════════════════════════════════════════════════════════════════
def _wmean_ci(values: pd.Series, weights: pd.Series):
    m = values.replace([np.inf, -np.inf], np.nan).notna()
    v, w = values[m], weights[m]
    if len(v) < 2:
        return None, None, int(len(v))
    mean = float(np.average(v, weights=w))
    se = float(np.sqrt(np.average((v - mean) ** 2, weights=w) / max(len(v) - 1, 1)))
    return mean, 1.96 * se, int(len(v))


def fit_b2b(bs: pd.DataFrame, tg: pd.DataFrame) -> dict:
    x = bs.merge(tg[['team_id', 'game_id', 'b2b']], on=['team_id', 'game_id'])
    x['fg2m'] = x.fgm - x.fg3m
    x['fg2a'] = x.fga - x.fg3a
    agg = x.groupby(['season', 'player_id', 'b2b']).agg(
        g=('game_id', 'count'), min=('min', 'sum'), fg2m=('fg2m', 'sum'), fg2a=('fg2a', 'sum'),
        fg3m=('fg3m', 'sum'), fg3a=('fg3a', 'sum'), tov=('tov', 'sum')).unstack('b2b')
    agg = agg[(agg[('g', True)] >= 3) & (agg[('g', False)] >= 10)]

    def per(col, b):
        return agg[(col, b)] / agg[('g', b)]

    mpg_other = per('min', False)
    roles = mpg_other.apply(lambda v: _role(v, B2B_ROLE_BOUNDS, B2B_ROLES))
    w = agg[('g', True)]
    metrics = {
        'minutes_delta': (per('min', True) - per('min', False), 0.0),
        'tov_per_min_ratio': ((agg[('tov', True)] / agg[('min', True)]) /
                              (agg[('tov', False)] / agg[('min', False)]).replace(0, np.nan), 1.0),
        'fg2_ratio': ((agg[('fg2m', True)] / agg[('fg2a', True)].replace(0, np.nan)) /
                      (agg[('fg2m', False)] / agg[('fg2a', False)].replace(0, np.nan)), 1.0),
        'fg3_ratio': ((agg[('fg3m', True)] / agg[('fg3a', True)].replace(0, np.nan)) /
                      (agg[('fg3m', False)] / agg[('fg3a', False)].replace(0, np.nan)), 1.0),
    }
    out = {'role_bounds': B2B_ROLE_BOUNDS, 'roles': {},
           'apply': {'minutes_delta': True, 'tov_per_min_ratio': True,
                     'fg2_ratio': False, 'fg3_ratio': False},
           'note': 'Shooting ratios are reference only (apply=false): no shooting drop is applied on B2B.',
           'team_games': int(len(tg)), 'b2b_share': round(float(tg.b2b.mean()), 4),
           'player_seasons': int(len(agg))}
    for role in B2B_ROLES:
        sel = roles == role
        out['roles'][role] = {}
        for name, (series, null) in metrics.items():
            mean, ci, n = _wmean_ci(series[sel], w[sel])
            significant = mean is not None and abs(mean - null) > ci
            out['roles'][role][name] = {
                'value': round(mean if significant else null, 4),
                'measured': None if mean is None else round(mean, 4),
                'ci95': None if ci is None else round(ci, 4),
                'n': n, 'significant': bool(significant)}
    return out


# ════════════════════════════════════════════════════════════════════════════
#  Returning players
# ════════════════════════════════════════════════════════════════════════════
def return_events(bs: pd.DataFrame, tg: pd.DataFrame, inj_set: set = None) -> pd.DataFrame:
    """One row per (return event, return game n) for n in 1..3 (consecutive played games)."""
    stats = bs.set_index(['team_id', 'game_id', 'player_id'])[['min', 'fga', 'reb', 'ast']]
    stat_map = {k: tuple(v) for k, v in zip(stats.index, stats.values)}
    rows = []
    for (season, team), g in tg.groupby(['season', 'team_id'], sort=False):
        games = list(zip(g.game_id, g.date.dt.strftime('%Y-%m-%d')))
        roster = bs[(bs.season == season) & (bs.team_id == team)]
        played = roster.groupby('player_id').game_id.apply(set)
        names = roster.groupby('player_id').player_name.first()
        for pid, pset in played.items():
            hist = []                      # (min, fga, reb, ast) of played games before
            missed, missed_dates = 0, []
            for i, (gid, d) in enumerate(games):
                if gid in pset:
                    if missed and len(hist) >= MIN_PRIOR_GAMES:
                        base = np.array(hist[-BASELINE_GAMES:])
                        b_min = base[:, 0].sum()
                        injury = bool(inj_set) and any((names[pid], x) in inj_set for x in missed_dates)
                        for n in RETURN_GAMES:
                            j = i + n - 1
                            if j >= len(games) or games[j][0] not in pset:
                                break           # not n consecutive played games
                            m, fga, reb, ast = stat_map[(team, games[j][0], pid)]
                            rows.append(dict(season=season, k=missed, n=n,
                                             base=b_min / len(base), base_min_sum=b_min,
                                             base_fga=base[:, 1].sum(), base_reb=base[:, 2].sum(),
                                             base_ast=base[:, 3].sum(),
                                             min=m, fga=fga, reb=reb, ast=ast, injury=injury))
                    hist.append(stat_map[(team, gid, pid)])
                    missed, missed_dates = 0, []
                elif hist:
                    missed += 1
                    missed_dates.append(d)
    ev = pd.DataFrame(rows)
    ev['role'] = ev.base.apply(lambda v: _role(v, RETURN_ROLE_BOUNDS, RETURN_ROLES))
    ev['bucket'] = ev.k.apply(_bucket)
    return ev


def control_ratios(bs: pd.DataFrame, tg: pd.DataFrame) -> dict:
    """
    Control group: player-games with NO absence before them (the player also played the
    previous team game) and >= MIN_PRIOR_GAMES earlier played games on the team.

    Grouping players by a 10-game baseline creates regression to the mean (low-baseline
    "bench" players drift up next game, "stars" drift down) whether or not they were
    absent. The model applies no such drift to anyone, so the absence effect must be
    measured RELATIVE to this control: effect = ratio(returning) / ratio(control).
    """
    t = tg.copy()
    t['gidx'] = t.groupby(['season', 'team_id']).cumcount()
    x = bs.merge(t[['team_id', 'game_id', 'gidx']], on=['team_id', 'game_id'])
    x = x.sort_values(['season', 'team_id', 'player_id', 'gidx'])
    g = x.groupby(['season', 'team_id', 'player_id'], sort=False)
    x['k_prior'] = g.cumcount()
    x['prev_gidx'] = g.gidx.shift(1)
    for col in ('min', 'fga', 'reb', 'ast'):
        x[f'roll_{col}'] = g[col].transform(
            lambda s: s.shift(1).rolling(BASELINE_GAMES, min_periods=1).sum())
    x['roll_n'] = g['min'].transform(lambda s: s.shift(1).rolling(BASELINE_GAMES, min_periods=1).count())
    c = x[(x.k_prior >= MIN_PRIOR_GAMES) & (x.gidx - x.prev_gidx == 1)].copy()
    c['base'] = c.roll_min / c.roll_n
    c['role'] = c.base.apply(lambda v: _role(v, RETURN_ROLE_BOUNDS, RETURN_ROLES))
    out = {'minutes': {}, 'per_minute': {}, 'n': {}}
    for role in RETURN_ROLES:
        r = c[c.role == role]
        out['minutes'][role] = float(r['min'].sum() / r.base.sum())
        out['n'][role] = int(len(r))
    for stat in ('fga', 'reb', 'ast'):
        out['per_minute'][stat] = float((c[stat].sum() / c['min'].sum()) /
                                        (c[f'roll_{stat}'].sum() / c.roll_min.sum()))
    return out


def _shrink(r_cell, n, r_prior, m=SHRINK_M):
    if n == 0 or r_cell is None or np.isnan(r_cell):
        return r_prior
    return (n * r_cell + m * r_prior) / (n + m)


def fit_returning(ev: pd.DataFrame, control: dict) -> dict:
    """
    minutes_effect[role][bucket][n] = (sum min / sum baseline of returns) / control ratio
    of the same role — the part attributable to the absence. Shrunk (m=30) toward the
    next-shorter absence bucket; the first bucket shrinks toward 1.0 (no effect).
    per_minute_ratio[stat][bucket][n] — same idea, pooled over roles, shrunk to 1.0.
    minutes_ratio_raw keeps the unadjusted ratios for reference.
    """
    buckets = [b[2] for b in ABSENCE_BUCKETS]
    effect, raw, cells_n = {}, {}, {}
    for n in RETURN_GAMES:
        e_n = ev[ev.n == n]
        for role in RETURN_ROLES:
            prior = 1.0
            for b in buckets:
                c = e_n[(e_n.role == role) & (e_n.bucket == b)]
                r = c['min'].sum() / c.base.sum() if len(c) else np.nan
                rel = r / control['minutes'][role] if len(c) else np.nan
                val = _shrink(rel, len(c), prior)
                effect.setdefault(role, {}).setdefault(b, {})[str(n)] = round(float(val), 4)
                raw.setdefault(role, {}).setdefault(b, {})[str(n)] = None if np.isnan(r) else round(float(r), 4)
                cells_n.setdefault(role, {}).setdefault(b, {})[str(n)] = int(len(c))
                prior = val
    per_min = {}
    for stat in ('fga', 'reb', 'ast'):
        per_min[stat] = {}
        for b in buckets:
            per_min[stat][b] = {}
            for n in RETURN_GAMES:
                c = ev[(ev.n == n) & (ev.bucket == b) & (ev['min'] > 0) & (ev.base_min_sum > 0)]
                base_rate = c[f'base_{stat}'].sum() / c.base_min_sum.sum() if len(c) else np.nan
                rate = c[stat].sum() / c['min'].sum() if len(c) else np.nan
                r = rate / base_rate / control['per_minute'][stat] if base_rate and base_rate > 0 else np.nan
                per_min[stat][b][str(n)] = {'value': round(float(_shrink(r, len(c), 1.0)), 4),
                                            'measured': None if np.isnan(r) else round(float(r), 4),
                                            'n': int(len(c))}
    return {'minutes_effect': effect, 'minutes_ratio_raw': raw, 'cell_n': cells_n,
            'per_minute_ratio': per_min,
            'control': {'minutes_ratio': {k: round(v, 4) for k, v in control['minutes'].items()},
                        'per_minute_ratio': {k: round(v, 4) for k, v in control['per_minute'].items()},
                        'n': control['n']}}


def fit_injury_reference(ev: pd.DataFrame) -> dict:
    """Injury-confirmed game-1 minutes ratio (reference only — not used at runtime)."""
    e = ev[(ev.injury) & (ev.n == 1)]
    out = {}
    for role in RETURN_ROLES:
        for _, _, b in ABSENCE_BUCKETS:
            c = e[(e.role == role) & (e.bucket == b)]
            if len(c):
                out.setdefault(role, {})[b] = {'ratio': round(float(c['min'].sum() / c.base.sum()), 4),
                                               'n': int(len(c))}
    return out


def minutes_effect(params_ret: dict, base: float, k: int, n: int) -> float:
    role = _role(base, RETURN_ROLE_BOUNDS, RETURN_ROLES)
    return params_ret['minutes_effect'][role][_bucket(k)][str(n)]


def legacy_cap(k: int, base: float) -> float:
    cap = 18.0 if k >= 10 else 24.0 if k >= 5 else 30.0
    return min(cap, base)


def holdout_check(bs_all: pd.DataFrame, holdout_season: str) -> dict:
    """
    Fit on seasons before holdout_season; score minutes MAE on holdout_season returns.
    Two fair comparisons (same model with vs without the absence effect):
      plain   : baseline            vs baseline x effect        (what the runtime does)
      regress : baseline x control  vs baseline x control x effect
    plus the legacy caps (game 1 only — the caps applied to the first game back).
    """
    train = bs_all[bs_all.season < holdout_season]
    test = bs_all[bs_all.season == holdout_season]
    tg_train = team_games(train)
    p = fit_returning(return_events(train, tg_train), control_ratios(train, tg_train))
    ev = return_events(test, team_games(test))
    ctrl = p['control']['minutes_ratio']

    def mae(pred, y):
        return round(float(np.mean(np.abs(np.asarray(pred) - np.asarray(y)))), 3)

    out = {'holdout_season': holdout_season, 'games': {}}
    for n in RETURN_GAMES:
        for label, sel in (('all', lambda e: e.k >= 1), ('k>=5', lambda e: e.k >= 5)):
            e = ev[(ev.n == n) & sel(ev)]
            if not len(e):
                continue
            eff = np.array([minutes_effect(p, b, k, n) for b, k in zip(e.base, e.k)])
            c = np.array([ctrl[_role(b, RETURN_ROLE_BOUNDS, RETURN_ROLES)] for b in e.base])
            row = {'n': int(len(e)),
                   'plain_baseline': mae(e.base, e['min']),
                   'plain_with_effect': mae(e.base * eff, e['min']),
                   'regress_baseline': mae(e.base * c, e['min']),
                   'regress_with_effect': mae(e.base * c * eff, e['min'])}
            if n == 1:
                row['legacy_caps'] = mae([legacy_cap(k, b) for k, b in zip(e.k, e.base)], e['min'])
            out['games'][f'g{n}_{label}'] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--last-season', default='2025-26',
                    help='Last season included in the fit (default 2025-26: keeps the 2026-27 replay out-of-sample).')
    ap.add_argument('--dry-run', action='store_true', help='Print only; write nothing.')
    args = ap.parse_args()

    print(f"[fit] loading box scores 2019-20 .. {args.last_season} (regular season) ...")
    bs = load_box_scores(args.last_season)
    tg = team_games(bs)
    print(f"[fit] {len(bs):,} player-games, {len(tg):,} team-games")

    b2b = fit_b2b(bs, tg)

    conn = sqlite3.connect(f"file:{NBA_DB}?mode=ro", uri=True)
    inj = pd.read_sql("SELECT game_date d, player FROM historical_injuries WHERE status LIKE 'Out%'", conn)
    conn.close()
    inj_set = set(zip(inj.player, inj.d))
    ev = return_events(bs, tg, inj_set)
    ret = fit_returning(ev, control_ratios(bs, tg))
    ret['injury_confirmed_reference'] = {
        'use': False,
        'note': ('historical_injuries covers 2025-10-21..2026-04-06 only and injury reports are not '
                 'stored daily, so live cannot tell injury from rest. Reference only.'),
        'game1_minutes_ratio': fit_injury_reference(ev)}
    ret.update({'role_bounds': RETURN_ROLE_BOUNDS, 'roles': RETURN_ROLES,
                'absence_buckets': [[lo, hi, label] for lo, hi, label in ABSENCE_BUCKETS],
                'return_games': RETURN_GAMES, 'min_prior_games': MIN_PRIOR_GAMES,
                'baseline_games': BASELINE_GAMES, 'shrink_m': SHRINK_M,
                'events_game1': int((ev.n == 1).sum()),
                'events_game1_injury_confirmed': int(((ev.n == 1) & ev.injury).sum())})
    ret['holdout_validation'] = holdout_check(bs, args.last_season)

    params = {
        'computed_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'source': 'scripts/fit_adjustments.py (Plans/STEP2_IMPLEMENTATION_AUDIT.md)',
        'data': {'db': 'database/nba_data.db', 'table': 'box_scores', 'game_type': 'regular season (002)',
                 'seasons': ['2019-20', args.last_season]},
        'b2b': b2b,
        'returning': ret,
    }

    # ── Report ────────────────────────────────────────────────────────────
    print("\n=== B2B (2nd night vs other games, within player-season) ===")
    print(f"team-games {b2b['team_games']:,} | B2B share {100*b2b['b2b_share']:.1f}% | player-seasons {b2b['player_seasons']:,}")
    for role, d in b2b['roles'].items():
        cells = '  '.join(f"{k}={v['measured']:+.3f}±{v['ci95']:.3f}{'' if v['significant'] else ' (ns→null)'}"
                          if 'delta' in k else
                          f"{k}={v['measured']:.3f}±{v['ci95']:.3f}{'' if v['significant'] else ' (ns→null)'}"
                          for k, v in d.items())
        print(f"  {role:9} {cells}")

    ctl = ret['control']
    print("\n=== Returning: control (no absence) minutes / baseline — regression to the mean ===")
    print('  ' + '  '.join(f"{r}={ctl['minutes_ratio'][r]:.3f} (n={ctl['n'][r]:,})" for r in RETURN_ROLES)
          + '  | per-minute ' + ' '.join(f"{k}={v:.3f}" for k, v in ctl['per_minute_ratio'].items()))
    print("\n=== Returning: absence EFFECT on minutes (returns / control, shrunk), raw ratio in [] ===")
    for role in RETURN_ROLES:
        for b in [x[2] for x in ABSENCE_BUCKETS]:
            vals, raws, ns = (ret['minutes_effect'][role][b], ret['minutes_ratio_raw'][role][b],
                              ret['cell_n'][role][b])
            print(f"  {role:9} {b:6} " + '  '.join(
                f"g{n}={vals[str(n)]:.3f} [{raws[str(n)] if raws[str(n)] is not None else float('nan'):.3f}] (n={ns[str(n)]})"
                for n in RETURN_GAMES))
    print("\n=== Returning: per-minute effect (pooled roles, / control, shrunk to 1.0) — g1,g2,g3 ===")
    for stat, d in ret['per_minute_ratio'].items():
        print(f"  {stat}: " + ' | '.join(f"{b} " + ','.join(f"{d[b][str(n)]['value']:.3f}" for n in RETURN_GAMES)
                                         for b in d))
    hv = ret['holdout_validation']
    print(f"\n=== Holdout {hv['holdout_season']} (fit on earlier seasons): minutes MAE ===")
    for key, row in hv['games'].items():
        extra = f" | legacy caps {row['legacy_caps']}" if 'legacy_caps' in row else ''
        print(f"  {key:9} n={row['n']:5}: baseline {row['plain_baseline']} -> with effect {row['plain_with_effect']}"
              f" | regression-aware {row['regress_baseline']} -> with effect {row['regress_with_effect']}{extra}")

    if args.dry_run:
        print("\n[dry-run] nothing written.")
        return
    if os.path.exists(OUT_PATH):
        arch = os.path.join(MODEL_DIR, 'config', 'archive')
        os.makedirs(arch, exist_ok=True)
        dst = os.path.join(arch, f"adjustment_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        shutil.copy2(OUT_PATH, dst)
        print(f"[fit] previous params archived to {dst}")
    with open(OUT_PATH, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(params, f, indent=2)
        f.write('\n')
    print(f"[fit] wrote {OUT_PATH}")


if __name__ == '__main__':
    main()
