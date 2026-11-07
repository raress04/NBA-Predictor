"""
Fit how strongly each player shooting / ball-security rate is pulled toward the league
average (T2 — Plans/T2_T3_IMPLEMENTATION_AUDIT.md).

Bug fixed: context_builder shrank every rate with CONFIDENCE_PRIOR_STRENGTH = 3000
"games", so a player's own numbers got ~0.5% weight and everyone shot like the league.

New rule (per stat):  rate = (made + prior * k) / (attempts + k)
  k is in ATTEMPTS of that stat, chosen so the estimate from the last 15 played games
  (EWMA-weighted exactly like context_builder) best predicts the rate over the NEXT
  10 games (attempt-weighted squared error).

Reads  : database/nba_data.db (read-only), regular season 2019-20 .. --last-season
Writes : config/shrinkage_params.json (previous file archived to config/archive/)

Run from advanced_model/:
    python3 -m scripts.fit_shrinkage --dry-run
    python3 -m scripts.fit_shrinkage
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
OUT_PATH = os.path.join(MODEL_DIR, 'config', 'shrinkage_params.json')

WINDOW = 15          # games the runtime uses (get_player_stats_matrix limit=15)
EWMA_SPAN = 10       # settings.EWMA_SPAN
HORIZON = 10         # games ahead used as the target
MIN_AHEAD = 5
K_GRID = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000]

# stat -> (made, attempts) column expressions; matrix field noted for the runtime
STATS = {
    'fg2_pct':     ('fg2m', 'fg2a'),     # 2P% per 2PA
    'fg3_pct_att': ('fg3m', 'fg3a'),     # 3P% per 3PA (runtime stores x p_shoot_3)
    'ft_pct':      ('ftm', 'fta'),       # FT%
    'p_shoot_3':   ('fg3a', 'fga'),      # share of FGA that are threes
    'fta_per_fga': ('fta', 'fga'),       # FTA per FGA
    'tov_rate':    ('tov', 'poss'),      # TOV per possession used
}
# legacy runtime (for comparison): (raw * games + prior * 3000) / (games + 3000),
# priors hard-coded in context_builder.LEAGUE_PRIORS
LEGACY_PRIORS = {'fg2_pct': 0.525, 'fg3_per_fga': 0.145, 'ft_pct': 0.782, 'tov_rate': 0.130,
                 'p_shoot_3': 0.40, 'fta_per_fga': 0.27}
LEGACY_STRENGTH = 3000


def load(last_season: str) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{NBA_DB}?mode=ro", uri=True)
    df = pd.read_sql("""SELECT player_id, season, substr(game_date,1,10) d, game_id, minutes,
                               fgm, fga, fg3m, fg3a, ftm, fta, tov
                        FROM box_scores WHERE game_id LIKE '002%' AND season >= '2019-20' AND season <= ?""",
                     conn, params=(last_season,))
    conn.close()
    df = df[pd.to_numeric(df.minutes, errors='coerce') > 0].copy()
    for c in ('fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta', 'tov'):
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    df['fg2m'] = df.fgm - df.fg3m
    df['fg2a'] = df.fga - df.fg3a
    df['poss'] = df.fga + 0.44 * df.fta + df.tov
    return df.sort_values(['player_id', 'd', 'game_id']).reset_index(drop=True)


def ewma_weights(n: int, span: int = EWMA_SPAN) -> np.ndarray:
    """pandas ewm(span, adjust=False).mean().iloc[-1] weights for a window of n values (oldest first)."""
    a = 2.0 / (span + 1.0)
    w = np.array([a * (1 - a) ** (n - 1 - i) for i in range(n)])
    w[0] = (1 - a) ** (n - 1)
    return w


def build_samples(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, game t) with >= WINDOW prior games and >= MIN_AHEAD next games:
    EWMA-window totals (as runtime) and next-HORIZON sums, for every made/attempt column."""
    cols = sorted({c for pair in STATS.values() for c in pair} | {'fgm', 'fga', 'fg3m', 'fta', 'tov'})
    w = ewma_weights(WINDOW) * WINDOW          # EWMA mean x n_games = runtime "totals"
    out = []
    for pid, g in df.groupby('player_id', sort=False):
        if len(g) < WINDOW + MIN_AHEAD:
            continue
        X = g[cols].to_numpy(float)
        seasons = g.season.to_numpy()
        win = np.lib.stride_tricks.sliding_window_view(X, WINDOW, axis=0)    # (n-W+1, ncols, W)
        tot = (win * w).sum(axis=2)                                            # window ending at i+W-1
        cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)])
        n = len(g)
        for t in range(WINDOW, n - MIN_AHEAD + 1):
            end = min(t + HORIZON, n)
            row = {'player_id': pid, 'season': seasons[t]}
            hist = tot[t - WINDOW]
            fut = cs[end] - cs[t]
            for j, c in enumerate(cols):
                row[f'h_{c}'] = hist[j]
                row[f'f_{c}'] = fut[j]
            out.append(row)
    return pd.DataFrame(out)


def league_prior(df: pd.DataFrame, made: str, att: str) -> float:
    return float(df[made].sum() / df[att].sum())


def loss(pred, s, made, att):
    a = s[f'f_{att}'].to_numpy()
    m = a > 0
    r = s[f'f_{made}'].to_numpy()[m] / a[m]
    return float(np.sum(a[m] * (r - pred[m]) ** 2) / np.sum(a[m]))


def predict(s, made, att, prior, k):
    hm, ha = s[f'h_{made}'].to_numpy(), s[f'h_{att}'].to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(ha + k > 0, (hm + prior * k) / (ha + k), prior)


def legacy_predict(s, stat):
    """What context_builder produces today (fg3 per attempt = fg3_per_fga / p_shoot_3)."""
    g = WINDOW
    f = lambda raw, p: (raw * g + p * LEGACY_STRENGTH) / (g + LEGACY_STRENGTH)
    with np.errstate(invalid='ignore', divide='ignore'):
        fga = s['h_fga'].to_numpy()
        safe = lambda num, den: np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
        if stat == 'fg3_pct_att':
            p3 = f(safe(s['h_fg3a'].to_numpy(), fga), LEGACY_PRIORS['p_shoot_3'])
            return f(safe(s['h_fg3m'].to_numpy(), fga), LEGACY_PRIORS['fg3_per_fga']) / p3
        made, att = STATS[stat]
        return f(safe(s[f'h_{made}'].to_numpy(), s[f'h_{att}'].to_numpy()), LEGACY_PRIORS[stat])


def fit_k(s, made, att, prior):
    losses = {k: loss(predict(s, made, att, prior, k), s, made, att) for k in K_GRID}
    best = min(losses, key=losses.get)
    return best, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--last-season', default='2025-26')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    df = load(args.last_season)
    print(f"[fit] {len(df):,} player-games 2019-20..{args.last_season}; building samples ...", flush=True)
    s = build_samples(df)
    print(f"[fit] {len(s):,} samples (>= {WINDOW} prior games, >= {MIN_AHEAD} next games)")
    holdout = args.last_season
    train, test = s[s.season < holdout], s[s.season == holdout]
    latest = df[df.season == args.last_season]

    params = {'computed_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
              'source': 'scripts/fit_shrinkage.py (Plans/T2_T3_IMPLEMENTATION_AUDIT.md)',
              'formula': 'rate = (made + prior * k) / (attempts + k); k in attempts of that stat',
              'data': {'seasons': ['2019-20', args.last_season], 'game_type': 'regular season (002)',
                       'window_games': WINDOW, 'ewma_span': EWMA_SPAN, 'horizon_games': HORIZON},
              'priors_from_season': args.last_season,
              'stats': {}}
    print(f"\n{'stat':12} {'prior':>7} {'k(train)':>8} {'k(all)':>7} | holdout {holdout} error (x1e4): "
          f"{'legacy':>8} {'no shrink':>9} {'fitted':>8}")
    for stat, (made, att) in STATS.items():
        p_train = league_prior(df[df.season < holdout], made, att)
        k_train, _ = fit_k(train, made, att, p_train)
        p_test = league_prior(df[df.season < holdout], made, att)   # prior known before the holdout
        e_leg = loss(legacy_predict(test, stat), test, made, att)
        e_raw = loss(predict(test, made, att, p_test, 0), test, made, att)
        e_fit = loss(predict(test, made, att, p_test, k_train), test, made, att)
        p_all = league_prior(df, made, att)
        k_all, losses_all = fit_k(s, made, att, p_all)
        prior_latest = league_prior(latest, made, att)
        params['stats'][stat] = {
            'made': made, 'attempts': att, 'k': k_all, 'prior': round(prior_latest, 5),
            'k_train': k_train, 'holdout': {'season': holdout, 'mse_legacy': e_leg,
                                            'mse_no_shrink': e_raw, 'mse_fitted': e_fit}}
        print(f"{stat:12} {prior_latest:7.4f} {k_train:8} {k_all:7} | {'':27} {e_leg*1e4:8.2f} {e_raw*1e4:9.2f} {e_fit*1e4:8.2f}")

    if args.dry_run:
        print("\n[dry-run] nothing written.")
        return
    if os.path.exists(OUT_PATH):
        arch = os.path.join(MODEL_DIR, 'config', 'archive')
        os.makedirs(arch, exist_ok=True)
        shutil.copy2(OUT_PATH, os.path.join(arch, f"shrinkage_params_{datetime.now():%Y%m%d_%H%M%S}.json"))
    with open(OUT_PATH, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(params, f, indent=2)
        f.write('\n')
    print(f"\n[fit] wrote {OUT_PATH}")


if __name__ == '__main__':
    main()
