"""
Step 3 — calibration of the raw sim share into a true win probability
(Plans/STEP3_CALIBRATION_AUDIT.md, P2/P3).

Input  : raw = direction-specific % of sims that cleared the line (uncapped), per prop
         candidate; label = win (1/0) in that direction.
Sources: R = BACKTEST replay of the current engine (logs/november/step3_replay/candidates.csv)
         L = live shadow_picks, current engine (ERA_3_STEP2, game_date >= 2026-11-07)
         O = live shadow_picks, old engine (ERA_5_REFORMS) - consistency check only
Models : M0 raw | M1 base rate per group | M2 Platt pooled | M3 Platt per direction |
         M3c Platt per category | Mgi common slope + group intercepts |
         M4 Platt per group shrunk to pooled (L2, lambda by inner time CV) | M4u unpooled |
         ISO isotonic per direction (diagnostic: is the logistic shape adequate?)
Validation: rolling origin by date (train on dates < d, test on d), never shuffled;
         Brier, log-loss, ECE, AUC, reliability, top-20% win rate; bootstrap by date.

Read-only. Report: logs/november/step3/fit_report.txt (+ fit_results.json).
Run from advanced_model/:  python3 -m scripts.fit_calibration
"""
import argparse
import json
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BET_DB = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')
R_CSV = os.path.join(MODEL_DIR, 'logs', 'november', 'step3_replay', 'candidates.csv')
OUT_DIR = os.path.join(MODEL_DIR, 'logs', 'november', 'step3')
LIVE_START = '2026-11-07'          # first slate on the current engine (T2/T2b live)
LAMBDAS = [0.1, 0.3, 1, 3, 10, 30, 100]
GROUPS = [f"{c}_{d}" for c in ('POINTS', 'REBOUNDS', 'ASSISTS') for d in ('OVER', 'UNDER')]
EPS = 0.005


# ── data ────────────────────────────────────────────────────────────────────
def _frame(df, src):
    df = df.copy()
    df['cat'] = df['cat'].str.upper()
    df['dir'] = df['dir'].str.upper()
    df = df[df['cat'].isin(['POINTS', 'REBOUNDS', 'ASSISTS']) & df['raw'].notna() & df['y'].notna()]
    df['group'] = df['cat'] + '_' + df['dir']
    df['y'] = df['y'].astype(int)
    df['p_raw'] = (df['raw'] / 100.0).clip(EPS, 1 - EPS)
    df['x'] = np.log(df['p_raw'] / (1 - df['p_raw']))
    df['src'] = src
    return df.reset_index(drop=True)


def load_R():
    c = pd.read_csv(R_CSV)
    c = c.rename(columns={'date': 'date', 'direction': 'dir', 'win': 'y'})
    c['real_odds'] = c['line_source'].eq('historical_odds_cache')
    return _frame(c[['date', 'player', 'cat', 'dir', 'line', 'projection', 'raw', 'y', 'gate', 'odds',
                     'real_odds']], 'R')


def load_live():
    conn = sqlite3.connect(f"file:{BET_DB}?mode=ro", uri=True)
    q = """SELECT game_date AS date, player, stat_category AS cat, direction AS dir, line, projection,
                  raw_sim_freq AS raw, actual_result AS y, ban_reason AS gate, model_version,
                  CASE WHEN UPPER(direction)='OVER' THEN over_odds ELSE under_odds END AS odds
           FROM shadow_picks WHERE COALESCE(superseded,0)=0 AND actual_result IS NOT NULL
             AND stat_category IN ('POINTS','REBOUNDS','ASSISTS') AND game_date >= '2026-10-20'"""
    s = pd.read_sql(q, conn)
    conn.close()
    s['real_odds'] = s['odds'].notna() & (s['odds'].round(3) != 1.909)
    L = _frame(s[(s.model_version == 'ERA_3_STEP2') & (s.date >= LIVE_START)], 'L')
    O = _frame(s[s.model_version == 'ERA_5_REFORMS'], 'O')
    return L, O


# ── logistic models ────────────────────────────────────────────────────────
def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class Logit:
    """
    z = a + b*x + a_g + b_g*x ; penalty lam*(sum a_g^2 + b_g^2) on the group offsets.
    groups=None -> pooled Platt. slope_offsets=False -> common slope, group intercepts.
    lam ~ 0 -> unpooled per-group fit.
    """
    def __init__(self, group_col=None, lam=1e-6, slope_offsets=True):
        self.group_col, self.lam, self.slope_offsets = group_col, lam, slope_offsets

    def _design(self, df):
        x = df['x'].to_numpy()
        cols = [np.ones_like(x), x]
        if self.group_col:
            for g in self.levels:
                m = (df[self.group_col].to_numpy() == g).astype(float)
                cols.append(m)
                if self.slope_offsets:
                    cols.append(m * x)
        return np.column_stack(cols)

    def fit(self, df):
        self.levels = sorted(df[self.group_col].unique()) if self.group_col else []
        X, y = self._design(df), df['y'].to_numpy()
        n = len(y)
        pen = np.ones(X.shape[1]) * self.lam
        pen[:2] = 1e-6

        def f(w):
            z = X @ w
            p = _sig(z)
            nll = -np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)) / n
            grad = X.T @ (p - y) / n
            return nll + np.sum(pen * w * w) / n, grad + 2 * pen * w / n

        w0 = np.zeros(X.shape[1])
        w0[1] = 1.0
        self.w = minimize(f, w0, jac=True, method='L-BFGS-B').x
        return self

    def predict(self, df):
        X = self._design(df)
        return _sig(X @ self.w)

    def coefs(self):
        out = {'a': float(self.w[0]), 'b': float(self.w[1])}
        k = 2
        for g in self.levels:
            out[f'a_{g}'] = float(self.w[k]); k += 1
            if self.slope_offsets:
                out[f'b_{g}'] = float(self.w[k]); k += 1
        return out


class Separate:
    """One pooled Platt per level of `col` (fallback: pooled fit)."""
    def __init__(self, col):
        self.col = col

    def fit(self, df):
        self.pooled = Logit().fit(df)
        self.m = {g: Logit().fit(d) for g, d in df.groupby(self.col) if len(d) >= 30 and d.y.nunique() == 2}
        return self

    def predict(self, df):
        p = self.pooled.predict(df)
        for g, m in self.m.items():
            idx = (df[self.col] == g).to_numpy()
            if idx.any():
                p[idx] = m.predict(df[idx])
        return p


class Raw:
    def fit(self, df): return self
    def predict(self, df): return df['p_raw'].to_numpy()


class BaseRate:
    def fit(self, df):
        self.r = df.groupby('group').y.mean().to_dict()
        self.g = df.y.mean()
        return self

    def predict(self, df):
        return df['group'].map(self.r).fillna(self.g).to_numpy()


class Iso:
    def fit(self, df):
        self.m = {d: IsotonicRegression(out_of_bounds='clip', y_min=EPS, y_max=1 - EPS).fit(g['p_raw'], g['y'])
                  for d, g in df.groupby('dir')}
        return self

    def predict(self, df):
        p = df['p_raw'].to_numpy().copy()
        for d, m in self.m.items():
            idx = (df['dir'] == d).to_numpy()
            if idx.any():
                p[idx] = m.predict(df.loc[idx, 'p_raw'])
        return np.clip(p, EPS, 1 - EPS)


class M4:
    """Per-group Platt shrunk to pooled; lambda chosen by an inner rolling CV on the train dates."""
    def __init__(self, lams=LAMBDAS):
        self.lams = lams

    def fit(self, df):
        dates = sorted(df.date.unique())
        inner = dates[-max(2, len(dates) // 3):] if len(dates) >= 6 else []
        if inner:
            scores = {}
            for lam in self.lams:
                pr, yy = [], []
                for d in inner:
                    tr, te = df[df.date < d], df[df.date == d]
                    if tr.empty or te.empty:
                        continue
                    pr.append(Logit('group', lam).fit(tr).predict(te)); yy.append(te.y.to_numpy())
                scores[lam] = brier(np.concatenate(yy), np.concatenate(pr))
            self.lam = min(scores, key=scores.get)
        else:
            self.lam = 10.0
        self.m = Logit('group', self.lam).fit(df)
        return self

    def predict(self, df):
        return self.m.predict(df)


def models():
    return {
        'M0 raw': lambda: Raw(),
        'M1 base rate': lambda: BaseRate(),
        'M2 Platt pooled': lambda: Logit(),
        'M3 Platt per direction': lambda: Separate('dir'),
        'M3c Platt per category': lambda: Separate('cat'),
        'Mgi slope common, group intercepts': lambda: Logit('group', 1e-6, slope_offsets=False),
        'M4 per group shrunk': lambda: M4(),
        'M4u per group unpooled': lambda: Logit('group', 1e-6),
        'ISO isotonic per direction': lambda: Iso(),
    }


# ── metrics ────────────────────────────────────────────────────────────────
def brier(y, p): return float(np.mean((p - y) ** 2))


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, p, side='right') - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum()
                     for b in range(bins) if (idx == b).any()) / len(y))


def auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float('nan')


def top20(y, p):
    k = max(1, int(len(p) * 0.2))
    return float(y[np.argsort(-p)[:k]].mean())


def summary(y, p):
    return dict(n=len(y), brier=brier(y, p), logloss=logloss(y, p), ece=ece(y, p), auc=auc(y, p),
                mean_p=float(p.mean()), win=float(y.mean()), top20=top20(y, p))


def reliability(y, p, edges=(0, .5, .55, .6, .65, .7, .75, .8, .85, 1.0001)):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            rows.append(dict(band=f"{lo*100:.0f}-{min(hi,1)*100:.0f}%", n=int(m.sum()),
                             stated=round(float(p[m].mean()) * 100, 1), actual=round(float(y[m].mean()) * 100, 1)))
    return pd.DataFrame(rows)


def boot_diff(df, col_a, col_b, n=1000, seed=7):
    """95% CI of Brier(a) - Brier(b), resampling test DATES."""
    rng = np.random.default_rng(seed)
    by = {d: g for d, g in df.groupby('date')}
    keys = list(by)
    diffs = []
    for _ in range(n):
        s = pd.concat([by[k] for k in rng.choice(keys, len(keys))])
        diffs.append(brier(s.y.to_numpy(), s[col_a].to_numpy()) - brier(s.y.to_numpy(), s[col_b].to_numpy()))
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ── rolling origin ─────────────────────────────────────────────────────────
def rolling(df, test_frac=1 / 3, window=None, model_set=None, min_train_dates=5):
    dates = sorted(df.date.unique())
    n_test = max(3, int(round(len(dates) * test_frac)))
    test_dates = [d for d in dates[-n_test:] if dates.index(d) >= min_train_dates]
    ms = model_set or models()
    out, coefs = [], []
    for d in test_dates:
        prior = [x for x in dates if x < d]
        if window:
            prior = prior[-window:]
        tr, te = df[df.date.isin(prior)], df[df.date == d].copy()
        for name, mk in ms.items():
            m = mk().fit(tr)
            te[name] = m.predict(te)
            if name == 'M2 Platt pooled':
                coefs.append(dict(date=d, **m.coefs()))
            if name == 'M4 per group shrunk':
                coefs.append(dict(date=d, lam=m.lam, **{f"M4_{k}": v for k, v in m.m.coefs().items()}))
        out.append(te)
    return pd.concat(out, ignore_index=True), coefs, test_dates


def table(oos, names, by=None):
    rows = []
    groups = [(None, oos)] if by is None else list(oos.groupby(by))
    for key, g in groups:
        y = g.y.to_numpy()
        for name in names:
            s = summary(y, g[name].to_numpy())
            rows.append(dict(group=key or 'ALL', model=name, **{k: round(v, 4) if isinstance(v, float) else v
                                                                  for k, v in s.items()}))
    return pd.DataFrame(rows)


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--write', action='store_true',
                    help="Fit the final M4 on all of R and write config/prob_calibration.json (old file archived).")
    ap.add_argument('--write-only', action='store_true', help="Skip the validation report; only --write.")
    ap.add_argument('--train-end', default=None,
                    help="Fit only on R dates <= this (out-of-sample replays). Use with --params-out.")
    ap.add_argument('--params-out', default=None, help="Write the params here instead of config/.")
    args = ap.parse_args()
    if args.write_only:
        R = load_R()
        if args.train_end:
            R = R[R.date <= args.train_end]
        write_params(R, args.params_out)
        return
    os.makedirs(args.out_dir, exist_ok=True)
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 30)
    L_ = []
    w = lambda s='': (L_.append(str(s)), print(s, flush=True))
    res = {'computed_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), 'data': 'R = BACKTEST'}

    R = load_R()
    Lv, O = load_live()
    names = list(models())
    w("STEP 3 — CALIBRATION FIT REPORT (read-only)   *** R = BACKTEST (replay, hindsight lineups) ***")
    w(f"R backtest: {len(R):,} rows, {R.date.nunique()} dates {R.date.min()}..{R.date.max()} | "
      f"L live current engine: {len(Lv):,} rows, {Lv.date.nunique()} dates | O live old engine: {len(O):,} rows")
    w("\nRows per group:")
    w(pd.DataFrame({'R': R.group.value_counts(), 'L': Lv.group.value_counts(), 'O': O.group.value_counts()})
      .reindex(GROUPS).fillna(0).astype(int).to_string())

    # 1. main rolling-origin comparison on R
    oos, coefs, tdates = rolling(R)
    w(f"\n1. ROLLING ORIGIN on R (expanding window; test dates {tdates[0]}..{tdates[-1]}, {len(tdates)} days, "
      f"{len(oos):,} test rows)")
    t1 = table(oos, names)
    w(t1.drop(columns='group').to_string(index=False))
    res['main'] = t1.to_dict('records')

    w("\n   Brier difference vs M0 raw and vs M1 base rate (95% CI, bootstrap by date; negative = better)")
    rows = []
    for name in names[2:]:
        d0 = boot_diff(oos, name, 'M0 raw'); d1 = boot_diff(oos, name, 'M1 base rate')
        rows.append(dict(model=name, vs_raw=f"{d0[0]:+.4f} [{d0[1]:+.4f},{d0[2]:+.4f}]",
                         vs_base=f"{d1[0]:+.4f} [{d1[1]:+.4f},{d1[2]:+.4f}]",
                         passes_gate=bool(d0[0] <= -0.005 and d0[2] < 0 and d1[0] <= 0)))
    w(pd.DataFrame(rows).to_string(index=False))
    res['gate'] = rows

    # 2. per group
    w("\n2. PER GROUP (out-of-sample Brier; AUC of raw shows whether the sim ranks winners)")
    key = ['M0 raw', 'M1 base rate', 'M2 Platt pooled', 'M3 Platt per direction', 'Mgi slope common, group intercepts',
           'M4 per group shrunk', 'M4u per group unpooled']
    tg = table(oos, key, by='group')
    piv = tg.pivot(index='group', columns='model', values='brier')[key]
    piv['n'] = tg.groupby('group').n.first()
    piv['win%'] = (tg.groupby('group').win.first() * 100).round(1)
    piv['AUC raw'] = tg[tg.model == 'M0 raw'].set_index('group').auc
    w(piv.round(4).to_string())
    w("\n   Per-group Brier difference vs M0 (95% CI by date) for M2, Mgi and M4")
    rows = []
    for g, d in oos.groupby('group'):
        r = {'group': g}
        for name in ('M2 Platt pooled', 'Mgi slope common, group intercepts', 'M4 per group shrunk'):
            m, lo, hi = boot_diff(d, name, 'M0 raw', n=500)
            r[name.split()[0]] = f"{m:+.4f} [{lo:+.4f},{hi:+.4f}]"
        rows.append(r)
    w(pd.DataFrame(rows).to_string(index=False))
    res['per_group'] = rows

    # 3. reliability
    w("\n3. RELIABILITY on R test days (stated vs actual win %)")
    for name in ('M0 raw', 'M2 Platt pooled', 'Mgi slope common, group intercepts', 'M4 per group shrunk'):
        w(f"   {name}")
        w(reliability(oos.y.to_numpy(), oos[name].to_numpy()).to_string(index=False))
    w("\n   Raw share 60-80% only (the band near the gates)")
    band = oos[(oos.p_raw >= .6) & (oos.p_raw < .8)]
    w(table(band, ['M0 raw', 'M2 Platt pooled', 'Mgi slope common, group intercepts', 'M4 per group shrunk'])
      .drop(columns='group').to_string(index=False))

    # 4. training window length
    w("\n4. TRAINING WINDOW LENGTH (same test days)")
    sub = {k: models()[k] for k in ('M0 raw', 'M2 Platt pooled', 'Mgi slope common, group intercepts',
                                    'M4 per group shrunk')}
    rows = []
    for win in (None, 10, 7, 5):
        o, _, _ = rolling(R, window=win, model_set=sub)
        for name in sub:
            rows.append(dict(window='expanding' if win is None else f'last {win} days', model=name,
                             brier=round(brier(o.y.to_numpy(), o[name].to_numpy()), 4),
                             ece=round(ece(o.y.to_numpy(), o[name].to_numpy()), 4)))
    w(pd.DataFrame(rows).pivot(index='model', columns='window', values='brier').to_string())

    # 5. hindsight-sensitive rows (minutes risk) and line source
    w("\n5. ROBUSTNESS")
    noMR = R[~R.gate.astype(str).str.startswith('MINUTES_RISK')]
    o, _, _ = rolling(noMR, model_set=sub)
    w(f"   a) Excluding MINUTES_RISK rows (hindsight-sensitive), n test={len(o):,}")
    w(table(o, list(sub)).drop(columns='group')[['model', 'brier', 'ece', 'auc', 'top20']].to_string(index=False))
    w("   b) By line source on the main test rows (real odds = historical_odds_cache)")
    w(table(oos, list(sub), by='real_odds')[['group', 'model', 'n', 'brier', 'ece', 'mean_p', 'win']].to_string(index=False))
    w("   c) Test fraction 1/2 instead of 1/3")
    o, _, td = rolling(R, test_frac=0.5, model_set=sub)
    w(table(o, list(sub)).drop(columns='group')[['model', 'n', 'brier', 'ece']].to_string(index=False))

    # 6. coefficient stability
    w("\n6. COEFFICIENT STABILITY across folds")
    cf = pd.DataFrame(coefs)
    m2 = cf[cf.a.notna()] if 'a' in cf else pd.DataFrame()
    if not m2.empty:
        w(f"   M2: a mean {m2.a.mean():+.3f} sd {m2.a.std():.3f} | b mean {m2.b.mean():+.3f} sd {m2.b.std():.3f}")
    m4 = cf[cf.get('lam').notna()] if 'lam' in cf else pd.DataFrame()
    if not m4.empty:
        w(f"   M4 lambda chosen per fold: {m4.lam.tolist()}")
        w(m4.drop(columns=[c for c in m4.columns if c in ('a', 'b')]).dropna(axis=1, how='all')
          .describe().loc[['mean', 'std']].round(3).T.to_string())

    # 7. engine consistency: fit on all of R, score L and O
    w("\n7. ENGINE CONSISTENCY — fit on ALL of R (backtest), score live data")
    full = {k: models()[k]().fit(R) for k in sub}
    for nm, D in (('L live, current engine', Lv), ('O live, old engine', O)):
        if D.empty:
            w(f"   {nm}: no rows"); continue
        D = D.copy()
        for k, m in full.items():
            D[k] = m.predict(D)
        w(f"   {nm}: n={len(D):,}, dates {D.date.min()}..{D.date.max()}")
        w(table(D, list(sub)).drop(columns='group')[['model', 'brier', 'logloss', 'ece', 'auc', 'mean_p', 'win']]
          .to_string(index=False))
        if D.date.nunique() >= 2:
            for k in list(sub)[1:]:
                m, lo, hi = boot_diff(D, k, 'M0 raw', n=500)
                w(f"      {k} vs raw: {m:+.4f} [{lo:+.4f},{hi:+.4f}]")
        w(reliability(D.y.to_numpy(), D['M2 Platt pooled'].to_numpy()).rename(
            columns={'stated': 'M2 stated'}).to_string(index=False))
        w(f"   per group (actual win % vs M0 / M4 stated):")
        gg = D.groupby('group').agg(n=('y', 'size'), win=('y', 'mean'), raw=('M0 raw', 'mean'),
                                     m4=('M4 per group shrunk', 'mean'))
        w((gg.assign(win=gg.win * 100, raw=gg.raw * 100, m4=gg.m4 * 100)).round(1).to_string())
    res['final_coefs_on_R'] = {k: (m.coefs() if hasattr(m, 'coefs') else (m.m.coefs() if hasattr(m, 'm') and
                                    hasattr(m.m, 'coefs') else None)) for k, m in full.items()}
    if hasattr(full['M4 per group shrunk'], 'lam'):
        res['final_M4_lambda'] = full['M4 per group shrunk'].lam

    # 8. economic sanity (BACKTEST, real-odds rows only)
    w("\n8. ECONOMIC SANITY — BACKTEST, real-odds rows of the test days only; flat 1 unit per pick")
    ro = oos[oos.real_odds & oos.odds.notna() & (oos.odds > 1.01)].copy()
    rows = []
    for name in ('M0 raw', 'M2 Platt pooled', 'Mgi slope common, group intercepts', 'M4 per group shrunk'):
        for margin in (0.0, 0.02, 0.05):
            sel = ro[ro[name] > 1 / ro.odds + margin]
            if len(sel):
                profit = np.where(sel.y == 1, sel.odds - 1, -1).sum()
                rows.append(dict(model=name, margin=margin, picks=len(sel), win=round(sel.y.mean() * 100, 1),
                                 roi=round(profit / len(sel) * 100, 1)))
            else:
                rows.append(dict(model=name, margin=margin, picks=0, win=None, roi=None))
    w(pd.DataFrame(rows).to_string(index=False))
    res['economic'] = rows

    # 9. economic sanity with uncertainty, and against selectors that ignore the sim
    w("\n9. ECONOMIC SANITY WITH 95% CI (bootstrap by date) — BACKTEST, real-odds test rows, margin 0.02")
    w("   'all rows' bets every real-odds candidate; 'M1 base rate' ignores the sim (odds + group only).")
    rng = np.random.default_rng(11)
    ro['all rows'] = 1.0
    rows = []
    for name in ('all rows', 'M0 raw', 'M1 base rate', 'M2 Platt pooled', 'Mgi slope common, group intercepts',
                 'M4 per group shrunk'):
        sel = ro if name == 'all rows' else ro[ro[name] > 1 / ro.odds + 0.02]
        if sel.empty:
            rows.append(dict(selector=name, picks=0)); continue
        sel = sel.assign(pl=np.where(sel.y == 1, sel.odds - 1, -1.0))
        by = {d: g.pl.to_numpy() for d, g in sel.groupby('date')}
        keys = list(by)
        boots = []
        for _ in range(1000):
            s = np.concatenate([by[k] for k in rng.choice(keys, len(keys))])
            boots.append(s.mean() * 100)
        rows.append(dict(selector=name, picks=len(sel), win=round(sel.y.mean() * 100, 1),
                         avg_odds=round(sel.odds.mean(), 3), roi=round(sel.pl.mean() * 100, 1),
                         roi_ci=f"[{np.percentile(boots, 2.5):+.1f}, {np.percentile(boots, 97.5):+.1f}]"))
    w(pd.DataFrame(rows).to_string(index=False))
    res['economic_ci'] = rows

    # 10. discrimination: does anything the sim produces rank winners?
    w("\n10. DISCRIMINATION (AUC; 0.50 = no better than a coin). Signals: raw share, and the projection gap")
    w("    gap = (projection - line)/line, signed so that positive favours the pick's direction")
    rows = []
    for nm, D in (('R backtest', R), ('L live current', Lv), ('O live old', O)):
        D = D[D.projection.notna() & (D.line > 0)].copy()
        D['gap'] = np.where(D.dir == 'OVER', 1, -1) * (D.projection - D.line) / D.line
        for g, d in [('ALL', D)] + list(D.groupby('group')):
            rows.append(dict(source=nm, group=g, n=len(d), auc_raw=round(auc(d.y.to_numpy(), d.p_raw.to_numpy()), 3),
                             auc_gap=round(auc(d.y.to_numpy(), d.gap.to_numpy()), 3)))
    t10 = pd.DataFrame(rows)
    w(t10.pivot(index='group', columns='source', values='auc_raw').round(3).add_prefix('raw | ').join(
        t10.pivot(index='group', columns='source', values='auc_gap').round(3).add_prefix('gap | ')).to_string())
    w("    AUC 95% CI on R (bootstrap by date):")
    D = R[R.projection.notna()].copy()
    by = {d: g for d, g in D.groupby('date')}
    keys = list(by)
    for g in ['ALL'] + GROUPS:
        boots = []
        for _ in range(500):
            s = pd.concat([by[k] for k in rng.choice(keys, len(keys))])
            if g != 'ALL':
                s = s[s.group == g]
            boots.append(auc(s.y.to_numpy(), s.p_raw.to_numpy()))
        w(f"      {g:15} raw AUC 95% CI [{np.nanpercentile(boots, 2.5):.3f}, {np.nanpercentile(boots, 97.5):.3f}]")
    res['discrimination'] = rows

    with open(os.path.join(args.out_dir, 'fit_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L_) + '\n')
    with open(os.path.join(args.out_dir, 'fit_results.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\n[fit] report written to {args.out_dir}")
    if args.write:
        write_params(R)


# Engine the R replay ran (replay_step2 --compare step3): calibration is only valid for it.
R_ENGINE = {'SHRINKAGE_MODEL': 'fitted', 'ENGINE_CALIBRATION': 'v2', 'B2B_MODEL': 'fitted',
            'RETURNING_MODEL': 'fitted'}
PARAMS_PATH = os.path.join(MODEL_DIR, 'config', 'prob_calibration.json')


def write_params(R, out_path=None):
    """Final M4 on all of R (BACKTEST). L is not merged: R already contains the same games (DV4)."""
    global PARAMS_PATH
    if out_path:
        PARAMS_PATH = out_path
    m = M4().fit(R)
    c = m.m.coefs()
    groups = {g: {'a_offset': round(c.get(f'a_{g}', 0.0), 6), 'b_offset': round(c.get(f'b_{g}', 0.0), 6),
                  'n': int((R.group == g).sum()), 'win_rate': round(float(R[R.group == g].y.mean()), 4)}
              for g in GROUPS}
    params = {
        'schema': 2, 'model': 'M4', 'input': 'logit(raw_sim_freq/100), clipped to [0.005, 0.995]',
        'formula': 'cal = sigmoid(a + a_offset + (b + b_offset) * x)',
        'computed_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'source': 'scripts/fit_calibration.py (Plans/STEP3_CALIBRATION_AUDIT.md)',
        'data': {'set': 'R = BACKTEST replay of the current engine (hindsight lineups)',
                 'file': os.path.relpath(R_CSV, MODEL_DIR), 'dates': [R.date.min(), R.date.max()],
                 'rows': int(len(R))},
        'engine': R_ENGINE, 'lambda': m.lam,
        'global': {'a': round(c['a'], 6), 'b': round(c['b'], 6)}, 'groups': groups,
    }
    if os.path.exists(PARAMS_PATH) and not out_path:
        arch = os.path.join(MODEL_DIR, 'config', 'archive')
        os.makedirs(arch, exist_ok=True)
        import shutil
        shutil.copy2(PARAMS_PATH, os.path.join(arch, f"prob_calibration_{datetime.now():%Y%m%d_%H%M%S}.json"))
    with open(PARAMS_PATH, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(params, f, indent=2)
        f.write('\n')
    print(f"[fit] wrote {PARAMS_PATH} (lambda {m.lam})")
    return params


if __name__ == '__main__':
    main()
