"""
Step 4 — fit the pick filters on cal_prob (Plans/STEP4_GATES_AUDIT.md, P3).

Read-only unless --write. Input: the BACKTEST candidate dump with every filter's outcome
(scripts/replay_step2.py --dump-candidates, Step 4 P1) and the LIVE shadow_picks rows.

  Core gate : cal_prob/100 > 1/odds + m          (m chosen on TUNE)
  Filters   : each existing filter is tested on top of the core gate; kept only if it
              raises ROI on TUNE with a bootstrap CI (by date) above 0
  Ranking   : EV = cal_prob/100 * odds - 1, daily cap chosen on TUNE
Rules are chosen on TUNE, frozen, then reported on TEST-B (backtest) and TEST-L (live).
Profit is judged on real-odds rows only (default 1.909 odds excluded).

Run from advanced_model/ (WSL):
    python3 -m scripts.fit_gates                      # report only
    python3 -m scripts.fit_gates --write              # also write config/gate_params.json
"""
import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime

import numpy as np
import pandas as pd

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from utils.prob_calibration import calibrate, load_params      # noqa: E402

CAND_CSV = os.path.join(MODEL_DIR, 'logs', 'november', 'step4_replay', 'candidates.csv')
BET_DB = os.path.join(MODEL_DIR, 'database', 'bet_tracker.db')
OUT_DIR = os.path.join(MODEL_DIR, 'logs', 'november', 'step4')
PARAMS_OUT = os.path.join(MODEL_DIR, 'config', 'gate_params.json')

TUNE = ('2026-10-20', '2026-11-08')
TEST_B = ('2026-11-10', '2026-11-17')
TEST_L = ('2026-11-09', '2026-11-17')
MARGINS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
CAPS = [3, 5, 10, 15, None]
FILTERS = ['f_min_gap', 'f_minutes', 'f_line_vol', 'f_stage1', 'f_variance', 'f_blowout', 'f_restriction']
LIVE_COMPUTABLE = {'f_min_gap', 'f_minutes', 'f_line_vol'}     # rebuildable from shadow_picks columns
MIN_GAP = {'POINTS': 2.0, 'REBOUNDS': 1.8, 'ASSISTS': 1.5}       # settings.LIVE_MIN_GAP
N_BOOT = 1000


# ── data ────────────────────────────────────────────────────────────────────
def _prep(df, params):
    df = df[df['cat'].isin(['POINTS', 'REBOUNDS', 'ASSISTS']) & df['raw'].notna()
            & df['win'].notna() & df['odds'].notna()].copy()
    df['real_odds'] = df['odds'].round(3) != 1.909
    df['cal'] = [calibrate(c, d, r, params=params) for c, d, r in zip(df.cat, df.direction, df.raw)]
    df = df[df['cal'].notna()].copy()
    df['p'] = df['cal'] / 100.0
    df['be'] = 1.0 / df['odds']
    df['edge_be'] = df['p'] - df['be']
    df['ev'] = df['p'] * df['odds'] - 1.0
    df['ret'] = df['win'] * df['odds'] - 1.0          # flat 1-unit stake profit
    df['group'] = df['cat'] + '_' + df['direction']
    if 'gap' not in df.columns:
        df['gap'] = np.where(df.direction == 'OVER', df.projection - df.line, df.line - df.projection)
    for f in FILTERS:
        if f in df.columns:
            df[f] = df[f].map(lambda v: v if isinstance(v, bool) else str(v) == 'True'
                              if pd.notna(v) else np.nan)
    return df


def load_backtest(params, path=CAND_CSV, odds_fix=None):
    c = pd.read_csv(path)
    if odds_fix:
        # Re-price with corrected Action Network odds (over/under were stored identical before
        # the 2026-11-19 scraper fix). Rows without a corrected price get the 1.909 default,
        # so they drop out of every profit number (real-odds rows only).
        f = pd.read_csv(odds_fix).rename(columns={'date_x': 'date', 'player_name': 'player',
                                                  'stat_category': 'cat'})
        f = f.drop_duplicates(['date', 'player', 'cat', 'line'])[['date', 'player', 'cat', 'line',
                                                                    'over_new', 'under_new']]
        c = c.merge(f, on=['date', 'player', 'cat', 'line'], how='left')
        fixed = np.where(c.direction == 'OVER', c.over_new, c.under_new)
        c['odds'] = np.where(pd.notna(fixed), fixed, 1.909)
    return _prep(c, params)


def load_live(params):
    conn = sqlite3.connect(f"file:{BET_DB}?mode=ro", uri=True)
    s = pd.read_sql("""
        SELECT game_date AS date, player, stat_category AS cat, UPPER(direction) AS direction, line,
               projection, raw_sim_freq AS raw, min_mean, min_std, ban_reason AS gate,
               actual_result AS win,
               CASE WHEN UPPER(direction)='OVER' THEN over_odds ELSE under_odds END AS odds
        FROM shadow_picks
        WHERE COALESCE(superseded,0)=0 AND actual_result IS NOT NULL AND model_version='ERA_3_STEP2'
          AND game_date >= ? AND game_date <= ?""", conn, params=TEST_L)
    conn.close()
    s['gap'] = np.where(s.direction == 'OVER', s.projection - s.line, s.line - s.projection)
    s['f_min_gap'] = s['gap'] >= s['cat'].map(MIN_GAP)
    s['f_line_vol'] = s['line'] >= 5.5
    s['f_minutes'] = ~((s['min_std'].fillna(0) > 6.5) | (s['min_mean'].fillna(30) < 18.0))
    return _prep(s, params)


def window(df, w):
    return df[(df.date >= w[0]) & (df.date <= w[1])]


# ── metrics ─────────────────────────────────────────────────────────────────
def stats(x):
    if len(x) == 0:
        return dict(picks=0, days=0, win=np.nan, roi=np.nan, be=np.nan, over=np.nan)
    return dict(picks=len(x), days=x.date.nunique(), win=x.win.mean() * 100, roi=x.ret.mean() * 100,
                be=x.be.mean() * 100, over=(x.direction == 'OVER').mean() * 100)


def boot_roi(x, n=N_BOOT, seed=11):
    """95% CI of ROI (%) resampling DATES."""
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    by = {d: g.ret.to_numpy() for d, g in x.groupby('date')}
    keys = list(by)
    vals = []
    for _ in range(n):
        r = np.concatenate([by[k] for k in rng.choice(keys, len(keys))])
        vals.append(r.mean() * 100 if len(r) else np.nan)
    return (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))


def boot_gain(base, mask, n=N_BOOT, seed=13):
    """95% CI of ROI(base & mask) - ROI(base), resampling dates."""
    rng = np.random.default_rng(seed)
    by = {d: (g.ret.to_numpy(), mask.loc[g.index].to_numpy(bool)) for d, g in base.groupby('date')}
    keys = list(by)
    vals = []
    for _ in range(n):
        pick = rng.choice(keys, len(keys))
        r = np.concatenate([by[k][0] for k in pick])
        m = np.concatenate([by[k][1] for k in pick])
        if m.sum() and len(r):
            vals.append((r[m].mean() - r.mean()) * 100)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def apply_rule(df, m, filters, cap):
    x = df[df.real_odds & (df.edge_be > m)]
    for f in filters:
        if f in x.columns:
            x = x[x[f].fillna(True).astype(bool)]
    if cap:
        x = x.sort_values(['date', 'ev'], ascending=[True, False]).groupby('date').head(cap)
    return x


def fmt(s, ci=None):
    t = (f"{s['picks']:>5} picks ({s['picks'] / max(s['days'], 1):.1f}/day) | win {s['win']:.1f}% "
         f"(break-even {s['be']:.1f}%) | ROI {s['roi']:+.1f}%")
    if ci:
        t += f" [{ci[0]:+.1f}, {ci[1]:+.1f}]"
    return t + f" | OVER {s['over']:.0f}%"


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cal-params', default=None, help='calibration params file (default: live M4)')
    ap.add_argument('--tune', nargs=2, default=TUNE, metavar=('START', 'END'))
    ap.add_argument('--tag', default='main', help='report name suffix')
    ap.add_argument('--candidates', default=CAND_CSV, help='BACKTEST candidate dump with filter columns')
    ap.add_argument('--odds-fix', default=None, help='CSV of corrected AN prices (over_new/under_new)')
    ap.add_argument('--write', action='store_true', help='write config/gate_params.json')
    args = ap.parse_args()

    params = load_params(args.cal_params) if args.cal_params else load_params()
    if not params:
        sys.exit('no calibration params')
    B = load_backtest(params, args.candidates, args.odds_fix)
    L = load_live(params)
    tune = window(B, tuple(args.tune))
    tb, tl = window(B, TEST_B), L
    lines = []
    w = lines.append
    w(f"STEP 4 — FIT GATES ON cal_prob  ({datetime.now():%Y-%m-%d %H:%M}, tag={args.tag})")
    w("BACKTEST = hindsight lineups. Profit on real-odds rows only. CIs: bootstrap by date.")
    w(f"Backtest odds: {'corrected (' + args.odds_fix + ')' if args.odds_fix else 'as stored in historical_odds_cache'}")
    w(f"Calibration: {args.cal_params or 'config/prob_calibration.json'} "
      f"(fit dates {params.get('data', {}).get('dates')})")
    w(f"TUNE {args.tune[0]}..{args.tune[1]}: {len(tune)} rows ({tune.real_odds.sum()} real odds, "
      f"{tune.date.nunique()} dates) | TEST-B {TEST_B}: {len(tb)} rows | TEST-L live {TEST_L}: {len(tl)} rows")

    # 0. today's rule for reference
    w("\n0. TODAY'S RULE (raw gates), real-odds rows")
    for name, x in [('TUNE', tune), ('TEST-B', tb)]:
        y = x[(x.gate == 'PASSED') & x.real_odds]
        w(f"  {name:7}: {fmt(stats(y), boot_roi(y))}")

    # 1. margin
    w("\n1. CORE GATE cal > break-even + m (no other filter), TUNE")
    rows = {}
    for m in MARGINS:
        x = apply_rule(tune, m, [], None)
        rows[m] = x
        w(f"  m={m * 100:.0f}%: {fmt(stats(x), boot_roi(x))}")
    best = max(MARGINS, key=lambda m: rows[m].ret.mean() if len(rows[m]) else -9)
    chosen_m = best
    dates = sorted(tune.date.unique())

    def per_date(x):
        g = x.groupby('date').ret
        return np.array([g.get_group(d).sum() if d in g.groups else 0.0 for d in dates]), \
            np.array([g.get_group(d).size if d in g.groups else 0 for d in dates])

    bs, bn = per_date(rows[best])
    rng = np.random.default_rng(17)
    idx = [rng.integers(0, len(dates), len(dates)) for _ in range(N_BOOT)]
    for m in MARGINS:                       # smallest m whose ROI is within noise of the best
        if m >= best:
            break
        ms, mn = per_date(rows[m])
        diff = [(ms[i].sum() / max(mn[i].sum(), 1) - bs[i].sum() / max(bn[i].sum(), 1)) * 100 for i in idx]
        if np.percentile(diff, 97.5) >= 0:
            chosen_m = m
            break
    w(f"  best ROI at m={best * 100:.0f}%; chosen (smallest within noise of best): m={chosen_m * 100:.0f}%")

    # 2. filters on top of the core gate
    w(f"\n2. FILTERS on top of the core gate (m={chosen_m * 100:.0f}%), TUNE — keep if ROI gain CI > 0")
    base = apply_rule(tune, chosen_m, [], None)
    kept = []
    w(f"  {'filter':14} {'keeps':>6} {'win%':>6} {'ROI':>7} | {'rejects':>7} {'win%':>6} {'ROI':>7} | gain [95% CI]")
    for f in FILTERS:
        if f not in base.columns:
            continue
        mask = base[f].fillna(True).astype(bool)
        k, r = base[mask], base[~mask]
        lo, hi = boot_gain(base, mask)
        gain = (k.ret.mean() - base.ret.mean()) * 100 if len(k) else np.nan
        keep = len(r) > 0 and lo > 0
        if keep:
            kept.append(f)
        w(f"  {f:14} {len(k):>6} {k.win.mean() * 100:>6.1f} {k.ret.mean() * 100:>+7.1f} | {len(r):>7} "
          f"{(r.win.mean() * 100 if len(r) else float('nan')):>6.1f} "
          f"{(r.ret.mean() * 100 if len(r) else float('nan')):>+7.1f} | {gain:+.1f} [{lo:+.1f}, {hi:+.1f}]"
          f"{'  KEEP' if keep else ''}")
    w(f"  kept: {kept or 'none'}")

    # MIN_GAP threshold grid (information)
    w("\n   MIN_GAP scale (x today's 2.0/1.8/1.5), on top of the core gate, TUNE:")
    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = base[base.gap >= base.cat.map(MIN_GAP) * s]
        w(f"   x{s:.2f}: {fmt(stats(x))}")

    # 3. cap
    w(f"\n3. DAILY CAP by EV rank (core gate + kept filters), TUNE")
    caps = {}
    for cap in CAPS:
        x = apply_rule(tune, chosen_m, kept, cap)
        caps[cap] = x
        w(f"  cap={cap or 'none':>4}: {fmt(stats(x), boot_roi(x))}")
    chosen_cap = None
    best_cap = max(CAPS, key=lambda c: caps[c].ret.mean() if len(caps[c]) else -9)
    blo, bhi = boot_roi(caps[best_cap])
    for cap in [None, 15, 10, 5, 3]:        # simplest = no cap, then the loosest cap within noise
        if len(caps[cap]) and caps[cap].ret.mean() * 100 >= blo:
            chosen_cap = cap
            break
    w(f"  chosen: cap={chosen_cap or 'none'} (loosest whose ROI is inside the best cap's CI)")

    # 4. frozen rule on TEST windows
    rule = dict(margin=chosen_m, filters=kept, cap=chosen_cap)
    w(f"\n4. FROZEN RULE {rule} on the TEST windows")
    for name, x in [('TUNE', tune), ('TEST-B', tb), ('TEST-L', tl)]:
        fl = kept if name != 'TEST-L' else [f for f in kept if f in LIVE_COMPUTABLE]
        y = apply_rule(x, chosen_m, fl, chosen_cap)
        s = stats(y)
        ok = s['picks'] > 0 and s['win'] >= s['be'] + 2 and s['roi'] > 0
        note = '' if fl == kept else f"  (live: filters {sorted(set(kept) - set(fl))} not logged, skipped)"
        w(f"  {name:7}: {fmt(s, boot_roi(y))}  {'PASS' if ok else 'FAIL'} (win >= BE+2 and ROI > 0){note}")
        if name != 'TUNE' and len(y):
            g = y.groupby('group').agg(n=('win', 'size'), win=('win', 'mean'), roi=('ret', 'mean'))
            g['win'] = (g.win * 100).round(1)
            g['roi'] = (g.roi * 100).round(1)
            w('    ' + g.to_string().replace('\n', '\n    '))
    w("  (TEST-L today's rule: see Season-Updates entry 44 — 16 live picks, 43.8%)")

    os.makedirs(OUT_DIR, exist_ok=True)
    rep = os.path.join(OUT_DIR, f'fit_report_{args.tag}.txt')
    with open(rep, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f"\n[+] report: {rep}")

    if args.write:
        out = dict(schema=1, gate_model='step4', margin=chosen_m, filters=kept, cap=chosen_cap,
                   rank='ev', computed_at=datetime.now().isoformat(timespec='seconds'),
                   source='scripts/fit_gates.py (Plans/STEP4_GATES_AUDIT.md)',
                   tune=list(args.tune), calibration=params.get('data'), engine=params.get('engine'),
                   n_tune=int(len(tune)))
        if os.path.exists(PARAMS_OUT):
            arch = os.path.join(MODEL_DIR, 'config', 'archive',
                                f"gate_params_{datetime.now():%Y%m%d_%H%M%S}.json")
            os.makedirs(os.path.dirname(arch), exist_ok=True)
            os.replace(PARAMS_OUT, arch)
        with open(PARAMS_OUT, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        print(f"[+] wrote {PARAMS_OUT}")


if __name__ == '__main__':
    main()
