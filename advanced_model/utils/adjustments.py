"""
utils/adjustments.py — data-driven back-to-back (B2B) and returning-player adjustments.

Step 2 — Plans/STEP2_IMPLEMENTATION_AUDIT.md §4.2 and §5.2.
Parameters come from config/adjustment_params.json, written by scripts/fit_adjustments.py
(re-run once a season). No effect sizes are hard-coded here.

How the simulator uses a player matrix: possessions/rebounds/assists/steals are handed
out in proportion to avg_fga / avg_reb / avg_ast / avg_stl within a team (shares are
normalised), so "more minutes" = a larger share; teammates' shares shrink automatically.
"""

import json
import os
from typing import Optional

import config.settings as settings

# Matrix fields that represent a player's share of team events (scaled by minutes).
SHARE_KEYS = ('avg_fga', 'avg_reb', 'avg_ast', 'avg_stl', 'avg_blk', 'avg_pts')
# Per-minute effect applied on top of minutes: matrix key -> fitted per-minute stat
PER_MINUTE_KEYS = {'avg_fga': 'fga', 'avg_pts': 'fga', 'avg_reb': 'reb', 'avg_ast': 'ast'}
MINUTES_FACTOR_CLIP = (0.3, 1.5)

_CACHE = {'path': None, 'mtime': None, 'params': None}


def load_params(path: Optional[str] = None) -> Optional[dict]:
    """Load adjustment params (cached; reloads if the file changes). None if missing/unreadable."""
    path = path or settings.ADJUSTMENT_PARAMS_PATH
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    if _CACHE['path'] == path and _CACHE['mtime'] == mtime:
        return _CACHE['params']
    try:
        with open(path, encoding='utf-8') as f:
            params = json.load(f)
    except (OSError, ValueError):
        params = None
    _CACHE.update(path=path, mtime=mtime, params=params)
    return params


def _role(value: float, bounds: list, roles: list) -> str:
    for b, r in zip(bounds, roles):
        if value < b:
            return r
    return roles[-1]


def _scale_shares(matrix: dict, factor: float) -> None:
    for k in SHARE_KEYS:
        if k in matrix and matrix[k] is not None:
            matrix[k] *= factor


# ── Back-to-back ──────────────────────────────────────────────────────────
def b2b_role(min_mean: float, params: dict) -> str:
    return _role(min_mean, params['b2b']['role_bounds'], ['bench', 'rotation', 'starter'])


def apply_b2b_adjustment(matrix: dict, params: Optional[dict] = None) -> dict:
    """
    2nd night of a back-to-back, measured effects per role (from min_mean):
      minutes  : min_mean += minutes_delta[role]; share stats x (new / old minutes)
      turnovers: tov_rate x tov_per_min_ratio[role]
      shooting : unchanged unless the JSON marks fg2/fg3 apply=true
    No-op (with the matrix untouched) if params are missing.
    """
    params = params if params is not None else load_params()
    if not params or 'b2b' not in params:
        return matrix
    b = params['b2b']
    mins = float(matrix.get('min_mean') or 0.0)
    if mins <= 0:
        return matrix
    role = b2b_role(mins, params)
    eff = b['roles'][role]
    apply = b.get('apply', {})
    delta = eff['minutes_delta']['value'] if apply.get('minutes_delta', True) else 0.0
    new_mins = max(mins + delta, 1.0)
    _scale_shares(matrix, new_mins / mins)
    matrix['min_mean'] = new_mins
    if apply.get('tov_per_min_ratio', True) and 'tov_rate' in matrix:
        matrix['tov_rate'] *= eff['tov_per_min_ratio']['value']
    if apply.get('fg2_ratio') and 'fg2_pct' in matrix:
        matrix['fg2_pct'] *= eff['fg2_ratio']['value']
    if apply.get('fg3_ratio') and 'fg3_pct' in matrix:
        matrix['fg3_pct'] *= eff['fg3_ratio']['value']
    matrix['b2b_adjustment'] = {'role': role, 'minutes_delta': delta}
    return matrix


# ── Returning players ─────────────────────────────────────────────────────
def absence_bucket(games_missed: int, params: dict) -> str:
    for lo, hi, label in params['returning']['absence_buckets']:
        if lo <= games_missed <= hi:
            return label
    raise ValueError(f"games_missed must be >= 1, got {games_missed}")


def return_role(baseline_min: float, params: dict) -> str:
    r = params['returning']
    return _role(baseline_min, r['role_bounds'], r['roles'])


def expected_return_minutes(info: dict, params: dict) -> float:
    """baseline (pre-absence) minutes x fitted absence effect[role][bucket][return game]."""
    r = params['returning']
    role = return_role(info['baseline_min'], params)
    bucket = absence_bucket(info['games_missed'], params)
    return info['baseline_min'] * r['minutes_effect'][role][bucket][str(info['return_game'])]


def apply_return_adjustment(matrix: dict, info: dict, params: Optional[dict] = None) -> dict:
    """
    info = {'games_missed': k>=1, 'return_game': 1..3, 'baseline_min': float}
    (from live_ingestion.evaluate_returning_player).

    expected = baseline x effect; share stats x (expected / matrix min_mean) [clipped],
    then the per-minute effect on FGA/PTS, REB, AST. min_mean becomes the expected
    minutes (the MINUTES_RISK gate sees them). minutes_restriction_flag is set on return
    game 1 only (same scope as before: UNDER ban + DB history ignored on the first game).
    """
    params = params if params is not None else load_params()
    if not params or 'returning' not in params or not info:
        return matrix
    mins = float(matrix.get('min_mean') or 0.0)
    if mins <= 0 or info.get('baseline_min', 0) <= 0:
        return matrix
    expected = expected_return_minutes(info, params)
    factor = min(max(expected / mins, MINUTES_FACTOR_CLIP[0]), MINUTES_FACTOR_CLIP[1])
    _scale_shares(matrix, factor)
    bucket = absence_bucket(info['games_missed'], params)
    pm = params['returning']['per_minute_ratio']
    n = str(info['return_game'])
    for key, stat in PER_MINUTE_KEYS.items():
        if key in matrix and matrix[key] is not None:
            matrix[key] *= pm[stat][bucket][n]['value']
    matrix['min_mean_baseline'] = mins
    matrix['min_mean'] = mins * factor
    matrix['return_info'] = dict(info, bucket=bucket, expected_min=round(expected, 2))
    if info['return_game'] == 1:
        matrix['minutes_restriction_flag'] = f"RETURNING_G1_{bucket}"
    return matrix
