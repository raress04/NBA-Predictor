"""
Step 3 — calibrated win probability for prop candidates (Plans/STEP3_CALIBRATION_AUDIT.md).

Model M4 (fitted by scripts/fit_calibration.py, stored in config/prob_calibration.json):
    x = logit(clip(raw/100, 0.005, 0.995))       raw = direction-specific % of sims clearing the line
    z = a + b*x + a_g + b_g*x                    g = CATEGORY_DIRECTION (e.g. POINTS_OVER)
    cal_prob = 100 * sigmoid(z)

The parameters record the engine they were fitted on. If the running engine differs
(e.g. SHRINKAGE_MODEL=legacy), calibrate() returns None: a curve fitted on one engine's
raw shares is not valid for another's.
"""
import json
import math
import os

ENGINE_KEYS = ('SHRINKAGE_MODEL', 'ENGINE_CALIBRATION', 'B2B_MODEL', 'RETURNING_MODEL')
EPS = 0.005
_cache = {'path': None, 'mtime': None, 'params': None, 'warned': set()}


def _settings():
    import config.settings as s
    return s


def engine_signature(settings=None) -> dict:
    s = settings or _settings()
    return {k: getattr(s, k, None) for k in ENGINE_KEYS}


def load_params(path: str = None):
    path = path or getattr(_settings(), 'PROB_CALIBRATION_PATH', None)
    if not path or not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    if _cache['path'] != path or _cache['mtime'] != mtime:
        try:
            with open(path, encoding='utf-8') as f:
                params = json.load(f)
            if params.get('schema') != 2 or params.get('model') != 'M4':
                params = None
        except (OSError, ValueError):
            params = None
        _cache.update(path=path, mtime=mtime, params=params)
    return _cache['params']


def _warn_once(key, msg):
    if key not in _cache['warned']:
        _cache['warned'].add(key)
        print(msg)


def calibrate(category: str, direction: str, raw_pct, params: dict = None, settings=None):
    """Calibrated win probability in % for this candidate, or None if not available."""
    if raw_pct is None:
        return None
    params = params if params is not None else load_params()
    if not params:
        _warn_once('missing', "  [CAL] config/prob_calibration.json missing or invalid - cal_prob not computed.")
        return None
    sig = engine_signature(settings)
    if params.get('engine') != sig:
        _warn_once(('engine', tuple(sig.items())),
                   f"  [CAL] calibration fitted on engine {params.get('engine')}, running {sig} - cal_prob not computed.")
        return None
    p = min(max(float(raw_pct) / 100.0, EPS), 1 - EPS)
    x = math.log(p / (1 - p))
    g = params['groups'].get(f"{str(category).upper()}_{str(direction).upper()}")
    if g is None:            # not a fitted group (combos, totals, spreads)
        return None
    z = (params['global']['a'] + g.get('a_offset', 0.0)
         + (params['global']['b'] + g.get('b_offset', 0.0)) * x)
    z = max(min(z, 30.0), -30.0)
    return 100.0 / (1.0 + math.exp(-z))
