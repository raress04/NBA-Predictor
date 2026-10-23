"""
Centralized path builder for prediction, result, and analysis reports/logs.
Supports live vs backtest mode and season/month hierarchical directories:
  predictions/
  └── <season>/
      ├── live_predictions/
      │   └── <month>/
      │       ├── predicts_<date>.txt
      │       └── predicts_report_<date>.pdf
      └── backtest_predictions/
          └── <month>/
              ├── predicts_<date>.txt
              └── predicts_report_<date>.pdf

  results/
  └── <season>/
      ├── live_results/
      │   └── <month>/
      │       └── yesterday_results_<date>.txt
      └── backtest_results/
          └── <month>/
              └── yesterday_results_<date>.txt

  analyze/
  └── <season>/
      ├── live_predictions/
      │   └── <month>/
      │       └── analyze_<date>.txt
      └── backtest_predictions/
          └── <month>/
              └── analyze_<date>.txt
"""

import os
from datetime import datetime
from typing import Optional


def get_nba_season(date_val) -> str:
    """
    Derives NBA season string (e.g. '2026-2027') from a date string or datetime.
    NBA season starts in October and ends the following summer.
    Months 10, 11, 12 -> year to year+1 (e.g. Oct 2026 -> 2026-2027)
    Months 1..9       -> year-1 to year (e.g. Apr 2027 -> 2026-2027)
    """
    if isinstance(date_val, str):
        dt = datetime.strptime(date_val[:10], '%Y-%m-%d')
    elif isinstance(date_val, datetime):
        dt = date_val
    else:
        raise ValueError(f"Invalid date_val: {date_val}")

    if dt.month >= 10:
        return f"{dt.year}-{dt.year + 1}"
    else:
        return f"{dt.year - 1}-{dt.year}"


# ── Predictions Paths ─────────────────────────────────────────────────────────

def get_prediction_dir(date_val, mode: str = 'live', base_dir: Optional[str] = None) -> str:
    """Returns <base_dir>/predictions/<season>/<mode>_predictions/<month>/"""
    if mode not in ('live', 'backtest'):
        raise ValueError(f"Mode must be 'live' or 'backtest', got: {mode}")

    if isinstance(date_val, str):
        dt = datetime.strptime(date_val[:10], '%Y-%m-%d')
    elif isinstance(date_val, datetime):
        dt = date_val
    else:
        raise ValueError(f"Invalid date_val: {date_val}")

    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    season = get_nba_season(dt)
    month = dt.strftime('%B').lower()
    mode_folder = f"{mode}_predictions"

    target_dir = os.path.join(base_dir, 'predictions', season, mode_folder, month)
    return target_dir


def get_prediction_filepath(date_val, mode: str = 'live', ext: str = 'txt', base_dir: Optional[str] = None) -> str:
    folder = get_prediction_dir(date_val, mode=mode, base_dir=base_dir)
    d_str = date_val[:10] if isinstance(date_val, str) else date_val.strftime('%Y-%m-%d')
    return os.path.join(folder, f"predicts_{d_str}.{ext.lstrip('.')}")


def get_report_filepath(date_val, mode: str = 'live', ext: str = 'pdf', base_dir: Optional[str] = None) -> str:
    folder = get_prediction_dir(date_val, mode=mode, base_dir=base_dir)
    d_str = date_val[:10] if isinstance(date_val, str) else date_val.strftime('%Y-%m-%d')
    return os.path.join(folder, f"predicts_report_{d_str}.{ext.lstrip('.')}")


def resolve_prediction_file(
    date_val,
    mode: str,
    base_dir: Optional[str] = None,
    allow_legacy: bool = False
) -> Optional[str]:
    """Finds a prediction txt file for date_val and mode ('live' or 'backtest')."""
    if mode not in ('live', 'backtest'):
        raise ValueError(f"Explicit mode ('live' or 'backtest') required, got: {mode}")

    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if isinstance(date_val, str):
        d_str = date_val[:10]
        dt = datetime.strptime(d_str, '%Y-%m-%d')
    elif isinstance(date_val, datetime):
        dt = date_val
        d_str = dt.strftime('%Y-%m-%d')
    else:
        return None

    # Primary check: predictions/<season>/<mode>_predictions/<month>/predicts_<date>.txt
    p = get_prediction_filepath(dt, mode=mode, ext='txt', base_dir=base_dir)
    if os.path.exists(p):
        return p

    # Secondary check without month folder
    season = get_nba_season(dt)
    alt_p = os.path.join(base_dir, 'predictions', season, f"{mode}_predictions", f"predicts_{d_str}.txt")
    if os.path.exists(alt_p):
        return alt_p

    if not allow_legacy:
        return None

    # Legacy fallback for older seasons
    month = dt.strftime('%B').lower()
    legacy_candidates = [
        os.path.join(base_dir, 'predictions', season, f"{mode}_new_gen_prediction", month, f"predicts_{d_str}.txt"),
        os.path.join(base_dir, 'predictions', season, "first_gen_prediction", month, f"predicts_{d_str}.txt"),
        os.path.join(base_dir, 'predictions', month, f"predicts_{d_str}.txt"),
    ]
    for c in legacy_candidates:
        if os.path.exists(c):
            return c

    return None


# ── Results Paths ────────────────────────────────────────────────────────────

def get_results_dir(date_val, mode: str = 'live', base_dir: Optional[str] = None) -> str:
    """Returns <base_dir>/results/<season>/<mode>_results/<month>/"""
    if mode not in ('live', 'backtest'):
        raise ValueError(f"Mode must be 'live' or 'backtest', got: {mode}")

    if isinstance(date_val, str):
        dt = datetime.strptime(date_val[:10], '%Y-%m-%d')
    elif isinstance(date_val, datetime):
        dt = date_val
    else:
        raise ValueError(f"Invalid date_val: {date_val}")

    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    season = get_nba_season(dt)
    month = dt.strftime('%B').lower()
    mode_folder = f"{mode}_results"

    target_dir = os.path.join(base_dir, 'results', season, mode_folder, month)
    return target_dir


def get_results_filepath(date_val, mode: str = 'live', ext: str = 'txt', base_dir: Optional[str] = None) -> str:
    folder = get_results_dir(date_val, mode=mode, base_dir=base_dir)
    d_str = date_val[:10] if isinstance(date_val, str) else date_val.strftime('%Y-%m-%d')
    return os.path.join(folder, f"yesterday_results_{d_str}.{ext.lstrip('.')}")


def resolve_results_file(
    date_val,
    mode: str = 'live',
    base_dir: Optional[str] = None,
    allow_legacy: bool = True
) -> Optional[str]:
    """Finds yesterday_results_<date>.txt across new hierarchy and legacy 2025-2026 folders."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if isinstance(date_val, str):
        d_str = date_val[:10]
        dt = datetime.strptime(d_str, '%Y-%m-%d')
    elif isinstance(date_val, datetime):
        dt = date_val
        d_str = dt.strftime('%Y-%m-%d')
    else:
        return None

    # Primary check: results/<season>/<mode>_results/<month>/yesterday_results_<date>.txt
    p = get_results_filepath(dt, mode=mode, ext='txt', base_dir=base_dir)
    if os.path.exists(p):
        return p

    # Also check without mode prefix in 2026-2027 if saved simply under results/<season>/<month>/
    season = get_nba_season(dt)
    month = dt.strftime('%B').lower()
    simple_p = os.path.join(base_dir, 'results', season, month, f"yesterday_results_{d_str}.txt")
    if os.path.exists(simple_p):
        return simple_p

    if not allow_legacy:
        return None

    legacy_candidates = [
        os.path.join(base_dir, 'results', season, month, f"yesterday_results_{d_str}.txt"),
        os.path.join(base_dir, 'results', month, f"yesterday_results_{d_str}.txt"),
        os.path.join(base_dir, f"yesterday_results_{d_str}.txt"),
        os.path.join(base_dir, "yesterday_results.txt"),
    ]
    for c in legacy_candidates:
        if os.path.exists(c):
            return c

    return None


# ── Analyze Paths ────────────────────────────────────────────────────────────

def get_analyze_dir(date_val, mode: str = 'live', base_dir: Optional[str] = None) -> str:
    """Returns <base_dir>/analyze/<season>/<mode>_predictions/<month>/"""
    if mode not in ('live', 'backtest'):
        raise ValueError(f"Mode must be 'live' or 'backtest', got: {mode}")

    if isinstance(date_val, str):
        dt = datetime.strptime(date_val[:10], '%Y-%m-%d')
    elif isinstance(date_val, datetime):
        dt = date_val
    else:
        raise ValueError(f"Invalid date_val: {date_val}")

    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    season = get_nba_season(dt)
    month = dt.strftime('%B').lower()
    mode_folder = f"{mode}_predictions"

    target_dir = os.path.join(base_dir, 'analyze', season, mode_folder, month)
    return target_dir


def get_analyze_filepath(date_val, mode: str = 'live', ext: str = 'txt', base_dir: Optional[str] = None) -> str:
    folder = get_analyze_dir(date_val, mode=mode, base_dir=base_dir)
    d_str = date_val[:10] if isinstance(date_val, str) else date_val.strftime('%Y-%m-%d')
    return os.path.join(folder, f"analyze_{d_str}.{ext.lstrip('.')}")


def get_analyze_report_filepath(date_val, mode: str = 'live', ext: str = 'pdf', base_dir: Optional[str] = None) -> str:
    folder = get_analyze_dir(date_val, mode=mode, base_dir=base_dir)
    d_str = date_val[:10] if isinstance(date_val, str) else date_val.strftime('%Y-%m-%d')
    return os.path.join(folder, f"analyze_report_{d_str}.{ext.lstrip('.')}")


def resolve_analyze_file(
    date_val,
    mode: str = 'live',
    base_dir: Optional[str] = None,
    allow_legacy: bool = True
) -> Optional[str]:
    """Finds analyze_report_<date>.pdf, analyze_<date>.pdf, or analyze_<date>.txt across new hierarchy and legacy folders."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if isinstance(date_val, str):
        d_str = date_val[:10]
        dt = datetime.strptime(d_str, '%Y-%m-%d')
    elif isinstance(date_val, datetime):
        dt = date_val
        d_str = dt.strftime('%Y-%m-%d')
    else:
        return None

    # Primary check: analyze_report_<date>.pdf in analyze/<season>/<mode>_predictions/<month>/
    p_report = get_analyze_report_filepath(dt, mode=mode, ext='pdf', base_dir=base_dir)
    if os.path.exists(p_report):
        return p_report

    # Secondary check: analyze_<date>.pdf
    p_pdf = get_analyze_filepath(dt, mode=mode, ext='pdf', base_dir=base_dir)
    if os.path.exists(p_pdf):
        return p_pdf

    # Tertiary check: analyze_<date>.txt
    p_txt = get_analyze_filepath(dt, mode=mode, ext='txt', base_dir=base_dir)
    if os.path.exists(p_txt):
        return p_txt

    if not allow_legacy:
        return None

    season = get_nba_season(dt)
    month = dt.strftime('%B').lower()
    legacy_candidates = [
        os.path.join(base_dir, 'analyze', season, month, f"analyze_report_{d_str}.pdf"),
        os.path.join(base_dir, 'analyze', season, month, f"analyze_{d_str}.pdf"),
        os.path.join(base_dir, 'analyze', season, month, f"analyze_{d_str}.txt"),
        os.path.join(base_dir, 'analyze', month, f"analyze_{d_str}.txt"),
        os.path.join(base_dir, f"analyze_{d_str}.txt"),
    ]
    for c in legacy_candidates:
        if os.path.exists(c):
            return c

    return None
