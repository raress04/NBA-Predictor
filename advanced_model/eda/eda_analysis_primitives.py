"""
eda_analysis_primitives.py
==========================
Pure-function analysis helpers for EDA.

Responsibilities:
  - Numeric / categorical column splitting.
  - Missingness computation.
  - Descriptive statistics.
  - Categorical frequency tables.
  - IQR-based outlier detection.
  - Correlation matrix computation.
  - Mutual information (classification) and Pearson/Spearman (regression).
  - Temporal period column addition.
  - Binned performance computation.

Rules:
  - No I/O in this module (no DB reads, no file writes, no plots).
  - All functions are pure: DataFrame in → dict/DataFrame out.
  - Never imported outside eda/.
"""

import re
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from eda.eda_config import (
    EDA_OUTLIER_IQR_FACTOR,
    EDA_HIGH_MISSING_THRESHOLD,
    EDA_HIGH_CORR_THRESHOLD,
    EDA_MAX_CATEGORICAL_LEVELS,
    EDA_N_DECILE_BINS,
)


# ── Phase 3.1 – Column type splitting ────────────────────────────────────────

def split_numeric_categorical(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    Split DataFrame columns into numeric and categorical lists.

    Returns
    -------
    (numeric_cols, categorical_cols)
    """
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns if c not in numeric]
    return numeric, categorical


# ── Phase 3.2 / 3.3 – Missingness ────────────────────────────────────────────

def compute_missingness(df: pd.DataFrame) -> List[Dict]:
    """
    Compute per-column missingness statistics.

    Returns
    -------
    List[dict] with keys: name, n_total, n_null, pct_null, high_missing
    """
    n_total = len(df)
    result = []
    for col in df.columns:
        n_null = int(df[col].isna().sum())
        pct    = n_null / n_total if n_total > 0 else 0.0
        result.append({
            'name':         col,
            'n_total':      n_total,
            'n_null':       n_null,
            'pct_null':     round(pct, 6),
            'high_missing': pct > EDA_HIGH_MISSING_THRESHOLD,
        })
    return result


def check_critical_columns_missing(
    missingness: List[Dict],
    critical_patterns: List[str],
) -> List[Dict]:
    """
    Assert that columns matching *critical_patterns* have zero nulls.

    Parameters
    ----------
    missingness       : output of compute_missingness()
    critical_patterns : list of exact names or regex patterns to match

    Returns
    -------
    List[dict] — issues found (empty list = all OK)
    """
    issues = []
    for entry in missingness:
        col = entry['name']
        for pat in critical_patterns:
            if re.fullmatch(pat, col, re.IGNORECASE):
                if entry['pct_null'] > 0:
                    issues.append({
                        'column':   col,
                        'pct_null': entry['pct_null'],
                        'n_null':   entry['n_null'],
                        'pattern':  pat,
                    })
    return issues


# ── Phase 3.4 – Descriptive statistics ───────────────────────────────────────

def compute_descriptive_stats(
    df: pd.DataFrame,
    numeric_cols: List[str],
) -> Dict[str, Dict]:
    """
    Compute summary statistics for each numeric column.

    Returns
    -------
    dict[col_name → {mean, median, std, min, max, p25, p75}]
    """
    stats = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(s) == 0:
            stats[col] = {k: None for k in ['mean','median','std','min','max','p25','p75','n']}
            continue
        stats[col] = {
            'n':      int(len(s)),
            'mean':   round(float(s.mean()),  6),
            'median': round(float(s.median()), 6),
            'std':    round(float(s.std()),   6),
            'min':    round(float(s.min()),   6),
            'max':    round(float(s.max()),   6),
            'p25':    round(float(s.quantile(0.25)), 6),
            'p75':    round(float(s.quantile(0.75)), 6),
        }
    return stats


# ── Phase 3.5 – Categorical frequencies ──────────────────────────────────────

def compute_value_counts(
    df: pd.DataFrame,
    col: str,
    max_levels: int = EDA_MAX_CATEGORICAL_LEVELS,
) -> List[Dict]:
    """
    Compute top-N value frequencies for a categorical column.

    Returns
    -------
    List[dict] with keys: value, count, pct
    """
    if col not in df.columns:
        return []
    vc  = df[col].value_counts(dropna=False).head(max_levels)
    n   = len(df)
    return [
        {'value': str(v), 'count': int(c), 'pct': round(c / n, 6)}
        for v, c in vc.items()
    ]


# ── Phase 4.1 – IQR outlier detection ────────────────────────────────────────

def compute_iqr_outliers(
    series: pd.Series,
    factor: float = EDA_OUTLIER_IQR_FACTOR,
) -> Dict:
    """
    Detect outliers using the Tukey IQR method.

    Returns
    -------
    dict with keys:
        q1, q3, iqr, lower_bound, upper_bound,
        n_outliers, pct_outliers, example_values (up to 10)
    """
    s = series.dropna()
    if len(s) == 0:
        return {k: None for k in [
            'q1','q3','iqr','lower_bound','upper_bound',
            'n_outliers','pct_outliers','example_values'
        ]}
    q1  = float(s.quantile(0.25))
    q3  = float(s.quantile(0.75))
    iqr = q3 - q1
    lo  = q1 - factor * iqr
    hi  = q3 + factor * iqr
    mask        = (s < lo) | (s > hi)
    n_outliers  = int(mask.sum())
    examples    = sorted(s[mask].tolist(), key=abs, reverse=True)[:10]
    return {
        'q1':           round(q1, 6),
        'q3':           round(q3, 6),
        'iqr':          round(iqr, 6),
        'lower_bound':  round(lo, 6),
        'upper_bound':  round(hi, 6),
        'n_outliers':   n_outliers,
        'pct_outliers': round(n_outliers / len(s), 6),
        'example_values': [round(v, 4) for v in examples],
    }


# ── Phase 4.2 – Odds integrity ────────────────────────────────────────────────

def check_odds_integrity(
    df: pd.DataFrame,
    odds_cols: List[str],
    odds_min: float = 1.01,
    odds_max: float = 15.0,
) -> List[Dict]:
    """
    Flag rows where odds values fall outside [odds_min, odds_max].

    Returns
    -------
    List[dict] — one entry per offending column with stats.
    """
    issues = []
    for col in odds_cols:
        if col not in df.columns:
            continue
        s          = pd.to_numeric(df[col], errors='coerce').dropna()
        below      = int((s < odds_min).sum())
        above      = int((s > odds_max).sum())
        total      = len(s)
        if below > 0 or above > 0:
            issues.append({
                'column':    col,
                'n_below':   below,
                'n_above':   above,
                'pct_below': round(below / total, 6) if total else 0,
                'pct_above': round(above / total, 6) if total else 0,
                'min_val':   round(float(s.min()), 4),
                'max_val':   round(float(s.max()), 4),
            })
    return issues


# ── Phase 5.1 – Correlation matrix ───────────────────────────────────────────

def compute_correlation_matrix(
    df: pd.DataFrame,
    cols: List[str],
) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix for the given numeric columns.

    Returns
    -------
    pd.DataFrame  (symmetric, index and columns = cols)
    """
    available = [c for c in cols if c in df.columns]
    return df[available].corr(method='pearson')


def find_high_corr_pairs(
    corr_df: pd.DataFrame,
    threshold: float = EDA_HIGH_CORR_THRESHOLD,
) -> List[Dict]:
    """
    Extract all (a, b) pairs with |corr| > threshold (a != b, no duplicates).

    Returns
    -------
    List[dict] with keys: col_a, col_b, correlation
    """
    pairs = []
    cols  = list(corr_df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_df.iloc[i, j]
            if not math.isnan(val) and abs(val) > threshold:
                pairs.append({
                    'col_a':       cols[i],
                    'col_b':       cols[j],
                    'correlation': round(float(val), 6),
                })
    return sorted(pairs, key=lambda x: abs(x['correlation']), reverse=True)


# ── Phase 5.3 – Mutual information (classification) ──────────────────────────

def compute_mutual_information(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
) -> Dict[str, float]:
    """
    Compute mutual information between each feature and a binary label.

    Returns
    -------
    dict[feature → MI score], sorted descending by score.
    """
    from sklearn.feature_selection import mutual_info_classif

    available = [f for f in features if f in df.columns and f != label_col]
    if label_col not in df.columns or not available:
        return {}

    sub = df[available + [label_col]].dropna()
    if len(sub) == 0:
        return {}

    X = sub[available].values
    y = sub[label_col].values.astype(int)
    scores = mutual_info_classif(X, y, random_state=42)
    result = {feat: round(float(s), 6) for feat, s in zip(available, scores)}
    return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))


# ── Phase 5.4 – Feature–target correlation (regression) ──────────────────────

def compute_feature_target_correlation(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    method: str = 'pearson',
) -> Dict[str, float]:
    """
    Compute Pearson or Spearman correlation between each feature and a target.

    Returns
    -------
    dict[feature → corr], sorted descending by |corr|.
    """
    available = [f for f in features if f in df.columns and f != target_col]
    if target_col not in df.columns or not available:
        return {}

    result = {}
    for feat in available:
        sub = df[[feat, target_col]].dropna()
        if len(sub) < 2:
            result[feat] = 0.0
            continue
        corr = sub[feat].corr(sub[target_col], method=method)
        result[feat] = round(float(corr) if not math.isnan(corr) else 0.0, 6)

    return dict(sorted(result.items(), key=lambda kv: abs(kv[1]), reverse=True))


# ── Phase 5.5 – Binned performance ───────────────────────────────────────────

def compute_binned_performance(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    n_bins: int = EDA_N_DECILE_BINS,
) -> List[Dict]:
    """
    Bin *feature_col* into *n_bins* quantile buckets and compute
    mean *target_col* per bin.

    Returns
    -------
    List[dict] with keys: bin_label, bin_min, bin_max, n, mean_target
    """
    if feature_col not in df.columns or target_col not in df.columns:
        return []

    sub = df[[feature_col, target_col]].dropna()
    if len(sub) < n_bins:
        return []

    sub = sub.copy()
    sub['_bin'] = pd.qcut(sub[feature_col], q=n_bins, duplicates='drop')

    result = []
    for interval, grp in sub.groupby('_bin', observed=True):
        result.append({
            'bin_label':   str(interval),
            'bin_min':     round(float(interval.left),  4),
            'bin_max':     round(float(interval.right), 4),
            'n':           int(len(grp)),
            'mean_target': round(float(grp[target_col].mean()), 6),
        })
    return result


# ── Phase 6.1 – Temporal period column ───────────────────────────────────────

def add_period_column(
    df: pd.DataFrame,
    date_col: str,
    freq: str = 'M',
) -> pd.DataFrame:
    """
    Add a 'period' column (e.g. '2026-01') derived from *date_col*.

    Parameters
    ----------
    df       : DataFrame (not mutated — copy returned)
    date_col : column with date-like values
    freq     : 'M' for month, 'W' for week, 'Q' for quarter

    Returns
    -------
    pd.DataFrame with new 'period' column (string).
    """
    df = df.copy()
    dt = pd.to_datetime(df[date_col], errors='coerce')
    if freq == 'M':
        df['period'] = dt.dt.to_period('M').astype(str)
    elif freq == 'W':
        df['period'] = dt.dt.to_period('W').astype(str)
    elif freq == 'Q':
        df['period'] = dt.dt.to_period('Q').astype(str)
    else:
        df['period'] = dt.dt.to_period(freq).astype(str)
    return df


def compute_period_stats(
    df: pd.DataFrame,
    numeric_cols: List[str],
    target_col: Optional[str] = None,
) -> List[Dict]:
    """
    Compute per-period aggregates (count + mean of numeric cols).
    Requires a 'period' column to be present (use add_period_column first).

    Returns
    -------
    List[dict] — one entry per period.
    """
    if 'period' not in df.columns:
        return []

    available = [c for c in numeric_cols if c in df.columns]
    result = []
    for period, grp in df.groupby('period', sort=True):
        entry: Dict = {'period': period, 'row_count': int(len(grp))}
        for col in available:
            s = pd.to_numeric(grp[col], errors='coerce').dropna()
            entry[f'mean_{col}'] = round(float(s.mean()), 6) if len(s) else None
        if target_col and target_col in grp.columns:
            s_tgt = pd.to_numeric(grp[target_col], errors='coerce').dropna()
            entry['target_rate'] = round(float(s_tgt.mean()), 6) if len(s_tgt) else None
        result.append(entry)
    return result


# ── Phase 7.3 – Class balance ─────────────────────────────────────────────────

def compute_class_balance(
    df: pd.DataFrame,
    label_col: str,
    group_cols: Optional[List[str]] = None,
) -> Dict:
    """
    Compute WIN/LOSS balance globally and per subgroup.

    Returns
    -------
    dict with keys: 'global' and each group in group_cols.
    """
    if label_col not in df.columns:
        return {}

    df = df.copy()
    df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
    sub = df[[label_col] + (group_cols or [])].dropna(subset=[label_col])
    n   = len(sub)

    def _balance(s):
        vc = s.value_counts(normalize=True)
        return {str(k): round(float(v), 6) for k, v in vc.items()}

    result: Dict = {
        'global': {
            'n':       n,
            'balance': _balance(sub[label_col]),
        }
    }

    for gcol in (group_cols or []):
        if gcol not in sub.columns:
            continue
        grp_stats = {}
        for val, grp in sub.groupby(gcol, sort=True):
            grp_stats[str(val)] = {
                'n':       int(len(grp)),
                'balance': _balance(grp[label_col]),
            }
        result[gcol] = grp_stats

    return result
