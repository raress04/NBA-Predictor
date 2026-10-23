"""
eda_plots.py
============
All EDA visualisation lives here.

Responsibilities:
  - Histogram / KDE plots.
  - Correlation heatmap.
  - Binned performance plots (feature vs win-rate / model_bias).

Rules:
  - Never imported outside eda/.
  - All outputs are saved as PNG files to EDA_PLOTS_DIR.
  - No DataFrame mutation; operates on copies.
  - matplotlib backend is set to 'Agg' (non-interactive) to allow headless runs.
"""

import os
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
from typing import List, Optional

from eda.eda_config import EDA_PLOTS_DIR


def _ensure_plots_dir():
    """Create EDA_PLOTS_DIR if it does not exist."""
    os.makedirs(EDA_PLOTS_DIR, exist_ok=True)


def _save(fig: plt.Figure, filename: str):
    """Save figure to EDA_PLOTS_DIR and close it."""
    _ensure_plots_dir()
    path = os.path.join(EDA_PLOTS_DIR, filename)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[EDA][plot] saved → {path}")
    return path


# ── Phase 4.4 – Histogram / KDE ──────────────────────────────────────────────

def plot_histogram(
    series: pd.Series,
    title: str,
    filename: str,
    bins: int = 50,
    kde: bool = True,
    color: str = '#4C8EBF',
) -> str:
    """
    Plot a histogram (optionally with KDE overlay) for *series*.

    Parameters
    ----------
    series   : pd.Series   — numeric data
    title    : str         — plot title
    filename : str         — output PNG filename (basename only)
    bins     : int         — number of histogram bins
    kde      : bool        — overlay a KDE curve
    color    : str         — bar colour

    Returns
    -------
    str  — absolute path to saved PNG
    """
    s = series.dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(s, bins=bins, color=color, alpha=0.75, edgecolor='white', label='histogram')

    if kde and len(s) > 1:
        try:
            from scipy.stats import gaussian_kde
            xs  = np.linspace(s.min(), s.max(), 300)
            kde_fn = gaussian_kde(s)
            # Scale KDE to match histogram y-axis (density → count)
            bin_width = (s.max() - s.min()) / bins if s.max() != s.min() else 1
            ax2 = ax.twinx()
            ax2.plot(xs, kde_fn(xs), color='#E05C2A', linewidth=2, label='KDE')
            ax2.set_ylabel('Density', fontsize=9)
            ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
        except ImportError:
            pass  # scipy not installed — skip KDE silently

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(series.name or '', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return _save(fig, filename)


# ── Phase 5.1 – Correlation heatmap ──────────────────────────────────────────

def plot_corr_heatmap(
    corr_df: pd.DataFrame,
    filename: str = 'corr_heatmap.png',
    max_cols: int = 40,
) -> str:
    """
    Plot a Pearson correlation heatmap.

    Large matrices are truncated to *max_cols* columns for readability.

    Returns
    -------
    str — absolute path to saved PNG
    """
    if corr_df.empty:
        print("[EDA][plot] Skipping empty correlation matrix.")
        return ''

    # Truncate if too wide
    cols = list(corr_df.columns)[:max_cols]
    mat  = corr_df.loc[cols, cols]

    n = len(cols)
    size = max(8, min(20, n * 0.5))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))

    im = ax.imshow(mat.values, vmin=-1, vmax=1, cmap='RdYlGn', aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cols, rotation=90, fontsize=max(5, 9 - n // 8))
    ax.set_yticklabels(cols, fontsize=max(5, 9 - n // 8))

    # Annotate cells for small matrices
    if n <= 20:
        for i in range(n):
            for j in range(n):
                val = mat.values[i, j]
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color='black')

    ax.set_title('Feature Correlation Matrix (Pearson)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    return _save(fig, filename)


# ── Phase 5.5 – Binned performance plots ─────────────────────────────────────

def plot_binned_performance(
    bin_stats: List[dict],
    feature_col: str,
    target_col: str,
    filename: Optional[str] = None,
) -> str:
    """
    Bar chart of mean target value per feature bin.

    Parameters
    ----------
    bin_stats   : output of eda_analysis_primitives.compute_binned_performance()
    feature_col : name of the feature that was binned
    target_col  : name of the target (used in labels)
    filename    : PNG basename; auto-generated if None

    Returns
    -------
    str — absolute path to saved PNG
    """
    if not bin_stats:
        print(f"[EDA][plot] No bin stats for {feature_col}, skipping.")
        return ''

    labels    = [b['bin_label'] for b in bin_stats]
    means     = [b['mean_target'] for b in bin_stats]
    ns        = [b['n'] for b in bin_stats]
    x         = range(len(labels))

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(x, means, color='#5B8DB8', edgecolor='white', alpha=0.85)

    # Annotate bar tops with sample size
    for bar, n_val in zip(bars, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f'n={n_val}',
            ha='center', va='bottom', fontsize=7, color='#444'
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel(feature_col, fontsize=10)
    ax.set_ylabel(f'Mean {target_col}', fontsize=10)
    ax.set_title(f'{feature_col} bins vs mean {target_col}', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    fig.tight_layout()

    fn = filename or f'{feature_col}_vs_{target_col}.png'
    return _save(fig, fn)


# ── Phase 6 – Temporal drift plot ────────────────────────────────────────────

def plot_temporal_stat(
    period_stats: List[dict],
    stat_key: str,
    filename: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """
    Line chart of a per-period stat (e.g. mean gap, win rate).

    Parameters
    ----------
    period_stats : output of eda_analysis_primitives.compute_period_stats()
    stat_key     : key within each period dict to plot (e.g. 'mean_gap', 'target_rate')
    filename     : PNG basename; auto-generated if None

    Returns
    -------
    str — absolute path to saved PNG
    """
    if not period_stats:
        return ''

    periods = [p['period'] for p in period_stats]
    # Coerce to float; replace None / non-numeric with np.nan
    raw     = [p.get(stat_key) for p in period_stats]
    values  = []
    for v in raw:
        try:
            values.append(float(v) if v is not None else float('nan'))
        except (TypeError, ValueError):
            values.append(float('nan'))

    # Skip plot if all values are NaN
    valid = [v for v in values if not (v != v)]  # NaN != NaN
    if not valid:
        print(f"[EDA][plot] Skipping temporal plot for '{stat_key}' — all values NaN.")
        return ''

    x_idx = list(range(len(periods)))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_idx, values, marker='o', color='#4C8EBF', linewidth=2)
    ax.fill_between(x_idx, values, alpha=0.15, color='#4C8EBF')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(periods, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(stat_key, fontsize=10)
    ax.set_title(title or f'{stat_key} over time', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fn = filename or f'temporal_{stat_key}.png'
    return _save(fig, fn)
