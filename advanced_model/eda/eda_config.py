"""
eda_config.py
=============
Single source of truth for all EDA path constants and EDA-scoped thresholds.

Rules:
  - No business-logic thresholds here; those live in config/settings.py.
  - Only EDA-specific knobs (sampling ratios, outlier factors, plot limits).
  - All paths are relative to the advanced_model/ project root.
"""

import os

# ── Project root (resolved at import time) ───────────────────────────────────
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)          # advanced_model/

# ── DB paths ─────────────────────────────────────────────────────────────────
BET_TRACKER_DB      = os.path.join(PROJECT_ROOT, 'database', 'bet_tracker.db')
NBA_DATA_DB         = os.path.join(PROJECT_ROOT, 'database', 'nba_data.db')
MASTER_TRAINING_DB  = os.path.join(PROJECT_ROOT, 'database', 'master_training_data.db')

# ── Feature matrix path ───────────────────────────────────────────────────────
# feature_matrix_v2.parquet is the current canonical output of ml/feature_pipeline.py
FEATURE_MATRIX_PATH = os.path.join(PROJECT_ROOT, 'database', 'feature_matrix_v2.parquet')

# ── Output paths ──────────────────────────────────────────────────────────────
EDA_OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'output', 'eda')
EDA_PLOTS_DIR    = os.path.join(PROJECT_ROOT, 'output', 'eda', 'plots')
EDA_REPORT_PATH  = os.path.join(PROJECT_ROOT, 'output', 'eda', 'eda_report.md')
EDA_FINDINGS_PATH = os.path.join(PROJECT_ROOT, 'output', 'eda', 'eda_findings.json')

# ── EDA-scoped analysis constants ────────────────────────────────────────────
# IQR multiplier for outlier bounds (Tukey method)
EDA_OUTLIER_IQR_FACTOR: float = 1.5

# Max number of rows to load from the feature matrix before falling back to sampling
MAX_FEATURE_ROWS: int = 200_000

# Sampling parameters for heavy tables
EDA_SAMPLE_RATIO: float = 0.20          # 20% random sample when table is large
EDA_SAMPLE_MAX_ROWS: int = 50_000       # hard ceiling on sampled row count

# Missingness alert threshold
EDA_HIGH_MISSING_THRESHOLD: float = 0.20  # 20% null → flagged as high-missing

# Correlation threshold for "high-correlation pair" flagging
EDA_HIGH_CORR_THRESHOLD: float = 0.90

# Max levels shown for categorical frequency tables
EDA_MAX_CATEGORICAL_LEVELS: int = 100

# Odds sanity bounds
EDA_ODDS_MIN: float = 1.01
EDA_ODDS_MAX: float = 15.0

# Number of decile bins for binned-performance plots
EDA_N_DECILE_BINS: int = 10

# First N rows loaded for metadata-only inspection of the feature matrix
EDA_METADATA_PREVIEW_ROWS: int = 5

# Whitelist for columns that sound like leakage but are targets
EDA_LEAKAGE_WHITELIST = ['model_bias', 'actual_value', 'result', 'is_win']
