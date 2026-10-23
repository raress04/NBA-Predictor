"""
eda_ml_data_loader.py
=====================
Responsible for loading ML-related data into DataFrames for EDA.

Responsibilities:
  - Feature matrix metadata + sampling.
  - Projection outcomes sampling from master_training_data.db.

Rules:
  - All DB reads go through eda_sql_helpers.execute_select.
  - No writes to any database or file (call eda_reports.save_findings for persistence).
  - Never imported outside eda/.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from eda.eda_config import (
    FEATURE_MATRIX_PATH,
    MASTER_TRAINING_DB,
    MAX_FEATURE_ROWS,
    EDA_SAMPLE_RATIO,
    EDA_SAMPLE_MAX_ROWS,
    EDA_METADATA_PREVIEW_ROWS,
)
from eda.eda_sql_helpers import get_connection_read_only, execute_select


# ── Target / label column name hints ─────────────────────────────────────────
_LABEL_PATTERNS: List[str] = [
    'is_win', 'label', 'model_bias', 'result', 'target',
    'actual_value', 'actualvalue',
]


def _dtype_hint(dtype) -> str:
    """Map a pandas dtype to a human-readable hint: 'numeric' or 'categorical'."""
    if pd.api.types.is_numeric_dtype(dtype):
        return 'numeric'
    return 'categorical'


# ── Feature matrix ────────────────────────────────────────────────────────────

def load_feature_matrix_metadata(path: str = FEATURE_MATRIX_PATH) -> Dict:
    """
    Read schema and first N rows of the feature matrix parquet file.

    Returns
    -------
    dict with keys:
        n_rows, n_cols, column_names, dtypes (col→hint), labels (list of label cols)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[EDA] Feature matrix not found: {path}")

    # Schema only via pyarrow (fast, no full load)
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow

    # Full row count from metadata
    n_rows = pf.metadata.num_rows
    n_cols = len(schema)
    column_names = [schema.field(i).name for i in range(n_cols)]

    # Dtype hints via preview rows
    preview_df = pf.read_row_groups([0]).to_pandas().head(EDA_METADATA_PREVIEW_ROWS)
    dtypes = {col: _dtype_hint(preview_df[col].dtype) for col in preview_df.columns}

    # Label detection
    labels = [c for c in column_names if c.lower() in _LABEL_PATTERNS]

    return {
        'path':         path,
        'n_rows':       n_rows,
        'n_cols':       n_cols,
        'column_names': column_names,
        'dtypes':       dtypes,
        'labels':       labels,
    }


def load_feature_matrix_sample(path: str = FEATURE_MATRIX_PATH) -> pd.DataFrame:
    """
    Load the full feature matrix if small, otherwise sample EDA_SAMPLE_MAX_ROWS rows.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[EDA] Feature matrix not found: {path}")

    df = pd.read_parquet(path)
    n = len(df)

    if n <= MAX_FEATURE_ROWS:
        print(f"[EDA] Feature matrix loaded fully: {n:,} rows")
        return df

    # Random sample
    sample_n = min(EDA_SAMPLE_MAX_ROWS, int(n * EDA_SAMPLE_RATIO))
    sampled = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    print(f"[EDA] Feature matrix sampled {sample_n:,}/{n:,} rows "
          f"(ratio={sample_n/n:.2%})")
    return sampled


def get_feature_matrix_sampling_info(path: str = FEATURE_MATRIX_PATH) -> Dict:
    """Return sampling metadata dict for eda_findings.json."""
    import pyarrow.parquet as pq
    n_rows = pq.ParquetFile(path).metadata.num_rows
    if n_rows <= MAX_FEATURE_ROWS:
        actual = n_rows
        ratio  = 1.0
        sampled = False
    else:
        actual  = min(EDA_SAMPLE_MAX_ROWS, int(n_rows * EDA_SAMPLE_RATIO))
        ratio   = actual / n_rows
        sampled = True
    return {
        'source':           path,
        'total_rows':       n_rows,
        'sampled_rows':     actual,
        'sample_ratio':     round(ratio, 4),
        'was_sampled':      sampled,
        'max_feature_rows': MAX_FEATURE_ROWS,
    }


# ── Projection outcomes ───────────────────────────────────────────────────────

def load_projection_outcomes_sample(
    db_path: str = MASTER_TRAINING_DB,
    sample_ratio: float = EDA_SAMPLE_RATIO,
    max_rows: int = EDA_SAMPLE_MAX_ROWS,
) -> pd.DataFrame:
    """
    Load a representative sample from the projection_outcomes table.

    Stratification: proportional sampling per category so each category
    is represented by roughly the same ratio.

    Returns
    -------
    pd.DataFrame
    """
    conn = get_connection_read_only(db_path)
    rows = execute_select(conn, "SELECT * FROM projection_outcomes")
    conn.close()

    df = pd.DataFrame(rows)
    n  = len(df)

    if n == 0:
        print("[EDA] projection_outcomes is empty.")
        return df

    if n <= max_rows:
        print(f"[EDA] projection_outcomes loaded fully: {n:,} rows")
        return df

    target_n = min(max_rows, int(n * sample_ratio))

    if 'category' in df.columns:
        # Stratified by category
        sampled_parts = []
        for cat, grp in df.groupby('category', sort=False):
            cat_n = max(1, int(target_n * len(grp) / n))
            sampled_parts.append(grp.sample(n=min(cat_n, len(grp)), random_state=42))
        sampled = pd.concat(sampled_parts).reset_index(drop=True)
    else:
        sampled = df.sample(n=target_n, random_state=42).reset_index(drop=True)

    print(f"[EDA] projection_outcomes sampled {len(sampled):,}/{n:,} rows")
    return sampled


# ── Generic table sampler ─────────────────────────────────────────────────────

def sample_table(
    conn,
    table: str,
    sample_ratio: float = EDA_SAMPLE_RATIO,
    stratify_by: Optional[str] = None,
    max_rows: Optional[int] = EDA_SAMPLE_MAX_ROWS,
) -> pd.DataFrame:
    """
    Load and optionally sample a DB table into a DataFrame.

    Parameters
    ----------
    conn         : read-only sqlite3.Connection
    table        : table name
    sample_ratio : fraction of rows to keep when sampling
    stratify_by  : column name to use for stratified sampling (optional)
    max_rows     : hard ceiling on output rows

    Returns
    -------
    pd.DataFrame
    """
    rows = execute_select(conn, f'SELECT * FROM "{table}"')
    df   = pd.DataFrame(rows)
    n    = len(df)

    if n == 0 or max_rows is None or n <= max_rows:
        return df

    target_n = min(max_rows, int(n * sample_ratio))

    if stratify_by and stratify_by in df.columns:
        parts = []
        for _, grp in df.groupby(stratify_by, sort=False):
            k = max(1, int(target_n * len(grp) / n))
            parts.append(grp.sample(n=min(k, len(grp)), random_state=42))
        return pd.concat(parts).reset_index(drop=True)

    return df.sample(n=target_n, random_state=42).reset_index(drop=True)
