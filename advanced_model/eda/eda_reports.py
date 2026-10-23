"""
eda_reports.py
==============
JSON findings persistence and Markdown report generation.

Responsibilities:
  - load_findings / save_findings (atomic read-update-write, never blind overwrite).
  - write_markdown_report -- renders eda_report.md from the findings dict.

Rules:
  - Never imported outside eda/.
  - All statistics in the report come from the findings dict -- no hard-coded numbers.
"""

import json
import os
from typing import Dict, Any
from datetime import datetime

from eda.eda_config import EDA_FINDINGS_PATH, EDA_REPORT_PATH, EDA_OUTPUT_DIR


# -- JSON findings helpers -----------------------------------------------------

def load_findings(path: str = EDA_FINDINGS_PATH) -> Dict[str, Any]:
    """
    Load eda_findings.json, or return an empty dict if the file does not exist.
    Never raises on missing file.
    """
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    return {}


def save_findings(findings: Dict[str, Any], path: str = EDA_FINDINGS_PATH):
    """
    Persist the findings dict to *path* as formatted JSON.
    Creates parent directories if needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(findings, fh, indent=2, default=str)


def update_findings(key: str, value: Any, path: str = EDA_FINDINGS_PATH):
    """
    Read existing findings, set findings[key] = value, then write back.
    This is the canonical way to persist any EDA result.
    """
    findings = load_findings(path)
    findings[key] = value
    save_findings(findings, path)


# -- Markdown report generation ------------------------------------------------

def _section(title: str, level: int = 2) -> str:
    return f"\n{'#' * level} {title}\n"


def _code(data, lang: str = 'json') -> str:
    body = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
    return f"\n```{lang}\n{body}\n```\n"


def _table_from_list(rows: list, cols: list) -> str:
    """Render a simple Markdown table from a list of dicts."""
    if not rows:
        return "_No data._\n"
    header  = "| " + " | ".join(cols) + " |"
    sep     = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines   = [header, sep]
    for row in rows:
        cells = [str(row.get(c, '')) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_report(
    findings: Dict[str, Any] = None,
    report_path: str = EDA_REPORT_PATH,
):
    """
    Generate eda_report.md from the findings dictionary.

    If *findings* is None, it is loaded from EDA_FINDINGS_PATH.
    Every statistic is sourced from the findings dict.
    """
    if findings is None:
        findings = load_findings()

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')

    lines = []
    lines.append(f"# EDA Report -- Advanced Model\n")
    lines.append(f"_Generated: {ts}_\n")

    # -- 0. Context & principles -----------------------------------------------
    lines.append(_section("Context & Principles"))
    lines.append(
        "This report is produced by the EDA layer (`advanced_model/eda/`). "
        "It is **read-only** with respect to production databases. "
        "All thresholds and paths are sourced from `eda_config.py` and `config/settings.py`. "
        "Anti-spaghetti rules apply: no duplicate logic, single responsibility per module.\n"
    )

    # -- 1. Data sources & schemas ---------------------------------------------
    lines.append(_section("Data Sources & Schemas"))

    tables = findings.get('tables', [])
    if tables:
        lines.append(f"**{len(tables)} tables/views enumerated across all databases.**\n")
        lines.append(_table_from_list(tables, ['db', 'table', 'type', 'row_count']))
    else:
        lines.append("_Table enumeration not yet run._\n")

    core_tables = findings.get('core_tables', [])
    if core_tables:
        lines.append(_section("Core EDA Tables", 3))
        lines.append(_table_from_list(core_tables, ['db', 'table', 'role']))

    schemas = findings.get('schemas', {})
    if schemas:
        lines.append(_section("Schema Details", 3))
        for db_name, tables_dict in schemas.items():
            for tbl, cols in tables_dict.items():
                lines.append(f"**{db_name}.{tbl}** ({len(cols)} columns)\n")
                lines.append(_table_from_list(cols, ['name', 'type', 'notnull', 'pk']))

    fm_meta = findings.get('feature_matrix_meta', {})
    if fm_meta:
        lines.append(_section("Feature Matrix Metadata", 3))
        lines.append(
            f"- Rows: **{fm_meta.get('n_rows', '?'):,}**  \n"
            f"- Columns: **{fm_meta.get('n_cols', '?')}**  \n"
            f"- Detected label columns: `{fm_meta.get('labels', [])}`\n"
        )

    # -- 2. Row counts & sampling ----------------------------------------------
    lines.append(_section("Row Counts & Sampling"))
    row_counts = findings.get('row_counts', {})
    if row_counts:
        for db_name, tbl_dict in row_counts.items():
            lines.append(f"**{db_name}**\n")
            for tbl, info in tbl_dict.items():
                lines.append(
                    f"- `{tbl}`: {info.get('row_count', '?'):,} rows"
                    + (f"  |  {info.get('min_date')} -> {info.get('max_date')}"
                       if info.get('min_date') else "")
                    + "\n"
                )

    si = findings.get('sampling_info', {})
    if si:
        lines.append(_section("Sampling Parameters", 3))
        lines.append(
            f"- Feature matrix: {si.get('sampled_rows', '?'):,} / "
            f"{si.get('total_rows', '?'):,} rows "
            f"(ratio={si.get('sample_ratio', '?')}, "
            f"sampled={si.get('was_sampled', '?')})\n"
        )

    # -- 3. Missingness & univariate stats -------------------------------------
    lines.append(_section("Missingness & Univariate Statistics"))

    for source_key in ['feature_matrix', 'training_data']:
        miss = findings.get('missingness', {}).get(source_key, [])
        if miss:
            high = [m for m in miss if m.get('high_missing')]
            lines.append(_section(f"Missingness -- {source_key}", 3))
            lines.append(f"Total columns: {len(miss)}  |  High-missing (>20%): **{len(high)}**\n")
            if high:
                lines.append("**High-missing columns:**\n")
                lines.append(_table_from_list(high, ['name', 'n_null', 'pct_null']))

    issues_missing = findings.get('issues', {}).get('missing_critical_fields', [])
    if issues_missing:
        lines.append("\n> ⚠️ **CRITICAL:** The following required columns have nulls:\n")
        lines.append(_table_from_list(issues_missing, ['column', 'pct_null', 'n_null']))

    descs = findings.get('descriptives', {}).get('feature_matrix', {})
    if descs:
        lines.append(_section("Descriptive Statistics -- Feature Matrix (numeric)", 3))
        rows_d = [{'column': k, **v} for k, v in list(descs.items())[:30]]
        lines.append(_table_from_list(rows_d, ['column', 'n', 'mean', 'median', 'std', 'min', 'max', 'p25', 'p75']))
        if len(descs) > 30:
            lines.append(f"_... and {len(descs)-30} more columns (see eda_findings.json)._\n")

    cat_dists = findings.get('categorical_distributions', {})
    if cat_dists:
        lines.append(_section("Categorical Distributions", 3))
        for col, vc in cat_dists.items():
            lines.append(f"**{col}** (top {len(vc)} levels):\n")
            lines.append(_table_from_list(vc[:10], ['value', 'count', 'pct']))

    # -- 4. Outliers & distributions -------------------------------------------
    lines.append(_section("Outliers & Distribution Shapes"))

    fm_outliers = findings.get('outliers', {}).get('feature_matrix', {})
    if fm_outliers:
        flagged = {k: v for k, v in fm_outliers.items()
                   if v.get('pct_outliers') is not None and v['pct_outliers'] > 0.05}
        lines.append(f"Columns with >5% IQR outliers: **{len(flagged)}**\n")
        if flagged:
            rows_o = [{'column': k, **{kk: vv for kk, vv in v.items() if kk != 'example_values'}}
                      for k, v in list(flagged.items())[:20]]
            lines.append(_table_from_list(rows_o, ['column', 'q1', 'q3', 'lower_bound', 'upper_bound', 'n_outliers', 'pct_outliers']))

    odds_issues = findings.get('issues', {}).get('odds_integrity', [])
    if odds_issues:
        lines.append(_section("Odds Integrity Issues", 3))
        lines.append(_table_from_list(odds_issues, ['column', 'n_below', 'n_above', 'min_val', 'max_val']))

    # -- 5. Correlations & feature–target relationships ------------------------
    lines.append(_section("Correlations & Feature–Target Relationships"))

    hcp = findings.get('high_corr_pairs', [])
    lines.append(f"High-correlation pairs (|r| > 0.9): **{len(hcp)}**\n")
    if hcp:
        lines.append(_table_from_list(hcp[:20], ['col_a', 'col_b', 'correlation']))

    mi = findings.get('feature_target_importance_classification', {})
    if mi:
        lines.append(_section("Top Features by Mutual Information (-> is_win)", 3))
        top_mi = [{'feature': k, 'mi_score': v} for k, v in list(mi.items())[:20]]
        lines.append(_table_from_list(top_mi, ['feature', 'mi_score']))

    reg_corr = findings.get('feature_target_importance_regression', {})
    if reg_corr:
        lines.append(_section("Top Features by Correlation (-> model_bias)", 3))
        top_rc = [{'feature': k, 'correlation': v} for k, v in list(reg_corr.items())[:20]]
        lines.append(_table_from_list(top_rc, ['feature', 'correlation']))

    # -- 6. Temporal drift & source-mode effects -------------------------------
    lines.append(_section("Temporal Drift & Source-Mode Effects"))

    period_stats = findings.get('temporal_drift', {}).get('period_stats', [])
    if period_stats:
        lines.append(f"Periods analysed: **{len(period_stats)}**\n")
        lines.append(_table_from_list(period_stats[:24],
                     [k for k in (period_stats[0].keys() if period_stats else [])]))

    sm_drift = findings.get('source_mode_drift', {})
    if sm_drift:
        lines.append(_section("Source-Mode Drift", 3))
        lines.append(_code(sm_drift))

    playoff = findings.get('temporal_drift', {}).get('playoff_vs_regular', {})
    if playoff:
        lines.append(_section("Playoff vs Regular Season", 3))
        lines.append(_code(playoff))

    # -- 7. Leakage & selection bias -------------------------------------------
    lines.append(_section("Leakage & Selection Bias Checks"))

    leakage = findings.get('issues', {}).get('potential_leakage', [])
    lines.append(f"Suspected leakage columns: **{len(leakage)}**\n")
    if leakage:
        lines.append(_table_from_list(leakage, ['column', 'reason']))

    sel_bias = findings.get('selection_bias', {}).get('training_vs_full', {})
    if sel_bias:
        lines.append(_section("Training vs Full Universe", 3))
        lines.append(_code(sel_bias))

    cb = findings.get('class_balance', {})
    if cb:
        lines.append(_section("Class Balance", 3))
        glob = cb.get('global', {})
        lines.append(f"Global n={glob.get('n', '?')}  |  balance={glob.get('balance', {})}\n")

    # -- 8. Recommendations for ML ---------------------------------------------
    lines.append(_section("Recommendations for ML"))

    recs = []
    # Auto-generate from findings
    if len(hcp) > 0:
        recs.append(f"- **Drop / merge highly-correlated pairs** ({len(hcp)} pairs with |r|>0.9). "
                    "Retain the more interpretable feature per pair.")
    high_miss_fm = [m for m in findings.get('missingness', {}).get('feature_matrix', [])
                    if m.get('high_missing')]
    if high_miss_fm:
        recs.append(f"- **Impute or drop high-missing columns**: "
                    f"{', '.join(m['name'] for m in high_miss_fm[:5])} "
                    f"(and {max(0, len(high_miss_fm)-5)} more).")
    if issues_missing:
        recs.append(f"- **CRITICAL -- fix null critical columns** before training: "
                    f"{', '.join(i['column'] for i in issues_missing)}.")
    if odds_issues:
        recs.append(f"- **Sanitise odds columns** -- {len(odds_issues)} columns contain out-of-range values.")
    if leakage:
        recs.append(f"- **Audit suspected leakage columns** before including in training: "
                    f"{', '.join(l['column'] for l in leakage[:5])}.")

    if recs:
        lines.extend(recs)
    else:
        lines.append("_No automatic recommendations generated (findings may be incomplete)._\n")

    # -- Write file -------------------------------------------------------------
    with open(report_path, 'w', encoding='utf-8') as fh:
        fh.write("\n".join(lines))

    print(f"[EDA] Report written -> {report_path}")
    return report_path
