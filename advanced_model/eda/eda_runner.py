"""
eda_runner.py
=============
Top-level EDA orchestrator.

Entry point:
    python -m eda.eda_runner

Execution order (plan-mandated, strictly sequential):
  Phase 0 – preconditions
  Phase 1 – data sources & schemas
  Phase 2 – row counts & sampling
  Phase 3 – missingness & univariate
  Phase 4 – outliers & distributions
  Phase 5 – correlations & feature–target
  Phase 6 – temporal drift & source-mode
  Phase 7 – leakage & selection bias
  Phase 8 – report generation

Rules:
  - Never import EDA modules from ETL / simulator / ML / analyze layers.
  - All DB access via eda_sql_helpers.execute_select only.
  - No writes to production DBs.
"""

import os
import sys
import re

# Ensure advanced_model/ is on sys.path when run as script
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

from eda.eda_config import (
    BET_TRACKER_DB, NBA_DATA_DB, MASTER_TRAINING_DB,
    FEATURE_MATRIX_PATH,
    EDA_OUTPUT_DIR, EDA_PLOTS_DIR, EDA_REPORT_PATH, EDA_FINDINGS_PATH,
    EDA_HIGH_CORR_THRESHOLD, EDA_LEAKAGE_WHITELIST,
)
from eda.eda_sql_helpers import (
    get_connection_read_only,
    list_tables,
    get_row_count,
    get_table_schema,
    get_date_range,
    detect_date_column,
)
from eda.eda_ml_data_loader import (
    load_feature_matrix_metadata,
    load_feature_matrix_sample,
    get_feature_matrix_sampling_info,
    load_projection_outcomes_sample,
    sample_table,
)
from eda.eda_analysis_primitives import (
    split_numeric_categorical,
    compute_missingness,
    check_critical_columns_missing,
    compute_descriptive_stats,
    compute_value_counts,
    compute_iqr_outliers,
    check_odds_integrity,
    compute_correlation_matrix,
    find_high_corr_pairs,
    compute_mutual_information,
    compute_feature_target_correlation,
    compute_binned_performance,
    add_period_column,
    compute_period_stats,
    compute_class_balance,
)
from eda.eda_plots import (
    plot_histogram,
    plot_corr_heatmap,
    plot_binned_performance,
    plot_temporal_stat,
)
from eda.eda_reports import (
    load_findings,
    save_findings,
    update_findings,
    write_markdown_report,
)


# -- Phase 1.2 – Core table rules ---------------------------------------------
# These are the tables EDA cares about. Each entry specifies which DB and the
# conceptual role of the table in the ML pipeline.
CORE_TABLE_RULES = [
    # bet_tracker.db
    {'db': 'bet_tracker', 'table': 'daily_picks',         'role': 'picks'},
    {'db': 'bet_tracker', 'table': 'pick_results',        'role': 'results'},
    {'db': 'bet_tracker', 'table': 'parlays',             'role': 'parlays'},
    {'db': 'bet_tracker', 'table': 'parlay_legs',         'role': 'parlay_legs'},
    {'db': 'bet_tracker', 'table': 'projection_outcomes', 'role': 'outcomes_tracker'},
    # nba_data.db
    {'db': 'nba_data',    'table': 'box_scores',          'role': 'box_scores'},
    {'db': 'nba_data',    'table': 'games',               'role': 'games'},
    {'db': 'nba_data',    'table': 'teams',               'role': 'teams'},
    {'db': 'nba_data',    'table': 'injuries',            'role': 'injuries'},
    # master_training_data.db
    {'db': 'master_training', 'table': 'projection_outcomes', 'role': 'training_data'},
]

# Critical columns that must have zero nulls in the training data
_CRITICAL_COL_PATTERNS = [
    r'bookmakerline', r'bookmaker_line',
    r'over_odds', r'under_odds',
    r'gap', r'vig',
    r'result', r'is_win',
    r'actualvalue', r'actual_value',
]

# Key histogram targets (Phase 4)
_HIST_COLS = ['gap', 'vig', 'sim_confidence', 'edge_pct',
              'projected_value', 'model_bias', 'conf', 'over_odds', 'under_odds']

# Key categorical columns (Phase 3.5)
_CAT_COLS = ['category', 'source_mode', 'is_playoff', 'team', 'opponent', 'direction', 'season']

# Key feature columns for binned-performance plots (Phase 5.5)
_BINNED_FEATURES = ['gap', 'vig', 'conf', 'edge_pct']

# Odds columns to validate (Phase 4.2)
_ODDS_COLS = ['over_odds', 'under_odds', 'decimal_odds']

# Leakage pattern words
_LEAKAGE_PATTERNS = ['future', 'target', 'postgame', 'post_game',
                     'actual', 'result', 'is_win', 'label', 'model_bias']


# ═════════════════════════════════════════════════════════════════════════════
# Phase 0 – Preconditions
# ═════════════════════════════════════════════════════════════════════════════

def check_eda_preconditions():
    """Verify all required files exist and create output directories."""
    print("\n" + "=" * 60)
    print("  EDA PRECONDITION CHECK")
    print("=" * 60)

    required_files = [
        ('BET_TRACKER_DB',     BET_TRACKER_DB),
        ('NBA_DATA_DB',        NBA_DATA_DB),
        ('MASTER_TRAINING_DB', MASTER_TRAINING_DB),
        ('FEATURE_MATRIX',     FEATURE_MATRIX_PATH),
    ]

    all_ok = True
    for label, path in required_files:
        exists = os.path.exists(path)
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {label:<22} {status}  ({path})")
        if not exists:
            all_ok = False

    os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)
    os.makedirs(EDA_PLOTS_DIR,  exist_ok=True)
    print(f"\n  Output dir  : {EDA_OUTPUT_DIR}")
    print(f"  Plots dir   : {EDA_PLOTS_DIR}")

    if not all_ok:
        print("\n[EDA] FAILED -- precondition check failed, aborting.")
        sys.exit(1)
    print("\n[EDA] OK -- all preconditions satisfied.\n")


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1 – Data Sources & Schemas
# ═════════════════════════════════════════════════════════════════════════════

def _db_label(db_path: str) -> str:
    return os.path.splitext(os.path.basename(db_path))[0]


def identify_core_tables(table_list, rules):
    """Cross-match actual table names with CORE_TABLE_RULES (case-insensitive)."""
    def normalise(s): return re.sub(r'[_\s]', '', s).lower()

    found = []
    for rule in rules:
        tgt = normalise(rule['table'])
        for entry in table_list:
            if (normalise(entry['table']) == tgt
                    and entry.get('db', '') == rule['db']):
                found.append({
                    'db':    rule['db'],
                    'table': entry['table'],
                    'role':  rule['role'],
                })
                break
    return found


def run_phase1():
    print("\n-- Phase 1: Data Sources & Schemas --")

    db_map = {
        'bet_tracker':    BET_TRACKER_DB,
        'nba_data':       NBA_DATA_DB,
        'master_training': MASTER_TRAINING_DB,
    }

    all_tables = []
    schemas    = {}

    for db_label, db_path in db_map.items():
        conn = get_connection_read_only(db_path)
        tbls = list_tables(conn)
        for t in tbls:
            t['table'] = t.pop('name')   # normalise key: name -> table
            t['db'] = db_label
            t['row_count'] = get_row_count(conn, t['table'])
        all_tables.extend(tbls)

        schemas[db_label] = {}
        for t in tbls:
            schemas[db_label][t['table']] = get_table_schema(conn, t['table'])

        conn.close()
        print(f"  [{db_label}] {len(tbls)} tables/views")

    update_findings('tables', all_tables)

    # Core tables
    core_tables = identify_core_tables(all_tables, CORE_TABLE_RULES)
    update_findings('core_tables', core_tables)
    print(f"  Core tables matched: {len(core_tables)}")

    update_findings('schemas', schemas)

    # Feature matrix metadata
    fm_meta = load_feature_matrix_metadata(FEATURE_MATRIX_PATH)
    update_findings('feature_matrix_meta', fm_meta)
    print(f"  Feature matrix: {fm_meta['n_rows']:,} rows × {fm_meta['n_cols']} cols")


# ═════════════════════════════════════════════════════════════════════════════
# Phase 2 – Row Counts & Sampling
# ═════════════════════════════════════════════════════════════════════════════

def run_phase2():
    print("\n-- Phase 2: Row Counts & Sampling --")

    findings = load_findings()
    schemas  = findings.get('schemas', {})

    db_map = {
        'bet_tracker':    BET_TRACKER_DB,
        'nba_data':       NBA_DATA_DB,
        'master_training': MASTER_TRAINING_DB,
    }

    row_counts = {}
    for db_label, db_path in db_map.items():
        conn = get_connection_read_only(db_path)
        row_counts[db_label] = {}
        for tbl_name in (schemas.get(db_label) or {}):
            n = get_row_count(conn, tbl_name)
            schema_cols = schemas[db_label].get(tbl_name, [])
            date_col    = detect_date_column(schema_cols)
            entry       = {'row_count': n}
            if date_col:
                entry.update(get_date_range(conn, tbl_name, date_col))
            row_counts[db_label][tbl_name] = entry
            print(f"  {db_label}.{tbl_name}: {n:,} rows")
        conn.close()

    update_findings('row_counts', row_counts)

    # Sampling info
    si = get_feature_matrix_sampling_info(FEATURE_MATRIX_PATH)
    update_findings('sampling_info', si)
    print(f"  Feature matrix sampling: {si['sampled_rows']:,}/{si['total_rows']:,} rows")


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3 – Missingness & Univariate
# ═════════════════════════════════════════════════════════════════════════════

def run_phase3(fm_df: pd.DataFrame, proj_df: pd.DataFrame):
    print("\n-- Phase 3: Missingness & Univariate --")

    numeric_cols, cat_cols = split_numeric_categorical(fm_df)
    update_findings('numeric_features',      numeric_cols)
    update_findings('categorical_features',  cat_cols)
    print(f"  Feature matrix: {len(numeric_cols)} numeric, {len(cat_cols)} categorical")

    # Missingness – feature matrix
    miss_fm = compute_missingness(fm_df)
    findings = load_findings()
    miss     = findings.get('missingness', {})
    miss['feature_matrix'] = miss_fm
    update_findings('missingness', miss)

    high_miss_fm = [m for m in miss_fm if m['high_missing']]
    print(f"  Feature matrix high-missing cols: {len(high_miss_fm)}")

    # Missingness – training data
    cls_df = proj_df.query("ml_usable_classification == 1") if 'ml_usable_classification' in proj_df.columns else proj_df
    reg_df = proj_df.query("ml_usable_regression == 1") if 'ml_usable_regression' in proj_df.columns else proj_df

    miss_train = compute_missingness(cls_df)
    miss_train_reg = compute_missingness(reg_df)

    miss = load_findings().get('missingness', {})
    miss['training_data'] = miss_train
    miss['training_data_regression'] = miss_train_reg
    update_findings('missingness', miss)

    # Critical column check
    crit_issues = check_critical_columns_missing(miss_train, _CRITICAL_COL_PATTERNS)
    iss = load_findings().get('issues', {})
    iss['missing_critical_fields'] = crit_issues
    update_findings('issues', iss)
    if crit_issues:
        print(f"  [CRITICAL] Missing fields: {[i['column'] for i in crit_issues]}")
    else:
        print("  [OK] All critical training columns have zero nulls.")

    # Descriptive stats – feature matrix numeric
    desc_fm = compute_descriptive_stats(fm_df, numeric_cols)
    descs = load_findings().get('descriptives', {})
    descs['feature_matrix'] = desc_fm
    update_findings('descriptives', descs)

    # Descriptive stats – training data
    num_proj, _ = split_numeric_categorical(proj_df)
    desc_proj = compute_descriptive_stats(proj_df, num_proj)
    descs = load_findings().get('descriptives', {})
    descs['training_data'] = desc_proj
    update_findings('descriptives', descs)

    # Categorical frequencies
    cat_dists = {}
    for col in _CAT_COLS:
        for df in [fm_df, proj_df]:
            if col in df.columns:
                cat_dists[col] = compute_value_counts(df, col)
                break
    update_findings('categorical_distributions', cat_dists)
    print(f"  Categorical distributions computed for: {list(cat_dists.keys())}")


# ═════════════════════════════════════════════════════════════════════════════
# Phase 4 – Outliers & Distributions
# ═════════════════════════════════════════════════════════════════════════════

def run_phase4(fm_df: pd.DataFrame, proj_df: pd.DataFrame):
    print("\n-- Phase 4: Outliers & Distributions --")

    numeric_cols = load_findings().get('numeric_features', [])

    # IQR outliers – feature matrix
    fm_outliers = {}
    for col in numeric_cols:
        if col in fm_df.columns:
            fm_outliers[col] = compute_iqr_outliers(fm_df[col])

    outliers = load_findings().get('outliers', {})
    outliers['feature_matrix'] = fm_outliers
    update_findings('outliers', outliers)
    flagged = sum(1 for v in fm_outliers.values()
                  if v.get('pct_outliers') and v['pct_outliers'] > 0.05)
    print(f"  Feature matrix: {flagged} cols with >5% IQR outliers")

    # IQR outliers – projections
    num_proj, _ = split_numeric_categorical(proj_df)
    proj_outliers = {col: compute_iqr_outliers(proj_df[col])
                     for col in num_proj if col in proj_df.columns}
    outliers = load_findings().get('outliers', {})
    outliers['projections'] = proj_outliers
    update_findings('outliers', outliers)

    descs = load_findings().get('descriptives', {})
    descs['projections'] = compute_descriptive_stats(proj_df, num_proj)
    update_findings('descriptives', descs)

    # Odds integrity
    odds_issues = check_odds_integrity(proj_df, _ODDS_COLS)
    iss = load_findings().get('issues', {})
    iss['odds_integrity'] = odds_issues
    update_findings('issues', iss)
    print(f"  Odds integrity issues: {len(odds_issues)}")

    # Histograms
    for col in _HIST_COLS:
        for df, label in [(fm_df, 'fm'), (proj_df, 'proj')]:
            if col in df.columns:
                plot_histogram(
                    df[col],
                    title=f'{col} distribution ({label})',
                    filename=f'hist_{col}_{label}.png',
                )
                break
    print(f"  Histograms saved to {EDA_PLOTS_DIR}")


# ═════════════════════════════════════════════════════════════════════════════
# Phase 5 – Correlations & Feature–Target
# ═════════════════════════════════════════════════════════════════════════════

def run_phase5(fm_df: pd.DataFrame, proj_df: pd.DataFrame):
    print("\n-- Phase 5: Correlations & Feature–Target --")

    numeric_cols = load_findings().get('numeric_features', [])
    available    = [c for c in numeric_cols if c in fm_df.columns]

    # Correlation matrix
    corr_df = compute_correlation_matrix(fm_df, available)
    corr_csv = os.path.join(EDA_OUTPUT_DIR, 'corr_matrix_features.csv')
    corr_df.to_csv(corr_csv)
    print(f"  Correlation matrix saved -> {corr_csv}")

    plot_corr_heatmap(corr_df, filename='corr_heatmap.png')

    # High-corr pairs
    hcp = find_high_corr_pairs(corr_df, threshold=EDA_HIGH_CORR_THRESHOLD)
    update_findings('high_corr_pairs', hcp)
    print(f"  High-corr pairs (|r|>{EDA_HIGH_CORR_THRESHOLD}): {len(hcp)}")

    # Mutual information (classification)
    label_cls = next((c for c in ['is_win', 'result', 'label'] if c in fm_df.columns), None)
    if label_cls:
        mi = compute_mutual_information(fm_df, available, label_cls)
        update_findings('feature_target_importance_classification', mi)
        top3 = list(mi.items())[:3]
        print(f"  Top-3 MI features (->{label_cls}): {top3}")
    else:
        print("  [skip] No classification label found in feature matrix.")

    # Pearson/Spearman (regression)
    label_reg = next((c for c in ['model_bias', 'actual_value'] if c in fm_df.columns), None)
    if label_reg:
        rc = compute_feature_target_correlation(fm_df, available, label_reg, method='pearson')
        update_findings('feature_target_importance_regression', rc)
        top3r = list(rc.items())[:3]
        print(f"  Top-3 corr features (->{label_reg}): {top3r}")
    else:
        print("  [skip] No regression target found in feature matrix.")

    # Binned performance plots
    for feat in _BINNED_FEATURES:
        for tgt in [label_cls, label_reg]:
            if tgt and feat in fm_df.columns and tgt in fm_df.columns:
                bstats = compute_binned_performance(fm_df, feat, tgt)
                plot_binned_performance(bstats, feat, tgt,
                                        filename=f'{feat}_vs_{tgt}.png')
                break


# ═════════════════════════════════════════════════════════════════════════════
# Phase 6 – Temporal Drift & Source-Mode
# ═════════════════════════════════════════════════════════════════════════════

def run_phase6(proj_df: pd.DataFrame):
    print("\n-- Phase 6: Temporal Drift & Source-Mode --")

    date_col = next((c for c in ['game_date', 'gamedate', 'date'] if c in proj_df.columns), None)

    if not date_col:
        print("  [skip] No date column found in training data.")
        update_findings('temporal_drift', {'period_stats': []})
        return

    df_period = add_period_column(proj_df, date_col, freq='M')

    stat_cols = [c for c in ['gap', 'vig', 'conf', 'sim_confidence', 'edge_pct']
                 if c in df_period.columns]
    target_col = next((c for c in ['is_win', 'result'] if c in df_period.columns), None)

    period_stats = compute_period_stats(df_period, stat_cols, target_col=target_col)
    td = load_findings().get('temporal_drift', {})
    td['period_stats'] = period_stats
    update_findings('temporal_drift', td)
    print(f"  {len(period_stats)} monthly periods computed")

    # Temporal plots
    for stat in (stat_cols[:3] + (['target_rate'] if target_col else [])):
        plot_temporal_stat(period_stats, f'mean_{stat}' if not stat.startswith('target') else stat,
                           filename=f'temporal_{stat}.png')

    # Source-mode drift
    sm_col = next((c for c in ['source_mode', 'mode'] if c in proj_df.columns), None)
    if sm_col:
        sm_drift = {}
        for mode, grp in proj_df.groupby(sm_col, sort=True):
            entry: dict = {'n': int(len(grp))}
            for col in stat_cols:
                if col in grp.columns:
                    s = pd.to_numeric(grp[col], errors='coerce').dropna()
                    entry[f'mean_{col}'] = round(float(s.mean()), 6) if len(s) else None
            if target_col and target_col in grp.columns:
                s_tgt = pd.to_numeric(grp[target_col], errors='coerce').dropna()
                entry['hit_rate'] = round(float(s_tgt.mean()), 6) if len(s_tgt) else None
            sm_drift[str(mode)] = entry
        update_findings('source_mode_drift', sm_drift)
        print(f"  Source-mode drift: {list(sm_drift.keys())}")

    # Playoff vs regular
    playoff_col = next((c for c in ['is_playoff', 'playoff'] if c in proj_df.columns), None)
    if playoff_col:
        pv = {}
        for val, grp in proj_df.groupby(playoff_col, sort=True):
            label = 'playoff' if val else 'regular'
            entry_pv: dict = {'n': int(len(grp))}
            for col in stat_cols:
                if col in grp.columns:
                    s = pd.to_numeric(grp[col], errors='coerce').dropna()
                    entry_pv[f'mean_{col}'] = round(float(s.mean()), 6) if len(s) else None
            if target_col and target_col in grp.columns:
                s_tgt = pd.to_numeric(grp[target_col], errors='coerce').dropna()
                entry_pv['hit_rate'] = round(float(s_tgt.mean()), 6) if len(s_tgt) else None
            pv[label] = entry_pv
        td = load_findings().get('temporal_drift', {})
        td['playoff_vs_regular'] = pv
        update_findings('temporal_drift', td)
        print(f"  Playoff vs regular: {pv}")


# ═════════════════════════════════════════════════════════════════════════════
# Phase 7 – Leakage & Selection Bias
# ═════════════════════════════════════════════════════════════════════════════

def run_phase7(fm_df: pd.DataFrame, proj_df: pd.DataFrame):
    print("\n-- Phase 7: Leakage & Selection Bias --")

    # Leakage audit on feature names
    all_cols  = list(fm_df.columns)
    suspected = []
    for col in all_cols:
        if col in EDA_LEAKAGE_WHITELIST:
            continue
        for pat in _LEAKAGE_PATTERNS:
            if pat in col.lower():
                suspected.append({'column': col, 'reason': f"name matches pattern '{pat}'"})
                break
    iss = load_findings().get('issues', {})
    iss['potential_leakage'] = suspected
    update_findings('issues', iss)
    print(f"  Suspected leakage columns: {len(suspected)}")

    # Training vs full universe (key feature distributions)
    key_feats = [c for c in ['gap', 'vig', 'conf', 'edge_pct', 'over_odds', 'under_odds']
                 if c in proj_df.columns and c in fm_df.columns]

    tv = {}
    for col in key_feats:
        full_s  = pd.to_numeric(proj_df[col], errors='coerce').dropna()
        train_s = pd.to_numeric(fm_df[col],   errors='coerce').dropna()
        tv[col] = {
            'full_mean':    round(float(full_s.mean()),  6) if len(full_s) else None,
            'train_mean':   round(float(train_s.mean()), 6) if len(train_s) else None,
            'full_std':     round(float(full_s.std()),   6) if len(full_s) else None,
            'train_std':    round(float(train_s.std()),  6) if len(train_s) else None,
            'full_n':       int(len(full_s)),
            'train_n':      int(len(train_s)),
        }
    sb = load_findings().get('selection_bias', {})
    sb['training_vs_full'] = tv
    update_findings('selection_bias', sb)

    # Class balance
    label_col  = next((c for c in ['is_win', 'result'] if c in fm_df.columns), None)
    group_cols = [c for c in ['category', 'source_mode', 'season'] if c in fm_df.columns]

    if label_col:
        cb = compute_class_balance(fm_df, label_col, group_cols)
        update_findings('class_balance', cb)
        g = cb.get('global', {})
        print(f"  Class balance (global): n={g.get('n')} {g.get('balance')}")
    else:
        print("  [skip] No classification label found for class balance.")


# ═════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  NBA ADVANCED MODEL -- EDA RUNNER")
    print("=" * 60)

    # -- Phase 0 ---------------------------------------------------------------
    check_eda_preconditions()

    # -- Phase 1 ---------------------------------------------------------------
    run_phase1()

    # -- Phase 2 ---------------------------------------------------------------
    run_phase2()

    # -- Load samples (used by phases 3-7) ------------------------------------
    print("\n-- Loading data samples --")
    fm_df   = load_feature_matrix_sample(FEATURE_MATRIX_PATH)
    proj_df = load_projection_outcomes_sample(MASTER_TRAINING_DB)

    # -- Phase 3 ---------------------------------------------------------------
    run_phase3(fm_df, proj_df)

    # -- Phase 4 ---------------------------------------------------------------
    run_phase4(fm_df, proj_df)

    # -- Phase 5 ---------------------------------------------------------------
    run_phase5(fm_df, proj_df)

    # -- Phase 6 ---------------------------------------------------------------
    run_phase6(proj_df)

    # -- Phase 7 ---------------------------------------------------------------
    run_phase7(fm_df, proj_df)

    # -- Phase 8 – Report ------------------------------------------------------
    print("\n-- Phase 8: Generating Report --")
    findings = load_findings()
    write_markdown_report(findings, EDA_REPORT_PATH)

    # -- Summary ---------------------------------------------------------------
    n_issues   = sum(len(v) for v in findings.get('issues', {}).values() if isinstance(v, list))
    n_hcp      = len(findings.get('high_corr_pairs', []))
    n_tables   = len(findings.get('tables', []))
    n_leakage  = len(findings.get('issues', {}).get('potential_leakage', []))

    print("\n" + "=" * 60)
    print("  EDA COMPLETE")
    print("=" * 60)
    print(f"  Tables enumerated : {n_tables}")
    print(f"  Issues flagged    : {n_issues}")
    print(f"  High-corr pairs   : {n_hcp}")
    print(f"  Leakage suspects  : {n_leakage}")
    print(f"\n  Report   -> {EDA_REPORT_PATH}")
    print(f"  Findings -> {EDA_FINDINGS_PATH}")
    print(f"  Plots    -> {EDA_PLOTS_DIR}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
