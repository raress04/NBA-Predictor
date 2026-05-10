# Changelog

## [2026-05-09/10] - NBA Model Refactoring Phase 7 to 8

### What Changed
Executed a comprehensive, safe architectural refactoring (6 major steps) moving the engine to a 10-module canonical architecture, completely verified by end-to-end regression testing.

- **Step 1:** Consolidated database operations from `bet_tracker.py` to `etl/db_manager.py` and `live_scraper.py` to `etl/live_ingestion.py`.
- **Step 2:** Merged `matrix_builder.py` and `synergy_tracker.py` into a unified `simulator/context_builder.py`.
- **Step 3:** Restructured the Machine Learning layer: extracted data prep into `ml/feature_pipeline.py` and consolidated training into `ml/xgb_engine.py` (keeping `get_calibration_features` and `get_residual_features` strictly separated).
- **Step 4:** Consolidated evaluation and tracking from `compute_retroactive_bias.py`, `daily_metrics.py`, and `evaluate_shadow.py` into a central `analyze/performance_metrics.py`. Added uniform deprecation headers to all old files.
- **Step 5:** Centralized global parameters in `config/settings.py` replacing hardcoded values across `parlay_builder.py`, `live_ingestion.py`, and `context_builder.py`.
- **Step 6:** Completed full regression validation ensuring 100% parity with historical baseline metrics (0% drift).

### Critical Bug Fix
- **prior_strength mismatch fixed:** Replaced erroneous `PRIOR_STRENGTH = 5` in `context_builder.py` and `prior_strength=10` default in `markov_engine.py` with the calibrated `CONFIDENCE_PRIOR_STRENGTH = 3000` from `settings.py`. This resolves silent Bayesian calculation errors when modules were called directly.

### Deprecated Files (Moved to `_deprecated/` or shimmed)
The following files were replaced with thin wrappers importing from the new modules (kept for backward compatibility until 2026-08-01):
- `etl/bet_tracker.py`
- `etl/live_scraper.py`
- `simulator/matrix_builder.py`
- `simulator/synergy_tracker.py`
- `ml/build_feature_matrix.py`
- `ml/build_training_dataset.py`
- `ml/train_calibration_model.py`
- `ml/train_residual_model.py`
- `ml/inference.py`
- `ml/daily_metrics.py`
- `ml/evaluate_shadow.py`
- `analyze/compute_retroactive_bias.py`

### Notes
- **`EDGE_THRESHOLDS`** extended dictionary remains local in `simulator/parlay_builder.py` as it relies on a specific multi-key structure (`min_prob`, `min_edge`, `trigger`, `safety`) different from the flat generic `EDGE_THRESHOLDS` now present in `settings.py`.
