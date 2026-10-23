# EDA Report -- Advanced Model

_Generated: 2026-05-10 23:26 UTC_


## Context & Principles

This report is produced by the EDA layer (`advanced_model/eda/`). It is **read-only** with respect to production databases. All thresholds and paths are sourced from `eda_config.py` and `config/settings.py`. Anti-spaghetti rules apply: no duplicate logic, single responsibility per module.


## Data Sources & Schemas

**17 tables/views enumerated across all databases.**

| db | table | type | row_count |
| --- | --- | --- | --- |
| bet_tracker | daily_picks | table | 31922 |
| bet_tracker | game_results | table | 211 |
| bet_tracker | historical_odds_cache | table | 57491 |
| bet_tracker | parlay_legs | table | 62 |
| bet_tracker | parlays | table | 26 |
| bet_tracker | pick_results | table | 31686 |
| bet_tracker | projection_outcomes | table | 9938 |
| bet_tracker | projection_outcomes_raw | table | 32382 |
| bet_tracker | shadow_picks | table | 2844 |
| bet_tracker | sqlite_sequence | table | 8 |
| nba_data | box_scores | table | 201899 |
| nba_data | games | table | 9318 |
| nba_data | historical_injuries | table | 19712 |
| nba_data | sqlite_sequence | table | 1 |
| nba_data | teams | table | 30 |
| master_training | projection_outcomes | table | 76068 |
| master_training | training_clean | view | 65787 |


### Core EDA Tables

| db | table | role |
| --- | --- | --- |
| bet_tracker | daily_picks | picks |
| bet_tracker | pick_results | results |
| bet_tracker | parlays | parlays |
| bet_tracker | parlay_legs | parlay_legs |
| bet_tracker | projection_outcomes | outcomes_tracker |
| nba_data | box_scores | box_scores |
| nba_data | games | games |
| nba_data | teams | teams |
| master_training | projection_outcomes | training_data |


### Schema Details

**bet_tracker.daily_picks** (22 columns)

| name | type | notnull | pk |
| --- | --- | --- | --- |
| id | INTEGER | False | True |
| date | TEXT | True | False |
| bet_type | TEXT | True | False |
| stat_category | TEXT | True | False |
| player_or_team | TEXT | True | False |
| game | TEXT | True | False |
| direction | TEXT | True | False |
| line | REAL | False | False |
| odds | REAL | False | False |
| sim_confidence | REAL | False | False |
| blended_confidence | REAL | False | False |
| edge_pct | REAL | False | False |
| projected_margin | REAL | False | False |
| safety_margin | REAL | False | False |
| market_spread | REAL | False | False |
| source_file | TEXT | False | False |
| raw_sim_freq | REAL | False | False |
| clv | REAL | False | False |
| source_mode | TEXT | False | False |
| model_version | TEXT | False | False |
| synthetic_flag | TEXT | False | False |
| ml_correction_applied | TEXT | False | False |

**bet_tracker.game_results** (6 columns)

| name | type | notnull | pk |
| --- | --- | --- | --- |
| id | INTEGER | False | True |
| date | TEXT | True | False |
| home_team | TEXT | True | False |
| away_team | TEXT | True | False |
| home_score | INTEGER | False | False |
| away_score | INTEGER | False | False |

**bet_tracker.historical_odds_cache** (8 columns)

| name | type | notnull | pk |
| --- | --- | --- | --- |
| event_id | TEXT | False | False |
| date | TEXT | False | False |
| player_name | TEXT | False | False |
| stat_category | TEXT | False | False |
| line | REAL | False | False |
| source | TEXT | False | False |
| over_odds | REAL | False | False |
| under_odds | REAL | False | False |

**bet_tracker.parlay_legs** (11 columns)

| name | type | notnull | pk |
| --- | --- | --- | --- |
| id | INTEGER | False | True |
| parlay_id | INTEGER | True | False |
| leg_number | INTEGER | True | False |
| pick_id | INTEGER | False | False |
| player | TEXT | False | False |
| stat_category | TEXT | False | False |
| direction | TEXT | False | False |
| line | REAL | False | False |
| odds | REAL | False | False |
| sim_confidence | REAL | False | False |
| edge_pct | REAL | False | False |

**bet_tracker.parlays** (9 columns)

| name | type | notnull | pk |
| --- | --- | --- | --- |
| id | INTEGER | False | True |
| game_date | TEXT | True | False |
| model_version | TEXT | False | False |
| combined_odds | REAL | False | False |
| stake | REAL | False | False |
| payout | REAL | False | False |
| result | TEXT | False | False |
| roi | REAL | False | False |
| created_at | TEXT | False | False |

**bet_tracker.pick_results** (7 columns)

| name | type | notnull | pk |
| --- | --- | --- | --- |
| id | INTEGER | False | True |
| pick_id | INTEGER | True | False |
| actual_value | REAL | False | False |
| is_win | INTEGER | True | False |
| diff_val | REAL | False | False |
| diff_pct | REAL | False | False |
| resolved_at | TEXT | False | False |

**bet_tracker.projection_outcomes** (12 columns)

| name | type | notnull | pk |
| --- | --- | --- | --- |
| id | INTEGER | False | True |
| game_date | TEXT | True | False |
| player | TEXT | True | False |
| category | TEXT | True | False |
| projected_value | REAL | True | False |
| bookmaker_line | REAL | False | False |
| actual_value | REAL | True | False |
| model_bias | REAL | True | False |
| source_mode | TEXT | True | False |
| source_file | TEXT | True | False |
| synthetic_flag | TEXT | False | False |
| model_version | TEXT | False | False |

**bet_tracker.projection_outcomes_raw** (15 columns)

| name | type | notnull | pk |
| --- | --- | --- | --- |
| id | INTEGER | False | True |
| game_date | TEXT | False | False |
| matchup | TEXT | False | False |
| player | TEXT | False | False |
| category | TEXT | False | False |
| projected_value | REAL | False | False |
| bookmaker_line | REAL | False | False |
| actual_value | REAL | False | False |
| model_bias | REAL | False | False |
| direction | TEXT | False | False |
| confidence | REAL | False | False |
| edge_pct | REAL | False | False |
| is_valuable | INTEGER | False | False |
| source_file | TEXT | False | False |
| source_mode | TEXT | False | False |

**bet_tracker.shadow_picks** (16 columns)

| name | type | notnull | pk |
| --- | --- | --- | --- |
| id | INTEGER | False | True |
| game_date | TEXT | False | False |
| player | TEXT | False | False |
| stat_category | TEXT | False | False |
| direction | TEXT | False | False |
| line | REAL | False | False |
| projection | REAL | False | False |
| sim_confidence | REAL | False | False |
| edge_pct | REAL | False | False |
| ban_reason | TEXT | False | False |
| created_at | TEXT | False | False |
| actual_value | REAL | False | False |
| actual_result | INTEGER | False | False |
| clv | REAL | False | False |
| model_version | TEXT | False | False |
| resolved_at | TEXT | False | False |

**bet_tracker.sqlite_sequence** (2 columns)

| name | type | notnull | pk |
| --- | --- | --- | --- |
| name |  | False | False |
| seq |  | False | False |

**nba_data.box_scores** (24 columns)

| name | type | notnull | pk |
| --- | --- | --- | --- |
| id | INTEGER | False | True |
| season | TEXT | False | False |
| game_id | TEXT | False | False |
| game_date | TEXT | False | False |
| team_id | INTEGER | False | False |
| player_id | INTEGER | False | False |
| player_name | TEXT | False | False |
| minutes | TEXT | False | False |
| fgm | INTEGER | False | False |
| fga | INTEGER | False | False |
| fg3m | INTEGER | False | False |
| fg3a | INTEGER | False | False |
| ftm | INTEGER | False | False |
| fta | INTEGER | False | False |
| oreb | INTEGER | False | False |
| dreb | INTEGER | False | False |
| reb | INTEGER | False | False |
| ast | INTEGER | False | False |
| stl | INTEGER | False | False |
| blk | INTEGER | False | False |
| tov | INTEGER | False | False |
| pf | INTEGER | False | False |
| pts | INTEGER | False | False |
| plus_minus | INTEGER | False | False |

**nba_data.games** (4 columns)

| name | type | notnull | pk |
| --- | --- | --- | --- |
| game_id | TEXT | False | True |
| game_date | TEXT | False | False |
| matchup | TEXT | False | False |
| season | TEXT | False | False |

**nba_data.historical_injuries** (5 columns)

| name | type | notnull | pk |
| --- | --- | --- | --- |
| game_date | TEXT | True | True |
| player | TEXT | True | True |
| team | TEXT | True | True |
| status | TEXT | True | False |
| reason | TEXT | False | False |

**nba_data.sqlite_sequence** (2 columns)

| name | type | notnull | pk |
| --- | --- | --- | --- |
| name |  | False | False |
| seq |  | False | False |

**nba_data.teams** (4 columns)

| name | type | notnull | pk |
| --- | --- | --- | --- |
| team_id | INTEGER | False | True |
| abbreviation | TEXT | True | False |
| city | TEXT | True | False |
| nickname | TEXT | True | False |

**master_training.projection_outcomes** (19 columns)

| name | type | notnull | pk |
| --- | --- | --- | --- |
| id | INTEGER | False | False |
| game_date | TEXT | False | False |
| player | TEXT | False | False |
| category | TEXT | False | False |
| projected_value | REAL | False | False |
| bookmaker_line | REAL | False | False |
| actual_value | REAL | False | False |
| over_odds | REAL | False | False |
| under_odds | REAL | False | False |
| model_bias | REAL | False | False |
| source_mode | TEXT | False | False |
| source_file | TEXT | False | False |
| season | TEXT | False | False |
| edge_pct | TEXT | False | False |
| conf | TEXT | False | False |
| direction | TEXT | False | False |
| is_default_odds | INTEGER | False | False |
| ml_usable_classification | INTEGER | False | False |
| ml_usable_regression | INTEGER | False | False |

**master_training.training_clean** (21 columns)

| name | type | notnull | pk |
| --- | --- | --- | --- |
| id | INTEGER | False | False |
| game_date | TEXT | False | False |
| player | TEXT | False | False |
| category | TEXT | False | False |
| projected_value | REAL | False | False |
| bookmaker_line | REAL | False | False |
| actual_value | REAL | False | False |
| over_odds | REAL | False | False |
| under_odds | REAL | False | False |
| model_bias | REAL | False | False |
| source_mode | TEXT | False | False |
| source_file | TEXT | False | False |
| season | TEXT | False | False |
| edge_pct | TEXT | False | False |
| conf | TEXT | False | False |
| direction | TEXT | False | False |
| is_default_odds | INTEGER | False | False |
| ml_usable_classification | INTEGER | False | False |
| ml_usable_regression | INTEGER | False | False |
| implied_spread |  | False | False |
| direction_implied |  | False | False |


### Feature Matrix Metadata

- Rows: **73,821**  
- Columns: **42**  
- Detected label columns: `['model_bias']`


## Row Counts & Sampling

**bet_tracker**

- `daily_picks`: 31,922 rows  |  2026-01-01 -> 2026-05-05

- `game_results`: 211 rows  |  2026-03-01 -> 2026-03-23

- `historical_odds_cache`: 57,491 rows  |  2025-10-21 -> 2026-04-19

- `parlay_legs`: 62 rows

- `parlays`: 26 rows  |  2026-03-21 -> 2026-05-01

- `pick_results`: 31,686 rows

- `projection_outcomes`: 9,938 rows  |  2026-01-01 -> 20260108

- `projection_outcomes_raw`: 32,382 rows  |  2026-01-01 -> 2026-04-30

- `shadow_picks`: 2,844 rows  |  2026-03-17 -> 2026-05-11

- `sqlite_sequence`: 8 rows

**nba_data**

- `box_scores`: 201,899 rows  |  2019-09-30T00:00:00 -> 2026-04-25T00:00:00

- `games`: 9,318 rows  |  2019-09-30T00:00:00 -> 2026-04-25T00:00:00

- `historical_injuries`: 19,712 rows  |  2025-10-21 -> 2026-04-06

- `sqlite_sequence`: 1 rows

- `teams`: 30 rows

**master_training**

- `projection_outcomes`: 76,068 rows  |  2025-10-21 -> 2026-03-31

- `training_clean`: 65,787 rows  |  2025-10-21 -> 2026-03-31


### Sampling Parameters

- Feature matrix: 73,821 / 73,821 rows (ratio=1.0, sampled=False)


## Missingness & Univariate Statistics


### Missingness -- feature_matrix

Total columns: 42  |  High-missing (>20%): **4**

**High-missing columns:**

| name | n_null | pct_null |
| --- | --- | --- |
| bookmaker_line | 28082 | 0.380407 |
| over_odds | 30696 | 0.415817 |
| under_odds | 30696 | 0.415817 |
| gap | 28082 | 0.380407 |


### Missingness -- training_data

Total columns: 19  |  High-missing (>20%): **4**

**High-missing columns:**

| name | n_null | pct_null |
| --- | --- | --- |
| id | 6070 | 0.399053 |
| edge_pct | 15105 | 0.993031 |
| conf | 15105 | 0.993031 |
| direction | 15105 | 0.993031 |


> ⚠️ **CRITICAL:** The following required columns have nulls:

| column | pct_null | n_null |
| --- | --- | --- |
| bookmaker_line | 0.09881 | 1503 |
| over_odds | 0.133127 | 2025 |
| under_odds | 0.133127 | 2025 |


### Descriptive Statistics -- Feature Matrix (numeric)

| column | n | mean | median | std | min | max | p25 | p75 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| projected_value | 73821 | 6.34247 | 4.258 | 5.731292 | 0.0 | 49.0 | 2.4258 | 8.0 |
| bookmaker_line | 45739 | 6.676327 | 4.5 | 5.77727 | 0.5 | 33.5 | 2.5 | 8.5 |
| over_odds | 43125 | 1.892475 | 1.9009 | 0.186214 | 1.1923 | 3.0 | 1.8 | 1.9091 |
| under_odds | 43125 | 1.892475 | 1.9009 | 0.186214 | 1.1923 | 3.0 | 1.8 | 1.9091 |
| edge_pct | 73821 | 0.239473 | 0.0 | 2.821594 | 0.0 | 49.6 | 0.0 | 0.0 |
| conf | 73821 | 52.177385 | 50.0 | 4.899539 | 33.333333 | 100.0 | 50.0 | 53.487377 |
| gap | 45739 | -0.065743 | -0.0608 | 1.676923 | -11.5 | 16.5396 | -0.7672 | 0.6192 |
| over_implied | 73821 | 0.519411 | 0.5 | 0.041424 | 0.333333 | 0.838715 | 0.5 | 0.534874 |
| under_implied | 73821 | 0.519411 | 0.5 | 0.041424 | 0.333333 | 0.838715 | 0.5 | 0.534874 |
| vig | 73821 | 0.047252 | 0.0 | 0.070719 | 0.0 | 0.67743 | 0.0 | 0.069748 |
| odds_skew | 73821 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| direction_bin | 73821 | 0.294848 | 0.0 | 0.455978 | 0.0 | 1.0 | 0.0 | 1.0 |
| roll5_pts | 72784 | 12.215598 | 11.2 | 6.51954 | 0.0 | 40.4 | 7.2 | 16.0 |
| roll10_pts | 72784 | 12.115566 | 11.1 | 6.20186 | 0.2 | 33.8 | 7.4 | 15.8 |
| roll5_std_pts | 72793 | 5.460734 | 5.128353 | 2.527142 | 0.0 | 27.226825 | 3.63318 | 6.841053 |
| roll5_reb | 72784 | 4.490486 | 4.0 | 2.427472 | 0.0 | 16.4 | 2.8 | 5.8 |
| roll10_reb | 72784 | 4.469914 | 4.0 | 2.284142 | 0.0 | 14.5 | 2.8 | 5.6 |
| roll5_std_reb | 72793 | 2.195127 | 2.04939 | 1.064171 | 0.0 | 8.899438 | 1.414214 | 2.774887 |
| roll5_ast | 72784 | 2.807628 | 2.4 | 1.983602 | 0.0 | 13.2 | 1.4 | 3.8 |
| roll10_ast | 72784 | 2.783162 | 2.3 | 1.8784 | 0.0 | 12.3 | 1.4 | 3.7 |
| roll5_std_ast | 72793 | 1.596818 | 1.48324 | 0.86539 | 0.0 | 6.685806 | 1.0 | 2.04939 |
| roll5_min | 72784 | 24.960053 | 25.438333 | 7.067845 | 2.332333 | 43.143667 | 20.149 | 30.477 |
| roll10_min | 72784 | 24.797362 | 25.350667 | 6.836358 | 2.647167 | 41.103333 | 20.246667 | 30.038667 |
| last_game_pts | 72784 | 12.35299 | 11.0 | 8.414836 | 0.0 | 83.0 | 6.0 | 17.0 |
| last_game_reb | 72784 | 4.532947 | 4.0 | 3.256906 | 0.0 | 25.0 | 2.0 | 6.0 |
| last_game_ast | 72784 | 2.852138 | 2.0 | 2.577581 | 0.0 | 22.0 | 1.0 | 4.0 |
| days_rest | 72793 | 2.640144 | 2.0 | 2.794404 | 1.0 | 30.0 | 2.0 | 3.0 |
| games_played_season | 72793 | 38.880126 | 40.0 | 17.560751 | 0.0 | 81.0 | 26.0 | 52.0 |
| is_starter | 72793 | 0.759427 | 1.0 | 0.427434 | 0.0 | 1.0 | 1.0 | 1.0 |
| home_flag | 72793 | 0.501051 | 1.0 | 0.500002 | 0.0 | 1.0 | 0.0 | 1.0 |

_... and 8 more columns (see eda_findings.json)._


### Categorical Distributions

**category** (top 3 levels):

| value | count | pct |
| --- | --- | --- |
| ASSISTS | 25561 | 0.346256 |
| REBOUNDS | 24206 | 0.327901 |
| POINTS | 24054 | 0.325842 |

**source_mode** (top 3 levels):

| value | count | pct |
| --- | --- | --- |
| retroactive_live | 9141 | 0.600947 |
| historical | 4597 | 0.302216 |
| live | 1473 | 0.096838 |

**direction** (top 3 levels):

| value | count | pct |
| --- | --- | --- |
| nan | 15105 | 0.993031 |
| UNDER | 91 | 0.005983 |
| OVER | 15 | 0.000986 |

**season** (top 3 levels):

| value | count | pct |
| --- | --- | --- |
| 2025_26_live_raw | 30696 | 0.415817 |
| 2025_26_jan_apr | 27794 | 0.376505 |
| 2025_26_oct_dec | 15331 | 0.207678 |


## Outliers & Distribution Shapes

Columns with >5% IQR outliers: **12**

| column | q1 | q3 | lower_bound | upper_bound | n_outliers | pct_outliers |
| --- | --- | --- | --- | --- | --- | --- |
| projected_value | 2.4258 | 8.0 | -5.9355 | 16.3613 | 5453 | 0.073868 |
| bookmaker_line | 2.5 | 8.5 | -6.5 | 17.5 | 2886 | 0.063097 |
| over_odds | 1.8 | 1.9091 | 1.63635 | 2.07275 | 8298 | 0.192417 |
| under_odds | 1.8 | 1.9091 | 1.63635 | 2.07275 | 8298 | 0.192417 |
| conf | 50.0 | 53.487377 | 44.768935 | 58.718442 | 8759 | 0.118652 |
| gap | -0.7672 | 0.6192 | -2.8468 | 2.6988 | 3990 | 0.087234 |
| over_implied | 0.5 | 0.534874 | 0.447689 | 0.587184 | 8201 | 0.111093 |
| under_implied | 0.5 | 0.534874 | 0.447689 | 0.587184 | 8201 | 0.111093 |
| vig | 0.0 | 0.069748 | -0.104621 | 0.174369 | 5738 | 0.077729 |
| days_rest | 2.0 | 3.0 | 0.5 | 4.5 | 5295 | 0.072741 |
| is_starter | 1.0 | 1.0 | 1.0 | 1.0 | 17512 | 0.240573 |
| model_bias | -1.6078 | 1.935 | -6.922 | 7.2492 | 5752 | 0.077918 |


## Correlations & Feature–Target Relationships

High-correlation pairs (|r| > 0.9): **15**

| col_a | col_b | correlation |
| --- | --- | --- |
| over_odds | under_odds | 1.0 |
| over_implied | under_implied | 1.0 |
| over_odds | conf | -0.983732 |
| over_odds | over_implied | -0.983732 |
| over_odds | under_implied | -0.983732 |
| under_odds | conf | -0.983732 |
| under_odds | over_implied | -0.983732 |
| under_odds | under_implied | -0.983732 |
| projected_value | bookmaker_line | 0.957766 |
| roll5_ast | roll10_ast | 0.947522 |
| roll5_pts | roll10_pts | 0.946041 |
| roll5_min | roll10_min | 0.937537 |
| roll5_reb | roll10_reb | 0.935863 |
| over_implied | vig | 0.921594 |
| under_implied | vig | 0.921594 |


### Top Features by Correlation (-> model_bias)

| feature | correlation |
| --- | --- |
| gap | -0.229107 |
| direction_bin | -0.105333 |
| roll5_reb | 0.047662 |
| bookmaker_line | 0.045348 |
| roll10_reb | 0.041419 |
| last_game_reb | 0.041013 |
| last_game_ast | 0.034452 |
| last_game_pts | 0.03339 |
| roll5_ast | 0.032726 |
| opp_def_ast_30 | 0.031485 |
| roll5_pts | 0.030891 |
| roll5_std_reb | 0.030295 |
| opp_def_pts_30 | 0.028834 |
| roll5_min | 0.028047 |
| roll5_std_ast | 0.027237 |
| opp_def_reb_30 | 0.027008 |
| roll10_pts | 0.026964 |
| roll10_ast | 0.025329 |
| days_rest | -0.018807 |
| roll10_min | 0.015672 |


## Temporal Drift & Source-Mode Effects

Periods analysed: **6**

| period | row_count | mean_conf | mean_edge_pct |
| --- | --- | --- | --- |
| 2025-10 | 443 | None | None |
| 2025-11 | 1519 | None | None |
| 2025-12 | 1341 | None | None |
| 2026-01 | 4823 | None | None |
| 2026-02 | 3373 | None | None |
| 2026-03 | 3712 | 80.896226 | 31.321698 |


### Source-Mode Drift


```json
{
  "historical": {
    "n": 4597,
    "mean_conf": null,
    "mean_edge_pct": null
  },
  "live": {
    "n": 1473,
    "mean_conf": 80.896226,
    "mean_edge_pct": 31.321698
  },
  "retroactive_live": {
    "n": 9141,
    "mean_conf": null,
    "mean_edge_pct": null
  }
}
```


## Leakage & Selection Bias Checks

Suspected leakage columns: **0**


### Training vs Full Universe


```json
{
  "conf": {
    "full_mean": 80.896226,
    "train_mean": 52.177385,
    "full_std": 7.882385,
    "train_std": 4.899539,
    "full_n": 106,
    "train_n": 73821
  },
  "edge_pct": {
    "full_mean": 31.321698,
    "train_mean": 0.239473,
    "full_std": 7.910331,
    "train_std": 2.821594,
    "full_n": 106,
    "train_n": 73821
  },
  "over_odds": {
    "full_mean": 1.893046,
    "train_mean": 1.892475,
    "full_std": 0.179922,
    "train_std": 0.186214,
    "full_n": 13186,
    "train_n": 43125
  },
  "under_odds": {
    "full_mean": 1.893193,
    "train_mean": 1.892475,
    "full_std": 0.179866,
    "train_std": 0.186214,
    "full_n": 13186,
    "train_n": 43125
  }
}
```


## Recommendations for ML

- **Drop / merge highly-correlated pairs** (15 pairs with |r|>0.9). Retain the more interpretable feature per pair.
- **Impute or drop high-missing columns**: bookmaker_line, over_odds, under_odds, gap (and 0 more).
- **CRITICAL -- fix null critical columns** before training: bookmaker_line, over_odds, under_odds.