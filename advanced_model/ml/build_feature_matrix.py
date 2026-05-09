"""
advanced_model/ml.build_feature_matrix.py
================================================
Build a feature matrix from master_training_data.db + nba_data.db for ML residual modeling.

Feature groups:
  A) Model outputs  : projected_value, bookmaker_line, gap, odds signals
  B) Player context : rolling 5/10 game averages, minutes, rest, home/away
  C) Matchup context: opponent defensive allowed stats (30-day rolling)

Strict no-look-ahead: all rolling windows use only data BEFORE game_date.

Output: database/feature_matrix.parquet

Usage:
    python3 -m ml.build_feature_matrix
"""

import os, sys
import sqlite3
import pandas as pd
import numpy as np

MODEL_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_DB   = os.path.join(MODEL_DIR, 'database', 'master_training_data.db')
NBA_DB      = os.path.join(MODEL_DIR, 'database', 'nba_data.db')
OUT_PATH    = os.path.join(MODEL_DIR, 'database', 'feature_matrix_v2.parquet')

CATEGORIES  = ['POINTS', 'REBOUNDS', 'ASSISTS']
SEASON_START = pd.Timestamp('2025-10-01')


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_date(s):
    """Normalize 2025-10-21T00:00:00 or 2025-10-21 → 2025-10-21."""
    return str(s)[:10]

def get_home_flag(abbrev: str, matchup: str) -> int:
    """Return 1 if the player's team is home, 0 if away."""
    matchup = matchup.strip()
    if 'vs.' in matchup:
        home = matchup.split('vs.')[0].strip()
        return 1 if abbrev == home else 0
    elif '@' in matchup:
        away = matchup.split('@')[0].strip()
        return 0 if abbrev == away else 1
    return 0


# ── Step 1: Load projection_outcomes ─────────────────────────────────────────

def load_projections() -> pd.DataFrame:
    print("[1] Loading projection_outcomes from master_training_data.db ...")
    conn = sqlite3.connect(MASTER_DB)
    df = pd.read_sql_query(
        "SELECT game_date, player, category, projected_value, bookmaker_line, "
        "actual_value, model_bias, over_odds, under_odds, edge_pct, conf, direction, season "
        "FROM projection_outcomes "
        "WHERE category IN ('POINTS','REBOUNDS','ASSISTS') "
        "AND actual_value IS NOT NULL",
        conn
    )
    conn.close()
    df['game_date'] = pd.to_datetime(df['game_date'].apply(normalize_date))
    print(f"    {len(df):,} rows after filtering to P/R/A with actual+line")
    return df


# ── Step 2: Load box_scores + games + teams ───────────────────────────────────

def load_box_scores() -> pd.DataFrame:
    print("[2] Loading box_scores + games + teams from nba_data.db ...")
    conn = sqlite3.connect(NBA_DB)
    bs = pd.read_sql_query(
        "SELECT b.game_id, b.game_date, b.player_name, b.team_id, "
        "b.pts, b.reb, b.ast, b.minutes "
        "FROM box_scores b",
        conn
    )
    games = pd.read_sql_query("SELECT game_id, matchup FROM games", conn)
    teams = pd.read_sql_query("SELECT team_id, abbreviation FROM teams", conn)
    conn.close()

    bs['game_date'] = pd.to_datetime(bs['game_date'].apply(normalize_date))
    # Parse minutes (stored as float-string like "40.983...")
    bs['minutes'] = pd.to_numeric(bs['minutes'], errors='coerce').fillna(0)

    bs = bs.merge(games, on='game_id', how='left')
    bs = bs.merge(teams, on='team_id', how='left')

    # Home flag per player row
    bs['home_flag'] = bs.apply(
        lambda r: get_home_flag(r['abbreviation'], r['matchup']) if pd.notna(r['matchup']) else 0,
        axis=1
    )

    # Opponent abbreviation
    def get_opp(abbrev, matchup):
        if not isinstance(matchup, str):
            return None
        if 'vs.' in matchup:
            parts = matchup.split('vs.')
            home, away = parts[0].strip(), parts[1].strip()
            return away if abbrev == home else home
        elif '@' in matchup:
            parts = matchup.split('@')
            away, home = parts[0].strip(), parts[1].strip()
            return home if abbrev == away else away
        return None

    bs['opp_abbrev'] = bs.apply(lambda r: get_opp(r['abbreviation'], r['matchup']), axis=1)
    # Map opp abbreviation → opp team_id
    abbrev_to_id = teams.set_index('abbreviation')['team_id'].to_dict()
    bs['opp_team_id'] = bs['opp_abbrev'].map(abbrev_to_id)

    print(f"    {len(bs):,} box score rows loaded")
    return bs


# ── Step 3: Rolling player features (strict no look-ahead) ───────────────────

def build_player_rolling(bs: pd.DataFrame) -> pd.DataFrame:
    print("[3] Computing per-player rolling features ...")
    bs = bs.sort_values(['player_name', 'game_date']).copy()

    results = []
    for player, grp in bs.groupby('player_name', sort=False):
        grp = grp.sort_values('game_date').reset_index(drop=True)

        # --- Stat rolling (shift(1) = exclude current game) ---
        for stat, col in [('pts', 'pts'), ('reb', 'reb'), ('ast', 'ast'), ('min', 'minutes')]:
            grp[f'roll5_{stat}']     = grp[col].shift(1).rolling(5,  min_periods=1).mean()
            grp[f'roll10_{stat}']    = grp[col].shift(1).rolling(10, min_periods=1).mean()
            grp[f'roll5_std_{stat}'] = grp[col].shift(1).rolling(5,  min_periods=2).std().fillna(0)

        grp['last_game_pts'] = grp['pts'].shift(1)
        grp['last_game_reb'] = grp['reb'].shift(1)
        grp['last_game_ast'] = grp['ast'].shift(1)

        # Days rest
        grp['days_rest'] = grp['game_date'].diff().dt.days.clip(upper=30).fillna(3)

        # Games played in current season (count of prior season games before current row)
        in_season = (grp['game_date'] >= SEASON_START).astype(int)
        grp['games_played_season'] = in_season.shift(1).cumsum().fillna(0)

        # Starter proxy: rolling avg minutes > 20
        grp['is_starter'] = (grp['roll10_min'] > 20).astype(int)

        results.append(grp)

    out = pd.concat(results, ignore_index=True)
    print(f"    Rolling features computed for {out['player_name'].nunique():,} players")
    return out


# ── Step 4: Opponent defensive context (30-day rolling allowed stats) ─────────

def build_opp_defense(bs: pd.DataFrame) -> pd.DataFrame:
    """
    For each (team_id, game_date), compute rolling 30-day mean of pts/reb/ast
    that team ALLOWED (i.e., what opponents scored against them).
    Returns a DataFrame keyed by (opp_team_id, game_date) with opp_def_* columns.
    """
    print("[4] Computing opponent defensive context ...")

    # Build game-level totals per team
    game_totals = (
        bs.groupby(['game_id', 'team_id', 'game_date'], as_index=False)
        .agg(team_pts=('pts', 'sum'), team_reb=('reb', 'sum'), team_ast=('ast', 'sum'))
    )

    # Self-join on game_id to find what the OTHER team scored (= what this team allowed)
    gt = game_totals[['game_id', 'team_id', 'game_date', 'team_pts', 'team_reb', 'team_ast']]
    gt_opp = gt.rename(columns={
        'team_id': 'opp_team_id',
        'team_pts': 'opp_scored_pts',
        'team_reb': 'opp_scored_reb',
        'team_ast': 'opp_scored_ast',
        'game_date': '_gd'
    })
    merged = gt.merge(gt_opp, on='game_id')
    merged = merged[merged['team_id'] != merged['opp_team_id']]

    # allowed[team_id, game_date] = what opponent scored = what team allowed
    allowed = merged[['team_id', 'game_date',
                       'opp_scored_pts', 'opp_scored_reb', 'opp_scored_ast']].copy()
    allowed = allowed.sort_values(['team_id', 'game_date'])

    # Per team: compute rolling 30-day mean of allowed stats (no look-ahead: shift(1) first)
    rolling_parts = []
    for tid, grp in allowed.groupby('team_id', sort=False):
        grp = grp.sort_values('game_date').set_index('game_date')
        rolled = (
            grp[['opp_scored_pts', 'opp_scored_reb', 'opp_scored_ast']]
            .shift(1)
            .rolling('30D', min_periods=1)
            .mean()
            .rename(columns={
                'opp_scored_pts': 'opp_def_pts_30',
                'opp_scored_reb': 'opp_def_reb_30',
                'opp_scored_ast': 'opp_def_ast_30',
            })
        )
        rolled['opp_team_id'] = tid
        rolling_parts.append(rolled.reset_index())

    opp_def = pd.concat(rolling_parts, ignore_index=True)
    print(f"    Opponent defense context built for {opp_def['opp_team_id'].nunique()} teams")
    return opp_def


# ── Step 5: Assemble feature matrix ──────────────────────────────────────────

def assemble(proj: pd.DataFrame, bs_rolling: pd.DataFrame, opp_def: pd.DataFrame) -> pd.DataFrame:
    print("[5] Assembling final feature matrix ...")

    # --- Group A: market/model signals ---
    proj['gap']              = proj['projected_value'] - proj['bookmaker_line']
    proj['over_implied']     = (1.0 / proj['over_odds'].replace(0, np.nan)).fillna(0.5)
    proj['under_implied']    = (1.0 / proj['under_odds'].replace(0, np.nan)).fillna(0.5)
    proj['vig']              = (proj['over_implied'] + proj['under_implied'] - 1.0).clip(0)
    proj['odds_skew']        = proj['over_implied'] - proj['under_implied']
    # Fill NULL edge_pct / conf (oct_dec rows) with 0 / implied_over respectively
    proj['edge_pct']         = pd.to_numeric(proj['edge_pct'], errors='coerce').fillna(0.0)
    proj['conf']             = pd.to_numeric(proj['conf'],     errors='coerce').fillna(proj['over_implied'] * 100)
    # Direction: derive from gap where NULL
    proj['direction_inferred'] = proj.apply(
        lambda r: r['direction'] if pd.notna(r['direction']) and r['direction'] != ''
        else ('OVER' if r['gap'] >= 0 else 'UNDER'),
        axis=1
    )
    proj['direction_bin']    = (proj['direction_inferred'] == 'OVER').astype(int)

    # --- Join rolling player features ---
    bs_key = bs_rolling[['player_name', 'game_date',
                          'roll5_pts', 'roll10_pts', 'roll5_std_pts',
                          'roll5_reb', 'roll10_reb', 'roll5_std_reb',
                          'roll5_ast', 'roll10_ast', 'roll5_std_ast',
                          'roll5_min', 'roll10_min',
                          'last_game_pts', 'last_game_reb', 'last_game_ast',
                          'days_rest', 'games_played_season', 'is_starter',
                          'home_flag', 'opp_team_id']].copy()
    bs_key = bs_key.rename(columns={'player_name': 'player'})
    # Keep only one row per (player, game_date) — take max (in case duplicate game entries)
    bs_key = bs_key.drop_duplicates(subset=['player', 'game_date'], keep='first')

    df = proj.merge(bs_key, on=['player', 'game_date'], how='left')

    # --- Join opponent defensive context ---
    opp_def['game_date'] = pd.to_datetime(opp_def['game_date'])
    df = df.merge(opp_def, on=['opp_team_id', 'game_date'], how='left')

    # --- One-hot encode category ---
    for cat in CATEGORIES:
        df[f'cat_{cat.lower()}'] = (df['category'] == cat).astype(int)

    # --- Drop leakage + non-feature columns ---
    drop_cols = ['actual_value', 'model_bias', 'direction', 'direction_inferred',
                 'source_mode', 'source_file', 'opp_abbrev', 'matchup',
                 'game_id', 'team_id', 'abbreviation']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print(f"    Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ── Step 6: Profile & save ────────────────────────────────────────────────────

def profile_and_save(df: pd.DataFrame):
    print("[6] NaN profile ...")
    total = len(df)
    nan_pct = (df.isna().sum() / total * 100).sort_values(ascending=False)
    high_nan = nan_pct[nan_pct > 5]
    if len(high_nan):
        print("    Columns with >5% NaN:")
        for col, pct in high_nan.items():
            print(f"      {col:<35} {pct:.1f}%")
    else:
        print("    All columns ≤5% NaN ✅")

    print(f"\n[7] Saving to {OUT_PATH} ...")
    df.to_parquet(OUT_PATH, index=False)
    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"    Saved. File size: {size_mb:.2f} MB")
    print(f"\n[✅] Feature matrix complete — {df.shape[0]:,} rows, {df.shape[1]} features")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  BUILD FEATURE MATRIX")
    print("=" * 60)

    proj      = load_projections()
    bs        = load_box_scores()
    bs_roll   = build_player_rolling(bs)
    opp_def   = build_opp_defense(bs)
    df        = assemble(proj, bs_roll, opp_def)

    # --- Target column (kept for train_test_split to use) ---
    # Re-attach model_bias from original projection_outcomes
    conn = sqlite3.connect(MASTER_DB)
    bias = pd.read_sql_query(
        "SELECT game_date, player, category, model_bias FROM projection_outcomes "
        "WHERE category IN ('POINTS','REBOUNDS','ASSISTS') "
        "AND actual_value IS NOT NULL",
        conn
    )
    conn.close()
    bias['game_date'] = pd.to_datetime(bias['game_date'].apply(normalize_date))
    bias = bias.drop_duplicates(subset=['game_date', 'player', 'category'])
    df = df.merge(bias, on=['game_date', 'player', 'category'], how='left')

    profile_and_save(df)


if __name__ == '__main__':
    main()
