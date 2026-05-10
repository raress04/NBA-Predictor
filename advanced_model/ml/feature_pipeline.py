"""
feature_pipeline.py
Consolidates ML feature extraction.
"""

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


# --- FROM build_training_dataset.py ---

"""
advanced_model/ml.build_training_dataset.py
==================================================
Merges Oct-Dec 2025 and Jan-Apr 2026 retrospective projection_outcomes
into a single master training database.

Usage:
    python3 -m ml.build_training_dataset
"""
import sqlite3
import os
import sys
import pandas as pd

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_OCT_DEC  = os.path.join(MODEL_DIR, 'database', 'retrospective_2025_26_oct_dec.db')
DB_JAN_APR  = os.path.join(MODEL_DIR, 'database', 'retrospective_db.db')
DB_MASTER   = os.path.join(MODEL_DIR, 'database', 'master_training_data.db')

LABEL_OCT_DEC = '2025_26_oct_dec'
LABEL_JAN_APR = '2025_26_jan_apr'


def load_outcomes(db_path: str, season_label: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        print(f"[!] DB not found: {db_path}")
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM projection_outcomes", conn)
    conn.close()
    df['season'] = season_label
    print(f"[+] {season_label}: {len(df):,} rows loaded from {os.path.basename(db_path)}")
    return df


def verify_schemas(df1: pd.DataFrame, df2: pd.DataFrame):
    """Warn on mismatched columns; fill missing ones with NULL."""
    cols1 = set(df1.columns) - {'id', 'season'}
    cols2 = set(df2.columns) - {'id', 'season'}
    missing_in_2 = cols1 - cols2
    missing_in_1 = cols2 - cols1
    if missing_in_2 or missing_in_1:
        print(f"[!] Schema difference detected (will fill with NULL):")
        if missing_in_2:
            print(f"    Missing in jan_apr → filling: {missing_in_2}")
            for col in missing_in_2:
                df2[col] = None
        if missing_in_1:
            print(f"    Missing in oct_dec → filling: {missing_in_1}")
            for col in missing_in_1:
                df1[col] = None
    else:
        print("[✅] Schemas match.")


def build():
    print(f"[+] Building master training dataset...")

    df_oct = load_outcomes(DB_OCT_DEC, LABEL_OCT_DEC)
    df_jan = load_outcomes(DB_JAN_APR, LABEL_JAN_APR)

    verify_schemas(df_oct, df_jan)

    # Drop auto-increment id, will regenerate in master
    for df in [df_oct, df_jan]:
        if 'id' in df.columns:
            df.drop(columns=['id'], inplace=True)

    combined = pd.concat([df_oct, df_jan], ignore_index=True)
    print(f"[+] Combined total: {len(combined):,} rows")

    # Write to master DB
    conn = sqlite3.connect(DB_MASTER)
    conn.execute("DROP TABLE IF EXISTS projection_outcomes")
    combined.to_sql('projection_outcomes', conn, index=True, index_label='id', if_exists='replace')

    # Unique index to prevent duplicates
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_outcome
        ON projection_outcomes (game_date, player, category, source_file)
    """)
    conn.commit()

    # Verify
    rows = conn.execute("SELECT season, COUNT(*) AS n FROM projection_outcomes GROUP BY season").fetchall()
    total = conn.execute("SELECT COUNT(*) FROM projection_outcomes").fetchone()[0]
    conn.close()

    print(f"\n[+] Master DB written to: {DB_MASTER}")
    print(f"{'─'*40}")
    for season, n in rows:
        print(f"  {season:<25} {n:>8,} rows")
    print(f"  {'TOTAL':<25} {total:>8,} rows")
    print(f"{'─'*40}")

    if total < 40_000:
        print(f"[!] WARNING: total {total:,} rows < expected 40,000")
    else:
        print(f"[✅] Gate passed: {total:,} rows >= 40,000")


if __name__ == '__main__':
    build()


# --- FROM inference.py (live feature generation) ---

def get_player_features(player_name: str, game_date: str, opp_team_name: str, is_home: bool):
    """
    Fetches the last 10 games for a player and computes rolling stats.
    Fetches opponent defensive stats for the last 30 days.
    """
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'nba_data.db')
    conn = sqlite3.connect(db_path)
    
    # Player rolling stats
    df = pd.read_sql_query(
        "SELECT game_date, pts, reb, ast, minutes FROM box_scores WHERE player_name=? AND game_date < ? ORDER BY game_date DESC LIMIT 10",
        conn, params=(player_name, game_date)
    )
    
    features = {}
    
    if len(df) > 0:
        df = df.iloc[::-1].reset_index(drop=True) # chronological order
        df['minutes'] = pd.to_numeric(df['minutes'], errors='coerce').fillna(0)
        
        last_game = df.iloc[-1]
        features['last_game_pts'] = float(last_game['pts'])
        features['last_game_reb'] = float(last_game['reb'])
        features['last_game_ast'] = float(last_game['ast'])
        
        # Days rest
        try:
            last_dt = datetime.strptime(last_game['game_date'][:10], "%Y-%m-%d")
            curr_dt = datetime.strptime(game_date[:10], "%Y-%m-%d")
            rest = (curr_dt - last_dt).days
            features['days_rest'] = min(rest, 30.0)
        except:
            features['days_rest'] = 3.0
            
        # Roll 5
        roll5 = df.tail(5)
        features['roll5_pts'] = float(roll5['pts'].mean())
        features['roll5_reb'] = float(roll5['reb'].mean())
        features['roll5_ast'] = float(roll5['ast'].mean())
        features['roll5_min'] = float(roll5['minutes'].mean())
        features['roll5_std_pts'] = float(roll5['pts'].std()) if len(roll5) > 1 else 0.0
        features['roll5_std_reb'] = float(roll5['reb'].std()) if len(roll5) > 1 else 0.0
        features['roll5_std_ast'] = float(roll5['ast'].std()) if len(roll5) > 1 else 0.0
        
        # Roll 10
        roll10 = df
        features['roll10_pts'] = float(roll10['pts'].mean())
        features['roll10_reb'] = float(roll10['reb'].mean())
        features['roll10_ast'] = float(roll10['ast'].mean())
        features['roll10_min'] = float(roll10['minutes'].mean())
        
        features['is_starter'] = 1.0 if features['roll10_min'] > 20.0 else 0.0
        
    else:
        # Defaults if no history
        for k in ['last_game_pts','last_game_reb','last_game_ast','roll5_pts','roll5_reb','roll5_ast','roll5_min',
                 'roll5_std_pts','roll5_std_reb','roll5_std_ast','roll10_pts','roll10_reb','roll10_ast','roll10_min']:
            features[k] = np.nan
        features['days_rest'] = 3.0
        features['is_starter'] = 0.0

    features['home_flag'] = 1.0 if is_home else 0.0
    
    # Games played season
    season_start = '2025-10-01' if game_date > '2025-10-01' else '2024-10-01'
    gps = conn.execute(
        "SELECT COUNT(*) FROM box_scores WHERE player_name=? AND game_date >= ? AND game_date < ?",
        (player_name, season_start, game_date)
    ).fetchone()[0]
    features['games_played_season'] = float(gps)
    
    # Opponent defensive stats
    opp_team_id = conn.execute("SELECT team_id FROM teams WHERE abbreviation=?", (opp_team_name,)).fetchone()
    if opp_team_id:
        opp_id = opp_team_id[0]
        # Get all games the opponent played in last 30 days
        def_df = pd.read_sql_query("""
            SELECT b.pts, b.reb, b.ast 
            FROM box_scores b
            WHERE b.team_id != ?
              AND b.game_id IN (SELECT DISTINCT game_id FROM box_scores WHERE team_id = ?)
              AND b.game_date >= date(?, '-30 days')
              AND b.game_date < ?
        """, conn, params=(opp_id, opp_id, game_date, game_date))
        
        if len(def_df) > 0:
            features['opp_def_pts_30'] = float(def_df['pts'].mean()) * 5 # Approx team points
            features['opp_def_reb_30'] = float(def_df['reb'].mean()) * 5
            features['opp_def_ast_30'] = float(def_df['ast'].mean()) * 5
        else:
            features['opp_def_pts_30'] = np.nan
            features['opp_def_reb_30'] = np.nan
            features['opp_def_ast_30'] = np.nan
    else:
        features['opp_def_pts_30'] = np.nan
        features['opp_def_reb_30'] = np.nan
        features['opp_def_ast_30'] = np.nan

    conn.close()
    return features


def build_inference_vector(cat: str, base_features: dict, projected: float, line: float, 
                           edge_pct: float, conf: float, over_odds: float, under_odds: float,
                           meta: dict):
    vec = base_features.copy()
    vec['projected_value'] = float(projected)
    vec['bookmaker_line'] = float(line)
    vec['edge_pct'] = float(edge_pct)
    vec['conf'] = float(conf)
    vec['gap'] = float(projected - line)
    
    oi = (1.0 / over_odds) if over_odds > 0 else 0.5
    ui = (1.0 / under_odds) if under_odds > 0 else 0.5
    vec['over_implied'] = oi
    vec['under_implied'] = ui
    vec['vig'] = max(0.0, oi + ui - 1.0)
    vec['odds_skew'] = oi - ui
    
    # Infer direction
    if vec['gap'] >= 0:
        vec['direction_bin'] = 1.0 # OVER
    else:
        vec['direction_bin'] = 0.0 # UNDER
        
    # Build list in exact order of feature_cols
    ordered = []
    medians = meta['medians']
    for c in meta['feature_cols']:
        val = vec.get(c, np.nan)
        if pd.isna(val) or val is None:
            val = medians.get(c, 0.0)
        ordered.append(float(val))
        
    return np.array(ordered).reshape(1, -1)


