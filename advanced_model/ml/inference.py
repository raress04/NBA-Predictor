"""
advanced_model/ml/inference.py
==============================
Live inference module for XGBoost Calibration and Residual Correction models.
Computes features on the fly using sqlite queries.
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from config import settings

_models_calibration = {}
_models_residual = {}
_metadata_calibration = {}
_metadata_residual = {}
_initialized = False

def init_models():
    global _initialized
    if _initialized: return
    _initialized = True

    try:
        import xgboost as xgb
    except ImportError:
        print("[!] XGBoost not installed. ML features disabled.")
        return

    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), settings.ML_MODEL_PATH)
    
    for cat in ['POINTS', 'REBOUNDS', 'ASSISTS']:
        # Load Calibration Models
        if settings.USE_ML_CALIBRATION:
            calib_path = os.path.join(model_dir, f'xgb_calibration_{cat}_v2.json')
            meta_path = calib_path.replace('.json', '_meta.json')
            if os.path.exists(calib_path) and os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    _metadata_calibration[cat] = json.load(f)
                bst = xgb.Booster()
                bst.load_model(calib_path)
                _models_calibration[cat] = bst

        # Load Residual Models
        if settings.USE_ML_RESIDUAL_CORRECTION:
            resid_path = os.path.join(model_dir, f'xgb_residual_{cat}_v2.json')
            meta_path = resid_path.replace('.json', '_meta.json')
            if os.path.exists(resid_path) and os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    _metadata_residual[cat] = json.load(f)
                bst = xgb.Booster()
                bst.load_model(resid_path)
                _models_residual[cat] = bst


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


def apply_ml_corrections(player_name: str, stat_category: str, game_date: str, opp_team_name: str, is_home: bool,
                         projected: float, line: float, edge_pct: float, conf: float, over_odds: float, under_odds: float):
    init_models()
    cat = stat_category.upper()
    
    result = {
        'ml_correction_applied': 'NONE',
        'projected': projected,
        'confidence': conf
    }
    
    if not settings.USE_ML_CALIBRATION and not settings.USE_ML_RESIDUAL_CORRECTION:
        return result
        
    if cat not in _metadata_calibration and cat not in _metadata_residual:
        return result
        
    import xgboost as xgb
    base_features = get_player_features(player_name, game_date, opp_team_name, is_home)
    
    applied = []
    
    # 1. Residual Correction
    if settings.USE_ML_RESIDUAL_CORRECTION and cat in _models_residual:
        meta = _metadata_residual[cat]
        X = build_inference_vector(cat, base_features, projected, line, edge_pct, conf, over_odds, under_odds, meta)
        dmatrix = xgb.DMatrix(X, feature_names=meta['feature_cols'])
        bias = _models_residual[cat].predict(dmatrix)[0]
        result['projected'] = projected + float(bias)
        applied.append('RESIDUAL')
        # Re-eval features for calibration with new projection
        projected = result['projected']
        
    # 2. Calibration
    if settings.USE_ML_CALIBRATION and cat in _models_calibration:
        meta = _metadata_calibration[cat]
        X = build_inference_vector(cat, base_features, projected, line, edge_pct, conf, over_odds, under_odds, meta)
        dmatrix = xgb.DMatrix(X, feature_names=meta['feature_cols'])
        prob = _models_calibration[cat].predict(dmatrix)[0]
        
        # The model predicts probability of OVER.
        # If the pick is UNDER (projected < line), the confidence is 1 - prob.
        if projected < line:
            prob = 1.0 - prob
            
        result['confidence'] = float(prob) * 100.0
        applied.append('CALIBRATION')
        
    if applied:
        result['ml_correction_applied'] = 'BOTH' if len(applied) == 2 else applied[0]
        
    return result

