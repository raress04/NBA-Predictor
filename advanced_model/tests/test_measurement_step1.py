"""
tests/test_measurement_step1.py
===============================
Step 1 of Plans/CRITICAL_BUGS_FOUND_06.11.md — "fix the measurement".

Covers:
  * ml_shadow_picks are graded from real box scores, never from the
    'live_pending' placeholder zeros in projection_outcomes (bug C4)
  * unplayed players stay unresolved, pushes are NULL, accents match
  * box_scores writes are idempotent (duplicate rows bug)
  * every prop candidate is logged once with direction-specific stage values,
    including candidates that used to be dropped silently (bug H3)
  * legacy stage classification used by the repair script
"""
import os
import sqlite3
import sys
import tempfile
import unittest
import unittest.mock

import pandas as pd

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

BOX_COLS = ['season', 'game_id', 'game_date', 'team_id', 'player_id', 'player_name',
            'minutes', 'pts', 'reb', 'ast']


def _tmp_path():
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    return f.name


def _make_nba_db(rows):
    path = _tmp_path()
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE box_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT, season TEXT, game_id TEXT, game_date TEXT,
        team_id INTEGER, player_id INTEGER, player_name TEXT, minutes TEXT,
        pts INTEGER, reb INTEGER, ast INTEGER)""")
    conn.executemany(f"INSERT INTO box_scores ({','.join(BOX_COLS)}) VALUES ({','.join('?' * len(BOX_COLS))})",
                     rows)
    conn.commit()
    conn.close()
    return path


def _make_bet_db():
    path = _tmp_path()
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE ml_shadow_picks (
        id INTEGER PRIMARY KEY AUTOINCREMENT, game_date TEXT, player TEXT, stat_category TEXT,
        direction TEXT, line REAL, projection REAL, sim_confidence REAL, xgb_confidence REAL,
        edge_pct REAL, model_version TEXT, cohort TEXT, created_at TEXT,
        actual_value REAL, is_win INTEGER, resolved_at TEXT)""")
    # The trap: live_pending placeholders with actual_value = 0.0
    conn.execute("""CREATE TABLE projection_outcomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT, game_date TEXT, player TEXT, category TEXT,
        projected_value REAL, bookmaker_line REAL, actual_value REAL, model_bias REAL,
        source_mode TEXT, source_file TEXT)""")
    conn.commit()
    conn.close()
    return path


D = '2026-11-04'
BOX = [
    ('2026-27', 'G1', f'{D}T00:00:00', 1, 101, 'Luka Dončić', '35.2', 41, 9, 8),
    ('2026-27', 'G1', f'{D}T00:00:00', 1, 101, 'Luka Dončić', '35.2', 41, 9, 8),  # duplicate row
    ('2026-27', 'G1', f'{D}T00:00:00', 1, 102, 'Bench Guy', '0', 0, 0, 0),         # DNP
    ('2026-27', 'G1', f'{D}T00:00:00', 2, 103, 'Push Man', '30.0', 20, 5, 4),
]


class TestMlShadowResolution(unittest.TestCase):

    def setUp(self):
        self.nba = _make_nba_db(BOX)
        self.bet = _make_bet_db()
        conn = sqlite3.connect(self.bet)
        picks = [
            (D, 'Luka Doncic', 'POINTS', 'OVER', 30.5),     # accent-less name, should WIN
            (D, 'Luka Doncic', 'ASSISTS', 'UNDER', 9.5),    # 8 < 9.5 -> WIN
            (D, 'Bench Guy', 'POINTS', 'UNDER', 4.5),       # DNP -> unresolved
            (D, 'Not Playing', 'POINTS', 'OVER', 10.5),     # no box row -> unresolved
            (D, 'Push Man', 'POINTS', 'OVER', 20.0),        # push -> NULL
        ]
        for gd, p, c, d, line in picks:
            conn.execute("INSERT INTO ml_shadow_picks (game_date, player, stat_category, direction, line) "
                         "VALUES (?,?,?,?,?)", (gd, p, c, d, line))
            conn.execute("INSERT INTO projection_outcomes (game_date, player, category, projected_value, "
                         "actual_value, model_bias, source_mode, source_file) VALUES (?,?,?,?,0.0,0,'live_pending','x')",
                         (gd, p, c, line))
        conn.commit()
        conn.close()

    def tearDown(self):
        os.unlink(self.nba)
        os.unlink(self.bet)

    def _rows(self):
        conn = sqlite3.connect(self.bet)
        rows = {(p, c): (a, w, r) for p, c, a, w, r in conn.execute(
            "SELECT player, stat_category, actual_value, is_win, resolved_at FROM ml_shadow_picks")}
        conn.close()
        return rows

    def test_graded_from_box_scores_not_placeholders(self):
        from ml.shadow_logger import resolve_ml_shadow_picks
        n = resolve_ml_shadow_picks(D, db_path=self.bet, nba_db=self.nba)
        rows = self._rows()
        self.assertEqual(rows[('Luka Doncic', 'POINTS')][:2], (41.0, 1),
                         "OVER must be graded against the real 41 pts, not the placeholder 0")
        self.assertEqual(rows[('Luka Doncic', 'ASSISTS')][:2], (8.0, 1))
        self.assertEqual(n, 3)

    def test_unplayed_and_push(self):
        from ml.shadow_logger import resolve_ml_shadow_picks
        resolve_ml_shadow_picks(D, db_path=self.bet, nba_db=self.nba)
        rows = self._rows()
        self.assertEqual(rows[('Bench Guy', 'POINTS')], (None, None, None), "DNP must stay unresolved")
        self.assertEqual(rows[('Not Playing', 'POINTS')], (None, None, None))
        actual, win, resolved = rows[('Push Man', 'POINTS')]
        self.assertEqual(actual, 20.0)
        self.assertIsNone(win, "push must not count as win or loss")
        self.assertIsNotNone(resolved)

    def test_force_regrades(self):
        from ml.shadow_logger import resolve_ml_shadow_picks
        conn = sqlite3.connect(self.bet)
        conn.execute("UPDATE ml_shadow_picks SET actual_value=0, is_win=0, resolved_at='old'")
        conn.commit()
        conn.close()
        self.assertEqual(resolve_ml_shadow_picks(D, db_path=self.bet, nba_db=self.nba), 0)
        resolve_ml_shadow_picks(D, force=True, db_path=self.bet, nba_db=self.nba)
        self.assertEqual(self._rows()[('Luka Doncic', 'POINTS')][:2], (41.0, 1))


class TestBoxScoresIO(unittest.TestCase):

    def test_insert_is_idempotent_and_replace_refreshes(self):
        from utils.box_scores_io import insert_box_scores, ensure_unique_index
        path = _make_nba_db([])
        conn = sqlite3.connect(path)
        df = pd.DataFrame([dict(zip(BOX_COLS, r)) for r in BOX[:1] + BOX[2:]])
        self.assertEqual(insert_box_scores(conn, df, BOX_COLS), 3)
        self.assertEqual(insert_box_scores(conn, df, BOX_COLS), 0, "second run must write nothing")
        self.assertTrue(ensure_unique_index(conn))
        df.loc[0, 'pts'] = 42
        self.assertEqual(insert_box_scores(conn, df, BOX_COLS, replace=True), 3)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM box_scores").fetchone()[0], 3)
        self.assertEqual(conn.execute("SELECT pts FROM box_scores WHERE player_id=101").fetchone()[0], 42)
        conn.close()
        os.unlink(path)

    def test_unique_index_refused_when_duplicates_exist(self):
        from utils.box_scores_io import ensure_unique_index
        path = _make_nba_db(BOX)
        conn = sqlite3.connect(path)
        self.assertFalse(ensure_unique_index(conn))
        conn.close()
        os.unlink(path)

    def test_load_box_actuals_collapses_duplicates(self):
        from utils.box_actuals import load_box_actuals
        path = _make_nba_db(BOX)
        actuals = load_box_actuals(D, nba_db=path)
        self.assertEqual(actuals['Luka Dončić']['POINTS'], 41)
        self.assertEqual(len(actuals), 3)
        os.unlink(path)


class TestCandidateLogging(unittest.TestCase):
    """build_parlays must log one row per evaluated prop with stage values."""

    def _run(self):
        from simulator import parlay_builder as pb
        under_pts = [16] * 4800 + [22] * 200          # P(OVER 20.5) = 4%  -> UNDER, raw 96%
        coin_reb = [4] * 2100 + [12] * 2900          # median 12, P(OVER 7.5) = 58% -> not valuable
        props = {
            'E1': {
                'Under Guy': {'points': {'line': 20.5, 'over_odds': 1.9, 'under_odds': 1.95, 'book': 'dk'}},
                'Coin Flip': {'rebounds': {'line': 7.5, 'over_odds': 1.9, 'under_odds': 1.9, 'book': 'dk'}},
            }
        }
        dists = {'Under Guy': {'PTS': under_pts}, 'Coin Flip': {'REB': coin_reb}}
        minutes = {
            'Under Guy': {'mean': 30.0, 'std': 8.0, 'team': 'Home'},   # -> MINUTES_RISK
            'Coin Flip': {'mean': 30.0, 'std': 3.0, 'team': 'Away'},
        }
        odds = {'E1': {'home_team': 'Home', 'away_team': 'Away'}}
        with unittest.mock.patch.object(pb, 'get_tiered_bias', return_value=0.0), \
             unittest.mock.patch.object(pb, 'persist_parlay', return_value=None), \
             unittest.mock.patch.object(pb, 'get_category_calibration', return_value=1.0):
            return pb.build_parlays([], odds, props, dists, {}, all_historical_rates={},
                                    all_player_minutes=minutes)

    def test_stage_values_are_direction_specific(self):
        out = self._run()
        rows = {r['player']: r for r in out['shadow_picks']}
        u = rows['Under Guy']
        self.assertTrue(u['ban_reason'].startswith('MINUTES_RISK'))
        self.assertEqual(u['direction'], 'UNDER')
        self.assertAlmostEqual(u['raw_sim_freq'], 96.0, places=6)
        self.assertAlmostEqual(u['sim_confidence'], 4.0, places=6, msg="legacy column keeps raw P(OVER)")
        self.assertEqual(u['conf_stage'], 'raw_over')
        self.assertEqual(u['min_std'], 8.0)
        self.assertEqual(u['under_odds'], 1.95)
        self.assertEqual(u['matchup'], 'Away @ Home')

    def test_previously_silent_candidates_are_logged(self):
        out = self._run()
        rows = {r['player']: r for r in out['shadow_picks']}
        c = rows['Coin Flip']
        self.assertTrue(c['ban_reason'].startswith('NOT_VALUABLE'), c['ban_reason'])
        self.assertAlmostEqual(c['raw_sim_freq'], 58.0, places=6)
        self.assertEqual(len(out['shadow_picks']), 2, "exactly one row per evaluated prop")

    def test_ingest_writes_stage_columns_on_old_schema(self):
        from etl import db_manager
        path = _tmp_path()
        conn = sqlite3.connect(path)
        conn.execute("""CREATE TABLE shadow_picks (id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date TEXT, player TEXT, stat_category TEXT, direction TEXT, line REAL,
            projection REAL, sim_confidence REAL, edge_pct REAL, ban_reason TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        conn.commit()
        conn.close()
        out = self._run()
        for r in out['shadow_picks']:
            r['game_date'] = D
        with unittest.mock.patch.object(db_manager, 'DB_PATH', path):
            self.assertEqual(db_manager.ingest_shadow_picks(out['shadow_picks']), 2)
        conn = sqlite3.connect(path)
        got = conn.execute("SELECT player, raw_sim_freq, conf_stage, min_std, superseded FROM shadow_picks "
                           "ORDER BY player").fetchall()
        conn.close()
        os.unlink(path)
        self.assertEqual(got[1][0], 'Under Guy')
        self.assertAlmostEqual(got[1][1], 96.0)
        self.assertEqual(got[1][2], 'raw_over')
        self.assertEqual(got[1][3], 8.0)
        self.assertEqual(got[1][4], 0)


class TestRepairStageClassification(unittest.TestCase):

    def test_classify_stage(self):
        from scripts.repair_measurement_20261106 import classify_stage
        cases = {
            ('MIN_GAP(gap=0.50<2.0)', 'POINTS'): 'raw_over',
            ('MINUTES_RISK(std=7.1,mean=25.0)', 'REBOUNDS'): 'raw_over',
            ('LOW_SAFE_CONF(60.0%<65.0%)', 'POINTS'): 'raw_over',
            ('LOW_CONF(60.2%<70.0%)', 'ASSISTS'): 'raw_over',
            ('LOW_EDGE(9.0%<15.0%)', 'POINTS'): 'raw_over',
            ('LOW_EDGE(min=2.0)', 'POINTS'): 'blended',
            ('LOW_CONFIDENCE(min=75.0)', 'POINTS'): 'blended',
            ('HIGH_VARIANCE(ci=0.30)', 'POINTS'): 'posterior',
            ('PARLAY_GATE_CLOSED (n<200 or MAE>8)', 'TOTAL'): 'posterior',
            ('BANNED_CATEGORY', 'SPREAD'): 'none',
            ('MARKET_INVERSION', 'SPREAD'): 'none',
        }
        for (reason, cat), expected in cases.items():
            self.assertEqual(classify_stage(reason, cat), expected, reason)


if __name__ == '__main__':
    unittest.main()
