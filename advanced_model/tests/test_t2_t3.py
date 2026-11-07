"""
T2 (fitted, attempt-based shrinkage) and T3 (daily injury report storage) tests.
Plans/T2_T3_IMPLEMENTATION_AUDIT.md.
"""
import asyncio
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

import config.settings as settings  # noqa: E402
from simulator import context_builder as cb  # noqa: E402

SP = cb.load_shrinkage_params()


def _player_db(rows_by_player):
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE box_scores (game_id TEXT, game_date TEXT, season TEXT, team_id INTEGER,
        player_name TEXT, minutes TEXT, fgm INTEGER, fga INTEGER, fg3m INTEGER, fg3a INTEGER, ftm INTEGER,
        fta INTEGER, oreb INTEGER, dreb INTEGER, ast INTEGER, stl INTEGER, blk INTEGER, tov INTEGER, pts INTEGER)""")
    for name, (fgm, fga, fg3m, fg3a, ftm, fta, tov) in rows_by_player.items():
        for i in range(15):
            conn.execute("INSERT INTO box_scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (f"00226{i:05d}", f"2026-10-{10 + i:02d}T00:00:00", '2026-27', 1, name, '30',
                          fgm, fga, fg3m, fg3a, ftm, fta, 1, 4, 3, 1, 0, tov,
                          2 * (fgm - fg3m) + 3 * fg3m + ftm))
    conn.commit()
    conn.close()
    return path


# Shooter: 6 3PA / 3 3PM, 4 2PA / 2 2PM, 4/4 FT per game. Big: no threes, 6/10 twos, 3/6 FT.
PLAYERS = {'Shooter': (5, 10, 3, 6, 4, 4, 1), 'Big': (6, 10, 0, 0, 3, 6, 2)}


class TestShrinkageMath(unittest.TestCase):
    def test_formula(self):
        self.assertAlmostEqual(cb._shrink_counts(45, 90, 0.36, 500), (45 + 0.36 * 500) / 590)
        self.assertEqual(cb._shrink_counts(0, 0, 0.36, 0), 0.36)

    @unittest.skipIf(SP is None, 'config/shrinkage_params.json missing')
    def test_params_sane(self):
        for stat in ('fg2_pct', 'fg3_pct_att', 'ft_pct', 'p_shoot_3', 'fta_per_fga', 'tov_rate'):
            self.assertIn(stat, SP)
            self.assertGreaterEqual(SP[stat]['k'], 0)
            h = SP[stat]['holdout']
            self.assertLessEqual(h['mse_fitted'], h['mse_legacy'] + 1e-12, stat)
            self.assertLessEqual(h['mse_fitted'], h['mse_no_shrink'] + 1e-12, stat)


@unittest.skipIf(SP is None, 'config/shrinkage_params.json missing')
class TestMatrixShrinkage(unittest.TestCase):
    def setUp(self):
        self.path = _player_db(PLAYERS)

    def tearDown(self):
        os.unlink(self.path)

    def _m(self, name, mode):
        with unittest.mock.patch.object(cb, 'DB_PATH', self.path), \
             unittest.mock.patch.object(settings, 'SHRINKAGE_MODEL', mode):
            return cb.get_player_stats_matrix(name, limit=15)

    def test_fitted_keeps_player_differences(self):
        s, b = self._m('Shooter', 'fitted'), self._m('Big', 'fitted')
        self.assertEqual(s['_shrinkage'], 'fitted')
        self.assertGreater(s['p_shoot_3'], 0.5)
        self.assertLess(b['p_shoot_3'], 0.2)
        self.assertGreater(s['ft_pct'], b['ft_pct'] + 0.15)

    def test_engine_contract_fg3(self):
        s = self._m('Shooter', 'fitted')
        p = SP['fg3_pct_att']
        attempts = 6 * 15
        expected_3p = (3 * 15 + p['prior'] * p['k']) / (attempts + p['k'])
        self.assertAlmostEqual(s['fg3_pct'] / s['p_shoot_3'], expected_3p, places=6)
        self.assertAlmostEqual(s['p_shoot_2'], 1 - s['p_shoot_3'])

    def test_legacy_is_league_average(self):
        s, b = self._m('Shooter', 'legacy'), self._m('Big', 'legacy')
        self.assertEqual(s['_shrinkage'], 'legacy')
        self.assertLess(abs(s['p_shoot_3'] - b['p_shoot_3']), 0.01, 'legacy bug reproduced')

    def test_missing_params_falls_back_to_legacy(self):
        with unittest.mock.patch.object(cb, 'load_shrinkage_params', return_value=None):
            s = self._m('Shooter', 'fitted')
        self.assertEqual(s['_shrinkage'], 'legacy')


class TestShrinkageSetting(unittest.TestCase):
    def _get(self, env):
        e = {k: v for k, v in os.environ.items() if k not in ('STEP2_PROFILE', 'SHRINKAGE_MODEL')}
        e.update(env)
        return subprocess.run([sys.executable, '-c', 'import config.settings as s; print(s.SHRINKAGE_MODEL)'],
                              cwd=MODEL_DIR, env=e, capture_output=True, text=True)

    def test_default_fitted(self):
        self.assertEqual(self._get({}).stdout.strip(), 'fitted')

    def test_env_legacy_and_step2_legacy(self):
        self.assertEqual(self._get({'SHRINKAGE_MODEL': 'legacy'}).stdout.strip(), 'legacy')
        self.assertEqual(self._get({'STEP2_PROFILE': 'legacy'}).stdout.strip(), 'legacy')

    def test_invalid_raises(self):
        self.assertNotEqual(self._get({'SHRINKAGE_MODEL': 'x'}).returncode, 0)


class TestInjuryStorage(unittest.TestCase):
    ROWS = [{'game_date': '2026-11-04', 'player': 'A B', 'team': 'Atlanta Hawks', 'status': 'Out', 'reason': 'x'},
            {'game_date': '2026-11-04', 'player': 'C D', 'team': 'Boston Celtics', 'status': 'Questionable', 'reason': 'y'}]

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        c = sqlite3.connect(self.path)
        c.execute("""CREATE TABLE historical_injuries (game_date TEXT NOT NULL, player TEXT NOT NULL,
                     team TEXT NOT NULL, status TEXT NOT NULL, reason TEXT, PRIMARY KEY (game_date, player, team))""")
        c.execute("INSERT INTO historical_injuries VALUES ('2026-11-04','A B','Atlanta Hawks','Out','x')")
        c.commit()
        c.close()

    def tearDown(self):
        os.unlink(self.path)

    def _count(self):
        c = sqlite3.connect(self.path)
        n = c.execute("SELECT COUNT(*) FROM historical_injuries").fetchone()[0]
        c.close()
        return n

    def _fake(self, rows):
        async def f(*a, **k):
            return list(rows)
        return f

    def test_dry_run_writes_nothing(self):
        from etl import scrape_historical_injuries as s
        with unittest.mock.patch.object(s, 'fetch_all', self._fake(self.ROWS)), \
             unittest.mock.patch('builtins.print') as pr:
            s.run('2026-11-04', '2026-11-04', dry_run=True, db_path=self.path)
        self.assertEqual(self._count(), 1)
        self.assertTrue(any('1 of 2 rows are new' in str(c) for c in pr.call_args_list))

    def test_apply_inserts_only_new(self):
        from etl import scrape_historical_injuries as s
        with unittest.mock.patch.object(s, 'fetch_all', self._fake(self.ROWS)), \
             unittest.mock.patch('builtins.print'):
            s.run('2026-11-04', '2026-11-04', db_path=self.path)
        self.assertEqual(self._count(), 2)

    def test_empty_report_is_not_an_error_in_nightly_mode(self):
        from etl import scrape_historical_injuries as s
        with unittest.mock.patch.object(s, 'fetch_all', self._fake([])), \
             unittest.mock.patch('builtins.print'):
            s.run('2026-11-03', '2026-11-03', allow_empty=True, db_path=self.path)   # no SystemExit
            with self.assertRaises(SystemExit):
                s.run('2026-11-03', '2026-11-03', db_path=self.path)
        self.assertEqual(self._count(), 1)


class TestEngineCalibration(unittest.TestCase):
    """T2b: free-throw rule and fitted shooting penalties."""

    M = {'fg2_pct': 0.55, 'fg3_pct': 0.36 * 0.4, 'p_shoot_3': 0.4, 'fta_per_fga': 0.30, 'ft_pct': 0.8,
         'tov_rate': 0.12, 'avg_fga': 15, 'avg_ast': 3, 'avg_reb': 5, 'avg_stl': 1}

    def test_foul_probability_matches_player_fta_rate(self):
        from simulator.markov_engine import MarkovSimulator
        m = dict(self.M)
        f = MarkovSimulator.foul_on_miss_probability(m)
        p2, p3, ps3 = 0.55, 0.36, 0.4
        e_fta = ps3 * (0.05 * p3 + (1 - p3) * f * 3) + (1 - ps3) * (0.05 * p2 + (1 - p2) * f * 2)
        self.assertAlmostEqual(e_fta, 0.30, places=9)
        self.assertLess(f, 0.30, 'legacy used fta_per_fga itself as the foul chance')

    def test_engine_switch(self):
        from simulator.markov_engine import MarkovSimulator
        for mode, expect in (('v2', True), ('legacy', False)):
            h, a = {'H': dict(self.M)}, {'A': dict(self.M)}
            with unittest.mock.patch.object(settings, 'ENGINE_CALIBRATION', mode):
                MarkovSimulator(h, a)
            self.assertEqual('_foul_on_miss' in h['H'], expect, mode)

    def test_simulated_fta_rate(self):
        import random
        from simulator.markov_engine import MarkovSimulator
        with unittest.mock.patch.object(settings, 'ENGINE_CALIBRATION', 'v2'):
            eng = MarkovSimulator({'H': dict(self.M)}, {'A': dict(self.M)})
        random.seed(7)
        tracker = {p: {'PTS': 0, 'REB': 0, 'OREB': 0, 'DREB': 0, 'AST': 0, 'STL': 0, 'BLK': 0, 'TOV': 0,
                       'FG3M': 0, 'FG3A': 0, 'FTA': 0, 'FTM': 0} for p in ('H', 'A')}
        n = 40000
        for _ in range(n):
            eng.simulate_possession(True, tracker)
        shots = n - tracker['H']['TOV']
        self.assertAlmostEqual(tracker['H']['FTA'] / shots, 0.30, delta=0.01)

    def test_engine_calibration_setting(self):
        def get(env):
            e = {k: v for k, v in os.environ.items()
                 if k not in ('STEP2_PROFILE', 'SHRINKAGE_MODEL', 'ENGINE_CALIBRATION')}
            e.update(env)
            return subprocess.run([sys.executable, '-c', 'import config.settings as s; print(s.ENGINE_CALIBRATION)'],
                                  cwd=MODEL_DIR, env=e, capture_output=True, text=True)
        self.assertEqual(get({}).stdout.strip(), 'v2')
        self.assertEqual(get({'SHRINKAGE_MODEL': 'legacy'}).stdout.strip(), 'legacy', 'follows shrinkage')
        self.assertEqual(get({'SHRINKAGE_MODEL': 'legacy', 'ENGINE_CALIBRATION': 'v2'}).stdout.strip(), 'v2')
        self.assertNotEqual(get({'ENGINE_CALIBRATION': 'x'}).returncode, 0)

    def _roles(self):
        return {'Star': {'avg_pts': 30, 'avg_fga': 22, 'min_mean': 35, 'games': 10},
                'Starter': {'avg_pts': 12, 'avg_fga': 9, 'min_mean': 30, 'games': 10},
                'Other': {'avg_pts': 14, 'avg_fga': 10, 'min_mean': 32, 'games': 10},
                'Active': {'avg_pts': 10, 'avg_fga': 8, 'min_mean': 28, 'games': 10}}

    def test_injury_matrix_v2_uses_fitted_values(self):
        from etl import live_ingestion as li
        ep = {'mean_baseline_efg': 0.5, 'injury': {
            'star': {'coef': 0.0, 'significant': False}, 'starter': {'coef': 0.005, 'significant': True},
            'rotation': {'coef': 0.002, 'significant': True}, 'collapse': {'coef': 0.004, 'significant': True}}}
        team = {'Active': {'fg2_pct': 0.5, 'fg3_pct': 0.15}, 'Other': {'fg2_pct': 0.5, 'fg3_pct': 0.15}}
        with unittest.mock.patch.object(li, 'get_team_player_roles', return_value=self._roles()), \
             unittest.mock.patch.object(li, '_engine_penalties', return_value=ep), \
             unittest.mock.patch.object(settings, 'ENGINE_CALIBRATION', 'v2'), \
             unittest.mock.patch('builtins.print'):
            li.apply_dynamic_injury_matrix(team, 1, 'TST', None)
        # missing: Star (no effect, ns) + Starter (+0.005) -> x (1 + 0.005/0.5)
        self.assertAlmostEqual(team['Active']['fg2_pct'], 0.5 * 1.01, places=9)

    def test_injury_matrix_legacy_unchanged(self):
        from etl import live_ingestion as li
        team = {'Active': {'fg2_pct': 0.5, 'fg3_pct': 0.15}, 'Other': {'fg2_pct': 0.5, 'fg3_pct': 0.15}}
        with unittest.mock.patch.object(li, 'get_team_player_roles', return_value=self._roles()), \
             unittest.mock.patch.object(settings, 'ENGINE_CALIBRATION', 'legacy'), \
             unittest.mock.patch('builtins.print'):
            li.apply_dynamic_injury_matrix(team, 1, 'TST', None)
        self.assertAlmostEqual(team['Active']['fg2_pct'], 0.5 * 0.94 * 0.97, places=9)

    def test_usage_penalty_v2(self):
        from etl import live_ingestion as li
        ep = {'mean_baseline_efg': 0.5, 'usage_boost': {
            'per_unit_fga_increase': {'coef': -0.02, 'significant': True}, 'inc_cap': 1.5}}
        team = {'A': {'avg_fga': 10.0, 'fg2_pct': 0.5, 'fg3_pct': 0.15},
                'B': {'avg_fga': 10.0, 'fg2_pct': 0.5, 'fg3_pct': 0.15}}
        with unittest.mock.patch.object(li, '_engine_penalties', return_value=ep), \
             unittest.mock.patch.object(settings, 'ENGINE_CALIBRATION', 'v2'), \
             unittest.mock.patch('builtins.print'):
            li.redistribute_usage(team, {'A': 10, 'B': 10, 'Gone': 10}, 'TST')
        # each gets +5 FGA = +50% -> shooting x (1 - 0.02/0.5*0.5) = 0.98 (legacy: 0.94)
        self.assertAlmostEqual(team['A']['fg2_pct'], 0.5 * 0.98, places=9)


if __name__ == '__main__':
    unittest.main()
