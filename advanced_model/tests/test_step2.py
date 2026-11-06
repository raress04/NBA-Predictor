"""
Step 2 tests — Plans/STEP2_IMPLEMENTATION_AUDIT.md §1-§6.

  - settings profile switches (STEP2_PROFILE)
  - A: runtime bias off / legacy on
  - B: stage-2 gate on the raw scale vs legacy posterior + 75% rule
  - C: back-to-back detection (date format, date-aware) and fitted effects
  - D: date-aware return detection and fitted minutes adjustment
  - backtest leak fixes (before_date)
"""
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
from utils import adjustments  # noqa: E402

PARAMS = adjustments.load_params()


def _tmp_db():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE box_scores (game_id TEXT, game_date TEXT, season TEXT, team_id INTEGER,
                    player_id INTEGER, player_name TEXT, minutes TEXT, pts INTEGER, fga INTEGER,
                    reb INTEGER, ast INTEGER, oreb INTEGER, dreb INTEGER)""")
    conn.execute("CREATE TABLE games (game_id TEXT, game_date TEXT, matchup TEXT, season TEXT)")
    return conn, path


def _add(conn, gid, date, team, player, minutes, pts=10, fga=8, reb=4, ast=2, season='2026-27', pid=None):
    conn.execute("INSERT INTO box_scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                 (gid, f"{date}T00:00:00", season, team, pid or abs(hash(player)) % 10**6, player,
                  str(minutes), pts, fga, reb, ast, 1, reb - 1))


class TestSettingsProfile(unittest.TestCase):
    def _flags(self, profile):
        env = dict(os.environ)
        if profile is None:
            env.pop('STEP2_PROFILE', None)
        else:
            env['STEP2_PROFILE'] = profile
        code = ("import config.settings as s;"
                "print(s.STEP2_PROFILE, s.USE_RUNTIME_BIAS, s.CONFIDENCE_GATE, s.B2B_MODEL, "
                "s.RETURNING_MODEL, s.SHADOW_MODEL_VERSION)")
        return subprocess.run([sys.executable, '-c', code], cwd=MODEL_DIR, env=env,
                              capture_output=True, text=True)

    def test_default_is_step2(self):
        out = self._flags(None)
        self.assertEqual(out.stdout.split(), ['step2', 'False', 'raw', 'fitted', 'fitted', 'ERA_3_STEP2'])

    def test_legacy_profile(self):
        out = self._flags('legacy')
        self.assertEqual(out.stdout.split(), ['legacy', 'True', 'legacy', 'off', 'legacy', 'ERA_3'])

    def test_invalid_profile_raises(self):
        out = self._flags('bogus')
        self.assertNotEqual(out.returncode, 0)
        self.assertIn('STEP2_PROFILE', out.stderr)


class TestRuntimeBias(unittest.TestCase):
    def test_off_in_step2(self):
        from utils import bias
        with unittest.mock.patch.object(settings, 'USE_RUNTIME_BIAS', False), \
             unittest.mock.patch.object(bias, 'load_all_bias_corrections',
                                        return_value={('POINTS', 'starter'): 0.53}):
            self.assertEqual(bias.get_tiered_bias('points', 25.0), 0.0)

    def test_on_in_legacy(self):
        from utils import bias
        with unittest.mock.patch.object(settings, 'USE_RUNTIME_BIAS', True), \
             unittest.mock.patch.object(bias, 'load_all_bias_corrections',
                                        return_value={('POINTS', 'starter'): 0.53}):
            self.assertAlmostEqual(bias.get_tiered_bias('points', 25.0), 0.53)

    def test_prop_edge_median_unshifted(self):
        from simulator import parlay_builder as pb
        dist = [10] * 2000 + [20] * 3000
        with unittest.mock.patch.object(settings, 'USE_RUNTIME_BIAS', False):
            e = pb.compute_prop_edge(dist, 14.5, 'points')
        self.assertEqual(e['median'], 20)


class TestConfidenceGate(unittest.TestCase):
    """REBOUNDS OVER, raw 72%, no DB history: passes on the raw scale, fails legacy."""

    def _run(self, gate):
        from simulator import parlay_builder as pb
        reb = [5] * 1400 + [10] * 3600          # P(OVER 7.5) = 72%, median 10 (gap 2.5)
        props = {'E1': {'Reb Guy': {'rebounds': {'line': 7.5, 'over_odds': 1.9,
                                                  'under_odds': 1.9, 'book': 'dk'}}}}
        dists = {'Reb Guy': {'REB': reb}}
        minutes = {'Reb Guy': {'mean': 30.0, 'std': 3.0, 'team': 'Home'}}
        odds = {'E1': {'home_team': 'Home', 'away_team': 'Away'}}
        with unittest.mock.patch.object(settings, 'CONFIDENCE_GATE', gate), \
             unittest.mock.patch.object(settings, 'USE_RUNTIME_BIAS', False), \
             unittest.mock.patch.object(pb, 'persist_parlay', return_value=None), \
             unittest.mock.patch.object(pb, 'get_category_calibration', return_value=1.0):
            return pb.build_parlays([], odds, props, dists, {}, all_historical_rates={},
                                    all_player_minutes=minutes, game_date='2026-11-06')

    def test_raw_gate_passes(self):
        out = self._run('raw')
        self.assertEqual(len(out['all_picks']), 1, out['shadow_picks'])
        p = out['all_picks'][0]
        self.assertAlmostEqual(p['confidence'], 72.0, places=6)
        self.assertAlmostEqual(p['raw_sim_freq_pct'], 72.0, places=6)
        self.assertLess(p['posterior_conf'], 70.0, "posterior still computed and logged")

    def test_legacy_gate_rejects_with_75(self):
        out = self._run('legacy')
        self.assertEqual(out['all_picks'], [])
        row = out['shadow_picks'][0]
        self.assertEqual(row['ban_reason'], 'LOW_CONFIDENCE(min=75.0)')
        self.assertIsNotNone(row['posterior_conf'])

    def test_shadow_rows_carry_model_version(self):
        out = self._run('legacy')
        self.assertEqual(out['shadow_picks'][0]['model_version'], settings.SHADOW_MODEL_VERSION)


class TestBackToBack(unittest.TestCase):
    def setUp(self):
        self.conn, self.path = _tmp_db()
        _add(self.conn, '0022600010', '2026-11-03', 1, 'A', 30)
        _add(self.conn, '0022600011', '2026-11-03', 2, 'B', 30)
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        os.unlink(self.path)

    def test_detects_previous_day_in_stored_format(self):
        from etl.live_ingestion import is_back_to_back
        self.assertTrue(is_back_to_back(1, self.conn, '2026-11-04'))
        self.assertFalse(is_back_to_back(1, self.conn, '2026-11-05'))
        self.assertFalse(is_back_to_back(3, self.conn, '2026-11-04'))

    def test_independent_of_clock(self):
        from etl import live_ingestion as li
        with unittest.mock.patch.object(li, 'CURRENT_GAME_DATE', '2026-11-04'):
            self.assertTrue(li.is_back_to_back(1, self.conn))

    @unittest.skipIf(PARAMS is None, 'config/adjustment_params.json missing')
    def test_fitted_effects(self):
        bench = {'min_mean': 12.0, 'avg_fga': 6.0, 'avg_reb': 3.0, 'avg_ast': 1.0, 'avg_stl': 0.5,
                 'avg_blk': 0.2, 'avg_pts': 7.0, 'tov_rate': 0.10, 'fg2_pct': 0.5, 'fg3_pct': 0.14}
        out = adjustments.apply_b2b_adjustment(dict(bench), PARAMS)
        d = PARAMS['b2b']['roles']['bench']
        self.assertGreater(d['minutes_delta']['value'], 0)
        self.assertAlmostEqual(out['min_mean'], 12.0 + d['minutes_delta']['value'])
        self.assertAlmostEqual(out['avg_fga'], 6.0 * out['min_mean'] / 12.0)
        self.assertAlmostEqual(out['tov_rate'], 0.10 * d['tov_per_min_ratio']['value'])
        self.assertEqual(out['fg2_pct'], 0.5, "no shooting change on B2B")
        self.assertEqual(out['fg3_pct'], 0.14)

    def test_missing_params_is_noop(self):
        m = {'min_mean': 12.0, 'avg_fga': 6.0, 'tov_rate': 0.1}
        with unittest.mock.patch.object(adjustments, 'load_params', return_value=None):
            self.assertEqual(adjustments.apply_b2b_adjustment(dict(m)), m)


class TestReturningPlayers(unittest.TestCase):
    """Team 1 plays games on Oct 21..Nov 3 (one per day). 'Star' plays the first 8, misses 3."""

    DATES = [f"2026-10-{d}" for d in range(21, 32)] + ["2026-11-01", "2026-11-02", "2026-11-03"]

    def setUp(self):
        self.conn, self.path = _tmp_db()
        for i, d in enumerate(self.DATES):
            gid = f"00226{i:05d}"
            _add(self.conn, gid, d, 1, 'Filler', 30, pid=1)
            if i < 8:
                _add(self.conn, gid, d, 1, 'Star', 34, pid=2)
            elif i in (8, 9, 10):
                _add(self.conn, gid, d, 1, 'Star', 0, pid=2)     # DNP rows count as missed
            elif i == 11:
                _add(self.conn, gid, d, 1, 'Star', 25, pid=2)    # return game 1 (Nov 1)
            if i in (5, 6, 7):
                _add(self.conn, gid, d, 1, 'Newbie', 20, pid=3)  # 3 games, then "absent"
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        os.unlink(self.path)

    def test_return_game_1(self):
        from etl.live_ingestion import evaluate_returning_player
        info = evaluate_returning_player('Star', 1, self.conn, '2026-11-01')   # games 0..10 visible
        self.assertEqual(info, {'games_missed': 3, 'return_game': 1, 'baseline_min': 34.0})

    def test_return_game_2_and_future_ignored(self):
        from etl.live_ingestion import evaluate_returning_player
        info = evaluate_returning_player('Star', 1, self.conn, '2026-11-02')   # played Nov 1 (game 1)
        self.assertEqual(info['return_game'], 2)
        self.assertEqual(info['games_missed'], 3)

    def test_no_return_after_three_games(self):
        from etl.live_ingestion import evaluate_returning_player
        _add(self.conn, '0022699998', '2026-11-04', 1, 'Star', 30, pid=2)
        _add(self.conn, '0022699999', '2026-11-05', 1, 'Star', 30, pid=2)
        # Nov 2 and Nov 3 rows for Star do not exist -> missed; add them as played
        _add(self.conn, '0022600012', '2026-11-02', 1, 'Star', 30, pid=2)
        _add(self.conn, '0022600013', '2026-11-03', 1, 'Star', 30, pid=2)
        self.conn.commit()
        self.assertIsNone(evaluate_returning_player('Star', 1, self.conn, '2026-11-06'))

    def test_new_arrival_is_not_a_return(self):
        from etl.live_ingestion import evaluate_returning_player
        # Newbie played 3 games, then missed 3: fewer than 5 prior games -> not a return
        self.assertIsNone(evaluate_returning_player('Newbie', 1, self.conn, '2026-11-01'))

    def test_season_of(self):
        from etl.live_ingestion import season_of
        self.assertEqual(season_of('2026-10-20'), '2026-27')
        self.assertEqual(season_of('2027-03-01'), '2026-27')

    @unittest.skipIf(PARAMS is None, 'config/adjustment_params.json missing')
    def test_star_long_absence_gets_fewer_minutes(self):
        m = {'min_mean': 34.0, 'avg_fga': 20.0, 'avg_reb': 6.0, 'avg_ast': 5.0, 'avg_stl': 1.0,
             'avg_blk': 0.5, 'avg_pts': 28.0}
        out = adjustments.apply_return_adjustment(
            dict(m), {'games_missed': 25, 'return_game': 1, 'baseline_min': 34.0}, PARAMS)
        eff = PARAMS['returning']['minutes_effect']['star']['20+']['1']
        self.assertAlmostEqual(out['min_mean'], 34.0 * eff, places=6)
        self.assertLess(out['min_mean'], 30.0)
        self.assertLess(out['avg_stl'], 1.0)
        self.assertEqual(out['minutes_restriction_flag'], 'RETURNING_G1_20+')
        self.assertEqual(out['min_mean_baseline'], 34.0)

    @unittest.skipIf(PARAMS is None, 'config/adjustment_params.json missing')
    def test_flag_only_on_first_game_back(self):
        m = {'min_mean': 34.0, 'avg_fga': 20.0}
        out = adjustments.apply_return_adjustment(
            dict(m), {'games_missed': 6, 'return_game': 2, 'baseline_min': 34.0}, PARAMS)
        self.assertNotIn('minutes_restriction_flag', out)
        self.assertIn('return_info', out)

    @unittest.skipIf(PARAMS is None, 'config/adjustment_params.json missing')
    def test_params_sane(self):
        r = PARAMS['returning']
        for role, d in r['minutes_effect'].items():
            for b, g in d.items():
                for n, v in g.items():
                    self.assertTrue(0.4 <= v <= 1.3, (role, b, n, v))
        self.assertNotIn('minutes_ratio', r, 'runtime must use the control-adjusted effect')


class TestLeakFixes(unittest.TestCase):
    def setUp(self):
        self.conn, self.path = _tmp_db()
        _add(self.conn, '0022600001', '2026-10-25', 1, 'Early', 30, pts=10, fga=10)
        _add(self.conn, '0022600002', '2026-11-05', 1, 'Late', 30, pts=40, fga=25)   # after the date
        _add(self.conn, '0022500001', '2026-03-01', 1, 'LastSeason', 30, season='2025-26')
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        os.unlink(self.path)

    def test_roles_ignore_future_and_other_seasons(self):
        from etl.live_ingestion import get_team_player_roles
        roles = get_team_player_roles(1, self.conn, before_date='2026-11-01')
        self.assertEqual(set(roles), {'Early'})

    def test_usage_ignores_future_and_roster_csv(self):
        from etl.live_ingestion import get_team_historical_usage
        usage = get_team_historical_usage(1, self.conn, before_date='2026-11-01')
        self.assertEqual(usage, {'Early': 10.0})

    def test_hit_rates_ignore_future(self):
        from simulator import context_builder as cb
        with unittest.mock.patch.object(cb, 'DB_PATH', self.path):
            hr = cb.get_player_historical_hit_rates('Late', {'PTS': 20.5}, before_date='2026-11-01')
            self.assertEqual(hr, {})
            hr_all = cb.get_player_historical_hit_rates('Late', {'PTS': 20.5})
            self.assertIn('PTS', hr_all)


if __name__ == '__main__':
    unittest.main()
