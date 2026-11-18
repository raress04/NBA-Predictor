"""Pipeline fixes A1-A4 (Plans/PIPELINE_FIXES_AND_NOV13_17_AUDIT.md §2)."""
import os
import sqlite3
import sys
import tempfile
import unittest
import unittest.mock

import pandas as pd

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODEL_DIR)

import config.settings as settings  # noqa: E402


class TestUnderRawLabel(unittest.TestCase):
    """A1: an UNDER pick shows the UNDER share as 'raw', not P(OVER)."""

    def _run(self):
        from simulator import parlay_builder as pb
        reb = [5] * 3600 + [10] * 1400          # P(UNDER 7.5) = 72%, P(OVER) = 28%
        props = {'E1': {'Reb Guy': {'rebounds': {'line': 7.5, 'over_odds': 1.9,
                                                  'under_odds': 1.9, 'book': 'dk'}}}}
        dists = {'Reb Guy': {'REB': reb}}
        minutes = {'Reb Guy': {'mean': 30.0, 'std': 3.0, 'team': 'Home'}}
        odds = {'E1': {'home_team': 'Home', 'away_team': 'Away'}}
        with unittest.mock.patch.object(settings, 'CONFIDENCE_GATE', 'raw'), \
             unittest.mock.patch.object(settings, 'USE_RUNTIME_BIAS', False), \
             unittest.mock.patch.object(pb, 'persist_parlay', return_value=None), \
             unittest.mock.patch.object(pb, 'get_category_calibration', return_value=1.0):
            return pb, pb.build_parlays([], odds, props, dists, {}, all_historical_rates={},
                                        all_player_minutes=minutes, game_date='2026-11-06')

    def test_pick_carries_direction_share(self):
        pb, out = self._run()
        self.assertEqual(len(out['all_picks']), 1, out['shadow_picks'])
        p = out['all_picks'][0]
        self.assertEqual(p['direction'].upper(), 'UNDER')
        self.assertAlmostEqual(p['raw_sim_freq'], 0.72, places=6)
        self.assertAlmostEqual(p['raw_sim_freq'] * 100, p['raw_sim_freq_pct'], places=6)
        self.assertIn('(raw: 72.0%)', p['pick'])

    def test_report_line_shows_direction_share(self):
        pb, out = self._run()
        p = out['all_picks'][0]
        text = pb.format_parlay_output({'parlay_2_3': [p, p], 'parlay_4': [], 'all_picks': [p, p]})
        self.assertIn('(raw: 72.0%)', text)
        self.assertNotIn('(raw: 28.0%)', text)


class _FakeSynergy:
    """Stands in for synergyplaytypes.SynergyPlayTypes: fails `fails` times, then returns df."""
    calls = 0

    def __init__(self, fails, df):
        self.fails, self.df = fails, df

    def __call__(self, **kwargs):
        _FakeSynergy.calls += 1
        if _FakeSynergy.calls <= self.fails:
            raise TimeoutError('Read timed out')
        df = self.df

        class R:
            def get_data_frames(self_inner):
                return [df]
        return R()


class TestSynergyRetries(unittest.TestCase):
    """A2: retries with backoff, clear fallback, stop after repeated failures."""

    def setUp(self):
        from simulator import context_builder as cb
        self.cb = cb
        cb._synergy_failed_tables = 0
        _FakeSynergy.calls = 0
        self.df = pd.DataFrame({'TEAM_NAME': ['A', 'B'], 'PPP': [0.9, 1.1]})

    def tearDown(self):
        self.cb._synergy_failed_tables = 0
        for f in (self.cb.get_league_defensive_data, self.cb.get_league_offensive_data,
                  self.cb.get_player_defensive_data):
            f.cache_clear()

    def _patch(self, fails):
        return unittest.mock.patch.object(self.cb.synergyplaytypes, 'SynergyPlayTypes',
                                          _FakeSynergy(fails, self.df))

    def test_recovers_after_two_timeouts(self):
        waits = []
        with self._patch(fails=2):
            df = self.cb._fetch_synergy('t', sleep=waits.append, play_type_nullable='Isolation')
        self.assertIsNotNone(df)
        self.assertEqual(_FakeSynergy.calls, 3)
        self.assertEqual(len(waits), 2)
        self.assertTrue(5 <= waits[0] <= 7.5 and 15 <= waits[1] <= 22.5, waits)
        self.assertEqual(self.cb._synergy_failed_tables, 0)

    def test_gives_up_and_stops_calling(self):
        with self._patch(fails=99):
            for _ in range(self.cb.SYNERGY_MAX_FAILED_TABLES):
                self.assertIsNone(self.cb._fetch_synergy('t', sleep=lambda s: None))
            n = _FakeSynergy.calls
            self.assertEqual(n, self.cb.SYNERGY_TRIES * self.cb.SYNERGY_MAX_FAILED_TABLES)
            self.assertIsNone(self.cb._fetch_synergy('t', sleep=lambda s: None))
            self.assertEqual(_FakeSynergy.calls, n, 'no more API calls once given up')

    def test_defensive_fallback_logged(self):
        with self._patch(fails=99), \
             unittest.mock.patch.object(self.cb.time, 'sleep'), \
             unittest.mock.patch('builtins.print') as pr:
            team_def, avgs = self.cb.get_league_defensive_data('2099-00')
        self.assertEqual(team_def, {})
        self.assertTrue(all(v == 0.95 for v in avgs.values()))   # today's fallback kept
        printed = ' '.join(str(c.args[0]) for c in pr.call_args_list if c.args)
        self.assertIn('[SYNERGY FALLBACK] team defense', printed)

    def test_defensive_success_no_fallback(self):
        with self._patch(fails=0), \
             unittest.mock.patch.object(self.cb.time, 'sleep'), \
             unittest.mock.patch('builtins.print') as pr:
            team_def, avgs = self.cb.get_league_defensive_data('2098-99')
        self.assertAlmostEqual(team_def['A']['Isolation'], 0.9)
        self.assertAlmostEqual(avgs['Isolation'], 1.0)
        printed = ' '.join(str(c.args[0]) for c in pr.call_args_list if c.args)
        self.assertNotIn('FALLBACK', printed)


class TestDailyMetricsOut(unittest.TestCase):
    """A3: ml.daily_metrics accepts --out (as run_pipeline.sh passes it)."""

    def test_out_writes_report(self):
        from ml import daily_metrics as dm
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, 'sub', 'daily_metrics_2026-11-18.txt')
            argv = ['daily_metrics', '--date', '2026-11-18', '--no-save', '--out', out]
            with unittest.mock.patch.object(sys, 'argv', argv), \
                 unittest.mock.patch.object(dm, 'compute_daily_metrics', return_value={}), \
                 unittest.mock.patch.object(dm, 'format_report', return_value='REPORT TEXT'), \
                 unittest.mock.patch('builtins.print'):
                dm.main()
            with open(out, encoding='utf-8') as f:
                self.assertEqual(f.read(), 'REPORT TEXT')

    def test_pipeline_args_match(self):
        """Every flag run_pipeline.sh passes to ml.daily_metrics is accepted."""
        with open(os.path.join(MODEL_DIR, 'scripts', 'run_pipeline.sh'), encoding='utf-8') as f:
            line = next(l for l in f if l.startswith('python3 -m ml.daily_metrics'))
        flags = [t for t in line.split() if t.startswith('--')]
        with open(os.path.join(MODEL_DIR, 'ml', 'daily_metrics.py'), encoding='utf-8') as f:
            src = f.read()
        for flag in flags:
            self.assertIn(f"'{flag}'", src, flag)


class TestCompareRetroLive(unittest.TestCase):
    """A4: reads bet_tracker.db, rolling window, no crash on an empty window."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, 'bet_tracker.db')
        c = sqlite3.connect(self.db)
        c.execute("""CREATE TABLE projection_outcomes (game_date TEXT, player TEXT, category TEXT,
                     projected_value REAL, bookmaker_line REAL, actual_value REAL, source_mode TEXT)""")
        c.execute("""CREATE TABLE daily_picks (date TEXT, player_or_team TEXT, stat_category TEXT,
                     direction TEXT, line REAL, sim_confidence REAL, edge_pct REAL)""")
        c.execute("INSERT INTO projection_outcomes VALUES ('2026-11-13','Reb Guy','REBOUNDS',5.0,7.5,6,'live_pending')")
        c.execute("INSERT INTO daily_picks VALUES ('2026-11-13','Reb Guy','REBOUNDS','UNDER',7.5,72,10)")
        c.commit()
        c.close()

    def tearDown(self):
        self.tmp.cleanup()

    def _main(self, *args):
        from ml import compare_retro_live as crl
        out = os.path.join(self.tmp.name, 'report.txt')
        with unittest.mock.patch.object(crl, 'RETRO_DB', self.db), \
             unittest.mock.patch.object(crl, 'LIVE_DB', self.db), \
             unittest.mock.patch.object(sys, 'argv', ['crl', *args, '--out', out]), \
             unittest.mock.patch('builtins.print'):
            crl.main()
        with open(out, encoding='utf-8') as f:
            return f.read()

    def test_default_db_is_bet_tracker(self):
        from ml import compare_retro_live as crl
        self.assertEqual(os.path.basename(crl.RETRO_DB), 'bet_tracker.db')

    def test_default_window(self):
        from ml import compare_retro_live as crl
        self.assertEqual(crl.default_window('2026-11-17'), ('2026-11-11', '2026-11-17'))

    def test_match_in_window(self):
        text = self._main('--end', '2026-11-17')
        self.assertIn('2026-11-11 → 2026-11-17', text)
        self.assertIn('live_pending=1', text)
        self.assertIn('2026-11-13', text)
        self.assertIn('100.0%', text)

    def test_empty_window_writes_report(self):
        text = self._main('--start', '2026-02-01', '--end', '2026-02-28')
        self.assertIn('nothing to compare', text)

    def test_missing_db_not_created(self):
        from ml import compare_retro_live as crl
        missing = os.path.join(self.tmp.name, 'nope.db')
        with unittest.mock.patch.object(crl, 'RETRO_DB', missing):
            with self.assertRaises(sqlite3.OperationalError):
                crl.retro_source_modes('2026-11-01', '2026-11-07')
        self.assertFalse(os.path.exists(missing))


if __name__ == '__main__':
    unittest.main()
