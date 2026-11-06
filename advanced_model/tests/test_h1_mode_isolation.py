"""
tests/test_h1_mode_isolation.py
===============================
H1 (Plans/CRITICAL_BUGS_FOUND_06.11.md), option B chosen by the owner on 2026-11-06:
  * a missing live prediction file never stops the nightly pipeline (warning + return 0),
  * backtest predictions are never substituted in a live run,
  * the grader attaches results only to picks of the same mode.
"""
import contextlib
import io
import os
import sqlite3
import sys
import tempfile
import unittest
import unittest.mock

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)


def _resolver(live_file, backtest_file):
    def resolve(date_iso, mode='live', base_dir=None, allow_legacy=True):
        return live_file if mode == 'live' else backtest_file
    return resolve


class TestPrepareAnalyzeNoCrossMode(unittest.TestCase):

    def _run(self, mode, live_file, backtest_file):
        from etl import prepare_analyze as pa
        out = io.StringIO()
        with unittest.mock.patch.object(pa, 'resolve_prediction_file',
                                        side_effect=_resolver(live_file, backtest_file)), \
             unittest.mock.patch.object(pa, 'validate_prediction_file', return_value=(True, 'ok')), \
             unittest.mock.patch.object(pa, 'check_games_and_box_scores',
                                        return_value=(7, 145, ['MIN @ OKC'])), \
             unittest.mock.patch.object(pa, 'resolve_results_file', return_value='/fake/results.txt'), \
             unittest.mock.patch.object(pa, 'validate_results_file', return_value=(True, 'ok')), \
             unittest.mock.patch.object(pa, 'generate_analysis_report') as gen_analysis, \
             unittest.mock.patch.object(pa, 'generate_results_file') as gen_results, \
             unittest.mock.patch.object(pa, 'fetch_box_scores_if_needed') as fetch_box, \
             contextlib.redirect_stdout(out):
            rc = pa.prepare_analyze(target_date='2026-03-15', mode=mode)
        return rc, out.getvalue(), (gen_analysis, gen_results, fetch_box)

    def test_live_missing_with_backtest_present_warns_and_continues(self):
        rc, log, mocks = self._run('live', live_file=None, backtest_file='/fake/backtest/predicts.txt')
        self.assertEqual(rc, 0, 'a missed live run must not stop the nightly pipeline')
        self.assertIn('live record has a gap', log)
        self.assertNotIn('backtest', log.lower(), 'the live run must not pick up the backtest file')
        for m in mocks:
            m.assert_not_called()   # nothing graded / generated from the backtest file

    def test_live_missing_without_backtest_warns_and_continues(self):
        rc, log, mocks = self._run('live', live_file=None, backtest_file=None)
        self.assertEqual(rc, 0)
        self.assertIn('live record has a gap', log)
        for m in mocks:
            m.assert_not_called()

    def test_backtest_missing_is_still_an_error(self):
        rc, log, mocks = self._run('backtest', live_file='/fake/live/predicts.txt', backtest_file=None)
        self.assertEqual(rc, 1)
        self.assertIn('ACTIONABLE ERROR', log)
        for m in mocks:
            m.assert_not_called()


class TestGraderSameModeOnly(unittest.TestCase):

    def setUp(self):
        fd, self.db = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        c = sqlite3.connect(self.db)
        c.executescript("""
            CREATE TABLE daily_picks (id INTEGER PRIMARY KEY, date TEXT, player_or_team TEXT,
                stat_category TEXT, direction TEXT, source_mode TEXT);
            CREATE TABLE pick_results (id INTEGER PRIMARY KEY AUTOINCREMENT, pick_id INTEGER,
                actual_value REAL, is_win INTEGER, diff_val REAL, diff_pct REAL, resolved_at TEXT,
                voided INTEGER DEFAULT 0);
            CREATE TABLE shadow_picks (id INTEGER PRIMARY KEY, game_date TEXT, player TEXT,
                stat_category TEXT, direction TEXT, line REAL, actual_value REAL, actual_result INTEGER);
            INSERT INTO daily_picks VALUES (1, '2026-03-15', 'Jamal Murray', 'ASSISTS', 'UNDER', 'live');
            INSERT INTO daily_picks VALUES (2, '2026-03-15', 'Jamal Murray', 'ASSISTS', 'UNDER', 'synthetic_circular');
        """)
        c.commit()
        c.close()
        self.props = [{'player': 'Jamal Murray', 'stat': 'ASSISTS', 'direction': 'UNDER',
                       'status': 'WON', 'actual': 6, 'diff_val': -1.5, 'diff_pct': -20.0}]

    def tearDown(self):
        os.unlink(self.db)

    def _graded_pick_ids(self):
        c = sqlite3.connect(self.db)
        ids = [r[0] for r in c.execute('SELECT pick_id FROM pick_results ORDER BY pick_id')]
        c.close()
        return ids

    def _grade(self, mode):
        from etl.analyze_predictions import write_db_results
        with contextlib.redirect_stdout(io.StringIO()):
            write_db_results('2026-03-15', self.props, [], {}, {}, mode=mode, db_path=self.db)

    def test_backtest_grading_never_touches_live_picks(self):
        self._grade('backtest')
        self.assertEqual(self._graded_pick_ids(), [])

    def test_live_grading_attaches_to_live_pick_only(self):
        self._grade('live')
        self.assertEqual(self._graded_pick_ids(), [1])


if __name__ == '__main__':
    unittest.main()
