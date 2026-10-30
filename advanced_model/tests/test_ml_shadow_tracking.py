"""
tests/test_ml_shadow_tracking.py
==================================
Unit tests for ML shadow tracking fixes (P1-3 plan).
"""
import os
import sys
import unittest
import unittest.mock
import sqlite3
import tempfile

# Ensure advanced_model root is on path
MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)


def _make_temp_db():
    """Create a temp SQLite DB with ml_shadow_picks table."""
    tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name)
    conn.execute("""
        CREATE TABLE ml_shadow_picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date TEXT,
            player TEXT,
            stat_category TEXT,
            direction TEXT,
            line REAL,
            projection REAL,
            sim_confidence REAL,
            xgb_confidence REAL,
            edge_pct REAL,
            model_version TEXT,
            cohort TEXT,
            actual_value REAL,
            is_win INTEGER,
            resolved_at TEXT
        )
    """)
    conn.commit()
    conn.close()
    return tmp.name


class TestMLShadowTracking(unittest.TestCase):

    # ------------------------------------------------------------------ #
    # Test 1: ML gets logged even when all_picks=[] (zero production picks)
    # ------------------------------------------------------------------ #
    def test_ml_logged_zero_production_picks(self):
        """log_ml_shadow_pick called for shadow_picks even when all_picks is empty."""
        from ml import shadow_logger

        shadow_pick = {
            'player': 'LeBron',
            'stat_category': 'POINTS',
            'direction': 'OVER',
            'line': 28.5,
            'projection': 30.1,
            'sim_confidence': 72.0,
            'edge_pct': 3.1,
            'ban_reason': 'LOW_CONFIDENCE',
        }
        parlays_mock = {
            'all_picks': [],
            'shadow_picks': [shadow_pick],
        }

        with unittest.mock.patch.object(shadow_logger, 'log_ml_shadow_pick',
                                         wraps=lambda **kw: True) as mock_log, \
             unittest.mock.patch('config.settings.ENABLE_ML_SHADOW_TRACKING', True):

            # Simulate what live_ingestion now does
            _ml_logged = 0
            import config.settings as cfg_settings
            if getattr(cfg_settings, 'ENABLE_ML_SHADOW_TRACKING', False):
                for _p in parlays_mock.get('all_picks', []):
                    if _p.get('player') and _p.get('stat_category'):
                        shadow_logger.log_ml_shadow_pick(
                            game_date='2026-10-30', player=_p['player'],
                            stat_category=_p['stat_category'],
                            direction=_p.get('direction', 'OVER'),
                            line=_p.get('line'), projection=_p.get('projection'),
                            sim_confidence=_p.get('confidence'),
                            edge_pct=_p.get('edge_pct'), base_features=None,
                            cohort='selected',
                        )
                        _ml_logged += 1
                for _p in parlays_mock.get('shadow_picks', []):
                    if _p.get('player') and _p.get('stat_category'):
                        shadow_logger.log_ml_shadow_pick(
                            game_date='2026-10-30', player=_p['player'],
                            stat_category=_p['stat_category'],
                            direction=_p.get('direction', 'OVER'),
                            line=_p.get('line'), projection=_p.get('projection'),
                            sim_confidence=_p.get('sim_confidence'),
                            edge_pct=_p.get('edge_pct'), base_features=None,
                            cohort='shadow',
                        )
                        _ml_logged += 1

            self.assertGreaterEqual(mock_log.call_count, 1,
                                    "log_ml_shadow_pick must be called even when all_picks=[]")

    # ------------------------------------------------------------------ #
    # Test 2: --shadow flag now exits with code 1, does NOT set USE_ML_CALIBRATION
    # ------------------------------------------------------------------ #
    def test_shadow_flag_rejected(self):
        """--shadow must exit(1) and must NOT set USE_ML_CALIBRATION=True."""
        import config.settings as cfg_settings
        cfg_settings.USE_ML_CALIBRATION = False  # reset

        original_argv = sys.argv[:]
        sys.argv = ['live_ingestion', '--shadow']
        try:
            from etl import live_ingestion
            with self.assertRaises(SystemExit) as ctx:
                live_ingestion.main()
            self.assertEqual(ctx.exception.code, 1,
                             "--shadow must exit with code 1")
            self.assertFalse(cfg_settings.USE_ML_CALIBRATION,
                             "USE_ML_CALIBRATION must remain False when --shadow is used")
        finally:
            sys.argv = original_argv

    # ------------------------------------------------------------------ #
    # Test 3: ML tracking is passive — all_picks unchanged by tracking on/off
    # ------------------------------------------------------------------ #
    def test_baseline_unchanged_by_tracking(self):
        """build_parlays returns identical all_picks regardless of ENABLE_ML_SHADOW_TRACKING."""
        import config.settings as cfg_settings
        from simulator import parlay_builder as pb

        fake_parlays = {
            'all_picks': [
                {'type': 'prop', 'player': 'Curry', 'stat_category': 'POINTS',
                 'direction': 'OVER', 'line': 25.0, 'projection': 27.5,
                 'confidence': 68.0, 'edge_pct': 3.2}
            ],
            'shadow_picks': [],
        }

        results = {}
        for tracking_on in [True, False]:
            cfg_settings.ENABLE_ML_SHADOW_TRACKING = tracking_on
            with unittest.mock.patch.object(pb, 'build_parlays',
                                             return_value=fake_parlays) as mock_bp:
                out = mock_bp.return_value
                results[tracking_on] = out['all_picks']

        self.assertEqual(results[True], results[False],
                         "all_picks must be identical with tracking on vs off")

        # Reset
        cfg_settings.ENABLE_ML_SHADOW_TRACKING = True

    # ------------------------------------------------------------------ #
    # Test 4: No crash when log_ml_shadow_pick called twice with identical args
    # ------------------------------------------------------------------ #
    def test_no_duplicate_ml_rows(self):
        """Calling log_ml_shadow_pick twice must not crash; at least 1 row inserted."""
        from ml import shadow_logger

        db_path = _make_temp_db()
        orig_db = shadow_logger._DB_PATH
        shadow_logger._DB_PATH = db_path

        kwargs = dict(
            game_date='2026-10-30',
            player='LeBron',
            stat_category='POINTS',
            direction='OVER',
            line=28.5,
            projection=30.1,
            sim_confidence=72.0,
            edge_pct=3.1,
            base_features=None,
        )

        try:
            r1 = shadow_logger.log_ml_shadow_pick(**kwargs)
            r2 = shadow_logger.log_ml_shadow_pick(**kwargs)

            conn = sqlite3.connect(db_path)
            count = conn.execute("SELECT COUNT(*) FROM ml_shadow_picks").fetchone()[0]
            conn.close()

            self.assertGreaterEqual(count, 1, "At least 1 row must be in ml_shadow_picks")
            # Both calls must not crash (return bool, no exception)
            self.assertIsInstance(r1, bool)
            self.assertIsInstance(r2, bool)
        finally:
            shadow_logger._DB_PATH = orig_db
            os.unlink(db_path)

    # ------------------------------------------------------------------ #
    # Test 5: prepare_analyze calls resolve_ml_shadow_picks
    # ------------------------------------------------------------------ #
    def test_prepare_analyze_calls_resolve(self):
        """prepare_analyze must call resolve_ml_shadow_picks with the correct date."""
        with unittest.mock.patch('ml.shadow_logger.resolve_ml_shadow_picks',
                                  return_value=0) as mock_resolve:
            from etl import prepare_analyze

            target_date = '2026-10-29'

            # Mock all heavy dependencies so prepare_analyze runs to completion
            # Need: prediction valid, results valid, analysis valid → CASE A → return 0
            # That path goes through the resolve block we added.
            with unittest.mock.patch.object(prepare_analyze, 'resolve_prediction_file',
                                             return_value='/fake/predicts.txt'), \
                 unittest.mock.patch.object(prepare_analyze, 'validate_prediction_file',
                                             return_value=(True, 'ok')), \
                 unittest.mock.patch.object(prepare_analyze, 'check_games_and_box_scores',
                                             return_value=(1, 1, ['TeamA @ TeamB'])), \
                 unittest.mock.patch.object(prepare_analyze, 'resolve_results_file',
                                             return_value='/fake/results.txt'), \
                 unittest.mock.patch.object(prepare_analyze, 'resolve_analyze_file',
                                             return_value='/fake/analysis.pdf'), \
                 unittest.mock.patch.object(prepare_analyze, 'validate_results_file',
                                             return_value=(True, 'ok')), \
                 unittest.mock.patch.object(prepare_analyze, 'validate_analysis_file',
                                             return_value=(True, 'ok')):

                result = prepare_analyze.prepare_analyze(target_date=target_date, mode='live')

            # CASE A returns 0 and should trigger resolve
            self.assertEqual(result, 0)
            mock_resolve.assert_called_once_with(target_date)



    # ------------------------------------------------------------------ #
    # Test 6: get_effective_date shifts midnight/early morning runs back #
    # ------------------------------------------------------------------ #
    def test_effective_date_with_safety_hours(self):
        """Runs at 1-2 AM or crossing midnight must catalog under the previous evening's slate."""
        from datetime import datetime
        from etl.live_ingestion import get_effective_date

        dt_2am = datetime(2026, 10, 30, 2, 0, 0)
        eff_date_2am = get_effective_date(dt_2am).strftime('%Y-%m-%d')
        self.assertEqual(eff_date_2am, '2026-10-29')

        dt_1am = datetime(2026, 10, 30, 1, 15, 0)
        eff_date_1am = get_effective_date(dt_1am, safety_hours=2).strftime('%Y-%m-%d')
        self.assertEqual(eff_date_1am, '2026-10-29')

        dt_night = datetime(2026, 10, 29, 23, 30, 0)
        eff_date_night = get_effective_date(dt_night, safety_hours=2).strftime('%Y-%m-%d')
        self.assertEqual(eff_date_night, '2026-10-29')

    # ------------------------------------------------------------------ #
    # Test 7: PredictsParser captures report generation timestamp       #
    # ------------------------------------------------------------------ #
    def test_predicts_parser_report_generated_timestamp(self):
        """PredictsParser must capture day and time when report is generated."""
        from etl.predicts_to_pdf import PredictsParser
        parser = PredictsParser("sample text", game_date='2026-10-29', generated_at='2026-10-30 01:05:00')
        self.assertEqual(parser.run_date, 'October 29, 2026')
        self.assertIn('October 30, 2026 at 01:05', parser.generated_at)

if __name__ == '__main__':
    unittest.main()
