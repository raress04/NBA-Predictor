"""
tests/test_prepare_analyze.py
===============================
Unit tests for etl/prepare_analyze.py covering all requirements:
1. Both artifacts valid -> skip without fetching or rewriting.
2. Valid results, missing analysis -> generates analysis only.
3. Missing results -> fetches box scores, generates results, generates analysis.
4. Missing results with existing analysis -> analysis is regenerated.
5. Empty / malformed artifacts -> detected and repaired.
6. Fetch or generation failure -> exits nonzero (1).
7. Pending games and missing predictions -> actionable failure (exit 1).
8. Verified no-game day -> successful no-op (exit 0).
9. Repeated execution -> idempotent, no duplicate writes.
10. Date override and month/year boundaries.
11. Atomic write reliability -> temp files and preservation on error.
12. Pipeline execution order in run_pipeline.sh.
"""

import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add advanced_model to path
MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from etl.prepare_analyze import (
    compute_target_date,
    validate_results_file,
    validate_analysis_file,
    validate_prediction_file,
    check_games_and_box_scores,
    generate_results_file,
    prepare_analyze,
)
from utils.prediction_paths import (
    get_nba_season,
    get_prediction_filepath,
    get_results_filepath,
    get_analyze_report_filepath,
)

SAMPLE_PREDICTS_TEXT = """[+] Execution mode: LIVE | Date: 2026-10-24
MATCHUP: Boston Celtics @ Miami Heat
PROJECTED SCORE: MIA 105.0 - BOS 110.0
Anthony Edwards      -> 29 PTS, 7 REB, 6 AST
Jayson Tatum         -> 26 PTS, 8 REB, 4 AST (O/U 24.5 points VALUABLE: Over (75% conf, 15.0% edge))
"""

SAMPLE_RESULTS_TEXT = """MATCHUP: BOS vs. MIA    110 - 105
Anthony Edwards        | 29 PTS, 7 REB,  6 AST
Jayson Tatum           | 26 PTS, 8 REB,  4 AST
Bam Adebayo            | 20 PTS, 10 REB,  5 AST
Jimmy Butler           | 18 PTS, 5 REB,  5 AST
Derrick White          | 15 PTS, 3 REB,  4 AST
"""


class TestPrepareYesterday(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_prep_yesterday_")
        self.target_date = "2026-10-24"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_prediction_file(self, content=SAMPLE_PREDICTS_TEXT, filename="predicts_2026-10-24.txt"):
        pred_path = get_prediction_filepath(self.target_date, mode='live', ext='txt', base_dir=self.test_dir)
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        with open(pred_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return pred_path

    def _create_results_file(self, content=SAMPLE_RESULTS_TEXT):
        res_path = get_results_filepath(self.target_date, mode='live', ext='txt', base_dir=self.test_dir)
        os.makedirs(os.path.dirname(res_path), exist_ok=True)
        with open(res_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return res_path

    def _create_analysis_file(self, size_bytes=15000):
        ana_path = get_analyze_report_filepath(self.target_date, mode='live', ext='pdf', base_dir=self.test_dir)
        os.makedirs(os.path.dirname(ana_path), exist_ok=True)
        with open(ana_path, 'wb') as f:
            # Write valid PDF header and dummy bytes
            f.write(b"%PDF-1.4 " + b"0" * size_bytes)
        return ana_path

    def test_both_artifacts_valid_skips(self):
        """Case A: Both results and analysis valid -> no fetching or rewriting."""
        self._create_prediction_file()
        self._create_results_file()
        self._create_analysis_file()

        with patch("etl.prepare_analyze.fetch_box_scores_if_needed") as mock_fetch, \
             patch("etl.prepare_analyze.generate_results_file") as mock_gen_res, \
             patch("etl.prepare_analyze.generate_analysis_report") as mock_gen_ana:
            
            code = prepare_analyze(target_date=self.target_date, mode='live', base_dir=self.test_dir)
            self.assertEqual(code, 0)
            mock_fetch.assert_not_called()
            mock_gen_res.assert_not_called()
            mock_gen_ana.assert_not_called()

    def test_valid_results_missing_analysis(self):
        """Case B: Valid results exist, missing analysis -> generates analysis only."""
        self._create_prediction_file()
        res_path = self._create_results_file()
        # Analysis PDF is not created

        fake_ana_path = get_analyze_report_filepath(self.target_date, mode='live', ext='pdf', base_dir=self.test_dir)

        with patch("etl.prepare_analyze.fetch_box_scores_if_needed") as mock_fetch, \
             patch("etl.prepare_analyze.generate_results_file") as mock_gen_res, \
             patch("etl.prepare_analyze.generate_analysis_report", return_value=fake_ana_path) as mock_gen_ana:
            
            code = prepare_analyze(target_date=self.target_date, mode='live', base_dir=self.test_dir)
            self.assertEqual(code, 0)
            mock_fetch.assert_not_called()
            mock_gen_res.assert_not_called()
            mock_gen_ana.assert_called_once()

    def test_missing_results_generates_all(self):
        """Case C: Missing results -> fetch box scores -> results -> analysis."""
        self._create_prediction_file()
        # Neither results nor analysis exists

        fake_res_path = get_results_filepath(self.target_date, mode='live', ext='txt', base_dir=self.test_dir)
        fake_ana_path = get_analyze_report_filepath(self.target_date, mode='live', ext='pdf', base_dir=self.test_dir)

        with patch("etl.prepare_analyze.fetch_box_scores_if_needed", return_value=True) as mock_fetch, \
             patch("etl.prepare_analyze.generate_results_file", return_value=fake_res_path) as mock_gen_res, \
             patch("etl.prepare_analyze.generate_analysis_report", return_value=fake_ana_path) as mock_gen_ana:
            
            code = prepare_analyze(target_date=self.target_date, mode='live', base_dir=self.test_dir)
            self.assertEqual(code, 0)
            mock_fetch.assert_called_once()
            mock_gen_res.assert_called_once()
            mock_gen_ana.assert_called_once()

    def test_missing_results_with_existing_analysis_regenerates_analysis(self):
        """Missing results with existing analysis -> analysis must be regenerated."""
        self._create_prediction_file()
        self._create_analysis_file()  # Pre-existing analysis
        # Results is missing

        fake_res_path = get_results_filepath(self.target_date, mode='live', ext='txt', base_dir=self.test_dir)
        fake_ana_path = get_analyze_report_filepath(self.target_date, mode='live', ext='pdf', base_dir=self.test_dir)

        with patch("etl.prepare_analyze.fetch_box_scores_if_needed", return_value=True), \
             patch("etl.prepare_analyze.generate_results_file", return_value=fake_res_path), \
             patch("etl.prepare_analyze.generate_analysis_report", return_value=fake_ana_path) as mock_gen_ana:
            
            code = prepare_analyze(target_date=self.target_date, mode='live', base_dir=self.test_dir)
            self.assertEqual(code, 0)
            mock_gen_ana.assert_called_once()

    def test_empty_or_malformed_artifacts_detected_and_repaired(self):
        """Empty (0-byte) or corrupted artifacts are rejected by validation and repaired."""
        self._create_prediction_file()
        
        # Corrupted results (0 bytes)
        empty_res = self._create_results_file(content="")
        valid_res, reason = validate_results_file(empty_res, self.target_date)
        self.assertFalse(valid_res)
        self.assertIn("empty", reason.lower())

        # Corrupted analysis (not a PDF)
        bad_ana = self._create_analysis_file(size_bytes=10)
        valid_ana, ana_reason = validate_analysis_file(bad_ana, self.target_date)
        self.assertFalse(valid_ana)

        # Ensure prepare_analyze detects them as invalid and triggers regeneration
        fake_res = get_results_filepath(self.target_date, mode='live', ext='txt', base_dir=self.test_dir)
        fake_ana = get_analyze_report_filepath(self.target_date, mode='live', ext='pdf', base_dir=self.test_dir)

        with patch("etl.prepare_analyze.fetch_box_scores_if_needed", return_value=True), \
             patch("etl.prepare_analyze.generate_results_file", return_value=fake_res) as mock_gen_res, \
             patch("etl.prepare_analyze.generate_analysis_report", return_value=fake_ana) as mock_gen_ana:
            
            code = prepare_analyze(target_date=self.target_date, mode='live', base_dir=self.test_dir)
            self.assertEqual(code, 0)
            mock_gen_res.assert_called_once()
            mock_gen_ana.assert_called_once()

    def test_fetch_failure_exits_nonzero(self):
        """Fetch failure exits nonzero (1) and does not proceed to file generation."""
        self._create_prediction_file()

        with patch("etl.prepare_analyze.fetch_box_scores_if_needed", return_value=False), \
             patch("etl.prepare_analyze.check_games_and_box_scores", return_value=(2, 0, ['CLE @ PHI', 'DEN @ OKC'])), \
             patch("etl.prepare_analyze.generate_results_file") as mock_gen_res:
            
            code = prepare_analyze(target_date=self.target_date, mode='live', base_dir=self.test_dir)
            self.assertEqual(code, 1)
            mock_gen_res.assert_not_called()

    def test_pending_games_and_missing_predictions(self):
        """Predictions missing but games occurred:
        live     -> warning + 0 (a missed live run must not stop the pipeline; H1 option B, 2026-11-06)
        backtest -> actionable failure (1)."""
        # Predictions NOT created
        with patch("etl.prepare_analyze.check_games_and_box_scores", return_value=(2, 45, ['BOS @ MIA', 'LAL @ GSW'])), \
             patch("etl.prepare_analyze.generate_results_file") as mock_gen_res, \
             patch("etl.prepare_analyze.generate_analysis_report") as mock_gen_ana:
            code = prepare_analyze(target_date=self.target_date, mode='live', base_dir=self.test_dir)
            self.assertEqual(code, 0)
            code = prepare_analyze(target_date=self.target_date, mode='backtest', base_dir=self.test_dir)
            self.assertEqual(code, 1)
            mock_gen_res.assert_not_called()
            mock_gen_ana.assert_not_called()

    def test_verified_no_game_day_success_noop(self):
        """0 games in schedule and 0 predictions -> successful no-op (code 0)."""
        # Predictions NOT created
        with patch("etl.prepare_analyze.check_games_and_box_scores", return_value=(0, 0, [])):
            code = prepare_analyze(target_date="2026-08-15", mode='live', base_dir=self.test_dir)
            self.assertEqual(code, 0)

    def test_repeated_execution_idempotency(self):
        """Repeated execution does not trigger duplicate writes or calls."""
        self._create_prediction_file()
        self._create_results_file()
        self._create_analysis_file()

        with patch("etl.prepare_analyze.generate_results_file") as mock_gen_res, \
             patch("etl.prepare_analyze.generate_analysis_report") as mock_gen_ana:
            
            # Run 1
            code1 = prepare_analyze(target_date=self.target_date, mode='live', base_dir=self.test_dir)
            self.assertEqual(code1, 0)
            # Run 2
            code2 = prepare_analyze(target_date=self.target_date, mode='live', base_dir=self.test_dir)
            self.assertEqual(code2, 0)

            mock_gen_res.assert_not_called()
            mock_gen_ana.assert_not_called()

    def test_date_override_and_boundaries(self):
        """Validates date override across month and year boundaries."""
        # 1. Target date override parsing
        d1 = compute_target_date("2026-10-31")
        self.assertEqual(d1, "2026-10-31")
        d2 = compute_target_date("2026-11-01")
        self.assertEqual(d2, "2026-11-01")

        # 2. Season boundary (month 10 vs month 4)
        self.assertEqual(get_nba_season("2026-10-24"), "2026-2027")
        self.assertEqual(get_nba_season("2027-01-15"), "2026-2027")
        self.assertEqual(get_nba_season("2027-04-10"), "2026-2027")

        # 3. Path resolution respects month folder
        p_oct = get_results_filepath("2026-10-31", mode='live', base_dir=self.test_dir)
        p_nov = get_results_filepath("2026-11-01", mode='live', base_dir=self.test_dir)
        self.assertIn("october", p_oct.replace('\\', '/'))
        self.assertIn("november", p_nov.replace('\\', '/'))

    def test_pipeline_ordering_in_shell_script(self):
        """Verifies scripts/run_pipeline.sh runs prepare_analyze BEFORE live_ingestion."""
        pipeline_sh = os.path.join(MODEL_DIR, 'scripts', 'run_pipeline.sh')
        self.assertTrue(os.path.exists(pipeline_sh))

        with open(pipeline_sh, 'r', encoding='utf-8') as f:
            content = f.read()

        prep_idx = content.find("python3 -m etl.prepare_analyze")
        model_idx = content.find("python3 -m etl.live_ingestion")

        self.assertGreater(prep_idx, -1, "prepare_analyze must be present in run_pipeline.sh")
        self.assertGreater(model_idx, -1, "live_ingestion must be present in run_pipeline.sh")
        self.assertLess(prep_idx, model_idx, "prepare_analyze must execute BEFORE live_ingestion")

    def test_atomic_write_cleans_tmp_on_error(self):
        """Ensures that any failure during results generation cleans up .tmp files."""
        # Non-existent DB causes generate_results_file to return None without leaving tmp
        res = generate_results_file(self.target_date, mode='live', base_dir=self.test_dir, db_path="nonexistent.db")
        self.assertIsNone(res)
        # Check no .tmp files exist in test directory
        for root, dirs, files in os.walk(self.test_dir):
            for f in files:
                self.assertFalse(f.endswith(".tmp") or ".tmp." in f)


if __name__ == '__main__':
    unittest.main()
