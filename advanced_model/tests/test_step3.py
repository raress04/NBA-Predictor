"""Step 3 — probability calibration (Plans/STEP3_CALIBRATION_AUDIT.md)."""
import math
import os
import subprocess
import sys
import unittest
import unittest.mock

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODEL_DIR)

import config.settings as settings  # noqa: E402
from utils import prob_calibration as pc  # noqa: E402

ENGINE = {'SHRINKAGE_MODEL': 'fitted', 'ENGINE_CALIBRATION': 'v2', 'B2B_MODEL': 'fitted', 'RETURNING_MODEL': 'fitted'}
PARAMS = {'schema': 2, 'model': 'M4', 'engine': ENGINE, 'global': {'a': 0.1, 'b': 0.2},
          'groups': {'POINTS_OVER': {'a_offset': -0.1, 'b_offset': 0.05}}}


class S:  # settings stand-in
    SHRINKAGE_MODEL, ENGINE_CALIBRATION, B2B_MODEL, RETURNING_MODEL = 'fitted', 'v2', 'fitted', 'fitted'


class TestCalibrateMath(unittest.TestCase):
    def test_formula(self):
        x = math.log(0.7 / 0.3)
        z = 0.1 - 0.1 + (0.2 + 0.05) * x
        self.assertAlmostEqual(pc.calibrate('points', 'Over', 70.0, PARAMS, S), 100 / (1 + math.exp(-z)), places=9)

    def test_none_cases(self):
        self.assertIsNone(pc.calibrate('POINTS', 'OVER', None, PARAMS, S))
        self.assertIsNone(pc.calibrate('PTS_REB', 'OVER', 70, PARAMS, S))          # group not fitted

        class Legacy(S):
            SHRINKAGE_MODEL = 'legacy'
        self.assertIsNone(pc.calibrate('POINTS', 'OVER', 70, PARAMS, Legacy))      # other engine
        with unittest.mock.patch.object(pc, 'load_params', return_value=None):
            self.assertIsNone(pc.calibrate('POINTS', 'OVER', 70))                  # no params file

    def test_clipping(self):
        self.assertTrue(0 < pc.calibrate('POINTS', 'OVER', 100.0, PARAMS, S) < 100)
        self.assertTrue(0 < pc.calibrate('POINTS', 'OVER', 0.0, PARAMS, S) < 100)


class TestFittedParams(unittest.TestCase):
    def test_file_sane(self):
        p = pc.load_params(settings.PROB_CALIBRATION_PATH)
        self.assertIsNotNone(p, "config/prob_calibration.json missing")
        self.assertEqual(set(p['groups']), {f"{c}_{d}" for c in ('POINTS', 'REBOUNDS', 'ASSISTS')
                                            for d in ('OVER', 'UNDER')})
        self.assertEqual(p['engine'], ENGINE)
        for g in p['groups'].values():
            self.assertGreaterEqual(g['n'], 200)          # A8 minimum sample

    def test_shrinks_overconfidence(self):
        """A raw 75% must come out well below 75% (the overconfidence found in P2)."""
        p = pc.load_params(settings.PROB_CALIBRATION_PATH)
        for g in p['groups']:
            cat, d = g.split('_')
            v = pc.calibrate(cat, d, 75.0, p, S)
            self.assertLess(v, 70.0, g)
            self.assertGreater(v, 40.0, g)


class TestSetting(unittest.TestCase):
    def _get(self, env):
        e = {k: v for k, v in os.environ.items() if k not in ('PROB_CALIBRATION', 'STEP2_PROFILE')}
        e.update(env)
        return subprocess.run([sys.executable, '-c', 'import config.settings as s; print(s.PROB_CALIBRATION)'],
                              cwd=MODEL_DIR, env=e, capture_output=True, text=True)

    def test_default_shadow(self):
        self.assertEqual(self._get({}).stdout.strip(), 'shadow')

    def test_env_and_legacy(self):
        self.assertEqual(self._get({'PROB_CALIBRATION': 'on'}).stdout.strip(), 'on')
        self.assertEqual(self._get({'STEP2_PROFILE': 'legacy'}).stdout.strip(), 'off')

    def test_invalid_raises(self):
        self.assertNotEqual(self._get({'PROB_CALIBRATION': 'maybe'}).returncode, 0)


class TestBuildParlays(unittest.TestCase):
    """REBOUNDS OVER, raw 72% (same fixture as test_step2.TestConfidenceGate)."""

    def _run(self, mode, gap_ok=True):
        from simulator import parlay_builder as pb
        reb = [5] * 1400 + [10] * 3600 if gap_ok else [7] * 1400 + [8] * 3600
        props = {'E1': {'Reb Guy': {'rebounds': {'line': 7.5, 'over_odds': 1.9, 'under_odds': 1.9, 'book': 'dk'}}}}
        dists = {'Reb Guy': {'REB': reb}}
        minutes = {'Reb Guy': {'mean': 30.0, 'std': 3.0, 'team': 'Home'}}
        odds = {'E1': {'home_team': 'Home', 'away_team': 'Away'}}
        with unittest.mock.patch.object(settings, 'PROB_CALIBRATION', mode), \
             unittest.mock.patch.object(settings, 'CONFIDENCE_GATE', 'raw'), \
             unittest.mock.patch.object(settings, 'USE_RUNTIME_BIAS', False), \
             unittest.mock.patch.object(pb, 'persist_parlay', return_value=None), \
             unittest.mock.patch.object(pb, 'get_category_calibration', return_value=0.85):
            return pb.build_parlays([], odds, props, dists, {}, all_historical_rates={},
                                    all_player_minutes=minutes, game_date='2026-11-10')

    def test_off_has_no_cal_prob(self):
        out = self._run('off')
        for p in out['all_picks']:
            self.assertIsNone(p.get('cal_prob'))

    def test_shadow_logs_but_decides_like_off(self):
        off, sh = self._run('off'), self._run('shadow')
        self.assertEqual([(p['player'], p['direction'], round(p['edge_pct'], 9)) for p in off['all_picks']],
                         [(p['player'], p['direction'], round(p['edge_pct'], 9)) for p in sh['all_picks']])
        self.assertEqual([r['ban_reason'] for r in off['shadow_picks']], [r['ban_reason'] for r in sh['shadow_picks']])
        rows = sh['all_picks'] + sh['shadow_picks']
        self.assertTrue(rows and all(r.get('cal_prob') is not None for r in rows))
        self.assertTrue(all(r['cal_prob'] < 72.0 for r in rows))

    def test_shadow_rejected_rows_carry_cal_prob(self):
        sh = self._run('shadow', gap_ok=False)            # median 8 vs line 7.5 -> MIN_GAP reject
        self.assertEqual(sh['all_picks'], [])
        self.assertTrue(sh['shadow_picks'] and sh['shadow_picks'][0].get('cal_prob') is not None)

    def test_on_uses_cal_prob_and_drops_category_penalty(self):
        on = self._run('on')
        p = on['all_picks'][0] if on['all_picks'] else None
        if p is None:   # rejected on the calibrated edge: the row must say so
            self.assertTrue(any(r['ban_reason'].startswith('LOW_EDGE') for r in on['shadow_picks']))
            return
        from simulator.parlay_builder import true_edge_pct
        self.assertAlmostEqual(p['edge_pct'], true_edge_pct(p['cal_prob'], 1.9), places=9)
        self.assertAlmostEqual(p['confidence'], 72.0, places=6)     # penalty x0.85 not applied under 'on'


if __name__ == '__main__':
    unittest.main()
