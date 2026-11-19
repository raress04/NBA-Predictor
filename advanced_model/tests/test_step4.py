"""Step 4 — pick filters on cal_prob (Plans/STEP4_GATES_AUDIT.md)."""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODEL_DIR)

import config.settings as settings  # noqa: E402
from simulator import parlay_builder as pb  # noqa: E402
from utils.prob_calibration import engine_signature, load_params  # noqa: E402

# REBOUNDS, line 7.5: 72% of sims under (median 5, gap 2.5). M4 puts REB UNDER at ~57%,
# so at odds 1.9 (break-even 52.6%) the edge over break-even is ~4.5 points.
REB_UNDER = [5] * 3600 + [10] * 1400


def _run(dist=REB_UNDER, line=7.5, odds=1.9, min_std=3.0, gate='raw', params=None, shadow=True,
         extra_players=None):
    props = {'E1': {'Reb Guy': {'rebounds': {'line': line, 'over_odds': odds, 'under_odds': odds, 'book': 'dk'}}}}
    dists = {'Reb Guy': {'REB': dist}}
    minutes = {'Reb Guy': {'mean': 30.0, 'std': min_std, 'team': 'Home'}}
    for name, (dd, ln, od) in (extra_players or {}).items():
        props['E1'][name] = {'rebounds': {'line': ln, 'over_odds': od, 'under_odds': od, 'book': 'dk'}}
        dists[name] = {'REB': dd}
        minutes[name] = {'mean': 30.0, 'std': 3.0, 'team': 'Home'}
    odds_map = {'E1': {'home_team': 'Home', 'away_team': 'Away'}}
    tmp = tempfile.TemporaryDirectory()
    path = os.path.join(tmp.name, 'gate_params.json')
    if params is not None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(params, f)
    pb._gate_cache.update(path=None, mtime=None, params=None, warned=False)
    with unittest.mock.patch.object(settings, 'CONFIDENCE_GATE', 'raw'), \
         unittest.mock.patch.object(settings, 'USE_RUNTIME_BIAS', False), \
         unittest.mock.patch.object(settings, 'GATE_MODEL', gate), \
         unittest.mock.patch.object(settings, 'PROB_CALIBRATION', 'on' if gate == 'step4' else 'shadow'), \
         unittest.mock.patch.object(settings, 'STEP4_SHADOW', shadow), \
         unittest.mock.patch.object(settings, 'GATE_PARAMS_PATH', path), \
         unittest.mock.patch.object(pb, 'persist_parlay', return_value=None), \
         unittest.mock.patch.object(pb, 'get_category_calibration', return_value=1.0), \
         unittest.mock.patch('builtins.print'):
        out = pb.build_parlays([], odds_map, props, dists, {}, all_historical_rates={},
                               all_player_minutes=minutes, game_date='2026-11-06')
    tmp.cleanup()
    pb._gate_cache.update(path=None, mtime=None, params=None, warned=False)
    return out


def _params(margin=0.02, filters=(), cap=None, **over):
    p = dict(schema=1, gate_model='step4', margin=margin, filters=list(filters), cap=cap, rank='ev',
             engine=engine_signature(), calibration=(load_params() or {}).get('data'))
    p.update(over)
    return p


class TestCandidateFilters(unittest.TestCase):
    """P1: every filter's outcome is recorded, not only the first failure."""

    def test_flags_on_pick(self):
        out = _run()
        f = out['all_picks'][0]['filters']
        self.assertAlmostEqual(f['gap'], 2.5)
        for k in ('f_min_gap', 'f_minutes', 'f_line_vol', 'f_stage1', 'f_banned', 'f_variance',
                  'f_blowout', 'f_restriction'):
            self.assertTrue(f[k], k)

    def test_flags_on_rejected_row_show_later_filters_too(self):
        # small gap AND minutes risk: MIN_GAP rejects first, but f_minutes is still recorded
        dist = [7] * 3600 + [10] * 1400                      # median 7, gap 0.5 < 1.8
        out = _run(dist=dist, min_std=8.0)
        row = out['shadow_picks'][0]
        self.assertTrue(row['ban_reason'].startswith('MIN_GAP'))
        self.assertFalse(row['filters']['f_min_gap'])
        self.assertFalse(row['filters']['f_minutes'])
        self.assertTrue(row['filters']['f_line_vol'])

    def test_low_line_flag(self):
        out = _run(dist=[2] * 3600 + [8] * 1400, line=4.5)
        self.assertFalse(out['shadow_picks'][0]['filters']['f_line_vol'])


class TestStep4Verdict(unittest.TestCase):
    def setUp(self):
        self.edge = pb.compute_prop_edge(REB_UNDER, 7.5, 'rebounds')
        self.flags = {'f_banned': True, 'f_minutes': True, 'f_min_gap': False}
        self.prop = {'over_odds': 1.9, 'under_odds': 1.9}

    def test_pass_and_math(self):
        v = pb.step4_verdict('rebounds', self.edge, self.prop, self.flags, _params(margin=0.02))
        self.assertTrue(v['passed'], v)
        self.assertAlmostEqual(v['raw'], 72.0)
        self.assertAlmostEqual(v['edge_be'], v['cal'] / 100 - 1 / 1.9)
        self.assertAlmostEqual(v['ev'], v['cal'] / 100 * 1.9 - 1)

    def test_margin_rejects(self):
        v = pb.step4_verdict('rebounds', self.edge, self.prop, self.flags, _params(margin=0.08))
        self.assertFalse(v['passed'])
        self.assertTrue(v['reason'].startswith('STEP4_LOW_EDGE'))

    def test_kept_filter_rejects(self):
        v = pb.step4_verdict('rebounds', self.edge, self.prop, self.flags,
                             _params(margin=0.0, filters=['f_minutes', 'f_min_gap']))
        self.assertEqual(v['reason'], 'STEP4_F_MIN_GAP')

    def test_banned(self):
        v = pb.step4_verdict('rebounds', self.edge, self.prop, dict(self.flags, f_banned=False), _params(0.0))
        self.assertEqual(v['reason'], 'BANNED_CATEGORY')

    def test_missing_odds_default(self):
        v = pb.step4_verdict('rebounds', self.edge, {}, self.flags, _params(0.0))
        self.assertAlmostEqual(v['odds'], 1.909)


class TestGateModelSwitch(unittest.TestCase):
    def test_raw_default_and_unchanged_by_shadow(self):
        self.assertEqual(settings.GATE_MODEL, 'raw')
        a = _run(params=_params(margin=0.0), shadow=True)
        b = _run(params=None, shadow=False)
        strip = lambda ps: [{k: v for k, v in p.items() if k != 'filters'} for p in ps]   # noqa: E731
        self.assertEqual(strip(a['all_picks']), strip(b['all_picks']))
        self.assertEqual(len(a['step4_shadow']), 1)
        self.assertEqual(b['step4_shadow'], [])

    def test_step4_picks_with_cal(self):
        out = _run(gate='step4', params=_params(margin=0.02))
        self.assertEqual(len(out['all_picks']), 1)
        p = out['all_picks'][0]
        self.assertAlmostEqual(p['confidence'], p['cal_prob'])
        self.assertAlmostEqual(p['edge_pct'], p['cal_prob'] - 100 / 1.9)
        self.assertTrue(out['step4_shadow'][0]['passed'])

    def test_step4_rejects_logged(self):
        out = _run(gate='step4', params=_params(margin=0.10))
        self.assertEqual(out['all_picks'], [])
        self.assertTrue(out['shadow_picks'][0]['ban_reason'].startswith('STEP4_LOW_EDGE'))

    def test_step4_without_params_falls_back_to_raw(self):
        a = _run(gate='step4', params=None)
        b = _run(gate='raw', params=None)
        self.assertEqual(len(a['all_picks']), len(b['all_picks']))

    def test_params_for_other_engine_ignored(self):
        p = _params(margin=0.0, engine=dict(engine_signature(), SHRINKAGE_MODEL='legacy'))
        out = _run(gate='step4', params=p)
        self.assertEqual(out['step4_shadow'], [])     # params rejected -> nothing recorded

    def test_rank_by_ev_and_cap(self):
        better = [5] * 4200 + [10] * 800                # 84% under -> higher cal -> higher EV
        out = _run(gate='step4', params=_params(margin=0.0, cap=1),
                   extra_players={'Better Guy': (better, 7.5, 1.9)})
        self.assertEqual([p['player'] for p in out['all_picks']], ['Better Guy'])
        self.assertTrue(any(r['ban_reason'] == 'DAILY_CAP' for r in out['shadow_picks']))

    def test_settings_validation(self):
        bad = [({'GATE_MODEL': 'bogus'}, 'GATE_MODEL'),
               ({'GATE_MODEL': 'step4', 'PROB_CALIBRATION': 'off'}, 'PROB_CALIBRATION')]
        for env, _ in bad:
            r = subprocess.run([sys.executable, '-c', 'import config.settings'], cwd=MODEL_DIR,
                               env={**os.environ, **env}, capture_output=True)
            self.assertNotEqual(r.returncode, 0, env)


class TestActionNetworkPrices(unittest.TestCase):
    """The AN API puts over/under in 'side'. Reading 'type' stored the same price for both."""

    def _fake(self):
        q = lambda book, side, v, odds: {'event_id': 1, 'player_id': 7, 'side': side,     # noqa: E731
                                          'type': 'core_bet_type_23_rebounds', 'value': v, 'odds': odds}
        return {'players': [{'id': 7, 'full_name': 'Reb Guy'}],
                'markets': {'15': {'event': {'core_bet_type_23_rebounds': [q(15, 'over', 2.5, 131),
                                                                           q(15, 'under', 2.5, -169)]}},
                            '30': {'event': {'core_bet_type_23_rebounds': [q(30, 'over', 2.5, 131),
                                                                           q(30, 'under', 2.5, -169)]}},
                            '68': {'event': {'core_bet_type_23_rebounds': [q(68, 'over', 3.5, -200),
                                                                           q(68, 'under', 3.5, 160)]}}}}

    def test_sides_kept_apart_and_line_matched(self):
        from etl import scrape_action_network as an
        resp = unittest.mock.Mock(status_code=200)
        resp.json.return_value = self._fake()
        res = {}
        with unittest.mock.patch.object(an.requests, 'get', return_value=resp), \
             unittest.mock.patch('builtins.print'):
            an.fetch_props('20261115', 'REBOUNDS', 'core_bet_type_23_rebounds', res)
        p = res[(1, 'Reb Guy')]['REBOUNDS']
        self.assertEqual(p['line'], 2.5)
        self.assertAlmostEqual(p['over_odds'], 2.31)            # +131
        self.assertAlmostEqual(p['under_odds'], round(1 + 100 / 169, 4))   # -169, not +131
        self.assertGreater(1 / p['over_odds'] + 1 / p['under_odds'], 1.0)   # a real bookmaker margin

    def test_consensus_prices_only_on_line(self):
        from etl.scrape_action_network import consensus_prices
        quotes = [(2.5, 'over', -110), (2.5, 'under', -110), (3.5, 'over', 150)]
        self.assertEqual(consensus_prices(quotes, 3.5), (2.5, 1.909))


class TestStep4ShadowCsv(unittest.TestCase):
    def test_write(self):
        from etl import live_ingestion as li
        rows = [dict(player='A', stat_category='REBOUNDS', passed=True, reason=None, ev=0.05),
                dict(player='B', stat_category='POINTS', passed=False, reason='STEP4_LOW_EDGE(1.0<=2)', ev=-0.01)]
        with tempfile.TemporaryDirectory() as d:
            fake = os.path.join(d, 'etl', 'live_ingestion.py')
            with unittest.mock.patch.object(li.os.path, 'abspath', return_value=fake), \
                 unittest.mock.patch('builtins.print'):
                li._write_step4_shadow(rows, '2026-11-19')
            path = os.path.join(d, 'logs', 'november', 'step4_shadow', 'step4_shadow_2026-11-19.csv')
            with open(path, encoding='utf-8') as f:
                got = list(csv.DictReader(f))
        self.assertEqual([r['player'] for r in got], ['A', 'B'])
        self.assertEqual(got[0]['game_date'], '2026-11-19')
        self.assertEqual(got[0]['gate_model'], 'raw')


if __name__ == '__main__':
    unittest.main()
