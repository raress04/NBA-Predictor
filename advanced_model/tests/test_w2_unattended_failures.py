"""
tests/test_w2_unattended_failures.py
====================================
W2 (Plans/W2_UNATTENDED_FAILURES_AUDIT.md): each unattended failure, simulated here,
leaves a clear "[ALERT ...]" line in the scheduled-log folder, and the night status
says OK / DEGRADED / FAILED.

No real database, cache folder, API or Windows pop-up is touched: alerts go to tmp_path
(PIPELINE_ALERTS_DIR), pop-ups are off (PIPELINE_NOTIFY=0), HTTP is mocked.
"""
import os
import stat
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from utils import run_alerts

SLATE = '2026-11-25'


@pytest.fixture(autouse=True)
def alerts_dir(tmp_path, monkeypatch):
    d = tmp_path / 'scheduled'
    monkeypatch.setenv('PIPELINE_ALERTS_DIR', str(d))
    monkeypatch.setenv('PIPELINE_NOTIFY', '0')
    monkeypatch.setattr(run_alerts, '_slate', lambda: SLATE)
    return d


def _alerts(alerts_dir):
    p = alerts_dir / f'alerts_{SLATE}.log'
    return p.read_text(encoding='utf-8') if p.exists() else ''


# ── utils/run_alerts ─────────────────────────────────────────────

def test_alert_line_format_and_file(alerts_dir, capsys):
    line = run_alerts.alert('error', 'espn', 'down -> no predictions')
    assert line == '[ALERT ERROR] espn: down -> no predictions'
    assert line in capsys.readouterr().out
    assert line in _alerts(alerts_dir)


def test_night_status_counts_only_latest_run(alerts_dir):
    run_alerts.start_run()
    run_alerts.alert('ERROR', 'a', 'old run')
    run_alerts.start_run()
    assert run_alerts.night_status() == ('OK', 0, 0)
    run_alerts.alert('WARN', 'b', 'x')
    run_alerts.alert('ERROR', 'c', 'y')
    assert run_alerts.night_status() == ('DEGRADED', 1, 1)
    assert run_alerts.night_status(failed=True)[0] == 'FAILED'
    assert run_alerts.status_line('DEGRADED', 1, 1) == 'NIGHT STATUS: DEGRADED (1 error, 1 warning)'


def test_summary_pops_up_only_when_not_ok(alerts_dir, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(run_alerts, 'notify', lambda t, b: calls.append((t, b)) or True)
    run_alerts.start_run()
    assert run_alerts.summarize() == 'OK'
    assert calls == []
    run_alerts.alert('WARN', 'odds_api', '100 credits left')
    assert run_alerts.summarize() == 'DEGRADED'
    assert 'NIGHT STATUS: DEGRADED (0 errors, 1 warning)' in capsys.readouterr().out
    assert calls and 'DEGRADED' in calls[-1][0]
    assert run_alerts.summarize(failed=True) == 'FAILED'
    assert 'No predictions tonight' in calls[-1][1]


def test_notify_is_off_in_tests_and_never_raises(monkeypatch):
    assert run_alerts.notify('t', 'b') is False          # PIPELINE_NOTIFY=0
    monkeypatch.setenv('PIPELINE_NOTIFY', '1')
    monkeypatch.setattr(run_alerts, 'toast_command', lambda t, b: ['/nonexistent/powershell.exe'])
    assert run_alerts.notify('t', 'b') is False


def test_toast_command_escapes_quotes():
    cmd = run_alerts.toast_command("it's", "a 'b'")
    assert "it''s" in cmd[-1] and "a ''b''" in cmd[-1]


# ── (c) Odds API credits and props ───────────────────────────────

def _resp(headers=None, json_data=None, status=200):
    r = MagicMock()
    r.headers = headers or {}
    r.json.return_value = json_data if json_data is not None else []
    r.status_code = status
    if status >= 400:
        err = requests.HTTPError(f"{status} Client Error")
        err.response = r
        r.raise_for_status.side_effect = err
    else:
        r.raise_for_status.return_value = None
    return r


def test_get_credits_remaining_reads_free_sports_endpoint():
    from etl import odds_fetcher
    with patch.object(odds_fetcher.requests, 'get',
                      return_value=_resp({'x-requests-remaining': '446', 'x-requests-used': '54'})) as g:
        assert odds_fetcher.get_credits_remaining(api_key='k') == (446, 54)
    assert g.call_args[0][0].endswith('/sports')


def test_props_fetched_from_eu_region_only():
    from etl import odds_fetcher
    with patch.object(odds_fetcher, '_load_cache', return_value=None), \
         patch.object(odds_fetcher, '_save_cache'), \
         patch.object(odds_fetcher.requests, 'get', return_value=_resp({}, {'bookmakers': []})) as g:
        odds_fetcher.fetch_player_props('evt123', api_key='k')
    assert g.call_args[1]['params']['regions'] == 'eu'


@pytest.fixture
def live():
    from etl import live_ingestion
    return live_ingestion


def test_credits_line_and_warning_below_threshold(live, alerts_dir, capsys):
    with patch.object(live, 'get_credits_remaining', return_value=(98, 402)):
        assert live.report_odds_credits(start_remaining=122, n_games=8, warn_below=120) == 98
    out = capsys.readouterr().out
    assert '[CREDITS] Odds API credits left: 98 (used tonight: 24, 8 games)' in out
    a = _alerts(alerts_dir)
    assert '[ALERT WARN] odds_api: 98 credits left (below 120), about 4 more night(s)' in a


def test_credits_ok_above_threshold_no_alert(live, alerts_dir):
    with patch.object(live, 'get_credits_remaining', return_value=(446, 54)):
        live.report_odds_credits(start_remaining=500, n_games=9, warn_below=120)
    assert _alerts(alerts_dir) == ''


def test_zero_credits_is_an_error(live, alerts_dir):
    with patch.object(live, 'get_credits_remaining', return_value=(0, 500)):
        live.report_odds_credits(start_remaining=6, n_games=1, warn_below=120)
    assert '[ALERT ERROR] odds_api: 0 credits left' in _alerts(alerts_dir)


def test_credits_unreadable_is_a_warning(live, alerts_dir):
    with patch.object(live, 'get_credits_remaining', side_effect=requests.ConnectionError('offline')):
        assert live.report_odds_credits(start_remaining=None, n_games=3) is None
    assert '[ALERT WARN] odds_api: could not read credits left: offline' in _alerts(alerts_dir)


def test_props_summary_some_games_missing(live, alerts_dir, capsys):
    st = [('A @ B', 'ok', None), ('C @ D', 'Odds API returned no props', None)]
    assert live.summarize_props(st, 2) == 1
    assert '[PROPS] Odds API props for 1 of 2 games' in capsys.readouterr().out
    assert '[ALERT WARN] odds_api: no props for C @ D: Odds API returned no props' in _alerts(alerts_dir)


def test_props_summary_out_of_credits_is_error(live, alerts_dir):
    st = [('A @ B', 'fetch failed: 401 Client Error', 401), ('C @ D', 'fetch failed: 401 Client Error', 401)]
    assert live.summarize_props(st, 2) == 0
    a = _alerts(alerts_dir)
    assert '[ALERT ERROR] odds_api: props call refused (HTTP 401/429)' in a
    assert '[ALERT ERROR] odds_api: no props for any of 2 games' in a


def test_props_summary_crashed_worker_is_warned(live, alerts_dir):
    live.summarize_props([('A @ B', 'ok', None)], 2)
    assert '[ALERT WARN] live_ingestion: 1 game(s) crashed' in _alerts(alerts_dir)


# ── (b) ESPN ─────────────────────────────────────────────────────

def _event(state, with_odds=True, home='Miami Heat', away='Boston Celtics'):
    comp = {'competitors': [{'homeAway': 'home', 'team': {'displayName': home}},
                            {'homeAway': 'away', 'team': {'displayName': away}}],
            'status': {'type': {'state': state}},
            'odds': [{'provider': {'name': 'DraftKings'}, 'overUnder': 220.5}] if with_odds else []}
    return {'id': f'{home}-{state}', 'date': '2026-11-25T23:00Z', 'competitions': [comp]}


@pytest.mark.parametrize('payload, state', [
    ({'events': []}, 'no_games'),
    ({'events': [_event('post'), _event('in', home='Utah Jazz')]}, 'all_started'),
    ({'events': [_event('pre', with_odds=False)]}, 'no_odds'),
    ({'events': [_event('pre')]}, 'ok'),
])
def test_espn_status_tells_cases_apart(payload, state):
    from etl import espn_odds
    with patch.object(espn_odds, '_load_cache', return_value=None), \
         patch.object(espn_odds.requests, 'get', return_value=_resp({}, payload)):
        espn_odds.fetch_espn_odds()
    assert espn_odds.LAST_FETCH_STATUS['state'] == state


def test_espn_fetch_failure_status():
    from etl import espn_odds
    with patch.object(espn_odds, '_load_cache', return_value=None), \
         patch.object(espn_odds.requests, 'get', side_effect=requests.ConnectionError('ESPN down')):
        assert espn_odds.fetch_espn_odds() == {}
    assert espn_odds.LAST_FETCH_STATUS['state'] == 'fetch_failed'
    assert 'ESPN down' in espn_odds.LAST_FETCH_STATUS['error']


def _run_live_with_espn(live, status):
    from etl import espn_odds

    def fake_fetch():
        espn_odds._set_status(**status)
        return {}
    with patch.object(live, 'get_injured_players', return_value=(set(), {})), \
         patch.object(live, 'fetch_espn_odds', side_effect=fake_fetch):
        return live.generate_todays_projections(target_date=SLATE)


def test_espn_outage_is_error_and_failed(live, alerts_dir):
    out = _run_live_with_espn(live, {'state': 'fetch_failed', 'error': 'HTTP 503'})
    assert out == 'failed'
    assert '[ALERT ERROR] espn: scoreboard fetch failed (HTTP 503) -> no predictions tonight' in _alerts(alerts_dir)


def test_espn_late_run_is_warning(live, alerts_dir):
    out = _run_live_with_espn(live, {'state': 'all_started', 'n_events': 9, 'n_started': 9})
    assert out == 'late'
    assert '[ALERT WARN] espn: all 9 games had already started' in _alerts(alerts_dir)


def test_no_game_day_raises_no_alert(live, alerts_dir):
    assert _run_live_with_espn(live, {'state': 'no_games'}) == 'no_games'
    assert _alerts(alerts_dir) == ''


def test_espn_exception_is_error(live, alerts_dir):
    with patch.object(live, 'get_injured_players', return_value=(set(), {})), \
         patch.object(live, 'fetch_espn_odds', side_effect=RuntimeError('boom')):
        assert live.generate_todays_projections(target_date=SLATE) == 'failed'
    assert "[ALERT ERROR] espn: could not load tonight's games (boom)" in _alerts(alerts_dir)


# ── (b) CBS injuries ─────────────────────────────────────────────

def test_cbs_fetch_failure_is_error(alerts_dir):
    from etl import injury_scraper
    with patch.object(injury_scraper.requests, 'get', side_effect=requests.Timeout('timed out')):
        raw, norm = injury_scraper.get_injured_players()
    assert raw == set() and norm == {}
    assert ('[ALERT ERROR] cbs_injuries: fetch failed (timed out) -> all players treated as healthy'
            in _alerts(alerts_dir))


def test_cbs_layout_change_is_error_and_no_fallback(alerts_dir):
    from etl import injury_scraper
    html = ('<html><body><a href="/nba/players/1/lebron-james/">LeBron James</a>'
            '<a href="/nba/players/2/stephen-curry/">Stephen Curry</a></body></html>')
    r = _resp()
    r.text = html
    with patch.object(injury_scraper.requests, 'get', return_value=r):
        raw, _ = injury_scraper.get_injured_players()
    assert raw == set(), "the old fallback would have marked these healthy players out"
    assert '[ALERT ERROR] cbs_injuries: page loaded but the injury table was not found' in _alerts(alerts_dir)


def test_cbs_normal_table_no_alert(alerts_dir):
    from etl import injury_scraper
    html = ('<table><tr class="TableBase-bodyTr"><td><span class="CellPlayerName--long">'
            '<a href="#">Joel Embiid</a></span></td><td></td><td></td><td></td>'
            '<td>Expected to be out until at least Dec 1</td></tr></table>')
    r = _resp()
    r.text = html
    with patch.object(injury_scraper.requests, 'get', return_value=r):
        raw, _ = injury_scraper.get_injured_players()
    assert raw == {'Joel Embiid'}
    assert _alerts(alerts_dir) == ''


# ── (a) grading catch-up ─────────────────────────────────────────

def _write(path, content, binary=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb' if binary else 'w', **({} if binary else {'encoding': 'utf-8'})) as f:
        f.write(content)


def _live_tree(base, dates_pred, dates_graded):
    from utils.prediction_paths import (get_prediction_filepath, get_results_filepath,
                                        get_analyze_report_filepath)
    for d in dates_pred:
        _write(get_prediction_filepath(d, mode='live', ext='txt', base_dir=base),
               f"[+] Execution mode: LIVE | Date: {d}\nMATCHUP: Boston Celtics @ Miami Heat\n"
               f"PROJECTED SCORE: MIA 105.0 - BOS 110.0\n"
               f"Jayson Tatum         -> 26 PTS, 8 REB, 4 AST\n")
    for d in dates_graded:
        _write(get_results_filepath(d, mode='live', ext='txt', base_dir=base),
               "MATCHUP: BOS vs. MIA    110 - 105\nJayson Tatum           | 26 PTS, 8 REB,  4 AST\n")
        _write(get_analyze_report_filepath(d, mode='live', ext='pdf', base_dir=base),
               b"%PDF-1.4 " + b"0" * 6000, binary=True)


def test_catch_up_finds_and_grades_missed_dates(tmp_path, alerts_dir):
    from etl.prepare_analyze import catch_up_missed_grading, find_ungraded_live_dates
    base = str(tmp_path / 'model')
    _live_tree(base, ['2026-11-20', '2026-11-22', '2026-11-23'], ['2026-11-22'])
    assert find_ungraded_live_dates('2026-11-24', base_dir=base) == ['2026-11-20', '2026-11-23']
    assert find_ungraded_live_dates('2026-11-24', base_dir=base, earliest='2026-11-21') == ['2026-11-23']

    runner = MagicMock(side_effect=[1, 0])      # first date fails, second still runs
    out = catch_up_missed_grading('2026-11-24', base_dir=base, runner=runner, earliest='2026-01-01')
    assert out == {'2026-11-20': 1, '2026-11-23': 0}
    assert [c.kwargs['target_date'] for c in runner.call_args_list] == ['2026-11-20', '2026-11-23']
    assert all(c.kwargs['mode'] == 'live' for c in runner.call_args_list)
    assert '[ALERT ERROR] prepare_analyze: catch-up grading of 2026-11-20 failed' in _alerts(alerts_dir)


def test_catch_up_respects_cutoff_setting(tmp_path):
    from etl.prepare_analyze import catch_up_missed_grading
    import config.settings as cfg
    base = str(tmp_path / 'model')
    _live_tree(base, ['2026-11-19', '2026-11-21'], [])
    runner = MagicMock(return_value=0)
    with patch.object(cfg, 'GRADING_CATCHUP_FROM', '2026-11-25'):
        assert catch_up_missed_grading('2026-11-26', base_dir=base, runner=runner) == {}
    runner.assert_not_called()


# ── (a) run_pipeline.sh: prepare failure no longer stops the night ──

STUB = r"""#!/bin/bash
echo "$*" >> "$STUB_CALLS"
case "$1 $2" in
  "-m utils.slate_date"|"-m utils.run_alerts") exec "$REAL_PY" "$@" ;;
  "-m etl.prepare_analyze") exit "${RC_PREPARE:-0}" ;;
  "-m etl.live_ingestion") exit "${RC_LIVE:-0}" ;;
  "-m ml.monitoring") exit "${RC_MONITORING:-0}" ;;
esac
exit 0
"""


def _run_pipeline(tmp_path, **rc):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    for name, body in (('python3', STUB), ('sqlite3', '#!/bin/bash\nexit 0\n')):
        p = bin_dir / name
        p.write_text(body, encoding='utf-8', newline='\n')
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    calls = tmp_path / 'calls.txt'
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ.get('PATH', '')}", REAL_PY=sys.executable,
               STUB_CALLS=str(calls), PIPELINE_LOG_ROOT=str(tmp_path / 'logs'),
               PIPELINE_ALERTS_DIR=str(tmp_path / 'scheduled'), PIPELINE_NOTIFY='0',
               **{k: str(v) for k, v in rc.items()})
    r = subprocess.run(['bash', os.path.join(MODEL_DIR, 'scripts', 'run_pipeline.sh')],
                       env=env, capture_output=True, text=True, timeout=120)
    return r, calls.read_text() if calls.exists() else ''


bash_only = pytest.mark.skipif(sys.platform.startswith('win'), reason='needs bash (run in WSL)')


@bash_only
def test_pipeline_prepare_failure_still_predicts(tmp_path):
    r, calls = _run_pipeline(tmp_path, RC_PREPARE=1, RC_MONITORING=2)
    assert r.returncode == 0, r.stdout + r.stderr
    assert '-m etl.live_ingestion' in calls, "tonight's predictions must still run"
    assert '[ALERT ERROR] prepare_analyze: grading yesterday failed (exit 1)' in r.stdout
    assert '[ALERT WARN] monitoring: exited with code 2' in r.stdout
    assert 'NIGHT STATUS: DEGRADED (1 error, 1 warning)' in r.stdout
    alerts = next((tmp_path / 'scheduled').glob('alerts_*.log')).read_text(encoding='utf-8')
    assert 'grading yesterday failed' in alerts


@bash_only
def test_pipeline_live_failure_is_failed_night(tmp_path):
    r, calls = _run_pipeline(tmp_path, RC_LIVE=1)
    assert r.returncode == 1
    assert '[ALERT ERROR] live_ingestion: exited with code 1' in r.stdout
    assert 'NIGHT STATUS: FAILED' in r.stdout
    assert '-m etl.seed_all_live' not in calls


@bash_only
def test_pipeline_clean_night_is_ok(tmp_path):
    r, _ = _run_pipeline(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'NIGHT STATUS: OK' in r.stdout
