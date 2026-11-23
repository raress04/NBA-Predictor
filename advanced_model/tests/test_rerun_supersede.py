"""W1 (Plans/W1_RERUN_AUDIT.md): a re-run of a slate replaces its rows instead of adding to them.

The latest run is the record, game by game: earlier rows are flagged superseded = 1, never
deleted, and a game the re-run did not simulate keeps its earlier rows.
"""
import os
import sqlite3
import sys
import tempfile
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import rerun  # noqa: E402
from database.bet_tracker_schema import ensure_rerun_columns  # noqa: E402

D = '2026-11-23'


def _tmp_db():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    return path


def _shadow_schema(conn):
    conn.execute("""CREATE TABLE shadow_picks (id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_date TEXT, player TEXT, stat_category TEXT, direction TEXT, line REAL,
        projection REAL, sim_confidence REAL, edge_pct REAL, ban_reason TEXT,
        actual_value REAL, actual_result INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")


def _row(player, cat, line=20.5, matchup=None, direction='OVER'):
    return {'game_date': D, 'player': player, 'stat_category': cat, 'direction': direction,
            'line': line, 'projection': line + 2, 'sim_confidence': 60.0, 'edge_pct': 10.0,
            'ban_reason': 'MIN_GAP', 'matchup': matchup}


class TestShadowPicksRerun(unittest.TestCase):
    def setUp(self):
        self.path = _tmp_db()
        c = sqlite3.connect(self.path)
        _shadow_schema(c)
        c.commit()
        c.close()

    def tearDown(self):
        os.unlink(self.path)

    def _ingest(self, rows):
        from etl import db_manager
        with unittest.mock.patch.object(db_manager, 'DB_PATH', self.path):
            return db_manager.ingest_shadow_picks(rows)

    def _current(self):
        c = sqlite3.connect(self.path)
        rows = c.execute("SELECT player, stat_category, line, matchup FROM shadow_picks "
                         "WHERE COALESCE(superseded,0)=0 ORDER BY player, stat_category, matchup").fetchall()
        total = c.execute("SELECT COUNT(*) FROM shadow_picks").fetchone()[0]
        c.close()
        return rows, total

    def test_rerun_replaces_same_candidates(self):
        self._ingest([_row('A Player', 'POINTS', 20.5), _row('B Player', 'REBOUNDS', 7.5)])
        self._ingest([_row('A Player', 'POINTS', 21.5), _row('B Player', 'REBOUNDS', 7.5)])
        current, total = self._current()
        self.assertEqual(total, 4, "nothing is deleted")
        self.assertEqual([(r[0], r[2]) for r in current], [('A Player', 21.5), ('B Player', 7.5)],
                         "only the latest run's rows are current")

    def test_game_not_in_rerun_keeps_its_rows(self):
        # first run: two games; the re-run only sees game 2 (game 1 already started)
        self._ingest([_row('Early Game', 'POINTS'), _row('Late Game', 'POINTS')])
        self._ingest([_row('Late Game', 'POINTS', 22.5)])
        current, _ = self._current()
        self.assertEqual([(r[0], r[2]) for r in current], [('Early Game', 20.5), ('Late Game', 22.5)])

    def test_totals_of_different_games_do_not_replace_each_other(self):
        run = [_row('Game Total', 'TOTAL', 228.5, 'A @ B'), _row('Game Total', 'TOTAL', 221.5, 'C @ D')]
        self._ingest(run)
        current, _ = self._current()
        self.assertEqual(len(current), 2, "two games in one run are both current")
        self._ingest([_row('Game Total', 'TOTAL', 227.5, 'A @ B')])
        current, _ = self._current()
        self.assertEqual(sorted((r[3], r[2]) for r in current), [('A @ B', 227.5), ('C @ D', 221.5)])

    def test_first_run_flags_nothing(self):
        self._ingest([_row('A Player', 'POINTS'), _row('A Player', 'ASSISTS', 5.5)])
        current, total = self._current()
        self.assertEqual((len(current), total), (2, 2))


class TestProjectionOutcomesRerun(unittest.TestCase):
    def setUp(self):
        self.path = _tmp_db()
        c = sqlite3.connect(self.path)
        c.execute("""CREATE TABLE projection_outcomes (id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date TEXT NOT NULL, player TEXT NOT NULL, category TEXT NOT NULL,
            projected_value REAL NOT NULL, bookmaker_line REAL, actual_value REAL NOT NULL,
            model_bias REAL NOT NULL, source_mode TEXT NOT NULL, source_file TEXT NOT NULL)""")
        ensure_rerun_columns(c)
        c.commit()
        c.close()

    def tearDown(self):
        os.unlink(self.path)

    def test_rerun_projection_replaces_earlier_row(self):
        from etl import live_ingestion as li
        with unittest.mock.patch.object(li, 'BET_TRACKER_DB', self.path):
            li.log_live_projection('A Player', 'POINTS', 21.0, 20.5, game_date=D, matchup='A @ B')
            li.log_live_projection('Game Total', 'TOTAL', 230.0, 228.5, game_date=D, matchup='A @ B')
            li.log_live_projection('Game Total', 'TOTAL', 220.0, 221.5, game_date=D, matchup='C @ D')
            # re-run
            li.log_live_projection('A Player', 'POINTS', 22.0, 20.5, game_date=D, matchup='A @ B')
            li.log_live_projection('Game Total', 'TOTAL', 231.0, 228.5, game_date=D, matchup='A @ B')
        c = sqlite3.connect(self.path)
        cur = c.execute("SELECT player, matchup, projected_value FROM projection_outcomes "
                        "WHERE COALESCE(superseded,0)=0 ORDER BY player, matchup").fetchall()
        total = c.execute("SELECT COUNT(*) FROM projection_outcomes").fetchone()[0]
        c.close()
        self.assertEqual(total, 5)
        self.assertEqual(cur, [('A Player', 'A @ B', 22.0), ('Game Total', 'A @ B', 231.0),
                               ('Game Total', 'C @ D', 220.0)])

    def test_old_schema_still_logs(self):
        path = _tmp_db()
        c = sqlite3.connect(path)
        c.execute("""CREATE TABLE projection_outcomes (id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date TEXT NOT NULL, player TEXT NOT NULL, category TEXT NOT NULL,
            projected_value REAL NOT NULL, bookmaker_line REAL, actual_value REAL NOT NULL,
            model_bias REAL NOT NULL, source_mode TEXT NOT NULL, source_file TEXT NOT NULL)""")
        c.commit()
        c.close()
        from etl import live_ingestion as li
        with unittest.mock.patch.object(li, 'BET_TRACKER_DB', path):
            li.log_live_projection('A Player', 'POINTS', 21.0, 20.5, game_date=D, matchup='A @ B')
        c = sqlite3.connect(path)
        n = c.execute("SELECT COUNT(*) FROM projection_outcomes").fetchone()[0]
        c.close()
        os.unlink(path)
        self.assertEqual(n, 1, "without the W1 columns the row is still logged")


class TestMlShadowRerun(unittest.TestCase):
    def test_supersede_ml_shadow(self):
        path = _tmp_db()
        c = sqlite3.connect(path)
        c.execute("""CREATE TABLE ml_shadow_picks (id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date TEXT NOT NULL, player TEXT NOT NULL, stat_category TEXT NOT NULL,
            direction TEXT NOT NULL, line REAL, projection REAL, sim_confidence REAL,
            xgb_confidence REAL, edge_pct REAL, model_version TEXT, cohort TEXT,
            created_at TEXT, actual_value REAL, is_win INTEGER, resolved_at TEXT)""")
        for p, cat in (('A Player', 'POINTS'), ('B Player', 'ASSISTS')):
            c.execute("INSERT INTO ml_shadow_picks (game_date, player, stat_category, direction) "
                      "VALUES (?, ?, ?, 'OVER')", (D, p, cat))
        c.commit()
        c.close()
        from ml.shadow_logger import supersede_ml_shadow
        n = supersede_ml_shadow(D, [('A Player', 'points')], db_path=path)
        c = sqlite3.connect(path)
        got = c.execute("SELECT player, superseded FROM ml_shadow_picks ORDER BY player").fetchall()
        c.close()
        os.unlink(path)
        self.assertEqual(n, 1)
        self.assertEqual(got, [('A Player', 1), ('B Player', 0)])


class TestParlaysRerun(unittest.TestCase):
    def setUp(self):
        self.path = _tmp_db()
        c = sqlite3.connect(self.path)
        c.execute("""CREATE TABLE parlays (id INTEGER PRIMARY KEY AUTOINCREMENT, game_date TEXT,
            model_version TEXT, combined_odds REAL, stake REAL, payout REAL, result TEXT, roi REAL,
            created_at TEXT)""")
        c.execute("""CREATE TABLE parlay_legs (id INTEGER PRIMARY KEY AUTOINCREMENT, parlay_id INTEGER,
            leg_number INTEGER, pick_id INTEGER, player TEXT, stat_category TEXT, direction TEXT,
            line REAL, odds REAL, sim_confidence REAL, edge_pct REAL)""")
        c.execute("""CREATE TABLE pick_results (id INTEGER PRIMARY KEY AUTOINCREMENT, pick_id INTEGER,
            actual_value REAL, is_win INTEGER, diff_val REAL, diff_pct REAL, resolved_at TEXT,
            voided INTEGER DEFAULT 0)""")
        ensure_rerun_columns(c)
        for _ in range(2):   # first run: 2 parlays
            c.execute("INSERT INTO parlays (game_date, stake) VALUES (?, 0.002)", (D,))
        c.execute("INSERT INTO parlays (game_date, stake) VALUES ('2026-11-22', 0.002)")
        c.commit()
        self.conn = c

    def tearDown(self):
        self.conn.close()
        os.unlink(self.path)

    def test_rerun_keeps_only_its_own_parlays(self):
        new_id = self.conn.execute("INSERT INTO parlays (game_date, stake) VALUES (?, 0.002)", (D,)).lastrowid
        n = rerun.supersede_parlays(self.conn, D, [new_id])
        cur = [r[0] for r in self.conn.execute(
            "SELECT id FROM parlays WHERE game_date=? AND superseded=0", (D,))]
        other_day = self.conn.execute(
            "SELECT superseded FROM parlays WHERE game_date='2026-11-22'").fetchone()[0]
        self.assertEqual((n, cur, other_day), (2, [new_id], 0))

    def test_rerun_with_no_parlay_supersedes_all(self):
        self.assertEqual(rerun.supersede_parlays(self.conn, D, []), 2)

    def test_superseded_parlay_is_not_settled(self):
        from etl.fetch_range_box_scores import resolve_parlays
        new_id = self.conn.execute("INSERT INTO parlays (game_date, stake) VALUES (?, 0.002)", (D,)).lastrowid
        self.conn.execute("INSERT INTO parlay_legs (parlay_id, leg_number, player, stat_category, direction, "
                          "line, odds) VALUES (?, 1, 'A Player', 'POINTS', 'OVER', 20.5, 1.9)", (new_id,))
        for pid in (1, 2):
            self.conn.execute("INSERT INTO parlay_legs (parlay_id, leg_number, player, stat_category, direction, "
                              "line, odds) VALUES (?, 1, 'A Player', 'POINTS', 'OVER', 20.5, 1.9)", (pid,))
        rerun.supersede_parlays(self.conn, D, [new_id])
        settled = resolve_parlays(self.conn.cursor(), D,
                                  {'A Player': {'POINTS': 25, 'REBOUNDS': 5, 'ASSISTS': 3}}, dry_run=True)
        self.assertEqual([s[0] for s in settled], [new_id])


class TestCurrentOnly(unittest.TestCase):
    def test_fragment_only_when_column_exists(self):
        c = sqlite3.connect(':memory:')
        c.execute("CREATE TABLE parlays (id INTEGER)")
        self.assertEqual(rerun.current_only(c, 'parlays'), '')
        c.execute("ALTER TABLE parlays ADD COLUMN superseded INTEGER DEFAULT 0")
        self.assertIn('superseded', rerun.current_only(c, 'parlays'))
        c.close()


class TestDailyCapRows(unittest.TestCase):
    def test_cap_rows_store_the_real_candidate(self):
        """DAILY_CAP rows used to store the pick label as the player and line 0."""
        from simulator import parlay_builder as pb
        import config.settings as settings
        dist = [30] * 900 + [10] * 100   # 90% over 20.5
        props, dists, minutes = {}, {}, {}
        for i in range(3):
            name = f'Player {i}'
            props[name] = {'points': {'line': 20.5, 'over_odds': 1.9, 'under_odds': 1.9, 'book': 'x'}}
            dists[name] = {'PTS': dist, 'REB': [5] * 1000, 'AST': [3] * 1000}
            minutes[name] = {'team': 'B', 'mean': 32, 'std': 2, 'usage': 15}
        odds = {'e1': {'home_team': 'B', 'away_team': 'A', 'spreads': None, 'totals': None}}
        with unittest.mock.patch.object(pb, 'persist_parlay', return_value=None), \
             unittest.mock.patch.object(pb, 'get_category_calibration', return_value=1.0), \
             unittest.mock.patch.object(pb, 'get_tiered_bias', return_value=0.0), \
             unittest.mock.patch.object(settings, 'MAX_VALUABLE_PICKS_PER_DAY', 1), \
             unittest.mock.patch('builtins.print'):
            out = pb.build_parlays([], odds, {'e1': props}, dists, {'e1': {'home': [100], 'away': [95]}},
                                   all_player_minutes=minutes, game_date=D)
        cap = [r for r in out['shadow_picks'] if r['ban_reason'] == 'DAILY_CAP']
        self.assertEqual(len(out['all_picks']), 1)
        self.assertEqual(len(cap), 2)
        for r in cap:
            self.assertTrue(r['player'].startswith('Player '), r['player'])
            self.assertEqual((r['stat_category'], r['line']), ('POINTS', 20.5))
            self.assertEqual(r['matchup'], 'A @ B')


if __name__ == '__main__':
    unittest.main()
