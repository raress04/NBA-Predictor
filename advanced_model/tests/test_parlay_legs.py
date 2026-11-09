"""persist_parlay must store the player's full name on each leg (bug found 2026-11-09)."""
import os
import sqlite3
import sys
import tempfile
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator import parlay_builder as pb  # noqa: E402


class TestPersistParlayNames(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        c = sqlite3.connect(self.path)
        c.execute("""CREATE TABLE parlays (id INTEGER PRIMARY KEY AUTOINCREMENT, game_date TEXT, model_version TEXT,
                     combined_odds REAL, stake REAL, payout REAL, result TEXT, roi REAL, created_at TEXT)""")
        c.execute("""CREATE TABLE parlay_legs (id INTEGER PRIMARY KEY AUTOINCREMENT, parlay_id INTEGER, leg_number INTEGER,
                     pick_id INTEGER, player TEXT, stat_category TEXT, direction TEXT, line REAL, odds REAL,
                     sim_confidence REAL, edge_pct REAL)""")
        c.commit()
        c.close()

    def tearDown(self):
        os.unlink(self.path)

    def test_full_player_name_is_stored(self):
        legs = [{'type': 'prop', 'player': 'Evan Mobley', 'pick': 'Evan Mobley Under 9.5 REB (Sim: 76%)',
                 'stat_category': 'REBOUNDS', 'direction': 'Under', 'line': 9.5, 'odds': 1.94, 'confidence': 75.9},
                {'type': 'prop', 'player': "Day'Ron Sharpe", 'pick': "Day'Ron Sharpe Under 8.5 REB",
                 'stat_category': 'REBOUNDS', 'direction': 'Under', 'line': 8.5, 'odds': 1.69, 'confidence': 76.4}]
        with unittest.mock.patch('etl.bet_tracker._get_conn', create=True,
                                 new=lambda: sqlite3.connect(self.path)):
            pid = pb.persist_parlay(legs, '2026-11-09')
        self.assertGreater(pid, 0)
        c = sqlite3.connect(self.path)
        names = [r[0] for r in c.execute("SELECT player FROM parlay_legs ORDER BY leg_number")]
        c.close()
        self.assertEqual(names, ['Evan Mobley', "Day'Ron Sharpe"])


class TestSettleParlays(unittest.TestCase):
    """Parlays are settled from box scores, not via the never-set pick_id (fixed 2026-11-09)."""

    def setUp(self):
        self.c = sqlite3.connect(':memory:')
        self.c.executescript("""
            CREATE TABLE parlays (id INTEGER PRIMARY KEY, game_date TEXT, stake REAL, combined_odds REAL,
                                  payout REAL, result TEXT, roi REAL);
            CREATE TABLE parlay_legs (id INTEGER PRIMARY KEY, parlay_id INTEGER, leg_number INTEGER, pick_id INTEGER,
                                      player TEXT, stat_category TEXT, direction TEXT, line REAL, odds REAL);
            CREATE TABLE pick_results (id INTEGER PRIMARY KEY, pick_id INTEGER, is_win INTEGER, voided INTEGER);
        """)
        self.box = {'Evan Mobley': {'POINTS': 20, 'REBOUNDS': 6, 'ASSISTS': 3},
                    'Ty Jerome': {'POINTS': 10, 'REBOUNDS': 2, 'ASSISTS': 5},
                    'Jamal Shead': {'POINTS': 8, 'REBOUNDS': 2, 'ASSISTS': 16}}

    def add(self, pid, legs):
        self.c.execute("INSERT INTO parlays (id, game_date, stake, combined_odds) VALUES (?, '2026-11-06', 0.01, 3.0)",
                       (pid,))
        for i, (pl, cat, d, line, odds) in enumerate(legs, 1):
            self.c.execute("INSERT INTO parlay_legs (parlay_id, leg_number, player, stat_category, direction, line, odds)"
                           " VALUES (?,?,?,?,?,?,?)", (pid, i, pl, cat, d, line, odds))

    def settle(self):
        from etl.fetch_range_box_scores import resolve_parlays
        return {r[0]: r for r in resolve_parlays(self.c.cursor(), '2026-11-06', self.box)}

    def test_win_loss_push_open(self):
        self.add(1, [('Evan Mobley', 'REBOUNDS', 'Under', 9.5, 1.9), ('Ty Jerome', 'ASSISTS', 'Under', 5.5, 1.8)])
        self.add(2, [('Evan Mobley', 'REBOUNDS', 'Under', 9.5, 1.9), ('Jamal Shead', 'ASSISTS', 'Under', 8.5, 1.8)])
        self.add(3, [('Evan Mobley', 'REBOUNDS', 'Under', 9.5, 1.9), ('Ty Jerome', 'ASSISTS', 'Over', 5.0, 1.8)])
        self.add(4, [('Evan Mobley', 'REBOUNDS', 'Under', 9.5, 1.9), ('Nobody Here', 'POINTS', 'Over', 9.5, 1.8)])
        r = self.settle()
        self.assertEqual(r[1][1], 'WIN')
        self.assertAlmostEqual(r[1][2], 0.01 * 1.9 * 1.8, places=6)
        self.assertEqual(r[2][1], 'LOSS')
        self.assertEqual(r[3][1], 'WIN')                 # Ty Jerome 5 vs 5.0 -> push, leg dropped
        self.assertAlmostEqual(r[3][2], 0.01 * 1.9, places=6)
        self.assertNotIn(4, r)                           # player missing -> stays open
        self.assertIsNone(self.c.execute("SELECT result FROM parlays WHERE id=4").fetchone()[0])

    def test_dry_run_writes_nothing(self):
        from etl.fetch_range_box_scores import resolve_parlays
        self.add(1, [('Evan Mobley', 'REBOUNDS', 'Under', 9.5, 1.9)])
        resolve_parlays(self.c.cursor(), '2026-11-06', self.box, dry_run=True)
        self.assertIsNone(self.c.execute("SELECT result FROM parlays WHERE id=1").fetchone()[0])


if __name__ == '__main__':
    unittest.main()
