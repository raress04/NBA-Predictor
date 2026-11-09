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


if __name__ == '__main__':
    unittest.main()
