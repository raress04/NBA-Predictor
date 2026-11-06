"""
tests/test_slate_date.py
========================
Slate date = US-Eastern calendar date of the run (handoff 06.11 "Prevention").
All inputs are timezone-aware, so results do not depend on the machine's timezone.
"""
import inspect
import os
import sys
import unittest
import unittest.mock
from datetime import datetime, timedelta, timezone

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from utils import slate_date as sd

UTC = timezone.utc

# (UTC time, expected US-Eastern slate date, why)
CASES = [
    (datetime(2026, 10, 29, 22, 59, tzinfo=UTC), '2026-10-29', 'EDT evening run from Europe after local midnight'),
    (datetime(2026, 10, 30, 3, 59, tzinfo=UTC), '2026-10-29', 'EDT 23:59'),
    (datetime(2026, 10, 30, 4, 0, tzinfo=UTC), '2026-10-30', 'EDT midnight'),
    (datetime(2026, 11, 6, 4, 30, tzinfo=UTC), '2026-11-05', 'EST 23:30 (old fixed -4 h said Nov 6)'),
    (datetime(2026, 11, 6, 5, 0, tzinfo=UTC), '2026-11-06', 'EST midnight'),
    (datetime(2026, 11, 1, 5, 30, tzinfo=UTC), '2026-11-01', 'DST end day, 01:30 EDT'),
    (datetime(2026, 11, 1, 6, 30, tzinfo=UTC), '2026-11-01', 'DST end day, 01:30 EST'),
    (datetime(2026, 11, 1, 4, 30, tzinfo=UTC), '2026-11-01', 'DST end day, 00:30 EDT'),
    (datetime(2027, 3, 14, 4, 59, tzinfo=UTC), '2027-03-13', 'DST start day, 23:59 EST'),
    (datetime(2027, 3, 14, 7, 30, tzinfo=UTC), '2027-03-14', 'DST start day, 03:30 EDT'),
    (datetime(2027, 3, 15, 3, 59, tzinfo=UTC), '2027-03-14', 'after DST start, 23:59 EDT'),
    (datetime(2027, 3, 15, 4, 0, tzinfo=UTC), '2027-03-15', 'after DST start, midnight EDT'),
]


class TestSlateDate(unittest.TestCase):

    def test_known_cases(self):
        for utc_dt, expected, why in CASES:
            with self.subTest(why=why):
                self.assertEqual(sd.slate_date(utc_dt), expected)
                self.assertEqual(sd.slate_date_compact(utc_dt), expected.replace('-', ''))

    def test_fallback_without_zoneinfo(self):
        """The DST fallback (used when the IANA tz database is missing) gives the same answers."""
        with unittest.mock.patch.object(sd, '_ET', None):
            for utc_dt, expected, why in CASES:
                with self.subTest(why=why):
                    self.assertEqual(sd.slate_date(utc_dt), expected)

    def test_fallback_matches_zoneinfo_hourly(self):
        if sd._ET is None:
            self.skipTest('zoneinfo/tzdata not available on this Python')
        t = datetime(2026, 1, 1, tzinfo=UTC)
        while t < datetime(2028, 1, 1, tzinfo=UTC):
            zi = sd.to_eastern(t).strftime('%Y-%m-%d %H')
            with unittest.mock.patch.object(sd, '_ET', None):
                fb = sd.to_eastern(t).strftime('%Y-%m-%d %H')
            self.assertEqual(zi, fb, msg=str(t))
            t += timedelta(hours=1)

    def test_parlay_builder_uses_slate_date_not_local_now(self):
        """persist_parlay() and the ML layer must get the slate date, never datetime.now()."""
        from simulator import parlay_builder as pb
        src = inspect.getsource(pb.build_parlays)
        self.assertNotIn('datetime.now', src)
        self.assertIn('game_date', inspect.signature(pb.build_parlays).parameters)

    def test_espn_fetch_uses_slate_date(self):
        from etl import espn_odds
        src = inspect.getsource(espn_odds.fetch_espn_odds)
        self.assertIn('slate_date_compact', src)
        self.assertNotIn('timedelta(hours=-4)', src)


if __name__ == '__main__':
    unittest.main()
