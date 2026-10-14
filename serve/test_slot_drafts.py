"""Backward-compatible per-slot draft accounting, CPU-only."""
import unittest
from serve.server import StrataEngine

class SlotDrafts(unittest.TestCase):
    def engine(self, **last):
        e = StrataEngine.__new__(StrataEngine)
        e.batch = 2
        e.last = last
        return e

    def test_slot_counts_add_to_admission(self):
        e = self.engine(generated=1, drafts_offered=4, drafts_accepted=2)
        e._apply_bdone('BDONE 1 10 stop 42.5 3 5'.split())
        self.assertEqual((e.last['drafts_accepted'], e.last['drafts_offered']), (5, 9))
        self.assertEqual((e.last['generated'], e.last['decode_ms']), (10, 42.5))

    def test_old_engine_line_preserves_admission(self):
        e = self.engine(drafts_offered=4, drafts_accepted=2)
        e._apply_bdone('BDONE 1 10 length 42.5'.split())
        self.assertEqual((e.last['drafts_accepted'], e.last['drafts_offered']), (2, 4))

    def test_no_counts_stays_unknown_for_old_engine(self):
        e = self.engine()
        e._apply_bdone('BDONE 1 10 length 42.5'.split())
        self.assertNotIn('drafts_offered', e.last)
        e._apply_bdone('BDONE 1 10 length 42.5 0 0'.split())
        self.assertEqual((e.last['drafts_accepted'], e.last['drafts_offered']), (0, 0))
