"""serve/test_slot_alloc.py - the batch slot choice with pipelined groups (STRATA_SLOT_LOW_FIRST)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

class LowFirstInGroup(unittest.TestCase):
    """STRATA_SLOT_LOW_FIRST: in an empty pipeline group the first request takes the group's slot 0, whatever the
    least-recently-used order says, so the group's window does not carry a pad row below it."""
    def test_empty_group_takes_its_lowest_slot(self):
        from serve.server import StrataEngine
        e = StrataEngine.__new__(StrataEngine)
        e.batch, e.slot_groups = 8, 4
        e.slot_order = [g * 2 + t for t in range(2) for g in range(4)]
        e.slot_group = [b // 2 for b in range(8)]
        e.slot_busy = [False] * 8
        e.slot_held = [[] for _ in range(8)]
        e.slot_used = [10, 1, 10, 1, 10, 1, 10, 1]          # every odd slot used longer ago
        e.slot_busy[0] = e.slot_busy[2] = e.slot_busy[4] = True
        self.assertEqual(e.pick_slot([1, 2, 3]), 6)          # the empty group 3: its slot 6, not 7
        e.slot_busy[6] = True
        self.assertIn(e.pick_slot([1, 2, 3]), (1, 3, 5, 7))  # then a second slot in some group


if __name__ == "__main__":
    unittest.main()
