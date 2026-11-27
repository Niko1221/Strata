"""The Inkling engine's conversation-cache log format: synthetic fixtures through ConvCacheLog's INK_* regexes.

    python -m unittest serve.test_inkling_cache_log_clear
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve.server import ConvCacheLog  # noqa: E402

MIB = 1024 * 1024


class InklingConversationCache(unittest.TestCase):
    """The inkling2 engine's log lines, on the INK_* format."""

    PARK = "strata serve: conversation cache: parked 67621 tokens (1.5 MiB; 5 parked, 6 MiB, cap 15000 MiB / 8 slots)\n"
    RESTORE = "strata serve: conversation cache: restored parked conversation 4000 tokens (4 parked, 3 MiB)\n"
    SKIPPED = "strata serve: conversation cache: skipped 5000 tokens (est 2.5 MiB; budget left 4.5 MiB)\n"
    DROPPED = "strata serve: conversation cache: dropped 1 superseded copy of this conversation; 4 parked, 5 MiB\n"
    EVICTED = "strata serve: conversation cache: evicted 40000 tokens (2 evictions total; 4 parked, 5 MiB)\n"
    EVICTED_THEN_PARKED = EVICTED + "strata serve: conversation cache: parked 3000 tokens (1.0 MiB; 3 parked, 2 MiB, cap 15000 MiB / 8 slots)\n"

    def poll(self, *lines, start=0):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text("".join(lines), encoding="utf-8")
            return ConvCacheLog().poll(str(log), start)

    def test_park_sets_parked_and_bytes(self):
        st = self.poll(self.PARK)
        self.assertEqual((st["parked"], st["bytes"]), (5, 6 * MIB))
        self.assertEqual((st["parks"], st["last_event"], st["last_tokens"]), (1, "parked", 67621))

    def test_restore_sets_parked_and_restores(self):
        st = self.poll(self.RESTORE)
        self.assertEqual((st["parked"], st["bytes"]), (4, 3 * MIB))
        self.assertEqual((st["restores"], st["last_event"], st["last_tokens"]), (1, "restored", 4000))

    def test_skipped_is_a_refused_park(self):
        st = self.poll(self.SKIPPED)
        self.assertEqual((st["refused"], st["refused_total"], st["last_refused"]), ({"put": 1}, 1, "put"))
        self.assertEqual((st["parked"], st["bytes"], st["parks"], st["restores"]), (0, 0, 0, 0))

    def test_dropped_sets_parked_bytes_without_a_park_or_restore(self):
        st = self.poll(self.DROPPED)
        self.assertEqual((st["parked"], st["bytes"]), (4, 5 * MIB))
        self.assertEqual((st["parks"], st["restores"], st["last_event"], st["last_tokens"]), (0, 0, None, None))

    def test_evicted_sets_evictions_and_parked_without_an_event(self):
        st = self.poll(self.EVICTED)
        self.assertEqual((st["evictions"], st["parked"], st["bytes"]), (2, 4, 5 * MIB))
        self.assertEqual((st["parks"], st["restores"], st["last_event"], st["last_tokens"]), (0, 0, None, None))

    def test_eviction_followed_by_a_park_overwrites_eviction_state(self):
        st = self.poll(self.EVICTED_THEN_PARKED)
        self.assertEqual((st["evictions"], st["parks"], st["last_event"], st["last_tokens"]), (2, 1, "parked", 3000))
        self.assertEqual((st["parked"], st["bytes"]), (3, 2 * MIB))

    def test_park_and_restore_sequence_keeps_the_restores(self):
        st = self.poll(self.PARK, self.RESTORE)
        self.assertEqual((st["parks"], st["restores"]), (1, 1))
        self.assertEqual((st["last_event"], st["last_tokens"]), ("restored", 4000))
        self.assertEqual((st["parked"], st["bytes"]), (4, 3 * MIB))   # the restore replaced the park's state

    def test_ink_lines_are_not_counted_by_the_qwen_event_regex(self):
        # The generic EVENT parser must not double-count an Inkling line.
        st = self.poll(self.PARK)
        self.assertEqual(st["parked"], 5)                             # not 5, not 0 — exactly the park's reading
        self.assertEqual((st["parks"], st["restores"]), (1, 0))       # no phantom park/restore from EVENT matching


if __name__ == "__main__":
    unittest.main()
