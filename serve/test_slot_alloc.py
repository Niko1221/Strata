"""serve/test_slot_alloc.py - which batch slot a request gets (StrataEngine.pick_slot), no engine needed.

    python -m unittest serve.test_slot_alloc -v
"""
from __future__ import annotations

import random
import sys
import unittest
import threading
import time
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import StrataEngine  # noqa: E402
from serve import slot_policy  # noqa: E402


def engine(batch, groups, pipelined=True):
    e = object.__new__(StrataEngine)          # pick_slot reads only the slot tables
    gs = batch // groups
    e.batch = batch
    e.slot_order = [g * gs + t for t in range(gs) for g in range(groups)]
    e.slot_busy = [False] * batch
    e.slot_held = [[] for _ in range(batch)]
    e.slot_used = [0.0] * batch
    e.slot_gs = gs if pipelined and groups > 1 else 0
    return e


class Pipelined(unittest.TestCase):
    def test_spreads_over_the_groups_first(self):
        e = engine(6, 3)
        got = []
        for _ in range(6):
            b = e.pick_slot([1, 2, 3])
            e.slot_busy[b] = True
            got.append(b)
        self.assertEqual(got, [0, 2, 4, 1, 3, 5])

    def test_fills_a_leading_hole_first(self):
        # #793's report: a group left as [_ B] costs a whole row; the next request goes into that hole
        e = engine(6, 3)
        e.slot_busy[3] = True                       # group 1 = [_ B]
        e.slot_used[0] = 0.0                        # least recently used: what the LRU fallback picked
        e.slot_used[2] = 99.0
        self.assertEqual(e.pick_slot([1]), 2)

    def test_never_opens_a_hole_below_the_pick(self):
        for batch, groups in ((6, 3), (8, 4), (8, 2), (4, 2)):
            gs = batch // groups
            e, rng = engine(batch, groups), random.Random(batch * 10 + groups)
            for _ in range(2000):
                busy = [b for b in range(batch) if e.slot_busy[b]]
                if busy and (rng.random() < 0.45 or len(busy) == batch):
                    e.slot_busy[rng.choice(busy)] = False          # a request ends, anywhere
                    continue
                b = e.pick_slot([7])
                g0 = b - b % gs
                with self.subTest(batch=batch, groups=groups):
                    self.assertTrue(all(e.slot_busy[x] for x in range(g0, b)),
                                    f"slot {b} picked with a free slot below it in its group: {e.slot_busy}")
                e.slot_busy[b] = True

    def test_held_tokens_are_ignored(self):
        # Affinity is opt-in: the default still prioritizes pipeline placement over held prefixes.
        e = engine(6, 3)
        e.slot_held[5] = [1, 2]
        self.assertEqual(e.pick_slot([1, 2, 3]), 0)


class CacheAffinity(unittest.TestCase):
    def engine(self):
        e = engine(6, 3)
        e.slot_cache_affinity = True
        e.slot_used = [time.time()] * 6
        return e

    def test_environment_survives_lazy_restart_setup(self):
        e = StrataEngine("missing", [], env={"STRATA_SLOT_CACHE_AFFINITY": "1"}, lazy=True)
        self.assertTrue(e.slot_cache_affinity)
        restarted = StrataEngine(*e.spawn[:5], lazy=True, cpus=e.spawn[6])
        self.assertTrue(restarted.slot_cache_affinity)
        self.assertFalse(StrataEngine("missing", [], env={}, lazy=True).slot_cache_affinity)


class OneGroup(unittest.TestCase):
    def test_reuse_then_least_recently_used(self):
        e = engine(4, 1, pipelined=False)
        e.slot_held[2] = [1, 2]
        self.assertEqual(e.pick_slot([1, 2, 3]), 2)        # its conversation's next turn
        e.slot_held[2] = [9]
        e.slot_held = [[5], [], [9], [6]]
        self.assertEqual(e.pick_slot([1, 2, 3]), 1)        # an empty slot before evicting a held one
        e.slot_held[1] = [8]
        e.slot_used = [5.0, 3.0, 9.0, 1.0]
        self.assertEqual(e.pick_slot([1, 2, 3]), 3)        # else the least recently used


class OneSlotGroups(unittest.TestCase):
    """--batch 2 on a two-GPU layer split: two pipelined groups of ONE slot.  No slot sits below another in its group,
    so no pad row writes an idle slot and each one keeps its chat (#2518): the next turn goes back to it."""

    def test_the_slot_of_the_chat(self):
        T = StrataEngine.TURN_ID
        e = engine(2, 2)
        self.assertEqual(e.slot_gs, 1)
        hist_a, hist_b = list(range(1000, 1700)), list(range(2000, 2700))
        e.slot_held = [hist_a + [T, 7, 8] + [50, 51], hist_b + [T, 7, 8] + [60]]
        e.slot_used = [10.0, 20.0]
        again = lambda h: h + [T, 5, 6] + list(range(3000, 3050))
        self.assertEqual(e.pick_slot(again(hist_b)), 1)
        self.assertEqual(e.pick_slot(again(hist_a)), 0)
        self.assertEqual(e.pick_slot(list(range(5000, 5100))), 0)      # a new chat: the least recently used one

    def test_groups_of_several_slots_still_ignore_held_tokens(self):
        T = StrataEngine.TURN_ID
        e = engine(4, 2)
        self.assertEqual(e.slot_gs, 2)
        hist = list(range(1000, 1700))
        e.slot_held[3] = hist + [T, 7, 8] + [50]
        self.assertEqual(e.pick_slot(hist + [T, 5, 6] + list(range(3000, 3050))), 0)


class RetentionPolicy(unittest.TestCase):
    NOW = 1000.0

    def e(self):
        e = engine(6, 3)
        e.slot_cache_affinity = True
        e.slot_used = [self.NOW] * 6
        return e

    def place(self, e, prompt, **kw):
        return slot_policy.place(prompt, e.slot_busy, e.slot_held, e.slot_used,
                                 e.slot_order, e.slot_gs, self.NOW, **kw)

    def test_padding_invalidity_survives_completion_of_higher_row(self):
        e = self.e()
        e.slot_cv = threading.Condition()
        e.slot_cache_valid = [False] * 6
        e._slot_native_active = [False] * 6
        e._slot_steps = [0] * 6
        e._slot_valid_until = [{} for _ in range(6)]
        e._observe_cache_line('BADM 2 1')
        e._observe_cache_line('BADM 3 1')
        e._observe_cache_line('BT 2 99')
        e._observe_cache_line('BDONE 2 2 stop 1')
        e._observe_cache_line('BT 3 88')  # same window: row 2 was not padding
        self.assertTrue(e.slot_cache_valid[2])
        e._observe_cache_line('BT 3 87')  # next window: row 2 was padding
        self.assertFalse(e.slot_cache_valid[2])
        e._observe_cache_line('BDONE 3 3 stop 1')
        self.assertFalse(e.slot_cache_valid[2])
        self.assertTrue(e.slot_cache_valid[3])
        # A later real request can restore the lower row's validity.
        e._observe_cache_line('BADM 2 1')
        e._observe_cache_line('BDONE 2 2 stop 1')
        self.assertTrue(e.slot_cache_valid[2])

    def test_new_higher_row_invalidates_lower_cache_on_its_first_window(self):
        e = self.e()
        e.slot_cv = threading.Condition()
        e.slot_cache_valid = [False] * 6
        e._slot_native_active = [False] * 6
        e._slot_steps = [0] * 6
        e._slot_valid_until = [{} for _ in range(6)]
        e._observe_cache_line('BADM 2 1')
        e._observe_cache_line('BDONE 2 2 stop 1')
        self.assertTrue(e.slot_cache_valid[2])
        e._observe_cache_line('BADM 3 1')  # row 3 was absent from row 2's final window
        e._observe_cache_line('BT 3 88')
        e._observe_cache_line('BDONE 3 2 stop 1')
        self.assertFalse(e.slot_cache_valid[2])


class ParkMainSession(unittest.TestCase):
    """Engine park=: before a side request replaces the main (solo) session, its conversation is copied into a slot."""
    NOW = 100000.0

    def e(self):
        e = engine(6, 3)
        e.slot_cache_affinity = True
        e.slot_padding_safe = True
        e.info = {"slot_park": 1}
        e.slot_used = [self.NOW] * 6
        e.slot_cv = threading.Condition()
        e.slot_cache_valid = [True] * 6
        e.slot_cache_peer = [None] * 6
        e._park_claims, e.slot_claims = set(), set()
        e.main_held = [7] * 50000
        e.main_peer = "192.0.2.9"
        return e

    def dest(self, e, exclude=()):
        return slot_policy.park_destination(e.main_held, e.slot_busy, e.slot_held, e.slot_used, e.slot_order,
                                            e.slot_gs, self.NOW, e.slot_cache_valid, True, exclude)

    def test_off_without_engine_capability_or_padding_safety(self):
        e = self.e()
        e.info = {}
        self.assertEqual(e._claim_park([1] * 60), "")
        e.info = {"slot_park": 1}
        e.slot_padding_safe = False
        self.assertEqual(e._claim_park([1] * 60), "")


class NativePaddingSafety(unittest.TestCase):

    def test_safe_native_steps_do_not_invalidate_finished_lower_slot(self):
        e = engine(6, 3)
        e.slot_padding_safe = True
        e.slot_cv = threading.Condition()
        e.slot_cache_valid = [False] * 6
        e._slot_native_active = [False] * 6
        e._slot_steps = [0] * 6
        e._slot_valid_until = [{} for _ in range(6)]
        e._observe_cache_line('BADM 2 1')
        e._observe_cache_line('BADM 3 1')
        e._observe_cache_line('BDONE 2 2 stop 1')
        for _ in range(3):
            e._observe_cache_line('BT 3 88')
        e._observe_cache_line('BDONE 3 4 stop 1')
        self.assertTrue(e.slot_cache_valid[2])
        self.assertTrue(e.slot_cache_valid[3])



class CacheObservability(unittest.TestCase):

    def test_busy_view_keeps_actual_reuse_separate_from_hint_and_peer(self):
        e = engine(2, 1)
        e.slot_cache_affinity = True
        e.slot_live = [{"state": "decoding", "prompt_tokens": 1000, "generated": 3,
                        "started": time.time(), "first_token": None, "peer_ip": "2001:db8::7",
                        "cache_reused_tokens": 0, "cache_match_hint": 900}, None]
        row = e.slots_view()[0]
        self.assertEqual((row["cache_reused_tokens"], row["cache_match_hint"]), (0, 900))
        self.assertEqual(row["peer_ip"], "2001:db8::7")

    def test_event_ring_is_bounded_and_returns_snapshot(self):
        e = engine(2, 1)
        for n in range(120):
            e.cache_event("allocation", slot=n % 2)
        events = e.cache_events_view()
        self.assertEqual(len(events), 96)
        events.clear()
        self.assertEqual(len(e.cache_events_view()), 96)

if __name__ == "__main__":
    unittest.main()
