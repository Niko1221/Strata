"""serve/test_slot_alloc.py - which batch slot a request gets (StrataEngine.pick_slot), no engine needed.

    python -m unittest serve.test_slot_alloc -v
"""
from __future__ import annotations

import os
import random
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import StrataEngine, slot_expire_s  # noqa: E402


def engine(batch, groups, pipelined=True):
    e = object.__new__(StrataEngine)          # pick_slot reads only the slot tables
    gs = batch // groups
    e.batch = batch
    e.slot_order = [g * gs + t for t in range(gs) for g in range(groups)]
    e.slot_group = [b // gs for b in range(batch)]
    e.slot_groups = groups
    e.slot_busy = [False] * batch
    e.slot_held = [[] for _ in range(batch)]
    e.slot_used = [0.0] * batch
    e.slot_gs = gs if pipelined and groups > 1 else 0
    e.SLOT_EXPIRE_S = 600.0                   # STRATA_SLOT_EXPIRE_S=600, whatever the environment says
    return e


def random_tables(e, rng, now, idle=(0, 1, 30, 599, 600, 601, 3600)):
    n = e.batch
    e.slot_busy = [rng.random() < 0.3 for _ in range(n)]
    e.slot_held = [[9] * rng.choice((0, 1, 50, 50, 5000)) for _ in range(n)]
    e.slot_used = [now - rng.choice(idle) for _ in range(n)]


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
        # the engine keeps no pipelined slot as a cache: a held prefix must not pull a request into a hole-making slot
        e = engine(6, 3)
        e.slot_held[5] = [1, 2]
        self.assertEqual(e.pick_slot([1, 2, 3]), 0)

    def test_the_expiry_does_not_move_the_pick(self):
        # #793's order stands: what the slots hold, when they were used and STRATA_SLOT_EXPIRE_S change nothing
        rng = random.Random(3)
        for _ in range(500):
            e, bare = engine(6, 3), engine(6, 3)
            random_tables(e, rng, time.time())
            bare.slot_busy = list(e.slot_busy)                  # the same busy slots, nothing held
            for expire in (0.0, 60.0, 600.0):
                e.SLOT_EXPIRE_S = expire
                self.assertEqual(e.pick_slot([1, 2, 3]), bare.pick_slot([1, 2, 3]))


class OneGroup(unittest.TestCase):
    def test_reuse_then_an_empty_slot(self):
        e = engine(4, 1, pipelined=False)
        e.slot_held[2] = [1, 2]
        self.assertEqual(e.pick_slot([1, 2, 3]), 2)        # its conversation's next turn
        e.slot_held = [[5], [], [9], [6]]
        e.slot_used = [time.time()] * 4
        self.assertEqual(e.pick_slot([1, 2, 3]), 1)        # an empty slot before evicting a held one

    def test_its_own_history_however_long_and_old(self):
        # a next turn takes its conversation's slot even when that slot is the longest and has expired
        e = engine(3, 1, pipelined=False)
        now = time.time()
        e.slot_held = [[1, 2] * 2000, [], [6] * 10]
        e.slot_used = [now - 3600, now, now - 900]
        self.assertEqual(e.pick_slot([1, 2] * 2000 + [3]), 0)

    def test_the_measured_case(self):
        # the PR's GPU run: a third stream decodes all along; A (40,251 tokens held) finished before B (1,777)
        a, b, side = [1] * 40251, [2] * 1777, [3] * 57
        for expire, side_takes in ((600.0, 1), (0.0, 0)):
            e = engine(3, 1, pipelined=False)
            e.SLOT_EXPIRE_S = expire
            now = time.time()
            e.slot_busy[2] = True
            e.slot_held[:2], e.slot_used[:2] = [a, b], [now - 20, now - 10]
            got = e.pick_slot(side)
            self.assertEqual(got, side_takes)               # B's slot; off (the old order): A's, the least recent
            e.slot_held[got], e.slot_used[got] = side, now
            # A's next turn: its own slot; off, its history is gone and it takes B's slot too
            self.assertEqual(e.pick_slot(a + [4] * 22), 0 if expire else 1)

    def test_the_shortest_history_goes_first(self):
        # a short side request must not take the slot of a long conversation that is still going
        e = engine(4, 1, pipelined=False)
        now = time.time()
        e.slot_held = [[5] * 1000, [6] * 10, [7] * 500, [8] * 2000]
        e.slot_used = [now - 1, now - 50, now - 2, now - 200]
        self.assertEqual(e.pick_slot([1, 2, 3]), 1)        # not the least recent (3): it holds the longest history

    def test_an_expired_slot_goes_first(self):
        e = engine(4, 1, pipelined=False)
        now = time.time()
        e.slot_held = [[5] * 1000, [6] * 10, [7] * 500, [8] * 2000]
        e.slot_used = [now - 1, now - 50, now - 2, now - e.SLOT_EXPIRE_S - 1]
        self.assertEqual(e.pick_slot([1, 2, 3]), 3)        # its client has moved on

    def test_expired_slots_least_recently_used_first(self):
        # among expired slots the length no longer counts
        e = engine(3, 1, pipelined=False)
        now = time.time()
        e.slot_held = [[5] * 10, [6] * 2000, [7] * 5]
        e.slot_used = [now - 700, now - 900, now - 1]
        self.assertEqual(e.pick_slot([1, 2, 3]), 1)

    def test_idle_for_exactly_the_limit_has_expired(self):
        e = engine(2, 1, pipelined=False)
        e.slot_held = [[5] * 1000, [6] * 10]
        e.slot_used = [1000.0 - e.SLOT_EXPIRE_S, 999.0]
        with mock.patch("time.time", return_value=1000.0):
            self.assertEqual(e.pick_slot([1, 2, 3]), 0)
            e.slot_used[0] += 0.001
            self.assertEqual(e.pick_slot([1, 2, 3]), 1)

    def test_a_busy_slot_is_never_taken(self):
        e = engine(4, 1, pipelined=False)
        now = time.time()
        e.slot_held = [[], [6] * 10, [7] * 500, [8] * 2000]
        e.slot_used = [now, now - 900, now - 1, now - 2]
        e.slot_busy[0] = e.slot_busy[1] = True             # the empty and the expired slot are in use
        self.assertEqual(e.pick_slot([1, 2, 3]), 2)        # so the shorter of the two held ones
        e.slot_busy = [True] * 4
        self.assertIsNone(e.pick_slot([1, 2, 3]))

    def test_with_batch_groups_the_least_busy_group_first(self):
        # --batch-groups without pipelining: the groups stay balanced first, then the rule within a group
        e = engine(4, 2, pipelined=False)                  # groups [0 1] [2 3]
        now = time.time()
        e.slot_busy[0] = True
        e.slot_held = [[], [6] * 10, [7] * 4000, [8] * 500]
        e.slot_used = [now, now - 5, now - 1, now - 2]
        self.assertEqual(e.pick_slot([1, 2, 3]), 3)        # group 1 is idle: its shorter history, not slot 1's

    def test_the_rule_on_random_tables(self):
        rng = random.Random(2)
        for _ in range(3000):
            e = engine(rng.randint(1, 8), 1, pipelined=False)
            now = 1e6
            random_tables(e, rng, now)
            with mock.patch("time.time", return_value=now):
                got = e.pick_slot([1, 2, 3])
            free = [b for b in range(e.batch) if not e.slot_busy[b]]
            if not free:
                self.assertIsNone(got)
                continue
            empty = [b for b in free if not e.slot_held[b]]
            expired = [b for b in free if e.slot_held[b] and now - e.slot_used[b] >= e.SLOT_EXPIRE_S]
            if empty:
                self.assertIn(got, empty)
            elif expired:
                self.assertIn(got, expired)
                self.assertEqual(e.slot_used[got], min(e.slot_used[b] for b in expired))
            else:
                shortest = min(len(e.slot_held[b]) for b in free)
                self.assertEqual(len(e.slot_held[got]), shortest)
                self.assertEqual(e.slot_used[got], min(e.slot_used[b] for b in free if len(e.slot_held[b]) == shortest))

    def test_equal_histories_least_recently_used(self):
        e = engine(4, 1, pipelined=False)
        now = time.time()
        e.slot_held = [[5] * 8, [6] * 8, [7] * 8, [8] * 8]
        e.slot_used = [now - 5, now - 3, now - 9, now - 1]
        self.assertEqual(e.pick_slot([1, 2, 3]), 2)        # as before

    def test_an_empty_slot_before_an_idle_one(self):
        e = engine(3, 1, pipelined=False)
        now = time.time()
        e.slot_held = [[5] * 100, [], [7] * 100]
        e.slot_used = [now - e.SLOT_EXPIRE_S - 1, now, now - 1]
        self.assertEqual(e.pick_slot([1, 2, 3]), 1)

    def test_strata_slot_expire_s(self):
        e = engine(3, 1, pipelined=False)
        now = time.time()
        e.slot_held = [[5] * 1000, [6] * 10, [7] * 500]
        e.slot_used = [now - 90, now - 1, now - 2]
        self.assertEqual(e.pick_slot([1, 2, 3]), 1)        # 90 s idle has not expired under 600 s
        e.SLOT_EXPIRE_S = 60.0
        self.assertEqual(e.pick_slot([1, 2, 3]), 0)        # it has under 60 s

    def test_off_is_the_old_order(self):
        e = engine(4, 1, pipelined=False)
        now = time.time()
        e.slot_held = [[5] * 1000, [6] * 10, [], [8] * 2000]
        e.slot_used = [now - 1, now - 50, now, now - 200]
        e.SLOT_EXPIRE_S = 0.0
        self.assertEqual(e.pick_slot([1, 2, 3]), 2)        # an empty slot first, as before
        e.slot_held[2] = [7] * 500
        self.assertEqual(e.pick_slot([1, 2, 3]), 3)        # then the least recently used, whatever it holds

    def test_off_is_the_old_order_on_random_tables(self):
        # off (0, the default): the earlier releases' key, (holds something, last used), gives the same slot every time
        rng = random.Random(1)
        for _ in range(3000):
            e = engine(rng.randint(1, 8), 1, pipelined=False)
            e.SLOT_EXPIRE_S = 0.0
            random_tables(e, rng, time.time())
            free = [b for b in e.slot_order if not e.slot_busy[b]]
            old = min(free, key=lambda b: (bool(e.slot_held[b]), e.slot_used[b])) if free else None
            self.assertEqual(e.pick_slot([1, 2, 3]), old)


class Setting(unittest.TestCase):
    def test_off_unless_set(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("STRATA_SLOT_EXPIRE_S", None)
            self.assertEqual(slot_expire_s(), 0.0)

    def test_strata_slot_expire_s_values(self):
        for value, want in (("600", 600.0), ("300", 300.0), ("90.5", 90.5), ("0", 0.0), ("-5", 0.0),
                            ("junk", 0.0), ("", 0.0)):
            with mock.patch.dict(os.environ, {"STRATA_SLOT_EXPIRE_S": value}):
                self.assertEqual(slot_expire_s(), want, value)

    def test_strata_slot_expire_s_from_the_run_config(self):
        # the config's "env" block reaches the engine's environment (child_env), which the engine reads it from
        from serve.server import child_env
        with mock.patch.dict(os.environ):
            os.environ.pop("STRATA_SLOT_EXPIRE_S", None)
            self.assertEqual(slot_expire_s(child_env({"env": {"STRATA_SLOT_EXPIRE_S": "600"}})), 600.0)
            self.assertEqual(slot_expire_s(child_env({})), 0.0)
        with mock.patch.dict(os.environ, {"STRATA_SLOT_EXPIRE_S": "300"}):
            self.assertEqual(slot_expire_s(child_env({})), 300.0)                                  # the server's, as before
            self.assertEqual(slot_expire_s(child_env({"env": {"STRATA_SLOT_EXPIRE_S": "60"}})), 60.0)  # the config wins
        self.assertEqual(slot_expire_s({"STRATA_SLOT_EXPIRE_S": "junk"}), 0.0)


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


if __name__ == "__main__":
    unittest.main()
