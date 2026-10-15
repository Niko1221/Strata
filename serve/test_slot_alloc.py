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

    def test_two_long_histories_stay_in_their_slots_over_alternating_turns(self):
        e = self.engine()
        a, b = list(range(1000)), list(range(2000, 3000))
        e.slot_held[0], e.slot_held[2] = list(a), list(b)
        for turn in range(8):
            for slot, history in ((2, b), (0, a)):
                history.append(4000 + turn)
                self.assertEqual(e.pick_slot(history), slot)
                e.slot_held[slot] = list(history)

    def test_longest_available_prefix_beats_group_spreading(self):
        e = self.engine()
        e.slot_held[0] = list(range(600))
        e.slot_held[5] = list(range(900))
        e.pick_slot(list(range(1000)))
        self.assertEqual(e.slot_last_plan.source_hint, 5)
        e.slot_busy[5] = True
        self.assertEqual(e.pick_slot(list(range(1000))), 0)

    def test_new_helper_uses_empty_group_without_padding_old_histories(self):
        e = self.engine()
        e.slot_held[0], e.slot_held[2] = list(range(600)), list(range(2000, 2600))
        self.assertEqual(e.pick_slot([9999]), 4)
        e.slot_busy[4] = True
        self.assertEqual(e.pick_slot([9999]), 5)

    def test_all_expired_histories_are_logically_empty(self):
        e = self.engine()
        e.slot_held = [[100 + b] for b in range(6)]
        e.slot_used = [8, 7, 6, 5, 1, 3]
        self.assertEqual(e.pick_slot([9999]), 0)

    def test_short_and_equal_histories_do_not_force_affinity(self):
        e = self.engine()
        e.slot_held[5] = [1, 2]
        self.assertEqual(e.pick_slot([1, 2, 3]), 0)
        e.slot_held[5] = list(range(900))
        self.assertEqual(e.pick_slot(list(range(900))), 0)

    def test_full_load_waits_and_empty_slots_still_spread(self):
        e = self.engine()
        got = []
        for _ in range(6):
            slot = e.pick_slot([9999])
            got.append(slot)
            e.slot_busy[slot] = True
        self.assertEqual(got, [0, 2, 4, 1, 3, 5])
        self.assertIsNone(e.pick_slot([9999]))

    def test_cold_request_uses_idle_group_before_filling_busy_group(self):
        e = self.engine()
        e.slot_busy[3] = e.slot_busy[5] = True
        # Every free slot has historical tokens, as in the reported live 2/3/5 placement.
        e.slot_held = [[100 + b] for b in range(6)]
        e.slot_used = [90, 90, 1, 90, 2, 90]
        self.assertEqual(e.pick_slot([9999]), 0)

    def test_padding_threatened_prefix_does_not_override_idle_group(self):
        e = self.engine()
        e.slot_busy[3] = e.slot_busy[5] = True
        e.slot_held = [[100 + b] for b in range(6)]
        e.slot_held[2] = list(range(1000))
        e.slot_used = [0] * 6
        self.assertEqual(e.pick_slot(list(range(1001))), 0)

    def test_valid_match_still_beats_group_load(self):
        e = self.engine()
        e.slot_busy[2] = True
        e.slot_held[3] = list(range(1000))
        # An upper slot's history is not padded by a lower busy slot.
        e.pick_slot(list(range(1001)))
        self.assertEqual(e.slot_last_plan.source_hint, 3)

    def test_cold_requests_spread_when_a_group_has_a_leading_hole(self):
        e = self.engine()
        e.slot_busy[3] = e.slot_busy[5] = True
        self.assertEqual(e.pick_slot([9999]), 0)
        e.slot_busy[0] = True
        # Now all groups have one request: filling a leading hole beats another row.
        self.assertEqual(e.pick_slot([9999]), 2)


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

    def test_ttl_curve_anchors_cap_and_monotonicity(self):
        for n, seconds in ((500, 0.1), (50000, 65), (100000, 90), (262144, 120), (900000, 120)):
            self.assertAlmostEqual(slot_policy.cache_ttl(n), seconds)
        curve = [slot_policy.cache_ttl(n) for n in range(0, 300000, 137)]
        self.assertEqual(curve, sorted(curve))

    def test_expiry_is_virtual_and_expired_history_still_matches(self):
        e = self.e()
        h = list(range(500))
        e.slot_held[2] = h
        e.slot_used[2] -= 10
        plan = self.place(e, h + [9999], allow_eviction=False)
        self.assertEqual(plan.source_hint, 2)
        self.assertEqual(plan.reused_hint, 500)
        self.assertEqual(e.slot_held[2], h)  # no token deletion on expiry

    def test_short_prompt_is_far_cheaper_than_fifty_k(self):
        tiny = slot_policy.cache_state(500, self.NOW, self.NOW)
        medium = slot_policy.cache_state(50000, self.NOW, self.NOW)
        long = slot_policy.cache_state(150000, self.NOW, self.NOW)
        self.assertLess(tiny['retention'], medium['retention'] / 100)
        self.assertGreater(medium['retention'], 0.5)
        self.assertLess(medium['retention'], long['retention'])

    def test_fresh_medium_outweighs_nearly_expired_long_history(self):
        a = slot_policy.cache_state(50000, self.NOW, self.NOW)
        b = slot_policy.cache_state(150000, self.NOW - 97, self.NOW)
        self.assertGreater(a['retention'], b['retention'])

    def test_unexpired_fifty_k_is_selected_before_similarly_fresh_150k(self):
        e = self.e()
        e.slot_held = [[b] * (50000 if b == 2 else 150000) for b in range(6)]
        self.assertEqual(self.place(e, [9999]).slot, 2)
        self.assertEqual(self.place(e, [9999], allow_eviction=False).reason, 'protected')

    def test_expired_short_helper_is_reclaimed_without_touching_long_histories(self):
        e = self.e()
        e.slot_held = [[b] * 100000 for b in range(6)]
        e.slot_held[4] = [4] * 500
        e.slot_used[4] -= 0.11
        self.assertEqual(self.place(e, [9999], allow_eviction=False).slot, 4)

    def test_cache_source_can_be_restored_into_another_pipeline_group(self):
        e = self.e()
        e.slot_busy[0] = True
        e.slot_held[1] = list(range(10000))
        plan = self.place(e, list(range(10001)))
        self.assertEqual(plan.source_hint, 1)
        self.assertEqual(plan.slot, 2)

    def test_close_matches_use_group_load_without_forcing_shorter_native_reuse(self):
        e = self.e()
        e.slot_busy[0] = True
        e.slot_held = [[b] * 50000 for b in range(6)]
        e.slot_held[1] = list(range(10000))
        e.slot_held[2] = list(range(9850))
        plan = self.place(e, list(range(10001)), prefill_rate=3000)
        self.assertEqual(plan.near_tokens, 200)
        self.assertEqual(plan.slot, 2)
        self.assertEqual(plan.reused_hint, 10000)
        e.slot_held[2] = list(range(9000))       # a superseded copy of slot 1: blank, in a less loaded group
        plan = self.place(e, list(range(10001)), prefill_rate=3000)
        self.assertEqual((plan.slot, plan.source_hint), (2, 1))
        e.slot_held[2] = [2] * 50000              # another chat: never overwritten just to balance groups
        self.assertEqual(self.place(e, list(range(10001)), prefill_rate=3000).slot, 1)

    def test_two_idle_conversations_do_not_overwrite_each_other(self):
        # Production 2026-10-14: A (slot 2) and B (slot 0), idle past their TTL between turns.  A was moved into
        # B's expired slot 0 and B back into slot 0 over A, so every turn re-read 60-100K tokens from 0.
        for safe in (False, True):
            e = self.e()
            a, b = [1] * 92000, [2] * 61000
            e.slot_held[2], e.slot_held[0] = list(a), list(b)
            e.slot_used[2] = e.slot_used[0] = self.NOW - 300
            for turn in range(6):
                for slot, h in ((2, a), (0, b)):
                    h += [3] * 500
                    plan = self.place(e, h + [4], padding_safe=safe)
                    self.assertEqual((plan.slot, plan.source_hint), (slot, slot), (safe, turn))
                    e.slot_held[slot] = list(h)
                    e.slot_used[slot] = self.NOW - 300

    def test_match_moves_only_into_a_blank_slot_of_a_less_loaded_group(self):
        e = self.e()
        e.slot_busy[0] = True
        e.slot_held[1] = list(range(10000))
        e.slot_held[2] = [2] * 50000
        e.slot_used[2] = self.NOW - 999            # expired, but still another chat's cache
        plan = self.place(e, list(range(10001)), padding_safe=True)
        self.assertEqual((plan.slot, plan.source_hint), (4, 1))   # blank slot 4, not expired slot 2
        e.slot_held[4], e.slot_held[5] = [4] * 50000, [5] * 50000
        e.slot_held[3] = [3] * 50000
        plan = self.place(e, list(range(10001)), padding_safe=True)
        self.assertEqual(plan.slot, 1)            # no blank slot elsewhere: stay in the busy group

    def test_superseded_copy_is_blank_and_needs_no_wait(self):
        e = self.e()
        e.slot_held = [[b] * 50000 for b in range(6)]   # all fresh: a new chat would have to wait
        e.slot_held[0] = list(range(9000))              # where the conversation was before it moved
        e.slot_held[4] = list(range(10000))             # where it is now
        self.assertEqual(slot_policy.superseded_slots(e.slot_held, e.slot_busy, [True] * 6), {0})
        plan = self.place(e, [9999], allow_eviction=False)
        self.assertEqual((plan.slot, plan.reason), (0, "reclaim"))

    def test_reclaim_overwrites_expired_history_only_after_blank_slots(self):
        e = self.e()
        e.slot_held = [[b] * 50000 for b in range(6)]
        e.slot_held[2] = [2] * 80000
        e.slot_held[3] = [3] * 40000
        e.slot_used[2] = e.slot_used[3] = self.NOW - 999      # both expired
        e.slot_held[5] = []
        self.assertEqual(self.place(e, [9999], padding_safe=True).slot, 5)
        e.slot_held[5] = [5] * 50000
        self.assertEqual(self.place(e, [9999], padding_safe=True).slot, 3)   # shorter expired history first

    def test_protected_history_is_kept_and_expired_ones_go_lowest_score_first(self):
        # Under the TTL rules an expired history is reclaimable and a protected one is not, whatever their lengths;
        # among expired ones the retention formula (freshness 0) decides.
        e = self.e()
        e.slot_busy[1] = e.slot_busy[4] = e.slot_busy[5] = True
        e.slot_held[0], e.slot_used[0] = [0] * 37365, self.NOW - 300   # expired
        e.slot_held[2], e.slot_used[2] = [2] * 1526, self.NOW - 1       # still inside its TTL: protected
        e.slot_held[3], e.slot_used[3] = [3] * 7848, self.NOW - 13      # expired
        self.assertEqual(self.place(e, [9999], padding_safe=True).slot, 3)
        self.assertLess(slot_policy.stale_retention(7848), slot_policy.stale_retention(37365))

    def test_new_chat_takes_a_worthless_slot_in_a_busier_group_before_an_expired_conversation(self):
        # GPU run 2026-10-15 on 0.1.44.1: B (slot 1, 9K, past its 11 s TTL) sat in the idle group; new chat C took
        # slot 1 on group load and the two overwrote each other every turn.  Old 1.5K filler histories were free.
        e = self.e()
        e.slot_busy[2] = e.slot_busy[4] = True
        e.slot_held[0], e.slot_used[0] = [0] * 37365, self.NOW - 300   # parked A, expired
        e.slot_held[1], e.slot_used[1] = [1] * 8970, self.NOW - 20     # B, expired
        e.slot_held[3], e.slot_used[3] = [3] * 1526, self.NOW - 30     # filler leftovers, expired
        e.slot_held[5], e.slot_used[5] = [5] * 1526, self.NOW - 30
        self.assertEqual(self.place(e, [9999], padding_safe=True).slot, 3)
        slot_policy.configure({"STRATA_SLOT_EXPIRED_KEEP_SCORE": "1"})      # keep nothing expired: load decides
        try:
            self.assertEqual(self.place(e, [9999], padding_safe=True).slot, 1)
        finally:
            slot_policy.configure({})

    def test_waits_for_a_side_request_slot_about_to_expire_instead_of_a_conversation(self):
        # GPU run 2026-10-15: a filler arriving 1 s after another filler finished overwrote expired 7.8K chat C,
        # while the finished filler's 1.5K history left its 1.4 s protection a moment later.
        e = self.e()
        e.slot_busy[1] = e.slot_busy[5] = True
        e.slot_held[0], e.slot_used[0] = [0] * 37365, self.NOW - 300   # expired, worthy
        e.slot_held[3], e.slot_used[3] = [3] * 7848, self.NOW - 15      # C: expired, worthy
        e.slot_held[2], e.slot_used[2] = [2] * 1526, self.NOW - 0.5     # filler: protected for under 1 s more
        e.slot_held[4], e.slot_used[4] = [4] * 1526, self.NOW - 0.2
        self.assertEqual(self.place(e, [9999], padding_safe=True, allow_eviction=False).reason, "protected")
        self.assertEqual(self.place(e, [9999], padding_safe=True, allow_eviction=True).slot, 3)   # after the wait
        e.slot_used[2] = self.NOW - 2                                     # the filler slot expired: no wait
        self.assertEqual(self.place(e, [9999], padding_safe=True, allow_eviction=False).slot, 2)

    def test_near_match_budget_respects_one_second_and_4096_cap(self):
        e = self.e()
        e.slot_held[0] = list(range(260000))
        p = list(range(260001))
        self.assertEqual(self.place(e, p, prefill_rate=500).near_tokens, 500)
        self.assertEqual(self.place(e, p, prefill_rate=10000).near_tokens, 4096)

    def test_busy_slots_never_match_or_get_overwritten_after_wait(self):
        e = self.e()
        e.slot_busy = [True] * 6
        e.slot_held[2] = list(range(500))
        self.assertIsNone(self.place(e, list(range(501)), allow_eviction=True).slot)
        with mock.patch('serve.server.time.monotonic', return_value=20):
            self.assertEqual(e._admission_slot(list(range(501)), None), (None, None))

    def test_padding_loss_protects_another_chat_in_a_lower_row(self):
        e = self.e()
        e.slot_held[0] = [77] * 100000
        e.slot_busy[2:] = [True] * 4
        # Slot 1 is empty, but using it would pad over the protected history at 0.
        self.assertEqual(self.place(e, [9999], allow_eviction=False).reason, 'protected')
        self.assertEqual(self.place(e, [9999], allow_eviction=True).slot, 0)

    def test_protection_window_once_then_eviction_and_early_reclaim(self):
        e = self.e()
        e.slot_held = [[b] * 50000 for b in range(6)]
        with mock.patch('serve.server.time.time', return_value=self.NOW), \
             mock.patch('serve.server.time.monotonic', return_value=10):
            slot, since = e._admission_slot([9999], None)
            self.assertIsNone(slot)
            self.assertEqual(since, 10)
        with mock.patch('serve.server.time.time', return_value=self.NOW), \
             mock.patch('serve.server.time.monotonic', return_value=14.9):
            self.assertEqual(e._admission_slot([9999], since), (None, 10))
        with mock.patch('serve.server.time.time', return_value=self.NOW), \
             mock.patch('serve.server.time.monotonic', return_value=15):
            self.assertIsNotNone(e._admission_slot([9999], since)[0])
        e.slot_held[4] = [4] * 500
        e.slot_used[4] = self.NOW - 0.2
        with mock.patch('serve.server.time.time', return_value=self.NOW), \
             mock.patch('serve.server.time.monotonic', return_value=10.2):
            self.assertEqual(e._admission_slot([9999], since)[0], 4)

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

    def test_another_conversation_is_told_by_the_main_history_up_to_its_last_turn(self):
        # main: system (turn 1 at 0), user turn at 20000, reply turn at 29000; a prompt that shares all of it up to
        # the reply's turn is the same conversation, one that diverges earlier is another (here: only the 20K system
        # prompt and tools in common, which is more than half of the 30K held)
        T = 248045
        e = self.e()
        e.info = {"slot_park": "1", "turn_token": str(T)}
        main = [T] + [5] * 19999 + [T] + [6] * 8999 + [T] + [8] * 999
        e.main_held = list(main)
        same = main[:29000] + [T] + [9] * 50 + [T] + [1] * 10      # its reply without the thinking, and a next turn
        self.assertEqual(e._claim_park(same), "")
        self.assertEqual(e._claim_park(main[:29001]), "")           # the reply regenerated
        other = main[:20000] + [T] + [3] * 500                      # another chat of the same client
        self.assertTrue(e._claim_park(other).startswith(" park="))  # another conversation: parked
        e.info = {"slot_park": "1"}                                 # an engine that does not say: any difference counts
        self.assertTrue(e._claim_park(same).startswith(" park="))

    def test_side_request_claims_a_blank_slot_and_parked_line_records_it(self):
        e = self.e()
        e.slot_busy[0] = True
        e.slot_held[2] = [2] * 30000                      # another chat: never a park target
        key = e._claim_park([1] * 60)
        self.assertEqual(key, " park=4")                  # blank slot of the least loaded group
        self.assertTrue(e.slot_busy[4])
        e._on_parked(4, 49999)
        self.assertEqual((len(e.slot_held[4]), e.slot_busy[4], e.slot_cache_peer[4]), (49999, False, "192.0.2.9"))
        self.assertEqual(e.cache_events_view()[0]["kind"], "parked")
        e._claim_park([1] * 60)                             # the engine held two tail tokens more than we know
        e._on_parked(e._park_claims and next(iter(e._park_claims)) or 0, 50002)
        self.assertIn(50000, [len(h) for h in e.slot_held])

    def test_continuing_short_or_already_kept_conversations_are_not_parked(self):
        e = self.e()
        self.assertEqual(e._claim_park(e.main_held + [1, 2, 3]), "")    # the same conversation continues
        e.main_held = [7] * 5000
        self.assertEqual(e._claim_park([1] * 60), "")                    # short: cheap to read again
        e.main_held = [7] * 50000
        e.slot_held[3] = [7] * 48000                                     # a slot holds 96% of it already
        self.assertEqual(e._claim_park([1] * 60), "")

    def test_own_older_copy_is_refreshed_before_using_a_blank_slot(self):
        e = self.e()
        e.slot_held[5] = [7] * 30000                      # this conversation, a few turns back
        self.assertEqual(self.dest(e), 5)

    def test_nowhere_to_park_without_evicting(self):
        e = self.e()
        e.slot_held = [[b] * 40000 for b in range(6)]     # every slot holds another fresh conversation
        self.assertIsNone(self.dest(e))
        self.assertEqual(e._claim_park([1] * 60), "")
        self.assertFalse(any(e.slot_busy))

    def test_admission_slot_is_never_the_park_slot_and_claim_is_released(self):
        e = self.e()
        self.assertNotEqual(self.dest(e, exclude=(0,)), 0)
        key = e._claim_park([1] * 60, exclude=(0,))
        b = int(key.split("=")[1])
        e._release_park(b)
        self.assertFalse(e.slot_busy[b])
        self.assertEqual(e.slot_held[b], [])

    def test_off_without_engine_capability_or_padding_safety(self):
        e = self.e()
        e.info = {}
        self.assertEqual(e._claim_park([1] * 60), "")
        e.info = {"slot_park": 1}
        e.slot_padding_safe = False
        self.assertEqual(e._claim_park([1] * 60), "")


class PolicyTunables(unittest.TestCase):
    def tearDown(self):
        slot_policy.configure({})

    def test_environment_overrides_and_bad_values_keep_defaults(self):
        got = slot_policy.configure({"STRATA_SLOT_TTL_MAX_S": "240", "STRATA_SLOT_PROTECT_WAIT_S": "2",
                                     "STRATA_SLOT_PARK_MIN_TOKENS": "x"})
        self.assertEqual(got, {"ttl_max_s": 240.0, "protect_wait_s": 2.0, "park_min_tokens": 8192,
                               "expired_keep_score": 0.01})
        self.assertAlmostEqual(slot_policy.cache_ttl(100000), 180.0)     # the whole curve doubles
        self.assertAlmostEqual(slot_policy.cache_ttl(10 ** 6), 240.0)
        self.assertEqual(slot_policy.configure({})["ttl_max_s"], 120.0)
        self.assertAlmostEqual(slot_policy.cache_ttl(100000), 90.0)

    def test_engine_reads_them_from_its_environment(self):
        e = StrataEngine("missing", [], env={"STRATA_SLOT_PARK_MIN_TOKENS": "20000"}, lazy=True)
        self.assertEqual(e.slot_policy_settings["park_min_tokens"], 20000)


class NativePaddingSafety(unittest.TestCase):
    def test_padding_safe_engine_keeps_lower_match_available(self):
        e = engine(6, 3)
        e.slot_busy[3] = True
        e.slot_held[2] = list(range(1000))
        p = list(range(1001))
        legacy = slot_policy.place(p, e.slot_busy, e.slot_held, e.slot_used, e.slot_order, 2, 1000)
        safe = slot_policy.place(p, e.slot_busy, e.slot_held, e.slot_used, e.slot_order, 2, 1000,
                                 padding_safe=True)
        self.assertEqual(legacy.reused_hint, 0)
        self.assertEqual(safe.source_hint, 2)
        self.assertEqual(safe.reused_hint, 1000)

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
    def test_expiry_is_reported_once_and_does_not_erase_match(self):
        e = engine(6, 3)
        e.slot_cache_affinity = True
        e.slot_live = [None] * 6
        e.slot_cv = threading.Condition()
        e.slot_cache_valid = [True] * 6
        e.slot_cache_peer = ["192.0.2.8"] * 6
        e.slot_held[0] = [1] * 500
        e.slot_used[0] = time.time() - 1
        for _ in range(3):
            row = e.slots_view()[0]
            self.assertTrue(row["cache_expired"])
            self.assertEqual(row["cache_peer_ip"], "192.0.2.8")
            self.assertEqual(row["held_tokens"], 500)
        self.assertEqual([x["kind"] for x in e.cache_events_view()], ["expired"])
        self.assertIsNotNone(e.pick_slot([1] * 500 + [2]))
        self.assertEqual(e.slot_last_plan.reused_hint, 500)

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
