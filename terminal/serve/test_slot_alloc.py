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
        plan = self.place(e, h + [9999])
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

    def test_expired_value_loses_more_each_interval_and_reaches_zero_at_total_idle_limit(self):
        ttl = slot_policy.cache_ttl(50000)
        base = slot_policy.stale_retention(50000)
        self.assertAlmostEqual(base, slot_policy.cache_state(50000, 0, ttl - 1e-8)['retention'])
        for age in (0, ttl / 2, ttl):
            self.assertEqual(slot_policy.stale_retention(50000, age), base)
        values = [slot_policy.stale_retention(50000, ttl + (1800 - ttl) * i / 4) for i in range(5)]
        self.assertAlmostEqual(values[2], base * 0.75)
        losses = [values[i] - values[i + 1] for i in range(4)]
        self.assertTrue(all(losses[i] < losses[i + 1] for i in range(3)))
        self.assertEqual(values[-1], 0)
        self.assertEqual(slot_policy.cache_ttl(50000), ttl)
        self.assertEqual(slot_policy.stale_retention(50000, 1e12), 0)
        self.assertEqual(slot_policy.stale_retention(0, 1e12), 0)

    def test_approved_lengths_cross_the_keep_threshold_before_thirty_minutes(self):
        for n, crossing in ((8000, 1585), (16000, 1697), (32000, 1745), (50000, 1762),
                            (100000, 1778), (200000, 1787), (262144, 1789)):
            with self.subTest(tokens=n):
                self.assertGreater(slot_policy.stale_retention(n, crossing - 1), 0.01)
                self.assertLess(slot_policy.stale_retention(n, crossing + 1), 0.01)
                self.assertEqual(slot_policy.stale_retention(n, 1800), 0)
                self.assertEqual(slot_policy.stale_retention(n, 1801), 0)
        for n in (500, 2000):
            self.assertLess(slot_policy.stale_retention(n), 0.01)

    def test_old_long_history_yields_before_a_recent_shorter_one_even_in_busier_group(self):
        e = self.e()
        e.slot_held = [[b] * 50000 for b in range(6)]
        e.slot_busy[1] = True
        e.slot_held[0], e.slot_used[0] = [0] * 150000, self.NOW - 1800
        e.slot_held[4], e.slot_used[4] = [4] * 8000, self.NOW - 15
        self.assertEqual(self.place(e, [9999], padding_safe=True).slot, 0)
        # A use refreshes the age: the long history is protected again and the expired shorter one yields.
        e.slot_used[0] = self.NOW
        self.assertEqual(self.place(e, [9999], padding_safe=True).slot, 4)

    def test_long_unused_history_still_matches_after_its_value_falls_below_keep_score(self):
        e = self.e()
        e.slot_held = [[b] * 50000 for b in range(6)]
        e.slot_held[2] = list(range(100000))
        e.slot_used[2] = self.NOW - 3600
        self.assertEqual(slot_policy.stale_retention(100000, self.NOW - e.slot_used[2]), 0)
        plan = self.place(e, list(range(100001)), padding_safe=True)
        self.assertEqual((plan.slot, plan.source_hint, plan.reused_hint), (2, 2, 100000))
        self.assertEqual(len(e.slot_held[2]), 100000)

    def test_decayed_history_is_reclaimed_while_a_side_request_is_protected(self):
        e = self.e()
        e.slot_held = [[b] * 50000 for b in range(6)]
        e.slot_used[0] = self.NOW - 1800
        e.slot_held[2], e.slot_used[2] = [2] * 1500, self.NOW - 0.5
        plan = self.place(e, [9999], padding_safe=True)
        self.assertEqual((plan.slot, plan.reason), (0, 'reclaim'))

    def test_unexpired_fifty_k_is_selected_before_similarly_fresh_150k(self):
        e = self.e()
        e.slot_held = [[b] * (50000 if b == 2 else 150000) for b in range(6)]
        self.assertEqual(self.place(e, [9999]).slot, 2)

    def test_expired_short_helper_is_reclaimed_without_touching_long_histories(self):
        e = self.e()
        e.slot_held = [[b] * 100000 for b in range(6)]
        e.slot_held[4] = [4] * 500
        e.slot_used[4] -= 0.11
        self.assertEqual(self.place(e, [9999]).slot, 4)

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

    def test_an_older_version_of_the_prompt_is_replaced_in_place_even_while_protected(self):
        # 2026-10-16: each Claude Code classifier query inserts its new tool calls before the closing text of the
        # last one's transcript, so it is no slot's continuation; every version took a blank slot of its own
        e = self.e()
        transcript = [1] * 42000
        e.slot_held[3] = transcript + [8] * 60 + [9] * 40        # the last query, protected (just used)
        e.slot_held[0] = [2] * 30000                              # another chat
        prompt = transcript + [8] * 60 + [7] * 300 + [9] * 40
        plan = self.place(e, prompt, padding_safe=True)
        self.assertEqual((plan.slot, plan.reason), (3, "sibling"))
        other = [1] * 20000 + [5] * 22000                         # shares under 98% of slot 3: a blank slot
        self.assertEqual(self.place(e, other, padding_safe=True).reason, "reclaim")
        self.assertEqual(self.place(e, prompt + [4], padding_safe=True).slot, 3)
        e.slot_held[5] = transcript[:40000]                       # a full-prefix source still decides first
        self.assertEqual(self.place(e, prompt, padding_safe=True).reason, "reuse")
        e.slot_held[5] = []
        e.slot_busy[3] = True                                     # a busy slot is never replaced
        self.assertEqual(self.place(e, prompt, padding_safe=True).reason, "reclaim")
        e.slot_busy[3] = False
        slot_policy.configure({"STRATA_SLOT_SIBLING_SHARE": "0"})
        try:
            self.assertEqual(self.place(e, prompt, padding_safe=True).reason, "reclaim")   # 0: off
        finally:
            slot_policy.configure({})

    def test_reported_checkpoints_give_the_hint_where_no_slot_continues_the_prompt(self):
        e = self.e()
        transcript = [1] * 42000
        e.slot_held[3] = transcript + [8] * 60 + [9] * 40
        e.slot_held[1] = [1] * 20000 + [6] * 5000                 # another chat sharing 20000 tokens
        checks = [()] * 6
        checks[3] = (32768, 42000)                                # periodic + the branch point of the last query
        checks[1] = (16384, 24990)
        prompt = transcript + [8] * 60 + [7] * 300 + [9] * 40
        plan = self.place(e, prompt, padding_safe=True, checks=checks)
        self.assertEqual((plan.slot, plan.reason, plan.source_hint, plan.reused_hint), (3, "sibling", 3, 42000))
        plan = self.place(e, [1] * 30000 + [2] * 10, padding_safe=True, checks=checks)
        self.assertEqual((plan.reason, plan.source_hint, plan.reused_hint), ("reclaim", 1, 16384))   # 24990: not a start
        checks[3] = (32768, 42000, 99999)                         # a length past what the slot holds is ignored
        self.assertEqual(self.place(e, prompt, padding_safe=True, checks=checks).reused_hint, 42000)
        e.slot_busy[3] = True                                     # a busy slot is no source
        plan = self.place(e, prompt, padding_safe=True, checks=checks)
        self.assertEqual((plan.source_hint, plan.reused_hint), (1, 16384))
        self.assertEqual(self.place(e, prompt, padding_safe=True).reused_hint, 0)   # nothing reported: no hint

    def test_common_prefix(self):
        for a, b, n in (([], [1], 0), ([1, 2, 3], [1, 2, 3], 3), ([1, 2, 3], [1, 2], 2), ([1, 2, 3], [1, 5, 3], 1),
                        (list(range(100000)), list(range(77777)) + [0], 77777)):
            self.assertEqual(slot_policy.common_prefix(a, b), n)
            self.assertEqual(slot_policy.common_prefix(b, a), n)

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

    def test_superseded_copy_is_blank_and_reclaimed_before_protected_histories(self):
        e = self.e()
        e.slot_held = [[b] * 50000 for b in range(6)]   # all fresh: every other one is protected
        e.slot_held[0] = list(range(9000))              # where the conversation was before it moved
        e.slot_held[4] = list(range(10000))             # where it is now
        self.assertEqual(slot_policy.superseded_slots(e.slot_held, e.slot_busy, [True] * 6), {0})
        plan = self.place(e, [9999])
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
        # among expired ones the retention at expiry, decayed by time past expiry, decides.
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
        e.slot_held[0], e.slot_used[0] = [0] * 37365, self.NOW - 100   # parked A, expired
        e.slot_held[1], e.slot_used[1] = [1] * 8970, self.NOW - 20     # B, expired
        e.slot_held[3], e.slot_used[3] = [3] * 1526, self.NOW - 30     # filler leftovers, expired
        e.slot_held[5], e.slot_used[5] = [5] * 1526, self.NOW - 30
        self.assertEqual(self.place(e, [9999], padding_safe=True).slot, 3)
        slot_policy.configure({"STRATA_SLOT_EXPIRED_KEEP_SCORE": "1"})      # keep nothing expired: load decides
        try:
            self.assertEqual(self.place(e, [9999], padding_safe=True).slot, 1)
        finally:
            slot_policy.configure({})

    def test_no_wait_for_a_side_request_slot_about_to_expire(self):
        # GPU run 2026-10-15: a filler arriving 1 s after another filler finished overwrote expired 7.8K chat C,
        # while the finished filler's 1.5K history left its 1.4 s protection a moment later.  A wait for it was
        # removed (2026-10-16): the protection rules decide, so C goes while the filler slot is protected.
        e = self.e()
        e.slot_busy[1] = e.slot_busy[5] = True
        e.slot_held[0], e.slot_used[0] = [0] * 37365, self.NOW - 300   # expired, worthy
        e.slot_held[3], e.slot_used[3] = [3] * 7848, self.NOW - 15      # C: expired, worthy
        e.slot_held[2], e.slot_used[2] = [2] * 1526, self.NOW - 0.5     # filler: protected for under 1 s more
        e.slot_held[4], e.slot_used[4] = [4] * 1526, self.NOW - 0.2
        self.assertEqual(self.place(e, [9999], padding_safe=True).slot, 3)
        e.slot_used[2] = self.NOW - 2                                     # the filler slot expired
        self.assertEqual(self.place(e, [9999], padding_safe=True).slot, 2)

    def test_near_match_budget_respects_one_second_and_4096_cap(self):
        e = self.e()
        e.slot_held[0] = list(range(260000))
        p = list(range(260001))
        self.assertEqual(self.place(e, p, prefill_rate=500).near_tokens, 500)
        self.assertEqual(self.place(e, p, prefill_rate=10000).near_tokens, 4096)

    def test_busy_slots_never_match_or_get_overwritten(self):
        e = self.e()
        e.slot_busy = [True] * 6
        e.slot_held[2] = list(range(500))
        self.assertIsNone(self.place(e, list(range(501))).slot)

    def test_padding_loss_counts_a_protected_history_in_a_lower_row(self):
        e = self.e()
        e.slot_held[0] = [77] * 100000
        e.slot_busy[2:] = [True] * 4
        # Slot 1 is empty, but using it would pad over the protected history at 0: the same loss as taking slot 0.
        self.assertEqual(self.place(e, [9999]).slot, 0)

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

    def setUp(self):
        clock = mock.patch('serve.server.time.time', return_value=self.NOW)
        clock.start()
        self.addCleanup(clock.stop)

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
        e.main_used = self.NOW
        return e

    def dest(self, e, exclude=(), main_value=0.0):
        return slot_policy.park_destination(e.main_held, e.slot_busy, e.slot_held, e.slot_used, e.slot_order,
                                            e.slot_gs, self.NOW, e.slot_cache_valid, True, exclude, main_value)

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

    def test_continuing_low_value_or_already_kept_conversations_are_not_parked(self):
        e = self.e()
        self.assertEqual(e._claim_park(e.main_held + [1, 2, 3]), "")    # the same conversation continues
        e.main_held = [7] * 2000
        self.assertEqual(e._claim_park([1] * 60), "")                    # worth less than PARK_MIN_VALUE (no length cut)
        self.assertEqual(e.cache_events_view()[0]["reason"], "main_low_value")
        e.main_held = [7] * 5000                                         # short, but worth more: parked
        self.assertTrue(e._claim_park([1] * 60).startswith(" park="))
        e = self.e()
        e.main_held = [7] * 50000
        e.slot_held[3] = [7] * 49000                                     # a slot holds 98% of it already
        self.assertEqual(e._claim_park([1] * 60), "")
        e.slot_held[3] = [7] * 48000                                     # 96%: the copy is cheaper than the 2K tail
        self.assertEqual(e._claim_park([1] * 60), " park=3")             # into its own older copy

    def test_parking_inherits_the_main_history_last_use_and_does_not_renew_its_value(self):
        e = self.e()
        e.main_used = self.NOW - 1700
        value = slot_policy.stale_retention(50000, 1700)
        key = e._claim_park([1] * 60)
        b = int(key.split('=')[1])
        # The main session may begin a new request before PARKED is processed; the snapshot owns this age.
        e.main_used = self.NOW
        e._on_parked(b, 49999)
        self.assertEqual(e.slot_used[b], self.NOW - 1700)
        self.assertAlmostEqual(slot_policy.stale_retention(len(e.slot_held[b]), self.NOW - e.slot_used[b]), value,
                               delta=0.00001)
        e.slot_live = [None] * 6
        with mock.patch('serve.server.time.time', return_value=self.NOW + 30):
            e.slots_view()
        self.assertEqual(e.slot_used[b], self.NOW - 1700)   # observation cannot renew the cache either

    def test_main_below_keep_threshold_is_not_parked_even_when_slots_are_blank(self):
        e = self.e()
        for age in (1770, 1800, 3600):
            e.main_used = self.NOW - age
            self.assertEqual(e._claim_park([1] * 60), '')
            self.assertFalse(any(e.slot_busy))
            self.assertFalse(e._park_claims)
            self.assertEqual(e.main_held, [7] * 50000)   # a skipped park does not itself erase the main cache
            self.assertEqual(e.cache_events_view()[0]['reason'], 'main_low_value')

    def test_main_above_keep_threshold_is_parked_with_its_old_age(self):
        e = self.e()
        e.main_used = self.NOW - 1760
        key = e._claim_park([1] * 60)
        self.assertTrue(key.startswith(' park='))
        b = int(key.split('=')[1])
        e._on_parked(b, 49999)
        self.assertEqual(e.slot_used[b], self.NOW - 1760)

    def test_park_min_value_is_its_own_threshold(self):
        # a 50K conversation idle for 25 minutes is worth about 0.07, for 20 minutes about 0.13
        slot_policy.configure({"STRATA_SLOT_PARK_MIN_VALUE": "0.1"})
        try:
            self.assertEqual(slot_policy.EXPIRED_KEEP_SCORE, 0.01)      # slot reclaim keeps its own threshold
            e = self.e()
            e.main_used = self.NOW - 1500
            self.assertLess(slot_policy.stale_retention(50000, 1500), 0.1)
            self.assertGreater(slot_policy.stale_retention(50000, 1500), slot_policy.EXPIRED_KEEP_SCORE)
            self.assertEqual(e._claim_park([1] * 60), '')
            self.assertEqual(e.cache_events_view()[0]['reason'], 'main_low_value')
            self.assertFalse(any(e.slot_busy))
            e = self.e()
            e.main_used = self.NOW - 1200
            self.assertGreater(slot_policy.stale_retention(50000, 1200), 0.1)
            self.assertTrue(e._claim_park([1] * 60).startswith(' park='))
        finally:
            slot_policy.configure({})

    def test_zero_value_main_can_still_continue_without_a_park_or_deletion(self):
        e = self.e()
        e.main_used = self.NOW - 3600
        self.assertEqual(slot_policy.stale_retention(len(e.main_held), 3600), 0)
        self.assertEqual(e._claim_park(e.main_held + [1, 2, 3]), '')
        self.assertEqual(e.main_held, [7] * 50000)
        self.assertFalse(e._park_claims)
        self.assertEqual(e.cache_events_view(), [])   # no low-value rejection of a continued conversation

    def test_an_invalid_copy_does_not_count_as_already_kept(self):
        e = self.e()
        e.slot_held[3] = [7] * 49000                      # 98% of the main conversation, but no longer usable
        e.slot_cache_valid[3] = False
        self.assertIsNotNone(self.dest(e))
        e.slot_cache_valid[3] = True
        self.assertIsNone(self.dest(e))                   # a usable copy: nothing to park

    def test_own_older_copy_is_refreshed_before_using_a_blank_slot(self):
        e = self.e()
        e.slot_held[5] = [7] * 30000                      # this conversation, a few turns back
        self.assertEqual(self.dest(e), 5)

    def test_an_older_version_is_replaced_by_the_park_protected_or_not(self):
        # the main session ran a classifier query; slot 2 holds the previous one (the same transcript with another
        # end), worth about as much as the main one, so the value ratio would never let the park take it
        e = self.e()
        e.slot_held = [[b] * 40000 for b in range(6)]                 # every slot holds another fresh conversation
        e.slot_held[2] = e.main_held[:49900] + [3] * 80
        self.assertEqual(self.dest(e), 2)
        e.slot_held[4] = e.main_held[:49950] + [3] * 30               # the version sharing more goes first
        self.assertEqual(self.dest(e), 4)
        self.assertEqual(self.dest(e, exclude=(4,)), 2)
        e.slot_held[1] = e.main_held[:30000]                          # its own older copy is still preferred
        self.assertEqual(self.dest(e), 1)
        e.slot_held[1] = [1] * 40000
        e.slot_held[2] = e.main_held[:40000] + [3] * 9000             # 82%: another conversation
        e.slot_held[4] = e.main_held[:40000] + [4] * 9000
        self.assertIsNone(self.dest(e))
        e.slot_held[2] = e.main_held[:49900] + [3] * 80
        slot_policy.configure({"STRATA_SLOT_SIBLING_SHARE": "0"})
        try:
            self.assertIsNone(self.dest(e))                           # 0: off
        finally:
            slot_policy.configure({})

    def test_engine_checkpoint_lines_are_kept_per_slot_and_forgotten_before_an_overwrite(self):
        e = self.e()
        e.slot_checks = [()] * 6
        e._observe_checks("BCHECKS 2 42729 32768 42000\n")
        e._observe_checks("BCHECKS 4 9000 9000 12000 512\n")      # only lengths inside what the slot holds
        e._observe_checks("BCHECKS x 1 2\n")                        # malformed: ignored
        e._observe_checks("BCHECKS 9 100 50\n")                     # no such slot
        self.assertEqual((e.slot_checks[2], e.slot_checks[4]), ((32768, 42000), (512,)))
        e.slot_held = [[b] * 40000 for b in range(6)]
        e.slot_held[2] = e.main_held[:49900] + [3] * 80               # an older version: the park takes it
        self.assertEqual(e._claim_park([1] * 60), " park=2")
        self.assertEqual(e.slot_checks[2], ())                      # the engine overwrites it now
        self.assertEqual(e.slot_checks[4], (512,))
        e.slot_busy[2] = False
        e._observe_checks("BCHECKS 2 50000 32768 49900\n")
        e.slot_held[2] = list(e.main_held)
        e.slot_live = [None] * 6
        view = {r["slot"]: r for r in e.slots_view()}
        self.assertEqual(view[2]["checkpoint_tokens"], [32768, 49900])

    def test_nowhere_to_park_without_evicting(self):
        e = self.e()
        e.slot_held = [[b] * 40000 for b in range(6)]     # every slot holds another fresh conversation
        self.assertIsNone(self.dest(e))
        self.assertEqual(e._claim_park([1] * 60), "")
        self.assertFalse(any(e.slot_busy))

    def test_an_expired_history_worth_less_than_the_keep_score_takes_the_park(self):
        e = self.e()
        e.slot_held = [[b] * 40000 for b in range(6)]     # every slot holds another conversation
        e.slot_used = [self.NOW - 100] * 6                 # all expired, still worth keeping
        self.assertIsNone(self.dest(e))                    # each still worth keeping: nowhere to park
        e.slot_held[1] = [1] * 1000                        # a short side chat, below EXPIRED_KEEP_SCORE
        e.slot_held[4] = [4] * 2000
        e.slot_busy[0] = True
        self.assertEqual(self.dest(e), 4)                  # the less loaded group (slot 1 shares one with busy 0)
        e.slot_busy[0] = False
        self.assertEqual(self.dest(e), 1)                  # same load: the lowest score
        e.slot_used[1] = self.NOW                          # protected: not a park target
        self.assertEqual(self.dest(e), 4)

    def test_park_uses_the_reclaim_order_a_lighter_group_before_a_blank_slot(self):
        e = self.e()
        e.slot_held = [[b] * 40000 for b in range(6)]
        e.slot_used = [self.NOW - 100] * 6                 # expired, worth keeping
        e.slot_held[1] = []                                # blank, in the group of busy slot 0
        e.slot_busy[0] = True
        e.slot_held[4] = [4] * 1000                        # not worth keeping, in an idle group
        self.assertEqual(self.dest(e), 4)                  # group load first, as place() does
        e.slot_busy[0] = False
        self.assertEqual(self.dest(e), 1)                  # the same load: the lower value (a blank slot is 0)

    def test_a_history_worth_keeping_takes_a_park_only_at_the_value_ratio(self):
        e = self.e()
        e.slot_held = [[b] * 40000 for b in range(6)]
        e.slot_used = [self.NOW - 100] * 6                 # all expired, worth keeping
        worth = slot_policy.stale_retention(40000, 100)
        self.assertIsNone(self.dest(e, main_value=worth * 1.9))      # less than twice as much: every one is kept
        self.assertEqual(self.dest(e, main_value=worth * 2), 0)      # twice as much: the slot order breaks the tie
        e.slot_busy[0] = True
        e.slot_used[3] = self.NOW - 1000                   # worth less than the others
        self.assertEqual(self.dest(e, main_value=worth * 2), 3)      # least loaded group, then the lowest value
        cheap = slot_policy.stale_retention(40000, 1000)
        self.assertEqual(self.dest(e, main_value=cheap * 2), 3)      # only slot 3 is cheap enough
        self.assertIsNone(self.dest(e, main_value=cheap * 1.5))
        slot_policy.configure({"STRATA_SLOT_PARK_VALUE_RATIO": "0"})
        try:
            self.assertIsNone(self.dest(e, main_value=100.0))         # 0: a park never takes one
        finally:
            slot_policy.configure({})

    def test_a_long_unused_history_can_take_a_park_but_busy_and_excluded_slots_cannot(self):
        e = self.e()
        e.slot_held = [[b] * 40000 for b in range(6)]
        e.slot_used = [self.NOW - 100] * 6
        self.assertIsNone(self.dest(e))
        e.slot_used[4] = self.NOW - 1800
        self.assertEqual(self.dest(e), 4)
        self.assertIsNone(self.dest(e, exclude=(4,)))
        e.slot_busy[4] = True
        self.assertIsNone(self.dest(e))
        e.slot_busy[4] = False
        e.slot_used[4] = self.NOW
        self.assertIsNone(self.dest(e))

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
        got = slot_policy.configure({"STRATA_SLOT_TTL_MAX_S": "240", "STRATA_SLOT_PARK_MIN_VALUE": "x"})
        self.assertEqual(got, {"ttl_max_s": 240.0,
                               "expired_keep_score": 0.01, "idle_value_zero_s": 1800.0, "park_value_ratio": 2.0,
                               "park_min_value": 0.01, "park_kept_share": 0.98,
                               "sibling_share": 0.98})
        self.assertAlmostEqual(slot_policy.cache_ttl(100000), 180.0)     # the whole curve doubles
        self.assertAlmostEqual(slot_policy.cache_ttl(10 ** 6), 240.0)
        self.assertEqual(slot_policy.configure({})["ttl_max_s"], 120.0)
        self.assertAlmostEqual(slot_policy.cache_ttl(100000), 90.0)

    def test_engine_reads_them_from_its_environment(self):
        e = StrataEngine("missing", [], env={"STRATA_SLOT_PARK_MIN_VALUE": "0.2"}, lazy=True)
        self.assertEqual(e.slot_policy_settings["park_min_value"], 0.2)

    def test_idle_limit_override_disable_and_invalid_values(self):
        ttl = slot_policy.cache_ttl(50000)
        base = slot_policy.stale_retention(50000)
        slot_policy.configure({"STRATA_SLOT_IDLE_VALUE_ZERO_S": "3600"})
        self.assertEqual(slot_policy.stale_retention(50000, 3600), 0)
        self.assertAlmostEqual(slot_policy.stale_retention(50000, (ttl + 3600) / 2), base * 0.75)
        slot_policy.configure({"STRATA_SLOT_IDLE_VALUE_ZERO_S": "0"})
        self.assertEqual(slot_policy.stale_retention(50000, ttl + 3600), base)
        for bad in ("nan", "inf", "-1", "bad"):
            got = slot_policy.configure({"STRATA_SLOT_IDLE_VALUE_ZERO_S": bad})
            self.assertEqual(got["idle_value_zero_s"], 1800)

    def test_a_configured_idle_limit_cannot_shorten_protection(self):
        slot_policy.configure({"STRATA_SLOT_IDLE_VALUE_ZERO_S": "30"})
        base = slot_policy.stale_retention(50000)
        self.assertEqual(slot_policy.stale_retention(50000, 64), base)
        self.assertEqual(slot_policy.stale_retention(50000, 65), 0)


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
