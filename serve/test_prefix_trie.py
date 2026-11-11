"""Autonomous tests for the prefix trie and the pin budget. No engine, no server.

    python -m unittest serve.test_prefix_trie -v
"""
from __future__ import annotations

import unittest

from serve.pin_registry import AutoPrefix, PinRegistry
from serve.prefix_trie import PrefixTrie


class Trie(unittest.TestCase):
    def fresh(self, **kw):
        kw.setdefault("ttl_s", None)
        kw.setdefault("idle_inserts", None)
        return PrefixTrie(**kw)

    def test_exact_and_partial(self):
        t = self.fresh()
        t.insert([1, 2, 3, 4], "a")
        self.assertEqual(t.longest_match([1, 2, 3, 4]), (4, "a"))
        self.assertEqual(t.longest_match([1, 2, 3, 4, 5]), (4, "a"))
        self.assertEqual(t.longest_match([1, 2]), (2, "a"))
        self.assertEqual(t.longest_match([9]), (0, None))
        self.assertEqual(t.longest_match([]), (0, None))

    def test_branches_diverge(self):
        t = self.fresh()
        t.insert([1, 2, 3, 4], "a")
        t.insert([1, 2, 9, 9], "b")
        self.assertEqual(t.longest_match([1, 2, 3, 4]), (4, "a"))
        self.assertEqual(t.longest_match([1, 2, 3, 7]), (3, "a"))
        self.assertEqual(t.longest_match([1, 2, 9]), (3, "b"))
        n, ref = t.longest_match([1, 2, 8])
        self.assertEqual(n, 2)
        self.assertIn(ref, ("a", "b"))

    def test_shorter_then_longer(self):
        t = self.fresh()
        t.insert([1, 2], "short")
        t.insert([1, 2, 3, 4], "long")
        self.assertEqual(t.longest_match([1, 2, 3, 4]), (4, "long"))
        self.assertEqual(t.longest_match([1, 2]), (2, "short"))

    def test_idle_inserts_drop_untouched_branches(self):
        t = self.fresh(idle_inserts=2)
        t.insert([1, 2, 3], "a")
        t.insert([4, 5, 6], "b")          # a has missed one insert
        t.insert([7], "c")                # a has missed two and is gone; b has missed one
        self.assertEqual(t.longest_match([1, 2, 3]), (0, None))
        self.assertEqual(t.longest_match([4, 5, 6])[0], 3)

    def test_a_match_keeps_a_branch_that_would_have_expired(self):
        t = self.fresh(idle_inserts=3)
        t.insert([1, 2, 3], "a")
        t.insert([4, 5, 6], "b")
        t.insert([7], "c")                                     # a has missed two, still inside the window of three
        self.assertEqual(t.longest_match([1])[0], 1)           # refresh a
        t.insert([8], "d")
        t.insert([9], "e")                                     # b has now missed three inserts, a has not
        self.assertEqual(t.longest_match([4, 5, 6]), (0, None))
        self.assertEqual(t.longest_match([1, 2, 3])[0], 3)

    def test_ttl_drops_a_cold_branch(self):
        now = [0.0]
        t = self.fresh(ttl_s=5, clock=lambda: now[0])
        t.insert([1, 2, 3], "a")
        now[0] = 3
        self.assertEqual(t.longest_match([1, 2]), (2, "a"))   # touch at t=3
        now[0] = 7
        t.insert([8], "b")                                    # a is 4s old, still inside 5s
        self.assertEqual(t.longest_match([1, 2, 3])[0], 3)
        now[0] = 20
        t.insert([9], "c")
        self.assertEqual(t.longest_match([1, 2, 3]), (0, None))

    def test_invalidate_one_ref_keeps_the_other_branch(self):
        t = self.fresh()
        t.insert([1, 2, 3, 4], "a")
        t.insert([1, 2, 9], "b")
        t.invalidate("a")
        self.assertEqual(t.longest_match([1, 2, 3, 4])[0], 2)
        self.assertEqual(t.longest_match([1, 2, 9]), (3, "b"))
        t.invalidate()
        self.assertEqual(len(t), 0)
        self.assertEqual(t.longest_match([1, 2, 9]), (0, None))


class Budget(unittest.TestCase):
    def test_below_minimum_is_not_offered(self):
        reg = PinRegistry(min_tokens=4, max_pins=2)
        self.assertIsNone(reg.propose([1, 2, 3, 4, 5], 3))
        self.assertEqual(len(reg), 0)

    def test_reuse_does_not_take_a_second_slot(self):
        reg = PinRegistry(min_tokens=2, max_pins=1)
        self.assertEqual(reg.propose([1, 2, 3, 4], 3), 3)
        self.assertEqual(reg.propose([1, 2, 3, 4, 9], 3), 3)
        self.assertIsNone(reg.propose([5, 6, 7, 8], 3))
        self.assertEqual(len(reg), 1)

    def test_refuse_when_full(self):
        reg = PinRegistry(min_tokens=2, max_pins=1, policy="refuse")
        self.assertEqual(reg.propose([1, 2, 3], 3), 3)
        self.assertIsNone(reg.propose([4, 5, 6], 3))

    def test_evict_least_used_not_oldest_insert(self):
        reg = PinRegistry(min_tokens=2, max_pins=2, policy="evict")
        reg.commit([1, 1, 1, 1], 4, "a", force=True)
        reg.commit([2, 2, 2, 2], 4, "b", force=True)
        self.assertEqual(reg.propose([2, 2, 2, 2, 9], 4), 4)
        self.assertEqual(reg.propose([2, 2, 2, 2, 8], 4), 4)     # b is the one in use
        self.assertEqual(reg.propose([3, 3, 3, 3, 9], 4), 4)     # a goes, b stays
        self.assertIsNone(reg.get(reg.key_for([1, 1, 1, 1], 4)))
        self.assertIsNotNone(reg.get(reg.key_for([2, 2, 2, 2], 4)))

    def test_release_drops_a_reservation_only(self):
        reg = PinRegistry(min_tokens=2, max_pins=1)
        reg.propose([1, 2, 3, 4], 4)
        reg.release([1, 2, 3, 4], 4)
        self.assertEqual(len(reg), 0)
        reg.propose([1, 2, 3, 4], 4)
        reg.get(reg.key_for([1, 2, 3, 4], 4)).established = True
        reg.release([1, 2, 3, 4], 4)
        self.assertEqual(len(reg), 1)

    def test_key_is_length_plus_prefix_hash(self):
        reg = PinRegistry(min_tokens=2, max_pins=2)
        self.assertNotEqual(reg.key_for([1, 2, 3, 9], 3), reg.key_for([1, 2, 4, 9], 3))
        self.assertNotEqual(reg.key_for([1, 2, 3, 9], 2), reg.key_for([1, 2, 3, 9], 3))
        self.assertEqual(reg.key_for([1, 2, 3, 9], 3), reg.key_for([1, 2, 3, 8], 3))


class Detector(unittest.TestCase):
    def detector(self, **reg):
        return AutoPrefix(PrefixTrie(ttl_s=None, idle_inserts=None),
                          PinRegistry(min_tokens=4, max_pins=2, **reg), enabled=True)

    def test_second_prompt_is_offered_and_a_miss_drops_the_pin(self):
        ap = self.detector()
        first = list(range(100))
        self.assertEqual(ap.propose(first), (None, None))
        ap.remember(first, None)
        ap.confirm_read(first, None, prompt_read=len(first), finish="stop")
        # The pin has to be longer than the re-read slack, or a full read is indistinguishable from overlap.
        second = first[:80] + list(range(1000, 1020))
        n, _ref = ap.propose(second)
        self.assertGreaterEqual(n, 80)
        ap.remember(second, n)
        ap.confirm_read(second, n, prompt_read=len(second), finish="stop")   # the read that creates the pin
        self.assertTrue(ap.registry.get(ap.registry.key_for(second, n)).established)
        third = first[:80] + list(range(2000, 2020))
        n3, _ref = ap.propose(third)
        self.assertEqual(n3, n)
        ap.remember(third, n3)
        ap.confirm_read(third, n3, prompt_read=len(third), finish="stop")     # prefix was read again: pin is gone
        self.assertIsNone(ap.registry.get(ap.registry.key_for(third, n3)))

    def test_cancel_before_the_read_forgets_the_prompt(self):
        ap = self.detector()
        ids = list(range(10))
        ap.remember(ids, None)
        ap.confirm_read(ids, None, prompt_read=2, finish="cancel")
        self.assertEqual(ap.propose(ids), (None, None))

    def test_epoch_change_drops_what_the_engine_no_longer_holds(self):
        ap = self.detector()
        ids = list(range(10))
        ap.sync(("pid", False))
        ap.remember(ids, None)
        ap.sync(("pid", False))
        self.assertEqual(ap.trie.longest_match(ids)[0], 10)
        ap.sync(("other", False))
        self.assertEqual(ap.trie.longest_match(ids), (0, None))

    def test_from_env_is_off_until_asked(self):
        self.assertFalse(AutoPrefix.from_env({}).enabled)
        on = AutoPrefix.from_env({"STRATA_AUTO_PREFIX": "1", "STRATA_AUTO_PREFIX_MIN": "128",
                                   "STRATA_AUTO_PREFIX_PINS": "3"})
        self.assertTrue(on.enabled)
        self.assertEqual(on.registry.min_tokens, 128)
        self.assertEqual(on.registry.max_pins, 3)
        self.assertFalse(AutoPrefix.from_env({"STRATA_AUTO_PREFIX": "0"}).enabled)


if __name__ == "__main__":
    unittest.main()
