"""strata_control/dsv41_fit.py: the memory plan of the official DeepSeek-V4.1-Flash sets (ISTA-DAS 1.75 / 2.00 / 2.25 bpw), and its
unified-memory branch for a Ryzen AI Max "Strix Halo" (gfx1151) PC, where the graphics memory is the RAM and is counted once.

The numbers of the 64 GB box are the measured ones (Windows, Adrenalin 26.10.1, engine 0.1.51): 2.00 bpw, Faster mode (5 experts),
DSpark auto: answers 14-16 tokens/s on code and 10-11 on chat, a 4K prompt 77-81 tokens/s and 8K 108-111.
"""
from __future__ import annotations

import unittest

from strata_control import dsv41_fit as F

# what setup hands the plan on the owner's Strix Halo: one integrated Radeon, its "VRAM" counting the 64 GB BIOS carve-out plus 32 GB
# shared with the 64 GB Windows sees (128 GB of memory in one pool)
HALO = [{"index": 0, "name": "AMD Radeon 8060S (Ryzen AI Max, Strix Halo, gfx1151)", "vram_gb": 96.0, "arch": "gfx1151",
         "uma": True, "integrated": True, "dedicated_gb": 64.0, "shared_gb": 32.0}]
SMALL = [{"index": 0, "name": "AMD Radeon 8040S (Ryzen AI Max, Strix Halo, gfx1151)", "vram_gb": 32.0, "arch": "gfx1151",
          "uma": True, "integrated": True, "dedicated_gb": 16.0, "shared_gb": 16.0}]
R9700 = [{"index": 0, "name": "Radeon AI PRO R9700", "vram_gb": 32.0, "arch": "gfx1201"}]
TWO_R9700 = [R9700[0], {**R9700[0], "index": 1}]


class UnifiedMemoryPlan(unittest.TestCase):
    def plan(self, model="IST-DAS-2.00bpw", ram=63.6, disk=700.0):
        return F.plan(HALO, ram, disk, 16384, model=model)

    def test_counts_the_whole_pool_once(self):
        p = self.plan()
        self.assertTrue(p["uma"])
        self.assertEqual(p["tp"], 1)                    # no multi-card split words
        self.assertEqual(p["args"], [])                 # and no --tp argument
        resident = 63.6 + 64.0 - F.ISTA_DENSE_GB - F.UMA_RESERVE_GB      # the RAM Windows sees plus the BIOS carve-out
        self.assertAlmostEqual(p["vram_experts_gb"], resident, places=1)
        self.assertAlmostEqual(p["streamed_gb"], 132.07 - resident, places=1)
        self.assertEqual(p["experts_in_vram"], int(resident / (132.07 / F.EXPERTS)))
        self.assertEqual(p["verdict"], "tight")         # it runs, most of it from memory: measured 14-16 tokens/s on code
        text = " ".join(p["lines"])
        self.assertIn("one pool of 128 GB", text)
        self.assertIn("read from the SSD", text)
        self.assertIn("14-16 tokens a second", text)

    def test_it_never_refuses_and_keeps_the_disk_numbers(self):
        p = self.plan("IST-DAS-2.25bpw")
        self.assertTrue(p["disk_ok"])
        self.assertAlmostEqual(p["download_gb"], F.ista_download_bytes("IST-DAS-2.25bpw") / 1e9, places=1)
        self.assertGreater(p["streamed_gb"], 30.0)      # 2.25 needs more from the SSD than 2.00
        self.assertTrue(any("274 GB" in l for l in p["lines"]))   # the set's own download

    def test_a_smaller_unified_pc_is_slower_never_refused(self):
        p = F.plan(SMALL, 16.0, 700.0, 8192, model="IST-DAS-2.00bpw")
        self.assertEqual(p["verdict"], "spills")
        self.assertTrue(any("1.75 bpw" in w for w in p["warnings"]))
        self.assertTrue(p["disk_ok"])

    def test_a_128_gb_unified_pc_nearly_holds_the_smallest_set(self):
        p = F.plan(HALO, 63.6, 700.0, 8192, model="IST-DAS-1.75bpw")
        self.assertGreater(p["vram_experts_gb"], 105.0)
        self.assertLess(p["streamed_gb"], 10.0)
        self.assertEqual(p["verdict"], "tight")         # 1.75 nearly fits the 128 GB Strix Halo: ~5 GB still come from the SSD

    def test_a_large_enough_unified_pc_fits(self):
        p = F.plan(HALO, 160.0, 700.0, 8192, model="IST-DAS-1.75bpw")
        self.assertEqual(p["verdict"], "fits")
        self.assertEqual(p["streamed_gb"], 0.0)

    def test_a_discrete_card_keeps_the_per_card_plan(self):
        p = F.plan(TWO_R9700, 251.0, 700.0, 8192, model="IST-DAS-2.00bpw")
        self.assertFalse(p.get("uma"))
        self.assertEqual(p["tp"], 2)
        self.assertEqual(len(p["cards"]), 2)
        self.assertTrue(any("Two cards share" in l for l in p["lines"]))

    def test_the_smaller_set_is_recommended_for_a_small_unified_pc(self):
        self.assertEqual(F.recommend(32.0), "IST-DAS-1.75bpw")
        self.assertEqual(F.recommend(63.6), "IST-DAS-2.00bpw")


if __name__ == "__main__":
    unittest.main()


class MeasuredCardAccounting(unittest.TestCase):
    """The plan's "experts on the cards" against what the engine's own startup line reported (f-042-maya-rc d93547aed, 2026-11-01): the
    old plan said 2,947 for one R9700 (the engine held 2,366-2,566) and 13,180 for four (it held 12,147-12,192).  Within 3% of the
    engine's counts on the R9700s; the other cards (boston 2 x A6000, an A4000, an RTX 5070) within 8%."""

    @staticmethod
    def cards(n, gb, vendor):
        return [{"index": i, "name": "card", "vram_gb": gb, "vendor": vendor, "arch": "gfx1201"} for i in range(n)]

    def plan(self, n, gb, vendor, model="IST-DAS-2.25bpw", ram=251.0):
        orig = F._on_windows
        F._on_windows = lambda: False
        try:
            return F.plan(self.cards(n, gb, vendor), ram, 500.0, model=model, vendor=vendor)
        finally:
            F._on_windows = orig

    def test_one_r9700_holds_what_the_engine_holds(self):
        got = self.plan(1, 32.0, "amd", ram=126.0)["experts_in_vram"]
        self.assertAlmostEqual(got, 2566, delta=2566 * 0.03)        # rc, the borrowed prompt slots; the 0.1.55 engine held 2,366
        self.assertLess(got, 2947)

    def test_four_r9700_hold_what_the_engine_holds(self):
        p = self.plan(4, 32.0, "amd")
        self.assertAlmostEqual(p["experts_in_vram"], 12192, delta=12192 * 0.03)
        self.assertLess(p["experts_in_vram"], 13180)
        self.assertEqual(len(p["cards"]), 4)

    def test_other_cards(self):
        self.assertAlmostEqual(self.plan(2, 48.0, "nvidia", ram=125.0)["experts_in_vram"], 8804, delta=8804 * 0.08)
        self.assertAlmostEqual(self.plan(1, 16.0, "nvidia", ram=64.0)["experts_in_vram"], 750, delta=750 * 0.08)
        self.assertAlmostEqual(self.plan(1, 12.0, "nvidia", ram=64.0)["experts_in_vram"], 287, delta=287 * 0.08)

    def test_the_cpu_share_follows_the_cards_share(self):
        p = self.plan(1, 32.0, "amd", ram=126.0)
        self.assertAlmostEqual(p["vram_experts_gb"] + p["cpu_experts_gb"], 149.06, delta=0.2)

    def test_the_bigger_dense_weights_of_an_unsloth_set_leave_less_room_for_experts(self):
        a = self.plan(1, 32.0, "amd", model="IST-DAS-2.25bpw", ram=126.0)["vram_experts_gb"]
        b = self.plan(1, 32.0, "amd", model="UD-IQ1_M", ram=126.0)["vram_experts_gb"]      # 7.6 GB of dense weights against 5.9
        self.assertLess(b, a)


class Rdna4Faster(unittest.TestCase):
    """rdna4_faster: the engine's `--experts auto` on RDNA4 cards, said by the screens and the app (ds4_experts_rule.hpp single_rdna4 / multi_rdna4)."""

    @staticmethod
    def cards(n, vendor="amd", arch="gfx1201", gb=32.0):
        return [{"index": i, "name": "card", "vram_gb": gb, "vendor": vendor, "arch": arch} for i in range(n)]

    def test_one_card_yes_two_no_three_and_four_yes_on_linux(self):
        orig = F._on_windows
        F._on_windows = lambda: False
        try:
            for model in ("IST-DAS-2.00bpw", "IST-DAS-2.25bpw", "UD-IQ1_S", "UD-IQ1_M"):
                self.assertTrue(F.rdna4_faster(self.cards(1), model, "amd"), model)
                self.assertFalse(F.rdna4_faster(self.cards(2), model, "amd"), model)     # 37-47% of the experts on two cards
                self.assertTrue(F.rdna4_faster(self.cards(3), model, "amd"), model)
                self.assertTrue(F.rdna4_faster(self.cards(4), model, "amd"), model)
        finally:
            F._on_windows = orig

    def test_not_for_other_cards_or_an_unknown_model(self):
        orig = F._on_windows
        F._on_windows = lambda: False
        try:
            self.assertFalse(F.rdna4_faster(self.cards(4, vendor="nvidia"), "IST-DAS-2.25bpw", "nvidia"))
            self.assertFalse(F.rdna4_faster(self.cards(4, arch="gfx1100"), "IST-DAS-2.25bpw", "amd"))
            self.assertFalse(F.rdna4_faster(self.cards(4), None, "amd"))
            self.assertFalse(F.rdna4_faster([], "IST-DAS-2.25bpw", "amd"))
        finally:
            F._on_windows = orig

    def test_windows_runs_one_amd_card(self):
        orig = F._on_windows
        F._on_windows = lambda: True
        try:
            self.assertTrue(F.rdna4_faster(self.cards(2), "IST-DAS-2.25bpw", "amd"))     # one card of the two: the single-card rule
        finally:
            F._on_windows = orig
