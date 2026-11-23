"""Inkling-2-Mini fit planner: it works from the memory that is FREE (card and RAM), keeps a margin, keeps more beside a screen,
and writes the engine's flags in GiB (#6662 #6644 #6792 #6739).  The numbers come from runs on an RTX 3060 12 GB, an A4000 16 GB and
an RTX 4090 24 GB (RESULTS-0221-INKCUDA.md): the engine's ledger showed 0.4-0.5 GiB of the card left after start-up at the old plans."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

TERMINAL = Path(__file__).resolve().parents[2]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

import setup as S
from strata_control import inkling_fit as F


class FreeMemoryPlan(unittest.TestCase):
    def test_vram_margin_is_taken_off_the_free_memory(self):
        # a partial placement: every GiB of safety is a GB less in the card, the experts' rest goes to RAM
        base = F.compute("SQ-15", 12.0, 64, 32768)
        # a card with 0.3 GiB less free memory (the same plan made on what is really free) holds that much less
        less = F.compute("SQ-15", 12.0 - F.VRAM_SAFETY_GB, 64, 32768)
        self.assertAlmostEqual(base["vram_gb"] - less["vram_gb"], F.VRAM_SAFETY_GB * F.GIB_TO_GB, places=1)
        self.assertGreater(F.VRAM_SAFETY_GB, 0.0)

    def test_a_card_that_drives_a_screen_keeps_a_larger_reserve(self):
        for build, vram in (("SQ-10", 8.0), ("SQ-15", 12.0)):
            plain = F.compute(build, vram, 64, 32768)
            shown = F.compute(build, vram, 64, 32768, desktop=True)
            self.assertAlmostEqual(plain["vram_gb"] - shown["vram_gb"], F.DESKTOP_RESERVE_GB, places=2)
            self.assertGreaterEqual(shown["ram_gb"], plain["ram_gb"])        # what leaves the card goes to RAM, not away
        # the 8 GiB card of #6739: a real desktop reserve instead of 159 MiB
        p8 = F.compute("SQ-10", 8.0, 31, 65536, desktop=True)
        p8_plain = F.compute("SQ-10", 8.0, 31, 65536)
        self.assertLessEqual(p8["vram_gb"], p8_plain["vram_gb"] - 0.75)

    def test_free_ram_caps_the_experts_held_in_ram(self):
        plenty = F.place("SQ-10", 4.0, 64, False, 32768)
        self.assertGreater(plenty["ram_gb"], 8.0)
        tight = F.place("SQ-10", 4.0, 64, False, 32768, ram_free_gib=6.0)
        self.assertLessEqual(tight["ram_gb"], 6.0 * F.GIB_TO_GB - F.FREE_RAM_MARGIN_GB)
        self.assertGreater(tight["ssd_gb"], 0.05)            # what RAM cannot hold streams from disk: said, not overcommitted
        # free RAM above the installed figure's own reserve changes nothing
        same = F.place("SQ-10", 4.0, 64, False, 32768, ram_free_gib=60.0)
        self.assertEqual(same["ram_gb"], plenty["ram_gb"])

    def test_unknown_free_ram_keeps_the_old_rule(self):
        self.assertEqual(F.place("SQ-10", 6.0, 32, False, 32768),
                         F.place("SQ-10", 6.0, 32, False, 32768, ram_free_gib=None))

    def test_flags_are_gib_not_gb(self):
        placed = {"vram_gb": 7.75, "ram_gb": 0.85}
        got = S.inkling_expert_flags(placed)
        self.assertEqual(got[0], "--vram-experts-gb")
        self.assertEqual(got[2], "--ram-experts-gb")
        self.assertAlmostEqual(float(got[1]), 7.75 / 1.073741824, delta=0.011)   # 7.22 GiB, not 7.75 (rounded up to the flag's 0.01)
        self.assertAlmostEqual(float(got[3]), 0.85 / 1.073741824, delta=0.011)
        self.assertGreaterEqual(float(got[1]), 7.75 / 1.073741824)
        self.assertGreaterEqual(float(got[3]), 0.85 / 1.073741824)

    def test_a_flag_never_cuts_an_expert_off_a_plan_that_holds_them_all(self):
        """UD-Q4_K_XL on a 32 GB card: the experts are 15.80 GB (3072 of them, 4.48 MB each in the NVFP4 layers). The cap written for the plan must
        hold every one: the 14.42 GiB of the old table (15.48 GB) left 38 experts on the host and put the engine on the hybrid path (RTX 5090:
        prefill -5.8%, decode -2.6%)."""
        placed = F.place("UD-Q4_K_XL", 31.0, 64, False, 32768, False, dspark=False, backend="cuda", ram_free_gib=56.0)
        self.assertTrue(placed["full_vram"], placed)
        cap_gb = float(S.inkling_expert_flags(placed)[1]) * 1.073741824
        self.assertGreaterEqual(cap_gb, 15.80 + 0.05)                            # the real bytes of the experts, and slack
        self.assertLess(cap_gb, 15.80 * 1.05)
        for gb in (0.0, 1.234, 8.0, 14.7119):                                    # no figure is ever rounded below itself
            p = {"vram_gb": gb, "ram_gb": gb / 3, "full_vram": False}
            f = S.inkling_expert_flags(p)
            self.assertGreaterEqual(float(f[1]) * 1.073741824 + 1e-9, gb)
            if gb / 3 > 0.05:                                                    # a plan with no RAM and no SSD leaves the RAM flag out (#6752)
                self.assertGreaterEqual(float(f[3]) * 1.073741824 + 1e-9, gb / 3)
            else:
                self.assertEqual(f, f[:2])


class MaxContext(unittest.TestCase):
    def test_a_bigger_model_on_a_smaller_card_is_planned_not_refused(self):
        # a MoE: the experts that do not fit the card go to RAM; every build gets a context on every card class
        for free in (3.7, 7.7, 11.63, 15.5):
            for build in ("SQ-10", "SQ-20", "UD-Q4_K_XL", "UD-Q2_K_XL"):
                pl = F.place(build, free, 64, False, 32768, backend="cuda", ram_free_gib=56.0)
                self.assertLessEqual(pl["ssd_gb"], 0.05, (free, build, pl))
                self.assertGreaterEqual(F.max_context(build, free, 64, False, backend="cuda", ram_free_gib=56.0), 32768, (free, build))

    def test_the_context_follows_the_ram_the_kv_goes_to(self):
        big = F.max_context("SQ-10", 7.7, 64, False, backend="cuda", ram_free_gib=56.0)
        small = F.max_context("SQ-10", 7.7, 16, False, backend="cuda", ram_free_gib=10.0)
        self.assertGreater(big, small)
        self.assertGreaterEqual(small, 8192)
        self.assertLessEqual(small, 131072)

    def test_a_kv_that_fits_neither_the_card_nor_ram_is_flagged(self):
        pl = F.place("SQ-10", 3.7, 16, False, 1048576, backend="cuda", ram_free_gib=10.0)
        self.assertFalse(pl.get("kv_fits", True), pl)
        ok = F.place("SQ-10", 3.7, 16, False, 32768, backend="cuda", ram_free_gib=10.0)
        self.assertTrue(ok.get("kv_fits", True), ok)

    def test_desktop_reserve_never_raises_the_context(self):
        for build in ("SQ-10", "SQ-20"):
            self.assertLessEqual(F.max_context(build, 7.7, 64, False, backend="cuda", desktop=True, ram_free_gib=56.0),
                                 F.max_context(build, 7.7, 64, False, backend="cuda", ram_free_gib=56.0))


class SetupReadsFreeMemory(unittest.TestCase):
    def test_ram_free_gb_linux_reads_memavailable(self):
        meminfo = "MemTotal:       65843200 kB\nMemFree:         1000000 kB\nMemAvailable:   33554432 kB\n"
        with mock.patch.object(S, "WIN", False), mock.patch.object(S.sys, "platform", "linux"), \
                mock.patch("builtins.open", mock.mock_open(read_data=meminfo)), \
                mock.patch.object(S, "_cgroup_memory_limit", return_value=None):
            self.assertAlmostEqual(S.ram_free_gb(), 32.0, places=3)

    def test_ram_free_gb_in_a_container_is_limited_by_the_cgroup(self):
        meminfo = "MemTotal:       65843200 kB\nMemAvailable:   33554432 kB\n"
        gib = 2 ** 30
        real_open = open

        def fake_open(name, *a, **k):
            if str(name).endswith("memory.current"):
                return mock.mock_open(read_data=str(14 * gib))()
            if str(name) == "/proc/meminfo":
                return mock.mock_open(read_data=meminfo)()
            return real_open(name, *a, **k)

        with mock.patch.object(S, "WIN", False), mock.patch.object(S.sys, "platform", "linux"), \
                mock.patch("builtins.open", fake_open), mock.patch.object(S, "_cgroup_memory_limit", return_value=16 * gib):
            self.assertAlmostEqual(S.ram_free_gb(), 2.0, places=3)         # a 16 GiB limit with 14 GiB used

    def test_plan_extra_carries_free_ram_and_the_display(self):
        gpu = {"index": 0, "name": "NVIDIA GeForce RTX 3060", "vram_gb": 12.0, "free_gb": 11.0}
        with mock.patch.object(S, "ram_free_gb", return_value=20.0), mock.patch.object(S, "gpu_drives_display", return_value=True):
            self.assertEqual(S.inkling_plan_extra(gpu), {"ram_free_gib": 20.0, "desktop": True})
        with mock.patch.object(S, "ram_free_gb", return_value=None), mock.patch.object(S, "gpu_drives_display", return_value=False):
            self.assertEqual(S.inkling_plan_extra(gpu), {})
        # a unified-memory machine has no card to keep a reserve on
        with mock.patch.object(S, "ram_free_gb", return_value=20.0), mock.patch.object(S, "gpu_drives_display", return_value=True):
            self.assertEqual(S.inkling_plan_extra({"index": 0, "uma": True}), {"ram_free_gib": 20.0})

    def test_planner_input_is_min_of_total_and_free_vram(self):
        gpu = {"index": 0, "name": "NVIDIA GeForce RTX 3060", "vram_gb": 12.0, "free_gb": 3.71, "arch": "86"}
        vram, ram, unified, igpu, arch = S.inkling_pc(gpu, 64.0)
        self.assertAlmostEqual(vram, 3.71, places=2)
        self.assertFalse(unified)


if __name__ == "__main__":
    unittest.main()
