"""`strata launch` for Inkling-2-Mini: "fit to your card" no longer plans more slots than keep their K/V on the card.

Olares (RTX 5090 Laptop 24 GB, UD-Q4_K_XL, 32K context, 3 slots from the VRAM table): the engine keeps 21,760 positions of every slot on the card and
11,008 on the host, and a decode step after 30K tokens reads the host part over PCIe: 86.3 tok/s after a short prompt, 81.7 after 4K, 44.3 after 30K.
Each slot reserves K/V for the whole context, so the number of slots decides whether the K/V fits next to the experts."""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

TERMINAL = Path(__file__).resolve().parents[2]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

from strata_cli.launch import settings as S
from strata_cli.launch.runner import fit_parallel_for
from strata_control import inkling_fit as F
from tests.terminal.launch_support import HW_12, HW_24, HW_32GB_RAM, Home, catalog


def _fit(hw, family, model, context, vram):
    with Home():
        return fit_parallel_for(catalog(hw), S.LaunchSettings(family=family, model=model, context=context), vram)


class ResidentParallel(unittest.TestCase):
    def test_a_24_gb_sm86_card_keeps_two_slots_on_the_card_at_32k_one_at_64k(self):
        # only compute capability 8.6 keeps the GGUF's int8 dense tensors (1.60 GB: INKSPEED, 8e208f1e6 / cd568d5e2); that card holds the K/V of two slots
        free = 24.0 - 0.75
        self.assertEqual(F.resident_parallel("UD-Q4_K_XL", free, 64.0, 32768, cap=4, backend="cuda", desktop=True, sm=86), 2)
        self.assertEqual(F.resident_parallel("UD-Q4_K_XL", free, 64.0, 65536, cap=4, backend="cuda", desktop=True, sm=86), 1)

    def test_a_24_gb_ada_or_blackwell_card_keeps_one_slot_with_the_bf16_dense_share(self):
        # sm 8.9 and 12.0 budget the BF16 dense share (2.45 GB, the int8 path lost at C1 there): 0.85 GB less room for K/V, so the second slot spills at 32K.
        # (the 4090 measurement of 2 slots at 30K, 79.8 tok/s, was made with the int8 trunk; with the BF16 trunk it is open, queued for 0.2.22)
        free = 24.0 - 0.75
        for sm in (89, 120):
            self.assertEqual(F.resident_parallel("UD-Q4_K_XL", free, 64.0, 32768, cap=4, backend="cuda", desktop=True, sm=sm), 1, sm)

    def test_the_slots_are_what_the_plan_keeps_on_the_card(self):
        # the answer is the largest N whose plan says "K/V on the card" with the experts' share of one slot
        for build in ("UD-Q4_K_XL", "SQ-20", "SQ-15", "SQ-10"):
            n = F.resident_parallel(build, 23.25, 64.0, 32768, cap=6, backend="cuda", desktop=True, sm=120)
            one = F.place(build, 23.25, 64.0, False, 32768, False, 1, True, backend="cuda", desktop=True, sm=120)
            at = F.place(build, 23.25, 64.0, False, 32768, False, n, True, backend="cuda", desktop=True, sm=120)
            self.assertEqual(at["kv_offload"], "off", build)
            self.assertGreaterEqual(at["vram_gb"], one["vram_gb"] - 0.05, build)
            if n < 6:
                more = F.place(build, 23.25, 64.0, False, 32768, False, n + 1, True, backend="cuda", desktop=True, sm=120)
                self.assertTrue(more["kv_offload"] != "off" or more["vram_gb"] < one["vram_gb"] - 0.05, build)

    def test_a_bigger_card_or_a_smaller_context_never_gives_fewer_slots(self):
        for build in ("UD-Q4_K_XL", "SQ-20"):
            last = 0
            for free in (11.0, 15.5, 18.5, 22.5, 30.0, 46.0):
                n = F.resident_parallel(build, free, 64.0, 32768, cap=8, backend="cuda", desktop=True, sm=120)
                self.assertGreaterEqual(n, last, (build, free))
                last = n
            last = 99
            for ctx in (8192, 32768, 65536, 131072):
                n = F.resident_parallel(build, 23.25, 64.0, ctx, cap=8, backend="cuda", desktop=True, sm=120)
                self.assertLessEqual(n, last, (build, ctx))
                last = n

    def test_never_below_one(self):
        self.assertEqual(F.resident_parallel("SQ-20", 8.0, 16.0, 131072, cap=4, backend="cuda", desktop=True), 1)
        self.assertEqual(F.resident_parallel("SQ-20", 0.0, 16.0, 32768, cap=4, backend="cuda"), 1)


class LaunchParallel(unittest.TestCase):
    def test_never_more_than_the_table(self):
        for table in (1, 2, 3, 4):
            for build in ("UD-Q4_K_XL", "SQ-10", "SQ-20"):
                self.assertLessEqual(F.launch_parallel(build, 46.0, 64.0, 32768, table, sm=120), table)
                self.assertGreaterEqual(F.launch_parallel(build, 46.0, 64.0, 32768, table, sm=120), 1)

    def test_an_unknown_build_keeps_the_table(self):
        self.assertEqual(F.launch_parallel("no-such-build", 24.0, 64.0, 32768, 3), 3)

    def test_the_24_gb_default_is_never_the_tables_three_for_the_gguf_at_32k(self):
        self.assertEqual(S.fit_parallel(24.0), 3)
        self.assertEqual(F.launch_parallel("UD-Q4_K_XL", 24.0 - 0.75, 64.0, 32768, 3, sm=86), 2)      # int8 dense share
        self.assertEqual(F.launch_parallel("UD-Q4_K_XL", 24.0 - 0.75, 64.0, 32768, 3, sm=120), 1)     # BF16 dense share
        self.assertEqual(F.launch_parallel("UD-Q4_K_XL", 24.0 - 0.75, 64.0, 32768, 3, sm=89), 1)


class InTheLaunch(unittest.TestCase):
    def test_inkling_on_a_24_gb_nvidia_card_gets_the_planned_number(self):
        got = _fit(HW_24, "inkling-2-mini", "UD-Q4_K_XL", 32768, 24.0)
        want = F.launch_parallel("UD-Q4_K_XL", 24.0 - 0.75, 64.0, 32768, S.fit_parallel(24.0), sm=None)
        self.assertEqual(got, want)
        self.assertLess(got, S.fit_parallel(24.0))                        # not the table's 3: that spills the K/V

    def test_a_bigger_context_means_fewer_slots_never_more(self):
        a = _fit(HW_24, "inkling-2-mini", "UD-Q4_K_XL", 32768, 24.0)
        b = _fit(HW_24, "inkling-2-mini", "UD-Q4_K_XL", 131072, 24.0)
        self.assertLessEqual(b, a)
        self.assertGreaterEqual(b, 1)

    def test_a_small_card_keeps_the_table(self):
        self.assertEqual(_fit(HW_12, "inkling-2-mini", "SQ-10", 32768, 12.0), S.fit_parallel(12.0))

    def test_an_unknown_model_name_keeps_the_table(self):
        self.assertEqual(_fit(HW_24, "inkling-2-mini", "my-own-gguf", 32768, 24.0), S.fit_parallel(24.0))
        self.assertEqual(_fit(HW_24, "inkling-2-mini", None, 32768, 24.0), S.fit_parallel(24.0))

    def test_other_families_are_unchanged(self):
        for vram in (12.0, 20.0, 24.0, 32.0):
            self.assertEqual(_fit(HW_24, "qwen", "IQ3_S", 131072, vram), S.fit_parallel(vram))
            self.assertEqual(_fit(HW_24, "hy4-flash", "Hy4-Flash", 32768, vram), S.fit_parallel(vram))

    def test_an_explicit_parallel_is_never_asked_for(self):
        # launch/runner.py: `parallel = self.s.parallel if self.s.parallel else (four_card_parallel or catalog_parallel())`: the fit is only the default
        text = (TERMINAL / "strata_cli" / "launch" / "runner.py").read_text(encoding="utf-8")
        self.assertTrue(re.search(r"parallel = self\.s\.parallel if self\.s\.parallel else \(", text))


if __name__ == "__main__":
    unittest.main()
