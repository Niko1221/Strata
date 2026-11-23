"""Inkling-2-Mini fit planner: the dense share of a GGUF build depends on the card class (c60259fa6).

On ONE CUDA card of compute capability 8.0 or newer with no layer split the loader keeps the file's Q8_0 matrices (and the embedding) as they are;
everywhere else it expands them to BF16 (a larger trunk). The two shares were calibrated on what strata-inkling placed with no cap on the card
(RTX 3060 Ti 7.66 GiB free, RTX 3060 11.63 GiB free, UD-IQ4_XS and UD-Q4_K_XL alike); one NVFP4 expert takes 4.479 MB on the card."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

TERMINAL = Path(__file__).resolve().parents[2]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

import setup as S
from strata_control import inkling_fit as F

EXPERT_GB = 4.479e-3
# experts placed on the card with the planner's cap lifted (drafter off, ctx 32768, free RAM 28 GiB, a card that drives a screen)
PLACED = {  # (free GiB, sm path): experts
    (7.66, "q8"): (703, 721), (11.63, "q8"): (1663, 1681),
    (7.66, "bf16"): (489, 489), (11.63, "bf16"): (1449, 1449),
}


def _plan(build: str, free: float, **kw) -> dict:
    return F.place(build, free, 32.0, False, 32768, False, dspark=False, backend="cuda", desktop=True, ram_free_gib=28.0, **kw)


class DenseShare(unittest.TestCase):
    def test_the_q8_path_plan_equals_what_the_engine_places(self):
        for (free, path), counts in PLACED.items():
            if path != "q8":
                continue
            engine = sum(counts) / len(counts) * EXPERT_GB
            for b in ("UD-IQ4_XS", "UD-Q4_K_XL"):
                self.assertAlmostEqual(_plan(b, free, sm=86)["vram_gb"], engine, delta=0.12, msg=f"{b} {free} GiB")

    def test_the_bf16_path_plan_equals_what_the_engine_places(self):
        for (free, path), counts in PLACED.items():
            if path != "bf16":
                continue
            engine = counts[0] * EXPERT_GB
            for b in ("UD-IQ4_XS", "UD-Q4_K_XL"):
                self.assertAlmostEqual(_plan(b, free, sm=61)["vram_gb"], engine, delta=0.2, msg=f"{b} {free} GiB")

    def test_the_card_class_picks_the_share(self):
        q8 = F.dense_gb_for("UD-Q4_K_XL", "cuda", 86)
        bf16 = F.BUILDS["UD-Q4_K_XL"]["dense_gb"]
        self.assertLess(q8, bf16)
        for sm in (None, 61, 70, 75, 80, 89, 120, 121):                               # unknown, older, A100, Ada, Blackwell (BF16 dense there): the larger share
            self.assertEqual(F.dense_gb_for("UD-Q4_K_XL", "cuda", sm), bf16, sm)
        self.assertEqual(F.dense_gb_for("UD-Q4_K_XL", "cuda", 86, split=True), bf16)  # a layer split keeps BF16
        for be in (None, "hip", "sycl", "metal"):
            self.assertEqual(F.dense_gb_for("UD-Q4_K_XL", be, 86), bf16, be)
        for b in ("SQ-10", "SQ-15", "SQ-20"):                                         # .strata packs have no Q8_0 dense path
            self.assertEqual(F.dense_gb_for(b, "cuda", 86), F.BUILDS[b]["dense_gb"])

    def test_an_unknown_card_never_gets_the_smaller_share(self):
        no_sm = _plan("UD-Q4_K_XL", 11.63)
        old_card = _plan("UD-Q4_K_XL", 11.63, sm=61)
        new_card = _plan("UD-Q4_K_XL", 11.63, sm=86)
        self.assertEqual(no_sm["vram_gb"], old_card["vram_gb"])
        self.assertGreater(new_card["vram_gb"], old_card["vram_gb"] + 0.5)

    def test_a_layer_split_is_planned_with_the_bf16_share(self):
        split = F.place("UD-Q4_K_XL", 11.63, 32.0, False, 32768, False, dspark=False, backend="cuda", desktop=True, ram_free_gib=28.0,
                        extra_vram_gib=11.63, sm=86)
        one = F.place("UD-Q4_K_XL", 11.63, 32.0, False, 32768, False, dspark=False, backend="cuda", desktop=True, ram_free_gib=28.0, sm=86)
        self.assertEqual(split["ram_gb"], 0.0)                     # two cards hold what one card could not
        self.assertGreater(one["vram_gb"], 6.0)

    def test_setup_tells_the_planner_the_cards_compute_capability(self):
        self.assertEqual(S.inkling_plan_extra({"index": 0, "arch": "86"}).get("sm"), 86)
        self.assertEqual(S.inkling_plan_extra({"index": 0, "arch": "120"}).get("sm"), 120)
        self.assertNotIn("sm", S.inkling_plan_extra({"index": 0}))
        self.assertNotIn("sm", S.inkling_plan_extra({"index": 0, "arch": "gfx1100"}))


if __name__ == "__main__":
    unittest.main()
