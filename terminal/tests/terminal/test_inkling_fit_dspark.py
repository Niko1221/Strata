"""Inkling-2-Mini fit planner: what the DSpark drafter takes off a discrete card.

Measured on an RTX 3060 Ti 8 GB (7.66 GiB free, ctx 32768, screen reserve, planner flags, the same plan with the drafter off and on; strata-inkling
memory ledger): the engine places 81 fewer experts for SQ-10 (968 -> 887, 0.23 GB) and 41 fewer for UD-Q4_K_XL (489 -> 448, 0.19 GB) with the
drafter on, and the ledger shows `DSpark took 78-80 MiB`. The planner used to charge 0.25 GB for SQ and 0.65 GB for GGUF (0.40 GB of it the
drafter file), 0.45 GB too much for a GGUF build on a tight card."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

TERMINAL = Path(__file__).resolve().parents[2]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

from strata_control import inkling_fit as F

# experts the engine placed on the card, and the bytes one of them takes (SF4 body of the first, loop-weighted, layers)
ENGINE = {"SQ-10": {"off": 968, "on": 887, "mb": 8.60e3 / 3072}, "UD-Q4_K_XL": {"off": 489, "on": 448, "mb": 4.57}}


def _plan(build: str, dspark: bool) -> dict:
    return F.place(build, 7.66, 32.0, False, 32768, False, dspark=dspark, backend="cuda", desktop=True, ram_free_gib=28.0)


class DrafterBudget(unittest.TestCase):
    def test_the_drafter_costs_what_the_engine_gives_up(self):
        for build, e in ENGINE.items():
            planner = _plan(build, False)["vram_gb"] - _plan(build, True)["vram_gb"]
            engine = (e["off"] - e["on"]) * e["mb"] / 1e3
            self.assertAlmostEqual(planner, engine, delta=0.12, msg=f"{build}: planner {planner:.2f} GB, engine {engine:.2f} GB")

    def test_a_gguf_drafter_file_is_not_charged_twice_on_a_discrete_card(self):
        on, off = _plan("UD-Q4_K_XL", True), _plan("UD-Q4_K_XL", False)
        self.assertLess(off["vram_gb"] - on["vram_gb"], F.DSPARK_GGUF_GB)        # not the 0.40 GB of the file on top of the scratch
        self.assertAlmostEqual(off["vram_gb"] - on["vram_gb"], F.dspark_gb(32768, True) + 0.0, delta=0.05)

    def test_the_drafter_is_the_same_charge_for_a_gguf_and_an_sq_build(self):
        d_sq = _plan("SQ-10", False)["vram_gb"] - _plan("SQ-10", True)["vram_gb"]
        d_gg = _plan("UD-Q4_K_XL", False)["vram_gb"] - _plan("UD-Q4_K_XL", True)["vram_gb"]
        self.assertAlmostEqual(d_sq, d_gg, delta=0.05)

    def test_unified_memory_still_holds_the_gguf_drafter_file(self):
        on = F.place("UD-Q4_K_XL", None, 24.0, True, 32768, False, dspark=True)
        off = F.place("UD-Q4_K_XL", None, 24.0, True, 32768, False, dspark=False)
        self.assertGreaterEqual(on["vram_gb"] + on["ssd_gb"], off["vram_gb"] + off["ssd_gb"] + F.DSPARK_GGUF_GB * 0.99)

    def test_the_download_still_counts_the_drafter_file(self):
        row = next(r for r in F.rows(8.0, 32.0) if r["build"] == "UD-Q4_K_XL")
        self.assertAlmostEqual(row["download_gb"], F.BUILDS["UD-Q4_K_XL"]["file_gb"] + F.DSPARK_GGUF_GB, places=2)


if __name__ == "__main__":
    unittest.main()
