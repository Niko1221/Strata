"""#6905: the Inkling planner keeps the global K/V in a CUDA card only with room to spare."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

TERMINAL = Path(__file__).resolve().parents[2]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

from strata_control import inkling_fit as F


class KvHeadroomOnCuda(unittest.TestCase):
    """#6905: RTX 5060 Ti 16 GB, SQ-10, 64K, vision: the plan kept the K/V in the card with 0.4 GB to spare, the browser took the next gigabyte, and growing the K/V ended the engine."""

    def test_the_reporters_card_puts_the_k_v_on_the_host_not_in_a_card_with_no_room(self):
        p = F.compute("SQ-10", 15.9, 128, 64000, vision=True, backend="cuda")
        self.assertEqual(p["kv_offload"], "on")
        self.assertTrue(p["full_vram"])                    # the experts all stay in the card
        self.assertGreater(p["host_kv_gb"], 1.0)

    def test_a_card_with_room_keeps_the_k_v_in_the_card(self):
        p = F.compute("SQ-10", 24.0, 128, 64000, vision=True, backend="cuda")
        self.assertEqual(p["kv_offload"], "off")

    def test_other_backends_and_an_explicit_choice_are_not_touched(self):
        # HIP / SYCL have no host K/V; --kv-offload off is the person's choice
        self.assertEqual(F.compute("SQ-10", 15.9, 128, 64000, vision=True, backend="hip")["kv_offload"], "off")
        self.assertEqual(F.compute("SQ-10", 15.9, 128, 64000, vision=True, backend="cuda", kv_offload="off")["kv_offload"], "off")

    def test_a_small_k_v_keeps_no_headroom_rule(self):
        # 8K context: the K/V is far under 1 GB, nothing to move
        self.assertEqual(F.compute("SQ-10", 15.9, 128, 8192, vision=False, backend="cuda")["kv_offload"], "off")


if __name__ == "__main__":
    unittest.main()
