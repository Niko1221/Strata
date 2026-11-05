"""Pictures on for DeepSeek on a card with memory of its own: setup raises the engine's per-card reserve (--reserve-mib, default 600) so the first picture's
hipBLASLt start and the tower's buffers have room.  Not on a Mac, not on a unified-memory APU, not with pictures off.  (The amount is provisional.)"""

import importlib.util
import unittest
from pathlib import Path

from strata_control import dsv41_fit

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("setup_vr", ROOT / "setup.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)

R9700 = {"index": 0, "name": "AMD Radeon AI PRO R9700", "vram_gb": 32, "vendor": "amd", "arch": "gfx1201", "usable": True}
STRIX = {"index": 0, "name": "AMD Radeon 8060S Graphics", "vram_gb": 96, "vendor": "amd", "arch": "gfx1151", "usable": True, "uma": True, "integrated": True}
RTX = {"index": 0, "name": "NVIDIA GeForce RTX 5070", "vram_gb": 12, "vendor": "nvidia", "usable": True}


class VisionReserve(unittest.TestCase):
    def plan(self, card):
        return dsv41_fit.plan([card], 125.0, None, 131072, model="ISTA-DAS-Coder")

    def test_a_discrete_card_with_pictures_on_gets_the_larger_reserve(self):
        for card in (R9700, RTX):
            args = setup.dsv41_vision_reserve_args("on", False, self.plan(card))
            self.assertEqual(args, ["--reserve-mib", str(setup.DSV41_VISION_RESERVE_MIB)], card["name"])
        self.assertGreater(setup.DSV41_VISION_RESERVE_MIB, 600)             # the engine's default

    def test_pictures_off_a_mac_and_a_unified_memory_apu_are_unchanged(self):
        self.assertEqual(setup.dsv41_vision_reserve_args("off", False, self.plan(R9700)), [])
        self.assertEqual(setup.dsv41_vision_reserve_args("on", True, self.plan(R9700)), [])
        strix = self.plan(STRIX)
        self.assertTrue(strix.get("uma"))
        self.assertEqual(setup.dsv41_vision_reserve_args("on", False, strix), [])

    def test_setup_puts_the_flag_after_the_plans_own_arguments(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn('*plan["args"], *dsv41_vision_reserve_args(ds_vision, mac, plan)]', src)


if __name__ == "__main__":
    unittest.main()
