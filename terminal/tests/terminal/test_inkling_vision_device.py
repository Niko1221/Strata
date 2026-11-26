"""Inkling-2-Mini: --vision gpu|cpu reaches strata-inkling as --vision-device (it was collapsed to "yes"), the fit budgets the
GPU picture reader's room, and an installed config with another reader device is written again."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

TERMINAL = Path(__file__).resolve().parents[2]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

from strata_cli.launch import runner as R  # noqa: E402
from strata_cli.launch import settings as S  # noqa: E402
from strata_control import inkling_fit  # noqa: E402

INK_CFG = {"exe": "component:strata-inkling", "arch": "inkling2_loop", "model_name": "inkling-2-mini-sq-20",
           "args": ["--native", "/m/Inkling-2-Mini-SQ-20.strata", "--max-context", "131072",
                    "--vision", "/m/Inkling-2-Mini-Vision.strata"]}


class TestFit(unittest.TestCase):
    def test_gpu_reader_is_budgeted(self):
        self.assertGreater(inkling_fit.VISION_GPU_RESERVE_MIB, 0)
        self.assertGreater(inkling_fit.VISION_GPU_GB, inkling_fit.VISION_GB)
        auto = inkling_fit.place("SQ-15", 16, 64, ctx=32768, vision=True, backend="cuda")
        gpu = inkling_fit.place("SQ-15", 16, 64, ctx=32768, vision=True, backend="cuda", vision_device="gpu")
        cpu = inkling_fit.place("SQ-15", 16, 64, ctx=32768, vision=True, backend="cuda", vision_device="cpu")
        self.assertLessEqual(gpu["vram_gb"], auto["vram_gb"])       # the GPU reader's room is not given to experts
        self.assertGreaterEqual(cpu["vram_gb"], auto["vram_gb"])     # the CPU reader takes no VRAM

    def test_default_placement_is_unchanged(self):
        self.assertEqual(inkling_fit.place("SQ-20", 24, 64, ctx=131072, vision=True, backend="cuda"),
                         inkling_fit.place("SQ-20", 24, 64, ctx=131072, vision=True, backend="cuda", vision_device=None))


class TestReaderDevice(unittest.TestCase):
    def test_desc_matches_compares_the_reader_device(self):
        s = S.LaunchSettings(family="inkling-2-mini", vision="gpu")
        self.assertTrue(R.desc_matches({"images": "gpu"}, s))
        self.assertFalse(R.desc_matches({"images": "on"}, s))
        self.assertFalse(R.desc_matches({"images": "cpu"}, s))

    def test_hy4_still_ignores_the_device(self):
        s = S.LaunchSettings(family="hy4-flash", vision="gpu")
        self.assertTrue(R.desc_matches({"images": "on"}, s))

    def test_apply_choices_writes_the_device(self):
        for dev in ("gpu", "cpu"):
            out = R.apply_choices(INK_CFG, S.LaunchSettings(family="inkling-2-mini", vision=dev))
            self.assertEqual(out["args"][out["args"].index("--vision-device") + 1], dev)
        out = R.apply_choices(INK_CFG, S.LaunchSettings(family="inkling-2-mini", vision="on"))
        self.assertNotIn("--vision-device", out["args"])            # "on" leaves the engine's own choice


if __name__ == "__main__":
    unittest.main()
