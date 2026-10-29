"""DeepSeek's valued picture option stays intact (#4507)."""
import unittest

from tools.quant_quality import engine_args


class VisionValue(unittest.TestCase):
    def test_quant_quality_keeps_valued_vision(self):
        base = ["--pack", "/p", "--eos", "1"]
        for vision in (["--vision", "off"], ["--vision", "on"], ["--vision"]):
            for tail in ([], ["--dspark", "none"]):
                args = base + vision + tail
                with self.subTest(vision=vision, tail=tail):
                    cfg = {"exe": "strata-dsv41", "arch": "deepseek41", "args": args}
                    expected = engine_args({**cfg, "args": base + tail}, None, 512)
                    if len(vision) == 2:
                        expected[2 + len(base):2 + len(base)] = vision
                    self.assertEqual(engine_args(cfg, None, 512), expected)
                    self.assertEqual(cfg["args"], args)
