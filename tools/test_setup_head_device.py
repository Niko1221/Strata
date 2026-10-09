"""delta 3 (layer): setup's head's card - the ranking by TIME (SMs x clock, the engine's own per-layer model) and
the card order that puts the head there.  No GPU and no engine: the probe is stubbed.
    python tools/test_setup_head_device.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import setup as S  # noqa: E402


# two cards of the machine this was written on, and a third so the ranking has something to reorder
FAST = {"index": 0, "name": "NVIDIA GeForce RTX 4080 SUPER", "vram_gb": 16.0, "arch": "89"}
BIG_SLOW = {"index": 1, "name": "NVIDIA GeForce RTX 3090", "vram_gb": 24.0, "arch": "86"}
SMALL_SLOW = {"index": 2, "name": "NVIDIA GeForce RTX 3060", "vram_gb": 12.0, "arch": "86"}

# what the probe answers for each card: (SMs, GHz) - the two inputs of the engine's layer model
PROBES = {0: (76, 2.61), 1: (82, 1.70), 2: (28, 1.78)}


def with_probe(cards):
    """`cards` with gpu_layer_ms stubbed by PROBES (and by index, so the order does not matter)."""
    orig = S.gpu_layer_ms
    S.gpu_layer_ms = lambda g, toolkit=13: (S.HEAD_MS_FIT / (PROBES[g["index"]][0] * PROBES[g["index"]][1])
                                            if g["index"] in PROBES else None)
    return orig


class HeadDevice(unittest.TestCase):
    def setUp(self):
        self.orig = with_probe(PROBES)

    def tearDown(self):
        S.gpu_layer_ms = self.orig

    def test_ms_is_the_engines_own_model(self):
        # 0.33 * 84 * 2.617 / (SMs x GHz): the constant the layer-split search was fitted with
        ms = S.gpu_layer_ms(FAST)
        self.assertAlmostEqual(ms, S.HEAD_MS_FIT / (76 * 2.61), places=6)
        self.assertLess(ms, S.gpu_layer_ms(BIG_SLOW))          # the 4080 is the faster card per layer
        self.assertLess(S.gpu_layer_ms(BIG_SLOW), S.gpu_layer_ms(SMALL_SLOW))

    def test_the_head_card_is_chosen_by_time_not_by_vram(self):
        # the card with the most VRAM is the SLOWEST: it must not win the head
        pick = S.head_card_by_time([FAST, BIG_SLOW, SMALL_SLOW])
        self.assertEqual(pick["index"], 0)
        self.assertGreater(pick["vram_gb"], 0)
        self.assertLess(FAST["vram_gb"], BIG_SLOW["vram_gb"])   # ... and it has less VRAM: that is the point
        # nothing to time: None, and setup keeps what it had
        orig = S.gpu_layer_ms
        S.gpu_layer_ms = lambda g, toolkit=13: None
        self.assertIsNone(S.head_card_by_time([FAST, BIG_SLOW]))
        S.gpu_layer_ms = orig

    def test_ties_keep_the_placement_the_engine_would_make(self):
        orig = S.gpu_layer_ms
        S.gpu_layer_ms = lambda g, toolkit=13: 0.5
        self.assertEqual(S.head_card_by_time([FAST, BIG_SLOW])["index"], BIG_SLOW["index"])   # the LAST of the two
        S.gpu_layer_ms = orig

    def test_the_config_order_puts_the_head_last(self):
        # --head-device N: that card goes last, the others keep their order, and the config says which one it is
        cfg = {"gpu": [0, 1, 2]}
        S.apply_head_device(cfg, "0", [FAST, BIG_SLOW, SMALL_SLOW])
        self.assertEqual(cfg["gpu"], [1, 2, 0])
        self.assertEqual(cfg["head_device"], 0)
        # auto: the fastest per layer, which is the same card here
        cfg = {"gpu": [1, 0, 2]}
        S.apply_head_device(cfg, "auto", [FAST, BIG_SLOW, SMALL_SLOW])
        self.assertEqual(cfg["head_device"], 0)
        self.assertEqual(cfg["gpu"], [1, 2, 0])

    def test_not_asked_moves_nothing(self):
        cfg = {"gpu": [0, 1]}
        S.apply_head_device(cfg, None, [FAST, BIG_SLOW])
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertNotIn("head_device", cfg)
        # ... and a single card has nowhere to put the head (said, nothing moved)
        cfg = {"gpu": 0}
        S.apply_head_device(cfg, "auto", [FAST])
        self.assertEqual(cfg["gpu"], 0)
        self.assertNotIn("head_device", cfg)

    def test_the_note_names_the_way_to_move_it(self):
        lines = S.head_device_note([FAST, BIG_SLOW])
        self.assertTrue(any("ms/layer" in x for x in lines))
        self.assertTrue(any("--head-device 0" in x or '"head_device": 0' in x for x in lines))
        # already where the estimate wants it: no "move it" line
        self.assertEqual(len(S.head_device_note([BIG_SLOW, FAST])), 2)
        self.assertEqual(S.head_device_note([FAST]), [])

    def test_no_estimate_says_so(self):
        orig = S.gpu_layer_ms
        S.gpu_layer_ms = lambda g, toolkit=13: None
        lines = S.head_device_note([FAST, BIG_SLOW])
        self.assertEqual(len(lines), 1)
        self.assertIn("could not be estimated", lines[0])
        S.gpu_layer_ms = orig


if __name__ == "__main__":
    unittest.main(verbosity=2)
