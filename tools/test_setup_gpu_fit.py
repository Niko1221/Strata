"""The owner's multi-GPU rule in setup (strata_control/gpu_fit.py): the model goes on the LARGEST card + RAM first; the second card is
offered as a layer split and is the default only when the model does not fit one card (with RAM) or would be too tight.

    python tools/test_setup_gpu_fit.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from test_setup_golden import card, install  # noqa: E402


def run(ram, found, model, answers=None, extra=()):
    argv = ["--family", "qwen", "--no-start", "--model", model, "--context", "32768", *extra]
    return install(ram, found, argv, answers=answers or {})


class SetupFitRule(unittest.TestCase):
    def test_one_card_has_no_question(self):
        code, out, cfg, asked = run(62.0, [card(0, "NVIDIA GeForce RTX 3070 Ti", 8.0, "86")], "IQ3_XXS")
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], 0) if "gpu" in cfg else None
        self.assertFalse(any("Which GPUs" in q for q in asked), asked)

    def test_two_unequal_cards_and_the_model_fits_one_defaults_to_the_largest_card_and_offers_the_split(self):
        found = [card(0, "NVIDIA GeForce RTX 3060 Ti", 8.0, "86"), card(1, "NVIDIA GeForce RTX 3070 Ti", 16.0, "86")]
        code, out, cfg, asked = run(62.0, found, "IQ3_XXS")
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], 1)                                   # the 16 GB card, alone
        self.assertNotIn("layer_split", cfg)
        self.assertIn("layer split", out)                                 # the split was offered
        self.assertIn("1) RTX 3070 Ti + RAM", out.replace("NVIDIA GeForce ", ""))
        self.assertTrue(any("Which GPUs" in q and "[1]" in q for q in asked), asked)
        # choosing the split is one answer away
        code, out, cfg, _ = run(62.0, found, "IQ3_XXS", answers={"Which GPUs?": "2"})
        self.assertEqual(code, 0, out)
        self.assertEqual(sorted(cfg["gpu"]), [0, 1])
        self.assertEqual(cfg["layer_split"], "auto")

    def test_two_small_cards_and_the_model_does_not_fit_one_defaults_to_the_split(self):
        found = [card(0, "NVIDIA GeForce RTX 3060", 12.0, "86"), card(1, "NVIDIA GeForce RTX 3060", 12.0, "86")]
        code, out, cfg, asked = run(26.0, found, "IQ3_XXS")
        self.assertEqual(code, 0, out)
        self.assertEqual(sorted(cfg["gpu"]), [0, 1])
        self.assertEqual(cfg["layer_split"], "auto")
        self.assertTrue(any("Which GPUs" in q and "[2]" in q for q in asked), asked)

    def test_naming_the_cards_still_wins(self):
        found = [card(0, "NVIDIA GeForce RTX 3070 Ti", 16.0, "86"), card(1, "NVIDIA GeForce RTX 3060 Ti", 8.0, "86")]
        code, out, cfg, _ = run(62.0, found, "IQ3_XXS", extra=("--gpus", "0,1"))
        self.assertEqual(code, 0, out)
        self.assertEqual(sorted(cfg["gpu"]), [0, 1])
        code, out, cfg, _ = run(62.0, found, "IQ3_XXS", extra=("--gpu", "1"))
        self.assertEqual(cfg["gpu"], 1)

    def test_a_bigger_amd_card_beside_the_nvidia_one_is_said_in_one_line_and_never_mixed(self):
        """RTX 5060 Ti (16 GB) + AMD R9700 (32 GB): the question offers the two makes apart, says one engine drives one make and that two
        models at once are not supported, and names the bigger AMD card; the NVIDIA answer runs on the NVIDIA card alone."""
        from unittest import mock
        found = [card(0, "NVIDIA GeForce RTX 5060 Ti", 16.0, "120")]
        amd = [{"index": 0, "name": "AMD Radeon AI PRO R9700", "vram_gb": 32.0, "arch": "gfx1201", "vendor": "amd"}]
        patches = [mock.patch.object(setup, "amd_gpus", lambda *a, **k: [dict(g) for g in amd]),
                   mock.patch.object(setup, "amd_problem", lambda g: None)]
        argv = ["--family", "qwen", "--no-start", "--model", "IQ3_XXS", "--context", "32768"]
        code, out, cfg, asked = install(62.0, found, argv, answers={"Which cards?": "1"}, extra=patches)
        self.assertEqual(code, 0, out)
        self.assertIn("This PC has NVIDIA and AMD cards Strata can use", out)
        self.assertIn("never split across NVIDIA and AMD cards", out)
        self.assertIn("two models at the same", out)
        self.assertIn("not supported yet", out)
        self.assertIn("This computer also has an AMD AI PRO R9700 (32 GB). The AMD build of Strata can use it", out)
        self.assertNotIn("Which GPUs?", " ".join(asked))              # one NVIDIA card: no split question
        self.assertEqual(cfg.get("gpu", 0), 0)


if __name__ == "__main__":
    unittest.main()
