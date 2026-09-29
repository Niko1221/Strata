"""Tests for setup.py's automatic rope configuration (the §6 flow): the setup resolves the method and the
factor for a context past the trained 262144, keeps explicit choices, refuses an explicit none there, and
adds nothing inside the trained range.  Pure functions - no GPU, no downloads, no prompts.

    python -m unittest tools.test_setup_rope
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class CoveringFactor(unittest.TestCase):
    def test_the_menu_contexts(self):
        self.assertEqual(setup.covering_factor(393216), 1.5)       # 262144 * 1.5 covers 384K exactly
        self.assertEqual(setup.covering_factor(524288), 2.0)       # 262144 * 2.0 covers 512K exactly
        self.assertEqual(setup.covering_factor(262144), 1.5)       # the smallest on the safe side

    def test_the_first_gap_takes_the_bigger_factor(self):
        self.assertEqual(setup.covering_factor(393217), 2.0)       # one token past 1.5's reach


class ResolveRope(unittest.TestCase):
    # ---- the automatic flow the docs promise (an omitted flag, --yes or the interactive default)
    def test_omitted_384k_gets_yarn_and_the_covering_factor(self):
        self.assertEqual(setup.resolve_rope(393216, None, None), ("yarn", 1.5))

    def test_omitted_512k_gets_yarn_and_the_covering_factor(self):
        self.assertEqual(setup.resolve_rope(524288, None, None), ("yarn", 2.0))

    def test_deterministic_for_yes(self):
        # --yes never prompts, so the resolution must be a pure function of its inputs
        for _ in range(3):
            self.assertEqual(setup.resolve_rope(524288, None, None), ("yarn", 2.0))

    def test_omitted_method_with_an_explicit_factor(self):
        # --rope-scale alone still expresses the intent to extend: the method defaults, the factor is kept
        self.assertEqual(setup.resolve_rope(393216, None, 1.8), ("yarn", 1.8))

    # ---- explicit selections are respected
    def test_explicit_method_gets_the_covering_factor(self):
        self.assertEqual(setup.resolve_rope(524288, "linear", None), ("linear", 2.0))

    def test_explicit_method_and_factor_are_kept_verbatim(self):
        self.assertEqual(setup.resolve_rope(393216, "linear", 2.0), ("linear", 2.0))
        self.assertEqual(setup.resolve_rope(524288, "yarn", 1.5), ("yarn", 1.5))

    # ---- an explicit none past the trained range is refused with an explanation, not overridden
    def test_explicit_none_past_trained_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            setup.resolve_rope(524288, "none", None)
        self.assertIn("262144", str(cm.exception))
        self.assertIn("yarn", str(cm.exception))                   # the message names the way out

    def test_explicit_none_within_trained_is_the_stock_model(self):
        self.assertEqual(setup.resolve_rope(131072, "none", None), (None, None))

    # ---- inside the trained range nothing turns on by itself
    def test_omitted_within_trained_adds_nothing(self):
        self.assertEqual(setup.resolve_rope(262144, None, None), (None, None))
        self.assertEqual(setup.resolve_rope(131072, None, None), (None, None))

    def test_a_scale_alone_within_trained_is_rejected(self):
        with self.assertRaises(ValueError):
            setup.resolve_rope(131072, None, 2.0)

    def test_explicit_method_within_trained_keeps_the_engine_default_factor(self):
        self.assertEqual(setup.resolve_rope(131072, "yarn", None), ("yarn", 2.0))

    # ---- §4 coordination: the RAM reduction lands BEFORE the rope config, so a reduced 384K needs nothing
    def test_a_reduced_context_resolves_as_the_reduced_one(self):
        # asked 393216, the RAM check brought it to 131072: the setup must not scale a trained-range context
        self.assertEqual(setup.resolve_rope(131072, None, None), (None, None))
        # ...but an explicit selection survives the reduction (the engine's in-range caveat is its own)
        self.assertEqual(setup.resolve_rope(131072, "yarn", None), ("yarn", 2.0))


if __name__ == "__main__":
    unittest.main()
