"""Tests for tools/paired_stats.py - the scipy/statsmodels wrappers.  Skipped when the packages are not installed."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired_stats as P  # noqa: E402


@unittest.skipUnless(P.available(), "scipy and statsmodels not installed")
class StatsTests(unittest.TestCase):
    def test_the_p_floor_of_three_and_nine_pairs(self):
        """The combinatorial floor: the all-same-sign rank sum is unique, so p cannot go below 2**(1-n)."""
        self.assertAlmostEqual(P.signed_rank([0.01, 0.02, 0.03])["p"], 0.25)
        self.assertAlmostEqual(P.signed_rank([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])["p"], 2 ** -8)

    def test_the_shift_estimate_and_its_ci_bracket_the_true_ratio(self):
        d = [math.log(1.05) + x for x in (0.002, -0.002, 0.003, -0.003, 0.001, -0.001, 0.0)]     # one zero pair dropped
        st = P.signed_rank(d)
        self.assertAlmostEqual(st["hl"], math.log(1.05), places=3)
        self.assertLess(st["ci_lo"], math.log(1.05))
        self.assertLess(math.log(1.05), st["ci_hi"])

    def test_significance_states_the_test_and_the_floor_cannot_hide_it(self):
        d = [0.02, 0.03, 0.01, 0.04, 0.02, 0.03, 0.015, 0.035, 0.025]                            # 9 pairs, all faster
        st = P.signed_rank(d)
        adj = P.holm([st["p"]])[0]
        self.assertLessEqual(adj, 0.05)
        self.assertTrue(P.significance(st, adj).startswith("yes"))
        self.assertEqual(P.verdict(st, adj, 0.01), "faster, over the floor")      # a low floor: the change clears it
        self.assertEqual(P.verdict(st, adj, 0.5), "faster, under the floor")      # a 50% floor only softens the word
        pn = P.power_note([0.01, -0.01, 0.02, -0.02, 0.015], 0.05 / 6, 0.02)
        self.assertEqual(pn["n_floor"], 8)                                    # 2**-7 <= 0.05/6 < 2**-6
        self.assertGreater(pn["n_need"], 5)
        self.assertIsNone(P.power_note([0.01], 0.05, 0.02))                   # n < 2: nothing to spread


if __name__ == "__main__":
    unittest.main()
