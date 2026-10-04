"""#642: setup recommends the expert pool's worker count on a hybrid CPU (the P-cores but the host loop's one, plus
half of the E-cores) and writes nothing new anywhere else; a calibration's measured count wins over the rule.

    python -m unittest tools.test_setup_hybrid
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from test_setup_golden import PROFILES, install  # noqa: E402


class HybridWorkers(unittest.TestCase):
    def test_rule(self):
        self.assertIsNone(setup.hybrid_pool_workers(None))
        self.assertEqual(setup.hybrid_pool_workers((8, 16)), 15)   # i9-14900K: the fork's measured 15
        self.assertEqual(setup.hybrid_pool_workers((6, 8)), 9)     # i5-13600K
        self.assertEqual(setup.hybrid_pool_workers((4, 8)), 7)     # Ryzen AI 9 HX 370 (Zen 5 + Zen 5c)
        self.assertIsNone(setup.hybrid_pool_workers((8, 8)))       # i7-13700K: all cores measured faster (AMD_HIP.md)
        self.assertIsNone(setup.hybrid_pool_workers((8, 4)))

    def test_recommend_keeps_a_given_count(self):
        with mock.patch.object(setup, "cpu_cores", lambda: (8, 16)), mock.patch.object(setup, "ok", lambda *a: None):
            self.assertEqual(setup.recommend_pool_workers(["--spec", "4"]), ["--spec", "4", "--pool-workers", "15"])
            given = ["--pool-workers", "12"]
            self.assertEqual(setup.recommend_pool_workers(given), given)
        with mock.patch.object(setup, "cpu_cores", lambda: None):
            self.assertEqual(setup.recommend_pool_workers(["--spec", "4"]), ["--spec", "4"])

    def test_install_on_a_hybrid_cpu(self):
        ram, found = PROFILES["96GB-1x16GB"]
        hybrid = [mock.patch.object(setup, "cpu_cores", lambda: (8, 16))]
        code, text, cfg, _ = install(ram, found, ["--family", "qwen", "--no-start", "--model", "Q2_0"], extra=hybrid)
        self.assertEqual(code, 0, text)
        a = cfg["args"]
        self.assertEqual(a[a.index("--pool-workers") + 1], "15")
        self.assertIn("hybrid CPU (8 performance + 16 efficiency cores)", text)
        # the same PC without the hybrid CPU: the config is the one it always was (no --pool-workers)
        code, text, plain, _ = install(ram, found, ["--family", "qwen", "--no-start", "--model", "Q2_0"])
        self.assertEqual(code, 0, text)
        self.assertNotIn("--pool-workers", plain["args"])
        i = a.index("--pool-workers")
        self.assertEqual(a[:i] + a[i + 2:], plain["args"])

    def test_calibration_wins(self):
        ram, found = PROFILES["96GB-1x16GB"]
        cal = {"settings": {"--pool-workers": "11"}, "date": "2026-10-03"}
        extra = [mock.patch.object(setup, "cpu_cores", lambda: (8, 16)),
                 mock.patch.object(setup, "saved_calibration", lambda cfg: cal)]
        code, text, cfg, _ = install(ram, found, ["--family", "qwen", "--no-start", "--model", "Q2_0"], extra=extra)
        self.assertEqual(code, 0, text)
        a = cfg["args"]
        self.assertEqual(a[a.index("--pool-workers") + 1], "11")
        self.assertEqual(a.count("--pool-workers"), 1)


if __name__ == "__main__":
    unittest.main()
