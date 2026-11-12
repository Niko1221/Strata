"""Tests for tools/calibrate_serving.py: the single/multi scoring and the env split.  No GPU, no server."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_serving as C  # noqa: E402


class Score(unittest.TestCase):
    def probe(self, tps, q=0.0, p=1.0, gap=0.0):
        return {"agg_tps": tps, "max_q": q, "sum_p": p, "max_gap": gap}

    def test_multi_puts_queue_wait_before_throughput(self):
        base = self.probe(10.0, q=8.0)
        low_q = self.probe(9.8, q=0.2)     # 2% slower, queue nearly gone: the winner
        fast_but_waiting = self.probe(12.0, q=9.0)
        self.assertLess(C.score(low_q, base, "multi")[0], C.score(fast_but_waiting, base, "multi")[0])

    def test_multi_refuses_a_throughput_loss_over_3_percent(self):
        base = self.probe(10.0)
        slow = self.probe(9.0)             # 10% slower
        self.assertGreater(C.score(slow, base, "multi")[0], 1e6)

    def test_single_puts_throughput_first(self):
        base = self.probe(10.0, p=2.0)
        fast = self.probe(12.0, p=2.0)
        slow = self.probe(10.5, p=1.0)     # a little slower, shorter prompt
        self.assertLess(C.score(fast, base, "single")[0], C.score(slow, base, "single")[0])

    def test_single_refuses_a_throughput_loss_over_2_percent(self):
        base = self.probe(10.0)
        self.assertGreater(C.score(self.probe(9.7), base, "single")[0], 1e6)


class EnvSplit(unittest.TestCase):
    def test_server_keys_are_not_written_to_the_engine_config(self):
        self.assertIn("STRATA_COALESCE_PREFILL", C.SERVER_ENV)
        self.assertNotIn("STRATA_PREFILL_ROUTED_STREAM", C.SERVER_ENV)
        self.assertNotIn("STRATA_PIPE_ADAPT_EVERY", C.SERVER_ENV)


class Prompt(unittest.TestCase):
    def test_same_seed_same_prompt(self):
        self.assertEqual(C.make_prompt(32, 7), C.make_prompt(32, 7))
        self.assertNotEqual(C.make_prompt(32, 7), C.make_prompt(32, 8))


if __name__ == "__main__":
    unittest.main()
