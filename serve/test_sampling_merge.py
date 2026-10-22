"""Offline tests for serve.server.merge_sampling: greedy must not inherit a model-card penalty (MTP stays on).

No engine, no GPU, no server:

    python serve/test_sampling_merge.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve.server import StrataEngine, merge_sampling  # noqa: E402

# Qwen3.6-35B-A3B's non-thinking card, the measured offender: presence_penalty 1.5.
CARD = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}


class GreedyPenaltyMerge(unittest.TestCase):
    def test_greedy_without_request_penalty_is_neutral(self):
        merged, note = merge_sampling(CARD, {"temperature": 0})
        self.assertEqual(merged["temperature"], 0)
        self.assertEqual(merged["presence_penalty"], 0.0)
        self.assertEqual(merged["frequency_penalty"], 0.0)
        self.assertEqual(merged["repetition_penalty"], 1.0)
        self.assertIn("[strata] greedy: penalty defaults not merged (MTP stays on)", note)
        self.assertNotIn("penalty_present", StrataEngine.sampling_keys(merged))

    def test_explicit_request_penalty_is_kept(self):
        merged, note = merge_sampling(CARD, {"temperature": 0, "presence_penalty": 1.5})
        self.assertEqual(merged["presence_penalty"], 1.5)
        self.assertIn("kept (MTP off for this request)", note)
        self.assertIn("penalty_present=1.5", StrataEngine.sampling_keys(merged))

    def test_temperature_above_zero_is_unchanged(self):
        merged, note = merge_sampling(CARD, {"temperature": 0.7})
        self.assertEqual(merged, CARD)
        self.assertEqual(note, "")
        self.assertIn("penalty_present=1.5", StrataEngine.sampling_keys(merged))

    def test_config_greedy_is_neutral_too(self):
        merged, note = merge_sampling({"temperature": 0.0, "presence_penalty": 1.5}, None)
        self.assertEqual(merged["presence_penalty"], 0.0)
        self.assertIn("MTP stays on", note)

    def test_explicit_penalty_at_temperature_above_zero_is_untouched(self):
        merged, note = merge_sampling(CARD, {"temperature": 0.7, "frequency_penalty": 0.3})
        self.assertEqual(merged["frequency_penalty"], 0.3)
        self.assertEqual(note, "")


if __name__ == "__main__":
    unittest.main()
