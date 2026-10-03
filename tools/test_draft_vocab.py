"""Tests for tools/draft_vocab.py --cover-text (#597): the most frequent tokens of a text, up to a share of its
occurrences.  Counts only - no model, no GPU.

    python tools/test_draft_vocab.py
"""
from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from draft_vocab import covering  # noqa: E402


class Covering(unittest.TestCase):
    def test_the_most_frequent_tokens_up_to_the_share(self):
        counts = Counter([7] * 6 + [3] * 3 + [9])                           # 10 occurrences: 60%, 30%, 10%
        self.assertEqual(covering(counts, 0.6), [7])
        self.assertEqual(covering(counts, 0.9), [7, 3])
        self.assertEqual(covering(counts, 0.99), [7, 3, 9])

    def test_equal_counts_keep_the_order_of_the_text(self):
        self.assertEqual(covering(Counter([5, 2, 5, 2, 8, 8]), 0.5), [5, 2])   # 5 is seen first, then 2, then 8
        self.assertEqual(covering(Counter(), 0.99), [])


if __name__ == "__main__":
    unittest.main()
