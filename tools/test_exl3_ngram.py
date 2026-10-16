"""Tests for the EXL3 n-gram ring codec (tools/exl3/ngram.py).  No model needed.

    python3 -m unittest discover -s tools -p test_exl3_ngram.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exl3                                                    # noqa: E402
from exl3 import ngram                                         # noqa: E402

ROW_DIM = 160


class CodebookTests(unittest.TestCase):
    def test_mul1_matches_exllama_formula(self):
        # ExLlamaV3 ngram_codec.mul1_codebook: h = 1024 + byte_sum(prod); (h * k_inv + k_bias) in fp16
        s = np.arange(1 << 16, dtype=np.int64)
        prod = (s * 0x83DCD12D) & 0xFFFFFFFF
        bsum = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255)
        h = (1024 + bsum).astype(np.float32)
        k_inv = np.array([0x1EEE], np.uint16).view(np.float16).astype(np.float32)
        k_bias = np.array([0xC931], np.uint16).view(np.float16).astype(np.float32)
        ref = (h * k_inv + k_bias).astype(np.float16)
        np.testing.assert_array_equal(ngram.mul1_codebook(), ref)


class RingTests(unittest.TestCase):
    @staticmethod
    def _canonical(symbols, K):
        dim = symbols.shape[0]
        st = np.zeros(dim, dtype=np.int64)
        for j in range((15 + K) // K):
            st |= np.roll(symbols, j) << (j * K)
        return (st & 0xFFFF).astype(np.uint16)

    def test_words_per_row(self):
        for K in range(1, 9):
            self.assertEqual(ngram.words_per_row(K), 1 + ROW_DIM * K // 16)
        self.assertEqual(ngram.words_per_row(5), 51)     # matches the shipped ng5 table
        self.assertEqual(ngram.ring_dim(51, 5), 160)

    def test_pack_unpack_roundtrip(self):
        rng = np.random.default_rng(3)
        cb = ngram.mul1_codebook()
        for K in range(1, 9):
            symbols = rng.integers(0, 1 << K, size=ROW_DIM, dtype=np.int64)
            states = self._canonical(symbols, K)
            scale = np.float16(rng.uniform(0.01, 0.1))
            packed = ngram.pack_row(states, scale, K)
            self.assertEqual(packed.shape[0], ngram.words_per_row(K))
            st2, sc2 = ngram.unpack_row(packed, K)
            np.testing.assert_array_equal(st2, states, "K=%d" % K)
            self.assertEqual(np.float16(sc2), np.float16(scale))
            out = ngram.dequant_row(packed, K, cb)
            self.assertEqual(out.shape, (ROW_DIM,))
            self.assertTrue(np.isfinite(out).all())

    def test_head_of_row(self):
        offsets = np.array([0, 20000003, 40000026], dtype=np.int64)
        self.assertEqual(ngram.head_of_row(0, offsets), 0)
        self.assertEqual(ngram.head_of_row(20000002, offsets), 0)
        self.assertEqual(ngram.head_of_row(20000003, offsets), 1)
        self.assertEqual(ngram.head_of_row(40000026, offsets), 2)


if __name__ == "__main__":
    unittest.main()
