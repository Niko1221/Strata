"""Tests for tools/mtp_pack.py's Q2_0 scale search (the MTP draft experts).

    python -m unittest tools.test_mtp_pack -v

The search picks, per 64-weight block, a scale d of either sign over [0.15, 1.0] x max|w|; the engine decodes a
block as (code - 1) * d with d the signed fp16 (include/strata/artifact/dequant.hpp:76, and the CPU and GPU expert
kernels are linear in d).  Needs llama.cpp's gguf-py like the packer itself ($STRATA_GGUF_PY); skipped without it.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import mtp_pack
except SystemExit as e:                      # tools/_paths.py found no gguf-py
    raise unittest.SkipTest(str(e))

BLOCKS = 20000
LAWS = {
    "normal": lambda rng, n: rng.standard_normal(n),
    "laplace": lambda rng, n: rng.laplace(size=n),
    "t3": lambda rng, n: rng.standard_t(3, size=n),
}


def q2_0_previous(w: np.ndarray) -> np.ndarray:
    """The search before the wider grid: scales in [0.5, 1.0] x max|w|, 17 steps, positive only."""
    x = w.reshape(-1, 64).astype(np.float32)
    amax = np.abs(x).max(axis=1, keepdims=True)
    best_err = np.full((x.shape[0], 1), np.inf, dtype=np.float32)
    best_d = np.zeros_like(amax)
    for f in np.linspace(0.5, 1.0, 17, dtype=np.float32):
        d = amax * f
        inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
        q = np.clip(np.rint(x * inv), -1, 2)
        err = ((q * d - x) ** 2).sum(axis=1, keepdims=True)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_d = np.where(better, d, best_d)
    d16 = best_d.astype(np.float16)
    d = d16.astype(np.float32)
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
    codes = (np.clip(np.rint(x * inv), -1, 2) + 1).astype(np.uint8)
    c = codes.reshape(-1, 16, 4)
    packed = (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).astype(np.uint8)
    out = np.empty((x.shape[0], 18), dtype=np.uint8)
    out[:, :2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = packed
    return out.reshape(-1)


def stored_scale(block: np.ndarray) -> np.float32:
    return np.float32(block[:2].copy().view(np.float16)[0])


def decode_block(block: np.ndarray) -> np.ndarray:
    """One 18-byte block as the engine does: out[j] = (code - 1) * d, the 2-bit codes four per byte, low bits first
    (dequantize_q2_0, include/strata/artifact/dequant.hpp:68-77)."""
    d = stored_scale(block)
    out = np.empty(64, dtype=np.float32)
    for j in range(64):
        code = (int(block[2 + j // 4]) >> ((j % 4) * 2)) & 3
        out[j] = np.float32(code - 1) * d
    return out


def relative_rms(blob: np.ndarray, x: np.ndarray) -> float:
    got = mtp_pack.dequant("q2_0", blob, x.size)
    return float(np.sqrt(((got - x) ** 2).mean()) / np.sqrt((x ** 2).mean()))


class ScaleSearchTests(unittest.TestCase):
    def test_wider_search_never_has_a_larger_error_than_the_previous_one(self):
        for name, law in LAWS.items():
            with self.subTest(law=name):
                x = law(np.random.default_rng(0), BLOCKS * 64).astype(np.float32)
                new, old = relative_rms(mtp_pack.q2_0(x), x), relative_rms(q2_0_previous(x), x)
                print(f"\n{name}: relative RMS previous {old:.5f}, now {new:.5f}, ratio {new / old:.4f}")
                self.assertLessEqual(new, old)
                self.assertLess(new / old, 0.90)            # the wider grid and the sign are worth 14 to 18 %

    def test_negative_scales_are_chosen_and_stored(self):
        x = np.random.default_rng(1).standard_normal(BLOCKS * 64).astype(np.float32)
        d = mtp_pack.q2_0(x).reshape(-1, 18)[:, :2].copy().view(np.float16).astype(np.float32)
        self.assertGreater(float((d < 0).mean()), 0.3)
        self.assertGreater(float((d > 0).mean()), 0.3)
        amax = np.abs(x.reshape(-1, 64)).max(axis=1, keepdims=True)
        self.assertTrue(np.all(np.abs(d) <= amax * 1.001))      # 1.0 x max|w| is the top of the grid (plus fp16 rounding)
        self.assertTrue(np.all(np.abs(d) >= amax * 0.149))      # 0.15 x max|w| the bottom

    def test_zero_block_keeps_a_zero_scale(self):
        blob = mtp_pack.q2_0(np.zeros(64, dtype=np.float32))
        self.assertEqual(blob[:2].tolist(), [0, 0])
        self.assertTrue(np.all(decode_block(blob) == 0))


class DequantTests(unittest.TestCase):
    def test_engine_formula_gives_the_values_the_search_optimized(self):
        x = np.random.default_rng(2).standard_normal(4000 * 64).astype(np.float32)
        blob = mtp_pack.q2_0(x).reshape(-1, 18)
        xb = x.reshape(-1, 64)
        negatives = 0
        for b in range(blob.shape[0]):
            d = stored_scale(blob[b])
            got = decode_block(blob[b])
            # the codes are the weights rounded against the STORED scale, then decoded as (code - 1) * d
            want = np.clip(np.rint(xb[b] * (np.float32(1) / d)), -1, 2).astype(np.float32) * d
            np.testing.assert_array_equal(got, want)
            negatives += d < 0
        self.assertGreater(negatives, 0)
        np.testing.assert_array_equal(mtp_pack.dequant("q2_0", blob.reshape(-1), x.size),
                                      np.concatenate([decode_block(r) for r in blob]))

    def test_negative_scale_block_decodes_with_the_engine_formula(self):
        block = np.zeros(18, dtype=np.uint8)
        block[:2] = np.array([-0.5], dtype=np.float16).view(np.uint8)
        block[2:] = 0xE4                                        # codes 0, 1, 2, 3 -> (-1, 0, 1, 2) x d
        got = decode_block(block)
        np.testing.assert_array_equal(got[:4], np.array([0.5, 0.0, -0.5, -1.0], dtype=np.float32))
        self.assertTrue(np.signbit(got[1]))                     # code 1 keeps the sign of the zero (tests/hip/q2_zero.cpp)


if __name__ == "__main__":
    unittest.main()
