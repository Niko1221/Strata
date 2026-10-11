"""Tests for tools/mtp_pack.py's Q2_0 scale search (the MTP draft experts).

    python -m unittest tools.test_mtp_pack -v

The search picks, per 64-weight block, a scale d of either sign over [0.15, 1.0] x max|w|; the engine decodes a
block as (code - 1) * d with d the signed fp16 (include/strata/artifact/dequant.hpp:76, and the CPU and GPU expert
kernels are linear in d).  The decode used here is tools/gguf_writer.dequantize_q2_0, the repository's Python mirror
of dequantize_row_q2_0; the engine's own C++ is proved against a scalar oracle by src/kernels/cpu/expert_multi_test.cpp.
Needs llama.cpp's gguf-py like the packer itself ($STRATA_GGUF_PY): without it the whole module is SKIPPED, with the
reason in the pytest report (run with -rs), never passed silently.
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
    raise unittest.SkipTest(f"tools/mtp_pack.py needs gguf-py (set STRATA_GGUF_PY or run setup.py): {e}")
from gguf_writer import dequantize_q2_0

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


def stored_scales(blocks: np.ndarray) -> np.ndarray:
    """The fp16 scale of each (N, 18) block, as float32."""
    return np.ascontiguousarray(blocks[:, :2]).view(np.float16).reshape(-1).astype(np.float32)


def decode(blob: np.ndarray) -> np.ndarray:
    """Blocks as the engine decodes them (dequantize_q2_0 of tools/gguf_writer.py), one row of 64 per block."""
    return dequantize_q2_0(blob.tobytes()).reshape(-1, 64)


def block_errors(blob: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Squared error of each block's decode against its 64 weights."""
    return ((decode(blob) - x.reshape(-1, 64)) ** 2).sum(axis=1)


def relative_rms(blob: np.ndarray, x: np.ndarray) -> float:
    got = mtp_pack.dequant("q2_0", blob, x.size)
    return float(np.sqrt(((got - x) ** 2).mean()) / np.sqrt((x ** 2).mean()))


class ScaleSearchTests(unittest.TestCase):
    def test_no_block_has_a_larger_error_than_with_the_previous_search(self):
        for name, law in LAWS.items():
            with self.subTest(law=name):
                x = law(np.random.default_rng(0), BLOCKS * 64).astype(np.float32)
                new, old = mtp_pack.q2_0(x), q2_0_previous(x)
                worse = block_errors(new, x) > block_errors(old, x)
                self.assertEqual(int(worse.sum()), 0, f"{int(worse.sum())} of {BLOCKS} blocks got a larger error")
                rms_new, rms_old = relative_rms(new, x), relative_rms(old, x)
                print(f"\n{name}: relative RMS previous {rms_old:.5f}, now {rms_new:.5f}, ratio {rms_new / rms_old:.4f}")
                self.assertLess(rms_new / rms_old, 0.90)    # the wider grid and the sign are worth 14 to 18 %

    def test_block_whose_best_scale_is_a_previous_grid_point_keeps_it(self):
        x = np.zeros(64, dtype=np.float32)
        x[:3] = (2.0, 1.0, -1.0)                            # exact only for d = 1.0 = 0.5 x max|w|, a previous point
        new, old = mtp_pack.q2_0(x), q2_0_previous(x)
        self.assertEqual(float(block_errors(old, x)[0]), 0.0)
        self.assertEqual(float(block_errors(new, x)[0]), 0.0)

    def test_negative_scales_are_chosen_and_stored(self):
        x = np.random.default_rng(1).standard_normal(BLOCKS * 64).astype(np.float32)
        d = stored_scales(mtp_pack.q2_0(x).reshape(-1, 18)).reshape(-1, 1)
        self.assertGreater(float((d < 0).mean()), 0.3)
        self.assertGreater(float((d > 0).mean()), 0.3)
        amax = np.abs(x.reshape(-1, 64)).max(axis=1, keepdims=True)
        self.assertTrue(np.all(np.abs(d) <= amax * 1.001))      # 1.0 x max|w| is the top of the grid (plus fp16 rounding)
        self.assertTrue(np.all(np.abs(d) >= amax * 0.149))      # 0.15 x max|w| the bottom

    def test_zero_block_keeps_a_zero_scale(self):
        blob = mtp_pack.q2_0(np.zeros(64, dtype=np.float32))
        self.assertEqual(blob[:2].tolist(), [0, 0])
        self.assertTrue(np.all(decode(blob) == 0))

    def test_block_too_large_for_an_fp16_scale_is_refused(self):
        x = np.zeros(64, dtype=np.float32)
        x[0] = 70000.0
        with self.assertRaises(ValueError):
            mtp_pack.q2_0(x)


class DequantTests(unittest.TestCase):
    def test_engine_decode_gives_the_values_the_search_optimized(self):
        x = np.random.default_rng(2).standard_normal(4000 * 64).astype(np.float32)
        blob = mtp_pack.q2_0(x).reshape(-1, 18)
        d = stored_scales(blob).reshape(-1, 1)
        # the codes are the weights rounded against the STORED scale, then decoded as (code - 1) * d
        want = np.clip(np.rint(x.reshape(-1, 64) * (np.float32(1) / d)), -1, 2).astype(np.float32) * d
        got = decode(blob)
        np.testing.assert_array_equal(got, want)
        self.assertGreater(int((d < 0).sum()), 0)
        np.testing.assert_array_equal(mtp_pack.dequant("q2_0", blob.reshape(-1), x.size), got.reshape(-1))

    def test_negative_scale_block_decodes_with_the_engine_formula(self):
        block = np.zeros(18, dtype=np.uint8)
        block[:2] = np.array([-0.5], dtype=np.float16).view(np.uint8)
        block[2:] = 0xE4                                        # codes 0, 1, 2, 3 -> (-1, 0, 1, 2) x d
        got = decode(block)[0]
        np.testing.assert_array_equal(got[:4], np.array([0.5, 0.0, -0.5, -1.0], dtype=np.float32))
        self.assertTrue(np.signbit(got[1]))                     # code 1 keeps the sign of the zero (tests/hip/q2_zero.cpp)


if __name__ == "__main__":
    unittest.main()
