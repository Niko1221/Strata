"""Tests for the EXL3 reference reconstruction: permutation, Hadamard, folded vs materialized.

    python3 -m unittest discover -s tools -p test_exl3_reconstruct.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # runnable from the repo root too
import exl3                                                 # noqa: E402
from exl3.codebook import CB_MUL1, decode, unpack_tile      # noqa: E402
from exl3.reconstruct import (folded_from_what, weight_from_what)  # noqa: E402


class PermTests(unittest.TestCase):
    def test_perm_is_bijection(self):
        p = exl3.tile_perm()
        self.assertEqual(sorted(int(x) for x in p), list(range(256)))
        np.testing.assert_array_equal(p[exl3.tile_perm_inv()], np.arange(256))
        # the first lane's 8 slots, from quantize.py: r0..r3 at c0, then r0..r3 at c1
        self.assertEqual([int(v) for v in p[:8]], [0, 16, 128, 144, 8, 24, 136, 152])


class HadamardTests(unittest.TestCase):
    def test_sylvester_and_orthonormal(self):
        h = exl3.had128()
        self.assertEqual(h.shape, (128, 128))
        hh = h @ h
        np.testing.assert_allclose(hh, np.eye(128), atol=1e-4)
        # H[i,j] = (-1)^popcount(i&j)/sqrt(128)
        i, j = np.mgrid[0:128, 0:128]
        sign = np.where((np.bitwise_count(i & j) & 1) == 0, 1.0, -1.0)
        np.testing.assert_allclose(h, sign / np.sqrt(128.0), atol=1e-6)


class VectorizedDecodeTests(unittest.TestCase):
    def test_fast_windows_match_slow(self):
        rng = np.random.default_rng(7)
        for bits in (2, 3, 4, 5, 8):
            packed = rng.integers(0, 1 << 16, size=256 * bits // 16, dtype=np.uint16)
            np.testing.assert_array_equal(exl3.windows_tile_fast(packed, bits),
                                          exl3.windows_tile(packed, bits), "bits=%d" % bits)

    def test_lut_matches_decode(self):
        lut = exl3.codebook_lut(CB_MUL1)
        win = np.random.default_rng(3).integers(0, 1 << 16, size=4096, dtype=np.uint32)
        np.testing.assert_array_equal(lut[win], decode(CB_MUL1, win))
        packed = np.arange(48, dtype=np.uint16)
        np.testing.assert_array_equal(exl3.decode_lut(CB_MUL1, packed, 3),
                                      unpack_tile(packed, 3, CB_MUL1))


class ReconstructTests(unittest.TestCase):
    @staticmethod
    def _synthetic(in_f=256, out_f=256, bits=3):
        rng = np.random.default_rng(11)
        trellis = rng.integers(-(1 << 15), 1 << 15, size=(in_f // 16, out_f // 16, 256 * bits // 16),
                               dtype=np.int16)
        suh = (rng.standard_normal(in_f) * 0.5).astype(np.float16)
        svh = (rng.standard_normal(out_f) * 0.5).astype(np.float16)
        return trellis, suh, svh

    def test_decode_weight_hat_shape(self):
        trellis, _, _ = self._synthetic(640, 2560)
        w_hat = exl3.decode_weight_hat(trellis, CB_MUL1)
        self.assertEqual(w_hat.shape, (640, 2560))
        self.assertTrue(np.isfinite(w_hat.astype(np.float32)).all())

    @staticmethod
    def _rel_err(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    def test_folded_equals_materialized(self):
        # The algebra test: y = x . W must equal H(x . suh) @ W_hat . H . svh for random W_hat.
        rng = np.random.default_rng(5)
        k, n = 256, 384
        w_hat = (rng.standard_normal((k, n)) * 0.1).astype(np.float16)
        suh = (rng.standard_normal(k) * 0.5).astype(np.float16)
        svh = (rng.standard_normal(n) * 0.5).astype(np.float16)
        x = (rng.standard_normal((4, k)) * 0.3).astype(np.float16)
        w = weight_from_what(w_hat, suh, svh)
        self.assertLess(self._rel_err(x.astype(np.float32) @ w, folded_from_what(x, w_hat, suh, svh)),
                        1e-3)

    def test_reconstruct_and_folded_match_on_trellis(self):
        trellis, suh, svh = self._synthetic()
        x = np.random.default_rng(9).standard_normal((3, 256)).astype(np.float16)
        w = exl3.reconstruct_weight(trellis, suh, svh, CB_MUL1).astype(np.float32)
        self.assertLess(self._rel_err(x.astype(np.float32) @ w,
                                      exl3.folded_forward(x, trellis, suh, svh, CB_MUL1)),
                        1e-3)


if __name__ == "__main__":
    unittest.main()
