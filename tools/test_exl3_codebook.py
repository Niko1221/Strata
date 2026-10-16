"""Codebook/bitstream tests for the EXL3 reference decoder.

    python3 -m unittest discover -s tools -p test_exl3_codebook.py

These need no model and no GPU: they pin the procedural codebook and the 16-bit sliding-window
extraction against hand-derived values, so a later C++/HIP port has a golden target.
"""
from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # runnable from the repo root too
import exl3                                                 # noqa: E402
from exl3.codebook import (_LOP3_IMM, _LOP3_MASK1, _LOP3_MASK2, CB_3INST,  # noqa: E402
                           CB_MCG, CB_MUL1, _decode_mul1)


def h(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


class CodebookTests(unittest.TestCase):
    def test_half_constants(self):
        # codebook.cuh: 0x1eee = 1/147.7, 0xc931 = -10.39
        self.assertAlmostEqual(h(0x1EEE), 0.0067672, places=6)
        self.assertAlmostEqual(h(0xC931), -10.3828, places=3)

    def test_lop3_truth_table(self):
        # SASS LOP3 with imm 0x6a has minterms {1,3,5,6} == c XOR (a AND b).
        r = exl3.codebook._lop3(np.array([0, 0, 0, 0, 1, 1, 1, 1], np.uint32),
                                np.array([0, 0, 1, 1, 0, 0, 1, 1], np.uint32),
                                np.array([0, 1, 0, 1, 0, 1, 0, 1], np.uint32),
                                _LOP3_IMM)
        got = tuple(int(v) for v in r)
        self.assertEqual(got, (0, 1, 0, 1, 0, 1, 1, 0))
        # the masks are the two others used by the codebooks
        self.assertEqual(_LOP3_MASK1, 0x8FFF8FFF)
        self.assertEqual(_LOP3_MASK2, 0x3B603B60)

    def test_mul1_value_range(self):
        # byte sum of x in [0, 1020]; h = half(0x6400 + sum) in [1024, 2047]; affine -> about [-3.46, 3.47]
        vals = _decode_mul1(np.arange(0, 1 << 16, dtype=np.uint32))
        self.assertLessEqual(float(vals.min()), -3.3)
        self.assertGreaterEqual(float(vals.max()), 3.3)
        self.assertLess(float(np.abs(vals).max()), 4.0)          # never explodes
        self.assertTrue(np.isfinite(vals.astype(np.float32)).all())
        # the endpoint of the pre-affine range: 0x6400 -> 1024.0
        self.assertEqual(h(0x6400), 1024.0)
        self.assertAlmostEqual(h(0x67FF), 2047.0, places=0)

    def test_mul1_matches_scalar_reference(self):
        inv = h(0x1EEE)
        bias = h(0xC931)
        for win in (0, 1, 2, 3, 65535, 0x1234, 0xABCD, 0x83DC):
            x = (win * 0x83DCD12D) & 0xFFFFFFFF
            s = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF) + 0x6400
            ref = np.float16(struct.unpack("<e", struct.pack("<H", s & 0xFFFF))[0] * inv + bias)
            got = _decode_mul1(np.array([win], np.uint32))[0]
            self.assertEqual(np.float16(got), ref, "mul1 window 0x%04x" % win)

    def test_codebooks_distinct_and_finite(self):
        win = np.arange(0, 1 << 16, dtype=np.uint32)
        outs = [exl3.decode(cb, win) for cb in (CB_3INST, CB_MCG, CB_MUL1)]
        for o in outs:
            self.assertTrue(np.isfinite(o.astype(np.float32)).all())
        self.assertFalse(np.array_equal(outs[0], outs[1]))
        self.assertFalse(np.array_equal(outs[1], outs[2]))
        with self.assertRaises(ValueError):
            exl3.decode(7, win)


class BitstreamTests(unittest.TestCase):
    @staticmethod
    def _bruteforce_windows(packed_u16, bits):
        """Independent reader: flatten to a bit list, then take 16-bit LE windows."""
        total = 256 * bits
        words = np.asarray(packed_u16, dtype=np.uint16).reshape(-1)
        bits_list = [(int(w) >> k) & 1 for w in words for k in range(16)]
        bits_list = bits_list[:total]
        out = []
        for t in range(256):
            start = (t * bits + bits - 16) % total
            v = 0
            for k in range(16):
                v |= bits_list[(start + k) % total] << k
            out.append(v)
        return np.array(out, dtype=np.uint32)

    def test_windows_match_bruteforce(self):
        rng = np.random.default_rng(1234)
        for bits in (1, 2, 3, 4, 5, 6, 7, 8):
            packed = rng.integers(0, 1 << 16, size=256 * bits // 16, dtype=np.uint16)
            got = exl3.windows_tile(packed, bits)
            want = self._bruteforce_windows(packed, bits)
            np.testing.assert_array_equal(got, want, "bits=%d" % bits)

    def test_unpack_tile_shape_and_order(self):
        packed = np.arange(48, dtype=np.uint16)          # bits=3 -> 48 words
        vals = exl3.unpack_tile(packed, 3, CB_MUL1)
        self.assertEqual(vals.shape, (256,))
        self.assertEqual(vals.dtype, np.float16)

    def test_bitrate_parsing(self):
        for K in (1.5, 2.5, 3.5):
            self.assertTrue(exl3.bits_from_K(K).half)
        self.assertEqual(exl3.bits_from_K(3.5).bits, 3)
        self.assertEqual(exl3.k2_from_K(3.5), 7)   # 2*3 + 1 half-bit
        # 3.05 is the model's *average* bpw, not a per-tensor K: bits_from_K is per tensor and rejects it
        with self.assertRaises(ValueError):
            exl3.bits_from_K(3.05)
        with self.assertRaises(ValueError):
            exl3.bits_from_K(9)


if __name__ == "__main__":
    unittest.main()
