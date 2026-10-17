"""CPU-only CSF byte round trips and malformed-stream checks.

    python tools/test_nvfp4_csf.py
"""
from __future__ import annotations

import unittest

import numpy as np
from nvfp4_csf import decode_scale_bytes, pack_expert


def encode_reference(raw):
    """Simple per-row encoder, independent of the vectorized decoder."""
    rows, cols = raw.shape
    slabs, exceptions = [], []
    for start in range(0, rows, 16):
        bases, packed = [], []
        for row in range(start, start + 16):
            base = min(int(raw[row].min()), 240)
            bases.append(base)
            codes = []
            for col, value in enumerate(raw[row]):
                delta = int(value) - base
                if not 0 <= delta <= 15:
                    exceptions.append((int(value) << 24) | (row * cols + col))
                    delta = 0
                codes.append(delta)
            packed.extend(codes[i] | (codes[i + 1] << 4) for i in range(0, cols, 2))
        slabs.append(bases + packed)
    return np.asarray(slabs, dtype=np.uint8), np.asarray(exceptions, dtype='<u4')


class Codec(unittest.TestCase):
    def test_shapes_values_and_exceptions(self):
        rng = np.random.default_rng(123)
        for shape in [(16, 4), (640, 160), (2560, 40)]:
            for outliers in [False, True]:
                with self.subTest(shape=shape, outliers=outliers):
                    raw = rng.integers(48, 64, shape, dtype=np.uint8)
                    if outliers:
                        raw.flat[::23] = 126
                    fixed, exceptions = encode_reference(raw)
                    np.testing.assert_array_equal(decode_scale_bytes(fixed, exceptions), raw)

    def test_full_byte_range_is_preserved(self):
        raw = np.arange(256, dtype=np.uint8).reshape(16, 16)
        fixed, exceptions = encode_reference(raw)
        np.testing.assert_array_equal(decode_scale_bytes(fixed, exceptions), raw)

    def test_invalid_streams_fail(self):
        fixed, exceptions = encode_reference(np.full((16, 4), 56, dtype=np.uint8))
        bad_base = fixed.copy()
        bad_base[0, 0] = 241
        cases = [(fixed.astype(np.int16), exceptions), (fixed[:, :-1], exceptions),
                 (fixed[:, :16], exceptions), (fixed, exceptions.astype(np.int64)),
                 (bad_base, exceptions), (fixed, np.array([64], dtype='<u4')),
                 (fixed, np.array([2, 2], dtype='<u4')),
                 (fixed, np.array([2, 1], dtype='<u4'))]
        for a, b in cases:
            with self.subTest(shape=a.shape, dtype=b.dtype), self.assertRaises(ValueError):
                decode_scale_bytes(a, b)

    def test_superblock_nibble_order_and_scale_bytes(self):
        weight = np.tile(np.array([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=np.uint8), (1, 4))
        scales = np.array([[16, 32, 48, 64]], dtype=np.uint8)
        packed = pack_expert(weight, scales)
        expected_codes = [0x80, 0x91, 0xA2, 0xB3, 0xC4, 0xD5, 0xE6, 0xF7] * 4
        np.testing.assert_array_equal(packed, np.array([[16, 32, 48, 64, *expected_codes]], dtype=np.uint8))

    def test_packer_rejects_nan_and_negative_scales(self):
        weight = np.zeros((1, 32), dtype=np.uint8)
        for value in [127, 128, 255]:
            with self.assertRaises(ValueError):
                pack_expert(weight, np.full((1, 4), value, dtype=np.uint8))


if __name__ == '__main__':
    unittest.main()
