"""EXL3 procedural codebook and 16-bit sliding-window dequantization, in numpy.

Transcribed from turboderp-org/exllamav3, `exllamav3/exllamav3_ext/quant/codebook.cuh` and
`exl3_dq.cuh`.  Every function here is meant to be bit-exact with the CUDA device code, including
its fp16 rounding, so the same tests can be pointed at a kernel later.  The GPU path is:

    window_t = 16 bits of the packed stream starting at bit  (t*bits + bits - 16)  (mod 256*bits)
    value_t  = codebook_cb(window_t)

i.e. the "trellis" is nothing more than heavily overlapping 16-bit windows over a little-endian
bitstream of 256*K bits per 16x16 tile, and the procedural codebook maps each 16-bit window to one
weight.  (The encoder's Viterbi search is what makes the overlap useful; decoding is just this.)

Codebook ids follow ExLlamaV3: 0 = 3inst, 1 = mcg (0xCBAC1FED), 2 = mul1 (0x83DCD12D).
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

CB_3INST = 0
CB_MCG = 1
CB_MUL1 = 2

# mask/multiplier constants, from codebook.cuh
_MCG_MULT = 0xCBAC1FED
_MUL1_MULT = 0x83DCD12D
_3INST_MULT = 89226354
_3INST_ADD = 64248484
_LOP3_MASK1 = 0x8FFF8FFF
_LOP3_MASK2 = 0x3B603B60
_LOP3_IMM = 0x6A

# mul1 affine, half(0x1eee) = 1/147.7 and half(0xc931) = -10.39 (see the comment in codebook.cuh)
_MUL1_INV_BITS = 0x1EEE
_MUL1_BIAS_BITS = 0xC931
_MUL1_ACC = 0x6400                          # 0x6400 -> 1024.0 .. 0x67FF -> 2047.0


@dataclass(frozen=True)
class BitsK:
    bits: int
    half: bool


def bits_from_K(K: float) -> BitsK:
    """Split an EXL3 bitrate K (float) into integer bits and the half-rate flag.

    Integer 1..8, or 1.5 / 2.5 / 3.5 (mul1 only).  Mirrors BitsK::bits_from_K in bits_k.cuh.
    """
    bits = int(K)
    frac = float(K) - bits
    if not (1 <= bits <= 8) or not (frac == 0.0 or (frac == 0.5 and bits <= 3)):
        raise ValueError("unsupported EXL3 bitrate %r (integer 1..8, or 1.5/2.5/3.5)" % (K,))
    return BitsK(bits, frac == 0.5)


def k2_from_K(K: float) -> int:
    """Half-bit units (2*bits + half), for runtime switches over mixed bitrates."""
    b = bits_from_K(K)
    return 2 * b.bits + (1 if b.half else 0)


def _u32(x) -> np.ndarray:
    """Wrap to uint32 (numpy would otherwise upcast the product to int64)."""
    return np.asarray(x, dtype=np.uint64).astype(np.uint32)


def _half_from_bits(bits) -> np.ndarray:
    """Reinterpret uint16 bit patterns as IEEE half (fp16)."""
    return np.asarray(bits, dtype=np.uint16).view(np.float16)


def _half_bits_of(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float16).view(np.uint16)


def _lop3(a, b, c, imm: int) -> np.ndarray:
    """Bitwise ternary (a,b,c) selected by the 8-bit truth table `imm`, as SASS LOP3."""
    a = _u32(a)
    b = _u32(b)
    c = _u32(c)
    r = np.zeros(a.shape, dtype=np.uint32)
    for i in range(8):
        ta = a if (i >> 2) & 1 else ~a
        tb = b if (i >> 1) & 1 else ~b
        tc = c if (i >> 0) & 1 else ~c
        if (imm >> i) & 1:
            r |= (ta & tb & tc)
    return r


def _decode_3inst(x) -> np.ndarray:
    x = _u32(_u32(x) * np.uint32(_3INST_MULT) + np.uint32(_3INST_ADD))
    x = _lop3(x, _LOP3_MASK1, _LOP3_MASK2, _LOP3_IMM)
    lo = _half_from_bits((x & 0xFFFF).astype(np.uint16))
    hi = _half_from_bits((x >> 16).astype(np.uint16))
    return (lo + hi).astype(np.float16)


def _decode_mcg(x) -> np.ndarray:
    x = _u32(_u32(x) * np.uint32(_MCG_MULT))
    x = _lop3(x, _LOP3_MASK1, _LOP3_MASK2, _LOP3_IMM)
    lo = _half_from_bits((x & 0xFFFF).astype(np.uint16))
    hi = _half_from_bits((x >> 16).astype(np.uint16))
    return (lo + hi).astype(np.float16)


def _decode_mul1(x) -> np.ndarray:
    x = _u32(_u32(x) * np.uint32(_MUL1_MULT))
    byte_sum = ((x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)
                + np.uint32(_MUL1_ACC))
    h = _half_from_bits((byte_sum & 0xFFFF).astype(np.uint16)).astype(np.float64)
    inv = np.float64(_half_from_bits(np.uint16(_MUL1_INV_BITS)))
    bias = np.float64(_half_from_bits(np.uint16(_MUL1_BIAS_BITS)))
    # __hfma in fp16 = one rounding of the exact product+addend; float64 then one rounding matches it
    return (h * inv + bias).astype(np.float16)


_DECODERS = {CB_3INST: _decode_3inst, CB_MCG: _decode_mcg, CB_MUL1: _decode_mul1}


def decode(cb: int, window) -> np.ndarray:
    """Map 16-bit window value(s) to fp16 weight(s) for codebook `cb`."""
    try:
        fn = _DECODERS[cb]
    except KeyError:
        raise ValueError("unknown codebook %r (0=3inst, 1=mcg, 2=mul1)" % (cb,))
    return fn(window)


def _bitstream(packed) -> np.ndarray:
    """View a little-endian uint16 word array as a flat byte array (its exact memory order)."""
    return np.ascontiguousarray(np.asarray(packed, dtype="<u2")).view(np.uint8)


def _read_bits(data: np.ndarray, start_bit: int, nbits: int, total_bits: int) -> int:
    v = 0
    for k in range(nbits):
        bit = (start_bit + k) % total_bits
        v |= ((int(data[bit >> 3]) >> (bit & 7)) & 1) << k
    return v


def windows_tile(packed, bits: int) -> np.ndarray:
    """The 256 16-bit overlapping windows of one tile's packed trellis, in stored (TC) order.

    packed holds 256*bits bits as 256*bits/16 little-endian uint16 words.  Window t starts at bit
    (t*bits + bits - 16) mod 256*bits and wraps around the tile (tail-biting).  `bits` is the
    integer bitrate; half rates are handled separately.
    """
    packed = np.asarray(packed, dtype="<u2").reshape(-1)
    total_bits = 256 * bits
    if packed.size * 16 < total_bits:
        raise ValueError("packed tile too short: %d words for bits=%d" % (packed.size, bits))
    data = _bitstream(packed)
    starts = [(t * bits + bits - 16) % total_bits for t in range(256)]
    return np.fromiter((_read_bits(data, s, 16, total_bits) for s in starts),
                       dtype=np.uint32, count=256)


def unpack_tile(packed, bits: int, cb: int) -> np.ndarray:
    """Decode one 16x16 tile from its packed trellis to 256 fp16, in stored (TC-permuted) order.

    The row/column permutation back to row-major is undone by the reconstruct kernel; see
    docs/EXL3.md and the later tile-perm step.
    """
    return decode(cb, windows_tile(packed, bits))


def windows_tile_fast(packed, bits: int) -> np.ndarray:
    """Vectorized `windows_tile`: expand the tile to a bit array, then 16 gather-shifts.

    Same result as `windows_tile`, but ~100x faster (a batch of the 256 window reads as numpy
    operations instead of a per-bit Python loop).  This is the shape a CPU inner loop should take.
    """
    packed = np.asarray(packed, dtype="<u2").reshape(-1)
    total_bits = 256 * bits
    if packed.size * 16 < total_bits:
        raise ValueError("packed tile too short: %d words for bits=%d" % (packed.size, bits))
    bits_arr = np.zeros(packed.size * 16, dtype=np.uint8)
    for k in range(16):                                   # LSB-first bit order within each word
        bits_arr[k::16] = ((packed >> k) & 1).astype(np.uint8)
    bits_arr = bits_arr[:total_bits]
    doubled = np.concatenate([bits_arr, bits_arr])        # tail-biting wrap
    starts = ((np.arange(256, dtype=np.int64) * bits + bits - 16) % total_bits)
    out = np.zeros(256, dtype=np.uint32)
    for k in range(16):
        out |= (doubled[starts + k].astype(np.uint32) << k)
    return out


def codebook_lut(cb: int) -> np.ndarray:
    """A 65536-entry fp16 lookup table for a codebook: `lut[window] == decode(cb, window)`.

    128 KB, fits L2/constant memory.  Replacing the per-window multiply/dp4a chain with one indexed
    load is the single biggest decode optimization on both CPU and GPU.  Cached: building it calls
    `decode` over all 65536 windows, which must happen once, not per tile.
    """
    return _cached_lut(cb)


@functools.lru_cache(maxsize=None)
def _cached_lut(cb: int) -> np.ndarray:
    return decode(cb, np.arange(1 << 16, dtype=np.uint32))


def decode_lut(cb: int, packed, bits: int) -> np.ndarray:
    """Optimized `unpack_tile`: vectorized windows + cached table lookup."""
    return codebook_lut(cb)[windows_tile_fast(packed, bits)]


def windows_tiles_fast(packed2d, bits: int) -> np.ndarray:
    """Batch of `windows_tile_fast` over T tiles: `packed2d` is (T, 256*bits/16), returns (T, 256)."""
    packed2d = np.asarray(packed2d, dtype="<u2")
    if packed2d.ndim == 1:
        packed2d = packed2d[None, :]
    tiles, words = packed2d.shape
    total_bits = 256 * bits
    if words * 16 < total_bits:
        raise ValueError("packed tile too short: %d words for bits=%d" % (words, bits))
    bits_arr = np.zeros((tiles, words * 16), dtype=np.uint8)
    for k in range(16):
        bits_arr[:, k::16] = ((packed2d >> k) & 1).astype(np.uint8)
    bits_arr = bits_arr[:, :total_bits]
    doubled = np.concatenate([bits_arr, bits_arr], axis=1)
    starts = ((np.arange(256, dtype=np.int64) * bits + bits - 16) % total_bits)
    out = np.zeros((tiles, 256), dtype=np.uint32)
    for k in range(16):
        out |= (doubled[:, starts + k].astype(np.uint32) << k)
    return out


def decode_tiles_lut(cb: int, packed2d, bits: int) -> np.ndarray:
    """Decode T tiles at once: (T, 256*bits/16) packed -> (T, 256) fp16."""
    return codebook_lut(cb)[windows_tiles_fast(packed2d, bits)]
