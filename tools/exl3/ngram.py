"""EXL3 n-gram table codec (the `exl3_ngram_trellis` format), numpy reference.

Mirrors turboderp-org/exllamav3 `exllamav3/modules/quant/exl3_lib/ngram_codec.py`.

An n-gram row is a single 160-wide tail-biting trellis "ring" over the mul1 codebook plus an fp16
scale, packed as `1 + dim*K/16` little-endian uint16 words: word 0 is the scale's bit pattern, the
rest are the `dim*K`-bit ring.  Stream bits `[i*K, (i+1)*K)` are the low K bits of position i's
16-bit state; the state's higher bits are the preceding positions' symbols, K bits each (mod dim).
A row decodes to `codebook[state_i] * scale + head_bias[head]` (no Hadamard; only token-embedding
groups rotate).

This is a different geometry from the per-linear 16x16 tiles, so it has its own decoder.
"""
from __future__ import annotations

import numpy as np

from .codebook import CB_MUL1, codebook_lut

ROW_DIM = 160
HEAD_BIAS_DIM = 160


def words_per_row(K: int, dim: int = ROW_DIM) -> int:
    return 1 + dim * K // 16


def ring_dim(words: int, K: int) -> int:
    return (words - 1) * 16 // K


def mul1_codebook() -> np.ndarray:
    """All 65536 decoded mul1 values as fp16 (== ExLlamaV3's ngram_codec.mul1_codebook)."""
    return codebook_lut(CB_MUL1)


def unpack_row(packed, K: int) -> tuple[np.ndarray, np.float16]:
    """One packed ring -> (states (dim,) uint16, scale fp16)."""
    packed = np.asarray(packed, dtype=np.int16).reshape(-1)
    words = packed.shape[0] - 1
    dim = ring_dim(packed.shape[0], K)
    scale = np.array([packed[0]], dtype=np.int16).view(np.float16)[0]
    stream = packed[1:].view(np.uint16).astype(np.int64)
    bit = np.arange(dim, dtype=np.int64) * K
    i0 = bit >> 4
    i1 = ((bit >> 4) + 1) % words
    window = stream[i0] | (stream[i1] << 16)
    symbols = (window >> (bit & 15)) & ((1 << K) - 1)
    states = np.zeros(dim, dtype=np.int64)
    for j in range((15 + K) // K):
        states |= np.roll(symbols, j) << (j * K)
    return (states & 0xFFFF).astype(np.uint16), scale


def pack_row(states, scale, K: int) -> np.ndarray:
    """Inverse of unpack_row (states (dim,) -> packed ring), for tests and re-packing."""
    states = np.asarray(states, dtype=np.int64).reshape(-1)
    dim = states.shape[0]
    new_bits = states & ((1 << K) - 1)                          # (dim,)
    bits = (new_bits[:, None] >> np.arange(K, dtype=np.int64)) & 1   # (dim, K)
    words = bits.reshape(dim * K // 16, 16)
    stream = (words << np.arange(16, dtype=np.int64)).sum(-1).astype(np.uint16)
    scale_word = np.array([np.float16(scale)], dtype=np.float16).view(np.int16)
    return np.concatenate([scale_word, stream.view(np.int16)]).astype(np.int16)


def dequant_row(packed, K: int, codebook: np.ndarray, bias=None) -> np.ndarray:
    """One packed ring -> (dim,) float32, with optional per-row bias (already selected by head)."""
    states, scale = unpack_row(packed, K)
    out = codebook[states].astype(np.float32) * np.float32(scale)
    if bias is not None:
        out = out + np.asarray(bias, dtype=np.float32)
    return out


def head_of_row(row: int, head_offsets) -> int:
    """The hash head a global row belongs to (head_offsets is the running sum of vocab sizes)."""
    offs = np.asarray(head_offsets)
    return int(np.searchsorted(offs, row, side="right") - 1)


def decode_table_row(row: int, ring: np.ndarray, K: int, codebook: np.ndarray,
                     head_bias, head_offsets) -> np.ndarray:
    """Decode global `row` from its packed `ring`, adding its head's bias."""
    bias = None if head_bias is None else head_bias[head_of_row(row, head_offsets)]
    return dequant_row(ring, K, codebook, bias)
