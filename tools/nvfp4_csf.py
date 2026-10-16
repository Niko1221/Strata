"""CPU decoding for CSF NVFP4 block-scale streams (experimental).

Format: byte-window4-fixed-stream-u24-exceptions/1, described in
https://github.com/vllm-project/vllm/pull/60646 . Each 16-row slab starts
with 16 base bytes, followed by row-major, low-nibble-first 4-bit offsets.
Exception words store a 24-bit flattened position and an 8-bit replacement.
This module reconstructs bytes; it does not change quantization or kernels.
"""
from __future__ import annotations

import numpy as np


def decode_scale_bytes(fixed: np.ndarray, exceptions: np.ndarray) -> np.ndarray:
    """Return the original [rows, cols] E4M3 scale bytes, without rounding."""
    if fixed.dtype != np.dtype('uint8') or fixed.ndim != 2:
        raise ValueError('CSF fixed stream must be a 2D uint8 tensor')
    slabs, width = fixed.shape
    if not slabs or width <= 16 or width % 16:
        raise ValueError('CSF fixed stream has invalid slab dimensions')
    if exceptions.dtype != np.dtype('<u4') or exceptions.ndim != 1:
        raise ValueError('CSF exceptions must be a 1D little-endian uint32 tensor')
    rows, cols = slabs * 16, (width // 16 - 1) * 2
    bases = fixed[:, :16].reshape(rows, 1)
    if np.any(bases > 240):
        raise ValueError('CSF row bases must be at most 240')
    packed = fixed[:, 16:].reshape(rows, cols // 2)
    offsets = np.empty((rows, cols), dtype=np.uint8)
    offsets[:, 0::2] = packed & 15
    offsets[:, 1::2] = packed >> 4
    result = bases + offsets
    positions = exceptions & np.uint32(0xFFFFFF)
    if positions.size:
        if np.any(positions >= result.size):
            raise ValueError('CSF exception position is outside the scale matrix')
        if np.any(positions[1:] <= positions[:-1]):
            raise ValueError('CSF exception positions must be strictly increasing')
        result.reshape(-1)[positions] = (exceptions >> 24).astype(np.uint8)
    return result


def pack_expert(weight: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Repack ModelOpt's NVFP4 bytes into ggml's 36-byte superblocks.

    The per-expert FP32 weight_scale_2 is kept separately by Strata's pack.
    """
    if weight.dtype != np.uint8 or scales.dtype != np.uint8:
        raise ValueError('NVFP4 codes and scale bits must be uint8')
    if weight.ndim != 2 or scales.ndim != 2:
        raise ValueError('NVFP4 weights and scales must be matrices')
    rows, blocks = scales.shape
    if blocks % 4 or weight.shape != (rows, blocks * 8):
        raise ValueError('NVFP4 shape must have one scale per 16 values and 64-value superblocks')
    if np.any(scales >= 127):
        raise ValueError('NVFP4 scale must be finite and non-negative E4M3')
    source = weight.reshape(rows, blocks, 8)
    low, high = source & 15, source >> 4
    values = np.stack((low, high), axis=-1).reshape(rows, blocks, 16)
    codes = values[:, :, :8] | (values[:, :, 8:] << 4)
    return np.concatenate((scales.reshape(rows, blocks // 4, 4),
                           codes.reshape(rows, blocks // 4, 32)), axis=-1).reshape(rows, -1)


def e4m3_values(bits: np.ndarray) -> np.ndarray:
    """Decode E4M3FN bits into float32, including subnormals and signed zero."""
    bits = np.asarray(bits, dtype=np.uint8)
    exponent = ((bits >> 3) & 15).astype(np.int32)
    mantissa = (bits & 7).astype(np.float32)
    values = np.where(exponent == 0, mantissa * np.float32(2**-9),
                      (1 + mantissa / 8) * np.exp2(exponent - 7)).astype(np.float32)
    values = np.copysign(values, np.where(bits & 128, -1., 1.))
    return np.where((bits & 127) == 127, np.nan, values).astype(np.float32)


def dequant_nvfp4(weight: np.ndarray, scales: np.ndarray, scale2: float) -> np.ndarray:
    """ModelOpt NVFP4 definition: E2M1 code x E4M3 block scale x global scale."""
    if weight.dtype != np.uint8 or scales.dtype != np.uint8 or weight.ndim != 2 or scales.ndim != 2:
        raise ValueError('NVFP4 weights and scale bits must be 2D uint8 tensors')
    if scales.shape != (weight.shape[0], weight.shape[1] // 8) or weight.shape[1] % 8:
        raise ValueError('NVFP4 needs one block scale per 16 unpacked values')
    if np.any(scales >= 127) or not np.isfinite(scale2) or scale2 <= 0:
        raise ValueError('NVFP4 scales must be finite and non-negative; global scale must be positive')
    values = np.empty((weight.shape[0], weight.shape[1] * 2), dtype=np.uint8)
    values[:, 0::2], values[:, 1::2] = weight & 15, weight >> 4
    lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
    return lut[values] * np.repeat(e4m3_values(scales), 16, axis=1) * np.float32(scale2)


def dequant_mxfp8(weight: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """E4M3 values with an unsigned E8M0 scale per 32 consecutive values."""
    if weight.dtype != np.uint8 or scales.dtype != np.uint8 or weight.ndim != 2 or scales.ndim != 2:
        raise ValueError('MXFP8 weights and scale bits must be 2D uint8 tensors')
    if weight.shape[1] % 32 or scales.shape != (weight.shape[0], weight.shape[1] // 32):
        raise ValueError('MXFP8 needs one E8M0 scale per 32 values')
    if np.any(scales == 255):
        raise ValueError('MXFP8 scale has a NaN encoding')
    with np.errstate(over='ignore', invalid='ignore'):
        result = e4m3_values(weight) * np.repeat(np.exp2(scales.astype(np.float32) - 127), 32, axis=1)
    if not np.isfinite(result).all():
        raise ValueError('MXFP8 dequantization produced a non-finite value')
    return result
