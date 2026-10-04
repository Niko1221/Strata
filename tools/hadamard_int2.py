"""Reference codec for Strata's Hadamard-INT2 GGUF extension."""
from __future__ import annotations

import numpy as np

BLOCK_SIZE = 128
TYPE_ID = 144
TYPE_NAME = "HADAMARD_INT2"
BLOCK_BYTES = 34  # fp16 scale + 128 two-bit codes
CODEBOOK = np.asarray([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=np.float32)

_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB


def sign_vector(length: int, seed: int) -> np.ndarray:
    """Return SplitMix64 signs keyed by absolute input-column index."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if seed < 0 or seed > _MASK64:
        raise ValueError("seed must fit in an unsigned 64-bit integer")
    i = np.arange(length, dtype=np.uint64)
    z = np.uint64(seed) + (i + np.uint64(1)) * np.uint64(_GOLDEN)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(_MIX1)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(_MIX2)
    z ^= z >> np.uint64(31)
    return np.where((z & np.uint64(1)) != 0, 1.0, -1.0).astype(np.float32)


def rotate_rows(weights: np.ndarray, seed: int = 0) -> np.ndarray:
    """Apply normalized H(Dx) independently to each 128-value row block."""
    w = np.asarray(weights, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"weights must be a matrix, got {w.shape}")
    if w.shape[1] == 0 or w.shape[1] % BLOCK_SIZE:
        raise ValueError(f"input width must be a positive multiple of {BLOCK_SIZE}")
    if not np.isfinite(w).all():
        raise ValueError("weights contain NaN or infinity")

    rows, width = w.shape
    groups = width // BLOCK_SIZE
    y = np.array(w, dtype=np.float32, order="C", copy=True).reshape(rows, groups, BLOCK_SIZE)
    y *= sign_vector(width, seed).reshape(1, groups, BLOCK_SIZE)
    stride = 1
    while stride < BLOCK_SIZE:
        view = y.reshape(rows, groups, BLOCK_SIZE // (2 * stride), 2, stride)
        even = view[..., 0, :].copy()
        odd = view[..., 1, :].copy()
        view[..., 0, :] = even + odd
        view[..., 1, :] = even - odd
        stride *= 2
    y *= np.float32(1.0 / np.sqrt(BLOCK_SIZE))
    return y.reshape(rows, width)


def _nearest(groups: np.ndarray, scales: np.ndarray) -> np.ndarray:
    safe_scale = np.where(scales > 0.0, scales, 1.0)
    normalized = groups / safe_scale
    return np.argmin(np.abs(normalized[..., None] - CODEBOOK), axis=-1).astype(np.uint8)


def quantize_rows(
    weights: np.ndarray,
    *,
    seed: int = 0,
    importance: np.ndarray | None = None,
    scale_iters: int = 8,
) -> tuple[bytes, dict[str, float]]:
    """Rotate and quantize matrix rows into 34-byte blocks of 128 weights.

    ``importance`` contains non-negative activation second moments in the rotated
    input basis. One value per input channel is shared by every output row.
    """
    if scale_iters < 1:
        raise ValueError("scale_iters must be positive")
    rotated = rotate_rows(weights, seed)
    rows, width = rotated.shape
    n_groups = width // BLOCK_SIZE
    blocks = rotated.reshape(rows, n_groups, BLOCK_SIZE)

    if importance is None:
        omega = np.ones((1, n_groups, BLOCK_SIZE), dtype=np.float32)
    else:
        omega_flat = np.asarray(importance, dtype=np.float32)
        if omega_flat.shape != (width,):
            raise ValueError(f"importance must have shape ({width},), got {omega_flat.shape}")
        if not np.isfinite(omega_flat).all() or np.any(omega_flat < 0):
            raise ValueError("importance values must be finite and non-negative")
        omega = omega_flat.reshape(1, n_groups, BLOCK_SIZE)

    scales = np.max(np.abs(blocks), axis=-1, keepdims=True)
    for _ in range(scale_iters):
        codes = _nearest(blocks, scales)
        levels = CODEBOOK[codes]
        numerator = np.sum(omega * blocks * levels, axis=-1, keepdims=True)
        denominator = np.sum(omega * levels * levels, axis=-1, keepdims=True)
        scales = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        scales = np.maximum(scales, 0.0)

    scale16 = scales.astype("<f2")
    stored_scales = scale16.astype(np.float32)
    if not np.isfinite(stored_scales).all():
        raise ValueError("a block scale is outside the finite fp16 range")
    codes = _nearest(blocks, stored_scales)
    quantized = CODEBOOK[codes] * stored_scales

    packed = (
        codes[..., 0::4]
        | (codes[..., 1::4] << np.uint8(2))
        | (codes[..., 2::4] << np.uint8(4))
        | (codes[..., 3::4] << np.uint8(6))
    )
    encoded = np.empty((rows, n_groups, BLOCK_BYTES), dtype=np.uint8)
    encoded[..., :2] = scale16.view(np.uint8).reshape(rows, n_groups, 2)
    encoded[..., 2:] = packed.reshape(rows, n_groups, BLOCK_SIZE // 4)

    error = quantized - blocks
    unweighted_mse = float(np.mean(error * error))
    weighted_mse = float(np.sum(omega * error * error) / max(float(np.sum(omega)) * rows, 1e-30))
    signal_mse = float(np.mean(blocks * blocks))
    return encoded.tobytes(), {
        "weight_mse": unweighted_mse,
        "weight_nmse": unweighted_mse / max(signal_mse, 1e-30),
        "importance_weighted_mse": weighted_mse,
    }


def dequantize_blocks(raw: bytes, rows: int, width: int) -> np.ndarray:
    """Decode stored Hadamard-basis weights. The inverse basis is applied by the runtime."""
    if width <= 0 or width % BLOCK_SIZE:
        raise ValueError(f"width must be a positive multiple of {BLOCK_SIZE}")
    n_groups = width // BLOCK_SIZE
    expected = rows * n_groups * BLOCK_BYTES
    if len(raw) != expected:
        raise ValueError(f"expected {expected} bytes, got {len(raw)}")
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(rows, n_groups, BLOCK_BYTES)
    scales = blocks[..., :2].copy().view("<f2").astype(np.float32)[..., None]
    packed = blocks[..., 2:]
    codes = np.empty((rows, n_groups, BLOCK_SIZE), dtype=np.uint8)
    for lane in range(4):
        codes[..., lane::4] = (packed >> np.uint8(lane * 2)) & np.uint8(3)
    return (CODEBOOK[codes] * scales).reshape(rows, width)
