"""EXL3 tile permutation, Hadamard, and full weight reconstruction (numpy reference).

Transcribed from exllamav3 `modules/quant/exl3_lib/quantize.py` (tensor_core_perm),
`modules/quant/exl3.py` (get_weight_tensor), and the kernels `reconstruct.cu` / `hadamard.cu`.

The stored form is the rotated/quantized weight; the original-basis weight is

    W = diag(suh) . H128 . W_hat . H128 . diag(svh)

with H128 the natural-order Sylvester Hadamard scaled by 1/sqrt(128), applied in blocks of 128
along each dimension, and W_hat the decoded tile matrix.

The important consequence, and the main optimization, is that a GEMV never needs W at all:

    x W = (((x . suh) H128)) . W_hat . ((H128 . svh))
        = H128(x . suh)  @  W_hat  then  . H128  then  . svh

so only W_hat (the decoded tiles) has to be produced, on the fly, and the two Hamardard passes
run on the 1-row activation instead of the huge weight.  `folded_forward` below is that path;
`reconstruct_weight` is the materialized path used for prefill (reconstruct + GEMM).
"""
from __future__ import annotations

import numpy as np

from .codebook import decode_tiles_lut

HAD_BLOCK = 128


def tile_perm() -> np.ndarray:
    """tensor_core_perm: `tc[j] == row_major[perm[j]]` (index into the row-major 16x16 tile)."""
    perm = np.zeros(256, dtype=np.int64)
    for t in range(32):
        r0 = (t % 4) * 2
        r1, r2, r3 = r0 + 1, r0 + 8, r0 + 9
        c0 = t // 4
        c1 = c0 + 8
        perm[t * 8 + 0] = r0 * 16 + c0
        perm[t * 8 + 1] = r1 * 16 + c0
        perm[t * 8 + 2] = r2 * 16 + c0
        perm[t * 8 + 3] = r3 * 16 + c0
        perm[t * 8 + 4] = r0 * 16 + c1
        perm[t * 8 + 5] = r1 * 16 + c1
        perm[t * 8 + 6] = r2 * 16 + c1
        perm[t * 8 + 7] = r3 * 16 + c1
    return perm


def tile_perm_inv() -> np.ndarray:
    """argsort(tile_perm): map stored TC order back to row-major, `rm = tc[perm_inv]`."""
    return np.argsort(tile_perm())


def had128() -> np.ndarray:
    """Natural-order Sylvester Hadamard, scaled by 1/sqrt(128): H[i,j] = (-1)^popcount(i&j)/sqrt(128)."""
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < HAD_BLOCK:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(float(HAD_BLOCK))).astype(np.float32)


def _had_cols(m: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Apply h to blocks of 128 consecutive columns (the `_r` transform)."""
    rows, cols = m.shape
    m = m.reshape(rows, cols // HAD_BLOCK, HAD_BLOCK)
    m = np.einsum("bik,kj->bij", m.astype(np.float32), h)
    return m.reshape(rows, cols)


def _had_rows(m: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Apply h to blocks of 128 consecutive rows (the `_l` transform)."""
    rows, cols = m.shape
    m = m.reshape(rows // HAD_BLOCK, HAD_BLOCK, cols)
    m = np.einsum("ij,bjn->bin", h, m.astype(np.float32))
    return m.reshape(rows, cols)


def decode_weight_hat(trellis, cb: int) -> np.ndarray:
    """Undo the tile permutation: return W_hat (in x out), the decoded tiles in row-major order.

    `trellis` is [in/16, out/16, 256*K/16] int16; K is inferred from the last dimension.
    """
    tr = np.asarray(trellis)
    ki, nj, words = tr.shape
    bits = words * 16 // 256
    if bits < 1 or words * 16 != 256 * bits:
        raise ValueError("bad trellis tail dim %d (not 256*K/16)" % words)
    perm_i = tile_perm_inv()
    # Decode all tiles in one batch, then un-permute each 16x16 tile to row-major.
    flat = decode_tiles_lut(cb, tr.reshape(ki * nj, words), bits)   # (ki*nj, 256) TC order
    rm = flat[:, perm_i].reshape(ki, nj, 16, 16)                    # row-major tiles
    out = rm.transpose(0, 2, 1, 3).reshape(ki * 16, nj * 16)
    return out.astype(np.float16)


def weight_from_what(w_hat, suh, svh) -> np.ndarray:
    """The core algebra: W = diag(suh) . H . w_hat . H . diag(svh), in fp32.

    Kept separate from the decode so the Hadamard/sign algebra can be tested against the folded
    form without depending on the codebook.
    """
    h = had128()
    w = _had_rows(np.asarray(w_hat, dtype=np.float32), h)
    w = _had_cols(w, h)
    w = w * np.asarray(suh, dtype=np.float32)[:, None]
    w = w * np.asarray(svh, dtype=np.float32)[None, :]
    return w


def folded_from_what(x, w_hat, suh, svh) -> np.ndarray:
    """The same product without building W: y = H(x . suh) @ w_hat . H . svh."""
    x = np.asarray(x, dtype=np.float32)
    h = had128()
    xh = _had_cols(x * np.asarray(suh, dtype=np.float32)[None, :], h)
    z = xh @ np.asarray(w_hat, dtype=np.float32)
    return _had_cols(z, h) * np.asarray(svh, dtype=np.float32)[None, :]


def reconstruct_weight(trellis, suh, svh, cb: int) -> np.ndarray:
    """Materialize the original-basis weight W = diag(suh) . H . W_hat . H . diag(svh) (fp16)."""
    return weight_from_what(decode_weight_hat(trellis, cb), suh, svh).astype(np.float16)


def folded_forward(x, trellis, suh, svh, cb: int) -> np.ndarray:
    """The decode-time GEMV without materializing W: y = H(x . suh) @ W_hat . H . svh.

    `x` is (tokens, in).  This must equal `x @ reconstruct_weight(...)` up to fp16 rounding; the
    test checks exactly that, which pins the sign/Hadamard algebra of both paths.
    """
    return folded_from_what(x, decode_weight_hat(trellis, cb), suh, svh)
