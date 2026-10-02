"""Requantize per_layer_token_embd.weight (Q5_1) to IQ4_NL, in place as a new shard.

Why: the engine's PLE reader hard-requires IQ4_NL (src/kernels/ngram.cpp: "not IQ4_NL"),
and this model ships the table as Q5_1.  The encoder is a line-by-line port of
ggml's quantize_row_iq4_nl_impl (third_party/llama.cpp/ggml/src/ggml-quants.c:4966)
with ntry=7, no quant_weights, super_block_size == block_size == 32 -- so the
single-scale branch applies: dh[0] = fp16(scale), L recomputed with 1/scale (f32).
best_index_int8 tie-break (strictly-closer-left wins) is reproduced exactly.
Q5_1 dequant follows dequantize_row_q5_1 (ggml-quants.c): d,m fp16; low half =
low nibble | bit j of qh << 4; high half = high nibble | bit (j+16) of qh << 4.

The output file is the source shard's header with the tensor type field patched
Q5_1 -> IQ4_NL (same size, 4 bytes) followed by the re-encoded data, so the
engine's "the table alone fills its shard exactly" check (ngram.cpp) holds:
320001536 rows x 90 B = 28,800,138,240 data bytes.

Rows are the unit of work: the table is [160, 320001536] with 160 on the fast
axis, i.e. 320001536 contiguous rows of 160 values = 5 IQ4_NL blocks each.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import struct
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile  # noqa: E402

Q5_1 = 7          # ggml type id
IQ4_NL = 20       # ggml type id
Q5_1_BLOCK = 24   # d(2) + m(2) + qh(4) + qs(16), 32 values
IQ4_NL_BLOCK = 18 # d(2) + qs(16), 32 values
ROW_VALUES = 160  # fast axis; 5 blocks per row
VALUES_PER_ROW = ROW_VALUES
SRC_ROW_BYTES = ROW_VALUES // 32 * Q5_1_BLOCK    # 120
DST_ROW_BYTES = ROW_VALUES // 32 * IQ4_NL_BLOCK  # 90

KV_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                     1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)
GROUP_MAX_EPS = 1e-15  # ggml-quants.c:20
NTRY = 7               # quantize_iq4_nl passes ntry=7
# thresholds between adjacent codebook entries: strictly below -> left index
def best_index(al: np.ndarray) -> np.ndarray:
    """Vectorized best_index_int8(16, kvalues_iq4nl, x) (ggml-quants.c:28).

    searchsorted(side='right') gives mu with val[mu-1] <= x < val[mu], matching
    the C binary search invariant; the strict < tie-break then mirrors the C.
    Benchmarked faster than a 4-step midpoint search on this machine.
    """
    i = np.searchsorted(KV_IQ4NL, al.ravel(), side="right")
    np.clip(i, 1, 15, out=i)
    af = al.ravel()
    i -= (af - KV_IQ4NL[i - 1]) < (KV_IQ4NL[i] - af)
    return i.reshape(al.shape)


def dequant_q5_1(blk: np.ndarray) -> np.ndarray:
    """(nb, 24) raw blocks -> (nb, 32) float32 values."""
    d = blk[:, 0:2].copy().view(np.float16).astype(np.float32)
    m = blk[:, 2:4].copy().view(np.float16).astype(np.float32)
    qh = blk[:, 4:8].copy().view("<u4").astype(np.uint32)  # (nb, 1)
    ql = blk[:, 8:24]                                       # (nb, 16)
    bits = np.arange(16, dtype=np.uint32)
    lo = (ql & 0x0F).astype(np.uint32) | (((qh >> bits) & 1) << np.uint32(4))
    hi = (ql >> 4).astype(np.uint32) | (((qh >> (bits + 16)) & 1) << np.uint32(4))
    q = np.concatenate([lo, hi], axis=1).astype(np.float32)
    return q * d + m


def quantize_iq4_nl(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(nb, 32) float32 -> (L (nb,32) uint8, dh (nb,) float16), ggml-faithful.

    Candidate scales, in the C order: initial d0 = -max/values[0] (positive,
    values[0] = -127), then d_itry = max/(itry - 127) for itry in -7..7 (the
    id = (itry + values[0])/max form -- all negative).  A candidate wins when
    sumqx^2/sumq2 strictly beats the incumbent (the C test
    `sumqx*sumqx > best*sumq2` with best = d*sumqx of the incumbent).
    """
    nb = x.shape[0]
    w = x * x  # weight[j] = xb[j]*xb[j] (no quant_weights)
    amax = np.max(np.abs(x), axis=1)
    maxv = np.take_along_axis(x, np.argmax(np.abs(x), axis=1)[:, None], axis=1)[:, 0]
    live = amax >= GROUP_MAX_EPS
    max_safe = np.where(live, maxv, 1.0).astype(np.float32)

    al = np.empty_like(x)
    wx = w * x  # invariant across candidates: sumqx = sum(wx * q)
    best_d = np.zeros(nb, dtype=np.float32)
    best_g = np.zeros(nb, dtype=np.float32)

    def evaluate(idm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        np.multiply(idm[:, None], x, out=al)
        q = KV_IQ4NL[best_index(al)]
        sumqx = np.sum(wx * q, axis=1, dtype=np.float32)
        sumq2 = np.sum(w * q * q, axis=1, dtype=np.float32)
        d = np.divide(sumqx, sumq2, out=np.zeros(nb, np.float32), where=sumq2 > 0)
        g = np.where(sumq2 > 0, d * sumqx, np.float32(0))
        return d, g

    # initial: id0 = 1/d0 = -values[0]/max = 127/max  (d0 = -max/values[0])
    d, g = evaluate(np.where(live, 127.0 / max_safe, 0.0).astype(np.float32))
    best_d = np.where(live, d, 0.0).astype(np.float32)
    best_g = np.where(live, g, np.float32(0))
    for itry in range(-NTRY, NTRY + 1):
        d, g = evaluate(np.where(live, (itry - 127) / max_safe, 0.0).astype(np.float32))
        better = live & (g > best_g)
        best_d = np.where(better, d, best_d).astype(np.float32)
        best_g = np.where(better, g, best_g)

    dh = best_d.astype(np.float16)
    with np.errstate(divide="ignore", invalid="ignore"):
        idf = np.where(best_d != 0, 1.0 / best_d, 0.0).astype(np.float32)
    L = best_index(idf[:, None] * x).astype(np.uint8)  # (nb, 32)
    return L, dh


def pack_blocks(L: np.ndarray, dh: np.ndarray) -> bytes:
    qs = L[:, :16] | (L[:, 16:] << 4)  # low nibbles first, high second
    out = np.empty((L.shape[0], IQ4_NL_BLOCK), dtype=np.uint8)
    out[:, 0:2] = dh.view(np.uint8).reshape(-1, 2)
    out[:, 2:18] = qs
    return out.tobytes()


def patch_header(src: pathlib.Path) -> bytes:
    """The source shard's header with the tensor type patched to IQ4_NL."""
    f = GGUFFile(src)
    t = f.tensors[0]
    assert len(f.tensors) == 1, "shard 2 is expected to hold exactly one tensor"
    assert t.name == "per_layer_token_embd.weight", t.name
    assert list(t.shape) == [ROW_VALUES, 320001536], t.shape
    assert t.type_id == Q5_1, t.type_name
    header = src.read_bytes()[: f.data_start]
    # tensor info entry: u64 name_len | name | u32 n_dims | dims | u32 type | u64 offset
    name = t.name.encode()
    probe = struct.pack("<Q", len(name)) + name + struct.pack("<I", 2)
    pos = header.find(probe)
    assert pos >= 0, "tensor info entry not found in header"
    type_pos = pos + len(probe) + 16  # + two u64 dims
    (dtype,) = struct.unpack_from("<I", header, type_pos)
    assert dtype == Q5_1, dtype
    return header[:type_pos] + struct.pack("<I", IQ4_NL) + header[type_pos + 4:]


def process_chunk(task):
    src_path, out_path, header_size, r0, nrows = task
    with open(src_path, "rb") as f:
        f.seek(header_size + r0 * SRC_ROW_BYTES)
        raw = f.read(nrows * SRC_ROW_BYTES)
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(-1, Q5_1_BLOCK)
    x = dequant_q5_1(blk)
    assert np.isfinite(x).all(), "non-finite value in source"
    L, dh = quantize_iq4_nl(x)
    err = KV_IQ4NL[L].astype(np.float32) * dh.astype(np.float32)[:, None] - x
    se = float(np.sum(err.astype(np.float64) ** 2))
    me = float(np.max(np.abs(err)))
    with open(out_path, "r+b") as f:
        f.seek(header_size + r0 * DST_ROW_BYTES)
        f.write(pack_blocks(L, dh))
    return r0, nrows, se, me


def verify_sample(src_path: pathlib.Path, out_path: pathlib.Path, header_size: int,
                  r0: int, nrows: int) -> None:
    """Independent decode of written bytes through canonical_xcheck's decoder."""
    import canonical_xcheck as cx
    with open(src_path, "rb") as f:
        f.seek(header_size + r0 * SRC_ROW_BYTES)
        src = f.read(nrows * SRC_ROW_BYTES)
    with open(out_path, "rb") as f:
        f.seek(header_size + r0 * DST_ROW_BYTES)
        got = f.read(nrows * DST_ROW_BYTES)
    x_ref = dequant_q5_1(np.frombuffer(src, np.uint8).reshape(-1, Q5_1_BLOCK))
    x_got = cx.from_s4_iq4_nl(cx.to_s4_iq4_nl(got)).reshape(-1)
    err = x_got - x_ref.ravel()
    print(f"verify rows[{r0}..{r0 + nrows}): rmse={np.sqrt(np.mean(err**2)):.6f} "
          f"max={np.max(np.abs(err)):.6f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="original shard 2 (Q5_1)")
    ap.add_argument("--out", required=True, help="replacement shard 2 (IQ4_NL)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--rows", type=int, default=0, help="only the first N rows (test)")
    ap.add_argument("--verify-rows", type=int, default=200_000,
                    help="rows re-decoded via canonical_xcheck after the run")
    a = ap.parse_args()

    src, out = pathlib.Path(a.src), pathlib.Path(a.out)
    header = patch_header(src)
    f = GGUFFile(src)
    t = f.tensors[0]
    total_rows = int(t.shape[1])
    if a.rows:
        total_rows = min(total_rows, a.rows)
    print(f"rows={total_rows}  src_row={SRC_ROW_BYTES}B  dst_row={DST_ROW_BYTES}B  "
          f"header={len(header)}B  out_data={total_rows * DST_ROW_BYTES / 2**30:.2f} GiB")

    with open(out, "wb") as fo:
        fo.write(header)
        fo.seek(len(header) + total_rows * DST_ROW_BYTES - 1)
        fo.write(b"\0")

    CH = 1_048_576  # rows per chunk (5.2M blocks, ~0.6 GB of float32)
    tasks = []
    r0 = 0
    while r0 < total_rows:
        n = min(CH, total_rows - r0)
        tasks.append((str(src), str(out), len(header), r0, n))
        r0 += n

    t0 = time.time()
    done_rows = 0
    tot_se = 0.0
    tot_me = 0.0
    with Pool(a.workers) as pool:
        for r0, n, se, me in pool.imap_unordered(process_chunk, tasks):
            done_rows += n
            tot_se += se
            tot_me = max(tot_me, me)
            el = time.time() - t0
            print(f"[{done_rows / total_rows * 100:5.1f}%] rows={done_rows} "
                  f"el={el:6.1f}s eta={el / done_rows * (total_rows - done_rows):6.0f}s "
                  f"rmse={np.sqrt(tot_se / (done_rows * VALUES_PER_ROW)):.6f} "
                  f"maxerr={tot_me:.4f}", flush=True)

    print(f"done in {time.time() - t0:.0f}s  global_rmse="
          f"{np.sqrt(tot_se / (total_rows * VALUES_PER_ROW)):.6f}")
    vr = min(a.verify_rows, total_rows)
    verify_sample(src, out, len(header), 0, vr)
    if total_rows > 4 * a.verify_rows:
        verify_sample(src, out, len(header), total_rows // 2 - vr // 2, vr)
        verify_sample(src, out, len(header), total_rows - vr, vr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
