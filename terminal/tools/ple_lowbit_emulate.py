"""tools/ple_lowbit_emulate.py - emulate a 3-bit PLE n-gram table as a Q8_0 table-only GGUF, to measure its accuracy by teacher forcing,
or (--native) write it in the 3-bit format itself, which the engine reads (PLE formats C4, C3, C2s).

    python tools/ple_lowbit_emulate.py --format c4 --src <IQ4_NL shard 2 .gguf> --out <c4.gguf> [--rows A:B] [--workers N] [--native]

The candidates (docs: memory-goal/ple-representation.md section 4), all derived from B's IQ4_NL values:
    c4   3-bit Lloyd codebook, one fp16 scale per 160-value row            62 B/row
    c3   3-bit Lloyd codebook, 32-value blocks, E4M3 scale x one F32 table scale   65 B/row
    c2s  3-bit Lloyd codebook, 16-value blocks, E4M3 scale x one F32 table scale   70 B/row
Per row: decode IQ4_NL exactly as ggml/the engine does (ngram.cpp:139-170), quantize to the format, dequantize, and encode the
dequantized values as Q8_0 (ngram.cpp:238), which the engine already reads through --ple-gguf as a table-only file
(ngram.cpp:259-267). Q8_0 adds a small rounding of its own (about 3e-5 NMSE); it is reported separately.

Quantizer = the one that produced the doc's NMSE figures: the block scale comes from a local search around the max-abs element
mapped to either end of the grid (IQ4_NL-like, +-NTRY steps), then the least-squares scale, rounded to the scale type; the grid is
a Lloyd-Max fit (8 levels, 30 iterations) on block-max-normalized values sampled over the whole table with a fixed seed.
Stdout reports the NMSE against the IQ4_NL values; the codebook is saved as <out>.codebook.json and can be reused with --codebook.

--native writes the same quantizer's codes and scales packed (engine/include/strata/kernels/ple_lowbit.hpp): 60 B of 3-bit codes,
then one fp16 (c4) or one E4M3 byte per block (c3, c2s), as I8 bytes [row bytes, rows] marked strata.ple.format = c4/c3/c2s with
strata.ple.codebook (8 F32) and, for c3/c2s, strata.ple.scale (the F32 table scale). Every chunk is decoded back as the engine
decodes it and must give the quantizer's floats bit for bit. The Q8_0 emulation stores those floats after Q8_0's own rounding
(about 2.5e-5 NMSE more), so a native table is close to its emulation but not bit identical to it. With the same --seed and
--ntry the codebook fit is the same; --codebook <emulation>.codebook.json (e.g. ple-c4-q8.gguf.codebook.json for c4) reuses
the screened grid explicitly. It also writes <out>.ref.bin, about 4096 of its rows with those floats (int32 n,
uint32 rows[n], float32 values[n][160]; rows numbered in the output table): the reference src/kernels/ple_fp8_parity.cpp
checks the engine's reads against (PLE_EXPECT_FORMAT=C4|C3|C2s).
"""
import argparse
import json
import os
import pathlib
import struct
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile  # noqa: E402

TENSOR = "per_layer_token_embd.weight"
DIM, IQ_RB, Q8_RB = 160, 90, 170          # PLE_HEAD_DIM, IQ4_NL row bytes, Q8_0 row bytes (ngram.hpp:52,55)
GGML_Q8_0, GGML_I8 = 8, 24
ALIGN = 32
KV_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)  # ngram.cpp:134
E4M3_MAX = 448.0
# name -> (values per scale block, scale kind, scale bytes per row, 3-bit code bytes per row)
FORMATS = {"c4": (160, "f16", 2), "c3": (32, "e4m3", 5), "c2s": (16, "e4m3", 10)}
CODE_BYTES = DIM * 3 // 8                    # 60: every candidate stores 160 x 3 bits
SCALE_HEADROOM = 256.0                       # table scale T = max block scale / 256, so E4M3 (max 448) has 1.75x headroom


def row_bytes(fmt):
    return CODE_BYTES + FORMATS[fmt][2]


# ---------------------------------------------------------------- IQ4_NL decode, Q8_0 encode/decode
def iq4nl_decode(rows):
    """(n, 90) uint8 -> (n, 160) float32: d * kvalues[code], low nibbles = elements 0..15, high = 16..31 of each block."""
    n = rows.shape[0]
    b = np.ascontiguousarray(rows).reshape(n, 5, 18)
    d = np.ascontiguousarray(b[:, :, :2]).view("<f2")[:, :, 0].astype(np.float32)
    q = b[:, :, 2:]
    codes = np.concatenate([q & 15, q >> 4], axis=2)
    return (d[:, :, None] * KV_IQ4NL[codes]).reshape(n, DIM)


def q8_0_encode(x):
    """(n, 160) float32 -> (n, 170) uint8, ggml quantize_row_q8_0_ref: d = amax/127, q = roundf(x/d) (half away from zero), d as fp16."""
    n = x.shape[0]
    b = np.ascontiguousarray(x, dtype=np.float32).reshape(n, 5, 32)
    d = np.abs(b).max(axis=2) / np.float32(127)
    idv = np.divide(np.float32(1), d, out=np.zeros_like(d), where=d != 0)
    v = b * idv[:, :, None]
    q = (np.sign(v) * np.floor(np.abs(v) + np.float32(0.5))).astype(np.int8)
    out = np.empty((n, 5, 34), np.uint8)
    out[:, :, :2] = d.astype("<f2").view(np.uint8).reshape(n, 5, 2)
    out[:, :, 2:] = q.view(np.uint8)
    return out.reshape(n, Q8_RB)


def q8_0_decode(rows):
    n = rows.shape[0]
    b = np.ascontiguousarray(rows).reshape(n, 5, 34)
    d = np.ascontiguousarray(b[:, :, :2]).view("<f2")[:, :, 0].astype(np.float32)
    q = np.ascontiguousarray(b[:, :, 2:]).view(np.int8).astype(np.float32)
    return (d[:, :, None] * q).reshape(n, DIM)


# ---------------------------------------------------------------- the quantizers
def e4m3_round(x):
    """float32 -> nearest E4M3FN value (round half to even, saturating at 448, subnormals down to 2^-9)."""
    a = np.minimum(np.abs(x), np.float32(E4M3_MAX))
    _, ex = np.frexp(a)
    step = np.exp2(np.maximum(ex - 1, -6) - 3).astype(np.float32)
    q = np.minimum(np.round(a / step) * step, np.float32(E4M3_MAX))
    return np.copysign(q, x).astype(np.float32)


# E4M3FN bytes 0..126 as values (0x7F is NaN): increasing, so a value's byte is its index
E4M3_POS = np.array([np.ldexp(np.float32(m), -9) if e == 0 else np.ldexp(np.float32(8 + m), e - 10)
                     for e in range(16) for m in range(8)][:127], np.float32)


def e4m3_bytes(x):
    """float32 -> the E4M3FN byte of e4m3_round(x), sign bit included (also for -0)."""
    q = e4m3_round(x)
    return (np.searchsorted(E4M3_POS, np.abs(q)) | (np.signbit(q).astype(np.int64) << 7)).astype(np.uint8)


def e4m3_value(b):
    """E4M3FN bytes -> float32, as the engine reads them (ple_e4m3; the NaN bytes read as 0)."""
    v = np.append(E4M3_POS, np.float32(0))[b & 127]
    return np.where(b & 128, -v, v).astype(np.float32)


def quantize(Z, grid, bs, scale_kind, table_scale, ntry):
    """Z (n, 160) float32 -> (dequantized (n, 160) float32, codes (n, 160) uint8, scales (n, 160/bs) float32, n_saturated).
    scale_kind: "f16", "e4m3" (E4M3 x table_scale) or "f32" (unrounded, used to size the table scale)."""
    n = Z.shape[0]
    B = Z.reshape(n, -1, bs)
    g = grid.astype(np.float32)
    edges = (g[1:] + g[:-1]) / 2
    mx = np.take_along_axis(B, np.abs(B).argmax(2)[:, :, None], 2)[:, :, 0]
    best_err = np.full(B.shape[:2], np.inf, np.float32)
    best_c = np.zeros(B.shape, np.int8)
    best_s = np.zeros(B.shape[:2], np.float32)
    sat = 0
    for side in (g[0], g[-1]):
        for t in range(-ntry, ntry + 1):
            ref = side + t * (g[1] - g[0]) / (2 * ntry + 1) * np.sign(side)
            d = np.where(mx == 0, np.float32(1e-30), mx / ref).astype(np.float32)
            c = np.searchsorted(edges, B / d[:, :, None]).astype(np.int8)
            q = g[c]
            s = (q * B).sum(2) / np.maximum((q * q).sum(2), np.float32(1e-30))
            if scale_kind == "f16":
                s = s.astype(np.float16).astype(np.float32)
            elif scale_kind == "e4m3":
                s = e4m3_round(s / table_scale) * table_scale
            err = ((s[:, :, None] * q - B) ** 2).sum(2)
            upd = err < best_err
            best_err = np.where(upd, err, best_err)
            best_s = np.where(upd, s, best_s)
            best_c = np.where(upd[:, :, None], c, best_c)
    if scale_kind == "e4m3":
        sat = int((np.abs(best_s) / table_scale > E4M3_MAX * 0.999).sum())
    deq = (best_s[:, :, None] * g[best_c]).reshape(n, DIM).astype(np.float32)
    return deq, best_c.reshape(n, DIM).astype(np.uint8), best_s, sat


def pack_native(codes, scales, cb):
    """quantize()'s codes (n, 160) and scales (n, 160/block) -> (n, row_bytes) uint8, the engine's row (ple_lowbit.hpp)."""
    n = codes.shape[0]
    w = np.bitwise_or.reduce(codes.reshape(n, 20, 8).astype(np.uint32) << (3 * np.arange(8, dtype=np.uint32)), axis=2)
    out = [(w[:, :, None] >> np.array([0, 8, 16], np.uint32) & 255).astype(np.uint8).reshape(n, CODE_BYTES)]
    if cb["scale"] == "f16":
        out.append(scales.astype("<f2").view(np.uint8).reshape(n, 2))
    else:   # the scales are e4m3 x T, so e4m3_round(scale / T) is that e4m3 again (the decode check in _work proves it)
        out.append(e4m3_bytes(scales / np.float32(cb["table_scale"])))
    return np.concatenate(out, axis=1)


def native_dequant(rows, cb):
    """The engine's decode of pack_native's rows (ple_lowbit_dequant_row): value j = s[j // block] * grid[code j] in float32."""
    n, bs = rows.shape[0], cb["block"]
    b = rows[:, :CODE_BYTES].reshape(n, 20, 3).astype(np.uint32)
    w = b[:, :, 0] | b[:, :, 1] << 8 | b[:, :, 2] << 16
    codes = (w[:, :, None] >> (3 * np.arange(8, dtype=np.uint32)) & 7).reshape(n, DIM)
    sc = np.ascontiguousarray(rows[:, CODE_BYTES:])
    s = sc.view("<f2").astype(np.float32) if cb["scale"] == "f16" else e4m3_value(sc) * np.float32(cb["table_scale"])
    return (s[:, :, None] * np.array(cb["grid"], np.float32)[codes].reshape(n, -1, bs)).reshape(n, DIM)


def lloyd(x, k, iters=30):
    """Lloyd-Max levels for 1-D data: k levels, quantile start."""
    q = np.quantile(x, (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        a = np.searchsorted((q[1:] + q[:-1]) / 2, x)
        cnt = np.bincount(a, minlength=k)
        sm = np.bincount(a, weights=x, minlength=k)
        q = np.where(cnt > 0, sm / np.maximum(cnt, 1), q)
    return np.sort(q)


# ---------------------------------------------------------------- source table
class Source:
    def __init__(self, path):
        g = GGUFFile(path)
        t = [x for x in g.tensors if x.name == TENSOR]
        if len(t) != 1 or t[0].type_name != "IQ4_NL" or len(t[0].shape) != 2 or t[0].shape[0] != DIM:
            sys.exit("%s: no IQ4_NL [%d, rows] %s" % (path, DIM, TENSOR))
        self.path, self.n_rows, self.off = str(path), t[0].shape[1], g.data_start + t[0].offset
        if os.path.getsize(path) < self.off + self.n_rows * IQ_RB:
            sys.exit("%s is shorter than its table" % path)

    def read(self, f, a, b):
        f.seek(self.off + a * IQ_RB)
        buf = np.fromfile(f, np.uint8, (b - a) * IQ_RB)
        if buf.size != (b - a) * IQ_RB:
            sys.exit("short read at row %d" % a)
        return buf.reshape(b - a, IQ_RB)


def fit_codebook(src, fmt, seed, ntry, n_chunks=256, chunk=256, n_values=2_000_000):
    """Codebook from rows spread over the whole table (not the slice), so a slice run and the full run use the same grid."""
    bs, kind, _ = FORMATS[fmt]
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.integers(0, src.n_rows - chunk, n_chunks))
    with open(src.path, "rb") as f:
        Z = np.concatenate([iq4nl_decode(src.read(f, int(s), int(s) + chunk)) for s in starts])
    Bn = Z.reshape(-1, bs)
    Bn = (Bn / np.abs(Bn).max(1, keepdims=True)).ravel()
    grid = lloyd(Bn[rng.choice(Bn.size, min(n_values, Bn.size), replace=False)].astype(np.float64), 8).astype(np.float32)
    table_scale = 1.0
    if kind == "e4m3":   # size T from the unrounded block scales of 16384 of the fit rows
        _, _, s, _ = quantize(Z[:16384], grid, bs, "f32", 1.0, ntry)
        table_scale = float(np.abs(s).max() / SCALE_HEADROOM)
    return {"format": fmt, "block": bs, "scale": kind, "bytes_per_row": row_bytes(fmt), "grid": [float(v) for v in grid],
            "table_scale": table_scale, "seed": seed, "ntry": ntry, "fit_rows": int(Z.shape[0]), "fit_values": int(min(n_values, Bn.size)),
            "src": pathlib.Path(src.path).name}


# ---------------------------------------------------------------- the run
_G = {}


def _init(src_path, cb, native=False, lo=0, ref_stride=1):
    _G["src"], _G["cb"], _G["f"] = Source(src_path), cb, open(src_path, "rb")
    _G["grid"] = np.array(cb["grid"], np.float32)
    _G["native"], _G["lo"], _G["ref_stride"] = native, lo, ref_stride


def _work(span):
    a, b = span
    src, cb = _G["src"], _G["cb"]
    X = iq4nl_decode(src.read(_G["f"], a, b))
    Y, codes, scales, sat = quantize(X, _G["grid"], cb["block"], cb["scale"], np.float32(cb["table_scale"]), cb["ntry"])
    ref = None
    if _G["native"]:
        out = pack_native(codes, scales, cb)
        Y8 = native_dequant(out, cb)
        if not np.array_equal(Y8.view(np.uint32), Y.view(np.uint32)):
            raise RuntimeError("rows %d:%d: the packed rows do not decode to the quantizer's values" % (a, b))
        i = np.arange((_G["lo"] - a) % _G["ref_stride"], b - a, _G["ref_stride"])
        ref = (a + i, Y[i])
    else:
        out = q8_0_encode(Y)
        Y8 = q8_0_decode(out)
    X64, Y64, Y864 = X.astype(np.float64), Y.astype(np.float64), Y8.astype(np.float64)
    return a, out.tobytes(), ((X64 ** 2).sum(), ((Y64 - X64) ** 2).sum(), ((Y864 - X64) ** 2).sum(), ((Y864 - Y64) ** 2).sum()), sat, ref


def gguf_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def gguf_kv(k, v):
    if isinstance(v, str):
        return gguf_str(k) + struct.pack("<I", 8) + gguf_str(v)
    if isinstance(v, float):
        return gguf_str(k) + struct.pack("<If", 6, v)                                            # FLOAT32
    return gguf_str(k) + struct.pack("<IIQ", 9, 6, len(v)) + struct.pack("<%df" % len(v), *v)   # ARRAY of FLOAT32


def gguf_header(rows, kv, ggml_type=GGML_Q8_0, ne0=DIM):
    kvs = b"".join(gguf_kv(k, v) for k, v in kv.items())
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kv)) + kvs
    head += gguf_str(TENSOR) + struct.pack("<I", 2) + struct.pack("<QQ", ne0, rows) + struct.pack("<IQ", ggml_type, 0)
    return head + b"\0" * ((-len(head)) % ALIGN)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--format", required=True, choices=sorted(FORMATS))
    ap.add_argument("--src", required=True, help="the IQ4_NL GGUF holding per_layer_token_embd.weight (shard 2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", help="A:B, only these rows (the output table has B-A rows)")
    ap.add_argument("--codebook", help="reuse a saved codebook JSON instead of fitting one")
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--ntry", type=int, default=7, help="scale search steps each side of the max-abs mapping (the doc used 7)")
    ap.add_argument("--chunk-rows", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--idle-priority", action="store_true", help="Windows: run at idle priority")
    ap.add_argument("--native", action="store_true", help="write the packed 3-bit table the engine reads, not the Q8_0 emulation")
    a = ap.parse_args()
    if a.idle_priority and sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)

    src = Source(a.src)
    lo, hi = 0, src.n_rows
    if a.rows:
        lo, hi = (int(x) for x in a.rows.split(":"))
    if not 0 <= lo < hi <= src.n_rows:
        sys.exit("--rows %s outside 0:%d" % (a.rows, src.n_rows))
    out = pathlib.Path(a.out)
    cb_path = out.with_name(out.name + ".codebook.json")
    t0 = time.time()
    if a.codebook:
        cb = json.loads(pathlib.Path(a.codebook).read_text())
        if cb["format"] != a.format:
            sys.exit("codebook is for %s, not %s" % (cb["format"], a.format))
    else:
        cb = fit_codebook(src, a.format, a.seed, a.ntry)
    out.parent.mkdir(parents=True, exist_ok=True)
    cb_path.write_text(json.dumps(cb, indent=1))
    print("codebook (%s, %d B/row, %s scale, table scale %.6g), fitted in %.1f s -> %s" %
          (a.format, row_bytes(a.format), cb["scale"], cb["table_scale"], time.time() - t0, cb_path), flush=True)
    print("  grid:", " ".join("%.5f" % v for v in cb["grid"]), flush=True)

    spans = [(s, min(s + a.chunk_rows, hi)) for s in range(lo, hi, a.chunk_rows)]
    out_rb = row_bytes(a.format) if a.native else Q8_RB
    if a.native:
        kv = {"general.architecture": "strata-ple",
              "general.name": "PLE n-gram table, 3-bit %s (%d B a row)" % (a.format, out_rb),
              "strata.ple.format": a.format, "strata.ple.codebook": [float(v) for v in np.float32(cb["grid"])],
              "strata.ple.source": pathlib.Path(a.src).name, "strata.ple.rows": "%d:%d" % (lo, hi)}
        if cb["scale"] == "e4m3":
            kv["strata.ple.scale"] = float(np.float32(cb["table_scale"]))
        head = gguf_header(hi - lo, kv, GGML_I8, out_rb)
    else:
        kv = {"general.architecture": "strata-ple",
              "general.name": "PLE n-gram table emulating %s (3-bit dequantized values stored as Q8_0)" % a.format,
              "strata.ple.emulated_format": a.format, "strata.ple.emulated_from": pathlib.Path(a.src).name,
              "strata.ple.emulated_rows": "%d:%d" % (lo, hi)}
        head = gguf_header(hi - lo, kv)
    ref_stride = max(1, (hi - lo) // 4096)
    tmp = out.with_name(out.name + ".part")
    tot, sat, done, t1, refs = np.zeros(4), 0, 0, time.time(), []
    if a.workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(a.workers, _init, (a.src, cb, a.native, lo, ref_stride))
        results = pool.imap(_work, spans)
    else:
        _init(a.src, cb, a.native, lo, ref_stride)
        pool, results = None, map(_work, spans)
    with open(tmp, "wb") as w:
        w.write(head)
        for i, (s, buf, sums, sx, ref) in enumerate(results):
            w.write(buf)
            tot += sums
            sat += sx
            done += len(buf) // out_rb
            if ref is not None:
                refs.append(ref)
            if i % 50 == 49 or i == len(spans) - 1:
                el = time.time() - t1
                print("  %d/%d rows, %.0f rows/s, eta %.0f s, running NMSE %.5f" %
                      (done, hi - lo, done / el, el * (hi - lo - done) / done, tot[1] / tot[0]), flush=True)
    if pool:
        pool.close()
        pool.join()
    if tmp.stat().st_size != len(head) + (hi - lo) * out_rb:
        sys.exit("size check failed on " + str(tmp))
    tmp.replace(out)
    n = hi - lo
    if a.native:
        rows = np.concatenate([r for r, _ in refs])
        m = len(rows) // 16 * 16 or len(rows)   # ple_fp8_parity gathers them 16 rows (one token) at a time
        ref_path = out.with_name(out.name + ".ref.bin")
        ref_path.write_bytes(struct.pack("<i", m) + (rows[:m] - lo).astype("<u4").tobytes() +
                             np.concatenate([v for _, v in refs])[:m].astype("<f4").tobytes())
        print("format %s rows %d:%d (%d rows), written packed, %d B/row (I8, strata.ple.format = %s): %s" %
              (a.format, lo, hi, n, out_rb, a.format, out))
        print("NMSE vs IQ4_NL values (3-bit, as stored)       : %.5f" % (tot[1] / tot[0]))
        print("%d rows and their values for ple_fp8_parity: %s" % (m, ref_path))
    else:
        print("format %s rows %d:%d (%d rows), %d B/row in %s, written as Q8_0 (%d B/row): %s" %
              (a.format, lo, hi, n, row_bytes(a.format), "its own layout", Q8_RB, out))
        print("NMSE vs IQ4_NL values (3-bit dequantized)      : %.5f" % (tot[1] / tot[0]))
        print("NMSE vs IQ4_NL values (after Q8_0 rounding)    : %.5f" % (tot[2] / tot[0]))
        print("  Q8_0's own rounding, relative to the values   : %.2e" % (tot[3] / tot[0]))
    if cb["scale"] == "e4m3":
        print("E4M3 block scales saturated at 448: %d of %d" % (sat, n * DIM // cb["block"]))
    print("%.1f s total (%.0f rows/s)" % (time.time() - t0, n / (time.time() - t1)))


if __name__ == "__main__":
    main()
