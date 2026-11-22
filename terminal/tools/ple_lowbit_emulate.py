"""tools/ple_lowbit_emulate.py - emulate a low-bit PLE n-gram table as a Q8_0 table-only GGUF, to measure its accuracy by teacher
forcing, or (--native) write it in the low-bit format itself, which the engine reads (PLE formats C1, C2, C2s, C3, C4, C5, C6,
C5e, T58, T58e).

    python tools/ple_lowbit_emulate.py --format c4 --src <IQ4_NL shard 2 .gguf> --out <c4.gguf> [--rows A:B] [--workers N] [--native]

The candidates (docs: memory-goal/ple-representation.md section 4), all derived from B's IQ4_NL values:
    c1   4-bit IQ4_NL grid (kvalues / 127, not fitted), one fp16 scale per 160-value row   82 B/row
    c2   3-bit Lloyd codebook, 32-value blocks, fp16 scale                  70 B/row
    c2s  3-bit Lloyd codebook, 16-value blocks, E4M3 scale x one F32 table scale   70 B/row
    c3   3-bit Lloyd codebook, 32-value blocks, E4M3 scale x one F32 table scale   65 B/row
    c4   3-bit Lloyd codebook, one fp16 scale per 160-value row            62 B/row
    c5   3-bit codes through a trellis (below), one fp16 scale per row      62 B/row
    c6   2-bit Lloyd codebook, 32-value blocks, fp16 scale                  50 B/row (the doc's "16- or 32-value blocks":
         32 with fp16, the variant both of its NMSE figures were computed with; 16 with E4M3 has only the one vs B)
    c5e  c5's codes, one E4M3 scale per row x one F32 table scale            61 B/row
    t58  2.8-bit trellis (below): a 448-bit code stream, one fp16 scale      58 B/row
    t58e t58's code stream, one E4M3 scale per row x one F32 table scale     57 B/row
    (memory-goal/t-proto-report.md: held-out NMSE c5e 0.01830, t58 0.02402, t58e 0.02415)
Per row: decode IQ4_NL exactly as ggml/the engine does (ngram.cpp:139-170), quantize to the format, dequantize, and encode the
dequantized values as Q8_0 (ngram.cpp:238), which the engine already reads through --ple-gguf as a table-only file
(ngram.cpp:259-267). Q8_0 adds a small rounding of its own (about 3e-5 NMSE); it is reported separately.

Quantizer = the one that produced the doc's NMSE figures: the block scale comes from a local search around the max-abs element
mapped to either end of the grid (IQ4_NL-like, +-NTRY steps), then the least-squares scale, rounded to the scale type; the grid is
a Lloyd-Max fit (2^bits levels, 30 iterations) on block-max-normalized values sampled over the whole table with a fixed seed
(c1 keeps IQ4_NL's own grid, as the doc's C1 did).
c5 is a tail-biting bitshift trellis (QTIP-style, one value a step): a register of C5_STATE_BITS = 12 bits takes each 3-bit
code in at the bottom, starting with the row's own last codes, and value j = s * lut[state j], with lut 4096 floats. The codes
are the Viterbi path of the row over its RMS (terminal/tools/ple_c5_viterbi.cpp, a DLL built with cl: see C5_BUILD), s the
least-squares scale of that path as fp16. The lut is fitted like a Lloyd grid: from a random permutation of the normalized
values' quantiles, C5_FIT_ITERS rounds of encoding the fit rows and setting each state's value to the mean of the values
its paths put there (deterministic for a seed; --workers threads).
t58 is the same trellis at 2.8 bits a value (terminal/tools/ple_t_viterbi.cpp, its own DLL: see T_BUILD): a row's codes are
one circular 448-bit stream, value j shifts b_j = 2 3 3 3 3 (repeated) new bits into the 12-bit register, so its state is
the 12 stream bits ending at e_j = (j+1)*448/160, most significant first (the first values' states wrap to the stream's
end). The E4M3 forms (c5e, t58e) take the table scale T = the fit rows' largest least-squares scale / 256 (c3's headroom),
round the row's scale to e4m3 x T, run the Viterbi again on the row over that scale, and keep whichever path, with its own
least-squares scale rounded the same way, is closer (t-proto-report.md: "e4m3r").
Stdout reports the NMSE against the IQ4_NL values; the codebook is saved as <out>.codebook.json and can be reused with --codebook.

--native writes the same quantizer's codes and scales packed (engine/include/strata/kernels/ple_lowbit.hpp): 20 x bits bytes of
codes (a little-endian bit stream; t58/t58e: the 56 B stream, most significant bit first), then one fp16 per block (c1, c2,
c4, c6, c5, t58) or one E4M3 byte per block (c3, c2s, c5e, t58e), as I8 bytes [row bytes, rows] marked strata.ple.format =
<format> with strata.ple.codebook (2^bits F32; the trellis forms: their 4096-float lut, with strata.ple.trellis_bits = 12,
and for t58/t58e strata.ple.code_bits = 448) and, with E4M3 scales, strata.ple.scale (the F32 table scale). c5's rows are
c4's layout.
Every chunk is decoded back as the engine
decodes it and must give the quantizer's floats bit for bit. The Q8_0 emulation stores those floats after Q8_0's own rounding
(about 2.5e-5 NMSE more), so a native table is close to its emulation but not bit identical to it. With the same --seed and
--ntry the codebook fit is the same; --codebook <emulation>.codebook.json (e.g. ple-c4-q8.gguf.codebook.json for c4) reuses
the screened grid explicitly. It also writes <out>.ref.bin, about 4096 of its rows with those floats (int32 n,
uint32 rows[n], float32 values[n][160]; rows numbered in the output table): the reference src/kernels/ple_fp8_parity.cpp
checks the engine's reads against (PLE_EXPECT_FORMAT=C1|C2|C2s|C3|C4|C5|C6|C5e|T58|T58e).
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
# name -> (values per scale block, scale kind, scale bytes per row)
FORMATS = {"c4": (160, "f16", 2), "c3": (32, "e4m3", 5), "c2s": (16, "e4m3", 10),
           "c1": (160, "f16", 2), "c2": (32, "f16", 10), "c6": (32, "f16", 10), "c5": (160, "f16", 2),
           "c5e": (160, "e4m3", 1), "t58": (160, "f16", 2), "t58e": (160, "e4m3", 1)}
BITS = {"c1": 4, "c6": 2}                    # bits a code; 3 for the others (60 B of codes a row; t58/t58e: below)
FIXED_GRID = {"c1": KV_IQ4NL / np.float32(127)}   # C1 keeps IQ4_NL's grid (the doc's "4b_iq4nl_b160_f16"); the rest fit Lloyd
SCALE_HEADROOM = 256.0                       # table scale T = max block scale / 256, so E4M3 (max 448) has 1.75x headroom
C5_STATE_BITS = 12                           # PLE_C5_STATE_BITS (ple_lowbit.hpp): c5's lut has 2^12 floats
C5_FIT_ITERS = 100                           # lut fit rounds (rows 0:4000: NMSE 0.01844 after 30, 0.01818 after 100)
TRELLIS = ("c5", "c5e", "t58", "t58e")       # the formats whose codes are trellis states (a 4096-float lut)
T_FORMATS = ("t58", "t58e")                  # ... through the 2.8-bit stream (PLE_T58_CODE_BITS, ple_lowbit.hpp)
T_CODE_BITS = 448
T_SHIFTS = np.array([2, 3, 3, 3, 3] * 32, np.uint8)          # b_j: the new stream bits of value j (5 values a 14-bit step)
T_ENDS = np.cumsum(T_SHIFTS.astype(np.int64))                 # e_j = (j+1)*448/160: value j's state ends at stream bit e_j
assert T_ENDS[-1] == T_CODE_BITS and (T_ENDS == (np.arange(1, DIM + 1) * T_CODE_BITS) // DIM).all()
_ROOT = pathlib.Path(__file__).resolve().parents[2]
C5_DLL = _ROOT / "build" / "ple_c5_viterbi.dll"
C5_BUILD = r"cl /nologo /O2 /arch:AVX2 /LD /EHsc /Fo:build\ /Fe:build\ple_c5_viterbi.dll terminal\tools\ple_c5_viterbi.cpp"
T_DLL = _ROOT / "build" / "ple_t_viterbi.dll"
T_BUILD = r"cl /nologo /O2 /arch:AVX2 /LD /EHsc /Fo:build\ /Fe:build\ple_t_viterbi.dll terminal\tools\ple_t_viterbi.cpp"


def bits_of(fmt):
    return BITS.get(fmt, 3)


def bits_label(fmt):
    return "2.8" if fmt in T_FORMATS else str(bits_of(fmt))


def code_bytes(fmt):
    return (T_CODE_BITS if fmt in T_FORMATS else DIM * bits_of(fmt)) // 8


def row_bytes(fmt):
    return code_bytes(fmt) + FORMATS[fmt][2]


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
    """quantize()'s codes (n, 160) and scales (n, 160/block) -> (n, row_bytes) uint8, the engine's row (ple_lowbit.hpp):
    8 codes of cb["bits"] (default 3) bits in that many bytes, a little-endian word each, then the scales. t58/t58e: the
    codes are the trellis states, packed as their stream (t_pack)."""
    n, bits = codes.shape[0], cb.get("bits", 3)
    if cb.get("format") in T_FORMATS:
        out = [t_pack(codes)]
    else:
        w = np.bitwise_or.reduce(codes.reshape(n, 20, 8).astype(np.uint32) << (bits * np.arange(8, dtype=np.uint32)), axis=2)
        out = [(w[:, :, None] >> (8 * np.arange(bits, dtype=np.uint32)) & 255).astype(np.uint8).reshape(n, 20 * bits)]
    if cb["scale"] == "f16":
        out.append(scales.astype("<f2").view(np.uint8).reshape(n, -1))
    else:   # the scales are e4m3 x T, so e4m3_round(scale / T) is that e4m3 again (the decode check in _work proves it)
        out.append(e4m3_bytes(scales / np.float32(cb["table_scale"])))
    return np.concatenate(out, axis=1)


def native_dequant(rows, cb):
    """The engine's decode of pack_native's rows (ple_lowbit_dequant_row): value j = s[j // block] * grid[code j] in float32;
    the trellis forms (ple_c5_dequant_row, ple_c5e_dequant_row, ple_t58_dequant_row, ple_t58e_dequant_row): s * lut[state j]."""
    n, bs, bits = rows.shape[0], cb["block"], cb.get("bits", 3)
    nc = code_bytes(cb["format"]) if cb.get("format") in T_FORMATS else 20 * bits
    sc = np.ascontiguousarray(rows[:, nc:])
    s = sc.view("<f2").astype(np.float32) if cb["scale"] == "f16" else e4m3_value(sc) * np.float32(cb["table_scale"])
    if cb.get("format") in T_FORMATS:
        return s * np.array(cb["grid"], np.float32)[t_states(rows[:, :nc])]
    b = rows[:, :20 * bits].reshape(n, 20, bits).astype(np.uint32)
    w = np.bitwise_or.reduce(b << (8 * np.arange(bits, dtype=np.uint32)), axis=2)
    codes = (w[:, :, None] >> (bits * np.arange(8, dtype=np.uint32)) & ((1 << bits) - 1)).reshape(n, DIM)
    if cb.get("format") in ("c5", "c5e"):
        return s * np.array(cb["grid"], np.float32)[c5_states(codes)]
    return (s[:, :, None] * np.array(cb["grid"], np.float32)[codes].reshape(n, -1, bs)).reshape(n, DIM)


# ---------------------------------------------------------------- the trellis forms (c5, c5e, t58, t58e)
_ENC = {}


def _encoder(fmt):
    """The compiled search for fmt (c5/c5e: ple_c5_viterbi.cpp, t58/t58e: ple_t_viterbi.cpp), loaded once a process. A
    missing or other DLL is a RuntimeError, not a sys.exit: in a Pool worker SystemExit kills the worker, the pool replaces
    it and imap waits forever (main() checks the DLL before it starts the workers, so a worker does not get here first)."""
    t = fmt in T_FORMATS
    dll_path, build, abi_fn, fn = (T_DLL, T_BUILD, "ple_t_abi", "ple_t_viterbi") if t else \
                                  (C5_DLL, C5_BUILD, "ple_c5_abi", "ple_c5_viterbi")
    if fn not in _ENC:
        import ctypes
        if not dll_path.exists():
            raise RuntimeError("%s needs the compiled encoder %s; in a Visual Studio x64 prompt at the repository root run\n"
                               "    md build\n    %s" % (fmt, dll_path, build))
        try:
            dll = ctypes.CDLL(str(dll_path))
            abi = getattr(dll, abi_fn)()
        except (OSError, AttributeError) as e:
            raise RuntimeError("%s does not load as the %s encoder (%s): rebuild it (%s)" % (dll_path, fmt, e, build)) from None
        if abi != 1:
            raise RuntimeError("%s is not the encoder this script calls: rebuild it (%s)" % (dll_path, build))
        f = getattr(dll, fn)
        f.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * (2 if t else 1)
        _ENC[fn] = f
    return _ENC[fn]


def c5_encoder():
    return _encoder("c5")


def _search(fmt, Y, lut, threads, out_dtype):
    """Y (n, 160) float32 -> (n, 160) of the DLL's output for the tail-biting trellis path closest to Y in squared error
    (Viterbi), `threads` rows at once: c5's 3-bit codes, or t58's states."""
    f = _encoder(fmt)
    Y = np.ascontiguousarray(Y, np.float32)
    lut = np.ascontiguousarray(lut, np.float32)
    if lut.shape != (1 << C5_STATE_BITS,) or Y.ndim != 2 or Y.shape[1] != DIM:
        raise ValueError("%s search: Y %s, lut %s" % (fmt, Y.shape, lut.shape))
    out = np.empty(Y.shape, out_dtype)
    extra = (T_SHIFTS.ctypes.data,) if fmt in T_FORMATS else ()

    def run(span):   # the DLL call releases the GIL, so the threads run at once
        a, b = span
        if b > a and f(Y[a:b].ctypes.data, b - a, lut.ctypes.data, C5_STATE_BITS, *extra, out[a:b].ctypes.data) != 0:
            raise RuntimeError("the %s encoder refused L = %d" % (fmt, C5_STATE_BITS))
    cut = np.linspace(0, Y.shape[0], max(1, threads) + 1).astype(int)
    spans = list(zip(cut[:-1], cut[1:]))
    if threads > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(run, spans))
    else:
        run(spans[0])
    return out


def c5_codes(Y, lut, threads=1):
    """Y (n, 160) float32, the rows over their RMS -> (n, 160) uint8 codes: the tail-biting trellis path closest to Y
    in squared error (ple_c5_viterbi.cpp, Viterbi), `threads` rows at once."""
    return _search("c5", Y, lut, threads, np.uint8)


def t_codes(Y, lut, threads=1):
    """As c5_codes for t58/t58e (ple_t_viterbi.cpp): -> (n, 160) uint16 states of the path. They are their own codes: value
    j's new stream bits are state j's low b_j bits (t_pack)."""
    return _search("t58", Y, lut, threads, np.uint16)


def t_pack(states):
    """(n, 160) t58 states of a trellis path -> (n, 56) uint8, the code stream (stream bit t is bit 7 - t % 8 of byte t // 8):
    each 14-bit step k is stream bits 14k .. 14k + 13, the low 2 bits of state 5k (ending at 14k + 2) and state 5k + 4 (the
    12 bits ending at 14k + 14), four steps to 7 bytes. (States that are no path pack to a stream that decodes to other
    states: native_rows' decode check stops there.)"""
    st = states.astype(np.int64)
    w = ((st[:, 0::5] & 3) << 12) | st[:, 4::5]
    v = (w.reshape(-1, 8, 4) << np.array([42, 28, 14, 0], np.int64)).sum(2)
    return ((v[:, :, None] >> (8 * np.arange(6, -1, -1, dtype=np.int64))) & 255).astype(np.uint8).reshape(-1, T_CODE_BITS // 8)


def t_states(code_bytes):
    """(n, 56) uint8 code streams -> (n, 160) states, as ple_t58_dequant_row makes them: value j's state is the 12 bits of
    the circular stream ending at e_j, most significant first (read from 3 bytes of the stream with its last 2 bytes put
    in front, so the first values' states wrap)."""
    b = np.ascontiguousarray(code_bytes).astype(np.int64)
    ext = np.concatenate([b[:, -2:], b, np.zeros((b.shape[0], 1), np.int64)], 1)   # stream bit t is ext bit t + 16
    o = T_ENDS - C5_STATE_BITS + 16
    q, r = o // 8, o % 8
    return (((ext[:, q] << 16) | (ext[:, q + 1] << 8) | ext[:, q + 2]) >> (12 - r)) & ((1 << C5_STATE_BITS) - 1)


def trellis_path(fmt, Y, lut, threads=1):
    """-> (codes as pack_native takes them, states): the tail-biting path closest to Y."""
    if fmt in T_FORMATS:
        st = t_codes(Y, lut, threads)
        return st, st.astype(np.int64)
    codes = c5_codes(Y, lut, threads)
    return codes, c5_states(codes)


def c5_states(codes):
    """(n, 160) codes -> (n, 160) trellis states, as ple_c5_dequant_row makes them: the register takes each code in at the
    bottom and starts with the row's own last codes (tail-biting), so state j = codes j, j-1, ... in 3-bit groups, mod 2^12."""
    k = (C5_STATE_BITS - 1) // 3                      # the codes before code 0 that reach into state 0: 157, 158, 159
    c = np.concatenate([codes[:, DIM - k:], codes], 1).astype(np.int64)
    st = np.zeros(c.shape, np.int64)
    for j in range(0, C5_STATE_BITS, 3):
        st[:, j // 3:] |= c[:, :c.shape[1] - j // 3] << j
    return st[:, k:] & ((1 << C5_STATE_BITS) - 1)


def c5_normalized(Z):
    r = np.sqrt((Z.astype(np.float64) ** 2).mean(1)).astype(np.float32)
    return (Z / np.where(r > 0, r, np.float32(1))[:, None]).astype(np.float32)


def ls_scale(q, Z):
    return (q * Z).sum(1) / np.maximum((q * q).sum(1), np.float32(1e-30))


def quantize_trellis(Z, fmt, lut, table_scale=1.0, threads=1):
    """Z (n, 160) float32 -> (dequantized (n, 160) float32, codes, scales (n, 1) float32, n_saturated): the trellis path of
    the row over its RMS, then the least-squares scale of that path, rounded to fp16 (c5, t58; as quantize() rounds c4's).
    c5e/t58e: that scale as e4m3 x table_scale, then the path of the row over that scale and its own scale rounded the
    same way, whichever is closer (t-proto-report.md: "e4m3r", the Viterbi twice)."""
    codes, st = trellis_path(fmt, c5_normalized(Z), lut, threads)
    q = lut[st]
    s = ls_scale(q, Z)
    if FORMATS[fmt][1] == "f16":
        s = s.astype(np.float16).astype(np.float32)[:, None]
        return s * q, codes, s, 0
    T = np.float32(table_scale)
    s1 = e4m3_round(s / T) * T
    codes2, st2 = trellis_path(fmt, Z / np.where(s1 != 0, s1, np.float32(1))[:, None], lut, threads)
    q2 = lut[st2]
    s2 = e4m3_round(ls_scale(q2, Z) / T) * T
    Y1, Y2 = s1[:, None] * q, s2[:, None] * q2
    better = ((Y2 - Z) ** 2).sum(1) < ((Y1 - Z) ** 2).sum(1)
    s = np.where(better, s2, s1)[:, None]
    sat = int((np.abs(s) / T > E4M3_MAX * 0.999).sum())
    return np.where(better[:, None], Y2, Y1), np.where(better[:, None], codes2, codes), s, sat


def fit_trellis_lut(Z, seed, iters, threads=1, fmt="c5"):
    """Lloyd for the trellis: start from a random permutation of the normalized values' 4096 quantiles; each round encodes
    the rows and sets every state's value to the mean of the values the paths put on it (an unused state keeps its own).
    c5e's lut is c5's, t58e's is t58's (the same fit)."""
    Y = c5_normalized(Z)
    S = 1 << C5_STATE_BITS
    lut = np.random.default_rng(seed).permutation(np.quantile(Y.ravel(), (np.arange(S) + 0.5) / S)).astype(np.float32)
    for _ in range(iters):
        st = trellis_path(fmt, Y, lut, threads)[1].ravel()
        cnt = np.bincount(st, minlength=S)
        sm = np.bincount(st, weights=Y.ravel().astype(np.float64), minlength=S)
        lut = np.where(cnt > 0, sm / np.maximum(cnt, 1), lut).astype(np.float32)
    return lut


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
        if buf.size != (b - a) * IQ_RB:   # (in a Pool worker too: a RuntimeError reaches main(), a SystemExit hangs the pool)
            raise RuntimeError("%s: short read at row %d" % (self.path, a))
        return buf.reshape(b - a, IQ_RB)


def fit_codebook(src, fmt, seed, ntry, n_chunks=256, chunk=256, n_values=2_000_000, threads=1, c5_iters=C5_FIT_ITERS):
    """Codebook from rows spread over the whole table (not the slice), so a slice run and the full run use the same grid."""
    bs, kind, _ = FORMATS[fmt]
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.integers(0, src.n_rows - chunk, n_chunks))
    with open(src.path, "rb") as f:
        Z = np.concatenate([iq4nl_decode(src.read(f, int(s), int(s) + chunk)) for s in starts])
    if fmt in TRELLIS:
        lut = fit_trellis_lut(Z, seed, c5_iters, threads, fmt)
        table_scale = 1.0
        if kind == "e4m3":   # T from the unrounded least-squares scales of all the fit rows' paths (as t-proto measured)
            table_scale = float(np.abs(ls_scale(lut[trellis_path(fmt, c5_normalized(Z), lut, threads)[1]], Z)).max() / SCALE_HEADROOM)
        cb = {"format": fmt, "bits": 3, "block": bs, "scale": kind, "bytes_per_row": row_bytes(fmt),
              "grid": [float(v) for v in lut], "table_scale": table_scale,
              "trellis_bits": C5_STATE_BITS, "iters": c5_iters, "seed": seed, "fit_rows": int(Z.shape[0]),
              "src": pathlib.Path(src.path).name}
        if fmt in T_FORMATS:
            cb["code_bits"] = T_CODE_BITS
        return cb
    Bn = Z.reshape(-1, bs)
    Bn = (Bn / np.abs(Bn).max(1, keepdims=True)).ravel()
    if fmt in FIXED_GRID:
        grid = FIXED_GRID[fmt]
    else:
        grid = lloyd(Bn[rng.choice(Bn.size, min(n_values, Bn.size), replace=False)].astype(np.float64), 1 << bits_of(fmt)).astype(np.float32)
    table_scale = 1.0
    if kind == "e4m3":   # size T from the unrounded block scales of 16384 of the fit rows
        _, _, s, _ = quantize(Z[:16384], grid, bs, "f32", 1.0, ntry)
        table_scale = float(np.abs(s).max() / SCALE_HEADROOM)
    return {"format": fmt, "bits": bits_of(fmt), "block": bs, "scale": kind, "bytes_per_row": row_bytes(fmt), "grid": [float(v) for v in grid],
            "table_scale": table_scale, "seed": seed, "ntry": ntry, "fit_rows": int(Z.shape[0]), "fit_values": int(min(n_values, Bn.size)),
            "src": pathlib.Path(src.path).name}


# ---------------------------------------------------------------- the run
_G = {}


def _init(src_path, cb, native=False, lo=0, ref_stride=1):
    _G["src"], _G["cb"], _G["f"] = Source(src_path), cb, open(src_path, "rb")
    _G["grid"] = np.array(cb["grid"], np.float32)
    _G["native"], _G["lo"], _G["ref_stride"] = native, lo, ref_stride


def quantize_any(X, cb, grid, threads=1):
    """The format's quantizer with the codebook cb (grid: its values as float32) -> (floats, codes, scales, n_saturated)."""
    if cb["format"] in TRELLIS:
        return quantize_trellis(X, cb["format"], grid, np.float32(cb["table_scale"]), threads)
    return quantize(X, grid, cb["block"], cb["scale"], np.float32(cb["table_scale"]), cb["ntry"])


def native_rows(X, cb, grid, threads=1, what="rows"):
    """X (n, 160) float32 -> (the packed rows (n, row bytes), the quantizer's floats, n_saturated); the rows decode as the
    engine decodes them to exactly those floats, or this raises."""
    Y, codes, scales, sat = quantize_any(X, cb, grid, threads)
    out = pack_native(codes, scales, cb)
    if not np.array_equal(native_dequant(out, cb).view(np.uint32), Y.view(np.uint32)):
        raise RuntimeError("%s: the packed rows do not decode to the quantizer's values" % what)
    return out, Y, sat


def _work(span):
    a, b = span
    src, cb = _G["src"], _G["cb"]
    X = iq4nl_decode(src.read(_G["f"], a, b))
    ref = None
    if _G["native"]:
        out, Y, sat = native_rows(X, cb, _G["grid"], what="rows %d:%d" % (a, b))
        Y8 = Y
        i = np.arange((_G["lo"] - a) % _G["ref_stride"], b - a, _G["ref_stride"])
        ref = (a + i, Y[i])
    else:
        Y, _, _, sat = quantize_any(X, cb, _G["grid"])
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
    if isinstance(v, int):
        return gguf_str(k) + struct.pack("<II", 4, v)                                            # UINT32
    return gguf_str(k) + struct.pack("<IIQ", 9, 6, len(v)) + struct.pack("<%df" % len(v), *v)   # ARRAY of FLOAT32


def gguf_header(rows, kv, ggml_type=GGML_Q8_0, ne0=DIM):
    kvs = b"".join(gguf_kv(k, v) for k, v in kv.items())
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kv)) + kvs
    head += gguf_str(TENSOR) + struct.pack("<I", 2) + struct.pack("<QQ", ne0, rows) + struct.pack("<IQ", ggml_type, 0)
    return head + b"\0" * ((-len(head)) % ALIGN)


def native_kv(fmt, cb, src, lo, hi):
    """The metadata of a --native table of rows lo:hi (PleTable::open reads and checks it)."""
    kv = {"general.architecture": "strata-ple",
          "general.name": "PLE n-gram table, %s-bit %s (%d B a row)" % (bits_label(fmt), fmt, row_bytes(fmt)),
          "strata.ple.format": fmt, "strata.ple.codebook": [float(v) for v in np.float32(cb["grid"])],
          "strata.ple.source": pathlib.Path(src).name, "strata.ple.rows": "%d:%d" % (lo, hi)}
    if cb["scale"] == "e4m3":
        kv["strata.ple.scale"] = float(np.float32(cb["table_scale"]))
    if fmt in TRELLIS:
        kv["strata.ple.trellis_bits"] = C5_STATE_BITS
    if fmt in T_FORMATS:
        kv["strata.ple.code_bits"] = T_CODE_BITS
    return kv


def load_codebook(path, fmt):
    """A saved codebook JSON for fmt, or sys.exit saying why it is not one."""
    cb = json.loads(pathlib.Path(path).read_text())
    if cb["format"] != fmt:
        sys.exit("codebook is for %s, not %s" % (cb["format"], fmt))
    if fmt in TRELLIS and (cb.get("trellis_bits") != C5_STATE_BITS or len(cb["grid"]) != 1 << C5_STATE_BITS):
        sys.exit("codebook is a %s-bit trellis of %d values, not %d bits" % (cb.get("trellis_bits"), len(cb["grid"]), C5_STATE_BITS))
    if fmt in T_FORMATS and cb.get("code_bits") != T_CODE_BITS:
        sys.exit("codebook is for a %s-bit code stream, not %d" % (cb.get("code_bits"), T_CODE_BITS))
    if cb["scale"] == "e4m3" and not 0 < cb["table_scale"] < float("inf"):
        sys.exit("codebook has the table scale %r" % cb["table_scale"])
    return cb


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
    ap.add_argument("--native", action="store_true", help="write the packed low-bit table the engine reads, not the Q8_0 emulation")
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
    if a.format in TRELLIS:   # here, not first in a worker (with --codebook this process would not load it at all)
        try:
            _encoder(a.format)
        except RuntimeError as e:
            sys.exit(str(e))
    out = pathlib.Path(a.out)
    cb_path = out.with_name(out.name + ".codebook.json")
    t0 = time.time()
    if a.codebook:
        cb = load_codebook(a.codebook, a.format)
    else:
        cb = fit_codebook(src, a.format, a.seed, a.ntry, threads=a.workers)
    out.parent.mkdir(parents=True, exist_ok=True)
    cb_path.write_text(json.dumps(cb, indent=1))
    print("codebook (%s, %d B/row, %s scale, table scale %.6g), fitted in %.1f s -> %s" %
          (a.format, row_bytes(a.format), cb["scale"], cb["table_scale"], time.time() - t0, cb_path), flush=True)
    g = cb["grid"]
    print("  grid:", " ".join("%.5f" % v for v in g[:16]) + (" ... (%d values)" % len(g) if len(g) > 16 else ""), flush=True)

    spans = [(s, min(s + a.chunk_rows, hi)) for s in range(lo, hi, a.chunk_rows)]
    out_rb = row_bytes(a.format) if a.native else Q8_RB
    if a.native:
        head = gguf_header(hi - lo, native_kv(a.format, cb, a.src, lo, hi), GGML_I8, out_rb)
    else:
        kv = {"general.architecture": "strata-ple",
              "general.name": "PLE n-gram table emulating %s (%s-bit dequantized values stored as Q8_0)" % (a.format, bits_label(a.format)),
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
        print("NMSE vs IQ4_NL values (%s-bit, as stored)       : %.5f" % (bits_label(a.format), tot[1] / tot[0]))
        print("%d rows and their values for ple_fp8_parity: %s" % (m, ref_path))
    else:
        print("format %s rows %d:%d (%d rows), %d B/row in %s, written as Q8_0 (%d B/row): %s" %
              (a.format, lo, hi, n, row_bytes(a.format), "its own layout", Q8_RB, out))
        print("NMSE vs IQ4_NL values (%s-bit dequantized)      : %.5f" % (bits_label(a.format), tot[1] / tot[0]))
        print("NMSE vs IQ4_NL values (after Q8_0 rounding)    : %.5f" % (tot[2] / tot[0]))
        print("  Q8_0's own rounding, relative to the values   : %.2e" % (tot[3] / tot[0]))
    if cb["scale"] == "e4m3":
        print("E4M3 block scales saturated at 448: %d of %d" % (sat, n * DIM // cb["block"]))
    print("%.1f s total (%.0f rows/s)" % (time.time() - t0, n / (time.time() - t1)))


if __name__ == "__main__":
    main()
