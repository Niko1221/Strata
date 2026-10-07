#!/usr/bin/env python3
"""The DFlash stage-parity harness (docs/DFLASH.md): the engine's dumped stage tensors for one
draft cycle vs an independent numpy implementation reading the SAME standalone GGUF.

    STRATA_DF_PARITY=/tmp/dfparity [STRATA_DF_PARITY_CYCLE=0] ./build/strata ... \
        --dflash DFLASH.gguf --max-new 6
    python tools/dflash_stage_parity.py --dflash DFLASH.gguf --dir /tmp/dfparity

Every dumped [K, width] tensor is judged PER ROW (cos / max-abs / mean-abs / rel-L2 for each of
the K rows) - an aggregate cosine over all K rows hides one bad anchor row among five exact
mask rows.  The report names the FIRST divergent stage overall and the FIRST stage whose
ROW 0 (the anchor row, the only row on the real-token path) diverges.

Recomputed from the dumped inputs and the GGUF weights: the fusion (fc -> hidden_norm), each
checked layer's query path (input_layernorm -> q/k/v -> per-head q/k norms -> FULL-head NeoX
rope at each row's position), the attention itself (when the cycle dumps the pool: an oracle
that reads ONLY valid cells and fails hard on an out-of-range id instead of emitting NaN),
the o projection + residual, the MLP block, layer 1's query path from layer 0's output, and
the final norm.
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile  # noqa: E402

EPS = 1e-6
THETA = 1e7
SCALE = 1.0 / 16.0


def bf16(x):
    u = x.astype("<f4").view("<u4")
    r = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
    return (r << np.uint32(16)).view("<f4")


def rmsnorm(x, w):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(EPS)) * w


def rope_full(x, pos):
    """x [..., hd] at integer position(s) pos (broadcastable); full-head NeoX, theta 1e7."""
    hd = x.shape[-1]
    pair = np.arange(hd // 2, dtype=np.float32)
    inv = np.power(np.float32(THETA ** (-2.0 / hd)), pair)
    ang = np.float32(pos)[..., None] * inv
    c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    out = np.empty_like(x)
    out[..., : hd // 2] = x1 * c - x2 * s
    out[..., hd // 2:] = x2 * c + x1 * s
    return out


def softmax_rows(s):
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


def load(path):
    g = GGUFFile(path)
    data = open(path, "rb").read()
    canon = {"fc": "fc", "enc.output_norm": "hidden_norm", "output_norm": "norm"}
    for l in range(5):
        for a, b in [("attn_norm", "input_layernorm"), ("ffn_norm", "post_attention_layernorm"),
                     ("attn_q", "self_attn.q_proj"), ("attn_k", "self_attn.k_proj"),
                     ("attn_v", "self_attn.v_proj"), ("attn_output", "self_attn.o_proj"),
                     ("attn_q_norm", "self_attn.q_norm"), ("attn_k_norm", "self_attn.k_norm"),
                     ("ffn_gate", "mlp.gate_proj"), ("ffn_up", "mlp.up_proj"), ("ffn_down", "mlp.down_proj")]:
            canon[f"blk.{l}.{a}"] = f"layers.{l}.{b}"
    out = {}
    for t in g.tensors:
        n = int(np.prod(t.shape))
        raw = np.frombuffer(data, "<u2", n, g.data_start + t.offset)
        f = (raw.astype("<u4") << np.uint32(16)).view("<f4")
        if len(t.shape) == 2:
            f = f.reshape(t.shape[1], t.shape[0])   # torch [out, in]
        out[canon.get(t.name.removesuffix(".weight"), t.name.removesuffix(".weight"))] = f.astype(np.float32)
    return out


def read_bin(d, name):
    data = open(pathlib.Path(d) / f"{name}.bin", "rb").read()
    n = struct.unpack("<I", data[:4])[0]
    return np.frombuffer(data, "<f4", n, 4)


def has(d, name):
    return (pathlib.Path(d) / f"{name}.bin").exists()


def row_stats(got, ref):
    """Per-row (cos, max_abs, mean_abs, rel_l2); got/ref [K, width]."""
    out = []
    for r in range(got.shape[0]):
        a, b = got[r].astype(np.float64), ref[r].astype(np.float64)
        cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
        err = np.abs(a - b)
        rel = float(np.linalg.norm(err) / max(np.linalg.norm(a), 1e-30))
        out.append((cos, float(err.max()), float(err.mean()), rel))
    return out


class Report:
    def __init__(self):
        self.rows = []
        self.first_bad = None
        self.first_bad_row0 = None

    def stage(self, name, got, ref, cos_gate=0.9999, rel_gate=0.02):
        w = ref.shape[-1] if ref.ndim > 1 else ref.size
        got, ref = got.reshape(-1, w), ref.reshape(-1, w)
        rs = row_stats(got, ref)
        per_row_ok = [cos >= cos_gate and rel <= rel_gate for cos, _mx, _ma, rel in rs]
        agg = row_stats(got.reshape(1, -1), ref.reshape(1, -1))[0]
        all_ok = all(per_row_ok)
        self.rows.append((name, agg, rs, per_row_ok, all_ok))
        if not all_ok and self.first_bad is None:
            self.first_bad = name
        if not per_row_ok[0] and self.first_bad_row0 is None:
            self.first_bad_row0 = name
        return ref

    def print(self):
        print(f"{'stage':26s} {'agg cos':>9s} {'row0 cos':>9s} {'row0 max|d|':>11s} {'row0 relL2':>10s}  per-row")
        for name, agg, rs, ok, all_ok in self.rows:
            cells = " ".join(("." if g else "X") for g in ok)
            flag = "PASS" if all_ok else "FAIL"
            print(f"{name:26s} {agg[0]:9.6f} {rs[0][0]:9.6f} {rs[0][1]:11.3e} {rs[0][3]:10.3e}  [{cells}] {flag}")
        print("FIRST DIVERGENCE (any row): " + (self.first_bad if self.first_bad else "none"))
        print("FIRST ROW-0 DIVERGENCE:     " + (self.first_bad_row0 if self.first_bad_row0 else "none"))

    @property
    def ok(self):
        return all(all_ok for *_x, all_ok in self.rows)


def attn_oracle(q, kk, vv, page_table, page_size, nkv, n_ids, who):
    """Non-causal attention of every query row over cells [0, n_ids).

    q [K, nq, hd]; kk/vv [n_pool_rows, hd] in POOL ROW order (the flat [page][kvh][slot][hd]
    layout - the kv head is baked into the row index); page_table cell-page -> physical page.
    Reads ONLY cells with a valid entry and fails hard otherwise - no NaN is ever inserted: an
    id outside the dumped context raises.
    """
    K, nq, hd = q.shape
    G = nq // nkv
    attn = np.zeros((K, nq, hd), np.float32)
    for kvh in range(nkv):
        for j in range(K):
            rows = [(int(page_table[c // page_size]) if page_table is not None else c // page_size)
                    * nkv * page_size + kvh * page_size + c % page_size for c in range(n_ids)]
            for cell, row in enumerate(rows):
                if not (0 <= row < kk.shape[0]):
                    print(f"invalid reference cell {cell} -> pool row {row} ({who}: {kk.shape[0]} rows)",
                          file=sys.stderr)
                    raise SystemExit(2)
            k = kk[rows]                                         # [n_ids, hd]
            sc = (q[j, kvh * G:(kvh + 1) * G, :] @ k.T) * SCALE  # [G, n_ids]
            a = softmax_rows(sc)
            attn[j, kvh * G:(kvh + 1) * G, :] = a @ vv[rows]
    return attn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dflash", required=True)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    w = load(args.dflash)

    meta = open(pathlib.Path(args.dir) / "meta.bin", "rb").read()
    pos, K = struct.unpack("<2I", meta[:8])
    x = struct.unpack("<I", meta[8:12])[0] if len(meta) >= 12 else -1
    mask = struct.unpack("<I", meta[12:16])[0] if len(meta) >= 16 else -1
    page_size = struct.unpack("<I", meta[16:20])[0] if len(meta) >= 20 else 0
    H = w["hidden_norm"].shape[0]
    F = w["fc"].shape[1]
    emb = read_bin(args.dir, "emb").reshape(K, H)
    # the fusion's row count is the cycle's committed rows (1 after a full rejection), not K
    taps = read_bin(args.dir, "tapsin").reshape(-1, F)
    ctx_rows = taps.shape[0]

    rep = Report()

    # ---- the fusion
    rep.stage("fusion ctx", read_bin(args.dir, "ctx"), rmsnorm(bf16(taps @ w["fc"].T), w["hidden_norm"]))
    print(f"(cycle meta: anchor pos {pos}, K {K}, anchor token {x}, mask {mask}, fusion rows {ctx_rows})")

    # ---- the anchor embedding (step 8 invariant): row 0 is the anchor's row, rows 1.. mask rows
    if K > 2:
        same = max(np.abs(emb[r] - emb[1]).max() for r in range(1, K))
        print(f"emb: mask rows 1..{K-1} {'identical' if same == 0 else f'DIFFER (max {same:.3e})'}; "
              f"row0 vs row1 max|d| {np.abs(emb[0]-emb[1]).max():.3e}; |emb0| {np.linalg.norm(emb[0]):.3f}")

    # ---- layer 0's query path (from the DUMPED emb: verifies norm/proj/norm/rope given the input)
    def query_path(l, h_in):
        p = f"layers.{l}"
        xn_f32 = rmsnorm(h_in, w[f"{p}.input_layernorm"])   # the dumped xn stage is the f32 norm
        xn = bf16(xn_f32)
        q_raw = (xn @ w[f"{p}.self_attn.q_proj"].T).reshape(K, 24, 256)
        kc = (xn @ w[f"{p}.self_attn.k_proj"].T).reshape(K, 2, 256)
        vc = (xn @ w[f"{p}.self_attn.v_proj"].T).reshape(K, 2, 256)
        q_normed = q_raw.copy()
        for j in range(K):
            q_normed[j] = bf16(rmsnorm(q_raw[j], w[f"{p}.self_attn.q_norm"]))
            kc[j] = rope_full(bf16(rmsnorm(kc[j], w[f"{p}.self_attn.k_norm"])), pos + j)
        q_roped = q_normed.copy()
        for j in range(K):
            q_roped[j] = rope_full(q_normed[j], pos + j)
        return xn_f32, q_raw, q_normed, q_roped, kc, vc

    def attn_oracle_inputs(l):
        kk = read_bin(args.dir, f"kpool{l}").reshape(-1, 256)
        vv = read_bin(args.dir, f"vpool{l}").reshape(-1, 256)
        n_ids = pos + K
        if page_size <= 0:
            print(f"attention oracle L{l}: the cycle meta has no page_size (rerun the fixture)", file=sys.stderr)
            raise SystemExit(2)
        pt = None
        if has(args.dir, f"pt{l}"):
            pt = read_bin(args.dir, f"pt{l}").view(np.int32)
            if not np.array_equal(pt, np.arange(len(pt), dtype=np.int32)):
                print(f"attention oracle L{l}: the page table is not the identity: {pt[:16].tolist()}")
        if has(args.dir, f"steps{l}"):
            st = read_bin(args.dir, f"steps{l}").view(np.int32).reshape(K, 4)
            if int(st[0, 3]) != n_ids or not (st == st[0]).all():
                print(f"attention oracle L{l}: the runtime's attention steps {st[0].tolist()} "
                      f"do not give n_ids {n_ids} for every row")
        return kk, vv, pt, n_ids

    # ---- the context cells: recompute the expected K/V of EVERY context cell from the dumped
    # per-chunk fused contexts (ctx_b<pos>.bin) and compare against the layer pools - this is what
    # judges the context path's values AND its rope positions (a wrong-position context K is
    # invisible to every other stage, and reads to the drafter as noise)
    pool_cache = {}
    for l in range(5):
        if has(args.dir, f"kpool{l}") and has(args.dir, f"vpool{l}"):
            pool_cache[l] = (read_bin(args.dir, f"kpool{l}").reshape(-1, 256),
                             read_bin(args.dir, f"vpool{l}").reshape(-1, 256))
    if pool_cache and page_size > 0:
        ctx_all = []
        for f in sorted(pathlib.Path(args.dir).glob("ctx_b*.bin"),
                        key=lambda p: int(p.stem.removeprefix("ctx_b"))):
            p0 = int(f.stem.removeprefix("ctx_b"))
            rows = read_bin(args.dir, f.stem).reshape(-1, H)
            ctx_all.append((p0, rows))
        ctx_checks = [[] for _ in range(5)]
        for p0, rows in ctx_all:
            for j in range(rows.shape[0]):
                cell = p0 + j
                if cell >= pos:   # the query block's cells are checked against k{l}/v{l} elsewhere
                    continue
                xn = bf16(rows[j])
                for l, (kk, vv) in pool_cache.items():
                    page = cell // page_size
                    base = page * 2 * page_size + cell % page_size
                    if base >= kk.shape[0]:
                        continue
                    p = f"layers.{l}"
                    kexp = rope_full(bf16(rmsnorm((xn @ w[f"{p}.self_attn.k_proj"].T).reshape(2, 256),
                                                  w[f"{p}.self_attn.k_norm"])), cell)
                    vexp = (xn @ w[f"{p}.self_attn.v_proj"].T).reshape(2, 256)
                    dk = max(np.abs(kk[base + kvh * page_size] - kexp[kvh]).max() for kvh in range(2))
                    dv = max(np.abs(vv[base + kvh * page_size] - vexp[kvh]).max() for kvh in range(2))
                    ctx_checks[l].append((cell, dk, dv))
        for l, checks in enumerate(ctx_checks):
            if not checks:
                continue
            worst_k = max(checks, key=lambda t: t[1])
            worst_v = max(checks, key=lambda t: t[2])
            bad_k = sum(1 for _c, dk, _dv in checks if dk > 2e-2)
            bad_v = sum(1 for _c, _dk, dv in checks if dv > 2e-2)
            print(f"context cells L{l}: {len(checks)} checked; k max|d| {worst_k[1]:.3e} (cell {worst_k[0]}), "
                  f"v max|d| {worst_v[1]:.3e} (cell {worst_v[0]}); bad(k) {bad_k}, bad(v) {bad_v}"
                  + ("  <-- CONTEXT MISMATCH" if (bad_k or bad_v) else ""))

    # ---- the five draft layers, one per-loop iteration, the reference rolled forward from the
    # DUMPED emb through the dumped pools (the oracle judges the attention AND feeds the roll)
    h = emb
    for l in range(5):
        xn_ref, q_raw, q_normed, q_roped, k_ref, v_ref = query_path(l, h)
        tag = f"L{l}"
        if has(args.dir, f"xn{l}"):
            rep.stage(f"{tag} xn (norm)", read_bin(args.dir, f"xn{l}"), xn_ref)
        if has(args.dir, f"qraw{l}"):
            rep.stage(f"{tag} q raw (gemv)", read_bin(args.dir, f"qraw{l}"), q_raw.reshape(K, -1))
            rep.stage(f"{tag} q normed", read_bin(args.dir, f"qnormed{l}"), q_normed.reshape(K, -1))
        rep.stage(f"{tag} q (norm+rope)", read_bin(args.dir, f"q{l}"), q_roped.reshape(K, -1))
        rep.stage(f"{tag} k (norm+rope)", read_bin(args.dir, f"k{l}"), k_ref.reshape(K, -1))
        rep.stage(f"{tag} v", read_bin(args.dir, f"v{l}"), v_ref.reshape(K, -1))
        # the attention: oracle over the layer's dumped pool, per row, with strict cell bounds
        if has(args.dir, f"kpool{l}") and has(args.dir, f"vpool{l}"):
            kk, vv, pt, n_ids = attn_oracle_inputs(l)
            # the query block's own cells [pos, pos+K) were appended from k{l}/v{l} - cross-check
            # the pool's copies against the block tensors before trusting the oracle's inputs
            blk_k = read_bin(args.dir, f"k{l}").reshape(K, 2, 256)
            blk_v = read_bin(args.dir, f"v{l}").reshape(K, 2, 256)
            for j in range(K):
                page = int(pt[(pos + j) // page_size]) if pt is not None else (pos + j) // page_size
                base = page * 2 * page_size + (pos + j) % page_size
                if base + page_size < kk.shape[0]:
                    dk = max(np.abs(kk[base + kvh * page_size] - blk_k[j, kvh]).max() for kvh in range(2))
                    dv = max(np.abs(vv[base + kvh * page_size] - blk_v[j, kvh]).max() for kvh in range(2))
                    if dk > 2e-3 or dv > 2e-3:   # the pool stores fp16; the block tensors are f32
                        print(f"attention oracle {tag}: pool cell {pos+j} vs block k/v max|d| {dk:.3e}/{dv:.3e}")
            attn = attn_oracle(read_bin(args.dir, f"q{l}").reshape(K, 24, 256), kk, vv, pt, page_size, 2,
                               n_ids, tag).reshape(K, -1)
            rep.stage(f"{tag} attn (oracle)", read_bin(args.dir, f"attn{l}"), attn)
        else:
            attn = read_bin(args.dir, f"attn{l}").reshape(K, -1)   # plumbing check only
        # o projection + residual, then the MLP block - the reference roll continues
        h = h + bf16(attn @ w[f"layers.{l}.self_attn.o_proj"].T)
        if has(args.dir, f"h_attn{l}"):
            rep.stage(f"{tag} h (attn+o+res)", read_bin(args.dir, f"h_attn{l}"), h)
        xn = bf16(rmsnorm(h, w[f"layers.{l}.post_attention_layernorm"]))
        g_ = xn @ w[f"layers.{l}.mlp.gate_proj"].T
        u_ = xn @ w[f"layers.{l}.mlp.up_proj"].T
        h = h + bf16((g_ / (1 + np.exp(-g_)) * u_) @ w[f"layers.{l}.mlp.down_proj"].T)
        if has(args.dir, f"h_mlp{l}"):
            rep.stage(f"{tag} h after mlp", read_bin(args.dir, f"h_mlp{l}"), h)

    # ---- the final norm - meaningful now that the reference rolled through all five layers
    rep.stage("final norm", read_bin(args.dir, "final_norm"), rmsnorm(h, w["norm"]))

    rep.print()
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
