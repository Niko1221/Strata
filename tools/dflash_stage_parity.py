#!/usr/bin/env python3
"""The DFlash stage-parity harness (docs/DFLASH.md): the engine's dumped stage tensors for one
draft cycle vs an independent numpy implementation reading the SAME standalone GGUF.

    STRATA_DF_PARITY=/tmp/dfparity ./build/strata ... --dflash DFLASH.gguf --max-new 6
    python tools/dflash_stage_parity.py --dflash DFLASH.gguf --dir /tmp/dfparity

Recomputes from the dumped inputs (emb, tapsin) and the GGUF weights: the fusion
(fc -> hidden_norm), and draft layer 0's query path (input_layernorm -> q/k/v -> per-head
q/k norms -> FULL-head NeoX rope at each row's position), the o projection + residual from
the dumped attention, the MLP block, and the final norm.  Reports cos / max-abs / mean-rel
per stage and names the FIRST divergent stage.  The attention's own combining (scores over
the pool K/V) is not recomputed - everything around it is, so a wrong combining is isolated
by elimination.
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


def bf16(x):
    u = x.astype("<f4").view("<u4")
    r = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
    return (r << np.uint32(16)).view("<f4")


def rmsnorm(x, w):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(EPS)) * w


def rope_full(x, pos):
    hd = x.shape[-1]
    pair = np.arange(hd // 2, dtype=np.float32)
    inv = np.power(np.float32(THETA ** (-2.0 / hd)), pair)
    ang = np.float32(pos) * inv
    c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    out = np.empty_like(x)
    out[..., : hd // 2] = x1 * c - x2 * s
    out[..., hd // 2:] = x2 * c + x1 * s
    return out


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


def stats(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
    err = np.abs(a - b)
    rel = float(err.mean() / max(np.abs(a).mean(), 1e-30))
    return cos, float(err.max()), rel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dflash", required=True)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    w = load(args.dflash)

    pos, K = struct.unpack("<2I", open(pathlib.Path(args.dir) / "meta.bin", "rb").read())
    H = w["hidden_norm"].shape[0]
    F = w["fc"].shape[1]
    emb = read_bin(args.dir, "emb").reshape(K, H)
    # the fusion's row count is the cycle's committed rows (1 after a full rejection), not K
    taps = read_bin(args.dir, "tapsin").reshape(-1, F)
    ctx_rows = taps.shape[0]

    stages = []
    ok = True
    first = None

    def stage(name, got, ref, cos_gate=0.9999, rel_gate=0.02):
        nonlocal ok, first
        cos, mx, rel = stats(got, ref)
        good = cos >= cos_gate and rel <= rel_gate
        stages.append((name, cos, mx, rel, good))
        if not good and first is None:
            first = name
        ok &= good
        return ref

    # ---- the fusion
    stage("fusion ctx", read_bin(args.dir, "ctx"), rmsnorm(bf16(taps @ w["fc"].T), w["hidden_norm"]))
    print(f"(cycle meta: anchor pos {pos}, K {K}, fusion rows {ctx_rows})")

    # ---- layer 0's query path
    p = "layers.0"
    h = emb
    xn = bf16(rmsnorm(h, w[f"{p}.input_layernorm"]))
    q = (xn @ w[f"{p}.self_attn.q_proj"].T).reshape(K, 24, 256)
    kc = (xn @ w[f"{p}.self_attn.k_proj"].T).reshape(K, 2, 256)
    vc = (xn @ w[f"{p}.self_attn.v_proj"].T).reshape(K, 2, 256)
    for j in range(K):
        q[j] = rope_full(bf16(rmsnorm(q[j], w[f"{p}.self_attn.q_norm"])), pos + j)
        kc[j] = rope_full(bf16(rmsnorm(kc[j], w[f"{p}.self_attn.k_norm"])), pos + j)
    stage("q0 (norm+rope)", read_bin(args.dir, "q0"), q.reshape(K, -1))
    stage("k0 (norm+rope)", read_bin(args.dir, "k0"), kc.reshape(K, -1))
    stage("v0", read_bin(args.dir, "v0"), vc.reshape(K, -1))

    # ---- o projection + residual, from the DUMPED attention
    attn = read_bin(args.dir, "attn0").reshape(K, -1)
    h = h + bf16(attn @ w[f"{p}.self_attn.o_proj"].T)
    stage("h after attn+o+res", read_bin(args.dir, "h_attn0"), h)

    # ---- the MLP block
    xn = bf16(rmsnorm(h, w[f"{p}.post_attention_layernorm"]))
    g_ = xn @ w[f"{p}.mlp.gate_proj"].T
    u_ = xn @ w[f"{p}.mlp.up_proj"].T
    h = h + bf16((g_ / (1 + np.exp(-g_)) * u_) @ w[f"{p}.mlp.down_proj"].T)
    stage("h after mlp", read_bin(args.dir, "h_mlp0"), h)

    # ---- the final norm (only row 0's layer-0 chain is exact; deeper layers accumulate rope/pool
    # differences, so gate the final norm on the whole [K, H] but with the pool caveat)
    stage("final norm", read_bin(args.dir, "final_norm"), rmsnorm(h, w["norm"]), cos_gate=0.999, rel_gate=0.2)

    print(f"{'stage':28s} {'cos':>10s} {'max|d|':>10s} {'rel mean':>10s}  gate")
    for name, cos, mx, rel, good in stages:
        print(f"{name:28s} {cos:10.6f} {mx:10.3e} {rel:10.3e}  {'PASS' if good else 'FAIL'}")
    print("FIRST DIVERGENCE: " + (first if first else "none - all dumped stages agree"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
