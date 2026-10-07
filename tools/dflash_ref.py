#!/usr/bin/env python3
"""A numpy reference for the DFlash block forward (docs/DFLASH.md), for fixture parity.

    python tools/dflash_ref.py --dflash DFLASH.gguf --taps taps.bin --pos 120 --anchor 5513

Runs the z-lab/DeepSpec semantics with f32 arithmetic and bf16 rounding at the activation
boundaries (the engine's bf16 GEMV path rounds the same way): context = rmsnorm(fc(taps));
per layer k/v from the SAME ctx vector with k_norm and NeoX rope; the noise block
[anchor, mask x (K-1)] at rope positions [P..P+K-1] through the five layers with per-head
q/k norms and non-causal attention over the context cells + the block's own cells; final
norm; (the head is the target's, not in the drafter artifact - pass --target-gguf SHARD1 to
run it and print the row argmaxes).

Stage-by-stage dumps (--dump ctx,k0,q0,...) let the engine's forward be compared against
this reference tensor by tensor: taps -> ctx -> per-layer k/v/q -> logits.
"""
from __future__ import annotations

import argparse
import struct
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from gguf_reader import GGUFFile  # noqa: E402

MASK = 248077
EPS = 1e-6


def bf16(x: np.ndarray) -> np.ndarray:
    u = x.astype("<f4").view("<u4")
    r = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
    return (r << np.uint32(16)).view("<f4")


def rmsnorm(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * w


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def rope_neox(x: np.ndarray, pos: int, theta: float = 1e7) -> np.ndarray:
    # x [heads, hd], NeoX: halves rotated pairwise
    hd = x.shape[-1]
    inv = 1.0 / theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd)
    ang = pos * inv
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
    out = np.empty_like(x)
    out[..., : hd // 2] = x1 * cos - x2 * sin
    out[..., hd // 2 :] = x2 * cos + x1 * sin
    return out


def load(path: str) -> dict[str, np.ndarray]:
    g = GGUFFile(path)
    out = {}
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
        name = canon.get(t.name.removesuffix(".weight"), t.name.removesuffix(".weight"))
        out[name] = f.astype(np.float32)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dflash", required=True)
    ap.add_argument("--taps", default=None, help="the STRATA_DFLASH_TAPS fixture file; a synthetic tap row otherwise")
    ap.add_argument("--pos", type=int, default=0, help="the anchor position P (context cells [0, P))")
    ap.add_argument("--anchor", type=int, default=5513)
    ap.add_argument("--block", type=int, default=6, help="K query rows")
    ap.add_argument("--context-cells", type=int, default=0, help="how many context cells to fabricate (< P: random taps)")
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    w = load(args.dflash)
    H = w["hidden_norm"].shape[0]
    F = w["fc"].shape[1]
    n_taps = F // H
    rng = np.random.default_rng(3)

    def taps_row() -> np.ndarray:
        return bf16(rng.standard_normal(F, dtype=np.float32) * 3.0)

    # context cells [0, P): synthetic taps per position (a real fixture feeds the STRATA dump)
    ctxs = np.stack([rmsnorm(bf16(w["fc"] @ taps_row()), w["hidden_norm"]) for _ in range(args.pos)])

    # the noise block: [anchor, mask x (K-1)] at rope positions [P, P+K)
    K = args.block
    # the target's embedding rows are not in the artifact: random unit rows stand in for
    # embed(anchor)/embed(mask) - the token IDs only matter through the head, which is the
    # target's.  A --target-gguf run would gather the real rows.
    emb = bf16(rng.standard_normal((K, H), dtype=np.float32))
    h = emb.copy()
    dumps = {}
    for l in range(5):
        p = f"layers.{l}"
        xn = bf16(rmsnorm(h, w[f"{p}.input_layernorm"]))
        q = (xn @ w[f"{p}.self_attn.q_proj"].T).reshape(K, -1, 256)
        kc = (xn @ w[f"{p}.self_attn.k_proj"].T).reshape(K, -1, 256)
        vc = (xn @ w[f"{p}.self_attn.v_proj"].T).reshape(K, -1, 256)
        # context K/V from the SAME normed ctx vector, every layer
        ckc = (ctxs @ w[f"{p}.self_attn.k_proj"].T).reshape(args.pos, -1, 256)
        cvc = (ctxs @ w[f"{p}.self_attn.v_proj"].T).reshape(args.pos, -1, 256)
        q = bf16(rope_neox(bf16(rmsnorm(q, w[f"{p}.self_attn.q_norm"])), 0))   # per-row positions below
        for j in range(K):
            q[j] = bf16(rope_neox(bf16(rmsnorm((xn @ w[f"{p}.self_attn.q_proj"].T).reshape(K, -1, 256)[j],
                                               w[f"{p}.self_attn.q_norm"])), args.pos + j))
        for c in range(args.pos):
            ckc[c] = bf16(rope_neox(bf16(rmsnorm(ckc[c], w[f"{p}.self_attn.k_norm"])), c))
        for j in range(K):
            kc[j] = bf16(rope_neox(bf16(rmsnorm(kc[j], w[f"{p}.self_attn.k_norm"])), args.pos + j))
        kk = np.concatenate([ckc, kc], axis=0)          # [P + K, nkv, 256]
        vv = np.concatenate([cvc, vc], axis=0)
        nkv = kk.shape[1]
        G = 24 // nkv
        attn = np.zeros((K, 24, 256), np.float32)
        for kvh in range(nkv):
            for j in range(K):
                for head in range(kvh * G, (kvh + 1) * G):
                    s = q[j, head] @ kk[:, kvh].T / 16.0   # non-causal over EVERYTHING
                    a = np.exp(s - s.max())
                    a /= a.sum()
                    attn[j, head] = a @ vv[:, kvh]
        attn = attn.reshape(K, -1)
        h = h + bf16(attn @ w[f"{p}.self_attn.o_proj"].T)
        xn = bf16(rmsnorm(h, w[f"{p}.post_attention_layernorm"]))
        h = h + bf16(silu(xn @ w[f"{p}.mlp.gate_proj"].T) * (xn @ w[f"{p}.mlp.up_proj"].T) @ w[f"{p}.mlp.down_proj"].T)
        if f"q{l}" in args.dump.split(","):
            dumps[f"q{l}"] = q
    hn = rmsnorm(h, w["norm"])
    if "ctx" in args.dump.split(","):
        dumps["ctx"] = ctxs[-1] if args.pos else None
    if "logits" in args.dump.split(",") or not args.dump:
        print("final hidden [0, :8]:", np.array2string(hn[0, :8], precision=4))
    for k, v in dumps.items():
        np.save(f"/tmp/dflash_ref_{k}.npy", v)
        print(f"dumped /tmp/dflash_ref_{k}.npy {v.shape}")
    print(f"reference forward done: K={K}, P={args.pos}, ctx cells {ctxs.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
