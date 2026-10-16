"""The engine-form transforms of the EXL3 loader, in numpy, checked against what the loader staged.

    strata-exl3-pack <model_dir> --dump <dir>            # the loader's staged bytes, one file per role
    python3 tools/exl3/check_engine_form.py <model_dir> <dir>

The oracle is `llama.cpp`'s converter (`conversion/qwen4exp.py` and the mixins it inherits from
`conversion/qwen.py`), which is what the engine's own GGUF packs are built with:

    - the v-head reorder (`_LinearAttentionVReorderBase`): HF stores the v heads grouped by k head,
      `[G0_v0 G0_v1 G0_v2 G1_v0 ...]`, the engine pairs v head `h` with k head `h % 16` (tiled).
    - `A_log` -> `-exp(A_log)`: the GDN gate multiplies the softplus by it.
    - the zero-centred gammas: HF stores `w`, the engine's RMSNorm reads `1 + w`.
    - the indexer q/k projections are stored [out][in] for `bf16_gemv` (a row of `n_in` per output).

A mismatch means the loader's staging is wrong, never the oracle.  This is a manual check: it needs
the downloaded model, and the role list is the one `strata-exl3-pack --dump` writes.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import exl3                                                              # noqa: E402
from exl3.codebook import CB_3INST, CB_MCG, CB_MUL1                      # noqa: E402

N_K, N_V, HEAD = 16, 48, 128        # config.json: linear_num_key_heads, linear_num_value_heads, *_head_dim
V_PER_K = N_V // N_K
CONV_CH, VALUE_DIM = 10240, 6144    # 2*128*16 + 128*48, 128*48
LAYER = "model.language_model.layers."

# (engine name, hf base, kind, transform); kind: f32, bf16, exl3
ROLES = [
    ("blk.0.ssm_a", LAYER + "0.linear_attn.A_log", "f32", "neg_exp_v"),
    ("blk.0.ssm_dt.bias", LAYER + "0.linear_attn.dt_bias", "f32", "v1"),
    ("blk.0.ssm_alpha.weight", LAYER + "0.linear_attn.in_proj_a.weight", "bf16", "vrow"),
    ("blk.0.ssm_beta.weight", LAYER + "0.linear_attn.in_proj_b.weight", "bf16", "vrow"),
    ("blk.0.ssm_conv1d.weight", LAYER + "0.linear_attn.conv1d.weight", "f32", "conv_v"),
    ("blk.0.hc_attn_norm.weight", LAYER + "0.attn_hyper_connection.hc_norm.weight", "f32", "norm1"),
    ("blk.0.hc_ffn_norm.weight", LAYER + "0.mlp_hyper_connection.hc_norm.weight", "f32", "norm1"),
    ("blk.3.attn_q_norm.weight", LAYER + "3.self_attn.q_norm.weight", "f32", "norm1"),
    ("blk.3.attn_k_norm.weight", LAYER + "3.self_attn.k_norm.weight", "f32", "norm1"),
    ("blk.3.indexer.q_norm.weight", LAYER + "3.self_attn.indexer.q_layernorm.weight", "f32", "norm1"),
    ("blk.3.indexer.k_norm.weight", LAYER + "3.self_attn.indexer.k_layernorm.weight", "f32", "norm1"),
    ("blk.1.ple_norm_key.weight", LAYER + "1.ple.norm_key.weight", "f32", "norm1"),
    ("blk.1.ple_norm_query.weight", LAYER + "1.ple.norm_query.weight", "f32", "norm1"),
    ("blk.1.ple_norm_conv.weight", LAYER + "1.ple.norm_conv.weight", "f32", "norm1"),
    ("output_hc_norm.weight", "model.language_model.hyper_connection_mixer.hc_norm.weight", "f32", "norm1"),
    ("blk.3.indexer.q_proj.weight", LAYER + "3.self_attn.indexer.index_qk_proj", "bf16", "split_q"),
    ("blk.3.indexer.k_proj.weight", LAYER + "3.self_attn.indexer.index_qk_proj", "bf16", "split_k"),
    ("blk.0.ffn_gate_inp.weight", LAYER + "0.mlp.gate.weight", "bf16", "id"),                 # control
    ("blk.0.ffn_gate_shexp.weight", LAYER + "0.mlp.shared_expert.gate_proj", "exl3", "id"),    # control
    ("blk.0.attn_qkv.weight", LAYER + "0.linear_attn.in_proj_qkv", "exl3", "trellis_qkv"),
    ("blk.0.attn_gate.weight", LAYER + "0.linear_attn.in_proj_z", "exl3", "trellis_z"),
    ("blk.0.ssm_out.weight", LAYER + "0.linear_attn.out_proj", "exl3", "trellis_in"),
]


def reorder_v(a: np.ndarray, dim: int, head_dim: int) -> np.ndarray:
    """`_LinearAttentionVReorderBase._reorder_v_heads`: grouped -> tiled along `dim`."""
    shape = list(a.shape)
    ns = shape[:dim] + [N_K, V_PER_K, head_dim] + shape[dim + 1:]
    a = a.reshape(ns)
    perm = list(range(len(ns)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return np.ascontiguousarray(a.transpose(perm).reshape(shape))


class Model:
    """The model's shards, read directly: the numpy backend of `safetensors` refuses BF16."""

    def __init__(self, d: str):
        self.dir = pathlib.Path(d)
        self.map = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self._hdr: dict[str, tuple[str, dict]] = {}

    def _header(self, shard: str):
        if shard not in self._hdr:
            import struct
            with open(self.dir / shard, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._hdr[shard] = (str(self.dir / shard), json.loads(f.read(n)))
        return self._hdr[shard]

    def get(self, name: str) -> np.ndarray:
        import struct
        path, hdr = self._header(self.map[name])
        t = hdr[name]
        dt = {"F32": np.float32, "F16": np.float16, "I16": np.int16, "BF16": np.uint16}[t["dtype"]]
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            base = 8 + n
            f.seek(base + t["data_offsets"][0])
            raw = f.read(t["data_offsets"][1] - t["data_offsets"][0])
        a = np.frombuffer(raw, dtype=dt).reshape(t["shape"])
        if dt is np.uint16:                                  # BF16
            return (a.astype(np.uint32) << 16).view(np.float32)
        return a

    def exl3(self, base: str):
        tr = self.get(base + ".trellis").astype(np.int16)
        suh = self.get(base + ".suh")
        svh = self.get(base + ".svh")
        cb = CB_MUL1 if base + ".mul1" in self.map else (CB_MCG if base + ".mcg" in self.map else CB_3INST)
        return tr, suh, svh, cb


def head_index(n: int) -> tuple[list[int], list[int]]:
    """(src, dst) for `n` heads: tiled head `t` comes from grouped head `(t % N_K) * V_PER_K + t / N_K`."""
    src = [((t % N_K) * V_PER_K + t // N_K) for t in range(N_V)]
    dst = list(range(N_V))
    return [i for s in src for i in range(s * n, (s + 1) * n)], [i for d in dst for i in range(d * n, (d + 1) * n)]


def exl3_form(m: Model, hf: str, what: str) -> bytes:
    """The loader's host trellis|suh|svh for one EXL3 linear, permuted where the converter reorders.

    A v head is 128 = 8 tiles, so the reorder moves whole 16-row tiles: `trellis_qkv`/`trellis_z` are
    the output dim (the j tiles and `svh`), `trellis_in` the input dim (the i tiles and `suh`).
    """
    tr, suh, svh, _ = m.exl3(hf)
    tr, suh, svh = tr.copy(), suh.copy(), svh.copy()
    tiles = HEAD // 16
    if what == "id":
        pass
    elif what in ("trellis_qkv", "trellis_z"):
        j0 = (CONV_CH - VALUE_DIM) // 16 if what == "trellis_qkv" else 0
        src = [j0 + ((t % N_K) * V_PER_K + t // N_K) * tiles + r for t in range(N_V) for r in range(tiles)]
        dst = [j0 + t * tiles + r for t in range(N_V) for r in range(tiles)]
        tr[:, dst, :] = tr[:, src, :]
        s_src, s_dst = head_index(HEAD)
        svh[[j0 * 16 + i for i in s_dst]] = svh[[j0 * 16 + i for i in s_src]]
    elif what == "trellis_in":
        src = [((t % N_K) * V_PER_K + t // N_K) * tiles + r for t in range(N_V) for r in range(tiles)]
        dst = [t * tiles + r for t in range(N_V) for r in range(tiles)]
        tr[dst, :, :] = tr[src, :, :]
        s_src, s_dst = head_index(HEAD)
        suh[s_dst] = suh[s_src]
    else:
        raise ValueError("unknown trellis transform " + what)
    return tr.tobytes() + suh.tobytes() + svh.tobytes()


def bf16(a: np.ndarray) -> np.ndarray:
    """The loader's f32->bf16 (`f32_to_bf16`): round by adding half an ulp to the magnitude bits."""
    bits = np.ascontiguousarray(a, dtype="<f4").view(np.uint32)
    return (((bits + np.uint32(0x8000)) & np.uint32(0xFFFF0000)) >> np.uint32(16)).astype(np.uint16)


def expected(m: Model, hf: str, kind: str, what: str):
    """(bytes, exact) - exact for the integer trellis, a tolerance comparison otherwise."""
    if kind == "exl3":
        return exl3_form(m, hf, what), True
    if what in ("split_q", "split_k"):
        tr, suh, svh, cb = m.exl3(hf)
        w = exl3.reconstruct_weight(tr, suh, svh, cb)          # [in, out]
        rows = slice(0, 512) if what == "split_q" else slice(512, None)
        a = np.ascontiguousarray(w[:, rows].T)                 # [out][in], the engine's bf16_gemv layout
        return bf16(a).tobytes(), False
    a = m.get(hf)
    if what == "norm1":
        a = a + np.float32(1.0)
    elif what == "neg_exp_v":
        a = reorder_v(-np.exp(a.astype(np.float64)).astype(np.float32), 0, 1)
    elif what in ("v1", "vrow"):
        a = reorder_v(a, 0, 1)
    elif what == "conv_v":
        a = a.reshape(a.shape[0], -1)
        a = np.concatenate([a[:CONV_CH - VALUE_DIM], reorder_v(a[CONV_CH - VALUE_DIM:], 0, HEAD)], axis=0)
    return (bf16(a).tobytes() if kind == "bf16" else np.ascontiguousarray(a, dtype="<f4").tobytes(), False)


def main() -> int:
    model_dir, dump_dir = sys.argv[1], pathlib.Path(sys.argv[2])
    m = Model(model_dir)
    bad = 0
    for engine, hf, kind, what in ROLES:
        want_bytes, exact = expected(m, hf, kind, what)
        got = np.fromfile(dump_dir / (engine + ".bin"), dtype=np.uint8)
        if exact:
            ok = got.tobytes() == want_bytes
            detail = "" if ok else " (%d vs %d bytes)" % (len(got), len(want_bytes))
        else:
            if kind == "bf16":
                want = (np.frombuffer(want_bytes, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
                g = (got.view("<u2").astype(np.uint32) << 16).view(np.float32)
            else:
                want = np.frombuffer(want_bytes, dtype="<f4")
                g = got.view("<f4")
            if g.shape != want.shape:
                ok, detail = False, " (shape %s vs %s)" % (g.shape, want.shape)
            else:
                ok = bool(np.allclose(g.astype(np.float64), want.astype(np.float64), rtol=2e-2, atol=1e-4))
                detail = "" if ok else " (max abs %.3g)" % float(np.max(np.abs(g - want)))
        print("%-32s %s%s" % (engine, "PASS" if ok else "FAIL", detail))
        bad += not ok
    print("check_engine_form: %d/%d roles match the converter" % (len(ROLES) - bad, len(ROLES)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
