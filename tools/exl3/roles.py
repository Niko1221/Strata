"""Engine role <-> EXL3 (HF) tensor mapping for Qwen3.8-Flash-Next, and a checker.

The engine (src/core/layout.cpp) names tensors `blk.<L>.<suffix>` (plus a few global
names).  The EXL3/HF pack names them `model.language_model.layers.<L>.<module>...`.
This module is the single place that says which HF tensor serves which engine role and
what shape/format transform is needed.  `python3 -m tools.exl3.roles --check DIR`
verifies every role resolves against the actual model.
"""
from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Role:
    # engine suffix: "{L}" is the layer index.  "exl3" roles carry an EXL3 trellis/suh/svh.
    engine: str
    hf: str                       # HF tensor base, "{L}" for layer. "" = not in this model.
    exl3: bool                    # the engine role is an EXL3 linear
    engine_ndim: int              # 2 = (ne0, ne1), 1 = 1-D
    # shape transform from HF to engine form: "id", "t", "conv1d", "split_qk"
    xform: str = "id"
    # for 1-D: the engine element count and whether it must be widened bf16/f16 -> f32
    elems: int = 0
    note: str = ""


# Every layer
LAYER_COMMON = [
    Role("hc_attn_down.weight", "attn_hyper_connection.input_mix_weight_down.weight", False, 2, "t"),
    Role("hc_attn_up.weight", "attn_hyper_connection.input_mix_weight_up.weight", False, 2, "t"),
    Role("hc_attn_inject.weight", "attn_hyper_connection.block_inject_weight.weight", False, 2, "t"),
    Role("hc_ffn_down.weight", "mlp_hyper_connection.input_mix_weight_down.weight", False, 2, "t"),
    Role("hc_ffn_up.weight", "mlp_hyper_connection.input_mix_weight_up.weight", False, 2, "t"),
    Role("hc_ffn_inject.weight", "mlp_hyper_connection.block_inject_weight.weight", False, 2, "t"),
    Role("ffn_gate_inp.weight", "mlp.gate.weight", False, 2, "t"),
    Role("ffn_gate_shexp.weight", "mlp.shared_expert.gate_proj", True, 2),
    Role("ffn_up_shexp.weight", "mlp.shared_expert.up_proj", True, 2),
    Role("ffn_down_shexp.weight", "mlp.shared_expert.down_proj", True, 2),
    Role("hc_attn_norm.weight", "attn_hyper_connection.hc_norm.weight", False, 1, "id", 10240),
    Role("hc_ffn_norm.weight", "mlp_hyper_connection.hc_norm.weight", False, 1, "id", 10240),
    Role("ffn_gate_inp_shexp.weight", "mlp.shared_expert_gate.weight", False, 1, "id", 2560),
]

GDN_ONLY = [
    Role("attn_qkv.weight", "linear_attn.in_proj_qkv", True, 2),
    Role("attn_gate.weight", "linear_attn.in_proj_z", True, 2),
    Role("ssm_out.weight", "linear_attn.out_proj", True, 2),
    Role("ssm_conv1d.weight", "linear_attn.conv1d.weight", False, 2, "conv1d"),
    Role("ssm_alpha.weight", "linear_attn.in_proj_a.weight", False, 2, "t"),
    Role("ssm_beta.weight", "linear_attn.in_proj_b.weight", False, 2, "t"),
    Role("ssm_a", "linear_attn.A_log", False, 1, "id", 48),
    Role("ssm_dt.bias", "linear_attn.dt_bias", False, 1, "id", 48),
    Role("ssm_norm.weight", "linear_attn.norm.weight", False, 1, "id", 128),
]

QSA_ONLY = [
    Role("attn_q.weight", "self_attn.q_proj", True, 2),
    Role("attn_k.weight", "self_attn.k_proj", True, 2),
    Role("attn_v.weight", "self_attn.v_proj", True, 2),
    Role("attn_output.weight", "self_attn.o_proj", True, 2),
    Role("attn_q_norm.weight", "self_attn.q_norm.weight", False, 1, "id", 256),
    Role("attn_k_norm.weight", "self_attn.k_norm.weight", False, 1, "id", 256),
    Role("indexer.q_proj.weight", "self_attn.indexer.index_qk_proj", True, 2, "split_qk"),
    Role("indexer.k_proj.weight", "self_attn.indexer.index_qk_proj", True, 2, "split_qk"),
    Role("indexer.q_norm.weight", "self_attn.indexer.q_layernorm.weight", False, 1, "id", 128),
    Role("indexer.k_norm.weight", "self_attn.indexer.k_layernorm.weight", False, 1, "id", 128),
]

GLOBAL = [
    Role("token_embd.weight", "model.language_model.embed_tokens.weight", False, 2, "t"),
    Role("output.weight", "lm_head", True, 2),
    Role("output_hc_norm.weight", "model.language_model.hyper_connection_mixer.hc_norm.weight", False, 1, "id", 10240),
    Role("output_hc_down.weight", "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight", False, 2, "t"),
    Role("output_hc_up.weight", "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight", False, 2, "t"),
]

PLE = [
    Role("blk.1.ple_key.weight", "ple.key_proj.weight", False, 2, "t"),
    Role("blk.1.ple_value.weight", "ple.value_proj.weight", False, 2, "t"),
    Role("blk.1.ple_norm_key.weight", "ple.norm_key.weight", False, 1, "id", 10240),
    Role("blk.1.ple_norm_query.weight", "ple.norm_query.weight", False, 1, "id", 10240),
    Role("blk.1.ple_norm_conv.weight", "ple.norm_conv.weight", False, 1, "id", 10240),
]

LAYER_PREFIX = "model.language_model.layers."


def layer_roles(layer: int, qsa: bool) -> list[tuple[str, Role]]:
    out = []
    for r in LAYER_COMMON + (QSA_ONLY if qsa else GDN_ONLY):
        out.append((f"blk.{layer}.{r.engine}", r))
    return out


def _load_index(model: Path) -> dict[str, str]:
    idx = json.load(open(model / "model.safetensors.index.json"))
    return idx["weight_map"]


def _shape(model: Path, wm: dict, name: str):
    fn = wm.get(name)
    if fn is None:
        return None
    with open(model / fn, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr[name]["shape"], hdr[name]["dtype"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    model = Path(args.model)
    wm = _load_index(model)

    # EXL3 linears are stored as <base>.{trellis,suh,svh}; non-EXL3 as <base>[.weight].
    def hf_names(r: Role, layer: int) -> list[str]:
        base = r.hf if r.engine in ("token_embd.weight", "output.weight") or r.hf.startswith("model.") \
            else LAYER_PREFIX + str(layer) + "." + r.hf
        if r.hf.startswith("model."):
            base = r.hf
        if r.engine == "output.weight":
            base = "lm_head"
        if r.engine.startswith("blk.1.ple_"):
            base = LAYER_PREFIX + "1." + r.hf
        if r.exl3:
            return [base + ".trellis", base + ".suh", base + ".svh"]
        return [base + (".weight" if not base.endswith(".weight") and not base.endswith("A_log")
                        and not base.endswith("dt_bias") else "")]

    missing = 0
    total = 0
    for layer in range(48):
        qsa = (layer % 4) == 3
        for ename, r in layer_roles(layer, qsa):
            total += 1
            names = hf_names(r, layer)
            have = all(n in wm for n in names)
            if not have:
                missing += 1
                print(f"MISSING  {ename:32} <- {names}")
    for r in GLOBAL:
        total += 1
        base = "lm_head" if r.engine == "output.weight" else r.hf
        names = [base + s for s in (".trellis", ".suh", ".svh")] if r.exl3 else [base]
        if not all(n in wm for n in names):
            missing += 1
            print(f"MISSING  {r.engine:32} <- {names}")
    for r in PLE:
        total += 1
        base = LAYER_PREFIX + "1." + r.hf
        names = [base] if base.endswith(".weight") else [base + ".weight"]
        if not all(n in wm for n in names):
            missing += 1
            print(f"MISSING  {r.engine:32} <- {names}")

    print(f"\n{total - missing}/{total} engine roles resolved ({missing} missing)")
    if args.check and missing == 0:
        # spot-check a few shapes to confirm the transpose convention
        for ename, r, hf in [
            ("blk.0.ffn_gate_inp.weight", LAYER_COMMON[6], LAYER_PREFIX + "0.mlp.gate.weight"),
            ("blk.0.attn_qkv.weight", GDN_ONLY[0], LAYER_PREFIX + "0.linear_attn.in_proj_qkv.trellis"),
            ("blk.3.indexer.q_proj.weight", QSA_ONLY[6], LAYER_PREFIX + "3.self_attn.indexer.index_qk_proj.trellis"),
        ]:
            print(f"  {ename:32} <- {hf}  {_shape(model, wm, hf)}")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
