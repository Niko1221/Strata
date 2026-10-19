#!/usr/bin/env python3
"""Pack a DeepSpec DFlash checkpoint (safetensors) into the standalone GGUF Strata reads.

    python tools/dflash_gguf.py PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash/model.safetensors \
        -o dflash.gguf [--config config.json]

The published checkpoint (e.g. PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash) ships 58 BF16 tensors and
NO embedding and NO LM head - both are bound from the target at load.  Geometry comes from the
config.json next to the weights (or --config); everything written here is what
include/strata/core/dflash.hpp validates.  Tensor names are written in llama.cpp's GGUF family
(blk.N.attn_q.weight ...); the C++ loader also accepts the raw-HF names for files converted
elsewhere.

Self-checking: after writing, the file is read back with tools/gguf_reader.py and the round trip
through bf16 is asserted bit-exact (the checkpoint is already BF16, so repacking must not move a
bit).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile  # noqa: E402
from gguf_writer import GGUFWriter, bf16_to_f32  # noqa: E402

# HF checkpoint name -> (canonical name, GGUF shape from the torch shape [out, in] / [dim])
FIXED = {
    "fc.weight": "fc.weight",
    "hidden_norm.weight": "enc.output_norm.weight",
    "norm.weight": "output_norm.weight",
}
PER_LAYER = [
    ("input_layernorm.weight", "blk.{l}.attn_norm.weight"),
    ("post_attention_layernorm.weight", "blk.{l}.ffn_norm.weight"),
    ("self_attn.q_proj.weight", "blk.{l}.attn_q.weight"),
    ("self_attn.k_proj.weight", "blk.{l}.attn_k.weight"),
    ("self_attn.v_proj.weight", "blk.{l}.attn_v.weight"),
    ("self_attn.o_proj.weight", "blk.{l}.attn_output.weight"),
    ("self_attn.q_norm.weight", "blk.{l}.attn_q_norm.weight"),
    ("self_attn.k_norm.weight", "blk.{l}.attn_k_norm.weight"),
    ("mlp.gate_proj.weight", "blk.{l}.ffn_gate.weight"),
    ("mlp.up_proj.weight", "blk.{l}.ffn_up.weight"),
    ("mlp.down_proj.weight", "blk.{l}.ffn_down.weight"),
]



def is_dflash2(cfg: dict) -> bool:
    return "DFlash2DraftModel" in (cfg.get("architectures") or [])


def dflash2_names(cfg: dict) -> dict[str, str]:
    names = {
        "candidate_selector.predecessor_codebook.weight": "selector_predecessor.weight",
        "candidate_selector.successor_codebook.weight": "selector_successor.weight",
        "candidate_selector.hidden_projection.weight": "selector_hidden.weight",
    }
    for l in range(cfg["num_hidden_layers"]):
        for hf, gg in (("attention_conv", "attn_conv"), ("mlp_conv", "ffn_conv")):
            names[f"layers.{l}.{hf}.base_kernel"] = f"blk.{l}.{gg}_base"
            names[f"layers.{l}.{hf}.kernel_projection.weight"] = f"blk.{l}.{gg}_proj.weight"
    return names


def normalize_config(cfg: dict) -> dict:
    original = cfg
    cfg = dict(cfg)
    if is_dflash2(cfg):
        cfg.update(cfg.get("dflash_config") or {})
    for key, default in (("hidden_act", "silu"), ("rms_norm_eps", 1e-6),
                         ("attention_bias", False), ("is_causal", False),
                         ("input_embedding_scale", 1.0), ("output_multiplier", 1.0),
                         ("final_logit_softcapping", 0.0), ("partial_rotary_factor", 1.0),
                         ("markov_rank", 0), ("enable_confidence_head", False)):
        if cfg.get(key, default) != default:
            raise ValueError(f"unsupported {key}; runtime requires {default} (non-causal attention)")
    rope = cfg.get("rope_parameters") or {}
    theta = rope.get("rope_theta", cfg.get("rope_theta", 1e7))
    if not isinstance(theta, (int, float)) or not math.isfinite(theta) or theta <= 0:
        raise ValueError("rope_theta must be finite and positive")
    if rope.get("rope_type", "default") != "default" or cfg.get("rope_scaling") or rope.get("partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("only unscaled, full RoPE is supported")
    for key in ("hidden_size", "intermediate_size", "head_dim", "num_attention_heads",
                "num_key_value_heads", "num_hidden_layers", "vocab_size", "block_size"):
        if not isinstance(cfg.get(key), int) or isinstance(cfg[key], bool) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["num_attention_heads"] % cfg["num_key_value_heads"]:
        raise ValueError("query heads must be divisible by KV heads")
    taps = cfg.get("target_layer_ids")
    if not isinstance(taps, list) or not taps or any(type(x) is not int or x < 0 for x in taps) or taps != sorted(set(taps)):
        raise ValueError("target_layer_ids must be nonempty and strictly increasing")
    if type(cfg.get("mask_token_id")) is not int or not 0 <= cfg["mask_token_id"] < cfg["vocab_size"]:
        raise ValueError("mask_token_id must be inside the vocabulary")
    types = cfg.get("layer_types")
    if types is not None and (len(types) != cfg["num_hidden_layers"] or
                              any(t not in ("full_attention", "sliding_attention") for t in types)):
        raise ValueError("invalid layer_types")
    sliding = types is not None and "sliding_attention" in types
    if sliding and (not cfg.get("use_sliding_window", False) or
                    types != ["sliding_attention"] * cfg["num_hidden_layers"]):
        raise ValueError("runtime requires uniform sliding attention consistent with use_sliding_window")
    if not is_dflash2(cfg):
        if cfg.get("sample_from_anchor", True) is not True or cfg.get("target_hc_mean", False):
            raise ValueError("plain DFlash requires the anchor layout and standard target taps")
        if sliding or cfg.get("use_sliding_window", False):
            raise ValueError("plain DFlash requires full attention")
        return original
    H = cfg["hidden_size"]
    for key in ("conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k", "block_size"):
        if not isinstance(cfg.get(key), int) or isinstance(cfg[key], bool) or cfg[key] <= 0:
            raise ValueError(f"DFlash2 config requires a positive integer {key}")
    if H % cfg["conv_group_size"] or not 2 <= cfg["block_size"] <= 8 or cfg["selector_top_k"] > min(64, cfg["vocab_size"]) or cfg["selector_rank"] > 1024 or cfg["conv_kernel_size"] > 8:
        raise ValueError("DFlash2 geometry exceeds the experimental runtime limits")
    if cfg.get("is_causal", False):
        raise ValueError("DFlash2 requires non-causal block attention")
    if cfg.get("sample_from_anchor", False):
        raise ValueError("DFlash2 requires sample_from_anchor=false")
    if cfg.get("hidden_act", "silu") != "silu" or cfg.get("rms_norm_eps", 1e-6) != 1e-6:
        raise ValueError("experimental runtime requires SiLU and rms_norm_eps=1e-6")
    if cfg.get("attention_bias", False):
        raise ValueError("experimental runtime requires attention without biases")
    if cfg.get("use_sliding_window", False) and not cfg.get("sliding_window"):
        raise ValueError("sliding_window is required when enabled")
    window = cfg.get("sliding_window") if cfg.get("use_sliding_window", False) else 0
    if window and (not isinstance(window, int) or isinstance(window, bool) or window < 1):
        raise ValueError("sliding_window must be a positive integer")
    layer_types = cfg.get("layer_types")
    if window and cfg.get("is_causal") is not False:
        raise ValueError("sliding attention requires explicit non-causal is_causal=false")
    if window and layer_types != ["sliding_attention"] * cfg["num_hidden_layers"]:
        raise ValueError("DFlash2 requires uniform sliding attention layers")
    cfg["sliding_window"] = window or 0
    official = (H == 2560 and cfg.get("head_dim") == 128 and
                cfg.get("num_attention_heads") == 32 and cfg.get("num_key_value_heads") == 8 and
                cfg.get("target_layer_ids") == [5, 15, 25, 35, 45])
    cfg["target_hc_mean"] = cfg.get("target_hc_mean", official)
    if not isinstance(cfg["target_hc_mean"], bool):
        raise ValueError("target_hc_mean must be a boolean")
    if cfg.get("output_multiplier", 1.0) != 1.0 or cfg.get("final_logit_softcapping", 0.0):
        raise ValueError("experimental runtime does not implement logit scaling/softcapping")
    return cfg

def read_safetensors(path: pathlib.Path) -> dict[str, tuple[str, list[int], int]]:
    """The header: name -> (dtype, torch shape, absolute byte offset of the payload)."""
    size = path.stat().st_size
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors header")
        n = struct.unpack("<Q", prefix)[0]
        if n > size - 8:
            raise ValueError("safetensors header exceeds the file")
        header = json.loads(f.read(n))
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    base = 8 + n
    out, ranges = {}, []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if spec.get("dtype") not in ("BF16", "bfloat16"):
            raise ValueError(f"{name}: expected BF16 weights")
        shape, offsets = spec.get("shape"), spec.get("data_offsets")
        if (not isinstance(shape, list) or not 1 <= len(shape) <= 3 or
                any(type(x) is not int or x < 1 for x in shape) or
                not isinstance(offsets, list) or len(offsets) != 2 or
                any(type(x) is not int for x in offsets)):
            raise ValueError(f"{name}: invalid shape or offsets")
        start, end = offsets
        if start < 0 or end - start != math.prod(shape) * 2 or end > size - base:
            raise ValueError(f"{name}: invalid tensor byte range")
        ranges.append((start, end))
        out[name] = (spec["dtype"], shape, base + start)
    ordered = sorted(ranges)
    if any(right[0] < left[1] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("overlapping safetensors payloads")
    return out


def payload(path: pathlib.Path, off: int, elems: int) -> np.ndarray:
    with path.open("rb") as f:
        f.seek(off)
        return np.frombuffer(f.read(elems * 2), dtype="<u2")


def expected(cfg: dict) -> dict[str, list[int]]:
    """Every tensor the checkpoint must hold -> its torch shape [out, in] or [dim]."""
    H = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    D = cfg["head_dim"]
    Q = cfg["num_attention_heads"] * D
    KV = cfg["num_key_value_heads"] * D
    L = cfg["num_hidden_layers"]
    F = H * len(cfg["target_layer_ids"])
    want = {"fc.weight": [H, F], "hidden_norm.weight": [H], "norm.weight": [H]}
    for l in range(L):
        for hf, _ in PER_LAYER:
            # PER_LAYER entries are already relative to layers.N.
            shape = {
                "input_layernorm.weight": [H], "post_attention_layernorm.weight": [H],
                "self_attn.q_proj.weight": [Q, H], "self_attn.k_proj.weight": [KV, H],
                "self_attn.v_proj.weight": [KV, H], "self_attn.o_proj.weight": [H, Q],
                "self_attn.q_norm.weight": [D], "self_attn.k_norm.weight": [D],
                "mlp.gate_proj.weight": [I, H], "mlp.up_proj.weight": [I, H],
                "mlp.down_proj.weight": [H, I],
            }[hf]
            want[f"layers.{l}.{hf}"] = shape
    if is_dflash2(cfg):
        rank = cfg["selector_rank"]
        kernel = cfg["conv_kernel_size"]
        projected = 2 * kernel * (H // cfg["conv_group_size"])
        want.update({
            "candidate_selector.predecessor_codebook.weight": [cfg["vocab_size"], rank],
            "candidate_selector.successor_codebook.weight": [cfg["vocab_size"], rank],
            "candidate_selector.hidden_projection.weight": [rank, H],
        })
        for l in range(L):
            for side in ("attention_conv", "mlp_conv"):
                want[f"layers.{l}.{side}.base_kernel"] = [2, kernel, H]
                want[f"layers.{l}.{side}.kernel_projection.weight"] = [projected, H]
    return want


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("safetensors", type=pathlib.Path)
    ap.add_argument("-o", "--output", type=pathlib.Path, required=True)
    ap.add_argument("--config", type=pathlib.Path, default=None,
                    help="the drafter's config.json (default: next to the weights)")
    args = ap.parse_args()

    if args.safetensors.resolve() == args.output.resolve():
        raise ValueError("source and output must differ")

    cfg_path = args.config or args.safetensors.parent / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"no config.json at {cfg_path}; pass --config")
    cfg = normalize_config(json.loads(cfg_path.read_text()))
    for key in ("hidden_size", "intermediate_size", "head_dim", "num_attention_heads",
                "num_key_value_heads", "num_hidden_layers", "vocab_size", "block_size",
                "mask_token_id", "target_layer_ids"):
        if key not in cfg:
            raise SystemExit(f"config.json lacks {key}")

    want = expected(cfg)
    have = read_safetensors(args.safetensors)
    if is_dflash2(cfg):
        for name in ("predecessor_codebook", "successor_codebook"):
            bare = "candidate_selector." + name
            if bare in have:
                if bare + ".weight" in have:
                    raise ValueError(f"duplicate selector tensor: {bare}")
                have[bare + ".weight"] = have.pop(bare)

    missing = sorted(set(want) - set(have))
    extra = sorted(set(have) - set(want))
    if missing:
        raise SystemExit(f"missing tensors: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if extra:
        raise SystemExit(f"unexpected tensors (the DeepSpec drafter strips embed/lm_head): {extra[:8]}")
    for name, shape in want.items():
        if list(have[name][1]) != shape:
            raise SystemExit(f"{name}: shape {have[name][1]}, expected {shape}")

    src_sha = hashlib.sha256()
    with args.safetensors.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            src_sha.update(chunk)

    w = GGUFWriter()
    rope = cfg.get("rope_parameters", {}) or {}
    w.add("general.architecture", "dflash")
    w.add("general.name", cfg.get("_name_or_path", "DFlash drafter"))
    w.add("dflash.embedding_length", cfg["hidden_size"], "u32")
    w.add("dflash.block_count", cfg["num_hidden_layers"], "u32")
    w.add("dflash.attention.head_count", cfg["num_attention_heads"], "u32")
    w.add("dflash.attention.head_count_kv", cfg["num_key_value_heads"], "u32")
    w.add("dflash.attention.key_length", cfg["head_dim"], "u32")
    w.add("dflash.feed_forward_length", cfg["intermediate_size"], "u32")
    w.add("dflash.vocab_size", cfg["vocab_size"], "u32")
    w.add("dflash.block_size", cfg["block_size"], "u32")
    w.add("dflash.mask_token_id", cfg["mask_token_id"], "u32")
    w.add("dflash.target_layers", [int(x) for x in cfg["target_layer_ids"]], "array:u32")
    w.add("dflash.rope.frequency_base", float(rope.get("rope_theta", cfg.get("rope_theta", 1e7))), "f64")
    w.add("dflash.sample_from_anchor", "false" if is_dflash2(cfg) else "true")          # the DeepSpec anchor layout
    w.add("dflash.attention.causal", "false")
    if is_dflash2(cfg):
        for key in ("conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k"):
            w.add("dflash." + key, cfg[key], "u32")
        # HF target_layer_ids refer to OUTPUTS; llama.cpp converts them to INPUT boundaries.
        # Retain the HF IDs and mark their interpretation explicitly for Strata.
        w.add("dflash.target_layer_inputs", "false")
        w.add("dflash.attention.sliding_window", cfg["sliding_window"], "u32")
        # The official Flash-Next checkpoint uses completed HC stream means.
        w.add("dflash.target_hc_mean", "true" if cfg["target_hc_mean"] else "false")

    w.add("dflash.markov_rank", 0, "u32")
    w.add("dflash.has_confidence_head", "false")
    w.add("dflash.source.sha256", src_sha.hexdigest())

    # torch [out, in] rows are written in file order; add_bf16 emits the GGUF shape [in, out].
    named = {hf: gg for hf, gg in FIXED.items()}
    for l in range(cfg["num_hidden_layers"]):
        for hf, gg in PER_LAYER:
            named[f"layers.{l}.{hf}"] = gg.format(l=l)
    if is_dflash2(cfg):
        named.update(dflash2_names(cfg))
    for hf_name, shape in want.items():
        gguf_name = named[hf_name]
        _, torch_shape, off = have[hf_name]
        elems = int(np.prod(torch_shape))
        raw = payload(args.safetensors, off, elems)
        flat = bf16_to_f32(raw)
        if len(shape) == 1:
            w.add_bf16(gguf_name, flat.reshape(1, -1), shape=[int(shape[0])])
        elif len(shape) == 3:
            w.add_bf16(gguf_name, flat.reshape(-1, shape[-1]), shape=list(reversed(shape)))
        else:
            w.add_bf16(gguf_name, flat.reshape(torch_shape))
    destination = args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dflash-export-", dir=destination.parent) as folder:
        args.output = pathlib.Path(folder) / destination.name
        w.write(args.output)

        # Read back: the inventory, the metadata and a bit-exact bf16 round trip on every tensor.
        g = GGUFFile(args.output)
        ok = True
        by = {t.name: t for t in g.tensors}
        if len(g.tensors) != len(want):
            print(f"FAIL: {len(g.tensors)} tensors on disk, expected {len(want)}")
            ok = False
        if g.metadata.get("general.architecture") != "dflash":
            print("FAIL: architecture"); ok = False
        if g.metadata.get("dflash.block_size") != cfg["block_size"]:
            print("FAIL: block_size"); ok = False
        if list(g.metadata.get("dflash.target_layers") or []) != [int(x) for x in cfg["target_layer_ids"]]:
            print("FAIL: target_layers"); ok = False
        bits = 0
        for hf_name, shape in want.items():
            t = by.get(named[hf_name])
            if t is None or t.type_name != "BF16" or list(t.shape) != list(reversed(shape)):
                print(f"FAIL: {named[hf_name]} dir {t and (t.type_name, t.shape)}"); ok = False
                continue
            elems = int(np.prod(shape))
            with args.output.open("rb") as f:
                f.seek(g.data_start + t.offset)
                disk = np.frombuffer(f.read(elems * 2), dtype="<u2")
            raw = payload(args.safetensors, have[hf_name][2], elems)
            if not np.array_equal(disk, raw):
                print(f"FAIL: {named[hf_name]} moved bits in the bf16 round trip")
                ok = False
            bits += elems * 2
        print(f"wrote {destination}  ({args.output.stat().st_size} B, {bits} B of bf16 payloads)")
        print(f"source sha256 {src_sha.hexdigest()}")
        print("SELF-CHECK " + ("PASS" if ok else "FAIL"))
        if ok:
            args.output.replace(destination)
        return 0 if ok else 1



if __name__ == "__main__":
    raise SystemExit(main())
