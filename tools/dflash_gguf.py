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
import pathlib
import struct
import sys

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


def read_safetensors(path: pathlib.Path) -> dict[str, tuple[str, list[int], int]]:
    """The header: name -> (dtype, torch shape, absolute byte offset of the payload)."""
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    base = 8 + n
    out = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if spec["dtype"] != "BF16" and spec["dtype"] != "bfloat16":
            raise SystemExit(f"{name}: dtype {spec['dtype']}; the DFlash checkpoint is BF16")
        out[name] = (spec["dtype"], spec["shape"], base + spec["data_offsets"][0])
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
    return want


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("safetensors", type=pathlib.Path)
    ap.add_argument("-o", "--output", type=pathlib.Path, required=True)
    ap.add_argument("--config", type=pathlib.Path, default=None,
                    help="the drafter's config.json (default: next to the weights)")
    args = ap.parse_args()

    cfg_path = args.config or args.safetensors.parent / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"no config.json at {cfg_path}; pass --config")
    cfg = json.loads(cfg_path.read_text())
    for key in ("hidden_size", "intermediate_size", "head_dim", "num_attention_heads",
                "num_key_value_heads", "num_hidden_layers", "vocab_size", "block_size",
                "mask_token_id", "target_layer_ids"):
        if key not in cfg:
            raise SystemExit(f"config.json lacks {key}")

    want = expected(cfg)
    have = read_safetensors(args.safetensors)
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
    w.add("dflash.rope.frequency_base", float(rope.get("rope_theta", 1e7)), "f64")
    w.add("dflash.sample_from_anchor", "true")          # the DeepSpec anchor layout
    w.add("dflash.attention.causal", "false")
    w.add("dflash.markov_rank", 0, "u32")
    w.add("dflash.has_confidence_head", "false")
    w.add("dflash.source.sha256", src_sha.hexdigest())

    # torch [out, in] rows are written in file order; add_bf16 emits the GGUF shape [in, out].
    named = {hf: gg for hf, gg in FIXED.items()}
    for l in range(cfg["num_hidden_layers"]):
        for hf, gg in PER_LAYER:
            named[f"layers.{l}.{hf}"] = gg.format(l=l)
    for hf_name, shape in want.items():
        gguf_name = named[hf_name]
        _, torch_shape, off = have[hf_name]
        elems = int(np.prod(torch_shape))
        raw = payload(args.safetensors, off, elems)
        flat = bf16_to_f32(raw)
        if len(shape) == 1:
            w.add_bf16(gguf_name, flat.reshape(1, -1), shape=[int(shape[0])])
        else:
            w.add_bf16(gguf_name, flat.reshape(torch_shape))
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
    print(f"wrote {args.output}  ({args.output.stat().st_size} B, {bits} B of bf16 payloads)")
    print(f"source sha256 {src_sha.hexdigest()}")
    print("SELF-CHECK " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
