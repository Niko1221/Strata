r"""tools/glm53_geometry.py - measure the GLM-5.3 container, and check the numbers the engine hard-codes.

The checkpoint is 141 safetensors shards, not a GGUF, so there is no metadata block to read, and every tensor is
stored FLAT: a U8 entry's recorded shape is its byte count, not [rows, cols].  So the geometry has to come from
`config.json`, and this tool's job is to check that the config's geometry reproduces every recorded byte span.
It does, for all 116,915 tensors - which is what makes the shape table below a measurement rather than a guess.

    python tools/glm53_geometry.py [model_dir] [--json]

WHAT THE HEADER DOES SAY: name, dtype, byte span.  A quantized tensor is TWO entries, `name` (the packed codes,
U8) and `name.qs` (the scales, F32).  Bits per code and group size are then solvable from the two spans, and the
answer is the same for every tensor in a family.

WHAT IT DOES NOT SAY: bits per code, group size, or which axis is contiguous.  All three are inferred here from
byte counts.  `config.json`'s own `quantization_config` says fp8 e4m3 with 128x128 blocks - that is the ORIGINAL
checkpoint's config and it is wrong for this container, which is int4 group-64.  Trusting it would have made the
code plane 4x too small and the scale plane 2x too small.

Measured on D:\models\GLM-5.3-colibri-int4-g64: 141 shards, 116,915 tensors, 419,282,314,240 B of data.  Every
weight tensor is int4 with one f32 scale per 64 elements, EXCEPT `model.embed_tokens.weight` and `lm_head.weight`,
which are int8 with one scale per row.  The container carries NO indexer tensors, NO MTP tensors, NO
hyper-connection tensors and NO shared-expert gate, so those parts of the port have to come from elsewhere.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import struct
import sys
from dataclasses import dataclass

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DEFAULT_MODEL = pathlib.Path(r"D:\models\GLM-5.3-colibri-int4-g64")

# What `derived_glm_dsa()` and `GlmDsaGuard` hard-code.  A CHECK, not a source: measure first, compare after.
CPP_PLAN = {
    "layers": 78, "hidden": 6144, "experts": 256, "active": 8, "heads": 64,
    "q_lora_rank": 2048, "kv_lora_rank": 512, "qk_nope_head_dim": 192, "qk_rope_head_dim": 64,
    "v_head_dim": 256, "moe_intermediate_size": 2048, "group_elems": 64, "code_bits": 4,
    "expert_blob": 21233664, "indexer_full_layers": 21,
}

EXPERT_MARK = ".mlp.experts."
ABSENT = ("indexer", "nextn", "mtp", "hc_", "shared_expert_gate", "kpool")


@dataclass(frozen=True)
class Tensor:
    name: str
    shard: int
    file: str
    dtype: str
    elems: int          # the flat length the header records (for U8 this is the byte count)
    nbytes: int
    data_off: int     # where the shard's data section starts
    src_off: int      # the tensor's offset inside that section, from the header's data_offsets


def read_header(path: pathlib.Path):
    """(header_bytes, header).  The header length is the data section's start, so it is part of the answer:
    a tensor's offset is relative to it, and a pack row that drops it points at the wrong bytes."""
    with path.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return n, json.loads(fh.read(n))


def read_shards(root: pathlib.Path):
    files = sorted(p for p in root.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no .safetensors under {root}")
    out = {}
    for i, p in enumerate(files):
        n, hdr = read_header(p)
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            if name in out:
                raise ValueError(f"duplicate tensor {name}")
            lo, hi = info["data_offsets"]
            elems = 1
            for d in info["shape"]:
                elems *= d
            out[name] = Tensor(name, i, p.name, info["dtype"], elems, hi - lo, 8 + n, lo)
    return files, out


def shape_of(name: str, cfg: dict) -> tuple:
    """[rows, cols] as the model's config defines it - the container itself stores the tensor flat."""
    h, moe, inter = cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["intermediate_size"]
    heads, qk, v = cfg["num_attention_heads"], cfg["qk_head_dim"], cfg["v_head_dim"]
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return (cfg["vocab_size"], h)
    if EXPERT_MARK in name or ".mlp.shared_experts." in name:
        return (moe, h)
    if re.search(r"\.mlp\.(gate|up|down)_proj\.weight$", name):
        return (inter, h)
    if "self_attn.q_a_proj" in name:
        return (cfg["q_lora_rank"], h)
    if "self_attn.q_b_proj" in name:
        return (heads * qk, cfg["q_lora_rank"])
    if "self_attn.kv_a_proj_with_mqa" in name:
        return (cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"], h)
    if "self_attn.kv_b_proj" in name:
        return (heads * (cfg["qk_nope_head_dim"] + v), cfg["kv_lora_rank"])
    if "self_attn.o_proj" in name:
        return (h, heads * v)
    if "mlp.gate.weight" in name:
        return (cfg["n_routed_experts"], h)
    if "e_score_correction_bias" in name:
        return (cfg["n_routed_experts"],)
    if "q_a_layernorm" in name:
        return (cfg["q_lora_rank"],)
    if "kv_a_layernorm" in name:
        return (cfg["kv_lora_rank"],)
    if "norm" in name:
        return (h,)
    raise ValueError(f"no shape rule for {name}")


def family_of(name: str) -> str:
    fam = re.sub(r"layers\.\d+", "layers.N", name)
    return re.sub(r"experts\.\d+", "experts.E", fam)


def bits_of(elems: int, nbytes: int) -> int:
    """Bits per code from the byte span alone: a span is `ceil(elems * bits / 8)`, so usually only one fits."""
    fits = [b for b in (2, 3, 4, 8) if (elems * b + 7) // 8 == nbytes]
    if not fits:
        raise ValueError(f"{nbytes} B is not a whole number of codes for {elems} elements")
    if len(fits) > 1:
        raise ValueError(f"{nbytes} B for {elems} elements fits {fits}: the format is ambiguous")
    return fits[0]


def group_of(rows: int, cols: int, scales: int) -> int:
    """Group size from the scale count: `scales == rows * ceil(cols / group)`.  rows == scales means per-row."""
    if rows == 0 or scales % rows:
        raise ValueError(f"{scales} scales is not a whole number of rows")
    ngroups = scales // rows
    if ngroups == 0 or cols % ngroups:
        raise ValueError(f"{scales} scales does not divide {cols} columns into whole groups")
    g = cols // ngroups
    if (cols + g - 1) // g != ngroups:
        raise ValueError(f"group {g} does not reproduce the recorded scale count")
    return g


def measure(root: pathlib.Path) -> dict:
    files, t = read_shards(root)
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))

    families, mismatch, float_ok = {}, [], 0
    for name, x in t.items():
        if name.endswith(".qs"):
            continue
        sh = shape_of(name, cfg)
        rows, cols = sh[0], (sh[1] if len(sh) > 1 else 1)
        qs = t.get(name + ".qs")
        if x.dtype == "U8":
            if qs is None:
                raise ValueError(f"{name}: codes with no .qs scale plane")
            key = (family_of(name), bits_of(rows * cols, x.nbytes), group_of(rows, cols, qs.elems))
            families[key] = families.get(key, 0) + 1
        else:
            if qs is not None:
                raise ValueError(f"{name}: a float tensor with a .qs plane")
            if x.elems != rows * cols:
                mismatch.append(name)
            else:
                float_ok += 1

    layers = sorted({int(x.name.split(".")[2]) for x in t.values() if x.name.startswith("model.layers.")})
    sparse = sorted({int(x.name.split(".")[2]) for x in t.values() if EXPERT_MARK in x.name})
    first = sorted((x for x in t.values() if EXPERT_MARK in x.name and f"layers.{sparse[0]}." in x.name
                    and not x.name.endswith(".qs")), key=lambda y: y.name)
    blob = sum(x.nbytes + t[x.name + ".qs"].nbytes for x in first)

    return {
        "shards": len(files),
        "tensors": len(t),
        "total_bytes": sum(x.nbytes for x in t.values()),
        "dense_bytes": sum(x.nbytes for x in t.values() if EXPERT_MARK not in x.name),
        "expert_bytes": sum(x.nbytes for x in t.values() if EXPERT_MARK in x.name),
        "layers": len(layers),
        "sparse_layers": len(sparse),
        "experts_per_layer": len({int(x.name.split(".")[5]) for x in first}),
        "hidden": cfg["hidden_size"],
        "experts": cfg["n_routed_experts"],
        "active": cfg["num_experts_per_tok"],
        "heads": cfg["num_attention_heads"],
        "group_elems": group_of(shape_of(first[0].name, cfg)[0], shape_of(first[0].name, cfg)[1],
                                t[first[0].name + ".qs"].elems),
        "code_bits": bits_of(shape_of(first[0].name, cfg)[0] * shape_of(first[0].name, cfg)[1], first[0].nbytes),
        "expert_blob": blob // len({int(x.name.split(".")[5]) for x in first}),
        "expert_blob_layer": blob,
        "float_tensors_checked": float_ok,
        "mismatched": mismatch[:5],
        "families": sorted((f, b, g, n) for (f, b, g), n in families.items()),
        "indexer_full_layers": sum(1 for v in cfg.get("indexer_types", []) if v == "full"),
        "present": {k: sum(1 for x in t.values() if k in x.name) for k in ABSENT},
        "config": {k: cfg.get(k) for k in ("model_type", "hidden_size", "moe_intermediate_size", "intermediate_size",
                                           "first_k_dense_replace", "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim",
                                           "qk_rope_head_dim", "v_head_dim", "index_topk", "index_head_dim",
                                           "index_n_heads", "topk_method", "scoring_func", "norm_topk_prob",
                                           "routed_scaling_factor", "num_nextn_predict_layers",
                                           "num_experts_per_tok", "n_routed_experts", "num_attention_heads")},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=str(DEFAULT_MODEL))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    m = measure(pathlib.Path(args.model))
    if args.json:
        print(json.dumps(m, indent=1, default=str))
        return 0

    print(args.model)
    for k in ("shards", "tensors", "total_bytes", "dense_bytes", "expert_bytes", "layers", "sparse_layers",
             "experts_per_layer", "expert_blob", "hidden", "experts", "active", "heads", "group_elems", "code_bits", "float_tensors_checked", "indexer_full_layers"):
        print(f"  {k:<26} {m[k]}")
    if m["mismatched"]:
        print(f"  flat lengths that do NOT match the shape table: {m['mismatched']}")
    print("  format per family (family bits group tensors)")
    for f, b, g, n in m["families"]:
        print(f"    {f:<52} {b} {g:>4} {n:>6}")
    print("  tensors the container does not carry")
    for k, n in m["present"].items():
        print(f"    {k:<26} {n}")
    print("  plan.hpp constants against the files")
    fail = 0
    for k, want in CPP_PLAN.items():
        got = m.get(k, m["config"].get(k))
        ok = got == want
        fail += 0 if ok else 1
        print(f"    {k:<26} {got}  {'ok' if ok else 'FAIL'}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())