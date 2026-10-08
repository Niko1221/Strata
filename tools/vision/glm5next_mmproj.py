"""tools/vision/glm5next_mmproj.py - GLM-5.3-Flash's vision tower as an mmproj GGUF (llama.cpp's clip format,
projector type "glm5next"; the encoder graph is tools/vision/glm5next/glm5next.cpp, added to llama.cpp's mtmd by
glm5next_patch.py).

The tower (transformers' Glm5NextVisionModel): a 2x14x14 conv3d patch embedding (a still image is two identical
frames), 24 pre-norm blocks (RMSNorm, fused qkv with bias, per-head q/k RMSNorm, axial 2D rope, SwiGLU clamped at 10),
a final RMSNorm, a 2x2 conv down to 4096, and the merger (linear, LayerNorm, GELU, SwiGLU 4096 -> 10240 -> 4096,
clamped).  Its 347 tensors are BF16 in the official checkpoint (the last shard): matrices go out as F16, norms and
biases as F32.

    python glm5next_mmproj.py --ckpt <GLM-5.3-Flash dir> --out mmproj-GLM-5.3-Flash-F16.gguf [--gguf-py <dir>]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

import numpy as np


def read_bf16(path: pathlib.Path, names: set[str]) -> dict[str, np.ndarray]:
    """The named BF16 tensors of one safetensors file, as float32."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k in names:
            meta = head[k]
            if meta["dtype"] != "BF16":
                raise SystemExit(f"{k}: {meta['dtype']}, expected BF16")
            a, b = meta["data_offsets"]
            f.seek(base + a)
            raw = np.frombuffer(f.read(b - a), dtype=np.uint16)
            out[k] = (raw.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf-py", default=None, help="llama.cpp's gguf-py (when the gguf package is not installed)")
    a = ap.parse_args()
    if a.gguf_py:
        sys.path.insert(0, a.gguf_py)
    import gguf

    ck = pathlib.Path(a.ckpt)
    cfg = json.load(open(ck / "config.json"))
    vc = cfg["vision_config"]
    pre = json.load(open(ck / "processor_config.json"))["image_processor"]
    index = json.load(open(ck / "model.safetensors.index.json"))["weight_map"]
    names = {k for k in index if k.startswith("model.visual.")}
    files = sorted({index[k] for k in names})
    w = {}
    for fn in files:
        w.update(read_bf16(ck / fn, {k for k in names if index[k] == fn}))
    V = "model.visual."
    depth, hidden = int(vc["depth"]), int(vc["hidden_size"])

    out = gguf.GGUFWriter(a.out, arch="clip")
    out.add_string("general.type", "mmproj")
    out.add_string("general.name", "GLM-5.3-Flash vision")
    out.add_string("clip.projector_type", "glm5next")
    out.add_bool("clip.has_vision_encoder", True)
    out.add_bool("clip.use_silu", True)
    out.add_uint32("clip.vision.image_size", int(vc["image_size"]))
    out.add_uint32("clip.vision.patch_size", int(vc["patch_size"]))
    out.add_uint32("clip.vision.embedding_length", hidden)
    out.add_uint32("clip.vision.feed_forward_length", int(vc["intermediate_size"]))
    out.add_uint32("clip.vision.projection_dim", int(vc["out_hidden_size"]))
    out.add_uint32("clip.vision.block_count", depth)
    out.add_uint32("clip.vision.attention.head_count", int(vc["num_heads"]))
    out.add_float32("clip.vision.attention.layer_norm_epsilon", float(vc["rms_norm_eps"]))
    out.add_uint32("clip.vision.spatial_merge_size", int(vc["spatial_merge_size"]))
    out.add_array("clip.vision.image_mean", [float(x) for x in pre["image_mean"]])
    out.add_array("clip.vision.image_std", [float(x) for x in pre["image_std"]])
    out.add_float32("clip.vision.ffn_clamp", float(vc.get("swiglu_limit", 10.0)))
    out.add_uint32("clip.vision.image_min_tokens", int(pre.get("min_image_tokens", 16)))
    out.add_uint32("clip.vision.image_max_tokens", int(pre.get("max_image_tokens", 8000)))

    def mat(name, src):
        out.add_tensor(name, np.ascontiguousarray(w[src]).astype(np.float16))

    def vec(name, src):
        out.add_tensor(name, np.ascontiguousarray(w[src]).astype(np.float32))

    # the conv3d over two frames = two conv2d whose outputs add (llama.cpp's qwen2vl / glm4v form)
    pe = w[V + "patch_embed.proj.weight"]                    # [1024, 3, 2, 14, 14]
    out.add_tensor("v.patch_embd.weight", np.ascontiguousarray(pe[:, :, 0]).astype(np.float16))
    out.add_tensor("v.patch_embd.weight.1", np.ascontiguousarray(pe[:, :, 1]).astype(np.float16))
    vec("v.patch_embd.bias", V + "patch_embed.proj.bias")
    for i in range(depth):
        B, D = f"{V}blocks.{i}.", f"v.blk.{i}."
        vec(D + "ln1.weight", B + "norm1.weight")
        vec(D + "ln2.weight", B + "norm2.weight")
        mat(D + "attn_qkv.weight", B + "attn.qkv.weight")
        vec(D + "attn_qkv.bias", B + "attn.qkv.bias")
        vec(D + "attn_q_norm.weight", B + "attn.q_norm.weight")
        vec(D + "attn_k_norm.weight", B + "attn.k_norm.weight")
        mat(D + "attn_out.weight", B + "attn.proj.weight")
        vec(D + "attn_out.bias", B + "attn.proj.bias")
        for r in ("gate", "up", "down"):
            mat(D + f"ffn_{r}.weight", B + f"mlp.{r}_proj.weight")
            vec(D + f"ffn_{r}.bias", B + f"mlp.{r}_proj.bias")
    vec("v.post_ln.weight", V + "post_layernorm.weight")
    mat("mm.patch_merger.weight", V + "downsample.weight")
    vec("mm.patch_merger.bias", V + "downsample.bias")
    mat("mm.model.fc.weight", V + "merger.proj.weight")
    vec("mm.post_norm.weight", V + "merger.post_projection_norm.weight")
    vec("mm.post_norm.bias", V + "merger.post_projection_norm.bias")
    for r in ("gate", "up", "down"):
        mat(f"mm.{r}.weight", V + f"merger.{r}_proj.weight")
    used = 2 + depth * 14 + 1 + 2 + 1 + 2 + 3    # patch (w, b), blocks, post_ln, downsample, fc, post_norm, merger mlp
    if used != len(names):
        raise SystemExit(f"{len(names)} vision tensors in the checkpoint, {used} written")
    big = max(float(np.abs(v).max()) for v in w.values())
    if big > 60000:
        raise SystemExit(f"a weight of {big} does not fit F16")
    out.write_header_to_file()
    out.write_kv_data_to_file()
    out.write_tensors_to_file()
    out.close()
    print(f"{a.out}: {len(names)} tensors (largest |w| {big:.2f})")


if __name__ == "__main__":
    main()
