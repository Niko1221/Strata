"""Strata P0.S9 - build the pack-layout expert arena from the real GGUF.

The architecture (§3.2) defines one expert blob as 1,382,400 bytes:

    gate_up.codes   2 x 640 rows x 640 B   rows interleaved g0,u0,g1,u1,...  (share input x)
    down.codes      2,560 rows x 160 B
    gate_up.scales  2 x 640 x 40 x FP16
    down.scales     2,560 x 10 x FP16

and that totals exactly 819,200 + 409,600 + 102,400 + 51,200 = 1,382,400 B, matching the artifact's
Q2_0 expert size to the byte. The conversion is a pure relayout: Q2_0 codes and their FP16 block
scale are copied unchanged, so dequant(pack) == dequant(ggml) bit-exactly.

Why build this at all rather than read the GGUF in place: the GGUF stores gate, up and down as three
separate tensors, so a fused gate/up pass would read two streams from three arrays. The pack
interleaves gate and up ROWS so one pass over the activation vector serves both, which is the whole
point of the blob layout.

Expert index is the LAST axis of [in, out, experts] (P0.S6), and ne[0] varies fastest, so expert `e`
is one CONTIGUOUS 460,800-byte run inside each role tensor. Extraction is a strided copy.

Run: python tools/pack_layer.py [n_layers]     (default 1; 48 = the full 34 GB arena)
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
STRATA = REPO.parents[min(1, len(REPO.parents) - 1)]   # the dev layout's default shards (setup.py passes --gguf)
SHARD1 = STRATA / "Q2_0" / "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"

QK, BLOCK_BYTES = 64, 18          # Q2_0 (shape-independent: one fp16 scale per 64 weights, 18 bytes/block)

# H, FF, NE and everything derived from them used to be fixed at Qwen3.8-Flash-Next's own shape (2560/640/512).
# `configure()` recomputes all of this for whatever shape the target binary was compiled for
# (`strata::core::kCompiledGeometry`, read indirectly via --shape at build time) - see tools/strata_pack.py's
# `build()`, which calls it with the shape read from the GGUF itself before packing. The defaults below keep
# `python tools/pack_layer.py` runnable standalone against the original dev-layout checkpoint, unchanged.
H, FF, NE = 2560, 640, 512
ROW_BYTES_GU = ROW_BYTES_D = SCALES_GU = SCALES_D = ROLE_BYTES = BLOB_BYTES = 0
OFF: dict[str, int] = {}


def configure(h: int, ff: int, ne: int) -> None:
    """Recompute every shape-derived constant for an arbitrary (H, FF, NE) - the inverse of this file's old
    hardcoded assertion that only 2560/640/512 was valid.  QK/BLOCK_BYTES are Q2_0's own fixed block geometry,
    not the model's; H and FF only need to be multiples of QK (one fp16 scale per 64 weights)."""
    global H, FF, NE, ROW_BYTES_GU, ROW_BYTES_D, SCALES_GU, SCALES_D, ROLE_BYTES, BLOB_BYTES, OFF
    if h % QK or ff % QK:
        raise ValueError("H=%d and FF=%d must both be multiples of QK=%d (one Q2_0 scale per %d weights)"
                          % (h, ff, QK, QK))
    H, FF, NE = h, ff, ne
    ROW_BYTES_GU = H * 2 // 8         # codes for one H-wide row
    ROW_BYTES_D = FF * 2 // 8         # codes for one FF-wide row
    SCALES_GU = H // QK               # blocks per gate/up row
    SCALES_D = FF // QK               # blocks per down row
    ROLE_BYTES = H * FF * BLOCK_BYTES // QK     # bytes per role per expert
    BLOB_BYTES = 3 * ROLE_BYTES
    assert 2 * FF * ROW_BYTES_GU + H * ROW_BYTES_D + 2 * FF * SCALES_GU * 2 + H * SCALES_D * 2 == BLOB_BYTES
    OFF = blob_offsets()


def shape_from_gguf(gguf_path, layer: int = 0) -> tuple[int, int, int]:
    """(H, FF, NE) read from one layer's expert tensors - no architecture-specific metadata key needed. Expert
    index is the LAST axis of [in, out, experts] (gate/up: [H, FF, NE]; down: [FF, H, NE], the transpose), cross-
    checked between gate and down so a model that stores them inconsistently is refused rather than mis-packed."""
    g = GGUFFile(gguf_path)
    by_name = {t.name: t for t in g.tensors}
    gate = by_name[f"blk.{layer}.ffn_gate_exps.weight"]
    down = by_name[f"blk.{layer}.ffn_down_exps.weight"]
    h, ff, ne = int(gate.shape[0]), int(gate.shape[1]), int(gate.shape[2])
    if (int(down.shape[0]), int(down.shape[1]), int(down.shape[2])) != (ff, h, ne):
        raise ValueError("blk.%d: gate_exps is [%d,%d,%d] but down_exps is %s - expected the transpose [FF,H,NE]"
                          % (layer, h, ff, ne, list(down.shape)))
    return h, ff, ne


def blob_offsets() -> dict[str, int]:
    gu_codes = 0
    d_codes = gu_codes + 2 * FF * ROW_BYTES_GU
    gu_scales = d_codes + H * ROW_BYTES_D
    d_scales = gu_scales + 2 * FF * SCALES_GU * 2
    end = d_scales + H * SCALES_D * 2
    assert end == BLOB_BYTES, end
    return {"gate_up_codes": gu_codes, "down_codes": d_codes,
            "gate_up_scales": gu_scales, "down_scales": d_scales}


configure(H, FF, NE)   # the original hardcoded shape, so every module-level name below stays valid on import


def build(n_layers: int, out_path: pathlib.Path) -> dict:
    g = GGUFFile(SHARD1)
    by_name = {t.name: t for t in g.tensors}
    src = open(SHARD1, "rb")
    manifest = {"blob_bytes": BLOB_BYTES, "offsets": OFF, "layers": [], "n_layers": n_layers}
    total = n_layers * NE * BLOB_BYTES
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as out:
        for layer in range(n_layers):
            roles = {}
            for role in ("gate", "up", "down"):
                t = by_name[f"blk.{layer}.ffn_{role}_exps.weight"]
                roles[role] = t
                assert t.type_name == "Q2_0", t.type_name
                # expert e is a contiguous run inside the role tensor
                per_expert = t.elements // NE
                assert per_expert * BLOCK_BYTES // QK == ROLE_BYTES, per_expert
            layer_off = out.tell()
            # vectorized over the layer's 512 experts (the same bytes the per-block loop wrote; tools/strata_pack.py
            # verify checks them): each role is (512 experts, rows, blocks per row, 18 bytes)
            def role_blocks(role, rows, per_row):
                t = roles[role]
                n = NE * ROLE_BYTES
                src.seek(g.data_start + t.offset)
                a = np.frombuffer(src.read(n), dtype=np.uint8)
                assert a.size == n, (role, a.size)
                return a.reshape(NE, rows, per_row, BLOCK_BYTES)
            gate = role_blocks("gate", FF, SCALES_GU)
            up = role_blocks("up", FF, SCALES_GU)
            down = role_blocks("down", H, SCALES_D)
            gu = np.stack([gate, up], axis=2).reshape(NE, 2 * FF, SCALES_GU, BLOCK_BYTES)   # rows g0,u0,g1,u1,...
            blobs = np.concatenate([
                gu[:, :, :, 2:].reshape(NE, -1),        # gate/up codes
                down[:, :, :, 2:].reshape(NE, -1),      # down codes
                gu[:, :, :, :2].reshape(NE, -1),        # gate/up fp16 scales
                down[:, :, :, :2].reshape(NE, -1),      # down fp16 scales
            ], axis=1)
            assert blobs.shape == (NE, BLOB_BYTES), blobs.shape
            out.write(blobs.tobytes())
            manifest["layers"].append({"layer": layer, "offset": layer_off,
                                       "role_offsets": {r: by_name[f"blk.{layer}.ffn_{r}_exps.weight"].offset
                                                        for r in ("gate", "up", "down")}})
            print(f"  layer {layer}: wrote {NE} blobs at {layer_off}")
    src.close()
    manifest["total_bytes"] = total
    (out_path.with_suffix(".json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    dest = REPO / "pack" / ("layer0.bin" if n == 1 else f"experts-{n}l.bin")
    print(f"building {n} layer(s) -> {dest}  ({(n*NE*BLOB_BYTES)/2**30:.2f} GiB)")
    m = build(n, dest)
    print(f"done: {m['total_bytes']/2**30:.2f} GiB, {n*NE} blobs")
