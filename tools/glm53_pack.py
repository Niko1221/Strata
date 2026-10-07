r"""tools/glm53_pack.py - the pack index for the GLM-5.3 int4-g64 container.

    python tools/glm53_pack.py [model_dir] -o pack/glm53 [--write-data] [--json]

WHY THIS PACK IS A LAYOUT AND NOT A CONVERSION.  The container already stores every quantized tensor in the form
the engine's arena holds it: 4-bit codes, LSB-first, two per byte, bias -8, and one f32 scale per group of 64.
`tools/pack_index.py` exists because GGUF quants must be canonicalised and fp16 scales widened; here there is
nothing to canonicalise, so the pack is the source bytes placed at recorded offsets.  Every row is kind 0
(VERBATIM) or kind 2 (F32) and `scales_fp16` is 0 for every tensor, because the container's scales are already
f32 - the pack and the engine agree on the width.  No weight bit changes.

The two things the pack still has to decide, and cannot read off the header:

  1. The shape.  The container stores every tensor flat, so [rows, cols] comes from `config.json` through
     `glm53_geometry.shape_of`, and `ne0` is `cols` because `cols` is the contiguous axis.  Read the axis the other
     way and the scale plane does not fill the tensor - the failure `pack_index.py` documents.

  2. The blob layout.  `native_experts.txt` describes one expert as [gate rows | up rows | down rows] then the
     scales, gate and up rows INTERLEAVED so one pass over the activation serves both.  Here that is
     2 x 2,048 x 3,072 + 2,048 x 3,072 + 2 x 2,048 x 96 x 4 + 2,048 x 96 x 4 = 21,233,664 B, which is 15.4x the
     1,382,400 B blob `expert_layout` was built around.

WHAT THE ENGINE STILL CANNOT READ, stated rather than papered over: `native_fmt()` has no int4-g64 entry, so the
type columns carry `INT4G64 = 100` - a Strata-assigned id, not a ggml type - and `expert_layout_load()` requires
one line per layer, which the 3 dense layers cannot have because a dense layer has no blob.  Until both land this
pack is a specification, not a loadable artifact.

Refusals, all before anything is written: a `.weight` with no `.qs`, a scale count that does not match
rows * ceil(cols / group), a `cols` that is not a whole number of groups, a tensor name with a space (the engine
parses index.txt with sscanf and no quoting, so such a name is two fields), and a layer whose expert count is not
the config's.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import glm53_geometry as GM

FILE_ID = {"dense.bin": 0, "embd.bin": 1, "experts.bin": 2}
ALIGN = 64
VERBATIM, F32_COPY = 0, 2
INT4G64 = 100          # a Strata-assigned id, not a ggml type: native_fmt() has no int4-g64 entry yet
CODE_BIAS = -8                 # the int4 family; the int8 family is -(1 << 7) = -128, measured
EMBD = "model.embed_tokens.weight"
EXPERT_MARK = ".mlp.experts."
ROLES = ("gate_proj", "up_proj", "down_proj")
COLUMNS = ("name", "file", "kind", "src_off", "src_bytes", "dst_off", "dst_bytes", "ne0", "ne1",
          "code_bits", "code_bias", "group_elems", "codebook", "has_offset", "codes_bytes", "scales_bytes",
          "offset_bytes", "scales_fp16", "act_kind")


def align_up(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def planes(t: GM.Tensor, tmap: dict, cfg: dict) -> dict:
    """One tensor as the engine form: codes plane then scale plane, in the order the loader reads them."""
    sh = GM.shape_of(t.name, cfg)
    r, c = sh[0], (sh[1] if len(sh) > 1 else 1)
    qs = tmap.get(t.name + ".qs")
    if t.dtype == "U8":
        if qs is None:
            raise ValueError(f"{t.name}: codes with no .qs scale plane")
        bits = GM.bits_of(r * c, t.nbytes)
        g = GM.group_of(r, c, qs.elems)
        if c % g:
            raise ValueError(f"{t.name}: ne0 {c} is not a whole number of groups of {g}")
        codes = (r * c * bits + 7) // 8
        scales = r * ((c + g - 1) // g) * 4
        if codes != t.nbytes or scales != qs.nbytes:
            raise ValueError(f"{t.name}: planes {codes} + {scales} do not fill the span {t.nbytes + qs.nbytes}")
        return dict(kind=VERBATIM, ne0=c, ne1=r, code_bits=bits, code_bias=-(1 << (bits - 1)), group_elems=g, codebook=0,
                    has_offset=0, codes_bytes=codes, scales_bytes=scales, offset_bytes=0, scales_fp16=0,
                    act_kind=0, dst_bytes=codes + scales, codes=t, scales=qs)
    if qs is not None:
        raise ValueError(f"{t.name}: a float tensor with a .qs plane")
    return dict(kind=F32_COPY, ne0=c, ne1=r, code_bits=0, code_bias=0, group_elems=0, codebook=0, has_offset=0,
                codes_bytes=t.nbytes, scales_bytes=0, offset_bytes=0, scales_fp16=0, act_kind=0,
                dst_bytes=t.nbytes, codes=t, scales=None)


def blob_layout(r: int, c: int, g: int, bits: int = 4) -> dict:
    """One expert blob: [gate rows | up rows | down rows], then [gate scales | up scales | down scales].

    `up_off` is the bytes per gate/up role and `down_off` is where the down scales start, which is exactly how
    `src/core/expert_source.cpp` reads them: `per[3] = {up_off, up_off, blob - down_off}`.
    """
    row_bytes = c * bits // 8
    ngroups = (c + g - 1) // g
    scale_bytes = ngroups * 4
    gu_codes = 2 * r * row_bytes
    d_codes = r * row_bytes
    gu_scales = 2 * r * scale_bytes
    d_scales = r * scale_bytes
    return dict(row_bytes=row_bytes, ngroups=ngroups, gu_codes=gu_codes, d_codes=d_codes, gu_scales=gu_scales,
                d_scales=d_scales, up_off=r * row_bytes, down_off=gu_codes + d_codes + gu_scales,
                blob=gu_codes + d_codes + gu_scales + d_scales)


def build(root: pathlib.Path) -> dict:
    files, tmap = GM.read_shards(root)
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    tensors = [x for x in tmap.values() if not x.name.endswith(".qs")]
    for x in tensors:
        if " " in x.name:
            raise ValueError(f"{x.name}: a tensor name with a space would break the index parse")

    # src_off is an offset inside the pack file named by `file`; dst_off is one running arena offset, as
    # tools/pack_index.py computes it.  The two are different numbers on purpose.
    used = {name: 0 for name in FILE_ID}
    rows, dst = [], 0
    for x in sorted(tensors, key=lambda y: y.name):
        if EXPERT_MARK in x.name:
            continue
        p = planes(x, tmap, cfg)
        fname = "embd.bin" if x.name == EMBD else "dense.bin"
        p["name"], p["file"] = x.name, FILE_ID[fname]
        p["src_bytes"] = p["codes_bytes"] + p["scales_bytes"] + p["offset_bytes"]
        p["src_off"] = used[fname]
        p["dst_off"] = dst
        used[fname] += align_up(p["dst_bytes"])
        dst += align_up(p["dst_bytes"])
        rows.append(p)

    # by_layer[layer][expert][role]: the three tensors are separate entries in the container, so the blob is
    # assembled from three places, not one.
    by_layer = {}
    for x in tensors:
        if EXPERT_MARK in x.name:
            parts = x.name.split(".")
            by_layer.setdefault(int(parts[2]), {}).setdefault(int(parts[5]), {})[parts[6]] = x
    sparse = sorted(by_layer)
    if not sparse:
        raise ValueError(f"{root}: no expert tensors - nothing to pack")
    for l in sparse:
        if len(by_layer[l]) != cfg["n_routed_experts"]:
            raise ValueError(f"layer {l}: {len(by_layer[l])} experts, not {cfg['n_routed_experts']}")

    blob = blob_layout(cfg["moe_intermediate_size"], cfg["hidden_size"], 64)
    experts, at = [], 0
    for l in sparse:
        for e in range(cfg["n_routed_experts"]):
            roles = [by_layer[l][e][r] for r in ROLES]
            experts.append(dict(layer=l, expert=e, offset=at, roles=roles))
            at += blob["blob"]
    expert_bytes = sum(x.nbytes + tmap[x.name + ".qs"].nbytes for x in tensors if EXPERT_MARK in x.name)
    if at != expert_bytes:
        raise ValueError(f"the blob layout accounts for {at} B but the container has {expert_bytes} B of experts")

    return dict(root=root, files=files, cfg=cfg, rows=rows, pool=dst, sparse=sparse, blob=blob, experts=experts,
                file_bytes=used, tmap=tmap)


def write(out: pathlib.Path, b: dict, write_data: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "index.txt").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("# strata pack index v3  --  generated by tools/glm53_pack.py; do not hand-edit\n")
        fh.write("# align %d pool %d tensors %d\n" % (ALIGN, b["pool"], len(b["rows"])))
        fh.write("# " + " ".join(COLUMNS) + "\n")
        for r in b["rows"]:
            fh.write(" ".join(str(r[c]) for c in COLUMNS) + "\n")

    blob = b["blob"]
    per_role = {r: blob["row_bytes"] + blob["ngroups"] * 4 for r in ROLES}
    with (out / "native_experts.txt").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("# strata native experts v3: int4-g64 (type %d is a Strata id, not a ggml type; "
                  "native_fmt() refuses it until the kernel lands) (n_expert %d)\n"
                  % (INT4G64, b["cfg"]["n_routed_experts"]))
        fh.write("# layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard]\n")
        # The offset is the running position over the layers that HAVE a blob, not layer * n_expert * blob:
        # a dense layer has no blob, so it contributes nothing here.  expert_layout_load() still wants one line
        # per layer, which is the gap this pack cannot close until the engine can represent a layer with no blob.
        at = 0
        for l in b["sparse"]:
            shards = {r: b["tmap"][f"model.layers.{l}.mlp.experts.0.{r}.weight"].file for r in ROLES}
            shard = shards["gate_proj"] if len(set(shards.values())) == 1 else ",".join(shards[r] for r in ROLES)
            fh.write("%d %d %d %d %d %d %d %d %s\n" % (
                l, INT4G64, INT4G64, at, blob["blob"], 0, blob["up_off"], blob["down_off"], shard))
            at += blob["blob"] * b["cfg"]["n_routed_experts"]

    with (out / "shards.txt").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("# shard file bytes\n")
        for i, p in enumerate(b["files"]):
            fh.write("%d %s %d\n" % (i, p.name, p.stat().st_size))

    if not write_data:
        return
    handles = {}

    def data(t: GM.Tensor) -> bytes:
        fh = handles.get(t.file)
        if fh is None:
            fh = handles[t.file] = (b["root"] / t.file).open("rb")
        fh.seek(t.data_off + t.src_off)
        return fh.read(t.nbytes)

    for fname in FILE_ID:
        with (out / fname).open("wb") as fh:
            for r in b["rows"]:
                if r["file"] != FILE_ID[fname]:
                    continue
                fh.seek(r["src_off"])
                fh.write(data(b["tmap"][r["name"]]))
                if r["scales_bytes"]:
                    fh.write(data(b["tmap"][r["name"] + ".qs"]))

    with (out / "experts.bin").open("wb") as fh:
        for blob_entry in b["experts"]:
            l, e = blob_entry["layer"], blob_entry["expert"]
            roles = [b["tmap"][f"model.layers.{l}.mlp.experts.{e}.{r}.weight"] for r in ROLES]
            scales = [b["tmap"][roles[i].name + ".qs"] for i in range(3)]
            codes = [data(t) for t in roles]
            sc = [data(t) for t in scales]
            row, ng = blob["row_bytes"], blob["ngroups"]
            fh.seek(blob_entry["offset"])
            for i in range(b["cfg"]["moe_intermediate_size"]):
                fh.write(codes[0][i * row:(i + 1) * row])
                fh.write(codes[1][i * row:(i + 1) * row])
            for i in range(b["cfg"]["moe_intermediate_size"]):
                fh.write(codes[2][i * row:(i + 1) * row])
            for i in range(3):
                for j in range(b["cfg"]["moe_intermediate_size"]):
                    fh.write(sc[i][j * ng * 4:(j + 1) * ng * 4])
    for fh in handles.values():
        fh.close()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=str(GM.DEFAULT_MODEL))
    ap.add_argument("-o", "--out", default="pack/glm53")
    ap.add_argument("--write-data", action="store_true", help="copy the bytes too; without it the pack is the index only")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    b = build(pathlib.Path(args.model))
    if args.write_data:
        write(pathlib.Path(args.out), b, True)
    else:
        write(pathlib.Path(args.out), b, False)
    if args.json:
        print(json.dumps({k: v for k, v in b.items() if k not in ("rows", "experts", "tmap", "files", "cfg")},
                         indent=1, default=str))
        return 0
    print(f"{args.out}: {len(b['rows'])} index rows, arena pool {b['pool']} B")
    print(f"  {len(b['experts'])} expert blobs of {b['blob']['blob']} B "
          f"({b['blob']['gu_codes']} + {b['blob']['d_codes']} + {b['blob']['gu_scales']} + {b['blob']['d_scales']})")
    print(f"  {len(b['sparse'])} sparse layers, {b['cfg']['first_k_dense_replace']} dense layers "
          f"(no blob, so no native_experts.txt line)")
    print("  not loadable yet: native_fmt() has no int4-g64 entry, and expert_layout_load() wants one line per layer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())