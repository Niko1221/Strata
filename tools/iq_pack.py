"""tools/iq_pack.py - plan v0.3 P6: a native pack for a model file whose tensors the engine reads in GGUF form
(Q2_0, IQ2_XS, IQ3_XXS, UD-Q4_K_XL).

    python tools/iq_pack.py --gguf <model>-00001-of-0000N.gguf --out pack/iq3_xxs            (standalone)
    python tools/iq_pack.py --gguf <model>-00001-of-00002.gguf --base pack/full --out ...    (share dense.bin)

The other shards of a split model are found by name.  The experts cannot be re-expressed in the Q2_0 pack form,
so this pack keeps every quantized tensor in its GGUF form:

  native_experts.txt   one line per layer: layer gu_type d_type offset blob_bytes.  The engine fills its expert
                       arena from the GGUF (tensors found by name); `offset` is the layer's place in that arena.
  experts.bin          optional (--experts-bin): the arena as a file, per layer 512 blobs of [gate rows | up rows |
                       down rows], the raw GGUF slices.  Blob size is per layer (the files mix formats).
  index.txt            the table the engine loads.  Quantized dense tensors, token_embd and output are served
                       natively from the GGUF by the engine (--native): their rows carry shape only.
  dense.bin            standalone: every other tensor in the float form the engine takes (FORM below; index kinds
                       4/5/2 = BF16/F16/F32), converted when the file stores it otherwise.
                       With --base: the base (Q2_0) pack's dense.bin, hard-linked - the float tensors are
                       byte-identical in all three ISTA model files (checked) - plus extra.bin for tensors that are
                       float here but quantized in the base pack (blk.1.ple_key).
  tokenizer/           exported from the GGUF (tools/strata_tokenizer.py), with the model's chat template.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_reader as G  # noqa: E402

FLOAT = {"BF16", "F32", "F16"}
ROLES = ("gate", "up", "down")
N_EXPERT = 512
ALIGN = 64
KIND = {"BF16": "4", "F16": "5", "F32": "2"}

# The float form the engine takes for each tensor it does not read from the GGUF (name without `blk.N.`).  A file
# that stores one otherwise is converted: UD-Q4_K_XL keeps the router, SSM gates and injections in F32 (their
# values are BF16's, so the conversion is exact), the hyper-connection projections and the PLE value in Q8_0
# (re-rounded to BF16) and the PLE conv in F32.
FORM = {
    "ffn_gate_inp.weight": "BF16", "ffn_gate_inp_shexp.weight": "BF16",
    "hc_attn_down.weight": "BF16", "hc_attn_up.weight": "BF16", "hc_attn_inject.weight": "BF16",
    "hc_ffn_down.weight": "BF16", "hc_ffn_up.weight": "BF16", "hc_ffn_inject.weight": "BF16",
    "output_hc_down.weight": "BF16", "output_hc_up.weight": "BF16",
    "indexer.k_proj.weight": "BF16", "indexer.q_proj.weight": "BF16",
    "ssm_alpha.weight": "BF16", "ssm_beta.weight": "BF16",
    "ple_key.weight": "BF16", "ple_value.weight": "BF16", "ple_conv1d.weight": "F16",
    "attn_q_norm.weight": "F32", "attn_k_norm.weight": "F32", "hc_attn_norm.weight": "F32",
    "hc_ffn_norm.weight": "F32", "output_hc_norm.weight": "F32", "indexer.q_norm.weight": "F32",
    "indexer.k_norm.weight": "F32", "ple_norm_conv.weight": "F32", "ple_norm_key.weight": "F32",
    "ple_norm_query.weight": "F32", "ssm_a": "F32", "ssm_conv1d.weight": "F32", "ssm_dt.bias": "F32",
    "ssm_norm.weight": "F32",
}
# Served from the GGUF when quantized in a type the engine has kernels for: the dense projections and the PLE key
# (NativeDense and NativeHead: native_mmvq_supported), the token embedding (NativeEmbed: iq_supported).
MMVQ_TYPES = {"Q4_0", "Q5_0", "Q8_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_NL", "IQ4_XS", "Q2_0", "IQ2_XXS",
              "IQ2_XS", "IQ3_XXS", "IQ3_S", "IQ2_S", "IQ1_M"}
EMBED_TYPES = {"IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS", "IQ1_S", "IQ1_M", "Q2_0", "Q3_K",
               "Q4_K", "Q5_K", "Q6_K", "Q5_1", "Q8_0"}
NATIVE = {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "attn_q.weight", "attn_k.weight",
          "attn_v.weight", "attn_output.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
          "ffn_down_shexp.weight", "ple_key.weight", "output.weight"}
PLE_TABLE = "per_layer_token_embd.weight"   # read from the GGUF by the engine (--ple-gguf)


class Model:
    """A model's shards (a split model's are found by name from the first), tensors looked up across them."""

    def __init__(self, first: pathlib.Path):
        m = re.fullmatch(r"(.*-)00001(-of-(\d{5})\.gguf)", first.name)
        names = [f"{m.group(1)}{i:05d}{m.group(2)}" for i in range(1, int(m.group(3)) + 1)] if m else [first.name]
        self.shards = []
        self.where = {}
        for name in names:
            path = first.parent / name
            if not path.exists():
                raise FileNotFoundError(f"missing model shard {path}")
            g = G.GGUFFile(path)
            mm = np.memmap(path, dtype=np.uint8, mode="r")
            for t in g.tensors:
                if t.name in self.where:
                    raise ValueError(f"tensor {t.name} is in two shards")
                self.where[t.name] = (g, mm, t)
            self.shards.append(g)

    def tensors(self):
        return [t for g in self.shards for t in g.tensors]

    def get(self, name):
        return self.where[name][2] if name in self.where else None

    def bytes(self, t) -> np.ndarray:
        g, mm, _ = self.where[t.name]
        return mm[g.data_start + t.offset: g.data_start + t.offset + t.expected_bytes()]


def suffix(name: str) -> str:
    return re.sub(r"^blk\.\d+\.", "", name)


def to_f32(raw: np.ndarray, type_name: str) -> np.ndarray:
    if type_name == "F32":
        return raw.view("<f4")
    if type_name == "F16":
        return raw.view("<f2").astype(np.float32)
    if type_name == "BF16":
        return (raw.view("<u2").astype(np.uint32) << 16).view(np.float32)
    if type_name == "Q8_0":
        b = raw.reshape(-1, 34)
        return (b[:, 2:].view(np.int8).astype(np.float32) * b[:, :2].copy().view("<f2").astype(np.float32)).reshape(-1)
    raise ValueError(f"no conversion from {type_name}")


def encode(x: np.ndarray, form: str) -> bytes:
    if form == "F32":
        return x.astype("<f4").tobytes()
    if form == "F16":
        return x.astype("<f2").tobytes()
    u = x.astype(np.float32).view(np.uint32)                   # BF16, round to nearest even
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2").tobytes()


def is_expert(name: str) -> bool:
    return name.startswith("blk.") and name.endswith(("_exps.weight",))


def read_index(path: pathlib.Path):
    rows, header = {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            header.append(line)
            continue
        f = line.split()
        rows[f[0]] = f
    return header, rows


def native_row(t) -> list[str]:
    ne0 = int(t.shape[0])
    ne1 = int(t.shape[1]) if len(t.shape) > 1 else 0
    return [t.name, "0", "0", "0", "0", "0", "0", str(ne0), str(ne1), "8", "0", "32"] + ["0"] * 7


def index_standalone(src, out, model: Model) -> int:
    """Every non-expert tensor: quantized ones the engine serves natively as shape-only rows, the others into
    dense.bin in their engine form."""
    rows, at, served, converted = [], 0, 0, 0
    with open(out / "dense.bin", "wb") as fo:
        for t in model.tensors():
            if is_expert(t.name) or t.name == PLE_TABLE:
                continue
            if len(t.shape) > 2:
                print("tensor %s has %d dimensions; the index holds two" % (t.name, len(t.shape)))
                return 1
            sfx = suffix(t.name)
            if t.name == "token_embd.weight" or (sfx in NATIVE and t.type_name not in FLOAT):
                ok = EMBED_TYPES if t.name == "token_embd.weight" else MMVQ_TYPES
                if t.type_name not in ok:
                    print("%s is %s, a type the engine has no native kernels for" % (t.name, t.type_name))
                    return 1
                served += 1
                rows.append(native_row(t))
                continue
            form = FORM.get(sfx)
            if form is None:
                print("tensor %s: no engine form known for it" % t.name)
                return 1
            raw = model.bytes(t)
            if t.type_name == form:
                data = raw.tobytes()
            else:
                data = encode(to_f32(raw, t.type_name), form)
                converted += 1
            ne0 = int(t.shape[0])
            ne1 = int(t.shape[1]) if len(t.shape) > 1 else 0
            rows.append([t.name, "0", KIND[form], str(at), str(len(data)), "0", str(len(data)), str(ne0), str(ne1),
                         "0", "0", "1"] + ["0"] * 7)
            fo.write(data)
            pad = (-len(data)) % ALIGN
            fo.write(b"\0" * pad)
            at += len(data) + pad
    if converted:
        print("dense.bin: %d tensors converted to their engine form" % converted)
    write_index(out, rows, src, served, 0)
    return 0


def write_index(out, rows, src, served, n_extra):
    at = 0
    for r in rows:
        r[5] = str(at)
        at += (int(r[6]) + ALIGN - 1) // ALIGN * ALIGN
    with open(out / "index.txt", "w", encoding="utf-8", newline="\n") as fo:
        fo.write("# strata pack index v3 -- generated by tools/iq_pack.py (native experts) from %s\n" % src.name)
        fo.write("# align %d pool %d tensors %d\n" % (ALIGN, at, len(rows)))
        for r in rows:
            fo.write(" ".join(r) + "\n")
    print("index.txt: %d tensors, %d served natively, %d in extra.bin, arena %.2f GiB"
          % (len(rows), served, n_extra, at / 2**30))


def index_from_base(src, base, out, model: Model) -> int:
    base_src = pathlib.Path(json.loads((base / "manifest.json").read_text(encoding="utf-8"))["source"]["shard1"])
    if not base_src.exists():
        print("cannot find the base pack's shard 1 from its manifest.json")
        return 1
    bm = Model(base_src)
    header, rows = read_index(base / "index.txt")
    new_rows, extra = [], []
    served = 0
    for name, f in rows.items():
        t, bt = model.get(name), bm.get(name)
        if t is None or bt is None:
            print("tensor %s missing from one of the models" % name)
            return 1
        if t.type_name in FLOAT and bt.type_name in FLOAT:
            if t.type_name != bt.type_name or t.shape != bt.shape or \
                    not np.array_equal(model.bytes(t), bm.bytes(bt)):
                print("float tensor %s differs from the base model; this pack cannot reuse its dense.bin" % name)
                return 1
            new_rows.append(list(f))
        elif t.type_name in FLOAT:
            if t.type_name != "BF16":
                print("unexpected float type %s for %s" % (t.type_name, name))
                return 1
            nbytes = t.expected_bytes()
            off = sum(len(b) + (-len(b)) % ALIGN for b in extra)
            extra.append(model.bytes(t).tobytes())
            # file 3 = extra.bin, raw BF16 (index kind 4)
            new_rows.append([name, "3", "4", str(off), str(nbytes), "0", str(nbytes), f[7], f[8]] + ["0"] * 10)
        else:
            served += 1
            new_rows.append([name, f[1], "0", "0", "0", "0", "0", f[7], f[8], "8", "0", "32"] + ["0"] * 7)
    write_index(out, new_rows, src, served, len(extra))
    with open(out / "extra.bin", "wb") as fo:
        for b in extra:
            fo.write(b)
            fo.write(b"\0" * ((-len(b)) % ALIGN))
    dense = out / "dense.bin"
    if not dense.exists():
        try:
            os.link(base / "dense.bin", dense)
        except OSError:
            shutil.copyfile(base / "dense.bin", dense)
    if (base / "tokenizer").exists() and not (out / "tokenizer").exists():
        shutil.copytree(base / "tokenizer", out / "tokenizer")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="the model's first shard (the others are found by name)")
    ap.add_argument("--base", help="optional: a Q2_0 canonical pack whose dense.bin holds the shared float tensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-experts", action="store_true", help="rewrite the index only")
    ap.add_argument("--experts-bin", action="store_true",
                    help="also write experts.bin (the engine otherwise reads the experts from the GGUF itself)")
    a = ap.parse_args()
    src = pathlib.Path(a.gguf).resolve()
    base = pathlib.Path(a.base).resolve() if a.base else None
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    model = Model(src)
    rc = index_from_base(src, base, out, model) if a.base else index_standalone(src, out, model)
    if rc:
        return rc
    if not (out / "tokenizer" / "vocab.json").exists() or not (out / "tokenizer" / "chat_template.jinja").exists():
        subprocess.run([sys.executable, str(HERE / "strata_tokenizer.py"), "--gguf", str(src), "--out", str(out)],
                       check=True)   # writes <out>/tokenizer/

    # ---- the experts
    names = [t.name for t in model.tensors()]
    n_layers = 1 + max(int(n.split(".")[1]) for n in names if n.startswith("blk.") and n.endswith("_exps.weight"))
    layout, offset = [], 0
    for l in range(n_layers):
        ts = [model.get("blk.%d.ffn_%s_exps.weight" % (l, r)) for r in ROLES]
        per = [t.expected_bytes() // N_EXPERT for t in ts]
        if per[0] != per[1] or ts[0].type_name != ts[1].type_name:
            print("layer %d: gate and up differ in type" % l)
            return 1
        blob = per[0] + per[1] + per[2]
        layout.append((l, ts[0].type_id, ts[2].type_id, offset, blob, ts))
        offset += blob * N_EXPERT
    with open(out / "native_experts.txt", "w", encoding="utf-8", newline="\n") as fo:
        fo.write("# strata native experts v3: layer gu_type d_type offset blob_bytes (n_expert %d, total %d; the "
                 "engine takes the experts from %s by tensor name, or from experts.bin)\n"
                 % (N_EXPERT, offset, src.name))
        for l, gt, dt, off, blob, ts in layout:
            fo.write("%d %d %d %d %d\n" % (l, gt, dt, off, blob))
    if a.skip_experts or not a.experts_bin:
        if (out / "experts.bin").exists() and not a.experts_bin:
            print("note: %s/experts.bin exists; the engine reads it instead of the GGUF" % out)
        return 0
    path = out / "experts.bin"
    if path.exists() and path.stat().st_size == offset:
        print("experts.bin exists with the right size; not rewritten")
        return 0
    with open(path, "wb") as fo:
        for l, gt, dt, off, blob, ts in layout:
            parts = [model.bytes(t).reshape(N_EXPERT, -1) for t in ts]
            chunk = np.concatenate(parts, axis=1)          # (512, blob): gate | up | down per expert
            assert chunk.shape == (N_EXPERT, blob)
            fo.write(chunk.tobytes())
            if l % 8 == 0:
                print("  layer %2d  %-8s/%-7s blob %8d  at %.2f GiB" % (l, ts[0].type_name, ts[2].type_name, blob,
                                                                        off / 2**30), flush=True)
    print("experts.bin: %d layers, %.2f GiB" % (n_layers, offset / 2**30))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
