"""tools/q8_dense_gguf.py - a copy of a model's first shard whose natively served dense projections and head are Q8_0,
for the Pascal GP100 decode kernel (STRATA_Q8_SM60=1, docs/OLDER_GPUS.md).

The GSQ-RCO packs keep those matrices in K- and i-quants (Q6_K, IQ4_XS, Q4_K, IQ3_S, Q5_K, Q2_0, IQ4_NL), whose
decode is what a GP100 die is slowest at; STRATA_Q8_SM60 only runs Q8_0 matrices. This writes:

  * the tensor-info block with the SAME byte length: names and dims are unchanged, only the u32 type and the u64
    offset of each converted tensor change, so the data section starts where it did and every other tensor - the
    routed experts above all - keeps its absolute offset. An existing native pack (its native_experts.txt) stays
    valid; nothing is repacked.
  * the original data section, then each converted tensor's Q8_0 blocks appended (the old bytes stay, unused).
    Qwen3.8-Flash-Next IQ3_XXS: 301 tensors, +3.5 GiB, the worst requantization error 0.41% of a tensor's |w|max
    (Q8_0 is finer than every source format).

The other shards of the model are hard-linked into the output folder (--copy copies them): the engine finds them
beside --native, so --ple-gguf must name the one in the new folder too (it refuses two shard 2s).

    python tools/q8_dense_gguf.py <folder>/<model>-00001-of-00002.gguf <new folder> [--copy]

Then point the config's --native and --ple-gguf at the new folder and run the engine with STRATA_Q8_SM60=1.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths  # noqa: E402

_paths.add_gguf_py()
from gguf import GGUFReader, GGMLQuantizationType as QT  # noqa: E402
from gguf.quants import dequantize, quantize  # noqa: E402
from gguf_writer import dequantize_q2_0  # noqa: E402

# what src/core/native_dense.cpp serves natively (eligible()), and the head
SUFFIXES = (".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight", ".attn_q.weight", ".attn_k.weight",
            ".attn_v.weight", ".attn_output.weight", ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight",
            ".ffn_down_shexp.weight")
Q2_0 = 42   # GSQ's 2-bit type: gguf-py has no member for it


def is_target(name: str) -> bool:
    return name == "output.weight" or (name.startswith("blk.") and name.endswith(SUFFIXES))


def to_f32(t) -> np.ndarray:
    ne0, ne1 = int(t.shape[0]), int(t.shape[1])
    raw = np.asarray(t.data)
    if int(t.tensor_type) == Q2_0:
        return np.asarray(dequantize_q2_0(raw.tobytes()), dtype=np.float32).reshape(ne1, ne0)
    return np.asarray(dequantize(raw, t.tensor_type), dtype=np.float32).reshape(ne1, ne0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("shard1", type=Path, help="the model's first GGUF shard (the one --native names)")
    ap.add_argument("out_dir", type=Path, help="folder for the new shard 1 and the other shards' links")
    ap.add_argument("--copy", action="store_true", help="copy the other shards instead of hard-linking them")
    a = ap.parse_args()
    src = a.shard1.resolve()
    out = a.out_dir.resolve()
    if out == src.parent:
        sys.exit("the output folder must not be the model's own folder")
    dst = out / src.name

    t0 = time.time()
    r = GGUFReader(src)
    align, data_start, src_size = r.alignment, r.data_offset, os.path.getsize(src)
    mm = r.data
    jobs = []
    for t in r.tensors:
        if not is_target(t.name) or int(t.tensor_type) == int(QT.Q8_0):
            continue
        p = t.field.offset                                   # the tensor-info entry: name, n_dims, dims, type, offset
        nlen = struct.unpack_from("<Q", mm, p)[0]; p += 8
        assert bytes(mm[p:p + nlen]).decode() == t.name; p += nlen
        ndim = struct.unpack_from("<I", mm, p)[0]; p += 4 + 8 * ndim
        ttype, toff = struct.unpack_from("<IQ", mm, p)
        assert ttype == int(t.tensor_type) and data_start + toff == t.data_offset, t.name
        if int(t.shape[0]) % 32:
            print(f"  {t.name}: rows of {int(t.shape[0])} are not whole Q8_0 blocks, kept")
            continue
        jobs.append((t, p))
    print(f"{src.name}: {len(r.tensors)} tensors, {len(jobs)} to convert to Q8_0", flush=True)

    blobs, worst = [], 0.0
    for t, p in jobs:
        f = to_f32(t)
        q = np.ascontiguousarray(quantize(f, QT.Q8_0)).view(np.uint8).reshape(-1)
        assert q.nbytes == int(t.shape[0]) // 32 * 34 * int(t.shape[1]), t.name
        back = dequantize(q, QT.Q8_0).reshape(f.shape)
        worst = max(worst, float(np.abs(back - f).max() / max(float(np.abs(f).max()), 1e-30)))
        blobs.append((t, p, q))
    print(f"converted in {time.time() - t0:.0f} s; worst requantization error {worst:.2e} of a tensor's |w|max; "
          f"{sum(b[2].nbytes for b in blobs) / 2**30:.2f} GiB to append", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, tmp)
    with open(tmp, "r+b") as fh:
        end = src_size
        for t, p, blob in blobs:
            at = (end + align - 1) // align * align
            fh.seek(end); fh.write(b"\0" * (at - end))
            fh.seek(at); fh.write(blob.tobytes())
            end = at + blob.nbytes
            fh.seek(p); fh.write(struct.pack("<IQ", int(QT.Q8_0), at - data_start))
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, dst)

    v = GGUFReader(dst)                                      # every other tensor where it was, every new one as written
    conv = {t.name: blob for t, _, blob in blobs}
    orig = {t.name: t for t in r.tensors}
    assert len(v.tensors) == len(r.tensors) and v.data_offset == data_start
    for t in v.tensors:
        if t.name in conv:
            assert t.tensor_type == QT.Q8_0 and np.array_equal(np.asarray(t.data).view(np.uint8).reshape(-1), conv[t.name]), t.name
        else:
            assert t.tensor_type == orig[t.name].tensor_type and t.data_offset == orig[t.name].data_offset, t.name
    print(f"wrote {dst} ({os.path.getsize(dst) / 2**30:.2f} GiB): {len(conv)} converted, {len(v.tensors) - len(conv)} "
          "untouched, offsets checked", flush=True)

    m = re.match(r"(.*-)(\d{5})-of-(\d{5})(\.gguf)$", src.name)
    if m:
        for k in range(1, int(m[3]) + 1):
            name = f"{m[1]}{k:05d}-of-{m[3]}{m[4]}"
            if k == int(m[2]) or (out / name).exists():
                continue
            if a.copy:
                shutil.copyfile(src.parent / name, out / name)
            else:
                os.link(src.parent / name, out / name)
            print(f"{'copied' if a.copy else 'linked'} {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
