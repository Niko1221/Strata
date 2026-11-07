#!/usr/bin/env python3
"""Strip tensors (and patch metadata) from a GGUF, streaming: one pass over the data.

Built for the GLM-5.3-Flash-UNCENSORED pipeline: the converter wrote the MTP/NextN draft
layer (blk.45) that mainline llama.cpp's glm5-next runtime neither builds by default
(load_mtp=false -> the 1541/1538 hard error) nor supports in llama-imatrix's --nextn
collector (GGML_ASSERT embd_state, core dump).  The reference UD-Q2_K_XL pack carries no
draft either - Strata's engine speculates with n-grams, not the MTP head.

    python tools/glm/strip_gguf_tensors.py --gguf <in> --out <out> \
        --drop-regex '^blk\\.45\\.' --unset glm5-next.nextn_predict_layers \
        --set glm5-next.block_count=45

KV values are copied through gguf-py's reader (one scalar patched, one key dropped);
tensor bytes are copied raw - no dequant/requant - chunked from the source file, so every
kept tensor is bit-identical.  Tensor byte offsets come from the reader's data_offset.
"""
from __future__ import annotations

import argparse
import re
import sys
import time

sys.path.insert(0, "/home/simon/work/llama.cpp/gguf-py")
import gguf  # noqa: E402


def decode_field(f) -> object:
    """ReaderField -> a python value / list the writer accepts.

    The reader explodes arrays into one part per element (f.data[i] = the part index of
    element i), so scalar arrays rebuild as python lists; string arrays decode per value.
    """
    t0 = f.types[0]
    if t0 == gguf.GGUFValueType.ARRAY:
        itype = f.types[1]
        if itype == gguf.GGUFValueType.STRING:
            return [f.parts[i].tobytes().decode("utf-8", "replace") for i in f.data]
        if itype == gguf.GGUFValueType.BOOL:
            return [bool(f.parts[i][0]) for i in f.data]
        return [f.parts[i][0].item() if hasattr(f.parts[i][0], "item") else f.parts[i][0]
                for i in f.data]
    v = f.parts[f.data[0]]
    if t0 == gguf.GGUFValueType.STRING:
        return v.tobytes().decode("utf-8", "replace")
    if t0 == gguf.GGUFValueType.BOOL:
        return bool(v[0])
    return v[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-regex", required=True)
    ap.add_argument("--set", action="append", default=[], help="KEY=VALUE")
    ap.add_argument("--unset", action="append", default=[])
    a = ap.parse_args()

    old_block_count, new_block_count = 46, 45           # GLM-5.3-Flash: 45 trunk layers + the MTP draft
    drop = re.compile(a.drop_regex)
    sets = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        sets[k] = int(v)
    unsets = set(a.unset)

    r = gguf.GGUFReader(a.gguf)
    keep = [t for t in r.tensors if not drop.match(t.name)]
    print(f"keep {len(keep)} / {len(r.tensors)} tensors (drop {len(r.tensors) - len(keep)})")

    arch = decode_field(r.fields["general.architecture"])
    writer = gguf.GGUFWriter(path=None, arch=arch, use_temp_file=False)

    for name, f in r.fields.items():                    # dict preserves file order
        if name.startswith("GGUF.") or name == "general.architecture":
            continue                                    # header pseudo-KVs; arch comes from the writer
        if name in unsets:
            continue
        if name in sets:
            writer.add_key_value(name, sets[name], gguf.GGUFValueType.UINT32)
            continue
        val = decode_field(f)
        # per-layer arrays sized to the old block count lose their last entry (the MTP layer's)
        if (f.types[0] == gguf.GGUFValueType.ARRAY and isinstance(val, list)
                and len(val) == old_block_count and new_block_count < old_block_count):
            val = val[:new_block_count]
        writer.add_key_value(name, val, f.types[0],
                             sub_type=f.types[1] if f.types[0] == gguf.GGUFValueType.ARRAY else None)

    for t in keep:
        # gguf-py's TI packing reverses the shape it is given (measured: passing the reader's
        # GGUF-order shape as-is produced transposed dims and the NVFP4 byte-row check then
        # failed on n_expert=288) - hand it numpy order, i.e. the stored dims reversed.
        writer.add_tensor_info(t.name, list(int(x) for x in t.shape[::-1]), gguf.GGMLQuantizationType.F32,
                               t.n_bytes, raw_dtype=t.tensor_type)

    writer.write_header_to_file(path=a.out)
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    src = open(a.gguf, "rb")
    fout = writer.fout[0]
    writer.write_padding(fout, fout.tell())             # align the data section (32 B default)
    t0, done, copied = time.time(), 0, 0
    step = 64 * 1024 * 1024
    for t in keep:
        writer.write_padding(fout, copied)
        src.seek(t.data_offset)
        left = t.n_bytes
        while left:
            chunk = src.read(min(step, left))
            if not chunk:
                raise RuntimeError(f"short read at {t.name}")
            fout.write(chunk)
            copied += len(chunk)
            left -= len(chunk)
        done += 1
        if done % 200 == 0 or done == len(keep):
            print(f"  {done}/{len(keep)} tensors, {done * 100 // len(keep)}%, {time.time() - t0:.0f} s", flush=True)
    writer.close()
    src.close()
    print(f"stripped GGUF written: {a.out} ({time.time() - t0:.0f} s, {copied} data bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
