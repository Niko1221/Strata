"""tools/ple_key_bf16.py - store a model's blk.1.ple_key as BF16 when its GGUF quantized it to anything but Q2_0.

    python tools/ple_key_bf16.py --gguf <model>-00001-of-0000N.gguf        (any shard; finds the one with the key)

With --native the engine takes the PLE key from the GGUF when it is Q2_0 (the GSQ-RCO files) and otherwise needs a
BF16 key in the pack: a key quantized to another type (OrcaRouter's IQ2_M: IQ2_S) stops it with "native PLE key is
absent or incompatible".  The shard that holds the key is rewritten once with the key dequantized to BF16 - still a
valid GGUF (contiguous offsets: llama.cpp and the vision encoder load it) - and replaces the original file.
Nothing is done when the key is already BF16 or Q2_0.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_reader as G  # noqa: E402
from iq_pack import Model, dequant_bf16  # noqa: E402

KEY = "blk.1.ple_key.weight"
BF16, Q2_0 = 30, 42
CHUNK = 64 << 20


def info_start(g: G.GGUFFile) -> int:
    """File position of the tensor-info section (right after the metadata)."""
    with g.path.open("rb") as fh:
        _, _, n_kv = struct.unpack("<IQQ", fh.read(4 + 20)[4:])
        for _ in range(n_kv):
            g._str(fh)
            g._value(fh)
        return fh.tell()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="a shard of the model")
    a = ap.parse_args()
    model = Model(pathlib.Path(a.gguf).resolve())
    if KEY not in model.where:
        print("%s: the model has no %s" % (a.gguf, KEY))
        return 1
    g, key, mm, path = model.where[KEY]
    if key.type_id in (BF16, Q2_0):
        return 0
    bf16 = dequant_bf16(mm, g, key)
    if bf16 is None:
        print("%s is %s, which gguf-py cannot dequantize" % (KEY, key.type_name))
        return 1

    # the same header with the key's type and every offset after it recomputed: the fields keep their width, so the
    # header - and where the data starts - does not move
    start, al = info_start(g), g.alignment
    head = bytearray(mm[:start].tobytes())
    sizes, off = [], 0
    for t in g.tensors:
        n = len(bf16) if t.name == KEY else t.expected_bytes()
        name = t.name.encode("utf-8")
        head += struct.pack("<Q", len(name)) + name + struct.pack("<I", len(t.shape))
        head += struct.pack(f"<{len(t.shape)}Q", *t.shape)
        head += struct.pack("<IQ", BF16 if t.name == KEY else t.type_id, off)
        sizes.append(n)
        off += (n + al - 1) // al * al
    if len(head) > g.data_start:
        print("the rewritten header is longer than the original")
        return 1
    head += b"\0" * (g.data_start - len(head))

    tmp = path.with_name(path.name + ".ple-bf16.part")
    print("rewriting %s with %s as BF16 (%s -> %d bytes) ..." % (path.name, KEY, key.type_name, len(bf16)), flush=True)
    with open(tmp, "wb") as fo:
        fo.write(head)
        for t, n in zip(g.tensors, sizes):
            if t.name == KEY:
                fo.write(bf16)
            else:
                src = g.data_start + t.offset
                for c in range(0, n, CHUNK):
                    fo.write(mm[src + c: src + min(n, c + CHUNK)].tobytes())
            fo.write(b"\0" * ((-n) % al))
    check = G.GGUFFile(tmp)
    want = {t.name: (t.shape, BF16 if t.name == KEY else t.type_id) for t in g.tensors}
    if {t.name: (t.shape, t.type_id) for t in check.tensors} != want or \
            os.path.getsize(tmp) != check.data_start + off:
        print("the rewritten file does not check out; the original is kept")
        return 1
    del model, g, mm                                     # release the memmap before replacing the file (Windows)
    import gc
    gc.collect()
    os.replace(tmp, path)
    print("%s: %s is BF16 now" % (path.name, KEY))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
