#!/usr/bin/env python3
"""Strip split.* KV pairs from a GGUF in place, preserving data_start exactly."""
import struct, sys, pathlib

p = pathlib.Path(sys.argv[1])

def rstr(f):
    n = struct.unpack("<Q", f.read(8))[0]; return f.read(n)

def skip_val_of(f, t):  # GGUF type enum
    if t in (0, 1, 7): f.read(1)          # u8 i8 bool
    elif t in (2, 3): f.read(2)           # u16 i16
    elif t in (4, 5, 6): f.read(4)        # u32 i32 f32
    elif t in (10, 11, 12): f.read(8)     # u64 i64 f64
    elif t == 8: rstr(f)                  # string
    elif t == 9:                          # array
        et = struct.unpack("<I", f.read(4))[0]; cnt = struct.unpack("<Q", f.read(8))[0]
        for _ in range(cnt): skip_val_of(f, et)
    else: raise ValueError(f"type {t}")

with open(p, "r+b") as f:
    assert f.read(4) == b"GGUF"
    ver = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    raw_pairs = []
    for _ in range(n_kv):
        k0 = f.tell(); key = rstr(f)
        t = struct.unpack("<I", f.read(4))[0]
        skip_val_of(f, t); v1 = f.tell()
        f.seek(k0); raw = f.read(v1 - k0); f.seek(v1)
        raw_pairs.append((key.decode(), raw))
    kv_end = f.tell()
    # parse tensor infos, keeping them for rewrite (offsets may need shifting)
    infos = []
    for _ in range(n_tensors):
        name = rstr(f); nd = struct.unpack("<I", f.read(4))[0]
        shape = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        tid = struct.unpack("<I", f.read(4))[0]; off = struct.unpack("<Q", f.read(8))[0]
        infos.append((name, shape, tid, off))
    table_end = f.tell()
    old_data_start = (table_end + 31) // 32 * 32

    kept_raw = [(k, raw) for k, raw in raw_pairs if not k.startswith("split.")]
    body = b"".join(raw for _, raw in kept_raw)
    new_hdr = (b"GGUF" + struct.pack("<I", ver) + struct.pack("<Q", n_tensors)
               + struct.pack("<Q", len(kept_raw)) + body)
    new_table = b"".join(
        struct.pack("<Q", len(n)) + n + struct.pack("<I", len(sh))
        + b"".join(struct.pack("<Q", d) for d in sh)
        + struct.pack("<I", t) + struct.pack("<Q", o)
        for (n, sh, t, o) in infos)
    new_data_start = (len(new_hdr) + len(new_table) + 31) // 32 * 32
    delta = old_data_start - new_data_start
    if delta:
        new_table = b"".join(
            struct.pack("<Q", len(n)) + n + struct.pack("<I", len(sh))
            + b"".join(struct.pack("<Q", d) for d in sh)
            + struct.pack("<I", t) + struct.pack("<Q", o + delta)
            for (n, sh, t, o) in infos)
    total = len(new_hdr) + len(new_table)
    assert total <= new_data_start + 32
    f.seek(0)
    f.write(new_hdr + new_table + b"\x00" * (old_data_start - total))
    print(f"n_kv {n_kv} -> {len(kept_raw)}; dropped {[k for k, _ in raw_pairs if k.startswith('split.')]}")
    print(f"data_start {old_data_start} -> {new_data_start}; all tensor offsets shifted by +{delta}")
