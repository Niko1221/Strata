#!/usr/bin/env python
"""Build the centering offset table from the engine's cell dump (bench/results/2026-11-10-centering/).

The dump (STRATA_KV_Q4_DUMP) holds the rotated K/V rows of the batched prompt path.  The centering
subtracts the per-(KV head, channel) offset **before** the rotation, so the table wants the offset in
the channel space.  The Hadamard is linear and an involution (fwht(fwht(x)) == x), so the offset of
the rotated cells, un-rotated once, is the channel-space offset.

The table (the file STRATA_KV_CENTER names) is:
    "STKC" | int32 version=1 | int32 entries | int32 heads | int32 dim
    per entry: int32 model_layer | heads*dim float32 K offsets | heads*dim float32 V offsets
Only the layers with cells are written; the engine puts every entry into its model-layer slot.
"""
import argparse
import struct

import numpy as np

MAGIC = b"STKC"
VERSION = 1


def fwht(x):
    """The engine's orthonormal Walsh-Hadamard (Sylvester, scale 1/16): its own inverse."""
    a = np.array(x, dtype=np.float64, copy=True)
    n = a.shape[-1]
    s = 1
    while s < n:
        idx = np.arange(n)
        left = idx[(idx // s) % 2 == 0]
        right = left + s
        tmp = a[..., left].copy()
        a[..., left] = tmp + a[..., right]
        a[..., right] = tmp - a[..., right]
        s *= 2
    return a / 16.0


def read_dump(path):
    """{model_layer: (K sum, V sum, count, heads, dim)} over the dump's records."""
    hdr = struct.Struct("<iiqiiii")
    acc = {}
    with open(path, "rb") as f:
        while True:
            head = f.read(hdr.size)
            if len(head) < hdr.size:
                break
            magic, layer, p0, T, nkv, hd, _pad = hdr.unpack(head)
            if magic != 0x51345042:
                raise SystemExit(f"bad magic {magic:#x}")
            n = T * nkv * hd
            k = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(T, nkv, hd).astype(np.float64)
            v = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(T, nkv, hd).astype(np.float64)
            e = acc.setdefault(layer, [np.zeros((nkv, hd)), np.zeros((nkv, hd)), 0, nkv, hd])
            e[0] += k.sum(axis=0)
            e[1] += v.sum(axis=0)
            e[2] += T
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    acc = read_dump(args.dump)
    if not acc:
        raise SystemExit("the dump has no records")
    heads = dim = None
    entries = []
    for layer in sorted(acc):
        ksum, vsum, count, nkv, hd = acc[layer]
        heads, dim = nkv, hd
        # the rotated mean per (head, channel), un-rotated into the channel space
        k_off = fwht(ksum / count)
        v_off = fwht(vsum / count)
        entries.append((layer, k_off.astype("<f4"), v_off.astype("<f4")))
        print(f"layer {layer:>3}: {count} cells, {nkv} heads, |K-Offset| median "
              f"{np.median(np.abs(k_off)):.5f}, |V-Offset| median {np.median(np.abs(v_off)):.5f}")

    with open(args.out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<iiii", VERSION, len(entries), heads, dim))
        for layer, k_off, v_off in entries:
            f.write(struct.pack("<i", layer))
            f.write(k_off.tobytes())
            f.write(v_off.tobytes())
    print(f"wrote {args.out}: {len(entries)} layers, {heads} heads, {dim} channels")


if __name__ == "__main__":
    main()
