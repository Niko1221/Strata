#!/usr/bin/env python3
"""Read a --dump-kv file and print the channel structure the synthetic cache cannot have.

THE FILE (SDKV v1, little-endian), written by `strata generate --dump-kv PATH --kv fp16`:

    int32 magic 0x564B4453 (SDKV), int32 version = 1, int32 n_layers, int32 n_cells,
    int32 n_head_kv, int32 head_dim, int32 layer_ids[n_layers]
    then for each dumped layer, in layer_ids order: K, then V,
    each n_cells*n_head_kv*head_dim uint16 (fp16 bits), in (cell, head, dim) order

`n_layers` is the number of DUMPED layers: every layer that HAS a KV cache (the QSA layers - layers 3, 7,
... for the artifact's interval of 4).  A GDN layer has no KV cache at all, so it is not in the file.

WHY THESE NUMBERS.  The 2-bit work is measured on i.i.d. Gaussian noise with a 3x per-cell spread
(src/kernels/kv_q2_parity.cpp).  Per-channel scaling exists to exploit a handful of channels being far
larger than the rest (KIVI, arXiv 2402.02750), and noise with every channel at the same RMS cannot show
that - so every per-channel number measured on it is a floor.  The ratios printed here are the question:

    channel max/median RMS   ~1  ->  every channel is the same size, per-channel scaling has nothing to
                                     exploit and the per-token scale is the honest choice
                              3x+ ->  the structure KIVI's result is about is there, and the per-channel
                                     arm is measuring something real
    top-8 variance share     how concentrated that structure is: 8 of the 512 channels holding a large
                             slice of the total variance is a handful of outlier channels, which is what
                             per-channel scaling is for
    cell max/median RMS      the per-cell spread, the axis the per-token scale exploits

The channel index is (head * head_dim + dim) over the n_head_kv * head_dim channels of the layer - the
same axis src/kernels/kv_q2_parity.cpp's channel_stats uses, and the same sums.  ONE LABEL DIFFERENCE,
and it is a factor of two in the exponent: that function takes its printed min/median/max from the
per-channel VARIANCES (cv[i] = chan[i]^2) while printing them under the words "channel RMS", so its
max/median is the SQUARE of the RMS ratio.  This file takes them from the RMS, as the label says, so a
"3x" here is a "9x" there.  The top-8 variance share is identical in both.

WHAT THE CACHE IS.  It is what the engine stored, read back through the engine's own fp16 gather:
post-RoPE, post-rms-norm.  NOT Hadamard-rotated - the rotation is applied on the way INTO the quantized
pools (kv_q4/kv_q2, and int8 with STRATA_KV_ROT=1), and --dump-kv requires --kv fp16, whose pools are
unrotated.  So these ratios describe the cache the rotation is applied TO, which is the cache whose
channel variance the rotation is supposed to equalize.  Reading the rotated form would mean applying the
WHT here, and the task says no re-rotation, so this file is the pre-rotation reference.

    python tools/kv_cache_dump.py kv.bin
    python tools/kv_cache_dump.py kv.bin --layer 3
    python tools/kv_cache_dump.py kv.bin --npz kv.npz
"""
from __future__ import annotations

import argparse
import struct
import sys

import numpy as np

MAGIC = 0x564B4453          # little-endian "SDKV"
VERSION = 1
HEADER = 6                  # int32 words before the layer id table


class Dump:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            raw = f.read()
        size = len(raw)
        if size < HEADER * 4:
            raise SystemExit(path + ": " + str(size) + " bytes, shorter than the " +
                             str(HEADER * 4) + "-byte header")
        magic, version, n_layers, n_cells, n_head_kv, head_dim = struct.unpack("<6i", raw[:HEADER * 4])
        if magic != MAGIC:
            raise SystemExit(path + ": magic 0x" + format(magic, "08X") + " is not 0x" + format(MAGIC, "08X") +
                             " (SDKV) - write the file with --dump-kv")
        if version != VERSION:
            raise SystemExit(path + ": version " + str(version) + " is not " + str(VERSION))
        if min(n_layers, n_cells, n_head_kv, head_dim) <= 0:
            raise SystemExit(path + ": header has a non-positive count: n_layers=" + str(n_layers) +
                             " n_cells=" + str(n_cells) + " n_head_kv=" + str(n_head_kv) +
                             " head_dim=" + str(head_dim))
        need = HEADER * 4 + 4 * n_layers
        if size < need:
            raise SystemExit(path + ": header promises a " + str(n_layers) +
                             "-entry layer id table, the file has " + str(size) + " bytes")
        layer_ids = list(struct.unpack("<" + str(n_layers) + "i", raw[HEADER * 4:need]))
        plane = n_cells * n_head_kv * head_dim
        self.n_layers, self.n_cells, self.n_head_kv, self.head_dim = n_layers, n_cells, n_head_kv, head_dim
        self.layer_ids, self.plane, self.size = layer_ids, plane, size
        self.expect = need + n_layers * 2 * plane * 2
        self.body = raw[need:]
        self.body_ok = len(self.body) == n_layers * 2 * plane * 2

    def layer(self, i):
        """Layer slot i as (K, V), each (n_cells, n_head_kv, head_dim) float32."""
        off = i * 2 * self.plane * 2
        k = np.frombuffer(self.body, dtype="<u2", count=self.plane, offset=off)
        v = np.frombuffer(self.body, dtype="<u2", count=self.plane, offset=off + self.plane * 2)
        # fp16 bits -> float32, exactly: `view` reads the same bits numpy's float16 holds.
        kf = k.view(np.float16).astype(np.float32).reshape(self.n_cells, self.n_head_kv, self.head_dim)
        vf = v.view(np.float16).astype(np.float32).reshape(self.n_cells, self.n_head_kv, self.head_dim)
        return kf, vf


def ratios(x, cells, heads, dims):
    """The channel and cell statistics, on (cells, heads, dims) float32.

    A channel is one (head, dim) column over the cells - the axis a per-channel scale is placed on.
    A cell is one row over the channels - the axis a per-token scale is placed on.  Both are the axes
    src/kernels/kv_q2_parity.cpp's channel_stats uses, computed the same way.
    """
    flat = np.asarray(x, dtype=np.float32).reshape(cells, heads * dims)
    n_ch = heads * dims
    chan = np.sqrt(np.sum(flat * flat, axis=0) / cells)          # per-channel RMS across cells
    cell = np.sqrt(np.sum(flat * flat, axis=1) / n_ch)           # per-cell RMS across channels
    var = chan * chan                                           # per-channel variance (mean square)
    total = float(np.sum(var))
    order = np.sort(var)[::-1]
    top8 = float(np.sum(order[:8])) if n_ch >= 8 else total
    med_c = float(np.median(chan))
    med_e = float(np.median(cell))
    return dict(
        chan_min=float(np.min(chan)), chan_med=med_c, chan_max=float(np.max(chan)),
        chan_ratio=(float(np.max(chan)) / med_c) if med_c > 0 else float("inf"),
        top8_share=(top8 / total) if total > 0 else float("nan"),
        cell_min=float(np.min(cell)), cell_med=med_e, cell_max=float(np.max(cell)),
        cell_ratio=(float(np.max(cell)) / med_e) if med_e > 0 else float("inf"),
        nonfinite=int(flat.size - np.count_nonzero(np.isfinite(flat))))


def print_block(tag, r):
    cmin, cmed, cmax, cr = r["chan_min"], r["chan_med"], r["chan_max"], r["chan_ratio"]
    emin, emed, emax, er = r["cell_min"], r["cell_med"], r["cell_max"], r["cell_ratio"]
    share = 100.0 * r["top8_share"]
    print("    " + tag.ljust(3) + " channel RMS  min " + format(cmin, ".6g") + "  median " +
          format(cmed, ".6g") + "  max " + format(cmax, ".6g") + "   max/median " + format(cr, ".3f"))
    print("         top-8 channel variance share  " + format(share, ".2f") + "% of total")
    print("    " + tag.ljust(3) + " cell RMS     min " + format(emin, ".6g") + "  median " +
          format(emed, ".6g") + "  max " + format(emax, ".6g") + "   max/median " + format(er, ".3f"))


def main():
    ap = argparse.ArgumentParser(
        description="Read a --dump-kv (SDKV v1) KV cache dump and print its channel structure.")
    ap.add_argument("file", help="the --dump-kv file")
    ap.add_argument("--layer", type=int, default=None, help="restrict the printout to one layer id")
    ap.add_argument("--npz", default=None, help="also write the arrays out for the gate/harness to consume")
    a = ap.parse_args()

    d = Dump(a.file)
    plane_bytes = d.plane * 2
    print(a.file + ": SDKV v1  n_layers=" + str(d.n_layers) + "  n_cells=" + str(d.n_cells) +
          "  n_head_kv=" + str(d.n_head_kv) + "  head_dim=" + str(d.head_dim))
    print("  layer ids: " + str(d.layer_ids))
    verdict = "MATCH" if d.size == d.expect else "MISMATCH"
    print("  size " + str(d.size) + " bytes, header implies " + str(d.expect) + "  (" + verdict + ")")
    print("  per plane (K or V): " + str(d.plane) + " fp16 = " + str(plane_bytes) +
          " bytes; per layer K+V = " + str(2 * plane_bytes) + " bytes")
    ok = (d.size == d.expect) and d.body_ok
    if not d.body_ok:
        print("  DEFECT: body is " + str(len(d.body)) + " bytes, the header implies " +
              str(d.n_layers * 2 * d.plane * 2))

    picked = list(range(d.n_layers))
    if a.layer is not None:
        picked = [i for i, layer_id in enumerate(d.layer_ids) if layer_id == a.layer]
        if not picked:
            print("layer " + str(a.layer) + " is not in the dump (ids: " + str(d.layer_ids) + ")", file=sys.stderr)
            return 2

    pooled_k, pooled_v = [], []
    for i in picked:
        k, v = d.layer(i)
        nf = int(np.count_nonzero(~np.isfinite(k))) + int(np.count_nonzero(~np.isfinite(v)))
        heads, dims, cells = d.n_head_kv, d.head_dim, d.n_cells
        print("\nlayer " + str(d.layer_ids[i]) + ": cells=" + str(cells) + " heads=" + str(heads) +
              " head_dim=" + str(dims) + " bytes=" + str(2 * plane_bytes))
        print_block("K", ratios(k, cells, heads, dims))
        print_block("V", ratios(v, cells, heads, dims))
        kv = np.concatenate([k.reshape(cells, -1), v.reshape(cells, -1)], axis=1)
        print_block("KV", ratios(kv, cells, 2 * heads, dims))
        if nf:
            print("    DEFECT: " + str(nf) + " non-finite values in this layer")
            ok = False
        pooled_k.append(k)
        pooled_v.append(v)

    if len(picked) > 1:
        k = np.concatenate(pooled_k, axis=0)
        v = np.concatenate(pooled_v, axis=0)
        n = int(k.shape[0])
        print("\npooled over " + str(len(picked)) + " layers: cells=" + str(n) + " heads=" + str(d.n_head_kv) +
              " head_dim=" + str(d.head_dim) + " bytes=" + str(int(k.size) * 2 + int(v.size) * 2))
        print_block("K", ratios(k, n, d.n_head_kv, d.head_dim))
        print_block("V", ratios(v, n, d.n_head_kv, d.head_dim))
        kv = np.concatenate([k.reshape(n, -1), v.reshape(n, -1)], axis=1)
        print_block("KV", ratios(kv, n, 2 * d.n_head_kv, d.head_dim))

    if a.npz:
        # Every dumped layer, not just the printed ones: a harness reading the npz wants the whole cache.
        arrays = dict()
        arrays["layer_ids"] = np.array(d.layer_ids, dtype=np.int32)
        arrays["n_cells"] = np.int32(d.n_cells)
        arrays["n_head_kv"] = np.int32(d.n_head_kv)
        arrays["head_dim"] = np.int32(d.head_dim)
        for i, layer_id in enumerate(d.layer_ids):
            k, v = d.layer(i)
            arrays["k" + str(layer_id)] = k
            arrays["v" + str(layer_id)] = v
        written = a.npz if a.npz.endswith(".npz") else a.npz + ".npz"   # numpy appends the suffix itself
        np.savez(a.npz, **arrays)
        print("\nwrote " + written + ": " + str(d.n_layers) + " layers x (K, V) of (" + str(d.n_cells) + ", " +
              str(d.n_head_kv) + ", " + str(d.head_dim) + ") float32, keys k<L>/v<L> plus " +
              "layer_ids/n_cells/n_head_kv/head_dim")

    if not ok:
        print("\nDEFECTS FOUND", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
