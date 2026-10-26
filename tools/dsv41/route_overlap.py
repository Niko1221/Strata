#!/usr/bin/env python3
"""How much would a batched MoE save?  Parse the engine's routing trace (DS4_TRACE=file) and measure how often two
sequences decoding at the same time pick the same experts.

    python3 route_overlap.py /data/works/route.trace

Two record kinds share the file: a decode (or DSpark verify) step's - u8 layer, u8 T, T x k_exp int16 expert ids - and a
prompt chunk's: u8 (layer | 0x80), u16 T, T x k_exp int16.  Only the decode records with T = 1 are tokens; with
--slot-burst 1 a two-slot engine alternates whole tokens (one record a layer, layers 0..N-1), so they group by their
layer cycle and the two slots alternate.

The read side of a batched MoE is  N*k  expert rows for N tokens of k experts; with the sets sharing s of their k it is
N*k - (N-1)*s, so the speedup on that side is  N*k / (N*k - (N-1)*s).
"""
from __future__ import annotations

import sys


def parse(data: bytes, k: int):
    recs, off = [], 0
    while off + 2 <= len(data):
        b0 = data[off]
        if b0 & 0x80:                      # a prompt chunk's record: layer | 0x80, u16 T
            if off + 3 > len(data):
                break
            layer, t, head, kind = b0 & 0x7F, int.from_bytes(data[off + 1:off + 3], "little"), 3, "c"
        else:                              # a decode (or verify) step's: u8 layer, u8 T
            layer, t, head, kind = b0, data[off + 1] & 0x7F, 2, "d"
        if t < 1 or t > 4096:
            return None
        need = head + t * k * 2
        if off + need > len(data):
            break
        ids = [int.from_bytes(data[off + head + 2 * i:off + head + 2 + 2 * i], "little", signed=True) for i in range(t * k)]
        recs.append((layer, t, ids, kind))
        off += need
    return recs


def main() -> int:
    path = sys.argv[1]
    data = open(path, "rb").read()
    best = None
    for k in (6, 8, 4, 16, 5, 7, 3, 2):
        recs = parse(data, k)
        if not recs or len(recs) < 64:
            continue
        if all(r[0] < 64 for r in recs) and all(0 <= e < 4096 for r in recs[:256] for e in r[2]):
            best = (k, recs)
            break
    if best is None:
        print("not a routing trace (or another k_exp):", len(data), "bytes")
        return 2
    k, recs = best
    t1 = [r for r in recs if r[1] == 1 and r[3] == "d"]
    n_layers = max(r[0] for r in recs if r[3] == "d") + 1
    print("trace: %d records (%d decode steps, %d prompt chunks), k_exp %d, %d layers, %d bytes" % (
        len(recs), len(t1), len(recs) - len(t1), k, n_layers, len(data)))
    tokens, cur = [], []
    for r in t1:
        if r[0] == 0 and cur:
            tokens.append(cur)
            cur = []
        cur.append(r)
    if cur:
        tokens.append(cur)
    full = [t for t in tokens if len(t) == n_layers and all(r[0] == i for i, r in enumerate(t))]
    print("decode tokens with every layer in order: %d (of %d groups)" % (len(full), len(tokens)))
    if len(full) < 2:
        print("not enough decode tokens to compare (two requests with --slot-burst 1)")
        return 1
    a = full[0::2]
    b = full[1::2]
    n = min(len(a), len(b))
    shared = total = 0
    for i in range(n):
        for li in range(n_layers):
            shared += len(set(a[i][li][2]) & set(b[i][li][2]))
            total += k
    cross = shared / total if total else 0.0
    same = tot_same = 0
    for stream in (a, b):
        for i in range(1, len(stream)):
            for li in range(n_layers):
                same += len(set(stream[i][li][2]) & set(stream[i - 1][li][2]))
                tot_same += k
    same_seq = same / tot_same if tot_same else 0.0
    print("expert overlap between the two sequences at the same step: %.1f%% (%.2f of %d a token)" % (
        100 * cross, cross * k, k))
    print("expert overlap between consecutive tokens of one sequence: %.1f%%" % (100 * same_seq))
    for pairs in (2, 3):
        rows = pairs * k - (pairs - 1) * cross * k
        print("  %d tokens batched: %.1f expert rows instead of %d -> %.2fx on the reads" % (
            pairs, rows, pairs * k, (pairs * k) / rows if rows > 0 else 0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
