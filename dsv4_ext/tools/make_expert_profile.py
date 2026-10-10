#!/usr/bin/env python3
"""Builds the expert profile .bin from routing traces written by `dsv4_run --dump-routing`.

Only the experts listed (count > 0) in this profile are copied into RAM at startup, and the hottest ones per layer
into VRAM.  The router still decides which experts run: the profile only decides WHERE they live.

  dsv4_run --model ... --prompt-ids ... --dump-routing run1.trace      (repeat with representative prompts)
  python3 tools/make_expert_profile.py run1.trace run2.trace -o expert-profile.bin [--merge old.bin] [--bytes-per-expert N]

Trace : "DSV4TRCE" u32 ver=1, u32 n_layers, u32 n_experts, u32 top_k, u64 model_hash, then int32[n_layers*top_k] per token.
Profile: "DSV4PROF" u32 ver=1, u32 n_layers, u32 n_experts, u64 model_hash, then u64[n_layers*n_experts] routing counts."""
import argparse, os, struct, sys
import numpy as np

def read_trace(path):
    with open(path, "rb") as f:
        if f.read(8) != b"DSV4TRCE": sys.exit("%s: not a DSV4 routing trace" % path)
        ver, nl, ne, k, mh = struct.unpack("<IIIIQ", f.read(24))
        if ver != 1: sys.exit("%s: unsupported trace version %d" % (path, ver))
        data = np.frombuffer(f.read(), dtype="<i4")
    rec = nl * k
    n = data.size // rec
    return nl, ne, k, mh, data[: n * rec].reshape(n, nl, k)

def read_profile(path):
    with open(path, "rb") as f:
        if f.read(8) != b"DSV4PROF": sys.exit("%s: not a DSV4 profile" % path)
        ver, nl, ne, mh = struct.unpack("<IIIQ", f.read(20))
        return nl, ne, mh, np.frombuffer(f.read(), dtype="<u8").reshape(nl, ne).copy()

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+"); ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--merge", help="existing profile whose counts are added"); ap.add_argument("--bytes-per-expert", type=float, default=0, help="average bytes per expert (MiB value from dsv4_plan) to estimate RAM")
    a = ap.parse_args()
    counts = None; meta = None; ntok = 0
    if a.merge:
        nl, ne, mh, counts = read_profile(a.merge); meta = (nl, ne, mh)
    for t in a.traces:
        nl, ne, k, mh, rec = read_trace(t)
        if meta is None: meta = (nl, ne, mh); counts = np.zeros((nl, ne), dtype=np.uint64)
        elif meta != (nl, ne, mh): sys.exit("%s: built for a different model/shape than the previous inputs" % t)
        for l in range(nl):
            col = rec[:, l, :].ravel(); col = col[col >= 0]
            counts[l] += np.bincount(col, minlength=ne).astype(np.uint64)
        ntok += rec.shape[0]
    nl, ne, mh = meta
    tmp = a.output + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"DSV4PROF" + struct.pack("<IIIQ", 1, nl, ne, mh)); f.write(counts.astype("<u8").tobytes()); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, a.output)
    used = int((counts > 0).sum()); total = nl * ne
    print("tokens: %d | layers: %d | experts/layer: %d" % (ntok, nl, ne))
    print("experts hit at least once: %d / %d (%.1f%%) -> these are the ONLY ones copied to RAM" % (used, total, 100.0 * used / total))
    for cov in (0.5, 0.8, 0.9, 0.99):
        need = 0
        for l in range(nl):
            s = np.sort(counts[l])[::-1].astype(np.float64); tot = s.sum()
            need += int(np.searchsorted(np.cumsum(s), cov * tot) + 1) if tot > 0 else 0
        print("  experts needed to cover %2.0f%% of the activations: %d (%.1f%%)" % (cov * 100, need, 100.0 * need / total))
    if a.bytes_per_expert > 0: print("estimated RAM for the profile set: %.2f GiB" % (used * a.bytes_per_expert / 1024.0))
    print("wrote %s" % a.output)

main()
