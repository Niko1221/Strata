"""tools/make_expert_profile.py - build a Strata expert-residency profile (`profile.bin`) from a routing trace.

    python tools/make_expert_profile.py --routing routing.bin --want-expert 256 --out my-profile.bin --slots 3000

The engine's `--dump-routing` writes one record per (layer, position):

    int32 layer, int32 k, k int32 expert ids, k float router weights      (repeated)

`--expert-profile` reads back a file that ranks (layer, expert) pairs by routing frequency, so the VRAM tier is
filled with the experts that are actually routed instead of the first ones a run happens to touch
(compulsory-miss).  The format is the engine's own (`src/core/expert_cache.cpp`, `read_expert_profile`):

    "STRP", uint32 version=1, uint32 n_layers, uint32 n_expert, uint32 slots, uint32 n_ranked,
    n_ranked x (uint16 layer, uint16 expert),      (ranked by descending frequency)
    n_layers * n_expert x float32 frequency        (the profile's own working; not read back)

WHY THIS SCRIPT EXISTS, AND WHY `--want-expert` IS REQUIRED FOR A PRUNED MODEL.  The header carries
`n_layers x n_expert` and the engine REFUSES a profile built for another artifact, so a 256-expert release (the
Coder) cannot reuse the 512-expert profile in `data/`.  A trace only contains the experts that were ROUTED, so the
highest id seen is a LOWER BOUND on the count, not the count: pass `--want-expert 256` so the header matches the
model.  Inferring it from the trace would produce a profile the engine accepts and then mis-indexes.
"""
from __future__ import annotations

import argparse
import struct
import sys
from collections import Counter
from pathlib import Path


def read_trace(path: Path):
    """Yield `(layer, ids)` per record.  Each record is self-describing through its `k` field."""
    raw = path.read_bytes()
    at, n = 0, len(raw)
    while at + 8 <= n:
        layer, k = struct.unpack_from("<ii", raw, at)
        at += 8
        need = 2 * k * 4                      # k int32 ids + k float weights
        if at + need > n:
            print(f"warning: the trace ends mid-record at byte {at} (k={k}); ignoring the tail", file=sys.stderr)
            break
        yield layer, struct.unpack_from(f"<{k}i", raw, at)
        at += need


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--routing", required=True, help="the file --dump-routing wrote")
    ap.add_argument("--out", required=True, help="the profile.bin to write")
    ap.add_argument("--want-expert", type=int, required=True,
                    help="the model's real experts per layer (256 for the pruned Coder release, 512 otherwise)")
    ap.add_argument("--slots", type=int, default=8000, help="how many slots the profile is built for")
    ap.add_argument("--top-layers", type=int, default=0, help="also print this many hottest layers")
    a = ap.parse_args()

    trace = Path(a.routing)
    if not trace.is_file() or trace.stat().st_size == 0:
        print(f"error: {trace} is missing or empty (--dump-routing writes DECODE steps only)", file=sys.stderr)
        return 1
    if a.want_expert <= 0:
        print("error: --want-expert must be positive", file=sys.stderr)
        return 1

    counts: Counter[tuple[int, int]] = Counter()
    per_layer: Counter[int] = Counter()
    n_layers = 0
    records = 0
    for layer, ids in read_trace(trace):
        records += 1
        per_layer[layer] += len(ids)
        n_layers = max(n_layers, layer + 1)
        for e in ids:
            # A pruned release has 256 experts.  An id outside that range means the trace and --want-expert come
            # from different models, and counting it would put a pair the engine refuses into the ranked list.
            if e < 0 or e >= a.want_expert:
                print(f"error: the trace routed expert {e} in layer {layer}, outside 0..{a.want_expert - 1}: "
                      f"the trace and --want-expert are from different models", file=sys.stderr)
                return 1
            counts[(layer, e)] += 1
    if records == 0:
        print("error: the trace has no records", file=sys.stderr)
        return 1

    ranked = [pair for pair, _ in counts.most_common()]
    n_ranked = min(len(ranked), a.slots)
    total = float(sum(counts.values())) or 1.0

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"STRP")
        f.write(struct.pack("<5I", 1, n_layers, a.want_expert, a.slots, n_ranked))
        for layer, expert in ranked[:n_ranked]:
            f.write(struct.pack("<HH", layer, expert))
        freq = [0.0] * (n_layers * a.want_expert)
        for (layer, expert), c in counts.items():
            freq[layer * a.want_expert + expert] = c / total
        f.write(struct.pack(f"<{len(freq)}f", *freq))

    covered = sum(counts[p] for p in ranked[:n_ranked])
    print(f"trace: {records} records, {n_layers} layers, {len(counts)} distinct pairs routed "
          f"of {n_layers * a.want_expert} possible")
    print(f"profile {out}: {n_layers} x {a.want_expert}, {a.slots} slots, {n_ranked} ranked pairs")
    print(f"  the {n_ranked} slots cover {100.0 * covered / total:.1f}% of the routed entries")
    if a.top_layers:
        print("  hottest layers by routed entries:")
        for layer, n in per_layer.most_common(a.top_layers):
            print(f"    layer {layer:3d}: {n} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
