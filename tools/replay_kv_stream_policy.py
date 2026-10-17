#!/usr/bin/env python3
"""Offline replay of KV block selections through the streaming page-pool policies (#2415).

    python replay_kv_stream_policy.py TRACE.bin [--layers 2 --slots 8192 --bytes 4224 --lambda 0.5 --decay-every 64]
    python replay_kv_stream_policy.py --synthetic agent --calls 2000 --universe 40000 --slots 8192

Trace record = int32 layer, int32 k, k int32 block ids, k float32 weights - self-describing, k may vary per
record.  One record is one kv_stream_resolve call (the blocks a selection named, ascending, one entry per block;
the engine's ids are cells, so a block is a cell id divided by --page).  The engine's --dump-kv-select writes this
trace (weights 1.0: the indexer's scores are not copied); --synthetic writes the patterns below in memory.

The pools are one per layer, as the engine's are. Policies per pool:

  clock   the engine's resolve_kernel second chance (kv_stream.cu): a used or loaded slot is referenced, a
          victim is the first unreferenced slot the hand reaches (a slot this call uses is never taken), and
          every slot the hand passes loses its reference. Modeled at slot granularity; the kernel's RT=1024
          chunking only orders ties inside one sweep.
  lru     evict the least recently used resident block not used by this call.
  lrfu    keep score KS survives eviction; every decay step (each --decay-every calls) all scores are KS *=
          lambda, and a hit adds --gain: KS = lambda * KS + gain (ActKV's rule, arXiv:2609.31395; kvmem's
          cadence is once per request). "Hit" is this call's top --top-frac weight mass: scores may vary
          within a call, and the weight the top hits cover is the range that accumulates - it is not a
          fraction of the pool. Evict the lowest-KS resident block not used by this call.
  oracle  Belady: evict the resident block whose next use is farthest (the offline ceiling).

Reports misses, the hit rate, the bytes re-read and how far above the compulsory load every policy sits
(each block is read into the pool at least once) - the same split #1089 reports for the expert cache.

The synthetic patterns: scan (fresh blocks every call - prefill-like), loop (overlapping windows over a
circular set), agent (an advancing tail plus a stable task set and one-offs - an agent's history), intermittent
(a hot set every call, a cold set every Nth) and spotty (a hot set every other call, fresh one-offs in the
rest - the case where a sweep clears the hot set's references between its two uses).
Standard library only; no GPU, no model pack.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import struct
import sys


# ---------------------------------------------------------------- trace

def load_trace(path: str) -> list[tuple[int, list[int], list[float]]]:
    """[(layer, block ids, weights)] from a binary trace.  Each record is self-describing - int32 layer, int32 k,
    k int32 block ids, k float32 weights - so k may vary per record, as the engine's --dump-kv-select writes it
    (one record per resolve call, a call's own blocks)."""
    with open(path, "rb") as f:
        raw = f.read()
    out, off = [], 0
    while off < len(raw):
        if len(raw) - off < 8:
            raise SystemExit(f"{path}: truncated record header at byte {off}")
        layer, k = struct.unpack_from("<ii", raw, off)
        if k <= 0 or k > (1 << 24) or len(raw) - off < 8 + 8 * k:
            raise SystemExit(f"{path}: truncated or malformed record at byte {off} (k={k})")
        ids = list(struct.unpack_from(f"<{k}i", raw, off + 8))
        ws = list(struct.unpack_from(f"<{k}f", raw, off + 8 + 4 * k))
        out.append((layer, ids, ws))
        off += 8 + 8 * k
    return out


def synthetic(kind: str, layers: int, calls: int, universe: int, width: int, tail: int, task: int,
              oneshot: int, every: int, seed: int) -> list[tuple[int, list[int], list[float]]]:
    """The call streams the patterns see; blocks are chosen in the universe, weights all 1.0 unless the pattern
    has a story for them. `calls` is the number of calls PER layer."""
    rng = random.Random(seed)
    out: list[tuple[int, list[int], list[float]]] = []
    for layer in range(layers):
        pos = 0                                       # the history's leading edge (blocks "written" so far)
        hot = rng.sample(range(universe), min(width, universe)) if kind in ("intermittent", "spotty") else []
        cold = rng.sample([b for b in range(universe) if b not in set(hot)],
                          min(oneshot, universe)) if kind == "intermittent" else []
        task_set = set(rng.sample(range(universe), min(task, universe))) if kind == "agent" else set()
        for t in range(calls):
            if kind == "scan":                        # prefill-like: new blocks every call, nothing comes back
                blocks = list(range(pos, min(pos + width, universe)))
                pos = min(pos + width, universe)
            elif kind == "loop":                      # overlapping windows over a circular working set
                blocks = [(pos + i) % universe for i in range(width)]
                pos = (pos + max(1, width // 2)) % universe
            elif kind == "agent":                     # a growing history: a recent tail, the task's own
                # entries (stable, old), and a few one-off reads - the ActKV/kvmem picture of an agent loop
                pos = min(pos + max(1, tail // 2), universe)
                blocks = list(range(max(0, pos - tail), pos)) + list(task_set)
                blocks += rng.sample(range(universe), min(oneshot, universe))
            elif kind == "intermittent":              # a hot set every call, a cold set every `every`-th call
                blocks = list(hot) + (list(cold) if t % max(1, every) == 0 else [])
            elif kind == "spotty":                    # a hot set every other call, fresh one-offs in the rest -
                # the scaled-up case where a sweep clears the hot set's references between its two uses
                if t % 2 == 0:
                    blocks = list(hot)
                else:
                    blocks = list(range(pos, min(pos + max(1, oneshot), universe)))
                    pos = min(pos + max(1, oneshot), universe)
            else:
                raise SystemExit(f"unknown --synthetic {kind!r} (scan|loop|agent|intermittent|spotty)")
            blocks = sorted(set(blocks))
            out.append((layer, blocks, [1.0] * len(blocks)))
    return out


# ---------------------------------------------------------------- policies

class ClockPool:
    """The engine's resolve_kernel second chance, modeled at slot granularity (see the module docstring)."""

    def __init__(self, slots: int):
        self.n = slots
        self.slot_block = [-1] * slots
        self.slot_ref = [0] * slots
        self.block_slot: dict[int, int] = {}
        self.hand = 0

    def take(self, used: set[int]) -> int | None:
        """Claim one victim slot for a miss; `used` holds every block this call names (its slots are never
        taken).  None on overflow - the kernel's 3*n scan limit, when every slot belongs to this call."""
        for _ in range(3 * self.n):
            sl = self.hand
            self.hand = (self.hand + 1) % self.n
            blk = self.slot_block[sl]
            if blk >= 0 and blk in used:
                continue                              # mine this call: the kernel never takes it
            if blk >= 0 and self.slot_ref[sl]:
                self.slot_ref[sl] = 0                 # second chance
                continue
            return sl
        return None

    def call(self, blocks: list[int], weights: list[float]) -> int:
        used = set(blocks)
        misses = []
        for b in blocks:
            sl = self.block_slot.get(b)
            if sl is not None:
                self.slot_ref[sl] = 1
            else:
                misses.append(b)
        for b in misses:                              # the kernel takes one victim per miss, in miss order
            sl = self.take(used)
            if sl is None:                            # overflow: the block is left out, as the kernel leaves it
                continue
            old = self.slot_block[sl]
            if old >= 0:
                del self.block_slot[old]
            self.slot_block[sl] = b
            self.slot_ref[sl] = 1                     # a block loaded this call gets the reference too
            self.block_slot[b] = sl
        return len(misses)


class LruPool:
    def __init__(self, slots: int):
        self.n = slots
        self.order = collections.OrderedDict()        # block -> None, oldest first

    def call(self, blocks: list[int], weights: list[float]) -> int:
        misses = 0
        for b in blocks:
            if b in self.order:
                self.order.move_to_end(b)
            else:
                misses += 1
                if len(self.order) >= self.n:
                    self.order.popitem(last=False)
                self.order[b] = None
        return misses


class LrfuPool:
    """ActKV's attention-aware LRFU at pool scale: KS = lambda * KS + gain on a hit, all scores decayed every
    `decay_every` calls, the lowest-KS resident block evicted.  KS survives eviction - an intermittent block
    keeps its history when it comes back."""

    def __init__(self, slots: int, lam: float = 0.5, gain: float = 1.0, top_frac: float = 0.9,
                 decay_every: int = 64):
        self.n = slots
        self.lam, self.gain, self.top_frac, self.decay_every = lam, gain, top_frac, max(1, decay_every)
        self.ks: dict[int, float] = {}
        self.resident: set[int] = set()
        self.loaded: dict[int, int] = {}              # block -> load order (tie-break)
        self.calls = 0
        self.seq = 0

    def call(self, blocks: list[int], weights: list[float]) -> int:
        self.calls += 1
        if self.calls % self.decay_every == 0:
            for b in list(self.ks):
                self.ks[b] *= self.lam
        # the hit range: the blocks covering the top `top_frac` of this call's weight mass, with ties at the
        # cut included (equal weights make the whole call one tie - the range is not an arbitrary 90% of the
        # entries).  It decides who accumulates score; it is not a fraction of the pool.
        order = sorted(range(len(blocks)), key=lambda i: -weights[i])
        total = sum(weights) or 1.0
        run, cut = 0.0, len(order) - 1
        for idx, i in enumerate(order):
            run += weights[i]
            if run >= self.top_frac * total:
                cut = idx
                break
        threshold = weights[order[cut]] if order else 0.0
        hits = {blocks[i] for i in order if weights[i] >= threshold}
        misses = 0
        used = set(blocks)
        for b in blocks:
            if b in hits:
                self.ks[b] = self.ks.get(b, 0.0) + self.gain   # the decay tick above is the lambda of KS = lambda*KS+gain
            if b not in self.resident:
                misses += 1
                if len(self.resident) >= self.n:
                    gone = min((x for x in self.resident if x not in used),
                               key=lambda x: (self.ks.get(x, 0.0), self.loaded.get(x, 0)), default=None)
                    if gone is None:                  # every resident is used by this call: nothing to evict
                        continue
                    self.resident.discard(gone)
                    self.loaded.pop(gone, None)
                self.resident.add(b)
                self.seq += 1
                self.loaded[b] = self.seq
        return misses


def oracle_misses(blocks_per_call: list[list[int]], slots: int) -> int:
    """Belady computed from a full next-use table (the offline ceiling)."""
    nxt: dict[int, list[int]] = {}
    for i, bs in enumerate(blocks_per_call):
        for b in set(bs):
            nxt.setdefault(b, []).append(i)
    cursor = {b: 0 for b in nxt}
    resident: set[int] = set()
    misses = 0
    never = len(blocks_per_call) + 1
    for i, bs in enumerate(blocks_per_call):
        used = set(bs)
        for b in bs:
            cursor[b] = cursor.get(b, 0) + 1
            if b not in resident:
                misses += 1
                if len(resident) >= slots:
                    def farthest(x: int) -> int:
                        c = cursor.get(x, 0)
                        seq = nxt.get(x, [])
                        return seq[c] if c < len(seq) else never
                    gone = max((x for x in resident if x not in used), key=farthest, default=None)
                    if gone is None:
                        continue
                    resident.discard(gone)
                resident.add(b)
    return misses


# ---------------------------------------------------------------- run

def run(trace, layers: int, slots: int, block_bytes: int, lam: float, gain: float, top_frac: float,
        decay_every: int) -> dict:
    """Every policy over every layer's call stream; the report sums the pools (the engine has one per layer)."""
    per_layer: dict[int, list[tuple[list[int], list[float]]]] = {l: [] for l in range(layers)}
    for layer, ids, ws in trace:
        if layer not in per_layer:
            per_layer[layer] = []
        per_layer[layer].append((ids, ws))
    totals = {"calls": 0, "lookups": 0, "unique": 0, "policies": {}}
    names = ["compulsory", "clock", "lru", "lrfu", "oracle"]
    counts = {n: 0 for n in names}
    for layer, calls in sorted(per_layer.items()):
        blocks_per_call = [ids for ids, _ in calls]
        unique = len({b for ids in blocks_per_call for b in ids})
        totals["calls"] += len(calls)
        totals["lookups"] += sum(len(ids) for ids in blocks_per_call)
        totals["unique"] += unique
        counts["compulsory"] += unique
        pools = {"clock": ClockPool(slots), "lru": LruPool(slots),
                 "lrfu": LrfuPool(slots, lam, gain, top_frac, decay_every)}
        for name, pool in pools.items():
            for ids, ws in calls:
                counts[name] += pool.call(ids, ws)
        counts["oracle"] += oracle_misses(blocks_per_call, slots)
    for name in names:
        m = counts[name]
        look = totals["lookups"]
        totals["policies"][name] = {
            "misses": m,
            "hit_pct": round(100.0 * (look - m) / look, 1) if look and name != "compulsory" else None,
            "read_mib": round(m * block_bytes / 1048576.0, 1),
            "over_compulsory": m - counts["compulsory"],
        }
    return totals


def report(t: dict, args) -> None:
    print(f"trace: {t['calls']} calls, {t['lookups']} block lookups, {t['unique']} distinct blocks, "
          f"{args.slots} slots/pool, {args.bytes} B/block")
    if args.synthetic:
        print(f"  synthetic: {args.synthetic} (layers={args.layers}, universe={args.universe}, width={args.width}, "
              f"tail={args.tail}, task={args.task}, oneshot={args.oneshot}, every={args.every}, seed={args.seed})")
    else:
        print(f"  {args.trace}")
    print(f"  lrfu: lambda={args.lamb} gain={args.gain} top-frac={args.top_frac} decay-every={args.decay_every}")
    print()
    print(f"{'policy':<22}{'misses':>10}{'hit%':>8}{'MiB read':>10}{'over compulsory':>22}")
    for name, p in t["policies"].items():
        over = p["over_compulsory"]
        over_s = "-" if name == "compulsory" else f"+{over} (+{100.0 * over / max(1, t['policies']['compulsory']['misses']):.1f}%)"
        hit = "-" if p["hit_pct"] is None else f"{p['hit_pct']:.1f}"
        print(f"{name:<22}{p['misses']:>10}{hit:>8}{p['read_mib']:>10}{over_s:>22}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?", help="a binary selection trace (see the module docstring)")
    ap.add_argument("--synthetic", choices=("scan", "loop", "agent", "intermittent", "spotty"),
                    help="run a built-in pattern instead of a trace file")
    ap.add_argument("--layers", type=int, default=2, help="layers for --synthetic (and pools to expect, default 2)")
    ap.add_argument("--calls", type=int, default=2000, help="--synthetic: calls per layer")
    ap.add_argument("--universe", type=int, default=40000, help="--synthetic: distinct blocks in play")
    ap.add_argument("--width", type=int, default=256, help="--synthetic: blocks read per call (scan/loop/intermittent hot)")
    ap.add_argument("--tail", type=int, default=128, help="--synthetic agent: the recent window")
    ap.add_argument("--task", type=int, default=64, help="--synthetic agent: the stable task-critical set")
    ap.add_argument("--oneshot", type=int, default=16, help="--synthetic agent/intermittent: one-off blocks per call")
    ap.add_argument("--every", type=int, default=8, help="--synthetic intermittent: the cold set comes every N calls")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--slots", type=int, default=8192, help="pool size per layer, in blocks (engine: --kv-resident cells / --page)")
    ap.add_argument("--page", type=int, default=4, help="cells per block (the engine's page_size; trace ids are blocks)")
    ap.add_argument("--bytes", type=int, default=4224, help="bytes per block (the engine's int8 K+V page)")
    ap.add_argument("--lambda", dest="lamb", type=float, default=0.5, help="lrfu decay per decay step")
    ap.add_argument("--gain", type=float, default=1.0, help="lrfu score a hit adds")
    ap.add_argument("--top-frac", type=float, default=0.9, help="the weight mass whose blocks accumulate score")
    ap.add_argument("--decay-every", type=int, default=64, help="lrfu decays every N calls (ActKV decays per compression step)")
    ap.add_argument("--json", action="store_true", help="print the numbers as JSON instead of a table")
    a = ap.parse_args()
    if not a.trace and not a.synthetic:
        ap.error("give a TRACE.bin or --synthetic PATTERN")
    tr = synthetic(a.synthetic, a.layers, a.calls, a.universe, a.width, a.tail, a.task, a.oneshot, a.every,
                   a.seed) if a.synthetic else load_trace(a.trace)
    if a.synthetic:
        a.layers = max(1, max((layer for layer, _, _ in tr), default=0) + 1)
    t = run(tr, a.layers, a.slots, a.bytes, a.lamb, a.gain, a.top_frac, a.decay_every)
    if a.json:
        print(json.dumps(t, indent=2))
    else:
        report(t, a)


if __name__ == "__main__":
    main()
