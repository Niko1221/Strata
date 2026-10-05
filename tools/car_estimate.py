#!/usr/bin/env python3
"""car_estimate - what cache-aware routing would do to a run's real routing, offline.

WHAT THIS ANSWERS, BEFORE ANY ENGINE CHANGE IS TRUSTED.  The decode path spends its time on the experts the
VRAM cache does not hold.  Cache-aware routing replaces an uncached pick with the best *resident* expert the
token did not already select, when the router's own score ratio clears a threshold.  Whether that is worth
anything on a given machine and cache size is a property of the ROUTING, not of the kernel, so it can be
measured from a trace: how many misses per token have a resident candidate at all, how many clear a given
threshold, which ranks get replaced, and how much CPU/PCIe expert work stops happening.

`--selftest` runs the rules against hand-computed cases and (when the engine's test binary is available) against
a golden case emitted by `car_substitute_test --emit-golden`, so this tool's copy of the rule cannot silently
drift from the engine's.

THE TRACE FORMAT (`STRCS1`) is what `STRATA_DUMP_ROUTER_SCORES=<path>` writes: little-endian

    magic  "STRCS1\\n" (7 bytes)
    u32    version (1)
    u32    n_layers
    u32    n_expert
    u32    flags (0)
    then records until EOF:
    u32 layer, u32 n_tok, u32 k, i32 ids[n_tok*k], f32 logits[n_tok*n_expert]

The ids are the router's own picks (so the engine's selection, including its tie-break, is exactly what is
scored) and the logits are the raw router scores of that whole row.

Usage:
    tools/car_estimate.py --trace scores.bin --slots 1006
    tools/car_estimate.py --trace scores.bin --profile data/expert-profile-coder.bin
    tools/car_estimate.py --trace scores.bin --frequency 21 --tau 0.2,0.35,0.5 --json out.json
    tools/car_estimate.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile

MAGIC = b"STRCS1\n"
NOT_RESIDENT = -1
# The engine's own clamp for a saturated score difference (`car.cpp`): an `inf` in a report is worse than a
# number nobody can reach.
RATIO_MAX = 3.0e38


class TraceError(ValueError):
    """A trace this tool refuses: the message says which part is wrong and where."""


class Trace:
    def __init__(self, n_layers: int, n_expert: int):
        self.n_layers = n_layers
        self.n_expert = n_expert
        self.records: list[tuple[int, int, int, list[int], list[float]]] = []  # (layer, n_tok, k, ids, logits)

    def add(self, layer: int, n_tok: int, k: int, ids: list[int], logits: list[float]) -> None:
        if n_tok <= 0 or k <= 0:
            raise TraceError("a record must have at least one token and one expert per token")
        if not 0 <= layer < max(self.n_layers, 1):
            raise TraceError(f"record layer {layer} is outside 0..{self.n_layers - 1}")
        if len(ids) != n_tok * k:
            raise TraceError(f"record has {len(ids)} ids for {n_tok} tokens x k={k}")
        if len(logits) != n_tok * self.n_expert:
            raise TraceError(f"record has {len(logits)} logits for {n_tok} tokens x {self.n_expert} experts")
        self.records.append((layer, n_tok, k, list(ids), list(logits)))

    def n_tokens(self) -> int:
        return sum(r[1] for r in self.records)


def read_trace(path: str, strict: bool = False) -> Trace:
    """Read a `STRCS1` trace.  Every failure names what was expected and what was found.

    **A TRACE AN ENGINE IS STILL WRITING ENDS MID-RECORD**, which is normal: the engine buffers a group's record
    and flushes when the buffer fills.  So an incomplete FINAL record is dropped with a warning (the records
    before it are complete and usable) unless `strict` is set.  An incomplete FIRST record is an error - there
    would be nothing to estimate.
    """
    if not os.path.exists(path):
        raise TraceError(f"{path}: no such file")
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 7 + 16 or data[:7] != MAGIC:
        raise TraceError(f"{path}: not a STRCS1 trace (magic {data[:7]!r})")
    off = 7
    version, n_layers, n_expert, flags = struct.unpack_from("<IIII", data, off)
    off += 16
    if version != 1:
        raise TraceError(f"{path}: version {version} is not 1")
    if flags != 0:
        raise TraceError(f"{path}: flags {flags} are not understood by this version")
    if n_layers <= 0 or n_expert <= 0:
        raise TraceError(f"{path}: n_layers {n_layers}, n_expert {n_expert}")
    trace = Trace(n_layers, n_expert)
    dropped = None
    while off < len(data):
        if off + 12 > len(data):
            dropped = f"a {len(data) - off}-byte record header at byte {off}"
            break
        layer, n_tok, k = struct.unpack_from("<III", data, off)
        off += 12
        n_ids = n_tok * k
        if off + 4 * n_ids + 4 * n_tok * n_expert > len(data):
            dropped = (f"the record at byte {off} (layer {layer}, {n_tok} tokens): it needs "
                       f"{4 * n_ids + 4 * n_tok * n_expert} bytes and {len(data) - off} are left")
            break
        ids = list(struct.unpack_from(f"<{n_ids}i", data, off))
        off += 4 * n_ids
        logits = list(struct.unpack_from(f"<{n_tok * n_expert}f", data, off))
        off += 4 * n_tok * n_expert
        trace.add(layer, n_tok, k, ids, logits)
    if dropped is not None:
        if strict or not trace.records:
            raise TraceError(f"{path}: truncated: {dropped}")
        print(f"car_estimate: note: {path} ends mid-write; {dropped} dropped (the engine may still be running)",
              file=sys.stderr)
    if not trace.records:
        raise TraceError(f"{path}: no records")
    return trace


def write_trace(path: str, n_layers: int, n_expert: int, records: list[tuple[int, int, int, list[int], list[float]]]) -> None:
    """Write a trace in the same format (the tests and `--selftest` use this)."""
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", 1, n_layers, n_expert, 0))
        for layer, n_tok, k, ids, logits in records:
            f.write(struct.pack("<III", layer, n_tok, k))
            f.write(struct.pack(f"<{len(ids)}i", *ids))
            f.write(struct.pack(f"<{len(logits)}f", *logits))


def read_profile(path: str) -> list[tuple[int, int]]:
    """`tools/make_profile.py`'s file: `STRP`, version, n_layers, n_expert, slots, n_ranked, the ranked
    (layer, expert) pairs, then the per-pair rank table.  Only the ranked pairs are used here."""
    if not os.path.exists(path):
        raise TraceError(f"{path}: no such file")
    with open(path, "rb") as f:
        head = f.read(4)
        if head != b"STRP":
            raise TraceError(f"{path}: not a profile (magic {head!r})")
        raw = f.read(4 * 5)
        if len(raw) < 20:
            raise TraceError(f"{path}: truncated header")
        version, n_layers, n_expert, slots, n_ranked = struct.unpack("<IIIII", raw)
        if version not in (1, 2):
            raise TraceError(f"{path}: profile version {version} is not 1 or 2")
        pairs = f.read(4 * n_ranked)
        if len(pairs) < 4 * n_ranked:
            raise TraceError(f"{path}: truncated ranking ({len(pairs)} bytes for {n_ranked} pairs)")
    return [(l, e) for l, e in struct.iter_unpack("<HH", pairs)]


def residency_from_profile(pairs: list[tuple[int, int]], n_layers: int, n_expert: int, per_layer: int) -> list[list[int]]:
    """The resident set the engine's profile fill produces: each layer keeps its first `per_layer` ranked
    experts (that is what per-layer admission does with `slots / n_layers` slots)."""
    res = [[] for _ in range(n_layers)]
    for layer, expert in pairs:
        if 0 <= layer < n_layers and 0 <= expert < n_expert and len(res[layer]) < per_layer:
            res[layer].append(expert)
    return res


def residency_by_frequency(trace: Trace, per_layer: int) -> list[list[int]]:
    """The best a cache of that size could do on THIS trace: the `per_layer` most-routed experts of each
    layer.  This is an upper bound (an oracle), and it is what tells a real profile's fill apart from a
    perfect one."""
    counts = [[0] * trace.n_expert for _ in range(trace.n_layers)]
    for layer, _n_tok, k, ids, _logits in trace.records:
        for e in ids:
            if 0 <= e < trace.n_expert:
                counts[layer][e] += 1
    out = []
    for layer in range(trace.n_layers):
        order = sorted(range(trace.n_expert), key=lambda e: (-counts[layer][e], e))
        out.append([e for e in order[:per_layer] if counts[layer][e] > 0])
    return out


def ratio_of(l_cand: float, l_orig: float) -> float:
    """`p_C / p_E` from raw logits - the engine's own definition (`car.cpp`), clamping included."""
    d = l_cand - l_orig
    if math.isnan(d):
        return 0.0
    if d > 88.0:
        return RATIO_MAX
    if d < -88.0:
        return 0.0
    return math.exp(d)


def substitute_row(ids: list[int], logits: list[float], res: list[int], n_expert: int, tau: float,
                   budget_per_token: int = 0, free_ratio: float = 0.0) -> dict:
    """The rule for ONE token, exactly as `car_substitute_test` pins it (best ratio first, ties by lower
    expert id, no expert twice in a token).  `ids`/`logits` are one token's row; returns the substituted ids
    and what was substituted.

    A threshold of 1.0 (or above) is OFF, not "substitute only when the ratio is at least 1": the engine does
    not even scan (`car.cpp`), and a resident expert scored ABOVE an uncached pick would otherwise clear it.
    """
    if not (tau < 1.0):
        misses = sum(1 for e in ids if 0 <= e < n_expert and res[e] == NOT_RESIDENT)
        return {"ids": list(ids), "subs": 0, "misses": misses,
                "with_candidate": 0, "free": 0, "budget_skipped": 0, "ratio_sum": 0.0, "ranks": []}
    used = [False] * n_expert
    for e in ids:
        if 0 <= e < n_expert:
            used[e] = True
    cand = [-1] * len(ids)
    ratio = [-1.0] * len(ids)

    def best(exclude_used: bool) -> tuple[int, float]:
        bi, bl = -1, 0.0
        for c in range(n_expert):
            if res[c] == NOT_RESIDENT:
                continue
            if exclude_used and used[c]:
                continue
            lc = logits[c]
            if math.isnan(lc):
                continue
            if bi < 0 or lc > bl:
                bi, bl = c, lc
        return bi, bl

    misses = 0
    with_candidate = 0
    for j, e in enumerate(ids):
        if not (0 <= e < n_expert) or res[e] != NOT_RESIDENT:
            continue
        misses += 1
        if math.isnan(logits[e]):
            continue
        c, lc = best(exclude_used=True)
        if c < 0:
            continue
        cand[j] = c
        ratio[j] = ratio_of(lc, logits[e])
        with_candidate += 1

    order = sorted((j for j in range(len(ids)) if cand[j] >= 0), key=lambda j: (-ratio[j], j))
    out_ids = list(ids)
    subs = 0
    free = 0
    spent = 0
    budget_skipped = 0
    ratio_sum = 0.0
    ranks = []
    mass = 0.0
    for j in order:
        e = ids[j]
        if used[cand[j]]:
            c, lc = best(exclude_used=True)
            if c < 0:
                continue
            cand[j] = c
            ratio[j] = ratio_of(lc, logits[e])
        if not (ratio[j] >= tau):
            continue
        is_free = free_ratio > 0.0 and ratio[j] >= free_ratio
        if budget_per_token > 0 and spent >= budget_per_token and not is_free:
            budget_skipped += 1
            continue
        out_ids[j] = cand[j]
        used[cand[j]] = True
        if is_free:
            free += 1
        else:
            spent += 1
        subs += 1
        ratio_sum += ratio[j]
        ranks.append(j)
        mass += ratio[j]
    return {"ids": out_ids, "subs": subs, "misses": misses, "with_candidate": with_candidate,
            "free": free, "budget_skipped": budget_skipped, "ratio_sum": ratio_sum, "ranks": ranks}


def estimate(trace: Trace, resident: list[list[int]], tau: float, k_expected: int | None = None,
             blob_bytes: int = 1382400, pcie_frac: float = 0.0) -> dict:
    """Run the rule over every record of the trace with the given per-layer resident sets."""
    n_expert = trace.n_expert
    rank_hist = [0] * 64
    total = {"tokens": 0, "entries": 0, "misses": 0, "with_candidate": 0, "subs": 0, "ratio_sum": 0.0,
             "budget_skipped": 0, "free": 0, "by_layer": {}}
    for layer, n_tok, k, ids, logits in trace.records:
        if k_expected is not None and k != k_expected:
            raise TraceError(f"record on layer {layer} has k={k}, expected {k_expected}")
        res = [NOT_RESIDENT] * n_expert
        for slot, e in enumerate(resident[layer] if layer < len(resident) else []):
            if 0 <= e < n_expert:
                res[e] = slot
        layer_stats = total["by_layer"].setdefault(layer, {"tokens": 0, "misses": 0, "subs": 0, "ratio_sum": 0.0})
        for t in range(n_tok):
            row_ids = ids[t * k:(t + 1) * k]
            row_logits = logits[t * n_expert:(t + 1) * n_expert]
            r = substitute_row(row_ids, row_logits, res, n_expert, tau)
            total["tokens"] += 1
            total["entries"] += k
            total["misses"] += r["misses"]
            total["with_candidate"] += r["with_candidate"]
            total["subs"] += r["subs"]
            total["ratio_sum"] += r["ratio_sum"]
            total["budget_skipped"] += r["budget_skipped"]
            total["free"] += r["free"]
            layer_stats["tokens"] += 1
            layer_stats["misses"] += r["misses"]
            layer_stats["subs"] += r["subs"]
            layer_stats["ratio_sum"] += r["ratio_sum"]
            for j in r["ranks"]:
                rank_hist[j if j < len(rank_hist) else len(rank_hist) - 1] += 1

    tokens = max(total["tokens"], 1)
    misses = max(total["misses"], 1)
    subs_per_token = total["subs"] / tokens
    miss_per_token = total["misses"] / tokens
    out = {
        "tau": tau,
        "tokens": total["tokens"],
        "entries": total["entries"],
        "misses": total["misses"],
        "misses_per_token": miss_per_token,
        "with_candidate": total["with_candidate"],
        "substitutions": total["subs"],
        "substitutions_per_token": subs_per_token,
        "substitution_rate_of_misses": total["subs"] / misses,
        "mean_ratio": (total["ratio_sum"] / total["subs"]) if total["subs"] else 0.0,
        "budget_skipped": total["budget_skipped"],
        "free": total["free"],
        "misses_after_per_token": (total["misses"] - total["subs"]) / tokens,
        "expert_mb_not_moved_per_token": subs_per_token * blob_bytes / (1024.0 * 1024.0),
        "pcie_mb_not_moved_per_token": subs_per_token * blob_bytes * pcie_frac / (1024.0 * 1024.0),
        "rank_histogram": {str(i): c for i, c in enumerate(rank_hist) if c},
        "resident_per_layer_max": max((len(r) for r in resident), default=0),
        "resident_per_layer_mean": (sum(len(r) for r in resident) / len(resident)) if resident else 0.0,
    }
    return out


def format_report(name: str, est: dict) -> str:
    lines = [
        f"== {name}: tau {est['tau']}, {est['tokens']} tokens, "
        f"{est['resident_per_layer_max']} resident experts/layer ==",
        f"   misses/token            {est['misses_per_token']:.2f} of {est['entries'] / max(est['tokens'], 1):.1f} routed",
        f"   substitutable misses    {est['with_candidate']} ({est['with_candidate'] / max(est['misses'], 1) * 100:.1f}%)",
        f"   substitutions           {est['substitutions']} = {est['substitutions_per_token']:.2f}/token "
        f"({est['substitution_rate_of_misses'] * 100:.1f}% of misses)",
        f"   misses left per token   {est['misses_after_per_token']:.2f}",
        f"   mean accepted ratio     {est['mean_ratio']:.3f}",
        f"   expert MB not moved     {est['expert_mb_not_moved_per_token']:.2f} MB/token",
    ]
    if est["rank_histogram"]:
        top = sorted(est["rank_histogram"].items(), key=lambda kv: int(kv[0]))
        lines.append("   substitutions by rank   " + " ".join(f"r{r}:{c}" for r, c in top))
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------
# --selftest: the rules against hand-computed cases, and (when it is built) the engine's own golden case
def _hand_cases() -> list[str]:
    problems = []

    def expect(cond: bool, what: str) -> None:
        if not cond:
            problems.append(what)

    # resident 0 at 10.0.  Miss 4 (9.0) -> ratio exp(1) = 2.718: substituted.  Miss 9 (12.0) -> exp(-2) = 0.135: not.
    res = [NOT_RESIDENT] * 16
    res[0] = 0
    logits = [-50.0] * 16
    logits[0], logits[4], logits[9] = 10.0, 9.0, 12.0
    r = substitute_row([9, 4], logits, res, 16, 0.35)
    expect(r["ids"] == [9, 0], f"ratio test: got {r['ids']}, want [9, 0]")
    expect(r["subs"] == 1, f"ratio test: {r['subs']} substitutions, want 1")
    expect(abs(r["ratio_sum"] - math.exp(1.0)) < 1e-9, "ratio test: the accepted ratio is exp(1)")

    # best ratio first: miss 4 (ratio exp(5.5)) beats miss 5 (exp(5)) for the only resident expert
    res = [NOT_RESIDENT] * 8
    res[0] = 0
    logits = [-50.0] * 8
    logits[0], logits[4], logits[5] = 10.0, 4.5, 5.0
    r = substitute_row([4, 5], logits, res, 8, 0.0)
    expect(r["ids"] == [0, 5], f"best ratio first: got {r['ids']}, want [0, 5]")

    # no candidate: an empty cache substitutes nothing
    res = [NOT_RESIDENT] * 8
    r = substitute_row([1, 2], [-50.0] * 8, res, 8, 0.0)
    expect(r["ids"] == [1, 2] and r["subs"] == 0 and r["misses"] == 2, "empty cache: nothing substituted")

    # the threshold 1.0 never substitutes
    res = [NOT_RESIDENT] * 8
    res[0] = 0
    logits = [-50.0] * 8
    logits[0], logits[3] = 10.0, -10.0
    r = substitute_row([3, 4], logits, res, 8, 1.0)
    expect(r["subs"] == 0 and r["ids"] == [3, 4], "tau 1.0 substitutes nothing")
    return problems


def _golden_check(problems: list[str]) -> str:
    """Compare this tool's rule with the engine's, using the engine's own golden case when it is built."""
    candidates = [os.path.join("build-cpu", "car_substitute_test"), os.path.join("build-hip", "car_substitute_test"),
                  os.path.join("build-plan-host", "car_substitute_test")]
    binary = next((c for c in candidates if os.path.isfile(c) and os.access(c, os.X_OK)), None)
    if binary is None:
        return "golden: SKIPPED (car_substitute_test is not built in build-cpu/ or build-hip/)"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "golden.txt")
        try:
            subprocess.run([binary, "--emit-golden", path], check=True, capture_output=True)
        except (subprocess.CalledProcessError, OSError) as exc:
            problems.append(f"golden: could not run {binary}: {exc}")
            return "golden: FAILED to run the engine's test binary"
        fields: dict[str, list[str]] = {}
        with open(path) as f:
            for line in f:
                parts = line.split()
                if parts:
                    fields[parts[0]] = parts[1:]
        try:
            n_expert, k, n_tok = int(fields["n_expert"][0]), int(fields["k"][0]), int(fields["n_tok"][0])
            tau = float(fields["threshold"][0])
            res = [int(x) for x in fields["res"]]
            logits = [float(x) for x in fields["logits"]]
            ids = [int(x) for x in fields["ids"]]
            expected = [int(x) for x in fields["expected"]]
        except (KeyError, IndexError, ValueError) as exc:
            problems.append(f"golden: unreadable case ({exc})")
            return "golden: FAILED to parse the engine's case"
        got = list(ids)
        subs = 0
        for t in range(n_tok):
            r = substitute_row(ids[t * k:(t + 1) * k], logits[t * n_expert:(t + 1) * n_expert], res, n_expert, tau)
            got[t * k:(t + 1) * k] = r["ids"]
            subs += r["subs"]
        if got != expected:
            problems.append(f"golden: the engine substituted to {expected}, this tool to {got}")
            return "golden: FAILED (the two implementations disagree)"
        return f"golden: OK ({subs} substitutions, ids match the engine's byte for byte)"


def selftest() -> int:
    problems = _hand_cases()
    golden = _golden_check(problems)

    # the trace reader: a round trip, and each way it must refuse a file
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.bin")
        write_trace(p, 2, 4, [(0, 1, 2, [3, 1], [-1.0, -2.0, -3.0, 7.0]),
                              (1, 2, 2, [0, 1, 2, 3], [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2])])
        t = read_trace(p)
        if t.n_layers != 2 or t.n_expert != 4 or t.n_tokens() != 3 or len(t.records) != 2:
            problems.append("trace: the round trip did not come back")
        with open(p, "r+b") as f:
            f.seek(0)
            f.write(b"NOPE!!!")
        try:
            read_trace(p)
            problems.append("trace: a bad magic was accepted")
        except TraceError:
            pass
        with open(p, "wb") as f:
            f.write(MAGIC + struct.pack("<IIII", 1, 2, 4, 0) + struct.pack("<III", 0, 2, 2) + b"\x00" * 4)
        try:
            read_trace(p)
            problems.append("trace: a truncated record was accepted")
        except TraceError:
            pass

        # the estimator end to end on a case whose answer is known by hand: one layer, one token, one
        # resident expert (3).  Both picks are misses; 1 (scored 9.0) against the resident expert's 10.0 is
        # exp(1) and is substituted, then 0 (9.5) loses the expert to it on the rescan and stays a miss.
        p2 = os.path.join(d, "est.bin")
        logits = [-50.0] * 4
        logits[3] = 10.0
        logits[1], logits[0] = 9.0, 9.5
        write_trace(p2, 1, 4, [(0, 1, 2, [1, 0], logits)])
        est = estimate(read_trace(p2), [[3]], tau=0.35)
        if est["misses"] != 2 or est["substitutions"] != 1:
            problems.append(f"estimate: want 2 misses and 1 substitution, got {est['misses']} and {est['substitutions']}")
        if abs(est["mean_ratio"] - math.exp(1.0)) > 1e-6:
            problems.append(f"estimate: mean ratio {est['mean_ratio']} is not exp(1)")
        if abs(est["misses_after_per_token"] - 1.0) > 1e-9:
            problems.append("estimate: one miss should be left per token")

        # the oracle residency ranks by routed frequency, ties by lower expert id
        p3 = os.path.join(d, "freq.bin")
        small = [-50.0] * 4
        write_trace(p3, 1, 4, [(0, 3, 1, [2, 2, 2], small[:4]), (0, 1, 1, [0], small[:4])])
        freq = residency_by_frequency(read_trace(p3), 1)
        if freq != [[2]]:
            problems.append(f"frequency ranking: want [[2]], got {freq}")

    for problem in problems:
        print(f"FAIL: {problem}")
    print(golden)
    if problems:
        print(f"car_estimate selftest: {len(problems)} failure(s)")
        return 1
    print("car_estimate selftest: all cases passed")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="estimate cache-aware routing from a STRCS1 router-score trace")
    ap.add_argument("--trace", help="a STRCS1 trace (STRATA_DUMP_ROUTER_SCORES=<path> writes one)")
    ap.add_argument("--profile", help="a profile.bin (tools/make_profile.py) to define the resident sets")
    ap.add_argument("--slots", type=int, help="cache slots; each layer gets slots // n_layers of them")
    ap.add_argument("--per-layer", type=int, help="resident experts per layer (overrides --slots)")
    ap.add_argument("--frequency", type=int, help="ORACLE: the N hottest experts per layer, from the trace itself")
    ap.add_argument("--tau", default="0.35", help="thresholds to score, comma separated (default 0.35)")
    ap.add_argument("--blob-bytes", type=int, default=1382400, help="bytes per expert blob (default 1382400)")
    ap.add_argument("--pcie-frac", type=float, default=0.0, help="the share of misses that would read over PCIe")
    ap.add_argument("--json", help="write the estimates to this file")
    ap.add_argument("--selftest", action="store_true", help="check the rules and the reader, then exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.trace:
        ap.error("--trace is required (or --selftest)")

    trace = read_trace(args.trace)
    per_layer = args.per_layer
    if per_layer is None and args.slots is not None:
        per_layer = max(args.slots // max(trace.n_layers, 1), 1)
    if per_layer is None:
        per_layer = 21

    sets: list[tuple[str, list[list[int]]]] = []
    if args.frequency is not None:
        sets.append((f"oracle (top {args.frequency}/layer by this trace's routing)",
                     residency_by_frequency(trace, args.frequency)))
    if args.profile:
        sets.append((f"profile {args.profile} ({per_layer}/layer)",
                     residency_from_profile(read_profile(args.profile), trace.n_layers, trace.n_expert, per_layer)))
    if not sets:
        sets.append((f"oracle (top {per_layer}/layer by this trace's routing)",
                     residency_by_frequency(trace, per_layer)))

    taus = [float(x) for x in args.tau.split(",") if x.strip()]
    out: dict = {"trace": args.trace, "n_layers": trace.n_layers, "n_expert": trace.n_expert,
                 "tokens": trace.n_tokens(), "per_layer": per_layer, "estimates": []}
    for name, resident in sets:
        for tau in taus:
            est = estimate(trace, resident, tau, blob_bytes=args.blob_bytes, pcie_frac=args.pcie_frac)
            est["resident_set"] = name
            out["estimates"].append(est)
            print(format_report(name, est))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except TraceError as exc:
        # A bad trace is a user's file, not a bug in this tool: say what is wrong and stop, with no traceback.
        print(f"car_estimate: {exc}", file=sys.stderr)
        sys.exit(2)
