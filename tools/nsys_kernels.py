#!/usr/bin/env python3
"""tools/nsys_kernels.py - what the GPU did in an Nsight Systems trace.

WHAT IT DOES
  It reads the SQLite that `nsys stats` writes next to a `.nsys-rep` and answers
  three questions about a window of the trace:

    1. which kernels ran, how often, how long each took, and what share of the
       GPU's time each one has
    2. how much time the host-to-device, device-to-host and device-to-device
       copies took, and how many bytes they moved
    3. how much of the window the GPU was busy, and where it sat idle

  The window matters: a trace of a real run holds the model load, the expert
  cache fill and the requests in one timeline.  The tail of the trace (the last
  request) is the decode; the beginning is the load.

    nsys stats --report cuda_gpu_kern_sum strata.nsys-rep   # writes strata.sqlite
    python tools/nsys_kernels.py --sqlite strata.sqlite --tail-seconds 67

  The table is written as JSON and as Markdown, so the numbers in a report can
  be re-derived from the trace.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def load_kernels(con, t0: int, t1: int) -> list[dict]:
    """Every kernel in the window, with its name, its times and its launch shape.

    The name columns in the SQLite are keys into StringIds, not text: a kernel
    launched from a CUDA graph has no name of its own there, and the key points
    at the graph node's number."""
    rows = con.execute(
        "select k.start, k.end, s.value, k.gridX, k.gridY, k.gridZ, "
        "       k.blockX, k.blockY, k.blockZ, k.staticSharedMemory, k.dynamicSharedMemory, k.streamId, "
        "       k.demangledName, k.shortName "
        "from CUPTI_ACTIVITY_KIND_KERNEL k left join StringIds s on s.id = k.demangledName "
        "where k.start < ? and k.end > ?", (t1, t0)).fetchall()
    out = []
    for r in rows:
        name = r[2] or f"node {r[12]}"
        out.append({"start": r[0], "end": r[1], "ns": r[1] - r[0], "name": name,
                    "short": name.split("(")[0][:70], "grid": [r[3], r[4], r[5]], "block": [r[6], r[7], r[8]],
                    "static_smem": r[9] or 0, "dynamic_smem": r[10] or 0, "stream": r[11],
                    "node": r[12], "short_id": r[13]})
    return out


def load_copies(con, t0: int, t1: int) -> list[dict]:
    """Every copy in the window, with its size and its time."""
    rows = con.execute(
        "select m.start, m.end, m.bytes, m.copyKind, m.streamId from CUPTI_ACTIVITY_KIND_MEMCPY m "
        "where m.start < ? and m.end > ?", (t1, t0)).fetchall()
    kinds = {1: "H2D", 2: "D2H", 3: "D2D", 4: "H2D (pinned)", 5: "D2H (pinned)"}
    return [{"start": r[0], "end": r[1], "ns": r[1] - r[0], "bytes": r[2] or 0,
             "kind": kinds.get(r[3], str(r[3])), "stream": r[4]} for r in rows]


def busy_time(intervals: list[tuple[int, int]]) -> int:
    """The time covered by the intervals, counted once (a kernel on another stream overlaps)."""
    if not intervals:
        return 0
    merged = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s for s, e in merged)


def kernel_table(kernels: list[dict], window_ns: int) -> list[dict]:
    """Per kernel: how often it ran, how long, and what share of the GPU's time it has."""
    agg: dict[str, dict] = {}
    for k in kernels:
        a = agg.setdefault(k["name"], {"name": k["name"], "short": k["short"], "instances": 0, "total_ns": 0,
                                       "max_ns": 0, "grid": k["grid"], "block": k["block"],
                                       "static_smem": k["static_smem"], "dynamic_smem": k["dynamic_smem"],
                                       "node": k.get("node")})
        a["instances"] += 1
        a["total_ns"] += k["ns"]
        a["max_ns"] = max(a["max_ns"], k["ns"])
    rows = []
    for a in agg.values():
        rows.append({"name": a["name"], "short": a["short"], "node": a["node"], "instances": a["instances"],
                     "total_ms": round(a["total_ns"] / 1e6, 3), "avg_us": round(a["total_ns"] / a["instances"] / 1e3, 2),
                     "max_us": round(a["max_ns"] / 1e3, 2),
                     "pct_of_window": round(100.0 * a["total_ns"] / window_ns, 2) if window_ns else None,
                     "grid": a["grid"], "block": a["block"], "static_smem_kb": round(a["static_smem"] / 1024, 1),
                     "dynamic_smem_kb": round(a["dynamic_smem"] / 1024, 1)})
    rows.sort(key=lambda r: -r["total_ms"])
    return rows


def copy_table(copies: list[dict]) -> list[dict]:
    agg: dict[str, dict] = {}
    for c in copies:
        a = agg.setdefault(c["kind"], {"kind": c["kind"], "count": 0, "total_ns": 0, "bytes": 0, "max_ns": 0})
        a["count"] += 1
        a["total_ns"] += c["ns"]
        a["bytes"] += c["bytes"]
        a["max_ns"] = max(a["max_ns"], c["ns"])
    return [{"kind": a["kind"], "count": a["count"], "total_ms": round(a["total_ns"] / 1e6, 3),
             "max_ms": round(a["max_ns"] / 1e6, 3), "total_mb": round(a["bytes"] / 1048576, 2),
             "avg_mb": round(a["bytes"] / a["count"] / 1048576, 3) if a["count"] else 0}
            for a in sorted(agg.values(), key=lambda x: -x["total_ns"])]


def buckets(kernels: list[dict], t0: int, t1: int, bucket_ns: int) -> list[dict]:
    """The GPU's busy and idle time per bucket, so the pattern of the run is visible."""
    out = []
    t = t0
    while t < t1:
        end = min(t + bucket_ns, t1)
        inb = [k for k in kernels if k["start"] < end and k["end"] > t]
        busy = busy_time([(max(k["start"], t), min(k["end"], end)) for k in inb])
        out.append({"t_s": round((t - t0) / 1e9, 2), "busy_ms": round(busy / 1e6, 2),
                    "idle_ms": round((end - t - busy) / 1e6, 2), "kernels": len(inb)})
        t = end
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sqlite", required=True, help="the SQLite `nsys stats` wrote next to the .nsys-rep")
    ap.add_argument("--tail-seconds", type=float, default=0.0, help="the window: the last N seconds of the trace")
    ap.add_argument("--from-seconds", type=float, default=0.0, help="the window's start, in seconds from the trace's start")
    ap.add_argument("--to-seconds", type=float, default=0.0, help="the window's end (0 = the trace's end)")
    ap.add_argument("--out", default=None, help="where the JSON and the Markdown table go")
    ap.add_argument("--top", type=int, default=25, help="how many kernels the table shows (default: %(default)s)")
    args = ap.parse_args(argv)

    con = sqlite3.connect(args.sqlite)
    row = con.execute("select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    t_min, t_max = row
    if not t_min:
        print("the trace has no kernels")
        return 1
    if args.tail_seconds:
        t0 = t_max - int(args.tail_seconds * 1e9)
    else:
        t0 = t_min + int(args.from_seconds * 1e9)
    t1 = t_max if not args.to_seconds else t_min + int(args.to_seconds * 1e9)
    window_ns = max(1, t1 - t0)

    kernels = load_kernels(con, t0, t1)
    copies = load_copies(con, t0, t1)
    ktab = kernel_table(kernels, window_ns)
    ctab = copy_table(copies)
    busy = busy_time([(k["start"], k["end"]) for k in kernels])
    out = {
        "sqlite": args.sqlite,
        "trace_start_ns": t_min, "trace_end_ns": t_max,
        "window": {"start_ns": t0, "end_ns": t1, "seconds": round(window_ns / 1e9, 3),
                   "from_trace_start_s": round((t0 - t_min) / 1e9, 3)},
        "gpu": {"busy_ms": round(busy / 1e6, 2), "idle_ms": round((window_ns - busy) / 1e6, 2),
                "busy_pct": round(100.0 * busy / window_ns, 2),
                "kernel_launches": len(kernels), "kernel_ms_sum": round(sum(k["ns"] for k in kernels) / 1e6, 2)},
        "copies": ctab,
        "kernels": ktab,
        "buckets_1s": buckets(kernels, t0, t1, 1_000_000_000),
    }
    text = json.dumps(out, indent=2)
    print(text if not args.out else f"written: {args.out}")
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        md = p.with_suffix(".md")
        lines = [f"# kernels in the window", "",
                 f"- window: {out['window']['seconds']} s "
                 f"({out['window']['from_trace_start_s']} s from the trace's start)",
                 f"- GPU busy: {out['gpu']['busy_ms']} ms ({out['gpu']['busy_pct']} %), "
                 f"idle: {out['gpu']['idle_ms']} ms",
                 f"- kernel launches: {out['gpu']['kernel_launches']} "
                 f"(sum of their times: {out['gpu']['kernel_ms_sum']} ms)", "",
                 "## copies", "", "| kind | count | total ms | max ms | total MB | avg MB |", "|---|---|---|---|---|---|"]
        for c in ctab:
            lines.append(f"| {c['kind']} | {c['count']} | {c['total_ms']} | {c['max_ms']} | {c['total_mb']} | {c['avg_mb']} |")
        lines += ["", "## kernels", "", "| # | kernel | launches | total ms | avg us | max us | % of window | grid | block | smem KB |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for i, k in enumerate(ktab[:args.top], 1):
            lines.append(f"| {i} | {k['short']} | {k['instances']} | {k['total_ms']} | {k['avg_us']} | {k['max_us']} | "
                         f"{k['pct_of_window']} | {k['grid']} | {k['block']} | "
                         f"{k['static_smem_kb']}/{k['dynamic_smem_kb']} |")
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"written: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
