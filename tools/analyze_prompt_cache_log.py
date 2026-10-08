"""Summarize Strata prompt-cache trace lines and identify expensive requests.

    python tools/analyze_prompt_cache_log.py server.log [server-2.log ...]
    python tools/analyze_prompt_cache_log.py --json server.log

Only lines containing ``strata serve: prompt cache:`` are considered.  Timestamps,
log levels, and other text before that prefix are allowed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PREFIX = "strata serve: prompt cache:"
EVENT_ORDER = ("decision", "checkpoint", "park", "restore")
TOP_N = 10

INT_FIELDS = {
    "decision": {"req", "prompt", "live", "checkpoint", "slot", "parked", "resume", "read_from",
                 "scan_entries", "scan_checkpoints", "images", "pin", "cvec_match", "ckpt"},
    "checkpoint": {"req", "tokens", "kept", "evicted_tokens", "gdn_bytes", "ple_bytes", "index_bytes"},
    "park": {"req", "tokens", "estimate_bytes", "fresh_estimate_bytes", "retained_bytes", "additional_bytes",
             "held_bytes", "stored"},
    "restore": {"req", "tokens", "bytes"},
}
FLOAT_FIELDS = {
    "decision": {"scan_ms"},
    "checkpoint": {"save_ms", "sync_ms", "copy_ms", "stage_sync_ms", "stage_copy_ms",
                   "alloc_ms", "gdn_ms", "ple_ms", "index_ms"},
    "park": {"save_ms", "estimate_ms", "capture_ms", "stage_sync_ms", "stage_capture_ms", "put_ms"},
    "restore": {"restore_ms"},
}
WORD_FIELDS = {
    "decision": {"source"},
    "checkpoint": {"kind", "evicted_kind"},
    "park": set(),
    "restore": {"source"},
}
BOOL_FIELDS = {"decision": {"cvec_match", "ckpt"}, "checkpoint": set(),
               "park": {"stored"}, "restore": set()}


def parse_line(line: str, file: str = "<input>", line_number: int = 0) -> dict | None:
    """Parse one supported trace event, or return None for unrelated/malformed input."""
    marker = line.find(PREFIX)
    if marker < 0:
        return None
    words = line[marker + len(PREFIX):].strip().split()
    if not words or words[0] not in EVENT_ORDER:
        return None
    event = words[0]
    fields = {}
    known = INT_FIELDS[event] | FLOAT_FIELDS[event] | WORD_FIELDS[event]
    for word in words[1:]:
        if "=" not in word:
            return None
        key, value = word.split("=", 1)
        if not key or not value:
            return None
        if key not in known:          # Permit future fields without guessing their types.
            continue
        try:
            if key in INT_FIELDS[event]:
                parsed = int(value)
                if parsed < 0 or key in BOOL_FIELDS[event] and parsed not in (0, 1):
                    return None
            elif key in FLOAT_FIELDS[event]:
                parsed = float(value)
                if parsed < 0 or parsed != parsed or parsed in (float("inf"), float("-inf")):
                    return None
            else:
                parsed = value
        except ValueError:
            return None
        fields[key] = parsed
    if not fields:
        return None
    return {"event": event, "file": file, "line": line_number, **fields}


def _rank(records: list[dict], metric: str) -> list[dict]:
    """Return the largest metrics first with source location as a deterministic tie-breaker."""
    return sorted(records, key=lambda row: (-row[metric], row["file"], row["line"]))[:TOP_N]


def analyze(records: list[dict], files: list[str] | None = None) -> dict:
    """Build the deterministic JSON-ready report used by both output modes."""
    groups = {event: [] for event in EVENT_ORDER}
    for record in records:
        groups[record["event"]].append(record)

    rereads = []
    for row in groups["decision"]:
        if "prompt" not in row:
            continue
        basis = "read_from" if "read_from" in row else "resume" if "resume" in row else None
        if basis is None:
            continue
        rereads.append({**row, "reread_tokens": max(0, row["prompt"] - row[basis]), "reread_basis": basis})

    decisions = groups["decision"]
    reused = sum(row.get("source", "").lower() not in ("", "none") for row in decisions)
    misses = [row for row in decisions if row.get("source", "").lower() == "none"]
    partial_rereads = [row for row in rereads
                       if row.get("source", "").lower() not in ("", "none") and row["reread_tokens"] > 0]
    cold_reads = [{**row, "cold_read_tokens": row["prompt"]} for row in misses if "prompt" in row]
    unclassified = sum("source" not in row for row in decisions)
    checkpoint_phases = {
        "checkpoint_sync": sum(row.get("sync_ms", 0) for row in groups["checkpoint"]),
        "checkpoint_copy": sum(row.get("copy_ms", 0) for row in groups["checkpoint"]),
        "checkpoint_stage_sync": sum(row.get("stage_sync_ms", 0) for row in groups["checkpoint"]),
        "checkpoint_stage_copy": sum(row.get("stage_copy_ms", 0) for row in groups["checkpoint"]),
        "checkpoint_allocation": sum(row.get("alloc_ms", 0) for row in groups["checkpoint"]),
        "checkpoint_gdn": sum(row.get("gdn_ms", 0) for row in groups["checkpoint"]),
        "checkpoint_ple": sum(row.get("ple_ms", 0) for row in groups["checkpoint"]),
        "checkpoint_index": sum(row.get("index_ms", 0) for row in groups["checkpoint"]),
    }
    checkpoint_phases["checkpoint_other"] = sum(max(0, row.get("save_ms", 0) - row.get("sync_ms", 0) -
                                                         row.get("copy_ms", 0) - row.get("stage_sync_ms", 0) -
                                                         row.get("stage_copy_ms", 0))
                                                       for row in groups["checkpoint"])
    park_phases = {
        "park_estimate": sum(row.get("estimate_ms", 0) for row in groups["park"]),
        "park_capture": sum(row.get("capture_ms", 0) for row in groups["park"]),
        "park_stage_sync": sum(row.get("stage_sync_ms", 0) for row in groups["park"]),
        "park_stage_capture": sum(row.get("stage_capture_ms", 0) for row in groups["park"]),
        "park_put": sum(row.get("put_ms", 0) for row in groups["park"]),
    }
    park_phases["park_other_save"] = sum(max(0, row.get("save_ms", 0) - row.get("capture_ms", 0) -
                                                row.get("stage_sync_ms", 0) - row.get("stage_capture_ms", 0) -
                                                row.get("put_ms", 0))
                                              for row in groups["park"])
    summary = {
        "event_counts": {event: len(groups[event]) for event in EVENT_ORDER},
        "events": len(records),
        "decisions": len(decisions),
        "reused": reused,
        "misses": len(misses),
        "unclassified": unclassified,
        "reuse_rate": reused / len(decisions) if decisions else None,
        "phase_totals_ms": {**checkpoint_phases, **park_phases},
    }
    hot_spots = {
        "largest_rereads": _rank(partial_rereads, "reread_tokens"),
        "cold_reads": _rank(cold_reads, "cold_read_tokens"),
        "slow_scans": _rank([row for row in decisions if "scan_ms" in row], "scan_ms"),
        "checkpoint_saves": _rank([row for row in groups["checkpoint"] if "save_ms" in row], "save_ms"),
        "park_saves": _rank([row for row in groups["park"] if "save_ms" in row], "save_ms"),
        "restores": _rank([row for row in groups["restore"] if "restore_ms" in row], "restore_ms"),
        "misses": sorted(misses, key=lambda row: (row["file"], row["line"]))[:TOP_N],
    }
    return {"files": list(files or []), "summary": summary, "hot_spots": hot_spots}


def read_logs(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                record = parse_line(line, str(path), line_number)
                if record is not None:
                    records.append(record)
    return records


def _location(row: dict) -> str:
    req = f" req={row['req']}" if "req" in row else ""
    return f"{row['file']}:{row['line']}{req}"


def format_text(report: dict) -> str:
    summary = report["summary"]
    rate = "n/a" if summary["reuse_rate"] is None else f"{100 * summary['reuse_rate']:.1f}%"
    lines = [
        f"Prompt cache: {summary['events']} events, {summary['decisions']} decisions",
        (f"Reuse: {summary['reused']}/{summary['decisions']} ({rate}); misses: {summary['misses']}; "
         f"unclassified: {summary['unclassified']}"),
    ]
    phases = summary["phase_totals_ms"]
    lines.extend([
        ("Checkpoint phases total: "
         f"sync={phases['checkpoint_sync']:g} ms copy={phases['checkpoint_copy']:g} ms "
         f"stage_sync={phases['checkpoint_stage_sync']:g} ms "
         f"stage_copy={phases['checkpoint_stage_copy']:g} ms other={phases['checkpoint_other']:g} ms"),
        ("Checkpoint copy detail: "
         f"allocation={phases['checkpoint_allocation']:g} ms gdn={phases['checkpoint_gdn']:g} ms "
         f"ple={phases['checkpoint_ple']:g} ms index={phases['checkpoint_index']:g} ms"),
        ("Park phases total: "
         f"estimate={phases['park_estimate']:g} ms capture={phases['park_capture']:g} ms "
         f"stage_sync={phases['park_stage_sync']:g} ms stage_capture={phases['park_stage_capture']:g} ms "
         f"put={phases['park_put']:g} ms other={phases['park_other_save']:g} ms"),
    ])
    specs = (
        ("Largest partial rereads", "largest_rereads", "reread_tokens", "tokens"),
        ("Largest cold reads", "cold_reads", "cold_read_tokens", "tokens"),
        ("Slow scans", "slow_scans", "scan_ms", "ms"),
        ("Checkpoint saves", "checkpoint_saves", "save_ms", "ms"),
        ("Park saves", "park_saves", "save_ms", "ms"),
        ("Restores", "restores", "restore_ms", "ms"),
    )
    for title, key, metric, unit in specs:
        rows = report["hot_spots"][key]
        lines.append(f"\n{title} (top {TOP_N}):")
        if not rows:
            lines.append("  none")
        for row in rows:
            detail = f" basis={row['reread_basis']}" if key == "largest_rereads" else ""
            if key == "checkpoint_saves":
                detail = (f" sync={row.get('sync_ms', 0):g} copy={row.get('copy_ms', 0):g}"
                           f" stage_sync={row.get('stage_sync_ms', 0):g}"
                           f" stage_copy={row.get('stage_copy_ms', 0):g}"
                           f" alloc={row.get('alloc_ms', 0):g} gdn={row.get('gdn_ms', 0):g}"
                           f" ple={row.get('ple_ms', 0):g} index={row.get('index_ms', 0):g}")
            elif key == "park_saves":
                detail = (f" estimate={row.get('estimate_ms', 0):g} capture={row.get('capture_ms', 0):g}"
                          f" stage_sync={row.get('stage_sync_ms', 0):g}"
                          f" stage_capture={row.get('stage_capture_ms', 0):g} put={row.get('put_ms', 0):g}")
            lines.append(f"  {row[metric]:g} {unit}  {_location(row)}{detail}")
    lines.append(f"\nMisses / source none (top {TOP_N}):")
    misses = report["hot_spots"]["misses"]
    lines.extend((f"  {_location(row)}" for row in misses))
    if not misses:
        lines.append("  none")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", type=Path, help="server log files")
    parser.add_argument("--json", action="store_true", help="write a deterministic machine-readable report")
    args = parser.parse_args(argv)
    try:
        records = read_logs(args.logs)
    except OSError as error:
        parser.error(str(error))
    report = analyze(records, [str(path) for path in args.logs])
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(format_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
