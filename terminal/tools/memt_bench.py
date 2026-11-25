"""tools/memt_bench.py - P7: the same prompts, the same card, the median of repeated runs.

Measures what a user feels on both implementations:

  * TTFT  - time to the first generated token (the prompt's read + prefill),
  * TPOT  - time per output token after the first ((end - TTFT) / (n_out - 1)),
  * tok/s - generated tokens over the whole request (end to end),

TPOT is derived from the request's end time because neither protocol reports a
per-token timestamp.  The engine flushes `T <id>` as it emits, so its TTFT is a
real first-token time; the reference streams SSE frames, and a server that
batches frames makes its TTFT later and its TPOT smaller (the gap shows up as
`frames < output_tokens - 1` in the report).  tok/s and the end-to-end latency
are unaffected by that.

for the memt_sglang server (`--side ref`) and for this fork's `strata-q36`
(`--side engine`), and prints the ratio.  Every run is greedy
(`temperature=0`), starts from an empty prompt cache (the engine is sent
`MEMFORGET`; the reference's Nano configuration keeps no retained KV), and
repeats `--runs` times after `--warmup` discarded requests; the report carries
the median of each field.

This tool reports; it does not invent.  A side that did not run is an error, not
a zero, and the numbers are useless unless both sides run on the same card with
the same prompt ids and `--max-new` — so two sides that generated different
numbers of tokens are reported as NOT COMPARABLE and the run fails.

    python tools/memt_bench.py --side both \\
        --ref-url http://127.0.0.1:30000 \\
        --engine /path/strata-q36 --engine-arg=--serve --engine-arg=--model --engine-arg nano.memt.gguf \\
        --prompt prompt.ids --max-new 128 --runs 3 --outdir runs/p7
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import memt_engine as ME  # noqa: E402
import memt_harness as H  # noqa: E402
import memt_ref as R  # noqa: E402

__all__ = ["rate", "summarize_side", "engine_measure", "reference_measure", "run_bench", "main"]

SAMPLING = H.GREEDY_SAMPLING   # greedy on both sides; no top_k/top_p to disagree about


def rate(tokens: int, seconds: Optional[float]) -> Optional[float]:
    if not seconds or seconds <= 0.0 or tokens <= 0:
        return None
    return tokens / seconds


def summarize_side(runs: Sequence[dict]) -> dict:
    """The median of each timing field, plus the output count every run agreed on."""
    summary = H.summarize(list(runs), ("ttft_ms", "e2e_ms", "tpot_ms", "tps", "decode_tps",
                                       "prompt_ms", "decode_ms"))
    output = [run["output_tokens"] for run in runs]
    summary["output_tokens"] = output[0] if output and len(set(output)) == 1 else None
    summary["output_tokens_per_run"] = output
    summary["identical_outputs"] = len(set(run.get("ids_digest") for run in runs)) <= 1
    summary["first_token_missing"] = sum(1 for run in runs if run.get("ttft_ms") is None)
    return summary


def _digest(ids: Sequence[int]) -> str:
    import hashlib
    return hashlib.sha256(",".join(str(int(x)) for x in ids).encode()).hexdigest()[:16]


def _run_record(ids: Sequence[int], ttft_ms: Optional[float], e2e_ms: float,
                prompt_ms: Optional[float] = None, decode_ms: Optional[float] = None,
                frames: Optional[int] = None, first_from_terminal: bool = False) -> dict:
    count = len(ids)
    tpot = None
    if count > 1 and ttft_ms is not None and e2e_ms > ttft_ms:
        tpot = (e2e_ms - ttft_ms) / (count - 1)
    return {
        "output_tokens": count,
        "frames": frames,
        "first_from_terminal": bool(first_from_terminal),
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "tpot_ms": tpot,
        "tps": rate(count, e2e_ms / 1000.0),
        "decode_tps": rate(count - 1, (e2e_ms - ttft_ms) / 1000.0) if count > 1 and ttft_ms is not None else None,
        "prompt_ms": prompt_ms,
        "decode_ms": decode_ms,
        "ids_digest": _digest(ids),
    }


def reference_measure(ref: R.Reference, ids: Sequence[int], *, max_new: int, runs: int,
                      warmup: int) -> dict:
    session = ref.create()
    lease = ref.acquire(session)
    records = []
    try:
        for index in range(warmup + runs):
            answer = ref.query(session, list(ids), lease=lease, max_new=max_new,
                               sampling=SAMPLING, stream=True)
            if index < warmup:
                continue
            records.append(_run_record(answer["ids"], answer["first_ms"], answer["end_ms"],
                                       frames=answer.get("frames"),
                                       first_from_terminal=answer.get("first_from_terminal", False)))
    finally:
        try:
            ref.close(session, lease)
        except R.ReferenceError:
            pass
    return {"runs": records, "summary": summarize_side(records)}


def engine_measure(engine: ME.MemtEngine, ids: Sequence[int], *, max_new: int, runs: int,
                   warmup: int) -> dict:
    records = []
    for index in range(warmup + runs):
        engine.forget()   # no cached prompt prefix: the number is prompt processing + decode
        timed = engine.query_timed(list(ids), max_new, **SAMPLING)
        if index < warmup:
            continue
        records.append(_run_record(timed["ids"], timed["first_ms"], timed["end_ms"],
                                   timed.get("prompt_ms"), timed.get("decode_ms")))
    return {"runs": records, "summary": summarize_side(records)}


def environment() -> dict:
    out = {"platform": platform.platform(), "machine": platform.machine(),
           "python": platform.python_version()}
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        if done.returncode == 0:
            out["gpus"] = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def compare_sides(reference: dict, engine: dict) -> dict:
    """The engine's speed relative to the reference (>1 means the engine is faster)."""
    out = {}
    for field in ("ttft_ms", "e2e_ms", "tpot_ms"):
        ref_value = reference["summary"].get(field)
        engine_value = engine["summary"].get(field)
        if ref_value and engine_value and engine_value > 0:
            out[field] = ref_value / engine_value
    ref_tps = reference["summary"].get("tps")
    engine_tps = engine["summary"].get("tps")
    if ref_tps and engine_tps:
        out["tps"] = engine_tps / ref_tps
    out["same_output_tokens"] = (reference["summary"].get("output_tokens") is not None
                                 and reference["summary"].get("output_tokens") == engine["summary"].get("output_tokens"))
    # different output counts make the ratio a comparison of different work
    out["comparable"] = out["same_output_tokens"]
    return out


def run_bench(*, prompts: Sequence[tuple[str, Sequence[int]]], max_new: int, runs: int,
              warmup: int, ref=None, engine=None, side: str = "both", outdir: Path = Path("runs/p7"),
              condition: Optional[dict] = None, report_name: str = "bench.json") -> dict:
    """Measure the requested sides and write the report.  `ref`/`engine` are already started."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "condition": dict(condition or {}, max_new=max_new, runs=runs, warmup=warmup,
                          sampling=SAMPLING, side=side,
                          started=time.strftime("%Y-%m-%dT%H:%M:%S%z")),
        "environment": environment(),
        "prompts": [],
        "ok": False,
    }
    notes = report.setdefault("notes", [])
    if side == "both" and (ref is None or engine is None):
        raise ValueError("--side both needs both the reference and the engine")
    for name, ids in prompts:
        row = {"name": name, "prompt_tokens": len(ids), "ids": list(ids), "sides": {}}
        if ref is not None:
            row["sides"]["reference"] = reference_measure(ref, ids, max_new=max_new, runs=runs, warmup=warmup)
        if engine is not None:
            row["sides"]["engine"] = engine_measure(engine, ids, max_new=max_new, runs=runs, warmup=warmup)
        if "reference" in row["sides"] and "engine" in row["sides"]:
            row["engine_vs_reference"] = compare_sides(row["sides"]["reference"], row["sides"]["engine"])
            if not row["engine_vs_reference"]["same_output_tokens"]:
                notes.append("%s: the two sides generated different token counts; the tok/s ratio compares "
                             "different work" % name)
        for label, measured in row["sides"].items():
            summary = measured["summary"]
            if summary.get("first_token_missing"):
                notes.append("%s/%s: %d run(s) had no token at all (max_new or stop)"
                             % (name, label, summary["first_token_missing"]))
            if summary.get("output_tokens") is None:
                notes.append("%s/%s: the runs generated different token counts; the medians are per-run"
                             % (name, label))
            if not summary.get("identical_outputs"):
                notes.append("%s/%s: the runs did not generate identical tokens (greedy should); the "
                             "timings are still medians of what ran" % (name, label))
            if label == "reference" and any(run.get("first_from_terminal") for run in measured["runs"]):
                notes.append("%s/reference: the server put every token in the terminal frame, so its TTFT "
                             "is that frame's arrival (the whole generation), not a first token" % name)
            elif label == "reference" and any(
                    run.get("frames") is not None and run["frames"] < run["output_tokens"] - 1
                    for run in measured["runs"]):
                notes.append("%s/reference: the server batched its streamed frames, so its TTFT is late "
                             "and its TPOT is a lower bound; tok/s and e2e are unaffected" % name)
        expected = runs
        measured_runs = {label: len(item["runs"]) for label, item in row["sides"].items()}
        if any(count != expected for count in measured_runs.values()):
            notes.append("%s: expected %d run(s) per side, got %r" % (name, expected, measured_runs))
            row["ok"] = False
        elif "engine_vs_reference" in row and not row["engine_vs_reference"]["comparable"]:
            row["ok"] = False   # the two sides did different amounts of work
        else:
            row["ok"] = True
        report["prompts"].append(row)
    report["ok"] = all(row.get("ok") for row in report["prompts"]) and bool(report["prompts"])
    path = outdir / report_name
    H.write_json(path, report)
    report["report_path"] = str(path)
    return report


def format_report(report: dict) -> str:
    lines = []
    for key, value in sorted(report.get("condition", {}).items()):
        lines.append("condition %s: %s" % (key, value))
    for gpu in report.get("environment", {}).get("gpus", []):
        lines.append("gpu: %s" % gpu)
    for row in report.get("prompts", []):
        lines.append("")
        lines.append("%s (%d prompt tokens)" % (row["name"], row["prompt_tokens"]))
        lines.append("  %-10s %8s %10s %10s %10s %10s" % ("side", "tokens", "ttft ms", "tpot ms", "tok/s", "e2e ms"))
        for label in ("reference", "engine"):
            measured = row["sides"].get(label)
            if measured is None:
                continue
            summary = measured["summary"]
            def show(value, fmt):
                return "-" if value is None else fmt % value
            lines.append("  %-10s %8s %10s %10s %10s %10s" % (
                label, show(summary.get("output_tokens"), "%d"),
                show(summary.get("ttft_ms"), "%.2f"), show(summary.get("tpot_ms"), "%.3f"),
                show(summary.get("tps"), "%.2f"), show(summary.get("e2e_ms"), "%.2f")))
        ratio = row.get("engine_vs_reference")
        if ratio and not ratio.get("comparable", True):
            lines.append("  engine vs reference: NOT COMPARABLE (different output token counts: "
                         "%s vs %s)" % (row["sides"]["reference"]["summary"].get("output_tokens"),
                                        row["sides"]["engine"]["summary"].get("output_tokens")))
            ratio = None
        if ratio:
            parts = []
            if "ttft_ms" in ratio:
                parts.append("ttft x%.2f" % ratio["ttft_ms"])
            if "tpot_ms" in ratio:
                parts.append("tpot x%.2f" % ratio["tpot_ms"])
            if "tps" in ratio:
                parts.append("tok/s x%.2f" % ratio["tps"])
            lines.append("  engine vs reference: %s%s" % (", ".join(parts),
                                                         "" if ratio["same_output_tokens"]
                                                         else "  (different output counts!)"))
    for note in report.get("notes", []):
        lines.append("note: %s" % note)
    lines.append("")
    lines.append("result: %s%s" % ("MEASURED" if report.get("ok") else "INCOMPLETE",
                                   " (%s)" % report["report_path"] if report.get("report_path") else ""))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=("ref", "engine", "both"), default="both")
    ap.add_argument("--ref-url", default=None)
    ap.add_argument("--ref-api-key", default=None)
    ap.add_argument("--engine", default=None, help="the strata-q36 executable")
    ap.add_argument("--engine-arg", action="append", default=[])
    ap.add_argument("--prompt", action="append", required=True,
                    help="token-id file to measure (repeatable; the file's stem names the row)")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--ready-timeout", type=float, default=30.0)
    ap.add_argument("--outdir", default="runs/p7")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.runs < 1 or args.warmup < 0 or args.max_new < 1:
        print("memt_bench: --runs >= 1, --warmup >= 0, --max-new >= 1", file=sys.stderr)
        return 2
    if args.side in ("ref", "both") and not args.ref_url:
        print("memt_bench: --side %s needs --ref-url" % args.side, file=sys.stderr)
        return 2
    if args.side in ("engine", "both") and not args.engine:
        print("memt_bench: --side %s needs --engine" % args.side, file=sys.stderr)
        return 2
    try:
        prompts = [(Path(path).stem, H.read_ids(path)) for path in args.prompt]
    except H.HarnessError as exc:
        print("memt_bench: %s" % exc, file=sys.stderr)
        return 2
    ref = engine = None
    try:
        if args.side in ("ref", "both"):
            ref = R.Reference(args.ref_url, api_key=args.ref_api_key)
            ref.wait_ready(timeout=args.ready_timeout)
        if args.side in ("engine", "both"):
            engine = ME.MemtEngine.spawn([args.engine, *args.engine_arg], ready_timeout=args.ready_timeout)
        condition = {"reference": args.ref_url, "engine_argv": [args.engine, *args.engine_arg]
                     if args.engine else None}
        report = run_bench(prompts=prompts, max_new=args.max_new, runs=args.runs, warmup=args.warmup,
                           ref=ref if args.side in ("ref", "both") else None,
                           engine=engine if args.side in ("engine", "both") else None,
                           side=args.side, outdir=Path(args.outdir), condition=condition)
    except (ME.MemtEngineError, R.ReferenceError, OSError, ValueError) as exc:
        print("memt_bench: %s" % exc, file=sys.stderr)
        return 2
    finally:
        if engine is not None:
            engine.close()
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else format_report(report))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
