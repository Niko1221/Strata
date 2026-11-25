"""tools/memt_parity.py - P6: does the engine's memory match memt_sglang?

Both implementations export `memt-memory-v1`, so parity is a procedure rather
than a model:

  1. the reference server creates a session, commits the same tokens from an
     empty memory, and exports its snapshot;
  2. this fork commits the same tokens from an empty memory and exports too;
  3. the two files are compared layer by layer (`memt_compare.py`: cosine,
     max-abs, shape, dtype, non-finite);
  4. both read the same prompts; the generated ids are compared;
  5. the engine's snapshot is copied into the reference's `--state-directory` and
     imported there, so the HF side really loads a C++-written file (and the
     prompts are compared again through that imported memory).

Steps 1-4 need a GPU host with both servers; step 5 needs the reference's state
directory to be writable by this tool.  Nothing here fabricates a number: a
missing server is an error, not a pass.

    python tools/memt_parity.py --ref-url http://127.0.0.1:30000 \\
        --engine /path/strata-q36 --engine-arg=--serve --engine-arg=--model --engine-arg nano.memt.gguf \\
        --ref-state-dir /data/memt-state --commit evidence.ids --prompt q1.ids \\
        --outdir runs/p6 --max-new 32

Exit status: 0 all comparisons pass, 1 a comparison failed, 2 a setup/transport
error (nothing was compared).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import memt_compare as C  # noqa: E402
import memt_engine as ME  # noqa: E402
import memt_harness as H  # noqa: E402
import memt_ref as R  # noqa: E402
import memt_snapshot as S  # noqa: E402

__all__ = ["token_verdict", "run_parity", "main"]


def token_verdict(reference: Sequence[int], engine: Sequence[int]) -> dict:
    """How two generations compare: identical, or the length of the common prefix."""
    same = list(reference) == list(engine)
    return {
        "identical": same,
        "reference_tokens": len(reference),
        "engine_tokens": len(engine),
        "identical_prefix": H.prefix_match(reference, engine),
    }


def _snapshot_compare(ref_path: Path, engine_path: Path, identity: str, min_cosine: float,
                      max_rel_l2: float = 0.25) -> dict:
    ref_layers = S.load(ref_path, identity)
    engine_layers = S.load(engine_path, identity)
    report = C.compare(ref_layers, engine_layers, min_cosine=min_cosine, max_rel_l2=max_rel_l2)
    report["layers_present"] = sorted(ref_layers)
    return report


def run_parity(ref, engine, *, commit_ids: Sequence[int], prompts: Sequence[tuple[str, Sequence[int]]],
               outdir: Path, identity: Optional[str] = None, ref_state_dir: Optional[Path] = None,
               max_new: int = 32, min_cosine: float = 0.999, max_rel_l2: float = 0.25,
               allow_token_drift: bool = False, roundtrip: bool = True,
               engine_model_dir: Optional[Path] = None, report_name: str = "parity.json") -> dict:
    """The whole P6 procedure.  `ref` is a memt_ref.Reference, `engine` a MemtEngine.

    Both sides generate greedily (`H.GREEDY_SAMPLING`): the memory state should
    agree regardless of sampling, but a token comparison only means something if
    the two are asked for the same distribution — the reference's default
    temperature is not 0.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    names = [name for name, _ in prompts]
    if len(set(names)) != len(names):
        raise ValueError("prompt names must be unique: %r" % names)
    caps = ref.capabilities()
    identity = identity or caps["snapshot_identity"]
    report = {
        "identity": identity,
        "reference_capabilities": {key: caps.get(key) for key in
                                   ("protocol", "epoch", "chunk_tokens", "max_prefill_tokens",
                                    "snapshot_identity", "memory_encoding")},
        "commit_tokens": len(commit_ids),
        "prompts": names,
        "max_new": max_new,
        "sampling": dict(H.GREEDY_SAMPLING),
        "min_cosine": min_cosine,
        "max_rel_l2": max_rel_l2,
        "comparisons": {},
        "ok": False,
    }
    notes = report.setdefault("notes", [])
    if engine_model_dir is not None:
        # the engine only compares the identity string it is given; recomputing it
        # from the checkpoint catches "the GGUF is not the model the server serves"
        computed = ME.identity_from_model_dir(engine_model_dir)
        report["engine_identity"] = {"checkpoint": str(engine_model_dir), "computed": computed,
                                     "matches": computed == identity}
        if computed != identity:
            raise ValueError("the engine checkpoint's snapshot identity is %s, the reference's is %s: "
                             "these are different memories" % (computed, identity))
    if len(commit_ids) >= int(caps["chunk_tokens"]):
        notes.append("the commit is %d tokens: at or over chunk_tokens=%d, so the reference splits the "
                     "write into more than one event while the engine tiles at 256; the state should "
                     "still agree, but a short commit is the cleaner P6 case"
                     % (len(commit_ids), caps["chunk_tokens"]))

    sessions = []

    def close_sessions():
        for session, lease in sessions:
            try:
                ref.close(session, lease)
            except Exception as exc:   # keep closing the rest; the report records it
                notes.append("close %s: %s" % (session, exc))

    try:
        # ---- the reference: commit from empty memory, export, query
        session = ref.create()
        lease = ref.acquire(session)
        sessions.append((session, lease))
        ref.begin(session, lease, "t1")
        committed = ref.commit(session, list(commit_ids), lease=lease, transaction="t1")
        ref.publish(session, lease, "t1")
        ref_name = "p6-ref-%s.safetensors" % stamp
        exported = ref.export(session, lease, ref_name)
        report["reference"] = {"commit_ms": committed["ms"], "snapshot": exported.get("file"),
                               "has_memory": exported.get("has_memory")}
        ref_runs = []
        for name, ids in prompts:
            answer = ref.query(session, list(ids), lease=lease, max_new=max_new,
                               sampling=H.GREEDY_SAMPLING, stream=True)
            ref_runs.append({"name": name, "ids": answer["ids"], "text": answer["text"],
                             "finish": answer["finish"], "first_ms": answer["first_ms"],
                             "end_ms": answer["end_ms"]})
        report["reference"]["queries"] = ref_runs

        # ---- the engine: same commit from empty memory, export, the same prompts
        engine.reset()
        engine.forget()
        done = engine.commit(list(commit_ids))
        engine_path = outdir / ("p6-engine-%s.safetensors" % stamp)
        engine.snapshot(engine_path, identity)
        report["engine"] = {"commit": done, "snapshot": engine_path.name}
        engine_runs = []
        for name, ids in prompts:
            engine.forget()
            timed = engine.query_timed(list(ids), max_new, **H.GREEDY_SAMPLING)
            engine_runs.append({"name": name, "ids": timed["ids"], "finish": timed["finish"],
                                "first_ms": timed["first_ms"], "end_ms": timed["end_ms"],
                                "prompt_ms": timed["prompt_ms"], "decode_ms": timed["decode_ms"]})
        report["engine"]["queries"] = engine_runs

        # ---- the facts P6 is about
        if ref_state_dir is None:
            report["comparisons"]["snapshot"] = {
                "ok": False,
                "skipped": "no --ref-state-dir: the reference's exported file cannot be read, so the "
                           "memory state was not compared",
            }
        else:
            ref_path = Path(ref_state_dir) / ref_name
            if not ref_path.exists():
                raise R.ReferenceError("the reference exported %r but %s does not exist (is "
                                       "--ref-state-dir the server's --state-directory?)"
                                       % (ref_name, ref_path))
            snapshot = _snapshot_compare(ref_path, engine_path, identity, min_cosine, max_rel_l2)
            snapshot.update(reference_file=str(ref_path), engine_file=str(engine_path))
            report["comparisons"]["snapshot"] = snapshot

        ref_by_name = {run["name"]: run for run in ref_runs}
        engine_by_name = {run["name"]: run for run in engine_runs}
        token_rows = []
        for name, ids in prompts:
            ref_run = ref_by_name[name]
            engine_run = engine_by_name[name]
            verdict = token_verdict(ref_run["ids"], engine_run["ids"])
            verdict.update(name=name, ok=verdict["identical"] or allow_token_drift)
            token_rows.append(verdict)
        report["comparisons"]["tokens"] = {
            "ok": all(row["ok"] for row in token_rows), "allow_drift": allow_token_drift, "rows": token_rows,
        }

        # ---- the HF side really loads the C++ file
        if ref_state_dir is None:
            report["comparisons"]["roundtrip"] = {
                "ok": False, "skipped": "no --ref-state-dir: the engine's snapshot was not imported by "
                                        "the reference"}
        elif not roundtrip:
            report["comparisons"]["roundtrip"] = {"ok": False, "skipped": "--no-roundtrip was given"}
        else:
            landed = Path(ref_state_dir) / engine_path.name
            shutil.copyfile(engine_path, landed)
            session2 = ref.create()
            lease2 = ref.acquire(session2)
            sessions.append((session2, lease2))
            ref.begin(session2, lease2, "t2")
            imported = ref.import_(session2, lease2, "t2", engine_path.name)
            if imported.get("format") != "memt-memory-v1" or imported.get("has_memory") is not True:
                raise R.ReferenceError("the reference did not take %r as a memory snapshot: %r"
                                       % (engine_path.name, imported))
            ref.publish(session2, lease2, "t2")
            roundtrip_rows = []
            for name, ids in prompts:
                answer = ref.query(session2, list(ids), lease=lease2, max_new=max_new,
                                   sampling=H.GREEDY_SAMPLING, stream=False)
                engine_run = engine_by_name[name]
                verdict = token_verdict(engine_run["ids"], answer["ids"])
                verdict.update(name=name, ok=verdict["identical"] or allow_token_drift)
                roundtrip_rows.append(verdict)
            report["comparisons"]["roundtrip"] = {
                "ok": all(row["ok"] for row in roundtrip_rows),
                "file": engine_path.name,
                "rows": roundtrip_rows,
            }
            # the tokens above are one signal; the stronger one is the memory itself:
            # export what the reference now holds and compare it with our file
            re_name = "p6-roundtrip-%s.safetensors" % stamp
            ref.export(session2, lease2, re_name)
            re_path = Path(ref_state_dir) / re_name
            if not re_path.exists():
                raise R.ReferenceError("the reference exported %r but %s does not exist"
                                       % (re_name, re_path))
            report["comparisons"]["roundtrip_state"] = _snapshot_compare(
                engine_path, re_path, identity, min_cosine, max_rel_l2)

        report["ok"] = all(item.get("ok") for item in report["comparisons"].values())
    except BaseException as exc:
        # an error is not a pass: the file keeps what was measured, and says why it stopped
        report["error"] = "%s: %s" % (type(exc).__name__, exc)
        raise
    finally:
        close_sessions()
        report["ok"] = ("error" not in report and bool(report.get("comparisons"))
                        and bool(report.get("prompts"))
                        and all(item.get("ok") for item in report["comparisons"].values()))
        report_path = outdir / report_name
        try:
            H.write_json(report_path, report)
            report["report_path"] = str(report_path)
        except OSError as exc:
            report.setdefault("notes", []).append("cannot write %s: %s" % (report_path, exc))
    return report


def format_report(report: dict) -> str:
    lines = ["identity: %s" % report.get("identity")]
    if report.get("error"):
        lines.append("error: %s" % report["error"])
    lines.append("commit: %d tokens, %d prompt(s) x %d new tokens"
                 % (report.get("commit_tokens", 0), len(report.get("prompts", [])),
                    report.get("max_new", 0)))
    for note in report.get("notes", []):
        lines.append("note: %s" % note)
    comparisons = report.get("comparisons", {})
    snapshot = comparisons.get("snapshot", {})
    if snapshot.get("skipped"):
        lines.append("snapshot: NOT COMPARED (%s)" % snapshot["skipped"])
    elif snapshot:
        lines.append("snapshot: %s (min cosine %.6f, max rel_l2 %.3f, %d layer(s))"
                     % ("PASS" if snapshot.get("ok") else "FAIL", snapshot.get("min_cosine", 0.0),
                        snapshot.get("max_rel_l2", 0.0), len(snapshot.get("layers", {}))))
        for layer, item in snapshot.get("layers", {}).items():
            if not item.get("same_shape"):
                lines.append("  layer %-3s shape %r vs %r" % (layer, item.get("ref_shape"), item.get("got_shape")))
                continue
            lines.append("  layer %-3s cosine %.9f  rel_l2 %.3e  max_abs %.3e  identical %s%s"
                         % (layer, item["cosine"], item["rel_l2"], item["max_abs"],
                            "yes" if item["bit_identical"] else "no",
                            "" if item.get("cosine", 0.0) >= snapshot.get("min_cosine", 0.999)
                            and item.get("rel_l2", 1e9) <= snapshot.get("max_rel_l2", 0.25) else "  <-- below"))
    tokens = comparisons.get("tokens", {})
    if tokens:
        lines.append("tokens: %s" % ("PASS" if tokens.get("ok") else "FAIL"))
        for row in tokens.get("rows", []):
            lines.append("  %-20s identical %s  prefix %d/%d"
                         % (row["name"], "yes" if row["identical"] else "no",
                            row["identical_prefix"], row["reference_tokens"]))
    roundtrip = comparisons.get("roundtrip", {})
    if roundtrip.get("skipped"):
        lines.append("roundtrip: NOT RUN (%s)" % roundtrip["skipped"])
    elif roundtrip:
        lines.append("roundtrip (HF loads the engine's file): %s" % ("PASS" if roundtrip.get("ok") else "FAIL"))
    roundtrip_state = comparisons.get("roundtrip_state")
    if roundtrip_state:
        lines.append("roundtrip state (HF's memory re-read by us): %s"
                     % ("PASS" if roundtrip_state.get("ok") else "FAIL"))
    lines.append("result: %s" % ("PASS" if report.get("ok") else "FAIL"))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-url", required=True, help="the memt_sglang server, e.g. http://127.0.0.1:30000")
    ap.add_argument("--ref-api-key", default=None)
    ap.add_argument("--ref-state-dir", default=None,
                    help="the reference's --state-directory (readable/writable here)")
    ap.add_argument("--engine", required=True, help="the strata-q36 executable")
    ap.add_argument("--engine-arg", action="append", default=[], help="one argument for the engine (repeatable)")
    ap.add_argument("--commit", required=True, help="token-id file to write into memory")
    ap.add_argument("--prompt", action="append", required=True,
                    help="token-id file to query (repeatable; the file's stem names the row)")
    ap.add_argument("--identity", default=None,
                    help="override the reference's snapshot identity (must match both sides)")
    ap.add_argument("--engine-model-dir", default=None,
                    help="the HF checkpoint the engine's GGUF was converted from: its identity must "
                         "equal the reference's (omit when only the GGUF is on this host)")
    ap.add_argument("--outdir", default="runs/p6")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--ready-timeout", type=float, default=30.0,
                    help="seconds to wait for the already-started reference server")
    ap.add_argument("--min-cosine", type=float, default=0.999)
    ap.add_argument("--max-rel-l2", type=float, default=0.25,
                    help="gross-error ceiling on ||ref-engine||/||ref|| per layer")
    ap.add_argument("--allow-token-drift", action="store_true",
                    help="report a token mismatch as a pass (a quantisation tolerance was agreed)")
    ap.add_argument("--no-roundtrip", action="store_true", help="skip importing the engine's file in HF")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.max_new < 1:
        print("memt_parity: --max-new must be positive", file=sys.stderr)
        return 2
    try:
        commit_ids = H.read_ids(args.commit)
        prompts = []
        for path in args.prompt:
            prompts.append((Path(path).stem, H.read_ids(path)))
    except H.HarnessError as exc:
        print("memt_parity: %s" % exc, file=sys.stderr)
        return 2
    ref = R.Reference(args.ref_url, api_key=args.ref_api_key)
    try:
        ref.wait_ready(timeout=args.ready_timeout)
    except R.ReferenceError as exc:
        print("memt_parity: the reference server is not up: %s" % exc, file=sys.stderr)
        return 2
    engine = None
    try:
        engine = ME.MemtEngine.spawn([args.engine, *args.engine_arg])
        report = run_parity(ref, engine, commit_ids=commit_ids, prompts=prompts,
                            outdir=Path(args.outdir), identity=args.identity,
                            ref_state_dir=Path(args.ref_state_dir) if args.ref_state_dir else None,
                            max_new=args.max_new, min_cosine=args.min_cosine,
                            max_rel_l2=args.max_rel_l2,
                            allow_token_drift=args.allow_token_drift, roundtrip=not args.no_roundtrip,
                            engine_model_dir=Path(args.engine_model_dir) if args.engine_model_dir else None)
    except (ME.MemtEngineError, R.ReferenceError, S.SnapshotError, H.HarnessError,
            ValueError, OSError) as exc:
        print("memt_parity: %s" % exc, file=sys.stderr)
        return 2
    finally:
        if engine is not None:
            engine.close()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_report(report))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
