"""Red/green controls for the two independent replica ownership bugs.

Run from the repository root:
  python tools/replay_replica_lifetime.py --output /tmp/replica-controls

Uses the real repository StrataEngine reader/pump, but synthetic CPU producers;
never loads a model, starts a server, or requests a GPU. Output must be new.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def close_only(source):
    """Add child close under the lock, deliberately leave shared last unchanged."""
    marker = "        held = False\n        nonfinite = False\n"
    assert source.count(marker) == 1, "baseline generator has changed"
    source = source.replace(marker, "        held = False\n        gen = None\n        nonfinite = False\n")
    start = source.index("        finally:\n            if held:", source.index("    def generate("))
    old = source[start:]
    assert old.startswith("        finally:\n            if held:")
    cleanup = old.removeprefix("        finally:\n")
    source = source[:start] + (
        "        finally:\n"
        "            try:\n"
        "                if gen is not None:\n"
        "                    gen.close()\n"
        "            finally:\n"
        + "".join("    " + line if line.strip() else line for line in cleanup.splitlines(True))
    )
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", default="3d1fb48090e8f2cefc22d7b5845fb7f5b71f33fa")
    p.add_argument("--baseline-file", type=Path,
                   help="optional captured baseline replicas.py for a source archive without Git history")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = args.baseline_file.read_text(encoding="utf-8") if args.baseline_file else subprocess.check_output(
        ["git", "show", args.baseline + ":serve/replicas.py"], cwd=ROOT
    ).decode("utf-8")
    from serve import test_replica_lifetime as contracts

    snapshots = {
        "original": original,
        "close_only": close_only(original),
        "fixed": (ROOT / "serve/replicas.py").read_text(encoding="utf-8"),
    }
    rows = []
    for name, source in snapshots.items():
        path = args.output / (name + ".py")
        path.write_text(source, encoding="utf-8", newline="\n")
        spec = importlib.util.spec_from_file_location("replica_control_" + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        contracts.ReplicaEngine = module.ReplicaEngine
        log = io.StringIO()
        suite = unittest.defaultTestLoader.loadTestsFromModule(contracts)
        with contextlib.redirect_stderr(log):
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        (args.output / (name + ".log")).write_text(log.getvalue(), encoding="utf-8")
        row = {
            "name": name,
            "tests": result.testsRun,
            "passed": result.testsRun - len(result.failures) - len(result.errors),
            "failures": [test.id() for test, _ in result.failures],
            "errors": [test.id() for test, _ in result.errors],
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    # The multi-caller statistics race may fail as well as the two forced statistics cases.
    # Keep every fixed contract mandatory; accept only these known statistics failures in close-only.
    stats = {
        "serve.test_replica_lifetime.ReplicaContracts.test_done_statistics_remain_request_owned_after_unlock",
        "serve.test_replica_lifetime.RealPipeContracts.test_real_pipe_queued_successor_keeps_output_and_statistics",
    }
    extra = "serve.test_replica_lifetime.ReplicaContracts.test_four_callers_many_turns_no_cross_output_or_stats"
    closed = set(rows[1]["failures"])
    ok = (all(row["tests"] == 18 and not row["errors"] for row in rows)
          and rows[0]["passed"] == 9
          and stats <= closed <= stats | {extra}
          and rows[2]["passed"] == 18)
    report = {"status": "PASS" if ok else "FAIL", "baseline": args.baseline, "rows": rows,
              "scope": "Original/close-only are intentional red controls. Real reader/pump, CPU fixtures, no GPU."}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
