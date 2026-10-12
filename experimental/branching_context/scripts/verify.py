#!/usr/bin/env python3
"""Cumulative gate runner. All missing real-backend gates fail closed.

No backend is included in this handoff. G0 is runnable; G1–G5 are obligations
Codex must implement and certify with actual data and an independent oracle.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
GATES = ["G0", "G1", "G2", "G3", "G4", "G5"]
sys.path.insert(0, str(ROOT / "src"))

PROFILES = {
    "G1": ("model_weights_sha256", "model_contract_id", "backend_revision", "oracle_revision",
           "oracle_fresh_no_cache_path", "execution_profile_id", "token_fixtures_sha256",
           "atol", "rtol", "max_observed_logit_error", "hardware"),
    "G2": ("filesystem", "checkpoint_manifest_sha256", "failure_points_executed",
           "model_restore_check", "process_restart_results"),
    "G3": ("prefix_tokens_prefilled", "shared_prefix_pages", "private_tail_pages",
           "kv_unique_bytes", "cold_prefill_comparison", "profiling_method"),
    "G4": ("evaluator_identity", "three_future_fixture_sha256", "adoption_receipt",
           "replayed_winner_sha256"),
    "G5": ("supported_profile_matrix", "randomized_sequences", "soak_duration_seconds",
           "mutation_kills", "access_control_results", "rollback_results"),
}


def suite_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from suite_tests(item)
        else:
            yield item


def load_g0():
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"), pattern="test_*.py")
    toy = ROOT / "reference" / "test_context_time_travel_toy.py"
    spec = importlib.util.spec_from_file_location("ctt_toy_reference", toy)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite.addTests(loader.loadTestsFromModule(module))
    return suite


def load_gate(gate):
    path = ROOT / "tests" / "gates" / gate
    if not path.is_dir() or not any(path.glob("test_*.py")):
        return None
    return unittest.defaultTestLoader.discover(str(path), pattern="test_*.py")


def validate_profile(gate):
    if gate == "G0":
        return None
    location = ROOT / "evidence" / "profiles" / (gate + ".json")
    if not location.is_file():
        return f"Missing evidence profile {location.relative_to(ROOT)}"
    try:
        values = json.loads(location.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return f"Invalid profile JSON: {exc}"
    for key in PROFILES[gate]:
        if key not in values or values[key] is None or values[key] == "" or values[key] == "TODO":
            return f"Incomplete evidence profile {gate}: {key}"
    if gate == "G1":
        for key in ("model_weights_sha256", "token_fixtures_sha256"):
            if not re.fullmatch("[0-9a-f]{64}", str(values[key])):
                return f"Invalid sha256 in profile: {key}"
        for key in ("atol", "rtol", "max_observed_logit_error"):
            if type(values[key]) not in (int, float) or values[key] < 0:
                return f"Invalid numeric profile field: {key}"
    if gate == "G5":
        try:
            num_sequences = int(values["randomized_sequences"])
        except (ValueError, TypeError):
            return "G5 randomized_sequences must be an integer"
        if num_sequences < 1000:
            return "G5 must cover >= 1000 independently reproducible randomized sequences"
    return None


def source_sha():
    h = hashlib.sha256()
    for directory in ("src", "spec", "tests", "reference", "docs", "scripts", "plans"):
        for f in sorted((ROOT / directory).rglob("*")):
            if f.is_file() and f.suffix in (".py", ".md", ".json"):
                rel = str(f.relative_to(ROOT)).encode("utf-8")
                h.update(len(rel).to_bytes(4, "big") + rel)
                h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def run_gate(gate, registry, *, quiet=False):
    cases = [c for c in registry["cases"] if c["gate"] == gate]
    suite = load_g0() if gate == "G0" else load_gate(gate)
    names = set(t._testMethodName for t in suite_tests(suite)) if suite else set()
    missing = [c["test"] for c in cases if c["test"] not in names]
    reasons = []
    if gate != "G0" and not cases:
        reasons.append("No acceptance cases registered")
    if suite is None or not names:
        reasons.append("No executable tests discovered")
    if missing:
        reasons.append(f"Missing {len(missing)} required test methods: " + ", ".join(missing[:5]) + ("..." if len(missing)>5 else ""))
    if reasons:
        return {"gate": gate, "status": "BLOCKED", "reasons": reasons,
                "required_cases": len(cases), "found_tests": len(names)}
    output = open(os.devnull, "w") if quiet else sys.stdout
    try:
        result = unittest.TextTestRunner(stream=output, verbosity=1 if quiet else 2).run(suite)
    finally:
        if quiet:
            output.close()
    counts = {"test_count": result.testsRun, "failures": len(result.failures),
              "errors": len(result.errors), "skipped": len(result.skipped),
              "expected_failures": len(result.expectedFailures),
              "unexpected_successes": len(result.unexpectedSuccesses)}
    good = result.testsRun > 0 and result.wasSuccessful() and not result.skipped and not result.expectedFailures
    profile_issue = validate_profile(gate) if good else None
    outcome = {"gate": gate, "status": "FAIL" if not good else "BLOCKED" if profile_issue else "PASS",
               "required_cases": len(cases), "found_tests": len(names), **counts,
               "failure_ids": [str(t.id()) for t, _ in result.failures + result.errors],
               "skipped_ids": [str(t.id()) for t, _ in result.skipped]}
    if profile_issue:
        outcome["reasons"] = [profile_issue]
    return outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="G0", choices=GATES + ["ALL"])
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-write", action="store_true", help="Suppress local evidence file")
    args = ap.parse_args()
    registry = json.loads((ROOT / "docs" / "acceptance.json").read_text())
    locked = json.loads((ROOT / "docs" / "locked-references.json").read_text())
    for relative_path, expected_sha in locked["sha256"].items():
        actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        if actual != expected_sha:
            raise SystemExit(f"ERROR: locked reference changed: {relative_path}; require versioned spec decision")
    if registry["case_count"] != len(registry["cases"]):
        raise SystemExit("ERROR: acceptance registry count inconsistent")
    selected = GATES if args.gate == "ALL" else GATES[:GATES.index(args.gate)+1]
    report = {"schema": 1, "tool": "scripts/verify.py",
              "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "source_tree_sha256": source_sha(),
              "python": sys.version, "platform": platform.platform(),
              "requested": args.gate, "gates": []}
    success = True
    for gate in selected:
        result = run_gate(gate, registry, quiet=args.quiet)
        report["gates"].append(result)
        print(f'{gate}: {result["status"]}: {result.get("test_count", 0)} tests; '
              f'{result.get("required_cases", 0)} required real-backend cases', flush=True)
        for reason in result.get("reasons", []):
            print("  - " + reason, flush=True)
        if result["status"] != "PASS":
            success = False
            print(f"Stopped at {gate}: no later gate can be certified", flush=True)
            break
    report["status"] = "PASS" if success else report["gates"][-1]["status"]
    if not args.no_write:
        directory = ROOT / "evidence" / "runs"
        directory.mkdir(parents=True, exist_ok=True)
        # Don't overwrite evidence from another run.
        filename = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + args.gate + ".json"
        destination = directory / filename
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Evidence: {destination.relative_to(ROOT)}", flush=True)
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
