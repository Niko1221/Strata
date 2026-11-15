"""Fail-closed comparison of complete bounded-run receipts, never qualification.

Only engine_toolchain and kv_storage are controlled axes. The latter permits
engine/toolchain identity and kv_k/kv_v changes, never model/router/context/MTP.
Missing native consumption observations block comparison. Synthetic receipt
tests exercise this contract; they are not real model A/A evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

try:
    from .q36_bounded_run import (
        DIMENSIONS,
        SETTINGS,
        integer,
        read_json,
        sha,
        token_hash,
    )
except ImportError:
    from q36_bounded_run import (
        DIMENSIONS,
        SETTINGS,
        integer,
        read_json,
        sha,
        token_hash,
    )


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _validate(run: dict) -> list[str]:
    errors = []
    if not isinstance(run, dict):
        return ["receipt must be an object"]
    if run.get("schema_version") != 1 or run.get("status") != "PASS":
        errors.append("run did not pass schema/protocol execution")
    if not isinstance(run.get("run_id"), str) or not run["run_id"]:
        errors.append("missing run identity")
    instrumentation = run.get("instrumentation")
    if (
        not isinstance(instrumentation, dict)
        or set(instrumentation)
        != {"mode", "hotspot_sync", "phase_counters", "startup_detail", "kernel_verify"}
        or instrumentation.get("mode")
        not in ("off", "host_counters", "diagnostic_sync")
        or any(
            type(instrumentation.get(k)) is not bool
            for k in (
                "hotspot_sync",
                "phase_counters",
                "startup_detail",
                "kernel_verify",
            )
        )
    ):
        errors.append("missing/invalid observed instrumentation")
    elif instrumentation["mode"] != (
        "diagnostic_sync"
        if instrumentation["hotspot_sync"] or instrumentation["kernel_verify"]
        else "host_counters" if instrumentation["phase_counters"] else "off"
    ):
        errors.append("inconsistent observed instrumentation")
    identities = run.get("identities", {})
    if (
        not isinstance(identities, dict)
        or any(
            not sha(identities.get(key))
            for key in (
                "engine_sha256",
                "model_sha256",
                "prompt_fixture_sha256",
                "toolchain_sha256",
                "harness_sha256",
                "guard_sha256",
                "resource_guard_sha256",
                "worker_interpreter_sha256",
            )
        )
        or ("pack_sha256" in identities and not sha(identities["pack_sha256"]))
    ):
        errors.append("missing artifact identity")
    argv = run.get("execution_argv")
    if not isinstance(argv, list) or any(not isinstance(arg, str) for arg in argv):
        errors.append("missing execution argument vector")
    experiment = run.get("experiment")
    if not isinstance(experiment, dict) or set(experiment) != {
        "dimensions",
        "max_relative_regression",
    }:
        errors.append("missing frozen experiment")
    else:
        dims = experiment["dimensions"]
        if (
            not isinstance(dims, list)
            or any(not isinstance(d, str) or d not in DIMENSIONS for d in dims)
            or len(set(dims)) != len(dims)
        ):
            errors.append("invalid comparison dimensions")
        threshold = experiment["max_relative_regression"]
        if not _finite(threshold) or threshold > 1:
            errors.append("invalid frozen threshold")
    requested = run.get("requested_settings")
    effective = run.get("effective_settings")
    if not isinstance(requested, dict) or set(requested) != set(SETTINGS):
        errors.append("missing requested settings")
    elif (
        any(
            not integer(requested[k], 1, 262144)
            for k in ("context", "prefill", "router_top_k")
        )
        or not integer(requested["parallel"], 1, 1)
        or not integer(requested["mtp"], 0, 64)
        or any(
            requested[k] not in ("f32", "f16", "q8_0", "nvfp4")
            for k in ("kv_k", "kv_v")
        )
        or not isinstance(requested["expert_quantization"], str)
        or not requested["expert_quantization"]
    ):
        errors.append("invalid requested settings")
    if not isinstance(effective, dict) or any(
        effective.get(k) is None for k in SETTINGS
    ):
        errors.append("missing effective observations")
    elif effective != requested or any(
        type(effective.get(k)) is not type(requested.get(k)) for k in SETTINGS
    ):
        errors.append("requested/effective settings mismatch")
    generation = run.get("startup_generation")
    if not isinstance(generation, str) or not generation:
        errors.append("missing startup generation")
    guard = run.get("guard")
    if (
        not isinstance(guard, dict)
        or guard.get("contained_before_resume") is not True
        or guard.get("termination_confirmed") is not True
        or guard.get("exit_code") != 0
        or run.get("child_exit_code") != 0
    ):
        errors.append("missing successful ownership/exit evidence")
    cases = run.get("cases")
    if not isinstance(cases, list) or not cases:
        return errors + ["missing cases"]
    expected = run.get("expected_case_ids")
    if (
        not isinstance(expected, list)
        or not expected
        or expected
        != [c.get("prompt_id") if isinstance(c, dict) else None for c in cases]
    ):
        errors.append("fixture expected case set/order mismatch")
    ids, requests = set(), set()
    for case in cases:
        if not isinstance(case, dict):
            errors.append("malformed case")
            continue
        ident, request = case.get("prompt_id"), case.get("request_id")
        if not isinstance(ident, str) or not ident or ident in ids:
            errors.append("missing/duplicate prompt ID")
        else:
            ids.add(ident)
        if not isinstance(request, str) or not request or request in requests:
            errors.append("missing/duplicate request ID")
        else:
            requests.add(request)
        if case.get("startup_generation") != generation:
            errors.append("case startup generation binding mismatch")
        if case.get("status") != "PASS":
            errors.append("case not completed")
        tokens = case.get("tokens")
        limit = case.get("requested_max_tokens")
        if not integer(limit, 1, 4096) or case.get("effective_max_tokens") != limit:
            errors.append("requested/effective generation limit mismatch")
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(not integer(t, 0, 2**31 - 1) for t in tokens)
        ):
            errors.append("missing/invalid committed tokens")
        elif integer(limit, 1, 4096) and len(tokens) > limit:
            errors.append("generation limit exceeded")
        stamps = case.get("token_times_s")
        if (
            not isinstance(stamps, list)
            or not isinstance(tokens, list)
            or len(stamps) != len(tokens)
            or not stamps
            or any(not _finite(s) for s in stamps)
            or any(b < a for a, b in zip(stamps, stamps[1:]))
            or case.get("ttft_s") != stamps[0]
            or not _finite(case.get("wall_s"))
            or case["wall_s"] < stamps[-1]
        ):
            errors.append("invalid monotonic token timing")
        consumed = case.get("consumed_input")
        if not isinstance(consumed, dict):
            errors.append("missing consumed input observations")
            continue
        count, submitted = case.get("submitted_input_count"), case.get(
            "submitted_input_sha256"
        )
        input_ids = consumed.get("ids")
        if (
            not integer(count, 1, 131072)
            or not sha(submitted)
            or not isinstance(input_ids, list)
            or len(input_ids) != count
            or any(not integer(i, 0, 2**31 - 1) for i in input_ids)
            or consumed.get("count") != count
            or consumed.get("sha256") != submitted
            or token_hash(input_ids) != submitted
            or consumed.get("final_position") != count
            or consumed.get("initial_position") != 0
            or consumed.get("request_id") != request
            or consumed.get("startup_generation") != generation
            or consumed.get("truncated") is not False
        ):
            errors.append("consumed/submitted input or binding mismatch")
        reservation, mtp = case.get("output_reservation"), case.get("mtp_reservation")
        capacity = requested.get("context") if isinstance(requested, dict) else None
        if (
            not integer(reservation, 1, 4096)
            or not integer(mtp, 0, 64)
            or not integer(capacity, 1, 262144)
            or not integer(count, 1, 131072)
            or not integer(limit, 1, 4096)
            or reservation < limit
            or count + reservation + mtp > capacity
        ):
            errors.append("input/output/MTP reservation exceeds capacity")
    return errors


def validate_pair(a: dict, b: dict) -> list[str]:
    """Return every comparability failure; unknown values never compare equal."""
    errors = ["baseline: " + reason for reason in _validate(a)]
    errors += ["candidate: " + reason for reason in _validate(b)]
    if errors:
        return errors
    if a["run_id"] == b["run_id"]:
        errors.append("independent run identities required")
    if a["experiment"] != b["experiment"]:
        errors.append("frozen experiments differ")
    # Receipts without instrumentation remain readable but cannot establish
    # comparability. A sync-enabled diagnostic run never supplies speed evidence.
    if a.get("instrumentation") != b.get("instrumentation"):
        errors.append("instrumentation differs or is unobserved in one run")
    for run in (a, b):
        if (run.get("instrumentation") or {}).get("mode") == "diagnostic_sync":
            errors.append("sync-enabled hotspot runs are diagnostic only")
    dims = set(a["experiment"]["dimensions"])
    allowed_identity = {"engine_sha256", "toolchain_sha256"} if dims else set()
    for key in set(a["identities"]) | set(b["identities"]):
        if key not in allowed_identity and a["identities"].get(key) != b[
            "identities"
        ].get(key):
            errors.append("artifact differs: " + key)
    for key in SETTINGS:
        if "kv_storage" in dims and key in ("kv_k", "kv_v"):
            continue
        if a["effective_settings"][key] != b["effective_settings"][key]:
            errors.append("effective setting differs: " + key)

    def arguments(run: dict) -> list[str]:
        argv = list(run["execution_argv"])
        if "kv_storage" in dims:
            for i, arg in enumerate(argv[:-1]):
                if arg in ("--cache-type-k", "--cache-type-v"):
                    argv[i + 1] = "<controlled-kv-storage>"
        return argv

    if arguments(a) != arguments(b):
        errors.append("execution arguments differ outside selected axis")
    ac, bc = a["cases"], b["cases"]
    if [r["prompt_id"] for r in ac] != [r["prompt_id"] for r in bc]:
        errors.append("case set/order differs (missing or extra cases)")
        return errors
    for left, right in zip(ac, bc):
        for key in (
            "submitted_input_count",
            "submitted_input_sha256",
            "requested_max_tokens",
            "output_reservation",
            "mtp_reservation",
        ):
            if left.get(key) != right.get(key):
                errors.append(left["prompt_id"] + ": " + key + " differs")
    return errors


def compare(a: dict, b: dict) -> dict:
    reasons = validate_pair(a, b)
    result = {
        "schema_version": 1,
        "status": "INCOMPARABLE" if reasons else "COMPARABLE",
        "reasons": reasons,
        "qualification": "NOT_RUN",
        "cases": [],
    }
    if reasons:
        return result
    for left, right in zip(a["cases"], b["cases"]):
        result["cases"].append(
            {
                "prompt_id": left["prompt_id"],
                "tokens_identical": left["tokens"] == right["tokens"],
                "ttft_delta_s": right["ttft_s"] - left["ttft_s"],
                "wall_delta_s": right["wall_s"] - left["wall_s"],
                "baseline_token_times_s": left["token_times_s"],
                "candidate_token_times_s": right["token_times_s"],
            }
        )
    result["experiment"] = a["experiment"]
    result["interpretation"] = (
        "paired observations only; no speedup or quality qualification"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(read_json(args.baseline), read_json(args.candidate))
    with args.out.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, allow_nan=False)
    return 0 if result["status"] == "COMPARABLE" else 1


if __name__ == "__main__":
    sys.exit(main())
