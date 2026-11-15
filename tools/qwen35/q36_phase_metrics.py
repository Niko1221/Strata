"""Private PR00 artifacts for the existing Q36 bounded runner; never qualification.

Native read counters cover explicit expert-store API reads, not physical SSD
traffic or mmap faults. Copy counters cover submitted expert bytes, not elapsed
GPU time. Missing observations remain null, including cancel/snapshot/unload.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Callable

PHASES = ("load", "warmup", "prefill", "decode", "snapshot", "cancel", "unload")
COUNTS = (
    "read_calls",
    "read_requested_bytes",
    "read_completed_bytes",
    "read_failed_calls",
    "expert_h2d_submitted_bytes",
    "expert_d2d_submitted_bytes",
    "cpu_jobs",
    "input_tokens",
    "reused_tokens",
    "processed_tokens",
    "committed_tokens",
)
TIMES = ("wall_ms", "read_wall_ms", "cpu_compute_ms", "route_wait_ms")
IDENTITIES = {
    "tokenizer",
    "chat_template",
    "sampler",
    "model_metadata",
    "build_receipt",
}
SHA = re.compile(r"[0-9a-f]{64}")


def validate_observation_spec(value: dict) -> None:
    """Validate the trusted local observation spec before any worker is spawned."""
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "family",
        "source_sha",
        "binary_source_sha",
        "environment_manifest",
        "identity_files",
    }:
        raise ValueError("observation_schema")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("observation_schema_version")
    if value["family"] != "qwen36":
        raise ValueError("observation_family_requires_qwen36_runner")
    for key in ("source_sha", "binary_source_sha"):
        if value[key] is not None and not (
            isinstance(value[key], str) and re.fullmatch(r"[0-9a-f]{40}", value[key])
        ):
            raise ValueError("observation_source_identity")
    identities = value["identity_files"]
    if not isinstance(identities, dict) or set(identities) - IDENTITIES:
        raise ValueError("observation_identity_files")
    for entry in [value["environment_manifest"], *identities.values()]:
        if entry is None:
            continue
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256"}
            or not isinstance(entry["path"], str)
            or not Path(entry["path"]).is_absolute()
            or not isinstance(entry["sha256"], str)
            or not SHA.fullmatch(entry["sha256"])
        ):
            raise ValueError("observation_file_identity")


def verify_observation_inputs(value: dict, digest: Callable[[Path], str]) -> None:
    """The caller supplies its resource-guarded streaming digest implementation."""
    validate_observation_spec(value)
    for entry in [value["environment_manifest"], *value["identity_files"].values()]:
        if entry is not None and digest(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("observation_file_digest_mismatch")


def parse_phase_line(line: str) -> dict | None:
    """Reject malformed/unknown telemetry rather than converting it to zeros."""
    if not line.startswith("Q36_PHASE "):
        return None
    words = line.split()[1:]
    if any(word.count("=") != 1 for word in words):
        raise ValueError("phase_fields")
    fields = dict(word.split("=", 1) for word in words)
    if len(fields) != len(words) or set(fields) != {
        "schema",
        "generation",
        "request",
        "phase",
        "state",
        "layer",
        "instrumentation",
        "gpu_time",
        "disk_physical_bytes",
        *COUNTS,
        *TIMES,
    }:
        raise ValueError("phase_schema_fields")
    if (
        fields["schema"] != "1"
        or fields["phase"] not in PHASES
        or fields["state"] not in {"complete", "cancelled", "incomplete"}
        or fields["instrumentation"] != "host_counters"
        or fields["gpu_time"] != "unavailable"
        or fields["disk_physical_bytes"] != "unavailable"
        or not re.fullmatch(r"[A-Za-z0-9-]{1,80}", fields["generation"])
    ):
        raise ValueError("phase_schema_values")
    for key in (*COUNTS, "request"):
        if not re.fullmatch(r"[0-9]{1,20}", fields[key]):
            raise ValueError("phase_counter")
        fields[key] = int(fields[key])
        if fields[key] > 2**64 - 1:
            raise ValueError("phase_counter_overflow")
    if not re.fullmatch(r"-1|[0-9]{1,2}", fields["layer"]):
        raise ValueError("phase_layer")
    fields["layer"] = int(fields["layer"])
    if not -1 <= fields["layer"] <= 64:
        raise ValueError("phase_layer")
    for key in TIMES:
        fields[key] = float(fields[key])
        if not math.isfinite(fields[key]) or fields[key] < 0:
            raise ValueError("phase_time")
    if (
        fields["read_completed_bytes"] > fields["read_requested_bytes"]
        or fields["read_failed_calls"] > fields["read_calls"]
    ):
        raise ValueError("phase_read_accounting")
    if fields["reused_tokens"] + fields["processed_tokens"] > fields["input_tokens"]:
        raise ValueError("phase_input_accounting")
    fields.update(schema=1, gpu_time=None, disk_physical_bytes=None)
    return fields


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank quantile; callers retain the actual sample count."""
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def protocol_metrics(case: dict) -> dict:
    """Count only protocol-committed tokens, separating TTFT from token intervals."""
    stamps = case.get("token_times_s", [])
    if any(
        type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in stamps
    ):
        raise ValueError("invalid_protocol_timing")
    if any(b < a for a, b in zip(stamps, stamps[1:])) or len(stamps) != len(
        case.get("tokens", [])
    ):
        raise ValueError("invalid_protocol_timing")
    intervals = [b - a for a, b in zip(stamps, stamps[1:])]
    elapsed = stamps[-1] - stamps[0] if len(stamps) > 1 else 0
    return {
        "phase": "request",
        "request_id": case["request_id"],
        "prompt_id": case["prompt_id"],
        "status": case["status"],
        "source": "bounded_protocol_host_clock",
        "committed_tokens": len(stamps),
        "ttft_s": case.get("ttft_s"),
        "wall_s": case.get("wall_s"),
        "interval_samples": len(intervals),
        "token_interval_p50_s": percentile(intervals, 0.50),
        "token_interval_p95_s": percentile(intervals, 0.95),
        "token_interval_p99_s": percentile(intervals, 0.99),
        "committed_after_first_per_s": (
            len(intervals) / elapsed if elapsed > 0 else None
        ),
        "input_consumption": (
            "OBSERVED" if case.get("consumed_input") else "UNAVAILABLE"
        ),
        "qualification": "NOT_RUN",
    }


def build_artifacts(
    result: dict, spec: dict, native_lines: list[str]
) -> tuple[dict, list[dict], str]:
    """Build bounded artifacts from all runs, preserving errors and missing phases."""
    observation = spec["observation"]
    validate_observation_spec(observation)
    instrumentation = result.get("instrumentation")
    metrics, errors, seen = [], [], set()
    generation = result.get("startup_generation")
    for line in native_lines:
        try:
            row = parse_phase_line(line)
            if row is None:
                continue
            if (
                generation is None
                or generation == "null"
                or row["generation"] != generation
            ):
                raise ValueError("phase_generation_mismatch")
            if row["request"] > len(result.get("cases", [])):
                raise ValueError("phase_request_mismatch")
            key = (row["request"], row["phase"], row["layer"])
            if key in seen:
                raise ValueError("duplicate_phase_record")
            seen.add(key)
            row.update(source="q36_native_host_counters", qualification="NOT_RUN")
            metrics.append(row)
        except (ValueError, OverflowError):
            # Do not copy arbitrary native text, paths, prompts or error payloads.
            errors.append("invalid_native_phase_record")
    phases = {row["phase"] for row in metrics if row.get("layer") == -1}
    for phase in PHASES:
        if phase not in phases:
            metrics.append(
                {
                    "phase": phase,
                    "status": "NOT_OBSERVED",
                    "value": None,
                    "reason": "no_native_phase_receipt",
                    "qualification": "NOT_RUN",
                }
            )
    metrics.extend(protocol_metrics(case) for case in result.get("cases", []))
    identity_files = {
        name: observation["identity_files"].get(name)
        or {"sha256": None, "reason": "not_supplied_or_not_independently_bound"}
        for name in sorted(IDENTITIES)
    }
    missing = [
        name for name, value in identity_files.items() if value.get("sha256") is None
    ]
    if not observation["environment_manifest"]:
        missing.append("environment_manifest")
    if not observation["binary_source_sha"]:
        missing.append("binary_source_sha")
    mode = (instrumentation or {}).get("mode")
    manifest = {
        "schema_version": 1,
        "family": "qwen36",
        "run_id": result["run_id"],
        "status": result["status"],
        "reason": result.get("reason"),
        "declared_source_sha": observation["source_sha"],
        "declared_binary_source_sha": observation["binary_source_sha"],
        "source_binding": "declarations_require_review_of_hashed_build_receipt",
        "identities": result.get("identities", {}),
        "declared_identities": {
            name: spec.get(name)
            for name in (
                "engine_sha256",
                "model_sha256",
                "pack_sha256",
                "prompt_fixture_sha256",
                "toolchain_sha256",
            )
        },
        "declared_identity_scope": "expected_spec_digests_not_proof_of_completed_preflight_verification",
        "identity_files": identity_files,
        "environment_manifest": observation["environment_manifest"],
        "requested": result.get("requested_settings", spec["requested_settings"]),
        "effective": result.get("effective_settings"),
        "observed": {
            "startup_generation": generation,
            "instrumentation": instrumentation,
            "initial_system_resources": result.get("initial_resources"),
            "gpu_memory": result.get("gpu_memory_observation"),
            "preflight_and_worker_start_s": result.get("preflight_and_worker_start_s"),
            "preflight_s": result.get("preflight_s"),
            "engine_start_to_ready_s": result.get("engine_start_to_ready_s"),
            "owned_job_peak_commit_bytes": result.get("guard", {}).get(
                "peak_job_commit_bytes"
            ),
            "job_scope": "worker_and_engine_including_observer_overhead",
            "native_process_memory": result.get("native_process_memory"),
        },
        "case_statuses": [
            {"prompt_id": c["prompt_id"], "status": c["status"]}
            for c in result.get("cases", [])
        ],
        "telemetry_errors": errors,
        "missing_reproducibility": missing,
        "performance_eligible": False,
        "performance_eligibility_reason": "requires_separate_native_input_state_and_resource_gates",
        "unavailable_counters": [
            "physical_ssd_read_bytes",
            "gpu_elapsed_ms",
            "h2d_elapsed_ms",
            "cuda_free_bytes",
            "dxgi_budget_bytes",
            "dxgi_usage_bytes",
            "shared_gpu_bytes",
            "pinned_peak_bytes",
            "hard_faults",
            "pagefile_bytes",
            "native_process_private_peak_bytes",
            "native_process_working_set_peak_bytes",
        ],
        "qualification": "NOT_RUN",
        "releaseQualified": False,
    }
    # The report contains only allowlisted identifiers, settings, statuses and
    # computed metrics. Arbitrary process/error text stays in the private raw log.
    statuses = {"PASS", "FAILED", "ABORTED", "NOT_RUN"}
    status = result["status"] if result["status"] in statuses else "UNKNOWN"
    report = [
        "# Q36 PR00 observation",
        "",
        f"Run status: {status}. Qualification: NOT_RUN.",
        "",
        "This is an observation, not a speedup, quality or 64k/131k qualification.",
        "Native reads count explicit expert-store API bytes, not physical SSD I/O.",
        "Expert copies count submitted bytes, not GPU elapsed time or proven overlap.",
        "Missing GPU/DXGI/phase/identity evidence remains unavailable.",
        "",
        "| Case | Status | Committed tokens | TTFT (s) | Interval samples |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in metrics:
        if row.get("phase") != "request":
            continue
        ident = (
            row["prompt_id"]
            if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", row["prompt_id"])
            else "redacted"
        )
        case_status = row["status"] if row["status"] in statuses else "UNKNOWN"
        ttft = row["ttft_s"]
        report.append(
            f"| {ident} | {case_status} | {row['committed_tokens']} | "
            f"{ttft if ttft is not None else 'unknown'} | {row['interval_samples']} |"
        )
    report.extend(
        [
            "",
            f"Native telemetry errors: {len(errors)}.",
            f"Missing identity fields: {', '.join(missing) or 'none'}.",
            f"Instrumentation: {mode if mode in {'off', 'host_counters', 'diagnostic_sync'} else 'unknown'}.",
            "TTFT percentiles need a separate request population; token intervals are not TTFT samples.",
            "Use the existing q36_compare_runs fail-closed input/state checks before comparing runs.",
            "",
        ]
    )
    return manifest, metrics, "\n".join(report)


def write_artifacts(output: Path, result: dict, spec: dict) -> None:
    """Reserve all new outputs before writing; never replace an earlier receipt."""
    paths = [
        output / name
        for name in ("run_manifest.json", "phase_metrics.jsonl", "report.md")
    ]
    if any(path.exists() for path in paths):
        raise FileExistsError("preserve_existing_observation_artifacts")
    stderr = output / "stderr.log"
    lines = stderr.read_text(encoding="utf-8").splitlines() if stderr.exists() else []
    manifest, metrics, report = build_artifacts(result, spec, lines)
    # These hashes bind private raw evidence without copying its contents.
    manifest["artifact_sha256"] = {}
    for name in (
        "result.json",
        "stdout.log",
        "stderr.log",
        "memory.jsonl",
        "native-process-memory.jsonl",
    ):
        path = output / name
        if path.is_file():
            with path.open("rb") as stream:
                manifest["artifact_sha256"][name] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    with paths[0].open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    with paths[1].open("x", encoding="utf-8") as stream:
        for row in metrics:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    with paths[2].open("x", encoding="utf-8") as stream:
        stream.write(report)
