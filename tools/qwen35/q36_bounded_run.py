"""Private, Windows-only, H-volume bounded sequential GEN evidence collection.

The spec is a trusted local execution instruction, never model output. argv is
the complete argument vector after engine_path (include --serve explicitly).
The JSON fixture is {"cases": [{"prompt_id", "input_ids", "max_tokens",
"output_reservation", "mtp_reservation"}]}. Only synthetic/public fixtures
belong here: raw streams and token IDs are private and are never published.

schema_version=1 also requires model_path, prompt_fixture_path, output_root,
toolchain_sha256, requested_settings and experiment. Settings require context,
prefill, parallel, kv_k, kv_v, mtp, router_top_k and expert_quantization.
experiment={"dimensions": [], "max_relative_regression": 0.05} is frozen before
execution. Optional pack_path/pack_sha256 binds a separate pack. Artifact hashes
are verified, not inferred from file names. A PASS means protocol execution only:
native consumed-input receipts are unavailable and deliberately remain unknown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Callable
from uuid import uuid4

try:
    from .q36_phase_metrics import (
        validate_observation_spec,
        verify_observation_inputs,
        write_artifacts,
    )
except ImportError:
    from q36_phase_metrics import (
        validate_observation_spec,
        verify_observation_inputs,
        write_artifacts,
    )

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from native_model_resource_guard import PressurePolicy, available_memory

GIB = 1024**3
MAX_JSON = 2 * 1024**2
MAX_LOG = 8 * 1024**2
MAX_LINE = 64 * 1024
SETTINGS = (
    "context",
    "prefill",
    "parallel",
    "kv_k",
    "kv_v",
    "mtp",
    "router_top_k",
    "expert_quantization",
)
DIMENSIONS = {"engine_toolchain", "kv_storage"}


class ResourceAbort(RuntimeError):
    """Carry the failing observation without losing it on the guard's error path."""

    def __init__(self, reason: str, sample: dict) -> None:
        super().__init__(reason)
        self.sample = sample


def digest(path: Path, check: Callable[[], None] = lambda: None) -> str:
    """Hash in bounded chunks, checking admission during potentially large reads."""
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024**2):
            check()
            value.update(block)
    return value.hexdigest()


def token_hash(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path, expected_sha256: str | None = None) -> dict:
    with path.open("rb") as source:
        data = source.read(MAX_JSON + 1)
    if len(data) > MAX_JSON:
        raise ValueError("json_size_limit")
    if (
        expected_sha256 is not None
        and hashlib.sha256(data).hexdigest() != expected_sha256
    ):
        raise ValueError("json_digest_mismatch")

    def unique(pairs: list[tuple[str, object]]) -> dict:
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate_json_key")
            obj[key] = value
        return obj

    value = json.loads(data, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("json_object_required")
    return value


def integer(value: object, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


OUTPUT_VOLUME = "H:"


def validate_spec(spec: dict) -> None:
    """Reject incomplete identities and attempts to weaken the resource floors."""
    required = {
        "schema_version",
        "run_id",
        "engine_path",
        "engine_sha256",
        "argv",
        "model_path",
        "model_sha256",
        "prompt_fixture_path",
        "prompt_fixture_sha256",
        "toolchain_sha256",
        "ready_timeout_s",
        "total_timeout_s",
        "min_available_ram_bytes",
        "output_root",
        "output_dir",
        "requested_settings",
        "experiment",
    }
    if not isinstance(spec, dict) or not required <= spec.keys():
        raise ValueError("missing_spec_fields")
    if spec.get("schema_version") != 1:
        raise ValueError("schema_version")
    if "observation" in spec:
        validate_observation_spec(spec["observation"])
    if not isinstance(spec["run_id"], str) or not re.fullmatch(
        r"[A-Za-z0-9_-]{1,80}", spec["run_id"]
    ):
        raise ValueError("run_id")
    for key in ("engine", "model", "prompt_fixture"):
        if (
            not sha(spec.get(key + "_sha256"))
            or not isinstance(spec[key + "_path"], str)
            or not Path(spec[key + "_path"]).is_absolute()
        ):
            raise ValueError(key + "_identity")
    if not sha(spec.get("toolchain_sha256")):
        raise ValueError("toolchain_identity")
    if "pack_path" in spec or "pack_sha256" in spec:
        if (
            not sha(spec.get("pack_sha256"))
            or not Path(spec["pack_path"]).is_absolute()
        ):
            raise ValueError("pack_identity")
    argv = spec.get("argv")
    if (
        not isinstance(argv, list)
        or len(argv) > 128
        or any(
            not isinstance(arg, str) or "\x00" in arg or len(arg) > 8192 for arg in argv
        )
    ):
        raise ValueError("argv")
    for key in ("ready_timeout_s", "total_timeout_s"):
        value = spec.get(key)
        if (
            type(value) not in (float, int)
            or not math.isfinite(value)
            or not 0 < value <= 3600
        ):
            raise ValueError(key)
    if spec["ready_timeout_s"] > spec["total_timeout_s"]:
        raise ValueError("deadline_order")
    for key, minimum in (
        ("min_available_ram_bytes", 3 * GIB),
        ("min_commit_headroom_bytes", 4 * GIB),
        ("min_disk_free_bytes", GIB),
    ):
        if not integer(spec.get(key, minimum), minimum, 1024 * GIB):
            raise ValueError(key)
    if spec.get("job_commit_bytes") is not None and not integer(
        spec["job_commit_bytes"], 64 * 1024**2, 24 * GIB
    ):
        raise ValueError("job_commit_bytes")
    settings = spec.get("requested_settings")
    if not isinstance(settings, dict) or set(settings) != set(SETTINGS):
        raise ValueError("requested_settings")
    for key in ("context", "prefill", "parallel", "router_top_k"):
        if not integer(settings[key], 1, 262144):
            raise ValueError(key)
    if settings["parallel"] != 1 or not integer(settings["mtp"], 0, 64):
        raise ValueError("single_session_and_mtp")
    if any(
        settings[key] not in ("f32", "f16", "q8_0", "nvfp4") for key in ("kv_k", "kv_v")
    ):
        raise ValueError("kv_storage")
    if (
        not isinstance(settings["expert_quantization"], str)
        or not settings["expert_quantization"]
    ):
        raise ValueError("expert_quantization")
    experiment = spec.get("experiment")
    if not isinstance(experiment, dict) or set(experiment) != {
        "dimensions",
        "max_relative_regression",
    }:
        raise ValueError("experiment")
    dims = experiment["dimensions"]
    if (
        not isinstance(dims, list)
        or any(not isinstance(d, str) or d not in DIMENSIONS for d in dims)
        or len(set(dims)) != len(dims)
    ):
        raise ValueError("experiment_dimensions")
    threshold = experiment["max_relative_regression"]
    if (
        type(threshold) not in (float, int)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("experiment_threshold")
    root, output = Path(spec["output_root"]), Path(spec["output_dir"])
    if not root.is_absolute() or not output.is_absolute() or not root.is_dir():
        raise ValueError("output_root")
    if root.resolve().drive.upper() != OUTPUT_VOLUME:
        raise ValueError("existing_guard_requires_H_volume_output")
    # An immediate new child of an explicitly selected root prevents traversal and overwrites.
    if (
        output.parent.resolve() != root.resolve()
        or output.exists()
        or output.is_symlink()
    ):
        raise ValueError("new_contained_output_required")


def load_cases(
    path: Path, capacity: int, expected_sha256: str | None = None
) -> list[dict]:
    fixture = read_json(path, expected_sha256)
    cases = fixture.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 32:
        raise ValueError("case_count")
    seen = set()
    total_output = 0
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "prompt_id",
            "input_ids",
            "max_tokens",
            "output_reservation",
            "mtp_reservation",
        }:
            raise ValueError("case_schema")
        ident = case["prompt_id"]
        if (
            not isinstance(ident, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", ident)
            or ident in seen
        ):
            raise ValueError("duplicate_or_invalid_prompt_id")
        seen.add(ident)
        ids = case["input_ids"]
        if (
            not isinstance(ids, list)
            or not 1 <= len(ids) <= 131072
            or any(not integer(i, 0, 2**31 - 1) for i in ids)
        ):
            raise ValueError("input_ids")
        if not integer(case["max_tokens"], 1, 4096):
            raise ValueError("max_tokens")
        if not integer(
            case["output_reservation"], case["max_tokens"], 4096
        ) or not integer(case["mtp_reservation"], 0, 64):
            raise ValueError("reservations")
        if len(ids) + case["output_reservation"] + case["mtp_reservation"] > capacity:
            raise ValueError("input_plus_reservations_exceeds_capacity")
        total_output += case["max_tokens"]
    if total_output > 32768:
        raise ValueError("total_output_token_limit")
    return cases


def _emit(event: dict) -> None:
    sys.stdout.write(json.dumps(event, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _worker(spec: dict) -> int:
    """Runs only inside the outer atomic Job. The stdin handshake gates target spawn."""
    if sys.stdin.readline(32) != "OWNED\n":
        return 125
    output = Path(spec["output_dir"])
    # Recheck executable bytes in the contained process immediately before spawn.
    if digest(Path(spec["engine_path"])) != spec["engine_sha256"]:
        _emit({"event": "failure", "reason": "engine_digest_changed"})
        return 125
    events: queue.Queue = queue.Queue(maxsize=128)
    failure: list[str] = []
    lock = threading.Lock()
    written = 0
    child_env = dict(os.environ)
    if "observation" in spec:
        child_env["STRATA_Q36_PHASE_METRICS"] = "1"
        child_env["STRATA_Q36_STARTUP_GENERATION"] = spec["observation_generation"]
    # The native switches are presence-based: even the string '0' enables
    # hotspot/verification synchronization in the existing executable.
    hotspot = "STRATA_Q36_HOTSPOT_PROFILE" in child_env
    verify = "STRATA_Q36_VERIFY" in child_env
    counters = child_env.get("STRATA_Q36_PHASE_METRICS") == "1"
    _emit(
        {
            "event": "instrumentation",
            "value": {
                "mode": (
                    "diagnostic_sync"
                    if hotspot or verify
                    else "host_counters" if counters else "off"
                ),
                "hotspot_sync": hotspot,
                "phase_counters": counters,
                "kernel_verify": verify,
                "startup_detail": child_env.get("STRATA_Q36_STARTUP_DETAIL") == "1",
            },
        }
    )
    spawned_at = time.monotonic()
    proc = subprocess.Popen(
        [spec["engine_path"], *spec["argv"]],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW,
        env=child_env,
    )
    _emit({"event": "child", "pid": proc.pid, "at": spawned_at})

    def pump(stream, name: str) -> None:
        nonlocal written
        try:
            with (output / (name + ".log")).open("xb", buffering=0) as raw:
                while line := stream.readline(MAX_LINE + 1):
                    with lock:
                        room = max(0, MAX_LOG - written)
                        raw.write(line[:room])
                        written += min(room, len(line))
                    if len(line) > room:
                        raise ValueError("output_limit")
                    if len(line) > MAX_LINE:
                        raise ValueError("line_limit")
                    events.put_nowait(
                        (
                            name,
                            line.decode("utf-8", errors="strict").rstrip("\r\n"),
                            time.monotonic(),
                        )
                    )
            events.put_nowait((name, None, time.monotonic()))
        except queue.Full:
            failure.append("queue_limit")
        except Exception as error:
            failure.append(
                str(error)
                if isinstance(error, ValueError)
                else "stream_observer_failure"
            )

    for stream, name in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
        threading.Thread(target=pump, args=(stream, name), daemon=True).start()
    closed = set()

    def line() -> tuple[str, float]:
        while True:
            if failure:
                raise RuntimeError(failure[0])
            try:
                source, text, stamp = events.get(timeout=0.05)
            except queue.Empty:
                continue
            if text is None:
                closed.add(source)
                if len(closed) == 2:
                    raise RuntimeError("child_eof")
                continue
            if source == "stderr":
                if text.startswith("Q36_EFFECTIVE "):
                    fields = dict(
                        word.split("=", 1) for word in text.split()[1:] if "=" in word
                    )
                    _emit({"event": "effective", "fields": fields})
                continue
            if text.startswith("ERR"):
                raise RuntimeError("engine_error")
            return text, stamp

    try:
        while True:
            text, stamp = line()
            if text == "READY" or text.startswith("READY "):
                _emit({"event": "ready", "at": stamp})
                break
        for index, case in enumerate(spec["cases"]):
            sent = time.monotonic()
            _emit({"event": "request", "index": index, "at": sent})
            command = (
                f"GEN {case['max_tokens']} "
                + ",".join(map(str, case["input_ids"]))
                + "\n"
            )
            proc.stdin.write(command.encode())
            proc.stdin.flush()
            count = 0
            while True:
                text, stamp = line()
                if text.startswith("T "):
                    parts = text.split()
                    if (
                        len(parts) != 2
                        or not parts[1].isdigit()
                        or int(parts[1]) > 2**31 - 1
                    ):
                        raise RuntimeError("malformed_token")
                    count += 1
                    if count > case["max_tokens"]:
                        raise RuntimeError("generation_limit_exceeded")
                    _emit(
                        {
                            "event": "token",
                            "index": index,
                            "id": int(parts[1]),
                            "at": stamp,
                        }
                    )
                elif text == "DONE" or text.startswith("DONE "):
                    if count == 0:
                        raise RuntimeError("empty_output")
                    _emit({"event": "done", "index": index, "at": stamp})
                    break
        proc.stdin.write(b"STOP\n")
        proc.stdin.close()
        # Drain both pipes before exit so late stderr/effective rows cannot be lost.
        while len(closed) < 2:
            try:
                line()
            except RuntimeError as error:
                if str(error) != "child_eof":
                    raise
        code = proc.wait(timeout=5)
        return 0 if code == 0 else 1
    except Exception as error:
        reason = str(error) if isinstance(error, RuntimeError) else "worker_failure"
        _emit({"event": "failure", "reason": reason})
        return 1
    finally:
        # The outer job owns descendants; never enumerate names, ports or PIDs.
        if proc.poll() is None:
            proc.terminate()
        code = proc.wait(timeout=5)
        _emit({"event": "exit", "code": code})


def run_bounded(spec: dict) -> dict:
    """Collect protocol evidence; PASS does not imply comparable or qualified."""
    validate_spec(spec)
    if os.name != "nt":
        raise ValueError("windows_job_required")
    from perf131k.windows_job import run_owned

    output = Path(spec["output_dir"]).resolve()
    start = time.monotonic()
    policy = PressurePolicy(
        physical_floor_bytes=spec["min_available_ram_bytes"],
        commit_floor_bytes=spec.get("min_commit_headroom_bytes", 4 * GIB),
        physical_grace_seconds=0,
    )
    disk_floor = spec.get("min_disk_free_bytes", GIB)

    def resources() -> dict:
        sample = available_memory()
        reason = policy.observe(sample, time.monotonic())
        sample["disk_free_bytes"] = shutil.disk_usage(output.parent).free
        if reason:
            raise ResourceAbort(reason, sample)
        if sample["disk_free_bytes"] < disk_floor:
            raise ResourceAbort("disk_pressure", sample)
        if time.monotonic() - start >= spec["total_timeout_s"]:
            raise ResourceAbort("total_deadline", sample)
        return sample

    def abort_preflight(error: Exception) -> dict:
        # Admission failure has no owned process. Keep a small receipt when the
        # destination is writable; an unwritable disk still surfaces its OSError.
        result = {
            "schema_version": 1,
            "run_id": spec["run_id"],
            "status": "ABORTED",
            "reason": (
                str(error)
                if isinstance(error, RuntimeError)
                else "resource_observer_failure"
            ),
            "last_stage": "preflight",
            "child_exit_code": None,
            "requests_status": "NOT_RUN",
            "cases": [],
            "qualification": "NOT_RUN",
            "initial_resources": (
                error.sample if isinstance(error, ResourceAbort) else None
            ),
        }
        output.mkdir()
        with (output / "result.json").open("x", encoding="utf-8") as target:
            json.dump(result, target, indent=2)
        if "observation" in spec:
            write_artifacts(output, result, spec)
        return result

    try:
        initial = resources()
    except (RuntimeError, OSError, ValueError) as error:
        return abort_preflight(error)
    try:
        if "observation" in spec:
            verify_observation_inputs(
                spec["observation"], lambda path: digest(path, resources)
            )
        for kind in ("engine", "model", "prompt_fixture", "pack"):
            if (
                kind + "_path" in spec
                and digest(Path(spec[kind + "_path"]), resources)
                != spec[kind + "_sha256"]
            ):
                raise ValueError(kind + "_digest_mismatch")
    except ResourceAbort as error:
        # Only resource/deadline failures become abort receipts. Invalid artifacts
        # remain distinct validation errors and never start an owned process.
        return abort_preflight(error)
    cases = load_cases(
        Path(spec["prompt_fixture_path"]),
        spec["requested_settings"]["context"],
        spec["prompt_fixture_sha256"],
    )
    # Bound spec copying and refuse unknown top-level fields to avoid persisting credentials.
    allowed = {
        "schema_version",
        "run_id",
        "argv",
        "output_root",
        "output_dir",
        "ready_timeout_s",
        "total_timeout_s",
        "min_available_ram_bytes",
        "min_commit_headroom_bytes",
        "min_disk_free_bytes",
        "job_commit_bytes",
        "requested_settings",
        "experiment",
        "toolchain_sha256",
        "observation",
    }
    allowed |= {
        kind + suffix
        for kind in ("engine", "model", "prompt_fixture", "pack")
        for suffix in ("_path", "_sha256")
    }
    if set(spec) - allowed:
        raise ValueError("unknown_spec_fields")
    output.mkdir()
    worker_spec = dict(spec, cases=cases)
    if "observation" in spec:
        worker_spec["observation_generation"] = uuid4().hex
    if len(json.dumps(worker_spec).encode()) > MAX_JSON:
        raise ValueError("worker_spec_size_limit")
    with (output / "spec.json").open("x", encoding="utf-8") as target:
        json.dump(worker_spec, target)
    (output / "handshake.txt").write_text("OWNED\n", encoding="ascii")
    result = {
        "schema_version": 1,
        "run_id": spec["run_id"],
        "status": "NOT_RUN",
        "reason": None,
        "last_stage": "preflight",
        "child_exit_code": None,
        "initial_resources": initial,
        "preflight_s": time.monotonic() - start,
        "gpu_memory_observation": None,
        "gpu_memory_reason": "GPU observation is outside this CPU resource harness",
        "execution_argv": spec["argv"],
        "identities": {
            key: value for key, value in spec.items() if key.endswith("_sha256")
        },
        "requested_settings": spec["requested_settings"],
        "experiment": spec["experiment"],
        "effective_settings": {},
        "startup_generation": None,
        "instrumentation": None,
        "cases": [],
        "expected_case_ids": [case["prompt_id"] for case in cases],
        "qualification": "NOT_RUN",
        "input_integrity": "NOT_RUN",
    }
    result["identities"].update(
        harness_sha256=digest(Path(__file__)),
        guard_sha256=digest(ROOT / "tools/perf131k/windows_job.py"),
        resource_guard_sha256=digest(ROOT / "tools/native_model_resource_guard.py"),
        worker_interpreter_sha256=digest(Path(sys.executable)),
        phase_metrics_sha256=digest(Path(__file__).with_name("q36_phase_metrics.py")),
    )
    for i, case in enumerate(cases):
        result["cases"].append(
            {
                "prompt_id": case["prompt_id"],
                "status": "NOT_RUN",
                "request_id": f"{spec['run_id']}:{i}",
                "submitted_input_count": len(case["input_ids"]),
                "submitted_input_sha256": token_hash(case["input_ids"]),
                "requested_max_tokens": case["max_tokens"],
                "output_reservation": case["output_reservation"],
                "mtp_reservation": case["mtp_reservation"],
                "consumed_input": None,
                "consumed_input_reason": "native_consumption_receipt_unavailable",
                "effective_max_tokens": None,
                "tokens": [],
                "token_times_s": [],
                "ttft_s": None,
                "wall_s": None,
                "request_started": None,
            }
        )
    pending = bytearray()
    abort_reason = None
    active = None
    ready = False
    last_sample = -1.0

    def consume_output(block: bytes) -> None:
        nonlocal abort_reason, active, ready
        pending.extend(block)
        if len(pending) > MAX_LINE * 2:
            raise ValueError("event_buffer_limit")
        while b"\n" in pending:
            raw, _, tail = pending.partition(b"\n")
            pending[:] = tail
            event = json.loads(raw)
            kind = event["event"]
            if kind == "ready":
                ready = True
                result["last_stage"] = "ready"
                result["ready_s"] = event["at"] - start
                result["engine_start_to_ready_s"] = (
                    event["at"] - result["engine_started_at"]
                )
            elif kind == "instrumentation":
                value = event["value"]
                if (
                    not isinstance(value, dict)
                    or set(value)
                    != {
                        "mode",
                        "hotspot_sync",
                        "phase_counters",
                        "startup_detail",
                        "kernel_verify",
                    }
                    or value["mode"] not in ("off", "host_counters", "diagnostic_sync")
                    or any(
                        type(value[k]) is not bool
                        for k in (
                            "hotspot_sync",
                            "phase_counters",
                            "startup_detail",
                            "kernel_verify",
                        )
                    )
                ):
                    raise ValueError("instrumentation_schema")
                result["instrumentation"] = value
            elif kind == "child":
                result["owned_engine_pid"] = event["pid"]
                result["engine_started_at"] = event["at"]
                result["preflight_and_worker_start_s"] = event["at"] - start
            elif kind == "effective":
                fields = event["fields"]
                if fields.get("schema") != "1":
                    raise ValueError("effective_schema_mismatch")
                generation = fields.pop("generation", None)
                if not generation or result["startup_generation"] not in (
                    None,
                    generation,
                ):
                    raise ValueError("startup_generation_mismatch")
                result["startup_generation"] = generation
                for key in SETTINGS:
                    if key in fields:
                        result["effective_settings"][key] = (
                            int(fields[key])
                            if key
                            in ("context", "prefill", "parallel", "mtp", "router_top_k")
                            else fields[key]
                        )
            elif kind == "request":
                active = event["index"]
                row = result["cases"][active]
                row.update(status="ABORTED", request_started=event["at"])
                result["last_stage"] = row["request_id"]
            elif kind == "token":
                row = result["cases"][event["index"]]
                elapsed = event["at"] - row["request_started"]
                row["tokens"].append(event["id"])
                row["token_times_s"].append(elapsed)
                if row["ttft_s"] is None:
                    row["ttft_s"] = elapsed
            elif kind == "done":
                row = result["cases"][event["index"]]
                row.update(status="PASS", wall_s=event["at"] - row["request_started"])
                active = None
            elif kind == "failure":
                abort_reason = event["reason"]
                if active is not None:
                    result["cases"][active]["status"] = "FAILED"
            elif kind == "exit":
                result["child_exit_code"] = event["code"]
            else:
                raise ValueError("unknown_worker_event")

    def on_output(block: bytes) -> None:
        nonlocal abort_reason
        try:
            consume_output(block)
        except Exception:
            abort_reason = "event_observer_failure"
            raise

    with (output / "memory.jsonl").open("x", encoding="utf-8") as memory:

        def sample(_: Path) -> dict:
            nonlocal abort_reason, last_sample
            try:
                value = resources()
            except Exception as error:
                abort_reason = (
                    str(error)
                    if isinstance(error, RuntimeError)
                    else "resource_observer_failure"
                )
                if isinstance(error, ResourceAbort):
                    memory.write(
                        json.dumps(
                            dict(
                                error.sample,
                                elapsed_s=time.monotonic() - start,
                                abort_reason=abort_reason,
                            )
                        )
                        + "\n"
                    )
                    memory.flush()
                raise
            elapsed = time.monotonic() - start
            if elapsed - last_sample >= 0.25:
                try:
                    memory.write(json.dumps(dict(value, elapsed_s=elapsed)) + "\n")
                    memory.flush()
                except OSError:
                    abort_reason = "memory_trace_write_failure"
                    raise
                last_sample = elapsed
            return {
                "available_ram": value["available_physical_bytes"],
                "commit_headroom": value["available_commit_bytes"],
                "H_free": value["disk_free_bytes"],
            }

        def tick() -> str | None:
            if abort_reason:
                return abort_reason
            # Large model/pack hashing is preflight, not engine startup. The
            # unchanged total deadline still covers preflight and the full run.
            ready_origin = result.get("engine_started_at", launch_started)
            if not ready and time.monotonic() - ready_origin >= spec["ready_timeout_s"]:
                return "ready_deadline"
            return None

        launch_started = time.monotonic()
        budget = SimpleNamespace(
            timeout_s=max(0.001, spec["total_timeout_s"] - (time.monotonic() - start)),
            job_commit_bytes=spec.get("job_commit_bytes"),
            max_processes=16,
            output_bytes=MAX_LOG,
            min_available_ram=policy.physical_floor_bytes,
            min_commit_headroom=policy.commit_floor_bytes,
            min_H_free=disk_floor,
        )
        result["guard"] = run_owned(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                str(output / "spec.json"),
            ],
            output,
            dict(os.environ),
            output / "events.jsonl",
            budget,
            sample,
            stdin_path=output / "handshake.txt",
            on_output=on_output,
            on_tick=tick,
            memory_diagnostics="observation" in spec,
        )
    guard = result["guard"]
    success = (
        guard["status"] == "FINISHED"
        and guard["exit_code"] == 0
        and guard["termination_confirmed"]
        and ready
        and not pending
        and not abort_reason
        and result["child_exit_code"] == 0
        and all(row["status"] == "PASS" for row in result["cases"])
    )
    result["status"] = "PASS" if success else "ABORTED"
    result["reason"] = (
        None
        if success
        else abort_reason
        or guard.get("observer_reason")
        or ("total_deadline" if guard["status"] == "TIMEOUT" else guard["status"])
    )
    if not success and (
        abort_reason in ("empty_output", "engine_error", "child_eof", "worker_failure")
        or result["child_exit_code"] not in (None, 0)
    ):
        result["status"] = "FAILED"
    for row in result["cases"]:
        row["startup_generation"] = result["startup_generation"]
        row.pop("request_started")
    result["elapsed_s"] = time.monotonic() - start
    result["child_exit_code_reason"] = (
        "observed"
        if result["child_exit_code"] is not None
        else "not_observed_before_job_cleanup"
    )
    with (output / "result.json").open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, allow_nan=False)
    if "observation" in spec:
        write_artifacts(output, result, spec)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--spec", type=Path)
    mode.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return _worker(read_json(args.worker))
    result = run_bounded(read_json(args.spec))
    sys.stdout.write(
        json.dumps({"status": result["status"], "reason": result["reason"]}) + "\n"
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
