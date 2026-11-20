"""Local P01-A test gate. No engine imports; preserves each receipt and raw output."""

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from owned_job import run_owned


def digest(path: Path) -> str:
    """Hash only small task source/artifacts, never model data."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def inputs(root: Path, command: list[str]) -> dict:
    """Bind evidence to the exact collector, analyzer, tests and local fixture executable."""
    paths = list((root / "tests/perf131k").glob("*.py")) + list((root / "tests/perf131k").glob("*.cpp"))
    paths += [root / "include/strata/perf131k/trace.hpp", root / "tools/perf131k/capture_trace.py"]
    paths += [
        root / path
        for path in (
            "src/program/generate.cpp",
            "src/core/verify.cpp",
            "src/core/expert_source.cpp",
            "src/prefill/prefill.cpp",
        )
    ]
    series = os.environ.get("P01_FIXTURE_SERIES", "02")
    if series.isalnum():
        paths += list((root / "_scratch/perf131k").glob(f"cpu-*-{series}.jsonl"))
    for value in [*command, os.environ.get("P01_ANALYZER_UNDER_TEST", "")]:
        if not value:
            continue
        candidate = Path(value)
        candidate = candidate if candidate.is_absolute() else root / candidate
        if candidate.is_file() and candidate.is_relative_to(root) and candidate.stat().st_size < 8 * 1024**2:
            paths.append(candidate)
    return {str(path.relative_to(root)): digest(path) for path in sorted(set(paths))}


class Memory(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong)
        for name in ("total_phys", "avail_phys", "total_page", "avail_page", "total_virt", "avail_virt", "avail_ext")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 < args.timeout <= 60 or not args.command:
        raise ValueError("command and timeout in (0,60] required")
    root = Path(__file__).resolve().parents[2]
    if root.drive.upper() != "H:" or not args.label.replace("-", "").isalnum():
        raise ValueError("H-side checkout and simple evidence label required")
    destination = root / "_docs/perf131k/tests"
    destination.mkdir(parents=True, exist_ok=True)
    receipt = destination / (args.label + ".json")
    raw = destination / (args.label + ".txt")
    memory = Memory()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise OSError("memory telemetry unavailable; no test launched")
    data = {
        "utc_started": datetime.now(timezone.utc).isoformat(),
        "cwd": str(root),
        "input_sha256": inputs(root, args.command),
        "available_ram": memory.avail_phys,
        "commit_headroom": memory.avail_page,
        "h_free": shutil.disk_usage(root).free,
        "estimated_peak_bytes": 256 * 1024**2,
        "timeout_seconds": args.timeout,
        "command": args.command,
        "status": "BLOCKED",
    }
    allowed = memory.avail_phys >= 768 * 1024**2 and memory.avail_page >= 1024**3 and data["h_free"] >= 5 * 1024**3
    # Exclusive evidence publication: even a repeated label cannot destroy a prior RED.
    with receipt.open("x", encoding="utf-8") as record:
        if allowed:
            temp = root / "_scratch/perf131k"
            temp.mkdir(parents=True, exist_ok=True)
            env = dict(
                os.environ,
                TEMP=str(temp),
                TMP=str(temp),
                UV_CACHE_DIR=str(temp / "uv"),
                BLACK_CACHE_DIR=str(temp / "black"),
                PYTHONDONTWRITEBYTECODE="1",
                PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            )
            allowlist = (
                "TEMP",
                "TMP",
                "UV_CACHE_DIR",
                "BLACK_CACHE_DIR",
                "PYTHONDONTWRITEBYTECODE",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                "P01_ANALYZER_UNDER_TEST",
                "P01_FIXTURE_SERIES",
                "STRATA_PERF131K_HOST_TRACE",
            )
            data["environment"] = {key: env.get(key) for key in allowlist}
            started = time.monotonic()
            data.update(run_owned(args.command, root, env, raw, args.timeout))
            data["elapsed_seconds"] = time.monotonic() - started
            data["raw_sha256"] = digest(raw) if raw.exists() else None
            data["input_sha256_after"] = inputs(root, args.command)
            data["utc_finished"] = datetime.now(timezone.utc).isoformat()
        record.write(json.dumps(data, indent=2) + "\n")
    sys.stdout.write(
        json.dumps(
            {
                key: data.get(key)
                for key in (
                    "status",
                    "exit_code",
                    "available_ram",
                    "commit_headroom",
                    "h_free",
                    "contained_before_resume",
                    "termination_confirmed",
                    "elapsed_seconds",
                )
            }
        )
        + "\n"
    )
    return data.get("exit_code", 77)


if __name__ == "__main__":
    raise SystemExit(main())
