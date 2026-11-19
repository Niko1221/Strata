"""One bounded B1 CPU job; retain input bytes, fixtures, raw output and receipts on H:."""
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
from run_bounded import Memory


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    label = sys.argv[1]
    if not label.replace("-", "").isalnum():
        raise ValueError("simple unique label required")
    root = Path(__file__).resolve().parents[2]
    destination = Path("H:/Strata/evidence/p01-b1-20261026/tests") / label
    destination.mkdir(parents=True, exist_ok=False)
    paths = [root / p for p in ("serve/server.py", "serve/perf131k_e2e.py",
             "tests/perf131k/test_e2e_consumer.py", "tests/perf131k/run_b1.py",
             "tests/perf131k/owned_job.py", "tests/perf131k/run_bounded.py")]
    mutant = os.environ.get("P01_E2E_UNDER_TEST")
    if mutant:
        paths.append(Path(mutant))
    server_mutant = os.environ.get("P01_E2E_SERVER_UNDER_TEST")
    if server_mutant:
        paths.append(Path(server_mutant))
    paths.append(Path("H:/Strata/evidence/p01-b1-20261026/freeze/server.py"))
    before = {str(p): digest(p) for p in paths}
    for i, p in enumerate(paths):
        if p.stat().st_size > 2 * 1024**2 or p.drive.upper() != "H:":
            raise ValueError("unexpected CPU source input")
        shutil.copyfile(p, destination / f"input-{i:02d}-{p.name}")
    memory = Memory()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise OSError("memory telemetry unavailable")
    data = {"utc_started": datetime.now(timezone.utc).isoformat(), "input_sha256": before,
            "cwd": str(root), "available_ram": memory.avail_phys,
            "commit_headroom": memory.avail_page, "H_free": shutil.disk_usage(root).free,
            "timeout_seconds": 60, "estimated_peak_bytes": 256 * 1024**2, "status": "BLOCKED"}
    if memory.avail_phys >= 768 * 1024**2 and memory.avail_page >= 1024**3 and data["H_free"] >= 5 * 1024**3:
        fixture_root = root / "_scratch/perf131k/b1-fixtures"
        fixture_root.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, TEMP=str(fixture_root), TMP=str(fixture_root),
                   PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        data["environment"] = {k: env.get(k) for k in ("TEMP", "TMP", "PYTHONDONTWRITEBYTECODE",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "P01_E2E_UNDER_TEST", "P01_E2E_SERVER_UNDER_TEST")}
        command = [sys.executable, "-B", "-m", "unittest", *sys.argv[2:]]
        data["command"] = command
        started = time.monotonic()
        data.update(run_owned(command, root, env, destination / "raw.txt", 60))
        data["elapsed_seconds"] = time.monotonic() - started
        data["raw_sha256"] = digest(destination / "raw.txt")
        data["input_sha256_after"] = {str(p): digest(p) for p in paths}
        if before != data["input_sha256_after"]:
            data.update(status="SOURCE_CHANGED", exit_code=78)
    data["utc_finished"] = datetime.now(timezone.utc).isoformat()
    with (destination / "receipt.json").open("x", encoding="utf-8") as out:
        json.dump(data, out, indent=2)
    print(json.dumps({k: data.get(k) for k in ("status", "exit_code", "elapsed_seconds", "job_active_after")}))
    return data.get("exit_code", 77)


if __name__ == "__main__":
    raise SystemExit(main())
