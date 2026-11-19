"""One B2 CPU target in a private Job; snapshots and all output retained on H:."""
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
    dest = Path("H:/Strata/evidence/p01-b2-20261026/tests") / label
    dest.mkdir(parents=True, exist_ok=False)
    files = [root / p for p in ("tests/perf131k/test_pilot.py", "tests/perf131k/run_b2.py",
             "tests/perf131k/owned_job.py", "tests/perf131k/run_bounded.py",
             "tools/perf131k/pilot.py", "tools/perf131k/windows_job.py", "tools/perf131k/cpu_child.py",
             "_docs/perf131k/model-inventory.json", "tools/perf131k/native_pilot.py",
             "tests/perf131k/test_native_pilot.py", "tools/perf131k/service_pilot.py",
             "tests/perf131k/test_service_pilot.py")]
    files += [Path(os.environ[k]) for k in ("P01_PILOT_UNDER_TEST", "P01_JOB_UNDER_TEST", "P01_NATIVE_UNDER_TEST", "P01_SERVICE_UNDER_TEST") if os.environ.get(k)]
    if os.environ.get("P01_PILOT_UNDER_TEST"):
        files.append(Path(os.environ["P01_PILOT_UNDER_TEST"]).with_name("cpu_child.py"))
    files = [p for p in files if p.exists()]
    before = {str(p): digest(p) for p in files}
    for i, path in enumerate(files):
        if path.stat().st_size > 2 * 1024**2 or path.drive.upper() != "H:":
            raise ValueError("unexpected CPU source")
        shutil.copyfile(path, dest / f"input-{i:02d}-{path.name}")
    mem = Memory()
    mem.length = ctypes.sizeof(mem)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
        raise OSError("memory telemetry unavailable")
    receipt = {"started": datetime.now(timezone.utc).isoformat(), "input_sha256": before,
               "available_ram": mem.avail_phys, "commit_headroom": mem.avail_page,
               "H_free": shutil.disk_usage(root).free, "estimated_peak_bytes": 512 * 1024**2,
               "timeout_s": 60, "status": "BLOCKED", "exit_code": 77}
    if mem.avail_phys >= 1024**3 and mem.avail_page >= 2 * 1024**3 and receipt["H_free"] >= 5 * 1024**3:
        temp = root / "_scratch/perf131k/b2-fixtures"
        temp.mkdir(parents=True, exist_ok=True)
        prior_fixtures = {p.name for p in temp.iterdir()}
        env = dict(os.environ, TEMP=str(temp), TMP=str(temp), PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, "-B", "-m", "unittest", *sys.argv[2:]]
        receipt["command"] = command
        receipt["mutation_selection"] = {k: env.get(k) for k in ("P01_PILOT_UNDER_TEST", "P01_JOB_UNDER_TEST", "P01_NATIVE_UNDER_TEST", "P01_SERVICE_UNDER_TEST")}
        started = time.monotonic()
        receipt.update(run_owned(command, root, env, dest / "raw.txt", 60))
        receipt["elapsed_s"] = time.monotonic() - started
        receipt["raw_sha256"] = digest(dest / "raw.txt")
        artifacts = []
        for folder in temp.iterdir():
            if folder.name in prior_fixtures:
                continue
            for path in folder.rglob("*"):
                if path.is_file():
                    if not path.resolve().is_relative_to(temp.resolve()) or path.stat().st_size > 2 * 1024**2:
                        raise ValueError("unexpected fixture artifact")
                    artifacts.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)})
        receipt["fixture_artifacts"] = artifacts
    receipt["input_sha256_after"] = {str(p): digest(p) for p in files}
    if receipt["input_sha256_after"] != before:
        receipt.update(status="SOURCE_CHANGED", exit_code=78)
    with (dest / "receipt.json").open("x", encoding="utf-8") as out:
        json.dump(receipt, out, indent=2)
    print(json.dumps({k: receipt.get(k) for k in ("status", "exit_code", "elapsed_s", "job_active_after")}))
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
