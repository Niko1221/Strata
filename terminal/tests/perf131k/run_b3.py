"""One CPU diagnostics gate; immutable source snapshots/raw receipts outside Git on H:."""
import json
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from tools.perf131k.pilot import system_resources
from tools.perf131k.windows_job import run_owned


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    label, *targets = sys.argv[1:]
    if not label.replace("-", "").isalnum() or not targets:
        raise ValueError("unique label and explicit unittest targets required")
    root = Path(__file__).resolve().parents[2]
    dest = Path("H:/Strata/evidence/p01-b3-diagnostics-20261026/tests") / label
    dest.mkdir(parents=True, exist_ok=False)
    paths = [root / p for p in (
        "src/core/expert_cache.cpp", "src/core/expert_source.cpp", "src/program/generate.cpp",
        "tools/perf131k/windows_job.py", "tools/perf131k/native_pilot.py",
        "tools/perf131k/pilot.py", "tests/perf131k/test_native_pilot.py", "tests/perf131k/test_pilot.py",
        "tests/perf131k/test_diagnostics.py", "tests/perf131k/diagnostics_cpu.cpp",
        "tests/perf131k/test_exit_status.py",
        "include/strata/perf131k/blob_failure.hpp", "tests/perf131k/test_blob_failure.py",
        "tests/perf131k/blob_failure_cpu.cpp",
        "tests/perf131k/run_b3.py")]
    for key in ("P01_JOB_UNDER_TEST", "P01_NATIVE_UNDER_TEST", "P01_CACHE_UNDER_TEST", "P01_GENERATE_UNDER_TEST", "P01_BLOB_SOURCE_UNDER_TEST"):
        if os.environ.get(key):
            paths.append(Path(os.environ[key]))
    if os.environ.get("P01_BLOB_HEADER_ROOT"):
        paths.append(Path(os.environ["P01_BLOB_HEADER_ROOT"]) / "strata/perf131k/blob_failure.hpp")
    before = {}
    for i, path in enumerate(paths):
        if path.drive.upper() != "H:" or path.stat().st_size > 2 * 1024**2:
            raise ValueError("bounded H source required")
        before[str(path)] = digest(path)
        shutil.copyfile(path, dest / f"input-{i:02d}-{path.name}")
    resources = system_resources(root)
    fixture_root = root / "_scratch/perf131k/b2-fixtures"
    previous_fixtures = {p.name for p in fixture_root.iterdir()} if fixture_root.exists() else set()
    result = {"status": "BLOCKED", "exit_code": 77, "input_sha256": before,
              "resources": resources, "GPU_model": "NOT_RUN"}
    blob_only = all(t.startswith("tests.perf131k.test_blob_failure") for t in targets)
    output_cap = 32 * 1024**2 if blob_only else None
    disk_admission = 5 * 1024**3
    result["disk_budget"] = {"admission_bytes": disk_admission, "running_floor_bytes": 4 * 1024**3,
                            "owned_output_cap_bytes": output_cap, "scope": "blob_CPU_only" if blob_only else "existing_CPU_gate"}
    if (resources["available_ram"] >= 1024**3 and resources["commit_headroom"] >= 3 * 1024**3
            and resources["H_free"] >= disk_admission):
        temp = dest / "tmp"
        temp.mkdir()
        env = dict(os.environ, TEMP=str(temp), TMP=str(temp), PYTHONDONTWRITEBYTECODE="1",
                   P01_DIAGNOSTICS_ARTIFACTS=str(dest))
        budget = SimpleNamespace(timeout_s=60, job_commit_bytes=2 * 1024**3, output_bytes=2 * 1024**2,
                                 max_processes=6, min_available_ram=512 * 1024**2,
                                 min_commit_headroom=1024**3, min_H_free=4 * 1024**3)
        command = [sys.executable, "-B", "-m", "unittest", *targets]
        result.update(command=command, budget=vars(budget), estimated_peak_bytes=512 * 1024**2)
        started = time.monotonic()
        def sample(path):
            if output_cap is not None and sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()) > output_cap:
                raise OSError("owned CPU output budget exceeded")
            return system_resources(path)
        result.update(run_owned(command, root, env, dest / "raw.txt", budget, sample))
        result["elapsed_s"] = time.monotonic() - started
        result["raw_sha256"] = digest(dest / "raw.txt")
        artifacts = list(dest.rglob("*"))
        if fixture_root.exists():
            for folder in fixture_root.iterdir():
                if folder.name not in previous_fixtures:
                    artifacts.extend(folder.rglob("*"))
        result["artifacts"] = []
        for path in artifacts:
            if path.is_file():
                if (not path.resolve().is_relative_to(dest.resolve())
                        and not path.resolve().is_relative_to(fixture_root.resolve())):
                    raise ValueError("artifact outside owned evidence/fixtures")
                if len(result["artifacts"]) >= 2048 or path.stat().st_size > 2 * 1024**2:
                    raise ValueError("artifact inventory limit exceeded")
                result["artifacts"].append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)})
    result["input_sha256_after"] = {str(p): digest(p) for p in paths}
    if before != result["input_sha256_after"]:
        result.update(status="SOURCE_CHANGED", exit_code=78)
    (dest / "receipt.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result.get(k) for k in ("status", "exit_code", "elapsed_s", "job_active_after")}))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
