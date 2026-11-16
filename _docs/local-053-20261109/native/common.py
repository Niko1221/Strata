"""Native-only paths, frozen-source gate and bounded owned-command receipts."""
from __future__ import annotations

import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
BASE = Path("H:/Strata/releases/local-053-20261109-opencode")
BUILD = BASE / "native-build"
NATIVE = BASE / "native"
COMPONENT = BASE / "native-component/0.5.3-windows-x86_64-cuda-cuda13"
GGML = Path("H:/Strata/deps/ggml-3cf03257")
GGML_PIN = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
NODE = Path("C:/Users/downl/AppData/Local/Programs/Strata/node/node.exe")
PYTHON = Path("C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe")
UV = Path("C:/Users/downl/.local/bin/uv.exe")
CUDA = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3")
CMAKE = Path("C:/Program Files/CMake/bin/cmake.exe")
CTEST = CMAKE.with_name("ctest.exe")
GIB = 1024**3
PHASES = ("build", "probe-diagnostic", "bind", "stage", "cpu", "gpu", "sanitizer",
          "spill", "model-bounded", "model-legacy7", "model-restart", "compare", "audit")


def require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: Any, *, fresh: bool = False) -> None:
    """Atomic checkpoints; immutable phase reservation prevents competing writers."""
    require(path.is_relative_to(BASE), "Receipt must stay inside the owned output")
    require(not fresh or not path.exists(), f"Retain prior evidence: {path}")
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def git(root: Path, *args: str) -> str:
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    done = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                          text=True, timeout=30, env=env, creationflags=subprocess.CREATE_NO_WINDOW)
    require(len(done.stdout) < 2 * 1024**2, "Unexpectedly large Git response")
    return done.stdout.strip()


def no_links(path: Path) -> None:
    for part in (path, *path.parents):
        require(not part.is_symlink() and not part.is_junction(), f"Linked path refused: {part}")


def freeze(root: Path, commit: str) -> dict[str, Any]:
    require(root.is_absolute() and root.drive.upper() == "H:", "Explicit absolute H: source root required")
    require(re.fullmatch(r"[0-9a-f]{40}", commit), "Full reviewed commit required")
    no_links(root)
    require(root.is_dir() and Path(git(root, "rev-parse", "--show-toplevel")).resolve() == root,
            "Source root must be the exact checkout root")
    require(not root.is_relative_to(BASE) and not BASE.is_relative_to(root), "Source/output overlap refused")
    require(git(root, "rev-parse", "HEAD") == commit, "Reviewed source commit changed")
    require(not git(root, "status", "--porcelain", "--untracked-files=all"), "A clean frozen source checkout is required")
    require(re.search(r"project\(strata VERSION 0\.5\.3\s", (root / "CMakeLists.txt").read_text(encoding="utf-8")),
            "Parent must freeze native 0.5.3 first")
    no_links(GGML)
    require(git(GGML, "rev-parse", "HEAD") == GGML_PIN, "ggml pin mismatch")
    require(not git(GGML, "status", "--porcelain", "--untracked-files=all"), "ggml must be clean")
    helpers = ("tools/build_hakua_engine.mjs", "apps/host/scripts/release-integrity.mjs",
               "tools/perf131k/windows_job.py", "tools/native_model_resource_guard.py",
               "packaging/package_engine.py", "tools/check_cuda_spills.py")
    return {"sourceRoot": str(root), "sourceCommit": commit, "ggmlRoot": str(GGML),
            "ggmlCommit": GGML_PIN, "helpers": {name: sha(root / name) for name in helpers},
            "adapters": {p.name: sha(p) for p in sorted(HERE.iterdir()) if p.suffix in (".py", ".mjs")},
            "expectedVersions": {"product": "0.5.3-local.1", "control": "0.5.4", "native": "0.5.3"}}


def environment() -> dict[str, str]:
    # Deliberately do not inherit user tuning, credentials, Python hooks or compiler flags.
    keep = {"SYSTEMROOT", "WINDIR", "COMSPEC", "SYSTEMDRIVE", "PROGRAMFILES", "PROGRAMFILES(X86)",
            "PROGRAMW6432", "PROGRAMDATA", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "NUMBER_OF_PROCESSORS"}
    env = {k: v for k, v in os.environ.items() if k.upper() in keep}
    git_exe = shutil.which("git.exe")
    require(git_exe, "Existing Git required")
    env["PATH"] = ";".join(map(str, (NODE.parent, PYTHON.parent, CMAKE.parent, CUDA / "bin/x64",
                                      CUDA / "bin", Path(git_exe).parent, Path("C:/Windows/System32"),
                                      Path("C:/Windows"))))
    env.update(TMP=str(BASE / "temp"), TEMP=str(BASE / "temp"), TMPDIR=str(BASE / "temp"),
               UV_CACHE_DIR=str(BASE / "cache/uv"), UV_OFFLINE="1", UV_PYTHON_DOWNLOADS="never",
               PIP_NO_INDEX="1", PIP_CACHE_DIR=str(BASE / "cache/pip"),
               CUDA_CACHE_PATH=str(BASE / "cache/cuda"), XDG_CACHE_HOME=str(BASE / "cache"),
               PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                CUDA_PATH=str(CUDA), CUDA_PATH_V13_3=str(CUDA), CMAKE_BUILD_PARALLEL_LEVEL="4",
               CTEST_PARALLEL_LEVEL="1", MSBUILDDISABLENODEREUSE="1", CMAKE_BUILD_TYPE="Release")
    return env


@lru_cache(maxsize=1)
def load_helpers(root: Path) -> tuple[Any, Any]:
    sys.path[:0] = [str(root / "tools"), str(root)]
    from native_model_resource_guard import available_memory
    from perf131k.windows_job import run_owned
    return available_memory, run_owned


def resources(root: Path) -> dict[str, int]:
    memory, _ = load_helpers(root)
    value = memory()
    return {"available_ram": value["available_physical_bytes"],
            "commit_headroom": value["available_commit_bytes"],
            "H_free": shutil.disk_usage("H:/").free, "C_free": shutil.disk_usage("C:/").free}


def require_current_job(root: Path) -> None:
    """Reject internal entrypoints invoked outside a Windows Job; creates/stops no process."""
    import ctypes
    from ctypes import wintypes
    load_helpers(root)
    from perf131k.windows_job import api
    present = wintypes.BOOL()
    require(api().IsProcessInJob(wintypes.HANDLE(-1), None, ctypes.byref(present)) and present.value,
            "Internal worker/driver must already be contained in its owning Job")


def admit(root: Path, *, allow_low_ram: bool = False) -> dict[str, int]:
    require(type(allow_low_ram) is bool, "Explicit boolean RAM exception required")
    value = resources(root)
    for key, floor in (("available_ram", 0 if allow_low_ram else 3 * GIB), ("commit_headroom", 4 * GIB),
                       ("H_free", 10 * GIB), ("C_free", 5 * GIB)):
        require(value[key] >= floor, f"Admission refused: {key} below {floor}; coordinate with parent")
    return value


def passed_run(run: dict[str, Any]) -> bool:
    return (run.get("status") == "FINISHED" and run.get("native_exit_code") == 0
            and run.get("contained_before_resume") is True and run.get("termination_confirmed") is True
            and run.get("job_active_after") == 0 and run.get("observed_dropped_bytes") == 0)


def owned(root: Path, name: str, command: list[str], seconds: int,
          *, env: dict[str, str] | None = None, on_output: Any = None,
          output_bytes: int = 8 * 1024**2, allow_low_ram: bool = False) -> dict[str, Any]:
    require(type(allow_low_ram) is bool, "Explicit boolean RAM exception required")
    require(re.fullmatch(r"[a-zA-Z0-9_.-]+", name) and 0 < seconds <= 5400, "Invalid owned command budget")
    receipt = BASE / "receipts" / (name + ".json")
    require(not receipt.exists(), f"Retain command receipt: {receipt}")
    row = {"name": name, "command": command, "started": time.time(), "passed": False,
           "deadlineSeconds": seconds, "outputBytes": output_bytes}
    ram_policy = {"allowLowRam": allow_low_ram, "nominalFloorBytes": 3 * GIB,
                  "effectiveFloorBytes": 0 if allow_low_ram else 3 * GIB,
                  "authorization": "user-explicit-2026-11-16-operation-only" if allow_low_ram else None,
                  "samples": 0, "nominalBreachSamples": 0, "minAvailableBytes": None}
    row["ramPolicy"] = ram_policy
    def observe(value):
        available = value["available_ram"]
        ram_policy["samples"] += 1
        ram_policy["nominalBreachSamples"] += int(available < 3 * GIB)
        previous = ram_policy["minAvailableBytes"]
        ram_policy["minAvailableBytes"] = available if previous is None else min(previous, available)
        return value
    save(receipt, row, fresh=True)
    try:
        row["admission"] = observe(admit(root, allow_low_ram=allow_low_ram))
        _, run_owned = load_helpers(root)
        budget = SimpleNamespace(job_commit_bytes=None, max_processes=64,
                                 min_available_ram=ram_policy["effectiveFloorBytes"], min_commit_headroom=4 * GIB,
                                 min_H_free=10 * GIB, timeout_s=seconds, output_bytes=output_bytes)
        row["run"] = run_owned(command, BASE, env or environment(),
            BASE / "receipts" / (name + ".txt"), budget, lambda _: observe(resources(root)),
            memory_diagnostics=True, on_output=on_output,
            on_tick=lambda: "C_disk_floor" if shutil.disk_usage("C:/").free < 5 * GIB else None)
        row["passed"] = passed_run(row["run"])
    except BaseException as error:
        row["error"] = f"{type(error).__name__}: {error}"
    finally:
        row["finished"] = time.time()
        save(receipt, row)
    return row


def text(name: str) -> str:
    path = BASE / "receipts" / (name + ".txt")
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else "[No output; consult failed command receipt]"


def configure(root: Path) -> list[str]:
    return [str(CMAKE), "-S", str(root), "-B", str(BUILD), "-G", "Visual Studio 17 2022",
            "-A", "x64", "-T", "cuda=13.3",
            "-DCMAKE_GENERATOR_INSTANCE=C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools",
            "-DCMAKE_CUDA_ARCHITECTURES=120", "-DSTRATA_GGML_DIR=" + str(GGML),
            "-DPython3_EXECUTABLE=" + str(PYTHON), "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
            "-DSTRATA_PORTABLE=ON", "-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_NATIVE_EXPERTS=ON",
            "-DSTRATA_ENABLE_SPIN8=ON", "-DSTRATA_ENABLE_NATIVE_CACHE=ON", "-DSTRATA_NATIVE_CACHE_OWNER=ON",
            "-DSTRATA_MMQ_KQUANTS=ON", "-DSTRATA_MMQ_TQ_OVERLAY=ON", "-DSTRATA_BUILD_TESTS=ON",
            "-DSTRATA_BUILD_DSV41=OFF", "-DSTRATA_BUILD_SPLIT=OFF", "-DSTRATA_PREFILL_MMQ=OFF",
            "-DSTRATA_CUDA_SPILL_BASELINE="]
