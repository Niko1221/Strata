"""Private release acceptance GPU observer and replayable identity policy.

C+G includes compute contexts. Only an explicitly pinned observed desktop
baseline or creation-verified descendants of this attempt may share the GPU.
No process names, retries, baseline generation, or process control are used.
"""

from __future__ import annotations

from datetime import datetime
import ctypes
from ctypes import wintypes
import hashlib
import json
import ntpath
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

SCHEMA = "strata.gpu-admission.v1"
IDENTITY = ("pid", "parentPID", "created", "executable", "sha256")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def baseline(binding: dict | None) -> list[dict]:
    """Read the explicit private artifact and verify its original byte pin."""
    if binding is None:
        return []
    require(set(binding) == {"path", "sha256"}, "Invalid baseline binding")
    path = Path(binding["path"])
    require(
        path.is_absolute() and re.fullmatch(r"[a-f0-9]{64}", binding["sha256"]),
        "Invalid baseline pin",
    )
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == binding["sha256"], "Baseline changed")
    value = json.loads(data)
    require(
        value.get("schema") == "strata.observed-desktop-gpu-baseline.v1", "Wrong baseline schema"
    )
    rows = value["processes"]
    require(isinstance(rows, list) and rows, "Empty explicit desktop baseline")
    indexed(rows)
    for row in rows:
        identity(row)
        require(row.get("type") == "C+G", "Baseline is not the observed desktop class")
    return rows


def identity(row: dict) -> None:
    require(isinstance(row, dict) and all(k in row for k in IDENTITY), "Missing process identity")
    require(
        type(row["pid"]) is int
        and row["pid"] > 0
        and type(row["parentPID"]) is int
        and row["parentPID"] >= 0,
        "Invalid process PID",
    )
    require(
        isinstance(row["executable"], str)
        and ntpath.isabs(row["executable"])
        and re.fullmatch(r"[a-f0-9]{64}", row["sha256"] or ""),
        "Unobservable process executable",
    )
    require(
        isinstance(row["created"], str) and row["created"].endswith("Z"),
        "Unobservable process creation",
    )
    datetime.fromisoformat(row["created"].replace("Z", "+00:00"))


def same(left: dict, right: dict) -> bool:
    return all(left[k] == right[k] for k in IDENTITY) and (
        "creationFiletime" not in right or left.get("creationFiletime") == right["creationFiletime"]
    )


def indexed(rows: list[dict]) -> dict[int, dict]:
    require(isinstance(rows, list), "Missing process observations")
    result = {row["pid"]: row for row in rows}
    require(len(result) == len(rows), "Duplicate PID observation")
    return result


def classify(raw: dict, binding: dict | None, owner: dict, minimum: int) -> dict:
    """Recompute policy from XML, CIM identities and the pinned baseline."""
    require(minimum in (768, 8192), "Invalid GPU floor")
    identity(owner)
    require(type(owner.get("creationFiletime")) is int, "Missing native worker creation time")
    creation_filetime(owner)
    desktops = indexed(baseline(binding))
    processes = indexed(raw["processes"])
    require(
        owner["pid"] in processes and same(processes[owner["pid"]], owner),
        "Worker identity changed",
    )
    xml = ET.fromstring(raw["gpuXML"])
    gpus = xml.findall("gpu")
    require(len(gpus) == 1 and xml.findtext("attached_gpus") == "1", "Another GPU observed")
    gpu = gpus[0]
    free_text = gpu.findtext("fb_memory_usage/free", "")
    require(re.fullmatch(r"[0-9]+ MiB", free_text) is not None, "Missing XML free memory")
    require(
        isinstance(gpu.findtext("uuid"), str)
        and gpu.findtext("uuid").startswith("GPU-")
        and len(raw["gpuRows"]) == 1
        and raw["gpuRows"][0][:2] == [0, gpu.findtext("uuid")]
        and type(raw["gpuRows"][0][2]) is int
        and type(raw["freeMiB"]) is int
        and raw["freeMiB"] == min(raw["gpuRows"][0][2], int(free_text.split()[0]))
        and raw["freeMiB"] >= minimum,
        "GPU identity or free-memory floor differs",
    )
    rows = gpu.findall("processes/process_info")
    observed = [int(row.findtext("pid", "0")) for row in rows]
    require(
        len(set(observed)) == len(observed) and all(pid > 0 for pid in observed),
        "Invalid GPU process enumeration",
    )
    compute = sorted(
        int(row.findtext("pid")) for row in rows if row.findtext("type") in ("C", "C+G")
    )
    require(sorted(raw["computeQueryPIDs"]) == compute, "GPU process queries disagree")
    desktop_pids, owned_pids = [], []
    for xml_row in rows:
        pid = int(xml_row.findtext("pid"))
        kind = xml_row.findtext("type")
        require(kind in ("C", "G", "C+G"), "Unknown GPU process class")
        row = processes.get(pid, {})
        identity(row)
        creation_filetime(row)
        if pid in desktops:
            require(
                kind == desktops[pid]["type"] and same(row, desktops[pid]),
                "Pinned desktop identity changed",
            )
            desktop_pids.append(pid)
            continue
        # Each parent must still exist and predate its child, preventing a
        # recycled parent PID from proving ownership of an unrelated process.
        seen = set()
        while row["pid"] != owner["pid"]:
            require(row["pid"] not in seen and len(seen) < 32, "Invalid ownership chain")
            seen.add(row["pid"])
            parent = processes.get(row["parentPID"], {})
            identity(parent)
            require(creation_filetime(parent) <= creation_filetime(row), "Reused ancestry PID")
            row = parent
        require(same(row, owner), "Foreign GPU owner")
        owned_pids.append(pid)
    return {
        "schema": SCHEMA,
        "device": 0,
        "freeMiB": raw["freeMiB"],
        "reserveMiB": 768,
        "minimumFreeMiB": minimum,
        "baseline": binding,
        "owner": owner,
        "observedPIDs": sorted(observed),
        "computePIDs": compute,
        "desktopPIDs": sorted(desktop_pids),
        "ownedPIDs": sorted(owned_pids),
        "unapprovedPIDs": [],
        "sharedDesktopGPU": bool(binding),
        "performanceQualification": "UNQUALIFIED",
        "gpuConcurrency": "UNQUALIFIED",
        "raw": raw,
    }


def validate(snapshot: dict, binding: dict | None, owner: dict, minimum: int) -> None:
    require(isinstance(snapshot, dict) and "raw" in snapshot, "Missing GPU observation evidence")
    require(
        snapshot == classify(snapshot["raw"], binding, owner, minimum),
        "GPU classification or binding changed",
    )


def creation_filetime(owner: dict) -> int:
    """Bind native FILETIME; CIM truncates its last 100-nanosecond digit."""
    identity(owner)
    text = owner["created"]
    require(
        re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{7}Z", text),
        "Missing precise worker creation identity",
    )
    delta = datetime.fromisoformat(text[:-1]) - datetime(1601, 1, 1)
    cim_time = ((delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds) * 10 + int(
        text[-2]
    )
    native_time = owner.get("creationFiletime")
    require(
        type(native_time) is int and native_time > 0 and native_time // 10 == cim_time // 10,
        "Native/CIM creation identity differs",
    )
    return native_time


def native_creation(pid: int) -> int:
    """Read one process creation time with query-only Windows access."""
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenProcess.restype = wintypes.HANDLE
    api.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.c_void_p] * 4)]
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = api.OpenProcess(0x1000, False, pid)
    require(bool(handle), "Native process identity unavailable")
    try:
        times = [ctypes.c_ulonglong() for _ in range(4)]
        require(
            bool(api.GetProcessTimes(handle, *[ctypes.byref(t) for t in times])),
            "Native process creation unavailable",
        )
        return times[0].value
    finally:
        require(bool(api.CloseHandle(handle)), "Native process observation handle not closed")


def command(argv: list[str]) -> str:
    result = subprocess.run(
        argv,
        capture_output=True,
        timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    require(result.returncode == 0, "GPU/process observation failed")
    return result.stdout.decode("utf-8-sig", errors="strict").strip()


def cim() -> list[dict]:
    """Observe no command lines, environment, tokens, or process arguments."""
    script = """$ErrorActionPreference='Stop';
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false);
$rows=@(Get-CimInstance Win32_Process | ForEach-Object {
@{pid=[int]$_.ProcessId;parentPID=[int]$_.ParentProcessId;
created=$_.CreationDate.ToUniversalTime().ToString('o');executable=$_.ExecutablePath}});
ConvertTo-Json -InputObject $rows -Compress"""
    return json.loads(
        command(
            [
                "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ]
        )
    )


def observe(binding: dict | None = None, owner: dict | None = None, minimum: int = 8192) -> dict:
    """Read NVIDIA and stable CIM identities; never allocate or stop a process."""
    raw = {}
    try:
        return _observe(binding, owner, minimum, raw)
    except Exception as error:
        # Preserve partial observation even when an identity disappears mid-read.
        # Callers keep this only in the private attempt; the original error wins.
        error.gpu_observation = {"raw": raw, "baseline": binding, "owner": owner}
        raise


def _observe(binding: dict | None, owner: dict | None, minimum: int, raw: dict) -> dict:
    desktop_pids = {row["pid"] for row in baseline(binding)}
    exe = "C:/Windows/System32/nvidia-smi.exe"
    gpu_rows = [
        [int(a), b.strip(), int(c)]
        for a, b, c in (
            line.split(",")
            for line in command(
                [exe, "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"]
            ).splitlines()
        )
    ]
    raw["gpuRows"] = gpu_rows
    require(len(gpu_rows) == 1, "Another GPU observed")
    xml = command([exe, "-q", "-x"])
    raw["gpuXML"] = xml
    compute = command([exe, "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    raw["computeQueryPIDs"] = [int(line) for line in compute.splitlines() if line]
    pids = [
        int(row.findtext("pid")) for row in ET.fromstring(xml).findall("gpu/processes/process_info")
    ]
    all_rows = indexed(cim())
    owner_pid = owner["pid"] if owner else os.getpid()
    needed = {owner_pid, *pids}
    for pid in pids:
        seen = set()
        while pid in all_rows and pid != owner_pid and pid not in seen and len(seen) < 32:
            seen.add(pid)
            # Pinned desktop rows need no ancestry outside the captured identity.
            if pid in desktop_pids:
                break
            pid = all_rows[pid]["parentPID"]
            needed.add(pid)
    observations = []
    raw["processes"] = observations
    for pid in sorted(needed):
        require(pid in all_rows, "GPU process identity unavailable")
        row = dict(all_rows[pid])
        require(isinstance(row.get("executable"), str), "GPU executable unavailable")
        observations.append(row)
        row["creationFiletime"] = native_creation(pid)
        row["sha256"] = digest(Path(row["executable"]))
        identity(row)
        creation_filetime(row)
    after = indexed(cim())
    require(
        all(
            pid in after
            and all(all_rows[pid][k] == after[pid][k] for k in IDENTITY if k != "sha256")
            for pid in needed
        ),
        "Process identity changed during observation",
    )
    require(
        all(native_creation(row["pid"]) == row["creationFiletime"] for row in observations),
        "Native process identity changed during observation",
    )
    if owner is None:
        owner = next(row for row in observations if row["pid"] == owner_pid)
    free_text = ET.fromstring(xml).findtext("gpu/fb_memory_usage/free", "")
    require(re.fullmatch(r"[0-9]+ MiB", free_text) is not None, "Missing XML free memory")
    raw["freeMiB"] = min(gpu_rows[0][2], int(free_text.split()[0]))
    return classify(raw, binding, owner, minimum)


if __name__ == "__main__":
    request = json.load(sys.stdin)
    try:
        snapshot = observe(request["baseline"], request["owner"], request["minimumFreeMiB"])
    except Exception as error:
        json.dump(
            {"error": type(error).__name__, "observation": getattr(error, "gpu_observation", None)},
            sys.stdout,
        )
        sys.exit(1)
    json.dump(snapshot, sys.stdout)
