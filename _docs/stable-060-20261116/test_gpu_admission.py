"""CPU-only GPU admission policy tests; never run a model or NVIDIA command."""

import copy
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path

import pytest

HERE = Path(__file__).absolute().parent
SPEC = importlib.util.spec_from_file_location("gpu_policy_test", HERE / "gpu-admission.py")
gpu = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gpu)


def process(pid: int, parent: int = 1, second: int = 1) -> dict:
    delta = datetime(2026, 11, 16, 13, 5, second, 700961) - datetime(1601, 1, 1)
    return {
        "pid": pid,
        "parentPID": parent,
        "created": f"2026-11-16T13:05:{second:02d}.7009610Z",
        "executable": f"C:\\Windows\\p{pid}.exe",
        "sha256": "a" * 64,
        "creationFiletime": ((delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds)
        * 10,
    }


def fixture(tmp_path: Path, desktop: bool = True) -> tuple:
    owner, display = process(100), process(200)
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "schema": "strata.observed-desktop-gpu-baseline.v1",
                "processes": [{**display, "type": "C+G"}],
            }
        )
    )
    binding = {"path": str(path), "sha256": gpu.digest(path)} if desktop else None
    raw = {
        "gpuRows": [[0, "GPU-test", 14811]],
        "freeMiB": 14811,
        "gpuXML": "<nvidia_smi_log><attached_gpus>1</attached_gpus><gpu><uuid>GPU-test</uuid>"
        "<fb_memory_usage><free>14811 MiB</free></fb_memory_usage>"
        "<processes>"
        + (
            "<process_info><pid>200</pid><type>C+G</type>"
            "<used_memory>N/A</used_memory></process_info>"
            if desktop
            else ""
        )
        + "</processes></gpu></nvidia_smi_log>",
        "computeQueryPIDs": [200] if desktop else [],
        "processes": [owner, display],
    }
    return raw, binding, owner


def test_baseline_unchanged_disappeared_and_no_baseline(tmp_path: Path) -> None:
    raw, binding, owner = fixture(tmp_path)
    snapshot = gpu.classify(raw, binding, owner, 8192)
    assert snapshot["observedPIDs"] == snapshot["desktopPIDs"] == snapshot["computePIDs"] == [200]
    assert snapshot["ownedPIDs"] == snapshot["unapprovedPIDs"] == []
    assert snapshot["sharedDesktopGPU"] and snapshot["gpuConcurrency"] == "UNQUALIFIED"
    gpu.validate(snapshot, binding, owner, 8192)
    raw["gpuXML"] = raw["gpuXML"].replace(
        "<process_info><pid>200</pid><type>C+G</type>"
        "<used_memory>N/A</used_memory></process_info>",
        "",
    )
    raw["computeQueryPIDs"] = []
    assert gpu.classify(raw, binding, owner, 8192)["desktopPIDs"] == []
    assert gpu.classify(raw, None, owner, 8192)["computePIDs"] == []


@pytest.mark.parametrize(
    "damage",
    [
        "pid_reuse",
        "path",
        "hash",
        "created",
        "pure_compute",
        "unknown_cg",
        "unknown_compute",
        "missing",
        "baseline_tamper",
        "gpu",
        "vram",
    ],
)
def test_reject_unapproved_or_changed_identity(tmp_path: Path, damage: str) -> None:
    raw, binding, owner = fixture(tmp_path)
    display = raw["processes"][1]
    if damage in ("pid_reuse", "created"):
        display["created"] = "2026-11-17T13:05:01.7009610Z"
    elif damage == "path":
        display["executable"] = "C:\\Windows\\other.exe"
    elif damage == "hash":
        display["sha256"] = "b" * 64
    elif damage == "pure_compute":
        raw["gpuXML"] = raw["gpuXML"].replace("C+G", "C")
    elif damage in ("unknown_cg", "unknown_compute"):
        raw["gpuXML"] = raw["gpuXML"].replace("200", "201")
        raw["computeQueryPIDs"] = [201]
        display["pid"] = 201
        if damage == "unknown_compute":
            raw["gpuXML"] = raw["gpuXML"].replace("C+G", "C")
    elif damage == "missing":
        display["executable"] = None
    elif damage == "baseline_tamper":
        Path(binding["path"]).write_text("{}")
    elif damage == "gpu":
        raw["gpuRows"].append([1, "GPU-other", 14811])
    else:
        raw["freeMiB"] = raw["gpuRows"][0][2] = 8191
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        gpu.classify(raw, binding, owner, 8192)


def test_owned_descendant_requires_live_creation_chain(tmp_path: Path) -> None:
    raw, binding, owner = fixture(tmp_path, desktop=False)
    raw["processes"] = [owner, process(300, 100, 2), process(400, 300, 3)]
    raw["gpuXML"] = raw["gpuXML"].replace(
        "<processes>", "<processes>" "<process_info><pid>400</pid><type>C</type></process_info>"
    )
    raw["computeQueryPIDs"] = [400]
    assert gpu.classify(raw, binding, owner, 8192)["ownedPIDs"] == [400]
    raw["processes"][1]["created"] = "2026-11-17T13:05:01.7009610Z"
    raw["processes"][1]["creationFiletime"] += (86400 - 1) * 10000000
    with pytest.raises(RuntimeError, match="Reused ancestry"):
        gpu.classify(raw, binding, owner, 8192)


@pytest.mark.parametrize(
    "damage", ["empty_compute", "empty_all", "missing_raw", "owner", "binding"]
)
def test_receipt_replays_raw_evidence(tmp_path: Path, damage: str) -> None:
    raw, binding, owner = fixture(tmp_path)
    snapshot = copy.deepcopy(gpu.classify(raw, binding, owner, 8192))
    if damage == "empty_compute":
        snapshot["computePIDs"] = []
    elif damage == "empty_all":
        snapshot["observedPIDs"] = []
    elif damage == "missing_raw":
        del snapshot["raw"]
    elif damage == "owner":
        snapshot["owner"]["sha256"] = "c" * 64
    else:
        snapshot["baseline"] = None
    with pytest.raises(RuntimeError):
        gpu.validate(snapshot, binding, owner, 8192)


@pytest.mark.parametrize("damage", [None, "reused_pid", "missing_identity", "query_disagreement"])
def test_observer_cim_xml_boundary(tmp_path: Path, monkeypatch, damage: str | None) -> None:
    raw, binding, owner = fixture(tmp_path)
    before = [{k: v for k, v in row.items() if k != "sha256"} for row in raw["processes"]]
    after = copy.deepcopy(before)
    if damage == "reused_pid":
        after[1]["created"] = "2026-11-17T13:05:01.7009610Z"
    elif damage == "missing_identity":
        before[1]["executable"] = None
    calls = iter([before, after])
    monkeypatch.setattr(gpu, "cim", lambda: next(calls))
    monkeypatch.setattr(gpu, "digest", lambda path: "a" * 64)
    monkeypatch.setattr(gpu, "native_creation", lambda pid: process(pid)["creationFiletime"])

    def command(argv: list[str]) -> str:
        if "-x" in argv:
            return raw["gpuXML"]
        if "--query-compute-apps=pid" in argv:
            return "201" if damage == "query_disagreement" else "200"
        return "0, GPU-test, 14811"

    monkeypatch.setattr(gpu, "command", command)
    if damage:
        with pytest.raises(RuntimeError):
            gpu.observe(binding, owner)
    else:
        assert gpu.observe(binding, owner)["desktopPIDs"] == [200]


def producer_module():
    spec = importlib.util.spec_from_file_location("gpu_producer_test", HERE / "real-model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_cli_requires_both_absolute_pins(tmp_path: Path) -> None:
    producer = producer_module()
    for argv in (
        ["--gpu-desktop-baseline", str(tmp_path / "base.json")],
        ["--gpu-desktop-baseline-sha256", "a" * 64],
        ["--gpu-desktop-baseline", "relative.json", "--gpu-desktop-baseline-sha256", "a" * 64],
    ):
        with pytest.raises(RuntimeError):
            producer.parsed(argv)
    args = producer.parsed(
        [
            "--gpu-desktop-baseline",
            str(tmp_path / "base.json"),
            "--gpu-desktop-baseline-sha256",
            "a" * 64,
        ]
    )
    assert producer.baseline_binding(args)["sha256"] == "a" * 64
    assert producer.baseline_binding(producer.parsed([])) is None


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "before_load",
        "before_reload",
        "runtime_owner",
        "runtime_hash",
        "runtime_pids",
        "runtime_reserve",
    ],
)
def test_producer_receipt_gpu_binding(tmp_path: Path, damage: str | None) -> None:
    spec = importlib.util.spec_from_file_location(
        "gpu_receipt_fixture", HERE / "test_real_model.py"
    )
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    report, contract = fixture_module.synthetic_observations(tmp_path)
    if damage == "before_load":
        del report["gpuBeforeLoad"]["raw"]
    elif damage == "before_reload":
        report["gpuBeforeReload"]["baseline"] = {"path": "changed"}
    elif damage == "runtime_owner":
        report["runtime"][0]["gpuObservation"]["ownedPIDs"] = []
    elif damage == "runtime_hash":
        report["runtime"][0]["executableSHA256"] = "f" * 64
    elif damage == "runtime_pids":
        report["runtime"][0]["gpuPIDs"] = []
    elif damage == "runtime_reserve":
        report["runtime"][0]["gpuObservation"]["freeMiB"] = 767
    if damage:
        with pytest.raises(RuntimeError):
            fixture_module.producer.validate_gpu_observations(report, contract)
    else:
        fixture_module.producer.validate_gpu_observations(report, contract)


def test_creation_identity_keeps_seventh_fractional_digit() -> None:
    owner = process(100)
    assert gpu.creation_filetime(owner) % 10 == 0
    owner["created"] = owner["created"].replace("7009610Z", "7009617Z")
    owner["creationFiletime"] += 7
    assert gpu.creation_filetime(owner) % 10 == 7


@pytest.mark.parametrize("free", [768, 767])
def test_runtime_reserve_floor(tmp_path: Path, free: int) -> None:
    raw, binding, owner = fixture(tmp_path, desktop=False)
    raw["freeMiB"] = raw["gpuRows"][0][2] = free
    raw["gpuXML"] = raw["gpuXML"].replace("14811 MiB", f"{free} MiB")
    if free == 768:
        assert gpu.classify(raw, binding, owner, 768)["freeMiB"] == free
    else:
        with pytest.raises(RuntimeError, match="floor"):
            gpu.classify(raw, binding, owner, 768)


@pytest.mark.skipif(os.name != "nt", reason="Windows CIM and process FILETIME contract")
def test_actual_cim_worker_creation_matches_windows_filetime(monkeypatch) -> None:
    """CPU-only current-process identity, no NVIDIA query or GPU allocation."""
    import ctypes
    from ctypes import wintypes

    monkeypatch.setenv("PATH", "C:/Windows/System32;C:/Windows")
    creation, exit_time, kernel, user = [ctypes.c_ulonglong() for _ in range(4)]
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.c_void_p] * 4)]
    assert api.GetProcessTimes(
        wintypes.HANDLE(-1),
        ctypes.byref(creation),
        ctypes.byref(exit_time),
        ctypes.byref(kernel),
        ctypes.byref(user),
    )
    row = next(p for p in gpu.cim() if p["pid"] == os.getpid())
    row["sha256"] = gpu.digest(Path(row["executable"]))
    row["creationFiletime"] = gpu.native_creation(os.getpid())
    assert gpu.creation_filetime(row) == creation.value
