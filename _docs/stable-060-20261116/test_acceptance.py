"""Boundary tests and read-only frozen captures; never execute production phases."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("acceptance_test_target", HERE / "acceptance.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
COMMIT = "a" * 40
CAPTURE = Path("H:/Strata/private-backups/cleanup-20261116/native-c775-profile-v1")
GPU_CAPTURE = {
    "gpu_binding": Path("H:/Strata/private-backups/cleanup-20261116/gpu-c775-v1/binding.json"),
    "gpu_report": Path("H:/Strata/evidence/upstream-sync-20261116/gpu-c775-v1.json"),
    "gpu_guard": Path("H:/Strata/evidence/upstream-sync-20261116/receipts/gpu-c775-v1.json"),
    "gpu_junit": Path("H:/Strata/evidence/upstream-sync-20261116/gpu-c775-v1.xml"),
}
MEASURED_SOURCE = "1c7811ccd9ebe8a853469b468ea196757a6d95332fe030b3e5f8d1512aedeaef"
MEASURED_BINARIES = {
    "strata.exe": "94e13464087b8944665dc226d798df4942cb74199022fed20947101b53d7f99c",
    "strata-q36.exe": "51d4a189cb6ab53641008a6ecbecd89af6f0a37ec33d45da3515bf4779a77b60",
}


def test_selected_profile_is_complete_actual_capture() -> None:
    """The reviewed capture includes a newly measured DLL, with no resource growth."""
    profile = HERE.parents[1] / a.PROFILE
    expected = "45c1fdb68fbe506c2790643c3295d59aa01f160cd1976ef9031469cd6ce028e5"
    a.artifact(profile, expected)
    a.artifact(CAPTURE / "candidate-measurements.json", expected)
    assert profile.read_bytes() == (CAPTURE / "candidate-measurements.json").read_bytes()
    comparison = a.release.load(CAPTURE / "comparison.json")
    a.artifact(
        CAPTURE / "comparison.json",
        "fa9154e1f6c2d8738a52956c297ade1f6e08a36d2200e36176cf6b227e0dbfaf",
    )
    assert comparison["measuredEntries"] == 451
    assert (
        comparison["addedEntries"]
        == comparison["missingEntries"]
        == comparison["changedAmounts"]
        == []
    )
    measured = a.release.module(
        "capture_measured", HERE.parents[1] / "tools/cuda_spill_baseline.py"
    )
    bindings = measured.bindings(
        a.native.PRIMARY,
        a.native.BUILD / "cuda-spill-build-key.txt",
        a.native.PRIMARY / "third_party/strata-ggml",
    )
    assert bindings == comparison["bindings"]
    assert len(measured.load(profile, bindings)) == 451
    libraries = a.compiled_libraries(a.release.load(profile), a.native.BUILD / "Release")
    assert (
        libraries["strata_kq.dll"]["sha256"]
        == "7f42796411387ef9d7d0d9cca8000bf3009bb85990a9f21633c6b10e6bab2bf8"
    )


@pytest.mark.parametrize("field", ("sourceDigest", "compilerKey", "ggmlCommit"))
def test_selected_profile_rejects_changed_identity(field: str) -> None:
    measured = a.release.module(
        "changed_capture_measured", HERE.parents[1] / "tools/cuda_spill_baseline.py"
    )
    profile = HERE.parents[1] / a.PROFILE
    bindings = a.release.load(profile)["bindings"]
    bindings[field] = "changed"
    with pytest.raises(ValueError, match="identity changed"):
        measured.load(profile, bindings)


@pytest.mark.parametrize("change", ("none", "stack", "local", "unlisted"))
def test_selected_profile_rejects_resource_growth(change: str) -> None:
    """Use an actual recorded entry with the production per-entry checker."""
    from tools import check_cuda_spills as checker
    from tools import cuda_spill_baseline as measured

    profile = HERE.parents[1] / a.PROFILE
    entries = measured.load(profile, a.release.load(profile)["bindings"])
    key, (stack, local) = next(iter(entries.items()))
    library, obj, sm, function = key.split("|", 3)
    if change == "stack":
        stack += 1
    elif change == "local":
        local += 1
    elif change == "unlisted":
        function += "_unlisted"
    rows = [(obj, sm, function, {"STACK": stack, "LOCAL": local})]
    names = {function: "strata::kernels::capture_boundary_kernel()"}
    assert checker.check_measured(library, rows, names, entries) == (0 if change == "none" else 1)


def frozen_gpu() -> tuple[dict, dict, dict, dict]:
    """Read immutable already-executed GPU proof; never launch GPU work."""
    for name, path in GPU_CAPTURE.items():
        a.artifact(path, a.GPU_PINS[name])
    a.validate_junit(GPU_CAPTURE["gpu_junit"])
    for name, digest in MEASURED_BINARIES.items():
        a.artifact(a.native.BUILD / "Release" / name, digest)
    return (
        *(a.release.load(GPU_CAPTURE[name]) for name in ("gpu_binding", "gpu_report", "gpu_guard")),
        dict(MEASURED_BINARIES),
    )


@pytest.mark.parametrize(
    "change",
    (
        "none",
        "old-source",
        "strata.exe",
        "strata-q36.exe",
        "failed",
        "undrained",
        "missing-job",
        "missing-samples",
        "ram-admission",
        "ram-observed",
        "commit",
        "H-disk",
        "C-disk",
        "vram",
    ),
)
def test_frozen_gpu_capture_and_fail_closed_boundaries(change: str) -> None:
    binding, report, guard, binaries = frozen_gpu()
    if change == "old-source":
        binding["sourceDigest"] = "8bc0448a35e217052eb0537285c7e38e724f0422a02f33f30554d2fb07d6a848"
    elif change in binaries:
        binaries[change] = "f" * 64
    elif change == "failed":
        guard["passed"] = False
    elif change == "undrained":
        guard["run"]["job_active_after"] = 1
    elif change == "missing-job":
        guard.pop("run")
    elif change == "missing-samples":
        guard["run"]["memory_diagnostics"]["samples"] = 0
    elif change == "ram-admission":
        guard["admission"]["available_ram"] = 3 * a.release.GIB - 1
    elif change == "ram-observed":
        # This capture's self-described exception has no authenticated outer flag.
        guard["run"]["memory_diagnostics"]["min_available_ram"] = 3 * a.release.GIB - 1
    elif change == "commit":
        guard["run"]["memory_diagnostics"]["min_commit_headroom"] = 4 * a.release.GIB - 1
    elif change == "H-disk":
        guard["run"]["memory_diagnostics"]["min_H_free"] = 10 * a.release.GIB - 1
    elif change == "C-disk":
        guard["run"]["last_resources"]["C_free"] = 5 * a.release.GIB - 1
    elif change == "vram":
        report["minimumFreeMiB"] = 767
    if change == "none":
        a.validate_gpu(binding, report, guard, MEASURED_SOURCE, binaries)
        assert binding["reportSHA256"] == a.GPU_PINS["gpu_report"]
        assert report["command"][report["command"].index("--parallel") + 1] == "1"
        assert guard["deadlineSeconds"] == 900
    else:
        with pytest.raises(RuntimeError):
            a.validate_gpu(binding, report, guard, MEASURED_SOURCE, binaries)


@pytest.mark.parametrize("name", tuple(GPU_CAPTURE))
def test_frozen_gpu_input_sha_rejects_changed_bytes(tmp_path: Path, name: str) -> None:
    original = GPU_CAPTURE[name]
    a.artifact(original, a.GPU_PINS[name])
    changed = tmp_path / original.name
    changed.write_bytes(original.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="digest"):
        a.artifact(changed, a.GPU_PINS[name])


def job() -> dict:
    """A complete synthetic successful Job record, never production evidence."""
    return {
        "status": "FINISHED",
        "exit_code": 0,
        "native_exit_code": 0,
        "contained_before_resume": True,
        "termination_confirmed": True,
        "job_active_after": 0,
        "observed_dropped_bytes": 0,
        "owned_root_pid": 1,
        "resource_samples": 2,
        "last_resources": {"C_free": 6 * a.release.GIB},
        "memory_diagnostics": {
            "status": "AVAILABLE",
            "samples": 2,
            "min_available_ram": 4 * a.release.GIB,
            "min_commit_headroom": 5 * a.release.GIB,
            "min_H_free": 11 * a.release.GIB,
        },
    }


def ci_run(workflow: str = "check.yml") -> dict:
    """Synthetic exact job multiset from the reviewed current workflow contract."""
    name, _, names = a.WORKFLOWS[workflow]
    return {
        "workflowName": name,
        "headSha": COMMIT,
        "headBranch": "main",
        "status": "completed",
        "conclusion": "success",
        "databaseId": 1,
        "jobs": [
            {"databaseId": i + 2, "name": n, "status": "completed", "conclusion": "success"}
            for i, n in enumerate(names)
        ],
    }


@pytest.mark.parametrize("workflow", a.WORKFLOWS)
def test_expanded_ci_contract(workflow: str) -> None:
    a.validate_ci_run(ci_run(workflow), workflow, COMMIT)
    file = a.native.PRIMARY / ".github/workflows" / workflow
    assert a.release.sha(file) == a.WORKFLOWS[workflow][1]


@pytest.mark.parametrize(
    "change",
    ("source", "branch", "workflow", "skipped", "cancelled", "missing", "extra", "duplicate", "job-id"),
)
def test_ci_rejects_incomplete_or_foreign_metadata(change: str) -> None:
    row = ci_run()
    if change == "source":
        row["headSha"] = "b" * 40
    elif change == "branch":
        row["headBranch"] = "other"
    elif change == "workflow":
        row["workflowName"] = "Intake"
    elif change in ("skipped", "cancelled"):
        row["jobs"][0]["conclusion"] = change
    elif change == "missing":
        row["jobs"].pop()
    elif change == "extra":
        row["jobs"].append({"name": "unexpected"})
    elif change == "duplicate":
        row["jobs"][-1] = row["jobs"][0]
    else:
        row["jobs"][1]["databaseId"] = row["jobs"][0]["databaseId"]
    with pytest.raises(RuntimeError):
        a.validate_ci_run(row, "check.yml", COMMIT)


def test_gh_reader_explicit_repo_and_suppressed_auth_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=b"[]", stderr=b"private diagnostic")

    monkeypatch.setattr(a.subprocess, "run", fake_run)
    data, command = a.gh_json(["run", "list", "--commit", COMMIT])
    assert data == [] and command[-2:] == ["--repo", "zapabob/Strata"]
    assert calls == [command]
    monkeypatch.setattr(
        a.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=b"", stderr=b"private diagnostic"),
    )
    with pytest.raises(RuntimeError, match="raw diagnostics suppressed") as error:
        a.gh_json(["run", "list"])
    assert "private diagnostic" not in str(error.value)


@pytest.mark.parametrize("count", (0, 2))
def test_ci_never_selects_missing_or_ambiguous_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    monkeypatch.setattr(a, "gh_json", lambda argv: ([ci_run()] * count, ["synthetic"]))
    with pytest.raises(RuntimeError, match="unambiguous"):
        a.ci(SimpleNamespace(reviewed_commit=COMMIT), tmp_path)


@pytest.mark.parametrize(
    "key,value",
    (
        ("status", "STOPPED"),
        ("exit_code", 1),
        ("native_exit_code", None),
        ("termination_confirmed", False),
        ("contained_before_resume", False),
        ("job_active_after", 1),
        ("observed_dropped_bytes", 2),
        ("resource_samples", 0),
    ),
)
def test_job_rejects_failed_or_undrained(key: str, value: object) -> None:
    record = job()
    record[key] = value
    with pytest.raises(RuntimeError):
        a.successful_job(record)


def test_job_rejects_memory_floor_violation() -> None:
    record = job()
    record["memory_diagnostics"]["min_available_ram"] = a.release.GIB
    with pytest.raises(RuntimeError):
        a.successful_job(record)


def test_unbound_policy_never_lowers_floor_but_nominal_gpu_observation_remains_valid() -> None:
    record = job()
    record["ramPolicy"] = a.release.ram_policy(True) | {
        "samples": 2,
        "nominalBreachSamples": 0,
        "minAvailableBytes": 4 * a.release.GIB,
    }
    a.successful_job(record)
    record["memory_diagnostics"]["min_available_ram"] = a.release.GIB
    with pytest.raises(RuntimeError, match="resource floor"):
        a.successful_job(record)


def tap() -> str:
    return "TAP version 13\nok 1 - required case\n1..1\n# tests 1\n# pass 1\n# fail 0\n# cancelled 0\n# skipped 0\n# todo 0\n"


def test_tap_requires_names_and_complete_summary() -> None:
    assert a.validate_tap(tap(), ["required case"])["pass"] == 1
    assert len(a.harness_contract(a.native.PRIMARY)["names"]) > 40


@pytest.mark.parametrize(
    "text",
    (
        tap().replace("ok 1", "not ok 1"),
        tap().replace("required case", "other"),
        tap().replace("required case", "required case # SKIP"),
        tap().replace("# fail 0\n", ""),
        tap().replace("1..1", "1..0"),
        tap() + "# tests 1\n",
    ),
)
def test_tap_rejects_incomplete_or_skipped(text: str) -> None:
    with pytest.raises(RuntimeError):
        a.validate_tap(text, ["required case"])


def nsis_rows() -> list[dict]:
    rows = []
    for name in sorted(a.SCENARIOS):
        blocked = name in {"locked", "junction", "root-junction"}
        rows.append(
            {
                "scenario": name,
                "ok": True,
                "userSentinelsPreserved": True,
                "oldEngineRemains": blocked,
                "newEngineCopied": not blocked,
                "exit": 2 if blocked else 0,
                "obsoleteManagedFolders": [],
                "uninstallExit": None if blocked else 0,
                "uninstallPreservedUserData": None if blocked else True,
            }
        )
    return rows


def test_nsis_observations() -> None:
    a.validate_nsis(nsis_rows())


@pytest.mark.parametrize("change", ("missing", "duplicate", "sentinel", "cleanup", "exit", "uninstall"))
def test_nsis_rejects_malformed_success(change: str) -> None:
    rows = nsis_rows()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    else:
        key = {
            "sentinel": "userSentinelsPreserved",
            "cleanup": "newEngineCopied",
            "exit": "exit",
            "uninstall": "uninstallPreservedUserData",
        }[change]
        rows[0][key] = 1 if change == "exit" else False
    with pytest.raises(RuntimeError):
        a.validate_nsis(rows)


def junit(path: Path) -> None:
    path.write_text(
        '<testsuite tests="10" failures="0" skipped="0">'
        + "".join(f'<testcase name="{name}" status="run" />' for name in sorted(a.GPU_NAMES))
        + "</testsuite>",
        encoding="utf-8",
    )


@pytest.mark.parametrize("change", ("none", "skip", "missing", "unexpected", "failure"))
def test_junit_exact_cases(tmp_path: Path, change: str) -> None:
    path = tmp_path / "gpu.xml"
    junit(path)
    if change == "none":
        a.validate_junit(path)
        return
    text = path.read_text(encoding="utf-8")
    if change == "skip":
        text = text.replace('status="run" />', 'status="run"><skipped /></testcase>', 1)
    elif change == "missing":
        text = text.replace('tests="10"', 'tests="9"')
    elif change == "unexpected":
        text = text.replace(sorted(a.GPU_NAMES)[0], "unexpected")
    else:
        text = text.replace('failures="0"', 'failures="1"')
    path.write_text(text, encoding="utf-8")
    with pytest.raises(RuntimeError):
        a.validate_junit(path)


def gpu() -> tuple[dict, dict, dict, dict]:
    binaries = {"strata.exe": "a" * 64, "strata-q36.exe": "b" * 64}
    hashes = {str(a.native.BUILD / "Release" / n): d for n, d in binaries.items()}
    binding = {
        "sourceRoot": str(a.native.PRIMARY),
        "sourceDigest": "c" * 64,
        "sourceUnchanged": True,
        "binariesUnchanged": True,
        "releaseQualified": False,
        "exitCode": 0,
        "before": hashes,
        "after": hashes,
    }
    report = {
        "registeredRequestedTests": sorted(a.GPU_NAMES),
        "exitCode": 0,
        "reason": None,
        "admissionFreeMiB": 9000,
        "reserveFloorMiB": 768,
        "minimumFreeMiB": 900,
        "samples": 3,
    }
    guard = {
        "passed": True,
        "run": job(),
        "admission": {
            "available_ram": 4 * a.release.GIB,
            "commit_headroom": 5 * a.release.GIB,
            "H_free": 11 * a.release.GIB,
            "C_free": 6 * a.release.GIB,
        },
    }
    return binding, report, guard, binaries


@pytest.mark.parametrize("change", ("none", "binary", "source", "source-attestation", "reserve", "guard"))
def test_gpu_carry_requires_binary_source_and_guard_equality(change: str) -> None:
    binding, report, guard, binaries = gpu()
    if change == "binary":
        binaries["strata-q36.exe"] = "d" * 64
    elif change == "source":
        binding["sourceDigest"] = "d" * 64
    elif change == "source-attestation":
        binding["sourceUnchanged"] = False
    elif change == "reserve":
        report["minimumFreeMiB"] = 767
    elif change == "guard":
        guard["run"]["job_active_after"] = 1
    if change == "none":
        a.validate_gpu(binding, report, guard, "c" * 64, binaries)
    else:
        with pytest.raises(RuntimeError):
            a.validate_gpu(binding, report, guard, "c" * 64, binaries)


def test_artifact_rejects_tamper_and_relative_path(tmp_path: Path) -> None:
    path = tmp_path / "strata.exe"
    path.write_bytes(b"fixture-only")
    expected = a.artifact(path)["sha256"]
    path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError):
        a.artifact(path, expected)
    with pytest.raises(RuntimeError):
        a.artifact(Path("relative.json"), expected)


def put(path: Path, value: object) -> Path:
    """Write labelled synthetic evidence exclusively inside pytest scratch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
    return path


def repin(path: Path, record: dict) -> str:
    """Model initial pinning of malformed evidence, rather than only later tampering."""
    raw = {k: v for k, v in record.items() if k not in ("receiptSHA256", "ownedJobSHA256", "workerLogSHA256")}
    put(path.parent / "receipt.json", raw)
    record.update(
        receiptSHA256=a.release.sha(path.parent / "receipt.json"),
        ownedJobSHA256=a.release.sha(path.parent / "owned-job.json"),
        workerLogSHA256=a.release.sha(path.parent / "worker.log"),
    )
    put(path, record)
    return a.release.sha(path)


def native_proof_fixture(raw: dict, work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic native evidence; only signature verification is isolated here.

    No actual build/engine/probe/signature/GPU work is performed. The existing
    production signature verifier is separately covered by the feed tests.
    """
    inputs, root, build = raw["inputs"], a.native.PRIMARY, a.native.BUILD
    put(root / "CMakeLists.txt", "synthetic source")
    put(root / "third_party/strata-ggml/UPSTREAM.md", "fixture")
    put(root / "third_party/strata-ggml/ggml/CMakeLists.txt", "fixture")
    put(build / "cuda-spill-build-key.txt", "f" * 64)
    measured = a.release.module(
        "acceptance_fixture_measured", HERE.parents[1] / "tools/cuda_spill_baseline.py"
    )
    bindings = measured.bindings(root, build / "cuda-spill-build-key.txt", root / "third_party/strata-ggml")
    rows = []
    for name in sorted(a.COMPILED_LIBRARIES):
        file = put(build / "Release" / name, "synthetic compiled " + name)
        rows.append({"library": name, "sha256": a.release.sha(file)})
        inputs["compiled:" + name] = a.artifact(file)
    profile = put(
        root / a.PROFILE,
        {"schema": 1, "bindings": bindings, "entries": {"fixture": [0, 0]}, "measuredLibraries": rows},
    )
    inputs["cudaProfile"] = a.artifact(profile)
    inputs["compilerKey"] = a.artifact(build / "cuda-spill-build-key.txt")
    before = {
        "sources": {"CMakeLists.txt": a.release.sha(root / "CMakeLists.txt")},
        "ggml": {"mode": "vendored", "commit": bindings["ggmlCommit"]},
        "baseline": {"path": a.PROFILE, "sha256": a.release.sha(profile)},
    }
    native_work = a.release.OUT / "native-input-r1"
    component_work = a.release.OUT / "component-r1"
    signed = component_work / "signed" / a.release.NAME
    for name in ("strata.exe", "strata-q36.exe", "toolkit.dll"):
        put(signed / name, "synthetic binary " + name)
        inputs[name if name.endswith(".exe") else "library:" + name] = a.artifact(signed / name)
    binaries = {n: inputs[n]["sha256"] for n in ("strata.exe", "strata-q36.exe")}
    put(
        signed / "native-build.json",
        {
            "schema": 1,
            "version": "0.6.0",
            "sources": before["sources"],
            "recipe": {"ggml": before["ggml"], "cudaSpillBaseline": before["baseline"]},
        },
    )
    inputs["nativeBuild"] = a.artifact(signed / "native-build.json")
    inputs["toolkitVersion"] = a.artifact(
        put(a.native.CUDA / "version.json", {"cuda": {"version": "13.3.1"}})
    )
    toolkit = {
        "versionSHA256": inputs["toolkitVersion"]["sha256"],
        "libraries": {"toolkit.dll": inputs["library:toolkit.dll"]["sha256"]},
    }
    for label, value in (
        ("inputs-before.json", before),
        ("inputs-after.json", before),
        ("toolkit-inputs.json", toolkit),
    ):
        inputs[label] = a.artifact(put(native_work / label, value))
    inputs["builderLog"] = a.artifact(put(native_work / "builder.log", "synthetic builder output"))
    builder = {"exitCode": 0, "logSHA256": inputs["builderLog"]["sha256"]}
    built = {
        "ok": True,
        "phase": "native-input",
        "sourceRoot": str(root),
        "sourceCommit": COMMIT,
        "releaseQualified": False,
        "files": binaries,
        "builder": builder,
    }
    inputs["native"] = a.artifact(put(native_work / "receipt.json", built))
    inputs["native_job"] = a.artifact(put(native_work / "owned-job.json", job()))
    inputs["component_job"] = a.artifact(put(component_work / "owned-job.json", job()))
    binding = {
        "receiptSHA256": inputs["native"]["sha256"],
        "ownedJobSHA256": inputs["native_job"]["sha256"],
        "builder": builder,
        "inputsBeforeSHA256": inputs["inputs-before.json"]["sha256"],
        "inputsAfterSHA256": inputs["inputs-after.json"]["sha256"],
        "toolkitInputsSHA256": inputs["toolkit-inputs.json"]["sha256"],
        "nativeBuildSHA256": inputs["nativeBuild"]["sha256"],
        "binaries": binaries,
    }
    inputs["nativeBinding"] = a.artifact(put(native_work / "native-binding.json", binding))
    component_files = {p.name: a.release.sha(p) for p in signed.iterdir()}
    for name in component_files:
        inputs["component:" + name] = a.artifact(signed / name)
    inputs["component"] = a.artifact(
        put(
            component_work / "receipt.json",
            {
                "signedComponent": str(signed),
                "files": component_files,
                "nativeBindingSHA256": inputs["nativeBinding"]["sha256"],
            },
        )
    )
    monkeypatch.setattr(
        a.feed,
        "receipt_assets",
        lambda path, digest, phase, commit: a.release.require(
            path == component_work / "receipt.json"
            and digest == inputs["component"]["sha256"]
            and phase == "component"
            and commit == COMMIT,
            "Synthetic signature boundary mismatch",
        ),
    )
    gpu_dir = work / "historical"
    gpu_dir.mkdir()
    junit(gpu_dir / "gpu.xml")
    inputs["gpu_junit"] = a.artifact(gpu_dir / "gpu.xml")
    historical, report, guard, _ = gpu()
    historical.update(
        sourceDigest=bindings["sourceDigest"],
        before={str(build / "Release" / n): d for n, d in binaries.items()},
        after={str(build / "Release" / n): d for n, d in binaries.items()},
    )
    report["command"] = ["synthetic-ctest", "--output-junit", str(gpu_dir / "gpu.xml")]
    inputs["gpu_report"] = a.artifact(put(gpu_dir / "report.json", report))
    historical["reportSHA256"] = inputs["gpu_report"]["sha256"]
    inputs["gpu_binding"] = a.artifact(put(gpu_dir / "binding.json", historical))
    inputs["gpuCommandLog"] = a.artifact(put(gpu_dir / "command.log", "synthetic GPU output"))
    guard["command_log"] = inputs["gpuCommandLog"]
    inputs["gpu_guard"] = a.artifact(put(gpu_dir / "guard.json", guard))
    monkeypatch.setattr(a, "GPU_PINS", {name: inputs[name]["sha256"] for name in a.GPU_PINS})
    equality = {
        "compiledLibraries": {n: inputs["compiled:" + n] for n in a.COMPILED_LIBRARIES},
        "profileBindings": bindings,
        "currentSourceDigest": bindings["sourceDigest"],
        "binaries": binaries,
        "versionProbes": {n: "synthetic 0.6.0" for n in binaries},
        "versionProbeCommands": [[str(signed / n), "--version"] for n in binaries],
        "historicalSourceDigest": bindings["sourceDigest"],
        "historicalSourceUnchangedAttested": True,
    }
    raw["results"]["equality"] = a.artifact(put(work / "native-equality.json", equality))


def acceptance_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str = "harness", *, low_ram: bool = False
) -> tuple[Path, dict]:
    """A substantive synthetic proof for each mode, never actual acceptance evidence."""
    original_root = a.native.PRIMARY
    root = tmp_path / "source"
    monkeypatch.setattr(a.release, "OUT", tmp_path / "out")
    monkeypatch.setattr(a.native, "PRIMARY", root)
    monkeypatch.setattr(a.native, "BUILD", tmp_path / "build")
    monkeypatch.setattr(a.native, "CUDA", tmp_path / "cuda")
    monkeypatch.setattr(a.release, "RES", root / "resources")
    monkeypatch.setattr(a.release, "NODE", put(root / "node.exe", "synthetic node"))
    monkeypatch.setattr(a, "GH", put(root / "gh.exe", "synthetic gh"))
    monkeypatch.setattr(a, "PWSH", put(root / "pwsh.exe", "synthetic PowerShell"))
    monkeypatch.setattr(a, "NSIS", put(root / "makensis.exe", "synthetic NSIS"))
    work = a.release.OUT / ("acceptance-" + mode + "-r1")
    work.mkdir(parents=True)
    raw = {
        "ok": True,
        "schema": a.SCHEMA,
        "phase": "acceptance-" + mode,
        "mode": mode,
        "scope": a.SCOPES[mode],
        "sourceCommit": COMMIT,
        "sourceRoot": str(root),
        "tree": "b" * 40,
        "reviewedSourceRoot": str(root),
        "releaseQualified": False,
        "inputs": {n: a.artifact(HERE / n) for n in a.ADAPTERS},
        "results": {},
    }
    inputs, results = raw["inputs"], raw["results"]
    inputs["nodeExecutable"] = a.artifact(a.release.NODE)

    def source(relative: str) -> Path:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_root / relative, path)
        return path

    if mode == "ci":
        inputs["ghExecutable"] = a.artifact(a.GH)
        commands = []
        for name in a.WORKFLOWS:
            inputs[name] = a.artifact(source(".github/workflows/" + name))
            row = {**ci_run(name), "workflowFile": name}
            results[name] = a.artifact(put(work / (name + ".json"), row))
            commands.extend(
                [
                    [
                        str(a.GH),
                        "run",
                        "list",
                        "--workflow",
                        name,
                        "--commit",
                        COMMIT,
                        "--branch",
                        "main",
                        "--limit",
                        "100",
                        "--json",
                        "databaseId,workflowName,headSha,headBranch,status,conclusion",
                        "--repo",
                        a.REPO,
                    ],
                    [
                        str(a.GH),
                        "run",
                        "view",
                        "1",
                        "--json",
                        "databaseId,workflowName,headSha,headBranch,status,conclusion,jobs",
                        "--repo",
                        a.REPO,
                    ],
                ]
            )
        results["commands"] = a.artifact(put(work / "commands.json", commands))
    elif mode == "harness":
        for name in a.HARNESS_FILES:
            inputs[name] = a.artifact(source("apps/host/" + name))
        for name in ("test-fs-guard.mjs", "test-environment.mjs"):
            inputs[name] = a.artifact(source("apps/host/scripts/" + name))
        inputs["harnessBudgetModule"] = a.artifact(
            source("apps/desktop/scripts/harness-budget.mjs")
        )
        budget = Path(inputs["harnessBudgetModule"]["path"]).read_text(encoding="utf-8")
        import re

        patch = {
            key: re.findall(r"\b" + key + r": '([^']+)'", budget)[0]
            for key in ("id", "file", "beforeSha256")
        }
        inputs["harnessPatch"] = a.artifact(
            put(a.release.RES / "harness" / patch["file"], "synthetic patched content")
        )
        patch["afterSha256"] = inputs["harnessPatch"]["sha256"]
        bundle = {"commit": "571902028a75c64c1d767f4ae5fd099485f0ddf0", "patches": [patch]}
        inputs["harnessBundle"] = a.artifact(put(a.release.RES / "harness/BUNDLE.json", bundle))
        inputs["harnessInventory"] = a.artifact(
            put(
                work / "harness-inventory.json",
                {
                    "BUNDLE.json": inputs["harnessBundle"]["sha256"],
                    patch["file"]: patch["afterSha256"],
                },
            )
        )
        identity = {
            "runtime": "v24.22.3",
            "commit": bundle["commit"],
            "bundleSHA256": inputs["harnessBundle"]["sha256"],
            "patch": patch,
            "importClosure": "complete",
        }
        contract = a.harness_contract(root)
        count = len(contract["names"])
        text = "TAP version 13\n" + "".join(
            f"ok {i + 1} - {n}\n" for i, n in enumerate(contract["names"])
        )
        text += f"1..{count}\n# tests {count}\n# pass {count}\n# fail 0\n# cancelled 0\n# skipped 0\n# todo 0\n"
        command = [
            str(a.release.NODE),
            "--import",
            (root / "apps/host/scripts/test-fs-guard.mjs").as_uri(),
            "--test",
            "--test-reporter=tap",
            "--test-concurrency=1",
            "--test-timeout=600000",
            *contract["files"],
        ]
        for name, value in (
            ("harness-contract.json", contract),
            ("harness-identity.json", identity),
            ("harness-command.json", command),
            ("harness.tap", text),
            ("harness-stderr.log", ""),
            ("harness-run.log", "synthetic launcher"),
        ):
            results[name] = a.artifact(put(work / name, value))
        raw["counters"] = a.validate_tap(text, contract["names"])
        raw["command"] = {
            "command": [
                str(a.release.NODE),
                str(HERE / "acceptance-harness.mjs"),
                str(root),
                str(work),
                "run",
            ],
            "exitCode": 0,
            "logSHA256": results["harness-run.log"]["sha256"],
        }
    elif mode == "nsis":
        inputs.update(
            script=a.artifact(source("tools/test_nsis_upgrade.ps1")),
            installer=a.artifact(source("apps/desktop/src-tauri/nsis/installer.nsi")),
            pwsh=a.artifact(a.PWSH),
            compiler=a.artifact(a.NSIS),
        )
        results["fixtures"] = a.artifact(put(work / "fixtures/results.json", nsis_rows()))
        results["source"] = a.artifact(
            put(work / "fixtures/source-sha256.txt", inputs["installer"]["sha256"])
        )
        results["log"] = a.artifact(put(work / "nsis.log", "synthetic NSIS output"))
        raw["powerShellVersion"] = "7.5.0"
        raw["command"] = {
            "command": [
                str(a.PWSH),
                "-NoProfile",
                "-NonInteractive",
                "-File",
                inputs["script"]["path"],
                "-OutputDir",
                str(work / "fixtures"),
                "-NsisCompiler",
                str(a.NSIS),
            ],
            "exitCode": 0,
            "logSHA256": results["log"]["sha256"],
        }
    else:
        native_proof_fixture(raw, work, monkeypatch)
    command = [
        sys.executable,
        "-B",
        str(HERE / "acceptance.py"),
        mode,
        "--execute",
        "--reviewed-commit",
        COMMIT,
        "--attempt",
        "r1",
        "--worker",
    ]
    if mode == "final-native":
        for name in a.EXTERNAL_INPUTS:
            option = "--" + name.replace("_", "-")
            command += [
                option + "-receipt",
                inputs[name]["path"],
                option + "-receipt-sha256",
                inputs[name]["sha256"],
            ]
    if low_ram:
        command.append("--allow-low-ram")
    results["workerCommand"] = a.artifact(put(work / "worker-command.json", command))
    reservation = {"identity": {k: raw[k] for k in ("sourceCommit", "sourceRoot", "tree")}}
    if low_ram:
        reservation.update(
            ramPolicy=a.release.ram_policy(True),
            command=command,
            helpers={
                n: a.release.sha(HERE / n) for n in ("release.py", "verify.mjs", "acceptance.py")
            },
        )
    put(work / "reservation.json", reservation)
    reservation_sha = a.release.sha(work / "reservation.json")
    results["workerContext"] = a.artifact(
        put(
            work / "worker-context.json",
            {
                "schema": "strata.acceptance.worker.v1",
                "command": command,
                "pid": 1,
                "creationFiletime": 123,
                "attemptRoot": str(work),
                "reservationSHA256": reservation_sha,
            },
        )
    )
    current_job = job()
    current_job.update(
        root_creation_filetime=123,
        acceptanceExecution={
            "schema": "strata.acceptance.execution.v1",
            "command": command,
            "cwd": str(root),
            "attemptRoot": str(work),
            "reservationSHA256": reservation_sha,
            "adapterSHA256": inputs["acceptance.py"]["sha256"],
        },
    )
    if low_ram:
        policy = a.release.ram_policy(True) | {
            "samples": 4,
            "nominalBreachSamples": 4,
            "minAvailableBytes": 2 * a.release.GIB,
        }
        raw["ramPolicy"] = policy
        current_job["ramPolicy"] = policy
        current_job["operationRAMSamples"] = [2 * a.release.GIB] * 4
        observed = {
            "available_ram": 2 * a.release.GIB,
            "commit_headroom": 5 * a.release.GIB,
            "H_free": 11 * a.release.GIB,
            "C_free": 6 * a.release.GIB,
        }
        current_job["operationAdmissions"] = [observed]
        current_job["memory_diagnostics"]["min_available_ram"] = 2 * a.release.GIB
        context = {
            "command": command,
            "identity": reservation["identity"],
            "ramPolicy": a.release.ram_policy(True),
            "pid": 1,
            "creationFiletime": 123,
            "attemptRoot": str(work),
            "reservationSHA256": reservation_sha,
            "admission": observed,
        }
        put(work / "operation-worker.json", context)
        current_job["operationExecution"] = {
            "command": command,
            "cwd": str(root),
            "attemptRoot": str(work),
            "reservationSHA256": reservation_sha,
            "workerContextSHA256": a.release.sha(work / "operation-worker.json"),
            "installAdmissionSHA256": None,
            "admission": {
                "available_ram": 2 * a.release.GIB,
                "commit_headroom": 5 * a.release.GIB,
                "H_free": 11 * a.release.GIB,
                "C_free": 6 * a.release.GIB,
            },
        }
    put(work / "owned-job.json", current_job)
    put(work / "worker.log", "synthetic worker output")
    path = work / "acceptance.json"
    repin(path, raw)
    return path, raw


@pytest.mark.parametrize(
    "damage", ["none", "flag", "source", "reservation", "minimum", "breaches", "policy", "stopped"]
)
def test_bound_operation_exception_through_pinned_consumer(
    tmp_path: Path, monkeypatch, damage: str
) -> None:
    path, record = acceptance_fixture(tmp_path, monkeypatch, low_ram=True)
    work = path.parent
    current_job = a.release.load(work / "owned-job.json")
    if damage == "none":
        a.validate_acceptance(path, a.release.sha(path), "harness", COMMIT)
        return
    if damage == "flag":
        current_job["operationExecution"]["command"].remove("--allow-low-ram")
    elif damage == "source":
        record["sourceCommit"] = "c" * 40
    elif damage == "reservation":
        reservation = a.release.load(work / "reservation.json")
        reservation["ramPolicy"] = a.release.ram_policy(False)
        put(work / "reservation.json", reservation)
    elif damage == "minimum":
        current_job["memory_diagnostics"]["min_available_ram"] -= 1
    elif damage == "breaches":
        record["ramPolicy"]["nominalBreachSamples"] = 0
    elif damage == "policy":
        del record["ramPolicy"]
    else:
        current_job.update(status="TIMEOUT", exit_code=124)
    put(work / "owned-job.json", current_job)
    repin(path, record)
    with pytest.raises(RuntimeError):
        a.validate_acceptance(path, a.release.sha(path), "harness", COMMIT)


@pytest.mark.parametrize(
    "phase,adapter", [("native-input", "native-input.py"), ("component", "components-feed.py")]
)
@pytest.mark.parametrize("damage", ["none", "absent", "mismatched", "tampered"])
def test_current_native_job_requires_its_bound_low_ram_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, adapter: str, damage: str
) -> None:
    path, record = acceptance_fixture(tmp_path, monkeypatch, low_ram=True)
    old = path.parent
    work = old.parent / (phase + "-r1")
    old.rename(work)
    record["phase"] = phase
    command = a.release.load(work / "operation-worker.json")["command"]
    command[2] = str(HERE / adapter)
    reservation = a.release.load(work / "reservation.json")
    reservation["command"] = command
    reservation["helpers"] = {
        name: a.release.sha(HERE / name) for name in ("release.py", "verify.mjs", adapter)
    }
    put(work / "reservation.json", reservation)
    context = a.release.load(work / "operation-worker.json")
    context.update(
        command=command, attemptRoot=str(work), reservationSHA256=a.release.sha(work / "reservation.json")
    )
    put(work / "operation-worker.json", context)
    current = a.release.load(work / "owned-job.json")
    current["operationExecution"].update(
        command=command,
        attemptRoot=str(work),
        reservationSHA256=a.release.sha(work / "reservation.json"),
        workerContextSHA256=a.release.sha(work / "operation-worker.json"),
    )
    if damage == "absent":
        del record["ramPolicy"]
    elif damage == "mismatched":
        record["phase"] = "other"
    elif damage == "tampered":
        context["admission"]["available_ram"] += 1
        put(work / "operation-worker.json", context)
    if damage == "none":
        a.successful_job(current, record, work)
    else:
        with pytest.raises(RuntimeError):
            a.successful_job(current, record, work)


def bind_current_native_jobs(record: dict) -> None:
    """Turn the fixture's current native and component jobs into bound low-RAM jobs."""
    inputs = record["inputs"]
    for phase, adapter, key in (
        ("native-input", "native-input.py", "native"),
        ("component", "components-feed.py", "component"),
    ):
        receipt = Path(inputs[key]["path"])
        work = receipt.parent
        data = a.release.load(receipt)
        data.update(
            phase=phase,
            sourceRoot=record["sourceRoot"],
            sourceCommit=COMMIT,
            tree=record["tree"],
            ramPolicy=a.release.ram_policy(True)
            | {"samples": 4, "nominalBreachSamples": 4, "minAvailableBytes": 2 * a.release.GIB},
        )
        identity = {k: data[k] for k in ("sourceRoot", "sourceCommit", "tree")}
        command = [
            sys.executable,
            "-B",
            str(HERE / adapter),
            "--execute",
            "--worker",
            "--reviewed-commit",
            COMMIT,
            "--attempt",
            "r1",
            "--allow-low-ram",
        ]
        policy = a.release.ram_policy(True)
        reservation = {
            "identity": identity,
            "command": command,
            "ramPolicy": policy,
            "helpers": {n: a.release.sha(HERE / n) for n in ("release.py", "verify.mjs", adapter)},
        }
        put(work / "reservation.json", reservation)
        resources = {
            "available_ram": 2 * a.release.GIB,
            "commit_headroom": 5 * a.release.GIB,
            "H_free": 11 * a.release.GIB,
            "C_free": 6 * a.release.GIB,
        }
        context = {
            "command": command,
            "identity": identity,
            "ramPolicy": policy,
            "pid": 1,
            "creationFiletime": 123,
            "attemptRoot": str(work),
            "reservationSHA256": a.release.sha(work / "reservation.json"),
            "admission": resources,
        }
        put(work / "operation-worker.json", context)
        current = job()
        current.update(
            root_creation_filetime=123,
            ramPolicy=data["ramPolicy"],
            operationRAMSamples=[2 * a.release.GIB] * 4,
            operationAdmissions=[resources],
            operationExecution={
                "command": command,
                "cwd": record["sourceRoot"],
                "attemptRoot": str(work),
                "reservationSHA256": a.release.sha(work / "reservation.json"),
                "workerContextSHA256": a.release.sha(work / "operation-worker.json"),
                "installAdmissionSHA256": None,
                "admission": resources,
            },
        )
        current["memory_diagnostics"]["min_available_ram"] = 2 * a.release.GIB
        put(work / "owned-job.json", current)
        if key == "native":
            inputs["native"] = a.artifact(put(receipt, data))
            inputs["native_job"] = a.artifact(work / "owned-job.json")
            binding_path = Path(inputs["nativeBinding"]["path"])
            binding = a.release.load(binding_path)
            binding.update(
                receiptSHA256=inputs["native"]["sha256"], ownedJobSHA256=inputs["native_job"]["sha256"]
            )
            inputs["nativeBinding"] = a.artifact(put(binding_path, binding))
        else:
            data["nativeBindingSHA256"] = inputs["nativeBinding"]["sha256"]
            inputs["component"] = a.artifact(put(receipt, data))
            inputs["component_job"] = a.artifact(work / "owned-job.json")


def test_final_native_consumer_rechecks_bound_current_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, record = acceptance_fixture(tmp_path, monkeypatch, "final-native")
    bind_current_native_jobs(record)
    monkeypatch.setattr(a.delivery, "receipt", lambda path, *_args: a.release.load(path))
    seen = []
    original = a.successful_job

    def capture(current, receipt=None, work=None):
        if receipt is not None:
            seen.append((receipt["phase"], work))
        return original(current, receipt, work)

    monkeypatch.setattr(a, "successful_job", capture)
    a.validate_native_proof(record, path.parent)
    assert seen == [
        ("native-input", Path(record["inputs"]["native"]["path"]).parent),
        ("component", Path(record["inputs"]["component"]["path"]).parent),
    ]


def test_final_native_producer_rechecks_bound_current_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, record = acceptance_fixture(tmp_path, monkeypatch, "final-native")
    inputs = record["inputs"]
    native_receipt = Path(inputs["native"]["path"])
    component_receipt = Path(inputs["component"]["path"])
    signed = Path(a.release.load(component_receipt)["signedComponent"])
    for backend in ("prism", "turboquant"):
        (signed / "local-backends" / backend).mkdir(parents=True)
    unsigned = native_receipt.parent / "unsigned" / a.release.NAME
    unsigned.mkdir(parents=True)
    for name in ("strata.exe", "strata-q36.exe"):
        shutil.copyfile(signed / name, unsigned / name)
        shutil.copyfile(signed / name, a.native.BUILD / "Release" / name)
    shutil.copyfile(signed / "native-build.json", a.native.BUILD / "Release" / "native-build.json")
    built = a.release.load(native_receipt)
    built.update(nativeComponent=str(unsigned), files=a.release.inventory(unsigned))
    put(native_receipt, built)
    put(native_receipt.parent / "worker.log", "native synthetic")
    put(component_receipt.parent / "worker.log", "component synthetic")
    bind_current_native_jobs(record)
    monkeypatch.setattr(a.delivery, "receipt", lambda path, *_args: a.release.load(path))
    monkeypatch.setattr(
        a.native,
        "configured_inputs",
        lambda _build: a.release.load(Path(inputs["inputs-before.json"]["path"])),
    )
    monkeypatch.setattr(
        a.native, "toolkit_libraries", lambda _path: {"toolkit.dll": inputs["library:toolkit.dll"]["sha256"]}
    )
    monkeypatch.setattr(a.release, "verify", lambda *_args: None)
    monkeypatch.setattr(a.release, "command", lambda *_args, **_kwargs: "0.6.0")
    real_module = a.release.module
    monkeypatch.setattr(
        a.release,
        "module",
        lambda name, module_path: (
            SimpleNamespace(uncovered=lambda *_args: [])
            if name == "acceptance_pe"
            else real_module(name, module_path)
        ),
    )
    args = SimpleNamespace(reviewed_commit=COMMIT)
    for name in a.EXTERNAL_INPUTS:
        setattr(args, name + "_receipt", Path(inputs[name]["path"]))
        setattr(args, name + "_receipt_sha256", inputs[name]["sha256"])
    seen = []
    original = a.successful_job

    def capture(current, receipt=None, work=None):
        if receipt is not None:
            seen.append((receipt["phase"], work))
        return original(current, receipt, work)

    monkeypatch.setattr(a, "successful_job", capture)
    producer_work = tmp_path / "producer-output"
    producer_work.mkdir()
    a.final_native(args, producer_work)
    assert seen == [("native-input", native_receipt.parent), ("component", component_receipt.parent)]


@pytest.mark.parametrize(
    "change", ("none", "stale-source", "phase", "schema", "result", "job", "log", "raw-receipt")
)
def test_consumer_requires_frozen_post_drain_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    path, record = acceptance_fixture(tmp_path, monkeypatch)
    expected = a.release.sha(path)
    if change == "none":
        assert a.validate_acceptance(path, expected, "harness", COMMIT)["schema"] == a.SCHEMA
        return
    if change in ("stale-source", "phase", "schema"):
        key = {"stale-source": "sourceCommit", "phase": "phase", "schema": "schema"}[change]
        record[key] = "foreign"
        path.write_text(json.dumps(record), encoding="utf-8")
        expected = a.release.sha(path)  # Even a caller pin cannot excuse the wrong contract.
    else:
        target = {
            "result": "harness.tap",
            "job": "owned-job.json",
            "log": "worker.log",
            "raw-receipt": "receipt.json",
        }[change]
        (path.parent / target).write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError):
        a.validate_acceptance(path, expected, "harness", COMMIT)


@pytest.mark.parametrize("mode", a.PHASES)
def test_default_cli_is_read_only_plan(mode: str, tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-B", str(HERE / "acceptance.py"), mode],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stderr
    assert '"execute": false' in proc.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mode", a.PHASES)
def test_complete_mode_specific_consumer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    path, _ = acceptance_fixture(tmp_path, monkeypatch, mode)
    assert a.validate_acceptance(path, a.release.sha(path), mode, COMMIT)["mode"] == mode


@pytest.mark.parametrize(
    "mode,key",
    (
        ("ci", "metal-build.yml"),
        ("harness", "harness.tap"),
        ("nsis", "fixtures"),
        ("final-native", "equality"),
    ),
)
def test_consumer_rejects_missing_required_mode_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, key: str
) -> None:
    path, record = acceptance_fixture(tmp_path, monkeypatch, mode)
    del record["results"][key]
    expected = repin(path, record)
    with pytest.raises(RuntimeError, match="Missing required evidence"):
        a.validate_acceptance(path, expected, mode, COMMIT)


@pytest.mark.parametrize("mode", a.PHASES)
def test_generic_hash_maps_are_never_mode_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    path, record = acceptance_fixture(tmp_path, monkeypatch, mode)
    generic = a.artifact(put(path.parent / "generic.txt", "fixture"))
    record["inputs"] = {"fixture": generic}
    record["results"] = {"fixture": generic}
    expected = repin(path, record)
    with pytest.raises(RuntimeError, match="Missing required evidence"):
        a.validate_acceptance(path, expected, mode, COMMIT)


@pytest.mark.parametrize(
    "change",
    (
        "mode",
        "attempt",
        "source",
        "missing-attempt",
        "foreign-script",
        "job-pid",
        "job-creation",
        "job-command",
        "reservation",
    ),
)
def test_consumer_rejects_replayed_or_foreign_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    path, record = acceptance_fixture(tmp_path, monkeypatch)
    command_path = path.parent / "worker-command.json"
    command = a.release.load(command_path)
    if change.startswith("job-"):
        current = a.release.load(path.parent / "owned-job.json")
        if change == "job-pid":
            current["owned_root_pid"] += 1
        elif change == "job-creation":
            current["root_creation_filetime"] += 1
        else:
            current["acceptanceExecution"]["command"][3] = "nsis"
        put(path.parent / "owned-job.json", current)
    elif change == "reservation":
        reservation = a.release.load(path.parent / "reservation.json")
        reservation["token"] = "new-unrelated-attempt"
        put(path.parent / "reservation.json", reservation)
    else:
        if change == "mode":
            command[3] = "nsis"
        elif change == "attempt":
            command[command.index("--attempt") + 1] = "r2"
        elif change == "source":
            command[command.index("--reviewed-commit") + 1] = "f" * 40
        elif change == "foreign-script":
            command[2] = str(path.parent / "unrelated.py")
        else:
            at = command.index("--attempt")
            command.pop(at)
            command.pop(at)
        put(command_path, command)
        record["results"]["workerCommand"] = a.artifact(command_path)
    expected = repin(path, record)
    with pytest.raises(RuntimeError):
        a.validate_acceptance(path, expected, "harness", COMMIT)


def test_native_consumer_links_external_argv_to_input_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, record = acceptance_fixture(tmp_path, monkeypatch, "final-native")
    command_path = path.parent / "worker-command.json"
    command = a.release.load(command_path)
    command[command.index("--native-receipt-sha256") + 1] = "f" * 64
    put(command_path, command)
    record["results"]["workerCommand"] = a.artifact(command_path)
    expected = repin(path, record)
    with pytest.raises(RuntimeError, match="external input argument"):
        a.validate_acceptance(path, expected, "final-native", COMMIT)


@pytest.mark.parametrize(
    "mode,target",
    (
        ("ci", "metal-build.yml"),
        ("harness", "harness.tap"),
        ("nsis", "fixtures"),
        ("final-native", "equality"),
    ),
)
def test_consumer_checks_rehashed_mode_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, target: str
) -> None:
    path, record = acceptance_fixture(tmp_path, monkeypatch, mode)
    artifact_path = Path(record["results"][target]["path"])
    if mode == "harness":
        put(artifact_path, artifact_path.read_text(encoding="utf-8").replace("ok 1 -", "not ok 1 -", 1))
    else:
        content = a.release.load(artifact_path)
        if mode == "ci":
            content["jobs"][0]["conclusion"] = "skipped"
        elif mode == "nsis":
            content[0]["userSentinelsPreserved"] = False
        else:
            content["binaries"]["strata.exe"] = "f" * 64
        put(artifact_path, content)
    record["results"][target] = a.artifact(artifact_path)
    expected = repin(path, record)
    with pytest.raises(RuntimeError):
        a.validate_acceptance(path, expected, mode, COMMIT)


@pytest.mark.parametrize(
    "change", ("none", "tamper", "missing", "extra", "duplicate", "toolkit-only", "extra-toolkit")
)
def test_exact_compiled_libraries_are_distinct_from_toolkit_dlls(tmp_path: Path, change: str) -> None:
    rows = []
    for name in a.COMPILED_LIBRARIES:
        file = put(tmp_path / name, "synthetic " + name)
        rows.append({"library": name, "sha256": a.release.sha(file)})
    profile = {"measuredLibraries": rows}
    if change == "tamper":
        put(tmp_path / "strata_kq.dll", "changed compiled DLL")
    elif change == "missing":
        rows.pop()
    elif change == "extra":
        rows.append({"library": "cudart64_13.dll", "sha256": "a" * 64})
    elif change == "duplicate":
        rows[-1] = rows[0]
    elif change == "toolkit-only":
        profile["measuredLibraries"] = [{"library": f"toolkit{i}.dll", "sha256": "a" * 64} for i in range(9)]
    elif change == "extra-toolkit":
        put(tmp_path / "cudart64_13.dll", "unrelated toolkit DLL")
    if change in ("none", "extra-toolkit"):
        assert set(a.compiled_libraries(profile, tmp_path)) == a.COMPILED_LIBRARIES
    else:
        with pytest.raises(RuntimeError):
            a.compiled_libraries(profile, tmp_path)


def test_missing_compiled_file_rejected(tmp_path: Path) -> None:
    profile = {"measuredLibraries": [{"library": n, "sha256": "a" * 64} for n in a.COMPILED_LIBRARIES]}
    with pytest.raises(RuntimeError, match="Artifact missing"):
        a.compiled_libraries(profile, tmp_path)


def test_execute_requires_explicit_attempt_before_controller(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> int:
        raise AssertionError("Controller must not be reached")

    monkeypatch.setattr(a.native, "run_controller", forbidden)
    with pytest.raises(RuntimeError, match="explicit --attempt"):
        a.main(["ci", "--execute", "--reviewed-commit", COMMIT])


@pytest.mark.parametrize("flags", (["--attempt", "r2"], ["--attempt=r2"]))
def test_explicit_attempt_forms_are_supported(flags: list[str]) -> None:
    assert a.parsed(["ci", "--execute", "--reviewed-commit", COMMIT, *flags]).attempt == "r2"


def test_guard_annotation_observes_real_invocation_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(a.release, "OUT", tmp_path)
    argv = ["ci", "--execute", "--reviewed-commit", COMMIT, "--attempt", "r2"]
    args = a.parsed(argv)
    work = tmp_path / "acceptance-ci-r2"
    put(work / "reservation.json", {"fixture": "private attempt"})
    seen = []

    def original(command: list[str], root: Path) -> dict:
        seen.append((command, root))
        return job()

    wrapped = a.owned_binding(original, args, argv)
    command = [sys.executable, "-B", str(HERE / "acceptance.py"), *argv, "--worker"]
    observed = wrapped(command, a.native.PRIMARY)
    assert seen == [(command, a.native.PRIMARY)]
    assert observed["acceptanceExecution"]["command"] == command
    assert observed["acceptanceExecution"]["reservationSHA256"] == a.release.sha(work / "reservation.json")
    with pytest.raises(RuntimeError, match="Unexpected owned"):
        wrapped([sys.executable, "unrelated.py"], a.native.PRIMARY)
