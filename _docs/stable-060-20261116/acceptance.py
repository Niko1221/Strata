"""Bounded acceptance producers; default is a read-only plan, never a release.

The production release controller owns admission, clean source checks and Job
containment. acceptance.json is published only after that controller drains.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("acceptance_delivery", HERE / "delivery.py")
delivery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(delivery)
feed, native, release = delivery.feed, delivery.native, delivery.release
require = release.require
SCHEMA = "strata.acceptance.v1"
REPO = "zapabob/Strata"
GH = Path("C:/Program Files/GitHub CLI/gh.exe")
PWSH = Path(
    "C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-611-0_26.1112.611.0_x64__3k8sg7r9htsxt"
    "/dependencies/native/powershell/pwsh.exe"
)
NSIS = Path("C:/Users/downl/AppData/Local/tauri/NSIS/Bin/makensis.exe")
HARNESS_FILES = tuple(
    "tests/" + name + ".test.mjs"
    for name in (
        "agent-context",
        "agent-tools",
        "agent-turn-guard",
        "harness-environment",
        "workflow-safety-harness",
        "agent-compaction",
        "agent-modes",
    )
)
# Reviewed expansion of current YAML, guarded by its exact file digest. A changed
# workflow requires a new review, rather than silently accepting a weaker job set.
WORKFLOWS = {
    "check.yml": (
        "Checks",
        "ce4a0a2ccbd830605c6074403b93fb72309141f242856a7f70567c94b7cc44c7",
        [
            "Windows / pinned Harness context and guard integration",
            *[
                "Linux / " + n
                for n in (
                    "cpu CMake preset",
                    "Python tests/",
                    "Python serve/",
                    "Python tools/",
                    "host typecheck + test + build",
                )
            ],
            *[
                "Windows / " + n
                for n in (
                    "cpu CMake preset (MSVC)",
                    "Python tests/",
                    "Python serve/",
                    "Python tools/",
                    "host typecheck + test + build",
                )
            ],
            "macOS / host typecheck + test + build",
            "macOS / Python tools/",
        ],
    ),
    "docs.yml": (
        "Docs",
        "b8fdb8bf7033f07e3a082e9e7ed4a32fc9035a3fe3741eca7c657298dcb86fa1",
        ["Links, release manifest, capability table"],
    ),
    "backends.yml": (
        "Backend compile checks",
        "57a05749c2168dd71e41c2c3f43df839d4bf66a08fe1f40ae308e4ef203c1305",
        ["CUDA 12.9", "CUDA 13.0", "HIP ROCm 7.2", "SYCL oneAPI 2026.1", "Metal strata + strata_ds4"],
    ),
    "native-preview.yml": (
        "Native preview CPU checks",
        "3d12f5c2e488aa879c5335c7c752202f361d3a0ae4481037d6f388ced7936880",
        ["CPU contracts (ubuntu-latest)", "CPU contracts (windows-latest)"],
    ),
    "terminal-portable-monitor.yml": (
        "Portable monitor client",
        "c38138a74b23c27c644fb7417506640f44cdba5e809b48cd3dd07af2cf1369a7",
        [f"{os} / Python {v}" for os in ("ubuntu-latest", "windows-latest") for v in ("3.10", "3.11")],
    ),
    "terminal-minimal-server.yml": (
        "Minimal serving runtime",
        "d50073e622b88dd540a811d96b8d1a499f28258ccfd8fb60e165671825f8b68c",
        ["Linux / runtime requirements only"],
    ),
    "snapshot.yml": (
        "Config snapshot",
        "95461d0adbe24b4fae4e3b307c537bdea82f1d80179cd590997bc55a29564c1b",
        ["ubuntu-24.04", "windows-2022"],
    ),
    "pages.yml": (
        "Deploy GitHub Pages",
        "1f47628a25ba667dbef2eca2e50fe09734801508abef7200e3edd048efe63027",
        ["deploy"],
    ),
    "metal-build.yml": (
        "Metal build",
        "4f2120b46b41c0c4584a1f5bafdd75b2835922c5eec5222827586da24fcdf1e5",
        ["metal-link"],
    ),
}
GPU_NAMES = {
    "expert_cache_segmented_test",
    "q36_kv_lifecycle_test",
    "q36_kv_cache_test",
    "prefill_fused_moe_test",
    "prefill_fused_iq_test",
    "prefill_fused_group_test",
    "prefill_fused_q36_test",
    "prefill_fused_q36_ww4_test",
    "native_cache_owner",
    "native_cache_bridge",
}
SCENARIOS = {"upgrade", "fresh", "empty", "locked", "junction", "root-junction"}
GPU_PINS = {
    "gpu_binding": "1bb5e5458d71ab880dce3f6b32f20557ca13ede4bca9f086a4ad80e5dbd25c07",
    "gpu_report": "dc7d0f138e03a888749b0753adc99fecefa6faf68e70dbaff7dbfe7ce6ba58f7",
    "gpu_guard": "013fc399513230419a28d2857f3af466b4669be3cd82c57c770b2d70a43d0ac5",
    "gpu_junit": "38faf161b70dc4e37c623eb0ee213f5c6db8164978e8328e68323af3fe7519dd",
}
EXTERNAL_INPUTS = ("native", "component", "native_job", "component_job", *GPU_PINS)
COMPILED_LIBRARIES = {
    "strata_kernels.lib",
    "strata_core.lib",
    "strata_engine.lib",
    "strata_prefill.lib",
    "strata_qwen35.lib",
    "strata_mmq.lib",
    "strata_pf_own.lib",
    "strata_mmq_w4a4.lib",
    "strata_kq.dll",
}
PROFILE = "tools/baselines/hakua060-sm120-cuda133-msvc1444.json"
ADAPTERS = ("acceptance.py", "acceptance-harness.mjs", "native-input.py", "delivery.py", "components-feed.py")
SCOPES = {
    "ci": "Observed remote CI at exact source; local Job ran metadata reader only",
    "harness": "Actual pinned Harness, synthetic model endpoints, serial source regressions; no model qualification",
    "nsis": "Six synthetic cleanup fixtures, no real app installation",
    "final-native": (
        "Historical GPU replay carried by native source digest and both actual binary equalities; "
        "not a final-source GPU execution"
    ),
}


def compiled_libraries(profile: dict, outputs: Path) -> dict:
    """Bind the nine measured native outputs, independently of toolkit PE DLLs."""
    rows = profile.get("measuredLibraries", [])
    require(
        isinstance(rows, list)
        and len(rows) == len(COMPILED_LIBRARIES)
        and {row.get("library") for row in rows} == COMPILED_LIBRARIES,
        "Exactly the nine measured compiled libraries required",
    )
    result = {}
    for row in rows:
        digest = row.get("sha256", "")
        require(
            isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest),
            "Invalid compiled library digest",
        )
        result[row["library"]] = artifact(outputs / row["library"], digest)
    return result


def artifact(path: Path, expected: str | None = None) -> dict:
    """Bind an existing regular absolute file; callers pin external inputs."""
    require(path.is_absolute(), "Absolute artifact path required")
    release.no_links(path)
    require(path.is_file(), "Artifact missing")
    digest = release.sha(path)
    if expected is not None:
        require(re.fullmatch(r"[a-f0-9]{64}", expected) and digest == expected, "Artifact digest mismatch")
    return {"path": str(path), "sha256": digest}


def check_artifacts(rows: dict) -> None:
    """Recheck all evidence bytes, including external logs and source inputs."""
    require(isinstance(rows, dict) and rows, "Nonempty artifact mapping required")
    for row in rows.values():
        artifact(Path(row["path"]), row["sha256"])


def successful_job(job: dict) -> None:
    """Reject stopped, undrained, truncated or unobserved runs."""
    require(
        job.get("status") == "FINISHED"
        and job.get("exit_code") == 0
        and job.get("native_exit_code") == 0
        and job.get("contained_before_resume") is True
        and job.get("termination_confirmed") is True
        and job.get("job_active_after") == 0
        and job.get("observed_dropped_bytes") == 0
        and isinstance(job.get("owned_root_pid"), int)
        and job["owned_root_pid"] > 0
        and job.get("resource_samples", 0) > 0,
        "Successful observed drained Job required",
    )
    diag = job.get("memory_diagnostics", {})
    require(diag.get("status") == "AVAILABLE" and diag.get("samples", 0) > 0, "Resource samples required")
    for name, floor in (("min_available_ram", 3), ("min_commit_headroom", 4), ("min_H_free", 10)):
        require(diag.get(name, 0) >= floor * release.GIB, "Observed resource floor violated")
    require(job.get("last_resources", {}).get("C_free", 0) >= 5 * release.GIB, "C reserve violated")


def validate_ci_run(run: dict, workflow: str, commit: str) -> None:
    """Validate actual gh metadata, with an exact complete expanded job multiset."""
    name, _, jobs = WORKFLOWS[workflow]
    require(
        run.get("workflowName") == name
        and run.get("headSha") == commit
        and run.get("headBranch") == "main"
        and run.get("status") == "completed"
        and run.get("conclusion") == "success"
        and isinstance(run.get("databaseId"), int)
        and run["databaseId"] > 0,
        "CI run identity/status mismatch",
    )
    rows = run.get("jobs", [])
    require(
        sorted(row.get("name", "") for row in rows) == sorted(jobs), "CI missing/duplicate/unexpected jobs"
    )
    ids = [row.get("databaseId") for row in rows]
    require(
        all(isinstance(i, int) and i > 0 for i in ids) and len(ids) == len(set(ids)), "CI job IDs invalid"
    )
    require(
        all(row.get("status") == "completed" and row.get("conclusion") == "success" for row in rows),
        "CI skipped/failed/incomplete job",
    )


def gh_json(argv: list[str]) -> tuple[Any, list[str]]:
    """Read bounded metadata only; never print auth diagnostics or raw logs."""
    command = [str(GH), *argv, "--repo", REPO]
    proc = subprocess.run(
        command,
        cwd=native.PRIMARY,
        capture_output=True,
        timeout=90,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    require(proc.returncode == 0, "GitHub metadata read failed; raw diagnostics suppressed")
    require(len(proc.stdout) <= 1024**2, "CI metadata exceeds limit")
    return json.loads(proc.stdout), command


def ci(args: argparse.Namespace, work: Path) -> dict:
    """Observe remote checks; the local owned Job executes only this reader."""
    inputs, results, commands = {"ghExecutable": artifact(GH)}, {}, []
    for filename, (name, digest, _) in WORKFLOWS.items():
        inputs[filename] = artifact(native.PRIMARY / ".github/workflows" / filename, digest)
        rows, command = gh_json(
            [
                "run",
                "list",
                "--workflow",
                filename,
                "--commit",
                args.reviewed_commit,
                "--branch",
                "main",
                "--limit",
                "100",
                "--json",
                "databaseId,workflowName,headSha,headBranch,status,conclusion",
            ]
        )
        commands.append(command)
        # Do not cherry-pick a successful older run when another run is pending/failed.
        require(isinstance(rows, list) and len(rows) == 1, "One unambiguous current workflow run required")
        row = rows[0]
        require(row.get("workflowName") == name, "Workflow file/name mismatch")
        run, command = gh_json(
            [
                "run",
                "view",
                str(row["databaseId"]),
                "--json",
                "databaseId,workflowName,headSha,headBranch,status,conclusion,jobs",
            ]
        )
        commands.append(command)
        require(all(run.get(k) == v for k, v in row.items()), "Remote run changed while read")
        validate_ci_run(run, filename, args.reviewed_commit)
        # Retain only selected fields, not job step output or service diagnostics.
        run["jobs"] = [
            {k: job[k] for k in ("databaseId", "name", "status", "conclusion")} for job in run["jobs"]
        ]
        run["workflowFile"] = filename
        release.save(work / (filename + ".json"), run)
        results[filename] = artifact(work / (filename + ".json"))
    release.save(work / "commands.json", commands)
    results["commands"] = artifact(work / "commands.json")
    return {
        "scope": "Observed remote CI at exact source; local Job ran metadata reader only",
        "inputs": inputs,
        "results": results,
    }


def harness_contract(root: Path) -> dict:
    """Require literal named test declarations; source changes cannot lose coverage silently."""
    names = []
    for filename in HARNESS_FILES:
        text = (root / "apps/host" / filename).read_text(encoding="utf-8")
        literals = re.findall(r"^test\(('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")\s*,", text, re.M)
        require(
            literals and len(literals) == len(re.findall(r"^test\(", text, re.M)),
            "Unsupported unnamed test declaration",
        )
        names.extend(ast.literal_eval(value) for value in literals)
    require(len(names) == len(set(names)), "Duplicate required test names")
    return {"files": list(HARNESS_FILES), "names": names}


def validate_tap(text: str, names: list[str]) -> dict:
    """All required named top-level cases and complete TAP counters must pass."""
    require(not re.search(r"^\s*(not ok|Bail out!)", text, re.M), "Failed TAP case")
    require(
        not re.search(r"^\s*(?:ok|not ok) \d+ .*#\s*(SKIP|TODO)\b", text, re.I | re.M),
        "Skipped/TODO TAP case",
    )
    actual = re.findall(r"^ok \d+ - (.+)$", text, re.M)
    require(sorted(actual) == sorted(names), "Missing/duplicate/unexpected TAP cases")
    counters = {}
    for key in ("tests", "pass", "fail", "cancelled", "skipped", "todo"):
        values = re.findall(r"^# " + key + r" (\d+)$", text, re.M)
        require(len(values) == 1, "Missing/duplicate TAP summary")
        counters[key] = int(values[0])
    require(
        counters["tests"] == counters["pass"] == len(names)
        and all(counters[k] == 0 for k in ("fail", "cancelled", "skipped", "todo")),
        "Incomplete TAP summary",
    )
    require(re.findall(r"^1\.\.(\d+)$", text, re.M) == [str(len(names))], "TAP plan differs")
    return counters


def harness(args: argparse.Namespace, work: Path) -> dict:
    """Use the production isolation guard and current pinned actual Harness."""
    root = native.PRIMARY
    contract = harness_contract(root)
    release.save(work / "harness-contract.json", contract)
    inputs = {name: artifact(root / "apps/host" / name) for name in HARNESS_FILES}
    for name in ("test-fs-guard.mjs", "test-environment.mjs"):
        inputs[name] = artifact(root / "apps/host/scripts" / name)
    before = release.inventory(release.RES / "harness")
    release.save(work / "harness-inventory.json", before)
    inputs["harnessInventory"] = artifact(work / "harness-inventory.json")
    inspect = [release.NODE, HERE / "acceptance-harness.mjs", root, work, "inspect"]
    identity = json.loads(release.command(inspect))
    inputs["harnessBundle"] = artifact(release.RES / "harness/BUNDLE.json")
    inputs["harnessPatch"] = artifact(release.RES / "harness" / identity["patch"]["file"])
    inputs["harnessBudgetModule"] = artifact(root / "apps/desktop/scripts/harness-budget.mjs")
    release.save(work / "harness-identity.json", identity)
    command = native.logged(
        [release.NODE, HERE / "acceptance-harness.mjs", root, work, "run"], work, "harness-run"
    )
    counters = validate_tap((work / "harness.tap").read_text(encoding="utf-8"), contract["names"])
    require(release.inventory(release.RES / "harness") == before, "Managed Harness changed during run")
    require(json.loads(release.command(inspect)) == identity, "Harness identity changed")
    return {
        "scope": "Actual pinned Harness, synthetic model endpoints, serial source regressions; no model qualification",
        "inputs": inputs,
        "results": {
            name: artifact(work / name)
            for name in (
                "harness-contract.json",
                "harness-identity.json",
                "harness-command.json",
                "harness.tap",
                "harness-stderr.log",
                "harness-run.log",
            )
        },
        "command": command,
        "counters": counters,
    }


def validate_nsis(rows: list[dict]) -> None:
    """Check fixture observations, not just the producer's ok flag or exit code."""
    require(
        isinstance(rows, list) and len(rows) == 6 and {r.get("scenario") for r in rows} == SCENARIOS,
        "Exactly six named NSIS fixtures required",
    )
    for row in rows:
        blocked = row["scenario"] in {"locked", "junction", "root-junction"}
        require(
            row.get("ok") is True
            and row.get("userSentinelsPreserved") is True
            and row.get("oldEngineRemains") is blocked
            and row.get("newEngineCopied") is not blocked,
            "NSIS cleanup/sentinel observations differ",
        )
        require(
            type(row.get("exit")) is int and (row["exit"] != 0 if blocked else row["exit"] == 0),
            "NSIS fixture exit differs",
        )
        if not blocked:
            require(
                row.get("obsoleteManagedFolders") == []
                and row.get("uninstallExit") == 0
                and row.get("uninstallPreservedUserData") is True,
                "NSIS uninstall failed",
            )


def nsis(args: argparse.Namespace, work: Path) -> dict:
    """Exercise production cleanup macros in six fresh synthetic fixture trees."""
    script = native.PRIMARY / "tools/test_nsis_upgrade.ps1"
    installer = native.PRIMARY / "apps/desktop/src-tauri/nsis/installer.nsi"
    inputs = {
        "script": artifact(script),
        "installer": artifact(installer),
        "pwsh": artifact(PWSH),
        "compiler": artifact(NSIS),
    }
    version = release.command(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", "$PSVersionTable.PSVersion.ToString()"]
    )
    require(re.fullmatch(r"7\.\d+\.\d+(?:[-+].*)?", version), "Trusted PowerShell 7 required")
    command = native.logged(
        [
            PWSH,
            "-NoProfile",
            "-NonInteractive",
            "-File",
            script,
            "-OutputDir",
            work / "fixtures",
            "-NsisCompiler",
            NSIS,
        ],
        work,
        "nsis",
        900,
    )
    output = work / "fixtures/results.json"
    validate_nsis(release.load(output))
    require(
        (work / "fixtures/source-sha256.txt").read_text(encoding="utf-8-sig").strip()
        == inputs["installer"]["sha256"],
        "NSIS production source differs",
    )
    return {
        "scope": "Six synthetic cleanup fixtures, no real app installation",
        "inputs": inputs,
        "results": {
            "fixtures": artifact(output),
            "source": artifact(work / "fixtures/source-sha256.txt"),
            "log": artifact(work / "nsis.log"),
        },
        "command": command,
        "powerShellVersion": version,
    }


def validate_junit(path: Path) -> None:
    """Require each expected numerical/lifecycle case exactly once and unskipped."""
    tree = ET.fromstring(path.read_bytes())
    require(tree.tag == "testsuite" and tree.get("tests") == "10", "GPU JUnit suite differs")
    for key in ("failures", "errors", "skipped", "disabled"):
        require(tree.get(key, "0") == "0", "GPU JUnit nonpassing summary")
    cases = tree.findall("testcase")
    require(len(cases) == 10 and {c.get("name") for c in cases} == GPU_NAMES, "GPU JUnit case set differs")
    require(
        all(
            c.get("status") == "run"
            and not any(c.find(k) is not None for k in ("failure", "error", "skipped"))
            for c in cases
        ),
        "GPU JUnit failed/skipped case",
    )


def validate_gpu(binding: dict, report: dict, guard: dict, digest: str, binaries: dict) -> None:
    """Carry only attested unchanged source digest and both exact product binaries."""
    require(
        Path(binding.get("sourceRoot", "")) == native.PRIMARY
        and binding.get("sourceDigest") == digest
        and binding.get("sourceUnchanged") is True
        and binding.get("binariesUnchanged") is True
        and binding.get("releaseQualified") is False
        and binding.get("exitCode") == 0,
        "GPU source attestation differs",
    )
    expected = {str(native.BUILD / "Release" / name): value for name, value in binaries.items()}
    # Windows-normalise paths; the historical record uses backslashes.
    for label in ("before", "after"):
        require(
            {str(Path(k)): v for k, v in binding.get(label, {}).items()} == expected,
            "GPU binaries differ; fresh bounded GPU proof required",
        )
    require(
        set(report.get("registeredRequestedTests", [])) == GPU_NAMES
        and len(report["registeredRequestedTests"]) == 10
        and report.get("exitCode") == 0
        and report.get("reason") is None
        and report.get("admissionFreeMiB", 0) >= 8192
        and report.get("reserveFloorMiB") == 768
        and report.get("minimumFreeMiB", 0) >= 768
        and report.get("samples", 0) > 0,
        "GPU report/floors differ",
    )
    require(guard.get("passed") is True, "GPU guard failed")
    successful_job(guard.get("run", {}))
    for key, floor in (("available_ram", 3), ("commit_headroom", 4), ("H_free", 10), ("C_free", 5)):
        require(guard.get("admission", {}).get(key, 0) >= floor * release.GIB, "GPU admission floor violated")


def final_native(args: argparse.Namespace, work: Path) -> dict:
    """Read fresh native/component receipts and qualify historical GPU equality only."""
    inputs = {}
    for name in EXTERNAL_INPUTS:
        path, digest = getattr(args, name + "_receipt"), getattr(args, name + "_receipt_sha256")
        require(path is not None and digest is not None, "Explicit pinned " + name + " input required")
        inputs[name] = artifact(path, digest)
        if name in GPU_PINS:
            require(digest == GPU_PINS[name], "Only the reviewed historical GPU evidence may be carried")
    built = delivery.receipt(
        args.native_receipt, args.native_receipt_sha256, "native-input", args.reviewed_commit
    )
    feed.receipt_assets(
        args.component_receipt, args.component_receipt_sha256, "component", args.reviewed_commit
    )
    component_record = release.load(args.component_receipt)
    component = Path(component_record["signedComponent"])
    for name, digest in component_record["files"].items():
        inputs["component:" + name] = artifact(component / name, digest)
    unsigned = Path(built["nativeComponent"])
    require(unsigned == args.native_receipt.parent / "unsigned" / release.NAME, "Foreign unsigned component")
    require(release.inventory(unsigned) == built["files"], "Unsigned native changed")
    binding_path = args.native_receipt.parent / "native-binding.json"
    binding = release.load(binding_path)
    require(
        release.sha(binding_path) == component_record["nativeBindingSHA256"],
        "Component native binding mismatch",
    )
    require(
        binding["receiptSHA256"] == args.native_receipt_sha256
        and binding["ownedJobSHA256"] == release.sha(args.native_receipt.parent / "owned-job.json"),
        "Native binding job mismatch",
    )
    require(
        binding.get("builder") == built.get("builder") and binding["builder"].get("exitCode") == 0,
        "Native builder receipt differs",
    )
    for label, key in (
        ("inputs-before.json", "inputsBeforeSHA256"),
        ("inputs-after.json", "inputsAfterSHA256"),
        ("toolkit-inputs.json", "toolkitInputsSHA256"),
    ):
        inputs[label] = artifact(args.native_receipt.parent / label, binding[key])
    inputs["builderLog"] = artifact(
        args.native_receipt.parent / "builder.log", binding["builder"]["logSHA256"]
    )
    release.verify("native", component, args, binding_path)
    build = release.load(component / "native-build.json")
    configured = native.configured_inputs(native.BUILD)
    require(
        configured["baseline"]["path"] == PROFILE, "Only the explicitly reviewed Windows profile is approved"
    )
    inputs["cudaProfile"] = artifact(
        native.PRIMARY / configured["baseline"]["path"], configured["baseline"]["sha256"]
    )
    require(
        release.load(args.native_receipt.parent / "inputs-before.json")
        == configured
        == release.load(args.native_receipt.parent / "inputs-after.json"),
        "Native configured inputs changed",
    )
    require(
        build["sources"] == configured["sources"]
        and build["recipe"]["ggml"] == configured["ggml"]
        and build["recipe"].get("cudaSpillBaseline") == configured["baseline"],
        "Fresh native source/profile differs",
    )
    require(
        release.sha(native.BUILD / "Release/native-build.json")
        == release.sha(component / "native-build.json"),
        "Current build metadata differs",
    )
    libraries = release.load(args.native_receipt.parent / "toolkit-inputs.json")
    require(
        libraries["libraries"]
        and libraries["libraries"] == native.toolkit_libraries(native.CUDA / "bin/x64"),
        "Current toolkit PE dependency inventory differs",
    )
    require(
        libraries["versionSHA256"] == release.sha(native.CUDA / "version.json"), "Toolkit version changed"
    )
    inputs["toolkitVersion"] = artifact(native.CUDA / "version.json", libraries["versionSHA256"])
    for name, digest in libraries["libraries"].items():
        inputs["library:" + name] = artifact(component / name, digest)
    binaries = {n: release.sha(component / n) for n in ("strata.exe", "strata-q36.exe")}
    probes = {}
    for name, digest in binaries.items():
        require(
            release.sha(unsigned / name) == release.sha(native.BUILD / "Release" / name) == digest,
            "Current native executable differs",
        )
        probes[name] = release.command([component / name, "--version"], timeout=20)
        require(re.search(r"(^|\s)0\.6\.0(\s|$)", probes[name]), "Native probe version differs")
        inputs[name] = artifact(component / name, digest)
    deps = release.module("acceptance_pe", native.PRIMARY / "packaging/binary_deps.py")
    for folder in (component, component / "local-backends/prism", component / "local-backends/turboquant"):
        require(
            not deps.uncovered(folder, [p.name for p in folder.iterdir() if p.is_file()]),
            "PE closure incomplete",
        )
    sys.path.insert(0, str(native.PRIMARY))
    from tools import cuda_spill_baseline as measured

    compiler_key = native.BUILD / "cuda-spill-build-key.txt"
    profile_path = native.PRIMARY / PROFILE
    profile_bindings = measured.bindings(
        native.PRIMARY, compiler_key, native.PRIMARY / "third_party/strata-ggml"
    )
    measured.load(profile_path, profile_bindings)
    measured_outputs = compiled_libraries(release.load(profile_path), native.BUILD / "Release")
    inputs.update({"compiled:" + name: row for name, row in measured_outputs.items()})
    inputs["compilerKey"] = artifact(compiler_key)
    digest = profile_bindings["sourceDigest"]
    gpu_binding, gpu_report, gpu_guard = (
        release.load(getattr(args, n + "_receipt")) for n in ("gpu_binding", "gpu_report", "gpu_guard")
    )
    validate_junit(args.gpu_junit_receipt)
    validate_gpu(gpu_binding, gpu_report, gpu_guard, digest, binaries)
    require(gpu_binding["reportSHA256"] == args.gpu_report_receipt_sha256, "GPU report binding differs")
    require(
        gpu_report["command"][-2:] == ["--output-junit", str(args.gpu_junit_receipt)],
        "GPU JUnit command binding differs",
    )
    log = gpu_guard["command_log"]
    inputs["gpuCommandLog"] = artifact(Path(log["path"]), log["sha256"])
    for name, receipt_path in (
        ("native_job", args.native_receipt),
        ("component_job", args.component_receipt),
    ):
        path = receipt_path.parent / "owned-job.json"
        require(path == getattr(args, name + "_receipt"), "Owned Job must belong to the pinned phase attempt")
        successful_job(release.load(path))
        inputs[name + "Log"] = artifact(receipt_path.parent / "worker.log")
    inputs["nativeBinding"] = artifact(binding_path)
    inputs["nativeBuild"] = artifact(component / "native-build.json")
    require(measured.source_digest(native.PRIMARY) == digest, "Native sources changed during acceptance")
    equality = {
        "claim": (
            "Historical GPU replay carried by native source digest and both actual binary equalities; "
            "not a final-source GPU execution"
        ),
        "historicalSourceDigest": gpu_binding["sourceDigest"],
        "historicalSourceUnchangedAttested": gpu_binding["sourceUnchanged"],
        "currentSourceDigest": digest,
        "profileBindings": profile_bindings,
        "compiledLibraries": measured_outputs,
        "binaries": binaries,
        "versionProbes": probes,
        "versionProbeCommands": [[str(component / name), "--version"] for name in binaries],
        "retainedLimitation": (
            "Generic spill-default profile has two IQ2 failures; explicit bound Windows profile only; "
            "model quality, full window and concurrency unqualified"
        ),
    }
    release.save(work / "native-equality.json", equality)
    return {
        "scope": equality["claim"],
        "inputs": inputs,
        "results": {"equality": artifact(work / "native-equality.json")},
    }


PHASES = {"ci": ci, "harness": harness, "nsis": nsis, "final-native": final_native}


BASE_ENVIRONMENT = native.environment


def environment(work: Path, token: str) -> dict[str, str]:
    """Extend only the existing filtered environment with trusted shell locations."""
    env = BASE_ENVIRONMENT(work, token)
    windows_ps = Path(os.environ["SYSTEMROOT"]) / "System32/WindowsPowerShell/v1.0"
    require(
        (windows_ps / "powershell.exe").is_file() and PWSH.is_file(), "Trusted shell runtimes unavailable"
    )
    env["PATH"] = str(PWSH.parent) + ";" + str(windows_ps) + ";" + env["PATH"]
    env["PATHEXT"] = ".COM;.EXE;.BAT;.CMD"
    return env


def phase(args: argparse.Namespace, work: Path) -> dict:
    """Bind the adapter, command and every result to one reviewed worker phase."""
    before = release.reviewed(args.reviewed_commit)
    inputs = {name: artifact(HERE / name) for name in ADAPTERS}
    inputs["nodeExecutable"] = artifact(release.NODE)
    command = [sys.executable, "-B", *sys.argv]
    release.save(work / "worker-command.json", command)
    import ctypes
    from ctypes import wintypes
    from perf131k.windows_job import api

    times = [ctypes.c_ulonglong() for _ in range(4)]
    require(
        api().GetProcessTimes(wintypes.HANDLE(-1), *[ctypes.byref(t) for t in times]),
        "Cannot observe worker identity",
    )
    release.save(
        work / "worker-context.json",
        {
            "schema": "strata.acceptance.worker.v1",
            "command": command,
            "pid": os.getpid(),
            "creationFiletime": times[0].value,
            "attemptRoot": str(work),
            "reservationSHA256": release.sha(work / "reservation.json"),
        },
    )
    result = PHASES[args.mode](args, work)
    result["inputs"].update(inputs)
    result["results"]["workerCommand"] = artifact(work / "worker-command.json")
    result["results"]["workerContext"] = artifact(work / "worker-context.json")
    check_artifacts(result["inputs"])
    check_artifacts(result["results"])
    require(release.reviewed(args.reviewed_commit) == before, "Acceptance source changed")
    return {"schema": SCHEMA, "reviewedSourceRoot": str(native.PRIMARY), "mode": args.mode, **result}


def parsed(argv: list[str]) -> argparse.Namespace:
    """The same argv contract is used before execution and by the pure consumer."""
    cli = native.parser(__doc__)
    cli.add_argument("mode", choices=PHASES)
    for name in EXTERNAL_INPUTS:
        option = name.replace("_", "-")
        cli.add_argument("--" + option + "-receipt", type=Path)
        cli.add_argument("--" + option + "-receipt-sha256")
    args = cli.parse_args(argv)
    if args.execute:
        require(
            sum(v == "--attempt" or v.startswith("--attempt=") for v in argv) == 1,
            "Execution requires one explicit --attempt before controller admission",
        )
    args.phase = "acceptance-" + args.mode
    return args


def owned_binding(original: Any, args: argparse.Namespace, argv: list[str]) -> Any:
    """Annotate the real existing guard result with the actual command it received."""
    expected = [sys.executable, "-B", str(HERE / "acceptance.py"), *argv, "--worker"]
    work = release.OUT / (args.phase + "-" + args.attempt)

    def run(command: list[str], root: Path, *positional: Any, **keywords: Any) -> dict:
        require(list(command) == expected and root == native.PRIMARY, "Unexpected owned acceptance launch")
        reservation_sha = release.sha(work / "reservation.json")
        result = original(command, root, *positional, **keywords)
        result["acceptanceExecution"] = {
            "schema": "strata.acceptance.execution.v1",
            "command": list(command),
            "cwd": str(root),
            "attemptRoot": str(work),
            "reservationSHA256": reservation_sha,
            "adapterSHA256": release.sha(HERE / "acceptance.py"),
        }
        return result

    return run


def evidence(record: dict, section: str, name: str, expected_path: Path | None = None) -> Path:
    """Require a named semantic artifact at its exact producer-owned location."""
    row = record.get(section, {}).get(name)
    require(isinstance(row, dict) and set(row) == {"path", "sha256"}, "Missing required evidence: " + name)
    path = Path(row["path"])
    require(expected_path is None or path == expected_path, "Foreign evidence path: " + name)
    artifact(path, row["sha256"])
    return path


def validate_execution(record: dict, work: Path, job: dict) -> None:
    """Bind argv, source, external pins, reservation and actual worker process identity."""
    command = release.load(evidence(record, "results", "workerCommand", work / "worker-command.json"))
    require(
        isinstance(command, list)
        and len(command) > 3
        and Path(command[0]) == Path(sys.executable)
        and command[1] == "-B"
        and Path(command[2]) == HERE / "acceptance.py",
        "Foreign acceptance command",
    )
    try:
        args = parsed(command[3:])
    except SystemExit as error:
        raise RuntimeError("Malformed worker command") from error
    require(
        args.execute
        and args.worker
        and args.mode == record["mode"]
        and args.reviewed_commit == record["sourceCommit"]
        and work == release.OUT / (args.phase + "-" + args.attempt),
        "Worker command mode/source/attempt differs",
    )
    for name in EXTERNAL_INPUTS:
        path, digest = getattr(args, name + "_receipt"), getattr(args, name + "_receipt_sha256")
        if args.mode == "final-native":
            require(
                path is not None and record["inputs"].get(name) == {"path": str(path), "sha256": digest},
                "Worker external input argument differs: " + name,
            )
        else:
            require(path is None and digest is None, "Unexpected external input arguments")
    context = release.load(evidence(record, "results", "workerContext", work / "worker-context.json"))
    execution = job.get("acceptanceExecution", {})
    reservation_sha = release.sha(work / "reservation.json")
    reservation = release.load(work / "reservation.json")
    require(
        reservation.get("identity") == {k: record[k] for k in ("sourceCommit", "sourceRoot", "tree")},
        "Attempt reservation source differs",
    )
    require(
        execution
        == {
            "schema": "strata.acceptance.execution.v1",
            "command": command,
            "cwd": str(native.PRIMARY),
            "attemptRoot": str(work),
            "reservationSHA256": reservation_sha,
            "adapterSHA256": record["inputs"]["acceptance.py"]["sha256"],
        },
        "Job was not the recorded acceptance command/attempt",
    )
    require(
        context
        == {
            "schema": "strata.acceptance.worker.v1",
            "command": command,
            "pid": job.get("owned_root_pid"),
            "creationFiletime": job.get("root_creation_filetime"),
            "attemptRoot": str(work),
            "reservationSHA256": reservation_sha,
        }
        and type(context.get("creationFiletime")) is int
        and context["creationFiletime"] > 0,
        "Worker identity does not match the successful owned Job",
    )


def validate_mode(record: dict, work: Path) -> None:
    """Revalidate required mode proof without rerunning production phases."""
    mode, inputs = record["mode"], record["inputs"]
    if mode == "ci":
        evidence(record, "inputs", "ghExecutable", GH)
        commands = []
        for filename, (name, digest, _) in WORKFLOWS.items():
            evidence(record, "inputs", filename, native.PRIMARY / ".github/workflows" / filename)
            require(inputs[filename]["sha256"] == digest, "Unreviewed CI workflow bytes")
            row = release.load(evidence(record, "results", filename, work / (filename + ".json")))
            require(row.get("workflowFile") == filename, "CI workflow file differs")
            validate_ci_run(row, filename, record["sourceCommit"])
            commands.extend(
                [
                    [
                        str(GH),
                        "run",
                        "list",
                        "--workflow",
                        filename,
                        "--commit",
                        record["sourceCommit"],
                        "--branch",
                        "main",
                        "--limit",
                        "100",
                        "--json",
                        "databaseId,workflowName,headSha,headBranch,status,conclusion",
                        "--repo",
                        REPO,
                    ],
                    [
                        str(GH),
                        "run",
                        "view",
                        str(row["databaseId"]),
                        "--json",
                        "databaseId,workflowName,headSha,headBranch,status,conclusion,jobs",
                        "--repo",
                        REPO,
                    ],
                ]
            )
        require(
            release.load(evidence(record, "results", "commands", work / "commands.json")) == commands,
            "CI commands differ from recorded workflow/source/run identities",
        )
    elif mode == "harness":
        for name in HARNESS_FILES:
            evidence(record, "inputs", name, native.PRIMARY / "apps/host" / name)
        for name in ("test-fs-guard.mjs", "test-environment.mjs"):
            evidence(record, "inputs", name, native.PRIMARY / "apps/host/scripts" / name)
        contract = release.load(
            evidence(record, "results", "harness-contract.json", work / "harness-contract.json")
        )
        require(contract == harness_contract(native.PRIMARY), "Harness required names/files differ")
        counters = validate_tap(
            evidence(record, "results", "harness.tap", work / "harness.tap").read_text(encoding="utf-8"),
            contract["names"],
        )
        require(counters == record.get("counters"), "Harness counters differ")
        identity = release.load(
            evidence(record, "results", "harness-identity.json", work / "harness-identity.json")
        )
        bundle = release.load(
            evidence(record, "inputs", "harnessBundle", release.RES / "harness/BUNDLE.json")
        )
        patch = identity.get("patch", {})
        require(
            identity.get("runtime") == "v24.22.3"
            and identity.get("commit") == bundle.get("commit") == "571902028a75c64c1d767f4ae5fd099485f0ddf0"
            and identity.get("bundleSHA256") == inputs["harnessBundle"]["sha256"]
            and identity.get("importClosure") == "complete",
            "Harness runtime/bundle/import proof differs",
        )
        budget = evidence(
            record,
            "inputs",
            "harnessBudgetModule",
            native.PRIMARY / "apps/desktop/scripts/harness-budget.mjs",
        ).read_text(encoding="utf-8")
        for field in ("id", "file", "beforeSha256"):
            expected = re.findall(r"\b" + field + r": '([^']+)'", budget)
            require(
                len(expected) == 1 and patch.get(field) == expected[0],
                "Harness budget patch contract differs",
            )
        require(
            [p for p in bundle.get("patches", []) if p.get("id") == patch["id"]] == [patch],
            "Harness bundle patch proof differs",
        )
        evidence(record, "inputs", "harnessPatch", release.RES / "harness" / patch["file"])
        require(patch.get("afterSha256") == inputs["harnessPatch"]["sha256"], "Harness patch bytes differ")
        inventory = release.load(
            evidence(record, "inputs", "harnessInventory", work / "harness-inventory.json")
        )
        require(
            inventory.get("BUNDLE.json") == inputs["harnessBundle"]["sha256"]
            and inventory.get(patch["file"]) == patch["afterSha256"],
            "Harness inventory proof differs",
        )
        command = [
            str(release.NODE),
            "--import",
            (native.PRIMARY / "apps/host/scripts/test-fs-guard.mjs").as_uri(),
            "--test",
            "--test-reporter=tap",
            "--test-concurrency=1",
            "--test-timeout=600000",
            *contract["files"],
        ]
        require(
            release.load(evidence(record, "results", "harness-command.json", work / "harness-command.json"))
            == command,
            "Harness test command differs",
        )
        evidence(record, "results", "harness-stderr.log", work / "harness-stderr.log")
        log = evidence(record, "results", "harness-run.log", work / "harness-run.log")
        require(
            record.get("command")
            == {
                "command": [
                    str(release.NODE),
                    str(HERE / "acceptance-harness.mjs"),
                    str(native.PRIMARY),
                    str(work),
                    "run",
                ],
                "exitCode": 0,
                "logSHA256": release.sha(log),
            },
            "Harness launcher proof differs",
        )
    elif mode == "nsis":
        for key, path in (
            ("script", native.PRIMARY / "tools/test_nsis_upgrade.ps1"),
            ("installer", native.PRIMARY / "apps/desktop/src-tauri/nsis/installer.nsi"),
            ("pwsh", PWSH),
            ("compiler", NSIS),
        ):
            evidence(record, "inputs", key, path)
        validate_nsis(release.load(evidence(record, "results", "fixtures", work / "fixtures/results.json")))
        require(
            evidence(record, "results", "source", work / "fixtures/source-sha256.txt")
            .read_text(encoding="utf-8-sig")
            .strip()
            == inputs["installer"]["sha256"],
            "NSIS source proof differs",
        )
        log = evidence(record, "results", "log", work / "nsis.log")
        require(
            re.fullmatch(r"7\.\d+\.\d+(?:[-+].*)?", record.get("powerShellVersion", "")),
            "NSIS PowerShell version proof missing",
        )
        require(
            record.get("command")
            == {
                "command": [
                    str(PWSH),
                    "-NoProfile",
                    "-NonInteractive",
                    "-File",
                    inputs["script"]["path"],
                    "-OutputDir",
                    str(work / "fixtures"),
                    "-NsisCompiler",
                    str(NSIS),
                ],
                "exitCode": 0,
                "logSHA256": release.sha(log),
            },
            "NSIS fixture command differs",
        )
    else:
        validate_native_proof(record, work)


def validate_native_proof(record: dict, work: Path) -> None:
    """Consume bound native/profile/GPU evidence with read-only production parsers."""
    inputs = record["inputs"]
    for name in EXTERNAL_INPUTS:
        evidence(record, "inputs", name)
        if name in GPU_PINS:
            require(inputs[name]["sha256"] == GPU_PINS[name], "Unreviewed historical GPU evidence")
    equality = release.load(evidence(record, "results", "equality", work / "native-equality.json"))
    component_receipt = Path(inputs["component"]["path"])
    feed.receipt_assets(component_receipt, inputs["component"]["sha256"], "component", record["sourceCommit"])
    component = release.load(component_receipt)
    signed = Path(component["signedComponent"])
    for name, digest in component["files"].items():
        evidence(record, "inputs", "component:" + name, signed / name)
        require(inputs["component:" + name]["sha256"] == digest, "Signed component proof differs")
    native_receipt = Path(inputs["native"]["path"])
    built = delivery.receipt(
        native_receipt, inputs["native"]["sha256"], "native-input", record["sourceCommit"]
    )
    binding = release.load(
        evidence(record, "inputs", "nativeBinding", native_receipt.parent / "native-binding.json")
    )
    require(
        inputs["nativeBinding"]["sha256"] == component["nativeBindingSHA256"]
        and binding["receiptSHA256"] == inputs["native"]["sha256"]
        and binding["ownedJobSHA256"] == inputs["native_job"]["sha256"],
        "Native builder binding differs",
    )
    for label, key in (
        ("inputs-before.json", "inputsBeforeSHA256"),
        ("inputs-after.json", "inputsAfterSHA256"),
        ("toolkit-inputs.json", "toolkitInputsSHA256"),
    ):
        evidence(record, "inputs", label, native_receipt.parent / label)
        require(inputs[label]["sha256"] == binding[key], "Native bound input receipt differs")
    before = release.load(Path(inputs["inputs-before.json"]["path"]))
    require(
        before == release.load(Path(inputs["inputs-after.json"]["path"])), "Native before/after proof differs"
    )
    require(
        binding.get("builder") == built.get("builder") and binding["builder"].get("exitCode") == 0,
        "Native builder result missing",
    )
    artifact(
        evidence(record, "inputs", "builderLog", native_receipt.parent / "builder.log"),
        binding["builder"]["logSHA256"],
    )
    for name, path in (("native_job", native_receipt), ("component_job", component_receipt)):
        successful_job(release.load(evidence(record, "inputs", name, path.parent / "owned-job.json")))
    build = release.load(evidence(record, "inputs", "nativeBuild", signed / "native-build.json"))
    require(
        inputs["nativeBuild"]["sha256"] == binding.get("nativeBuildSHA256"),
        "Native build receipt hash differs",
    )
    require(
        build.get("schema") == 1
        and build.get("version") == "0.6.0"
        and build.get("sources") == before.get("sources")
        and build["recipe"].get("ggml") == before.get("ggml"),
        "Native build/source/dependency proof differs",
    )
    profile_path = evidence(record, "inputs", "cudaProfile", native.PRIMARY / PROFILE)
    require(
        build["recipe"].get("cudaSpillBaseline")
        == {"path": PROFILE, "sha256": inputs["cudaProfile"]["sha256"]},
        "Native profile metadata differs",
    )
    require(
        before.get("baseline") == build["recipe"]["cudaSpillBaseline"], "Native configured profile differs"
    )
    toolkit = release.load(Path(inputs["toolkit-inputs.json"]["path"]))
    require(
        isinstance(toolkit.get("libraries"), dict) and toolkit["libraries"],
        "Toolkit PE dependency evidence missing",
    )
    artifact(
        evidence(record, "inputs", "toolkitVersion", native.CUDA / "version.json"), toolkit["versionSHA256"]
    )
    for name, digest in toolkit["libraries"].items():
        artifact(evidence(record, "inputs", "library:" + name, signed / name), digest)
    profile = release.load(profile_path)
    compiled = compiled_libraries(profile, native.BUILD / "Release")
    require(equality.get("compiledLibraries") == compiled, "Compiled-library equality proof differs")
    require(
        {key.removeprefix("compiled:") for key in inputs if key.startswith("compiled:")}
        == COMPILED_LIBRARIES,
        "Compiled-library input set differs",
    )
    for name, row in compiled.items():
        require(inputs["compiled:" + name] == row, "Compiled-library input differs")
    key_path = evidence(record, "inputs", "compilerKey", native.BUILD / "cuda-spill-build-key.txt")
    sys.path.insert(0, str(native.PRIMARY))
    from tools import cuda_spill_baseline as measured

    bindings = measured.bindings(native.PRIMARY, key_path, native.PRIMARY / "third_party/strata-ggml")
    measured.load(profile_path, bindings)
    require(
        equality.get("profileBindings") == bindings
        and equality.get("currentSourceDigest") == bindings["sourceDigest"],
        "Native source/compiler/dependency proof differs",
    )
    binaries = {name: inputs[name]["sha256"] for name in ("strata.exe", "strata-q36.exe")}
    require(binding.get("binaries") == binaries, "Native binding executable identities differ")
    for name, digest in binaries.items():
        evidence(record, "inputs", name, signed / name)
        require(
            built["files"].get(name) == digest
            and re.search(r"(^|\s)0\.6\.0(\s|$)", equality.get("versionProbes", {}).get(name, "")),
            "Native binary/version proof differs",
        )
    require(
        equality.get("binaries") == binaries
        and equality.get("versionProbeCommands") == [[str(signed / n), "--version"] for n in binaries],
        "Native probe command differs",
    )
    historical = release.load(Path(inputs["gpu_binding"]["path"]))
    report = release.load(Path(inputs["gpu_report"]["path"]))
    guard = release.load(Path(inputs["gpu_guard"]["path"]))
    validate_junit(Path(inputs["gpu_junit"]["path"]))
    validate_gpu(historical, report, guard, bindings["sourceDigest"], binaries)
    require(
        equality.get("historicalSourceDigest") == historical["sourceDigest"]
        and equality.get("historicalSourceUnchangedAttested") is True
        and historical["reportSHA256"] == inputs["gpu_report"]["sha256"],
        "GPU equality proof differs",
    )
    require(
        report["command"][-2:] == ["--output-junit", inputs["gpu_junit"]["path"]], "GPU JUnit linkage differs"
    )
    log = evidence(record, "inputs", "gpuCommandLog", Path(guard["command_log"]["path"]))
    artifact(log, guard["command_log"]["sha256"])


def validate_proof(record: dict, work: Path, job: dict) -> None:
    """Shared producer-after-drain and consumer semantic gate."""
    check_artifacts(record["inputs"])
    check_artifacts(record["results"])
    require(record.get("scope") == SCOPES[record["mode"]], "Acceptance scope differs")
    for name in ADAPTERS:
        evidence(record, "inputs", name, HERE / name)
    evidence(record, "inputs", "nodeExecutable", release.NODE)
    validate_execution(record, work, job)
    validate_mode(record, work)


def validate_acceptance(path: Path, expected_sha: str, mode: str, commit: str) -> dict:
    """Consumer API: explicit post-drain artifact pin, exact source and all bytes."""
    record = delivery.pinned(path, expected_sha)
    require(mode in PHASES and path.name == "acceptance.json", "Unknown acceptance mode/file")
    phase_name = "acceptance-" + mode
    receipt_path = path.parent / "receipt.json"
    raw = delivery.receipt(receipt_path, record["receiptSHA256"], phase_name, commit)
    require(
        record.get("schema") == SCHEMA
        and record.get("mode") == mode
        and record.get("sourceCommit") == commit
        and record.get("phase") == phase_name
        and record.get("reviewedSourceRoot") == str(native.PRIMARY)
        and record.get("releaseQualified") is False,
        "Acceptance identity differs",
    )
    require(
        {k: v for k, v in record.items() if k not in ("receiptSHA256", "ownedJobSHA256", "workerLogSHA256")}
        == raw,
        "Acceptance result differs from drained receipt",
    )
    artifact(path.parent / "owned-job.json", record["ownedJobSHA256"])
    artifact(path.parent / "worker.log", record["workerLogSHA256"])
    job = release.load(path.parent / "owned-job.json")
    successful_job(job)
    validate_proof(record, path.parent, job)
    artifact(path, expected_sha)
    return record


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parsed(argv)
    if args.execute and args.mode == "final-native":
        for name in EXTERNAL_INPUTS:
            path = getattr(args, name + "_receipt")
            digest = getattr(args, name + "_receipt_sha256")
            require(
                path is not None and path.is_absolute() and re.fullmatch(r"[a-f0-9]{64}", digest or ""),
                "Explicit absolute input and expected SHA required: " + name,
            )
    native.environment = environment
    if args.execute and not args.worker:
        sys.path.insert(0, str(native.PRIMARY / "tools"))
        from perf131k import windows_job

        original = windows_job.run_owned
        windows_job.run_owned = owned_binding(original, args, argv)
        try:
            result = native.run_controller(args, HERE / "acceptance.py", phase, argv)
        finally:
            windows_job.run_owned = original
    else:
        result = native.run_controller(args, HERE / "acceptance.py", phase, argv)
    if args.execute and not args.worker:
        work = release.OUT / (args.phase + "-" + args.attempt)
        raw = release.load(work / "receipt.json")
        job = release.load(work / "owned-job.json")
        successful_job(job)
        validate_proof(raw, work, job)
        release.save(
            work / "acceptance.json",
            {
                **raw,
                "receiptSHA256": release.sha(work / "receipt.json"),
                "ownedJobSHA256": release.sha(work / "owned-job.json"),
                "workerLogSHA256": release.sha(work / "worker.log"),
            },
        )
        validate_acceptance(
            work / "acceptance.json", release.sha(work / "acceptance.json"), args.mode, args.reviewed_commit
        )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        raise SystemExit(main())
    except Exception as error:
        logging.error(
            "Acceptance failed (%s); attempt evidence retained; raw diagnostics suppressed",
            type(error).__name__,
        )
        raise SystemExit(1) from None
