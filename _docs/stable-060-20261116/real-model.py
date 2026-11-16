"""One bounded installed 0.6 real-model attempt. Default: read-only plan.

Preparation tests never execute this controller. Explicit execution uses the
existing resource guard and an owned Windows Job; no download or publication.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("real_model_acceptance", HERE / "acceptance.py")
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)
delivery, native, release = acceptance.delivery, acceptance.native, acceptance.release
require, artifact = release.require, acceptance.artifact
SCHEMA = "strata.real-model.v1"
INPUTS = ("app", "extract", "install", "installed_qa", "component")
ADAPTERS = (
    "real-model.py",
    "real-model.mjs",
    "real-model-mcp.mjs",
    "real-model-runtime-fixture.mjs",
    "qa-installed.mjs",
    "acceptance.py",
    "delivery.py",
    "components-feed.py",
    "native-input.py",
    "release.py",
)
RESULTS = (
    "input.json",
    "node-command.json",
    "worker-command.json",
    "worker-context.json",
    "capability.json",
    "observations.json",
    "original-config-before.json",
    "original-config-after.json",
    "producer.log",
)
CSV = {
    "inventory": "id,version\nalpha,1\nbeta,2\n",
    "expected": "id,expected_version\nalpha,2\nbeta,2\ngamma,3\n",
    "requirements": "id,required_check,status\nalpha,checksum,passed\nbeta,checksum,failed\ngamma,checksum,passed\n",
}
MODEL_SHA = "8cf793a36e62fb078ac05c2e1bdb2121ff94736dccbede7cf81b76346e868949"
MODEL_SIZE = 5629111584
PRISM_COMMIT = "1965755ceee6738b9a04ceaf5d30e6305d6ccd00"
CHECKS = (
    "installed",
    "component",
    "runtime",
    "chat",
    "audit",
    "btw",
    "learning",
    "autoApply",
    "research",
    "memoryStore",
    "memoryRecall",
    "cleanup",
)


def parsed(argv: list[str]) -> argparse.Namespace:
    cli = native.parser(__doc__)
    for name in INPUTS:
        cli.add_argument("--" + name.replace("_", "-") + "-receipt", type=Path)
        cli.add_argument("--" + name.replace("_", "-") + "-receipt-sha256")
    args = cli.parse_args(argv)
    args.phase = "real-model"
    require(not args.worker or args.execute, "Worker requires execute")
    if args.execute:
        require(
            sum(v == "--attempt" or v.startswith("--attempt=") for v in argv) == 1,
            "One explicit attempt required",
        )
        require(re.fullmatch(r"r[1-9][0-9]*", args.attempt), "Invalid attempt")
        for name in INPUTS:
            path, sha = getattr(args, name + "_receipt"), getattr(args, name + "_receipt_sha256")
            require(
                path is not None
                and path.is_absolute()
                and re.fullmatch(r"[a-f0-9]{64}", sha or ""),
                "Explicit pinned input required: " + name,
            )
    return args


def gpu_snapshot() -> dict:
    """Only numeric, non-secret NVIDIA observations; never stop another owner."""
    executable = Path("C:/Windows/System32/nvidia-smi.exe")
    require(executable.is_file(), "GPU observation unavailable")

    def query(args: list[str]) -> str:
        result = subprocess.run(
            [str(executable), *args, "--format=csv,noheader,nounits"],
            capture_output=True,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        require(result.returncode == 0, "GPU observation failed")
        return result.stdout.decode("utf-8", errors="strict").strip()

    rows = [
        [int(n.strip()) for n in line.split(",")]
        for line in query(["--query-gpu=index,memory.free"]).splitlines()
    ]
    pids = [
        int(line.strip())
        for line in query(["--query-compute-apps=pid"]).splitlines()
        if line.strip()
    ]
    require(
        len(rows) == 1 and rows[0][0] == 0 and rows[0][1] >= 8192 and not pids,
        "GPU admission requires one idle GPU with 8192 MiB free",
    )
    return {"device": 0, "freeMiB": rows[0][1], "computePIDs": pids, "reserveMiB": 768}


def selected_original_hashes() -> dict:
    """Hash only fixed configuration files; never enumerate SOUL or memory DBs."""
    roots = delivery.USER_ROOTS[:3]
    rows = {}
    for root in roots:
        for relative in ("settings.json", "data/state.json", "desktop/shell-prefs.json"):
            path = root / relative
            release.no_links(path)
            rows[str(path)] = artifact(path) if path.is_file() else None
    return rows


def admission(args: argparse.Namespace, work: Path) -> dict:
    build, extracted, _ = delivery.extraction(args)
    installed = delivery.prior(args, "install")
    qa = delivery.prior(args, "installed_qa", "installed-qa")
    require(
        installed.get("installerSHA256") == build["installerSHA256"],
        "Installation belongs to another installer",
    )
    require(
        delivery.payload(delivery.APP) == extracted["payload"], "Physical installed payload differs"
    )
    require(
        qa.get("installerSHA256") == build["installerSHA256"],
        "Installed QA belongs to another installer",
    )
    delivery.validate_qa(qa["qa"], "installed", args.reviewed_commit, build["installerSHA256"])
    require(
        release.sha(args.installed_qa_receipt.parent / "qa-home/result.json") == qa["resultSHA256"],
        "Installed QA evidence changed",
    )
    component = delivery.feed.receipt_assets(
        args.component_receipt, args.component_receipt_sha256, "component", args.reviewed_commit
    )
    model = artifact(delivery.MODEL, MODEL_SHA)
    require(delivery.MODEL.stat().st_size == MODEL_SIZE, "Model size differs")
    delivery.quiescent()
    gpu = gpu_snapshot()
    # Same production modules and pinned SDK as the installed payload. The
    # synthetic proof is prevention by configuration, not a live endpoint claim.
    for path in (native.PRIMARY / "apps/host/server").rglob("*.mjs"):
        target = (
            delivery.APP / "host/server" / path.relative_to(native.PRIMARY / "apps/host/server")
        )
        require(
            target.is_file() and release.sha(path) == release.sha(target),
            "Installed host is not the tested source",
        )
    require(
        delivery.files(release.RES / "harness") == delivery.files(delivery.APP / "harness"),
        "Installed Harness differs",
    )
    require(
        release.sha(release.NODE) == release.sha(delivery.APP / "node/node.exe"),
        "Installed Node differs from tested Node",
    )
    proof_temp = work / "capability-scratch"
    proof_temp.mkdir()
    env = native.environment(work, os.environ["HAKUA060_WORKER_TOKEN"])
    env.update(
        TEMP=str(proof_temp), TMP=str(proof_temp), STRATA_HARNESS_DIR=str(delivery.APP / "harness")
    )
    command = [
        str(release.NODE),
        "--test",
        "--test-concurrency=1",
        "--test-reporter=tap",
        str(HERE / "real-model-runtime-fixture.mjs"),
    ]
    run = subprocess.run(
        command,
        cwd=native.PRIMARY,
        env=env,
        capture_output=True,
        timeout=175,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    (work / "capability.log").write_bytes(run.stdout + run.stderr)
    require(
        run.returncode == 0 and b"# pass 2" in run.stdout and b"# skipped 0" in run.stdout,
        "Actual-runtime synthetic admission failed",
    )
    proofs = list(proof_temp.glob("strata-real-model-synthetic-*/capability-admission.json"))
    require(len(proofs) == 1, "Missing unique runtime admission")
    proof = release.load(proofs[0])
    validate_capability(proof)
    release.save(
        work / "capability.json",
        {"command": command, "proof": proof, "log": artifact(work / "capability.log")},
    )
    return {
        "work": str(work),
        "app": str(delivery.APP),
        "sourceCommit": args.reviewed_commit,
        "ownerPID": os.getpid(),
        "installerSHA256": build["installerSHA256"],
        "shellSHA256": release.sha(delivery.APP / "strata-desktop.exe"),
        "model": {**model, "bytes": MODEL_SIZE},
        "gpuAdmission": gpu,
        "component": {
            "archive": str(Path(component["assets"]) / component["entry"]["url"]),
            "entry": component["entry"],
            "folder": release.NAME,
        },
        "capabilitySHA256": release.sha(work / "capability.json"),
    }


def validate_capability(proof: dict) -> None:
    require(
        proof.get("schema") == "strata.real-model.capability-admission.v1"
        and proof.get("syntheticOnly") is True,
        "Wrong capability proof",
    )
    require(
        proof.get("groupedCalls") == ["read", "write", "run_code", "subagent"],
        "Incomplete denial proof",
    )
    for key in (
        "arbitraryGroupedCallsSucceeded",
        "workingDirectoryChangedWhileDisabled",
        "syntheticOutsideInstructionsReachedModelWhileDisabled",
    ):
        require(proof.get(key) is False, "Capability escape")
    for key in (
        "explicitAgentsOverrideObserved",
        "defaultAgentsExcludedWhileOverridden",
        "filesOnWorkingDirectoryObserved",
        "absentOverrideDefaultAgentsObserved",
    ):
        require(proof.get(key) is True, "Incomplete instruction identity proof")
    require(
        set(proof.get("toolNames", []))
        == {
            "audit_inventory",
            "mcp__ownedAudit__read_audit_csv",
            "mcp__ownedAudit__save_audit",
            "list_mcp_resources",
            "list_mcp_resource_templates",
            "read_mcp_resource",
        },
        "Unexpected admitted tools",
    )
    require(
        set(proof.get("skillsOnToolNames", [])) == set(proof["toolNames"]) | {"skill"}
        and proof.get("forgedSkillNamesDenied") is True
        and proof.get("ownedCatalogSkillLoaded") is True,
        "Skills admission incomplete",
    )
    require(
        proof.get("dedicatedBtwChildToolfree") is True
        and proof.get("dedicatedBtwChildMaxTokens") == 1536,
        "Dedicated child admission incomplete",
    )


def phase(args: argparse.Namespace, work: Path) -> dict:
    before = release.reviewed(args.reviewed_commit)
    command = [sys.executable, "-B", *sys.argv]
    release.save(work / "worker-command.json", command)
    from ctypes import wintypes
    from perf131k.windows_job import api

    times = [ctypes.c_ulonglong() for _ in range(4)]
    require(
        api().GetProcessTimes(wintypes.HANDLE(-1), *[ctypes.byref(t) for t in times]),
        "Worker identity unavailable",
    )
    release.save(
        work / "worker-context.json",
        {
            "command": command,
            "pid": os.getpid(),
            "creationFiletime": times[0].value,
            "attemptRoot": str(work),
            "reservationSHA256": release.sha(work / "reservation.json"),
        },
    )
    inputs = {name: artifact(HERE / name) for name in ADAPTERS}
    inputs.update(
        {
            name: artifact(
                getattr(args, name + "_receipt"), getattr(args, name + "_receipt_sha256")
            )
            for name in INPUTS
        }
    )
    inputs["nodeExecutable"] = artifact(release.NODE)
    original = selected_original_hashes()
    release.save(work / "original-config-before.json", original)
    contract = admission(args, work)
    release.save(work / "input.json", contract)
    node_command = [
        str(release.NODE),
        str(HERE / "real-model.mjs"),
        "--execute",
        str(work / "input.json"),
        release.sha(work / "input.json"),
    ]
    release.save(work / "node-command.json", node_command)
    # Only the runner's sanitized error class goes to stdout/stderr. The private
    # structured checkpoints contain synthetic conversations, never access keys.
    run = subprocess.run(
        node_command,
        cwd=native.PRIMARY,
        capture_output=True,
        timeout=2400,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    (work / "producer.log").write_bytes(run.stdout + run.stderr)
    after = selected_original_hashes()
    release.save(work / "original-config-after.json", after)
    require(original == after, "Selected original configuration changed")
    require(run.returncode == 0, "Real-model runner failed; private partial observations retained")
    artifact(delivery.MODEL, MODEL_SHA)
    report = release.load(work / "observations.json")
    validate_observations(report, contract)
    require(release.reviewed(args.reviewed_commit) == before, "Source changed")
    results = {name: artifact(work / name) for name in RESULTS}
    return {
        "schema": SCHEMA,
        "mode": "installed-real-model",
        "actualModel": True,
        "syntheticResponses": False,
        "inputs": inputs,
        "results": results,
        "outcome": report["outcome"],
        "checks": report["checks"],
        "quality": report["quality"],
    }


def validate_memory_stream(frames: list) -> None:
    """Validate llama_memory.run's wrapped OpenAI chunks and separate done row."""
    require(isinstance(frames, list) and len(frames) >= 2, "Memory stream is incomplete")
    terminal = frames[-1]
    require(
        isinstance(terminal, dict)
        and set(terminal) == {"done", "memory"}
        and terminal["done"] is True
        and isinstance(terminal["memory"], str),
        "Memory terminal record malformed",
    )
    finished = False
    for frame in frames[:-1]:
        require(
            isinstance(frame, dict) and set(frame) == {"event"}, "Memory event wrapper malformed"
        )
        event = frame["event"]
        require(
            isinstance(event, dict)
            and "error" not in event
            and isinstance(event.get("choices"), list),
            "Memory completion chunk malformed",
        )
        choices = event["choices"]
        # OpenAI may emit an empty choices array with final usage after stop.
        if not choices:
            require(isinstance(event.get("usage"), dict), "Empty memory chunk without usage")
            continue
        require(not finished and len(choices) == 1, "Memory completion choice ordering invalid")
        choice = choices[0]
        require(
            isinstance(choice, dict)
            and type(choice.get("index", 0)) is int
            and choice.get("index", 0) == 0
            and isinstance(choice.get("delta"), dict)
            and "error" not in choice,
            "Memory completion choice malformed",
        )
        delta = choice["delta"]
        require(
            (delta.get("content") is None or isinstance(delta["content"], str))
            and not delta.get("tool_calls")
            and not delta.get("function_call")
            and "error" not in delta,
            "Memory completion must contain tool-free text",
        )
        reason = choice.get("finish_reason")
        require(reason in (None, "stop"), "Memory completion did not finish successfully")
        finished = reason == "stop"
    require(finished, "Memory completion is missing a successful terminal choice")


def validate_observations(report: dict, contract: dict) -> None:
    """Pure typed semantic consumer. Artifacts alone do not imply a pass."""
    require(report.get("schema") == "strata.real-model.observations.v1", "Wrong observations")
    require(report.get("sourceCommit") == contract["sourceCommit"], "Observation source differs")
    checks = report.get("checks", {})
    require(set(checks) == set(CHECKS), "Missing named checks")
    require(
        all(v in ("PASS", "FAIL", "NOT_RUN", "NOT_REUSABLE") for v in checks.values()),
        "Unknown outcome",
    )
    # Only one model-declined-learning outcome may be mechanically complete.
    require(
        all(checks[k] == "PASS" for k in CHECKS if k not in ("learning", "autoApply")),
        "Incomplete mechanical acceptance",
    )
    learn = report.get("learning", {})
    if checks["learning"] == "NOT_REUSABLE":
        require(
            checks["autoApply"] == "NOT_REUSABLE"
            and learn.get("last", {}).get("phase") == "skipped"
            and learn["last"].get("reason") == "not-reusable"
            and report.get("outcome") == "PARTIAL_NOT_REUSABLE",
            "False non-reusable outcome",
        )
    else:
        require(
            checks["learning"] == checks["autoApply"] == "PASS"
            and learn.get("last", {}).get("phase") == "saved"
            and report.get("outcome") == "MECHANICS_PASSED",
            "Learning did not save",
        )
        skill = report.get("learnedSkill", {})
        require(
            skill.get("id") == learn["last"]["skill"]["id"]
            and skill.get("body")
            and any(
                s.get("name") == skill.get("name")
                for f in report["turns"]["autoApply"]["frames"]
                if f.get("type") == "skills-used"
                for s in f.get("skills", [])
            ),
            "Saved skill/application mismatch",
        )
    require(
        report.get("instanceId") == "supervisor:" + report.get("variantId", "")
        and report.get("route") == "strata-local",
        "Foreign model route",
    )
    require(
        any(
            r.get("endpoint") == "/skill-learning"
            and r.get("state") == "done"
            and r.get("instanceId") == report["instanceId"]
            and r.get("tokensOut", 0) > 0
            for r in report.get("learningRequests", [])
        ),
        "No real same-model learning request",
    )
    settings = report.get("toolSettings", {})
    groups = {g.get("id"): g.get("enabled") for g in settings.get("groups", [])}
    expected_groups = {
        k: k == "skills"
        for k in (
            "shell",
            "files",
            "find",
            "jobs",
            "fetch",
            "search",
            "helpers",
            "goals",
            "questions",
            "todo",
            "skills",
            "present",
        )
    }
    require(
        groups == expected_groups
        and settings.get("browser", {}).get("enabled") is False
        and settings.get("webWithoutAsking") is False,
        "Audit capability configuration differs",
    )
    custom = settings.get("custom", [])
    require(
        len(custom) == 1 and custom[0].get("kind") == "server" and custom[0].get("enabled") is True,
        "Foreign custom tool",
    )
    prefix = "mcp__" + ("custom-" + custom[0]["id"].replace("_", "-"))[:32] + "__"
    require(
        set(report.get("auditTools", []))
        == {
            "audit_inventory",
            prefix + "read_audit_csv",
            prefix + "save_audit",
            "list_mcp_resources",
            "list_mcp_resource_templates",
            "read_mcp_resource",
            "skill",
        },
        "Uncontrolled allowed-schema list",
    )
    require(
        Path(custom[0].get("address", "")) == delivery.APP / "node/node.exe"
        and custom[0].get("args")
        == [str(HERE / "real-model-mcp.mjs"), str(Path(contract["work"]) / "real-home/workspace")],
        "Foreign MCP server arguments",
    )
    require(
        report.get("installed", {}).get("build") == {"host": "0.6.0", "shell": "0.6.0"}
        and report["installed"].get("shellSHA256") == contract["shellSHA256"],
        "Wrong installed build",
    )
    require(
        report.get("component", {}).get("archiveSHA256") == contract["component"]["entry"]["sha256"]
        and report["component"].get("job", {}).get("kind") == "install-engine-file"
        and report["component"]["job"].get("state") == "done",
        "No genuine production component import",
    )
    runtime = report.get("runtime", [])
    require(len(runtime) >= 3, "Missing runtime/reload observations")
    for row in runtime:
        require(
            row.get("backend") == "PrismML"
            and row.get("runtimeVersion") == "0.4.1"
            and row.get("runtimeCommit") == PRISM_COMMIT,
            "Wrong runtime",
        )
        require(
            row.get("modelSHA256") == MODEL_SHA
            and row.get("contextCapacity") == 32768
            and row.get("pid", 0) > 0
            and row.get("pid") in row.get("gpuPIDs", []),
            "Runtime/context/GPU binding differs",
        )
        require(
            row.get("ownedAncestry") is True
            and row.get("executableSHA256")
            and row.get("runtimeFiles"),
            "Missing runtime binary evidence",
        )
        require(
            row.get("modelArgumentMatched") is True
            and row.get("contextArgumentMatched") is True
            and row.get("gpuFreeMiB", 0) >= 768,
            "Unverified runtime arguments or reserve",
        )
    require(runtime[0]["pid"] != runtime[-1]["pid"], "Memory reload reused the original process")
    for name in ("chat", "audit", "research"):
        frames = report.get("turns", {}).get(name, {}).get("frames", [])
        starts = [f for f in frames if f.get("type") == "start"]
        require(
            starts
            and all(
                f.get("agent") is True and f.get("remoteModel") is False and f.get("demo") is False
                for f in starts
            ),
            "Non-agent or remote fallback",
        )
        require(
            any(
                f.get("type") == "done"
                and f.get("cancelled") is not True
                and f.get("finishReason") in ("stop", "completed")
                for f in frames
            )
            and not any(f.get("type") == "error" for f in frames),
            "Turn failed",
        )
        require(
            any(
                r.get("state") == "done"
                and r.get("instanceId") == report["instanceId"]
                and r.get("endpoint") == "/agent"
                and r.get("tokensOut", 0) > 0
                for r in report["turns"][name].get("serving", [])
            ),
            "Missing actual engine request",
        )
        require(
            any(
                f.get("type") == "context"
                and 0 < f.get("contextTokens", 0) <= 32768
                and f.get("contextMax") == 32768
                for f in frames
            ),
            "No measured agent context fit",
        )
    headers = report.get("headers", [])
    require(headers and any(h.get("tools") for h in headers), "Missing live durable schemas")
    for h in headers:
        names = [t.get("name") for t in h.get("tools", [])]
        allowed = (
            report.get("auditTools", [])
            if h.get("phase") == "audit"
            else ["web_search", "web_fetch"]
            if h.get("phase") == "research"
            else []
        )
        require(
            set(names) <= set(allowed)
            and h.get("provider") == report.get("route")
            and h.get("model") == "local",
            "Live request schema/route differs",
        )
    audit = report.get("audit", {})
    expected = {"total_mismatches": 1, "total_duplicates": 0, "total_missing": 1, "total_failed": 1}
    require(
        {k: audit.get(k) for k in expected} == expected,
        "Audit counts differ from controlled inputs",
    )
    require(
        [r.get("id") for r in audit.get("version_mismatches", [])] == ["alpha"]
        and [r.get("id") for r in audit.get("missing_ids", [])] == ["gamma"]
        and [r.get("id") for r in audit.get("failed_checks", [])] == ["beta"]
        and audit.get("duplicates") == [],
        "Audit classifications differ",
    )
    require(
        audit["version_mismatches"]
        == [{"id": "alpha", "inventory_version": "1", "expected_version": "2"}]
        and audit["missing_ids"] == [{"id": "gamma", "missingFrom": ["inventory"]}]
        and audit["failed_checks"]
        == [{"id": "beta", "required_check": "checksum", "status": "failed"}],
        "Audit values differ",
    )
    require(
        {k: v.get("content") for k, v in report.get("auditInputs", {}).items()} == CSV,
        "Uncontrolled audit fixture content",
    )
    done_tools = [
        f["tool"]
        for f in report["turns"]["audit"]["frames"]
        if f.get("type") == "agent-tool" and f.get("tool", {}).get("state") == "done"
    ]
    require(
        any(t.get("name") == "audit_inventory" for t in done_tools)
        and sum(t.get("name", "").endswith("__read_audit_csv") for t in done_tools) >= 3
        and any(t.get("name", "").endswith("__save_audit") for t in done_tools),
        "Audit did not execute real tools",
    )
    require(
        {
            t.get("args", {}).get("table")
            for t in done_tools
            if t.get("name") == prefix + "read_audit_csv"
        }
        == {"inventory", "expected", "requirements"},
        "Audit did not read all named inputs",
    )
    audit_call = next(t for t in done_tools if t.get("name") == "audit_inventory")
    require(
        audit_call.get("args")
        == {
            k + "_csv": report.get("auditInputs", {}).get(k, {}).get("content")
            for k in ("inventory", "expected", "requirements")
        },
        "Audit tool input binding differs",
    )
    save_call = next(t for t in done_tools if t.get("name") == prefix + "save_audit")
    require(
        json.loads(save_call.get("args", {}).get("audit_json", "null")) == audit,
        "Persisted audit is not the model tool output",
    )
    btw = report.get("turns", {}).get("btw", {}).get("frames", [])
    child = next((f for f in btw if f.get("type") == "btw-agent"), {})
    require(
        child.get("childSessionId")
        and child.get("childSessionId") != child.get("parentSessionId")
        and any(f.get("type") == "done" and not f.get("cancelled") for f in btw),
        "No separate completed btw child",
    )
    require(
        any(
            h.get("sessionId") == child["childSessionId"]
            and not h.get("tools")
            and h.get("maxTokens") == 1536
            for h in headers
        ),
        "Btw tool/output bounds missing",
    )
    research = report.get("research", {})
    require(
        len(set(research.get("queries", []))) >= 3
        and len(set(research.get("urls", []))) >= 4
        and research.get("sources")
        and research.get("workspaceBefore") == research.get("workspaceAfter"),
        "Incomplete or writing research",
    )
    web_done = [
        f["tool"]
        for f in report["turns"]["research"]["frames"]
        if f.get("type") == "agent-tool" and f.get("tool", {}).get("state") == "done"
    ]
    require(
        research["queries"]
        == [t.get("args", {}).get("query") for t in web_done if t.get("name") == "web_search"]
        and research["urls"]
        == [t.get("args", {}).get("url") for t in web_done if t.get("name") == "web_fetch"],
        "Research counts not derived from successful tools",
    )
    require(
        any(
            f.get("type") == "research"
            and f.get("sources") == research["sources"]
            and f.get("deep") is True
            for f in report["turns"]["research"]["frames"]
        ),
        "Research terminal frame missing",
    )
    require(
        report.get("proxyAbsent") is True and report.get("deniedUnexpectedTools") == 0,
        "Unsafe capability use",
    )
    memory = report.get("memory", {})
    require(
        memory.get("store", [])
        and memory["store"][-1].get("done") is True
        and memory["store"][-1].get("memory") == "stored",
        "Memory storage not confirmed",
    )
    require(
        memory.get("recall", [])
        and memory["recall"][-1].get("done") is True
        and memory.get("storeTurnId") != memory.get("recallTurnId")
        and memory.get("recallHistoryIncludesAnswer") is False,
        "Memory recall protocol invalid",
    )
    for phase_name, read, store_summary, turn_key in (
        ("memoryStore", False, True, "storeTurnId"),
        ("memoryRecall", True, False, "recallTurnId"),
    ):
        request = report.get("turns", {}).get(phase_name, {}).get("request", {})
        require(
            request.get("strata_memory")
            == {"read": read, "store_summary": store_summary, "turn_id": memory[turn_key]}
            and request.get("max_tokens") == 128
            and request.get("stream") is True,
            "Memory request bounds differ",
        )
        require(
            len(request.get("messages", [])) == 1 and request["messages"][0].get("role") == "user",
            "Memory request includes answer history",
        )
        validate_memory_stream(memory["recall" if read else "store"])
    require(
        "CERULEAN-684" not in json.dumps(report["turns"]["memoryRecall"]["request"]),
        "Recall prompt supplied the answer",
    )
    require(
        memory.get("profileRoot")
        == str(Path(contract["work"]) / "real-home/data/hakua-memory/local"),
        "Memory root escaped",
    )
    cleanup = report.get("cleanup", {})
    require(
        cleanup.get("turns") == []
        and cleanup.get("learningRunning") is False
        and cleanup.get("activeRequests") == 0
        and cleanup.get("runtimeExited") is True
        and cleanup.get("portClosed") is True
        and cleanup.get("shellExited") is True
        and cleanup.get("shellCloseRequested") is True,
        "Incomplete owned cleanup",
    )
    require(
        report.get("quality")
        == {
            "factualAccuracy": "NOT_REVIEWED",
            "memoryAnswerRecall": "NOT_REVIEWED",
            "generalModelQuality": "UNQUALIFIED",
            "fullWindow": "UNQUALIFIED",
            "gpuConcurrency": "UNQUALIFIED",
            "releaseQualified": False,
        },
        "Quality overclaim",
    )


def validate_proof(record: dict, work: Path, job: dict) -> None:
    require(
        set(record.get("inputs", {})) == set(ADAPTERS) | set(INPUTS) | {"nodeExecutable"}
        and set(record.get("results", {})) == set(RESULTS),
        "Missing/foreign named evidence",
    )
    for name in ADAPTERS:
        acceptance.evidence(record, "inputs", name, HERE / name)
    acceptance.evidence(record, "inputs", "nodeExecutable", release.NODE)
    for name in record["results"]:
        acceptance.evidence(record, "results", name, work / name)
    command = release.load(work / "worker-command.json")
    require(
        command[:3] == [sys.executable, "-B", str(HERE / "real-model.py")], "Foreign worker command"
    )
    args = parsed(command[3:])
    require(
        args.execute
        and args.worker
        and args.reviewed_commit == record["sourceCommit"]
        and work == release.OUT / ("real-model-" + args.attempt),
        "Worker source/attempt differs",
    )
    for name in INPUTS:
        path = getattr(args, name + "_receipt")
        acceptance.evidence(record, "inputs", name, path)
        require(
            record["inputs"][name]["sha256"] == getattr(args, name + "_receipt_sha256"),
            "Worker input pin differs",
        )
        delivery.receipt(
            path,
            getattr(args, name + "_receipt_sha256"),
            name.replace("_", "-"),
            args.reviewed_commit,
        )
    reservation_sha = release.sha(work / "reservation.json")
    require(
        release.load(work / "reservation.json").get("identity")
        == {k: record[k] for k in ("sourceCommit", "sourceRoot", "tree")},
        "Reservation source differs",
    )
    require(
        job.get("realModelExecution")
        == {
            "command": command,
            "cwd": str(native.PRIMARY),
            "attemptRoot": str(work),
            "reservationSHA256": reservation_sha,
            "adapterSHA256": record["inputs"]["real-model.py"]["sha256"],
        },
        "Unbound owned job",
    )
    require(
        release.load(work / "worker-context.json")
        == {
            "command": command,
            "pid": job.get("owned_root_pid"),
            "creationFiletime": job.get("root_creation_filetime"),
            "attemptRoot": str(work),
            "reservationSHA256": reservation_sha,
        },
        "Worker process differs",
    )
    contract = release.load(work / "input.json")
    require(
        contract["work"] == str(work)
        and contract["ownerPID"] == job["owned_root_pid"]
        and contract["sourceCommit"] == args.reviewed_commit,
        "Foreign runner input",
    )
    require(
        contract.get("model")
        == {"path": str(delivery.MODEL), "sha256": MODEL_SHA, "bytes": MODEL_SIZE}
        and Path(contract.get("app", "")) == delivery.APP,
        "Runner model/app identity differs",
    )
    require(
        contract.get("gpuAdmission", {}).get("freeMiB", 0) >= 8192
        and contract["gpuAdmission"].get("computePIDs") == []
        and contract["gpuAdmission"].get("reserveMiB") == 768,
        "Missing idle GPU admission",
    )
    build = release.load(args.app_receipt)
    require(
        contract.get("installerSHA256") == build.get("installerSHA256"), "Installer input differs"
    )
    component = delivery.feed.receipt_assets(
        args.component_receipt, args.component_receipt_sha256, "component", args.reviewed_commit
    )
    require(
        contract.get("component")
        == {
            "archive": str(Path(component["assets"]) / component["entry"]["url"]),
            "entry": component["entry"],
            "folder": release.NAME,
        },
        "Signed component input differs",
    )
    require(
        release.load(work / "node-command.json")
        == [
            str(release.NODE),
            str(HERE / "real-model.mjs"),
            "--execute",
            str(work / "input.json"),
            release.sha(work / "input.json"),
        ],
        "Runner invocation differs",
    )
    proof = release.load(work / "capability.json")
    validate_capability(proof["proof"])
    artifact(Path(proof["log"]["path"]), proof["log"]["sha256"])
    require(
        release.sha(work / "capability.json") == contract["capabilitySHA256"],
        "Admission input changed",
    )
    require(
        release.load(work / "original-config-before.json")
        == release.load(work / "original-config-after.json"),
        "Original config differs",
    )
    report = release.load(work / "observations.json")
    require(
        report.get("inputSHA256") == release.sha(work / "input.json"),
        "Runner observations input differs",
    )
    validate_observations(report, contract)
    home = work / "real-home"

    def stage_artifact(row: dict, expected: Path) -> None:
        require(
            set(row) == {"path", "sha256"} and Path(row["path"]) == expected,
            "Foreign stage artifact",
        )
        artifact(expected, row["sha256"])

    stage_artifact(report["auditFile"], home / "workspace/audit.json")
    require(
        release.load(home / "workspace/audit.json") == report["audit"],
        "Persisted audit bytes differ",
    )
    for key, text in CSV.items():
        row = report["auditInputs"][key]
        stage_artifact({k: row[k] for k in ("path", "sha256")}, home / "workspace" / (key + ".csv"))
        require(
            (home / "workspace" / (key + ".csv")).read_text(encoding="utf-8") == text,
            "Fixture input bytes differ",
        )
    for row in report["runtime"]:
        root = home / "components/engine" / release.NAME
        stage_artifact(row["componentManifest"], root / "engine-component.json")
        require(
            row["componentManifest"]["sha256"] == component["entry"]["manifest_sha256"],
            "Imported component manifest differs",
        )
        runtime_root = root / "local-backends/prism"
        stage_artifact(row["manifest"], runtime_root / "runtime.json")
        manifest = release.load(runtime_root / "runtime.json")
        require(
            manifest.get("version") == "0.4.1"
            and manifest.get("source", {}).get("commit") == PRISM_COMMIT
            and row["runtimeFiles"] == {f["path"]: f["sha256"] for f in manifest["files"]},
            "Runtime inventory differs",
        )
        for relative, digest in row["runtimeFiles"].items():
            require(
                not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                "Unsafe runtime member",
            )
            artifact(runtime_root / relative, digest)
        require(Path(row["executable"]).is_relative_to(runtime_root), "Runtime executable escaped")
        artifact(Path(row["executable"]), row["executableSHA256"])
    for header in report["headers"]:
        require(Path(header["file"]).is_relative_to(home / "data/agent"), "Foreign session log")
        artifact(Path(header["file"]), header["fileSHA256"])
    if report["checks"]["learning"] == "PASS":
        slug = report["learnedSkill"]["id"].removeprefix("my-")
        require(re.fullmatch(r"[a-z0-9-]+", slug), "Unsafe learned skill id")
        stage_artifact(report["learnedSkillFile"], home / "data/skills" / slug / "SKILL.md")
    require(
        record["checks"] == report["checks"]
        and record["outcome"] == report["outcome"]
        and record["quality"] == report["quality"],
        "Outcome changed",
    )
    originals = release.load(work / "original-config-before.json")
    expected_originals = {
        str(root / name)
        for root in delivery.USER_ROOTS[:3]
        for name in ("settings.json", "data/state.json", "desktop/shell-prefs.json")
    }
    require(
        set(originals) == expected_originals, "Original configuration observation scope differs"
    )
    for key, row in originals.items():
        require(
            row is None
            or (
                set(row) == {"path", "sha256"}
                and row["path"] == key
                and re.fullmatch(r"[a-f0-9]{64}", row["sha256"])
            ),
            "Malformed original hash observation",
        )
    require(
        report.get("modelMetadata") == {"architecture": "qwen35", "tensorCount": 427},
        "Unexpected GGUF metadata",
    )
    for row in report["runtime"]:
        chain = row.get("ancestry", [])
        require(
            chain
            and chain[0].get("pid") == row["pid"]
            and chain[-1].get("pid") == report["shellPID"]
            and all(a.get("parent") == b.get("pid") for a, b in zip(chain, chain[1:])),
            "Runtime ancestry does not reach the owned shell",
        )


def validate_real_model(path: Path, expected_sha: str, commit: str) -> dict:
    record = delivery.pinned(path, expected_sha)
    require(
        path.name == "real-model.json"
        and record.get("schema") == SCHEMA
        and record.get("mode") == "installed-real-model"
        and record.get("actualModel") is True
        and record.get("syntheticResponses") is False,
        "Wrong real-model result",
    )
    raw = delivery.receipt(
        path.parent / "receipt.json", record["receiptSHA256"], "real-model", commit
    )
    require(
        {
            k: v
            for k, v in record.items()
            if k not in ("receiptSHA256", "ownedJobSHA256", "workerLogSHA256")
        }
        == raw,
        "Result differs from drained receipt",
    )
    artifact(path.parent / "owned-job.json", record["ownedJobSHA256"])
    artifact(path.parent / "worker.log", record["workerLogSHA256"])
    job = release.load(path.parent / "owned-job.json")
    acceptance.successful_job(job)
    validate_proof(record, path.parent, job)
    artifact(path, expected_sha)
    return record


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parsed(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "execute": False,
                    "phase": args.phase,
                    "requiredPinnedReceipts": INPUTS,
                    "explicitAttempt": True,
                    "actualModel": "NOT_RUN",
                    "releaseQualified": False,
                }
            )
        )
        return 0
    if args.worker:
        return native.run_controller(args, HERE / "real-model.py", phase, argv)
    sys.path.insert(0, str(native.PRIMARY / "tools"))
    from perf131k import windows_job

    original = windows_job.run_owned
    work = release.OUT / (args.phase + "-" + args.attempt)
    expected = [sys.executable, "-B", str(HERE / "real-model.py"), *argv, "--worker"]

    def owned(command: list[str], root: Path, *positional: Any, **keywords: Any) -> dict:
        require(command == expected and root == native.PRIMARY, "Unexpected launch")
        reservation_sha = release.sha(work / "reservation.json")
        result = original(command, root, *positional, **keywords)
        result["realModelExecution"] = {
            "command": command,
            "cwd": str(root),
            "attemptRoot": str(work),
            "reservationSHA256": reservation_sha,
            "adapterSHA256": release.sha(HERE / "real-model.py"),
        }
        return result

    windows_job.run_owned = owned
    try:
        result = native.run_controller(args, HERE / "real-model.py", phase, argv)
    finally:
        windows_job.run_owned = original
    raw, job = release.load(work / "receipt.json"), release.load(work / "owned-job.json")
    acceptance.successful_job(job)
    validate_proof(raw, work, job)
    release.save(
        work / "real-model.json",
        {
            **raw,
            "receiptSHA256": release.sha(work / "receipt.json"),
            "ownedJobSHA256": release.sha(work / "owned-job.json"),
            "workerLogSHA256": release.sha(work / "worker.log"),
        },
    )
    validate_real_model(
        work / "real-model.json", release.sha(work / "real-model.json"), args.reviewed_commit
    )
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        logging.error(
            "Real-model attempt failed (%s); private partial observations retained",
            type(error).__name__,
        )
        raise SystemExit(1) from None
