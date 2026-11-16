"""Preparation boundary tests, not real-model acceptance or release evidence."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import copy
from datetime import datetime
import json
import io
import pytest

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("real_model_boundary_release", HERE / "release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
producer_spec = importlib.util.spec_from_file_location(
    "real_model_producer", HERE / "real-model.py"
)
producer = importlib.util.module_from_spec(producer_spec)
producer_spec.loader.exec_module(producer)


def test_operation_ram_option_is_explicit_and_does_not_change_plan_mode(monkeypatch) -> None:
    monkeypatch.setenv("HAKUA060_ALLOW_LOW_RAM", "1")
    assert producer.parsed([]).allow_low_ram is False
    args = producer.parsed(["--allow-low-ram"])
    assert args.allow_low_ram is True and args.execute is False


def test_production_gate_boundary() -> None:
    """Observe permission decisions with fake services; execute no proposed tools."""
    completed = subprocess.run(
        [
            str(release.NODE),
            "--test",
            "--test-reporter=tap",
            "--test-concurrency=1",
            str(HERE / "real-model-boundary-fixture.mjs"),
        ],
        cwd=release.ROOT,
        capture_output=True,
        timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    # Never print raw captured errors, which could contain inherited host details.
    passed = completed.returncode == 0 and all(
        counter in completed.stdout for counter in (b"# pass 3", b"# fail 0", b"# skipped 0")
    )
    assert passed, "Synthetic permission boundary fixture failed; raw output suppressed"


def test_actual_runtime_capability_admission() -> None:
    """Local scripted inference only; actual pinned registry and MCP transport."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper()
        in {
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "PATH",
            "PATHEXT",
            "TEMP",
            "TMP",
            "PROGRAMFILES",
            "PROGRAMFILES(X86)",
            "PROGRAMDATA",
            "SYSTEMDRIVE",
        }
    }
    env["STRATA_HARNESS_DIR"] = str(release.RES / "harness")
    completed = subprocess.run(
        [
            str(release.NODE),
            "--test",
            "--test-reporter=tap",
            "--test-concurrency=1",
            str(HERE / "real-model-runtime-fixture.mjs"),
        ],
        cwd=release.ROOT,
        env=env,
        capture_output=True,
        timeout=170,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    # The environment deliberately excludes credentials; do not expose raw errors.
    (Path(env["TEMP"]) / "real-model-runtime-synthetic.tap").write_bytes(
        completed.stdout + completed.stderr
    )
    passed = completed.returncode == 0 and all(
        counter in completed.stdout for counter in (b"# pass 2", b"# fail 0", b"# skipped 0")
    )
    assert passed, "Synthetic runtime admission failed; captured diagnostics retained privately"


def test_node_pure_producer_contracts() -> None:
    run = subprocess.run(
        [
            str(release.NODE),
            "--test",
            "--test-concurrency=1",
            "--test-reporter=tap",
            str(HERE / "real-model-contract-fixture.mjs"),
        ],
        cwd=release.ROOT,
        capture_output=True,
        timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert run.returncode == 0 and b"# pass 4" in run.stdout and b"# skipped 0" in run.stdout, (
        "Pure producer fixture failed; raw diagnostics suppressed"
    )


def memory_chunks() -> list[dict]:
    """Synthetic OpenAI wire chunks consumed by the actual LocalLlama parser."""
    base = {"id": "synthetic-memory", "object": "chat.completion.chunk", "model": "local"}
    return [
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Synthetic answer"},
                    "finish_reason": None,
                }
            ],
        },
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {
            **base,
            "choices": [],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        },
    ]


def memory_frames(status: str) -> list[dict]:
    return [{"event": event} for event in memory_chunks()] + [{"done": True, "memory": status}]


@pytest.mark.parametrize("store", [True, False])
def test_actual_memory_adapter_stream_contract(monkeypatch, store: bool) -> None:
    """Actual SSE parsing/run wrapper; fake transport and store, no model or DB."""
    from serve import llama_memory

    class SyntheticMemory:
        def principal(self, key):
            return "synthetic-principal"

        def begin(self, key, consent, messages, cancel):
            return consent["store_summary"], messages

        def complete(self, turn, answer, summarize):
            assert answer == "Synthetic answer"
            return "stored" if turn else None

    chunks = memory_chunks()
    wire = b"".join(b"data: " + json.dumps(row).encode() + b"\n\n" for row in chunks)
    monkeypatch.setattr(
        llama_memory.memory_harness, "from_config", lambda *a, **k: SyntheticMemory()
    )
    monkeypatch.setattr(
        llama_memory.LocalLlama, "response", lambda *a: io.BytesIO(wire + b"data: [DONE]\n\n")
    )
    monkeypatch.setattr(
        llama_memory.http.client,
        "HTTPConnection",
        lambda *a, **k: pytest.fail("Synthetic protocol test attempted network access"),
    )
    config = {
        "port": 54321,
        "server_key": "synthetic-local-transport",
        "model": "local",
        "args": [
            "--cache-ram",
            "0",
            "--cache-reuse",
            "0",
            "--no-cache-prompt",
            "--no-cache-idle-slots",
            "--no-slots",
            "--log-disable",
        ],
    }
    request = {
        "messages": [{"role": "user", "content": "Synthetic fact"}],
        "stream": True,
        "max_tokens": 128,
        "strata_memory": {"read": not store, "store_summary": store, "turn_id": "synthetic-turn"},
    }
    frames = []
    llama_memory.run(config, request, "synthetic-profile", frames.append)
    assert frames == memory_frames("stored" if store else "not-stored")
    producer.validate_memory_stream(frames)
    # Missing upstream [DONE] raises in production and leaves no successful
    # terminal record, even if a stop choice was received just before EOF.
    monkeypatch.setattr(llama_memory.LocalLlama, "response", lambda *a: io.BytesIO(wire))
    truncated = []
    with pytest.raises(RuntimeError, match="ended before completion"):
        llama_memory.run(config, request, "synthetic-profile", truncated.append)
    with pytest.raises(RuntimeError):
        producer.validate_memory_stream(truncated)


@pytest.mark.parametrize(
    "damage",
    [
        "invented_string",
        "missing_stop",
        "length",
        "content_filter",
        "cancelled",
        "wrong_choices",
        "multiple_choices",
        "wrong_index",
        "boolean_index",
        "wrong_delta",
        "nontext_content",
        "tool_call",
        "event_error",
        "wrapper_error",
        "empty_chunk",
        "choice_after_stop",
        "missing_done",
        "early_done",
    ],
)
def test_memory_stream_rejects_failed_malformed_and_truncated_chunks(damage: str) -> None:
    frames = memory_frames("stored")
    first = frames[0]["event"]
    choice = first["choices"][0]
    if damage == "invented_string":
        frames[0]["event"] = "completion"
    elif damage == "missing_stop":
        frames[1]["event"]["choices"][0]["finish_reason"] = None
    elif damage in ("length", "content_filter", "cancelled"):
        frames[1]["event"]["choices"][0]["finish_reason"] = damage
    elif damage == "wrong_choices":
        first["choices"] = {}
    elif damage == "multiple_choices":
        first["choices"].append(copy.deepcopy(choice))
    elif damage == "wrong_index":
        choice["index"] = 1
    elif damage == "boolean_index":
        choice["index"] = False
    elif damage == "wrong_delta":
        choice["delta"] = "text"
    elif damage == "nontext_content":
        choice["delta"]["content"] = 1
    elif damage == "tool_call":
        choice["delta"]["tool_calls"] = [{"id": "synthetic-tool"}]
    elif damage == "event_error":
        first["error"] = {"message": "synthetic failure"}
    elif damage == "wrapper_error":
        frames[0] = {"error": "synthetic failure"}
    elif damage == "empty_chunk":
        first["choices"] = []
    elif damage == "choice_after_stop":
        frames.insert(2, copy.deepcopy(frames[0]))
    elif damage == "missing_done":
        frames.pop()
    elif damage == "early_done":
        frames.insert(1, copy.deepcopy(frames[-1]))
    with pytest.raises(RuntimeError):
        producer.validate_memory_stream(frames)


def synthetic_observations(tmp_path: Path) -> tuple[dict, dict]:
    """Explicit synthetic data for the pure consumer, never a release receipt."""
    p = producer
    contract = {
        "work": str(tmp_path),
        "sourceCommit": "a" * 40,
        "shellSHA256": "b" * 64,
        "component": {"entry": {"sha256": "c" * 64}},
    }
    prefix = "mcp__custom-owned__"
    tools = [
        "audit_inventory",
        prefix + "read_audit_csv",
        prefix + "save_audit",
        "list_mcp_resources",
        "list_mcp_resource_templates",
        "read_mcp_resource",
        "skill",
    ]
    audit = {
        "criteria": "Synthetic unit fixture",
        "version_mismatches": [{"id": "alpha", "inventory_version": "1", "expected_version": "2"}],
        "duplicates": [],
        "missing_ids": [{"id": "gamma", "missingFrom": ["inventory"]}],
        "failed_checks": [{"id": "beta", "required_check": "checksum", "status": "failed"}],
        "total_mismatches": 1,
        "total_duplicates": 0,
        "total_missing": 1,
        "total_failed": 1,
    }
    frames = [
        {"type": "start", "agent": True, "remoteModel": False, "demo": False},
        {"type": "context", "contextTokens": 8000, "contextMax": 32768},
        {"type": "done", "finishReason": "stop"},
    ]
    serving = [
        {"endpoint": "/agent", "state": "done", "instanceId": "supervisor:test", "tokensOut": 10}
    ]
    turns = {
        k: {"frames": copy.deepcopy(frames), "serving": serving}
        for k in ("chat", "audit", "research", "autoApply")
    }

    def tool(name: str, args: dict) -> dict:
        return {"type": "agent-tool", "tool": {"name": name, "args": args, "state": "done"}}

    turns["audit"]["frames"] += [tool(prefix + "read_audit_csv", {"table": k}) for k in p.CSV]
    turns["audit"]["frames"] += [
        tool("audit_inventory", {k + "_csv": v for k, v in p.CSV.items()}),
        tool(prefix + "save_audit", {"audit_json": json.dumps(audit)}),
    ]
    sources = [{"url": "https://example.org/source"}]
    queries = ["synthetic query " + str(i) for i in range(3)]
    urls = ["https://example.org/" + str(i) for i in range(4)]
    turns["research"]["frames"] += (
        [tool("web_search", {"query": q}) for q in queries]
        + [tool("web_fetch", {"url": u}) for u in urls]
        + [{"type": "research", "deep": True, "sources": sources}]
    )
    turns["autoApply"]["frames"] += [
        {"type": "skills-used", "skills": [{"name": "Inventory reconciliation"}]}
    ]
    turns["btw"] = {
        "frames": [
            {"type": "btw-agent", "childSessionId": "child", "parentSessionId": "parent"},
            {"type": "done"},
        ]
    }
    memory = {
        "profileRoot": str(tmp_path / "real-home/data/hakua-memory/local"),
        "storeTurnId": "store-id",
        "recallTurnId": "recall-id",
        "recallHistoryIncludesAnswer": False,
        "store": memory_frames("stored"),
        "recall": memory_frames("not-stored"),
    }
    for phase, read, store, turn in (
        ("memoryStore", False, True, "store-id"),
        ("memoryRecall", True, False, "recall-id"),
    ):
        turns[phase] = {
            "request": {
                "messages": [
                    {"role": "user", "content": "synthetic fact" if store else "recall the fact"}
                ],
                "max_tokens": 128,
                "stream": True,
                "strata_memory": {"read": read, "store_summary": store, "turn_id": turn},
            }
        }
    runtime = [
        {
            "backend": "PrismML",
            "runtimeVersion": "0.4.1",
            "runtimeCommit": p.PRISM_COMMIT,
            "pid": i,
            "gpuPIDs": [i],
            "modelSHA256": p.MODEL_SHA,
            "contextCapacity": 32768,
            "ownedAncestry": True,
            "executableSHA256": "d" * 64,
            "runtimeFiles": {"llama-server.exe": "d" * 64},
            "modelArgumentMatched": True,
            "contextArgumentMatched": True,
            "gpuFreeMiB": 800,
        }
        for i in (1, 2, 3)
    ]
    owner = {
        "pid": 100,
        "parentPID": 50,
        "created": "2026-11-16T13:05:01.7009610Z",
        "executable": "C:\\Python\\python.exe",
        "sha256": "e" * 64,
    }
    delta = datetime(2026, 11, 16, 13, 5, 1, 700961) - datetime(1601, 1, 1)
    owner["creationFiletime"] = (
        (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds
    ) * 10

    def gpu_fixture(pid: int | None = None) -> dict:
        processes = [owner]
        node = ""
        if pid is not None:
            processes.append(
                {
                    "pid": pid,
                    "parentPID": 100,
                    "created": "2026-11-16T13:05:02.7009610Z",
                    "executable": "C:\\Prism\\llama-server.exe",
                    "sha256": "d" * 64,
                    "creationFiletime": owner["creationFiletime"] + 10000000,
                }
            )
            node = f"<process_info><pid>{pid}</pid><type>C</type></process_info>"
        free = 800 if pid is not None else 14811
        raw = {
            "gpuXML": "<nvidia_smi_log><attached_gpus>1</attached_gpus><gpu>"
            f"<uuid>GPU-fixture</uuid><fb_memory_usage><free>{free} MiB</free>"
            "</fb_memory_usage><processes>" + node + "</processes></gpu></nvidia_smi_log>",
            "gpuRows": [[0, "GPU-fixture", free]],
            "freeMiB": free,
            "computeQueryPIDs": [pid] if pid else [],
            "processes": processes,
        }
        return p.gpu_policy.classify(raw, None, owner, 768 if pid else 8192)

    contract.update(
        ownerPID=100, gpuPolicy={"baseline": None, "owner": owner}, gpuAdmission=gpu_fixture()
    )
    for row in runtime:
        row.update(executable="C:\\Prism\\llama-server.exe", gpuObservation=gpu_fixture(row["pid"]))
    settings = {
        "groups": [
            {"id": k, "enabled": k == "skills"}
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
        ],
        "custom": [
            {
                "id": "owned",
                "kind": "server",
                "enabled": True,
                "address": str(p.delivery.APP / "node/node.exe"),
                "args": [str(HERE / "real-model-mcp.mjs"), str(tmp_path / "real-home/workspace")],
            }
        ],
        "browser": {"enabled": False},
        "webWithoutAsking": False,
    }
    report = {
        "schema": "strata.real-model.observations.v1",
        "sourceCommit": contract["sourceCommit"],
        "checks": dict.fromkeys(p.CHECKS, "PASS"),
        "outcome": "MECHANICS_PASSED",
        "learning": {
            "last": {
                "phase": "saved",
                "skill": {"id": "my-learned-inventory", "name": "Inventory reconciliation"},
            }
        },
        "learnedSkill": {
            "id": "my-learned-inventory",
            "name": "Inventory reconciliation",
            "body": "Synthetic procedure",
        },
        "learningRequests": [
            {
                "endpoint": "/skill-learning",
                "state": "done",
                "instanceId": "supervisor:test",
                "tokensOut": 10,
            }
        ],
        "variantId": "test",
        "instanceId": "supervisor:test",
        "route": "strata-local",
        "toolSettings": settings,
        "auditTools": tools,
        "installed": {
            "build": {"host": "0.6.0", "shell": "0.6.0"},
            "shellSHA256": contract["shellSHA256"],
        },
        "component": {
            "archiveSHA256": "c" * 64,
            "job": {"kind": "install-engine-file", "state": "done"},
        },
        "runtime": runtime,
        "turns": turns,
        "headers": [
            {
                "phase": "audit",
                "provider": "strata-local",
                "model": "local",
                "tools": [{"name": x} for x in tools],
            },
            {
                "phase": "btw",
                "provider": "strata-local",
                "model": "local",
                "sessionId": "child",
                "maxTokens": 1536,
                "tools": [],
            },
        ],
        "audit": audit,
        "auditInputs": {k: {"content": v} for k, v in p.CSV.items()},
        "research": {
            "queries": queries,
            "urls": urls,
            "sources": sources,
            "workspaceBefore": {"audit.json": "synthetic"},
            "workspaceAfter": {"audit.json": "synthetic"},
        },
        "proxyAbsent": True,
        "deniedUnexpectedTools": 0,
        "memory": memory,
        "cleanup": {
            "turns": [],
            "learningRunning": False,
            "activeRequests": 0,
            "runtimeExited": True,
            "portClosed": True,
            "shellExited": True,
            "shellCloseRequested": True,
        },
        "quality": {
            "factualAccuracy": "NOT_REVIEWED",
            "memoryAnswerRecall": "NOT_REVIEWED",
            "generalModelQuality": "UNQUALIFIED",
            "fullWindow": "UNQUALIFIED",
            "gpuConcurrency": "UNQUALIFIED",
            "releaseQualified": False,
        },
    }
    report.update(gpuBeforeLoad=gpu_fixture(), gpuBeforeReload=gpu_fixture())
    return report, contract


def test_pure_consumer_mechanics_and_honest_non_reusable(tmp_path: Path) -> None:
    report, contract = synthetic_observations(tmp_path)
    producer.validate_observations(report, contract)
    report["checks"].update(learning="NOT_REUSABLE", autoApply="NOT_REUSABLE")
    report["outcome"] = "PARTIAL_NOT_REUSABLE"
    report["learning"] = {"last": {"phase": "skipped", "reason": "not-reusable"}}
    del report["learnedSkill"]
    del report["turns"]["autoApply"]
    producer.validate_observations(report, contract)


@pytest.mark.parametrize(
    "path,value",
    [
        (("checks", "runtime"), "NOT_RUN"),
        (("checks", "learning"), "NOT_REUSABLE"),
        (("outcome",), "PASS"),
        (("learningRequests",), []),
        (("toolSettings", "groups", 0, "enabled"), True),
        (("auditTools",), ["run_code"]),
        (("toolSettings", "custom", 0, "args"), ["unowned"]),
        (("runtime", 0, "runtimeVersion"), "0.4.0"),
        (("runtime", 0, "contextCapacity"), 8192),
        (("runtime", 0, "gpuPIDs"), []),
        (("runtime", 0, "ownedAncestry"), False),
        (("runtime", 0, "modelArgumentMatched"), False),
        (("component", "job", "state"), "running"),
        (("component", "archiveSHA256"), "0" * 64),
        (("turns", "chat", "frames", 0, "agent"), False),
        (("turns", "chat", "serving"), []),
        (("turns", "chat", "frames", 1, "contextTokens"), 40000),
        (("headers", 0, "tools"), [{"name": "read"}]),
        (("headers", 0, "provider"), "remote"),
        (("audit", "total_missing"), 0),
        (("audit", "missing_ids"), []),
        (("auditInputs", "inventory", "content"), "different"),
        (("headers", 1, "maxTokens"), 4096),
        (("research", "queries"), ["repeated"] * 3),
        (("research", "workspaceAfter"), {}),
        (("proxyAbsent",), False),
        (("memory", "store", 3, "memory"), "failed"),
        (("memory", "storeTurnId"), "recall-id"),
        (("turns", "memoryRecall", "request", "messages", 0, "content"), "CERULEAN-684"),
        (("cleanup", "runtimeExited"), False),
        (("quality", "factualAccuracy"), "PASS"),
    ],
)
def test_pure_consumer_rejects_unbound_or_false_success(
    tmp_path: Path, path: tuple, value: object
) -> None:
    report, contract = synthetic_observations(tmp_path)
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises((RuntimeError, ValueError, KeyError, TypeError)):
        producer.validate_observations(report, contract)


def test_default_plan_is_inert_and_execute_requires_explicit_attempt(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        producer.native, "run_controller", lambda *a: pytest.fail("Plan launched a controller")
    )
    assert producer.main([]) == 0
    assert json.loads(capsys.readouterr().out)["actualModel"] == "NOT_RUN"
    with pytest.raises(RuntimeError, match="explicit attempt"):
        producer.parsed(["--execute"])
    with pytest.raises(RuntimeError, match="explicit attempt"):
        producer.parsed(["--execute", "--attempt", "r1", "--attempt", "r2"])


def test_consumer_rejects_missing_execution_inputs_before_reading_job(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="named evidence"):
        producer.validate_proof({"inputs": {}, "results": {}}, tmp_path, {})


@pytest.mark.parametrize(
    "damage",
    ["command_source", "command_attempt", "input_pin", "reservation", "job_command", "worker_pid"],
)
def test_worker_job_source_attempt_and_pin_binding(
    tmp_path: Path, monkeypatch, damage: str
) -> None:
    """The earlier artifact/receipt helpers are stubbed; execution binding is real."""
    p = producer
    monkeypatch.setattr(p.release, "OUT", tmp_path)
    work = tmp_path / "real-model-r7"
    commit = "a" * 40
    argv = ["--execute", "--worker", "--attempt", "r7", "--reviewed-commit", commit]
    paths = {name: tmp_path / (name + ".json") for name in p.INPUTS}
    for name, path in paths.items():
        option = name.replace("_", "-")
        argv.extend(
            ["--" + option + "-receipt", str(path), "--" + option + "-receipt-sha256", "b" * 64]
        )
    command = [sys.executable, "-B", str(HERE / "real-model.py"), *argv]
    identity = {"sourceCommit": commit, "sourceRoot": str(p.native.PRIMARY), "tree": "c" * 40}
    inputs = {name: {"path": str(HERE / name), "sha256": "d" * 64} for name in p.ADAPTERS}
    inputs.update({name: {"path": str(path), "sha256": "b" * 64} for name, path in paths.items()})
    inputs["nodeExecutable"] = {"path": str(release.NODE), "sha256": "e" * 64}
    inputs["pythonExecutable"] = {"path": sys.executable, "sha256": "e" * 64}
    record = {
        **identity,
        "inputs": inputs,
        "results": {name: {"path": str(work / name), "sha256": "f" * 64} for name in p.RESULTS},
    }
    execution = {
        "command": copy.deepcopy(command),
        "cwd": str(p.native.PRIMARY),
        "attemptRoot": str(work),
        "reservationSHA256": "1" * 64,
        "adapterSHA256": "d" * 64,
    }
    job = {"realModelExecution": execution, "owned_root_pid": 345, "root_creation_filetime": 123456}
    context = {
        "command": copy.deepcopy(command),
        "pid": 345,
        "creationFiletime": 123456,
        "attemptRoot": str(work),
        "reservationSHA256": "1" * 64,
    }
    docs = {
        work / "worker-command.json": command,
        work / "worker-context.json": context,
        work / "reservation.json": {"identity": identity},
    }
    if damage == "command_source":
        command[command.index(commit)] = "9" * 40
    if damage == "command_attempt":
        command[command.index("r7")] = "r8"
    if damage == "input_pin":
        inputs["installed_qa"]["sha256"] = "8" * 64
    if damage == "reservation":
        docs[work / "reservation.json"] = {"identity": {**identity, "tree": "7" * 40}}
    if damage == "job_command":
        execution["command"] = ["different"]
    if damage == "worker_pid":
        context["pid"] = 346
    monkeypatch.setattr(
        p.acceptance, "evidence", lambda record, section, name, expected_path=None: expected_path
    )
    monkeypatch.setattr(p.delivery, "receipt", lambda *a: {})
    monkeypatch.setattr(p.release, "load", lambda path: docs[path])
    monkeypatch.setattr(p.release, "sha", lambda path: "1" * 64)
    with pytest.raises(RuntimeError):
        p.validate_proof(record, work, job)
