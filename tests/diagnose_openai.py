#!/usr/bin/env python3
"""Live OpenAI-compatible conversation and native function-tool diagnostics.

Runs deterministic, independent scenarios against Strata and saves complete requests and
responses under the ignored build directory. The API key is read from STRATA_API_KEY or
.strata-service.env; it is never written to the report.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "http://127.0.0.1:8088/v1"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_scheduled_jobs",
            "description": "List scheduled background jobs, including their IDs, names, status, and schedule.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_job_status",
            "description": "Get the current status and recent result for one scheduled job.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string", "description": "Scheduled job identifier"}},
                "required": ["job_id"],
                "additionalProperties": False,
            },
        },
    },
]


def api_key(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.environ.get("STRATA_API_KEY"):
        return os.environ["STRATA_API_KEY"]
    env_file = ROOT / ".strata-service.env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("STRATA_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("No API key: pass --api-key, set STRATA_API_KEY, or create .strata-service.env")


def tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    return message.get("tool_calls") or []


def call(session: requests.Session, base: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    response = session.post(f"{base.rstrip('/')}/chat/completions", json=payload, timeout=600)
    elapsed = time.monotonic() - started
    try:
        body = response.json()
    except ValueError:
        body = {"raw": response.text}
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {json.dumps(body, ensure_ascii=False)}")
    return body, elapsed


def message_of(body: dict[str, Any]) -> dict[str, Any]:
    return body["choices"][0]["message"]


def base_payload(messages: list[dict[str, Any]], max_tokens: int = 256) -> dict[str, Any]:
    return {
        "model": "qwen3.8-flash-next-q2_0",
        "messages": messages,
        "reasoning_effort": "none",
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }


def run(base: str, key: str) -> tuple[list[dict[str, Any]], int]:
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    report: list[dict[str, Any]] = []

    def scenario(name: str, payload: dict[str, Any], check) -> dict[str, Any]:
        try:
            response, elapsed = call(session, base, payload)
            msg = message_of(response)
            ok, detail = check(msg)
            row = {
                "name": name,
                "passed": bool(ok),
                "detail": detail,
                "elapsed_s": round(elapsed, 3),
                "request": payload,
                "response": response,
            }
        except Exception as exc:
            row = {"name": name, "passed": False, "detail": f"{type(exc).__name__}: {exc}", "request": payload}
        report.append(row)
        status = "PASS" if row["passed"] else "FAIL"
        print(f"[{status}] {name}: {row['detail']}", flush=True)
        return row

    scenario(
        "plain-tools-question",
        base_payload([{"role": "user", "content": "What tools do you have available? Answer directly."}]),
        lambda m: (bool(m.get("content")) and "tool" in m["content"].lower(), repr(m.get("content"))),
    )

    scenario(
        "multi-turn-recall",
        base_payload([
            {"role": "system", "content": "Follow the conversation and answer the latest user message."},
            {"role": "user", "content": "My UPS monitoring job is named ups-nightly and its identifier is job-4821."},
            {"role": "assistant", "content": "Understood. I will use that information in this conversation."},
            {"role": "user", "content": "What is the exact UPS job identifier? Answer with only the identifier."},
        ]),
        lambda m: ("job-4821" in (m.get("content") or "").lower(), repr(m.get("content"))),
    )

    required = base_payload([
        {"role": "system", "content": "You are an agent. Use the supplied functions when needed. Never invent tool results."},
        {"role": "user", "content": "Check the status of the UPS job you set up. First list the scheduled jobs."},
    ], 384)
    required.update({"tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "list_scheduled_jobs"}}})
    first = scenario(
        "required-native-tool-call",
        required,
        lambda m: (
            bool(tool_calls(m)) and tool_calls(m)[0].get("function", {}).get("name") == "list_scheduled_jobs",
            json.dumps(m, ensure_ascii=False),
        ),
    )

    if first["passed"]:
        first_message = message_of(first["response"])
        call_id = tool_calls(first_message)[0].get("id") or "call_list_jobs"
        continuation = base_payload([
            {"role": "system", "content": "You are an agent. Use supplied functions when needed. Never invent tool results."},
            {"role": "user", "content": "Check the status of the UPS job you set up. First list the scheduled jobs."},
            first_message,
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps({"jobs": [{"id": "job-4821", "name": "ups-nightly", "status": "failed"}]}),
            },
            {"role": "user", "content": "Now inspect that UPS job and tell me why it failed."},
        ], 384)
        continuation.update({"tools": TOOLS, "tool_choice": "auto"})
        scenario(
            "tool-result-continuation",
            continuation,
            lambda m: (
                bool(tool_calls(m)) and tool_calls(m)[0].get("function", {}).get("name") == "get_job_status"
                and "job-4821" in str(tool_calls(m)[0].get("function", {}).get("arguments", "")),
                json.dumps(m, ensure_ascii=False),
            ),
        )
    else:
        report.append({"name": "tool-result-continuation", "passed": False, "detail": "SKIP: prerequisite tool call failed"})
        print("[FAIL] tool-result-continuation: SKIP: prerequisite tool call failed", flush=True)

    scenario(
        "long-history-latest-turn",
        base_payload([
            {"role": "system", "content": "Answer the latest user request using the conversation history."},
            {"role": "user", "content": "We are discussing a UPS monitoring job."},
            {"role": "assistant", "content": "Understood."},
            *sum(([
                {"role": "user", "content": f"Background note {i}: routine diagnostics were reviewed."},
                {"role": "assistant", "content": f"Noted background item {i}."},
            ] for i in range(24)), []),
            {"role": "user", "content": "The verification phrase is copper-lantern-731."},
            {"role": "assistant", "content": "I have the verification phrase."},
            {"role": "user", "content": "Return only the verification phrase from my previous message."},
        ], 128),
        lambda m: ("copper-lantern-731" in (m.get("content") or "").lower(), repr(m.get("content"))),
    )

    return report, sum(not row.get("passed", False) for row in report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE))
    parser.add_argument("--api-key")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "diagnostics" / "openai.json")
    args = parser.parse_args()
    report, failures = run(args.base_url, api_key(args.api_key))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"base_url": args.base_url, "scenarios": report}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{len(report) - failures}/{len(report)} passed; report: {args.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
