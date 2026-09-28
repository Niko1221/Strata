#!/usr/bin/env python3
"""End-to-end native OpenAI tool loop that mirrors a Hermes agent conversation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

from diagnose_openai import DEFAULT_BASE, ROOT, TOOLS, api_key, base_payload, call, message_of, tool_calls


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--api-key")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "diagnostics" / "hermes-flow.json")
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key(args.api_key)}", "Content-Type": "application/json"})
    messages = [
        {
            "role": "system",
            "content": "You are Hermes, a tool-using agent. Use the provided tools instead of guessing. Maintain conversational context.",
        },
        {
            "role": "user",
            "content": "I need an update on the UPS monitoring job you set up. Use your tools to inspect it.",
        },
    ]
    transcript: list[dict] = []

    def complete(label: str, tool_choice="auto") -> dict:
        payload = base_payload(messages, 512)
        payload.update({"tools": TOOLS, "tool_choice": tool_choice})
        body, elapsed = call(session, args.base_url, payload)
        msg = message_of(body)
        transcript.append({"label": label, "request": payload, "response": body, "elapsed_s": round(elapsed, 3)})
        messages.append(msg)
        print(f"{label}: {json.dumps(msg, ensure_ascii=False)}", flush=True)
        return msg

    first = complete("discover-job")
    require(tool_calls(first), "Hermes did not call a tool to discover the UPS job")
    require(tool_calls(first)[0]["function"]["name"] == "list_scheduled_jobs", "Hermes called the wrong discovery tool")
    messages.append({
        "role": "tool",
        "tool_call_id": tool_calls(first)[0]["id"],
        "content": json.dumps({"jobs": [{"id": "job-4821", "name": "ups-nightly", "status": "failed"}]}),
    })

    second = complete("inspect-job")
    require(tool_calls(second), "Hermes ignored the failed job and did not inspect it")
    function = tool_calls(second)[0]["function"]
    require(function["name"] == "get_job_status", "Hermes called the wrong inspection tool")
    require("job-4821" in function["arguments"], "Hermes forgot the job ID returned by the first tool")
    messages.append({
        "role": "tool",
        "tool_call_id": tool_calls(second)[0]["id"],
        "content": json.dumps({
            "id": "job-4821",
            "name": "ups-nightly",
            "status": "failed",
            "last_error": "UPS host 192.0.2.45 timed out after 30 seconds",
        }),
    })

    third = complete("explain-result")
    answer = (third.get("content") or "").lower()
    require("ups-nightly" in answer or "job-4821" in answer, "Hermes lost the job identity after the tool result")
    require("timed out" in answer or "timeout" in answer, "Hermes did not report the actual tool error")

    messages.append({"role": "user", "content": "What tools are available to you in this conversation? Name them."})
    fourth = complete("remember-tools")
    answer = (fourth.get("content") or "").lower()
    names = {call_.get("function", {}).get("name") for call_ in tool_calls(fourth)}
    require(
        ("list_scheduled_jobs" in answer and "get_job_status" in answer)
        or names.intersection({"list_scheduled_jobs", "get_job_status"}),
        "Hermes did not recognize the native tools still available in the same conversation",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"base_url": args.base_url, "transcript": transcript}, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: complete Hermes native-tool workflow; report: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
