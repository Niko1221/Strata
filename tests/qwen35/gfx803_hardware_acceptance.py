#!/usr/bin/env python3
"""Guard-friendly Qwen3.6 API smoke: tool round-trip, stream cancel, recovery, restart."""
import argparse
import json
import pathlib
import subprocess
import time
import urllib.request


def request(base, model, messages, **options):
    body = {"model": model, "messages": messages, "max_tokens": 128,
            "temperature": 0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    body.update(options)
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    response = urllib.request.urlopen(req, timeout=60)
    if not body.get("stream"):
        with response:
            return json.load(response)
    chunks, content = [], ""
    try:
        for line in response:
            text = line.decode().strip()
            if not text.startswith("data: ") or text == "data: [DONE]":
                continue
            event = json.loads(text[6:]); chunks.append(event)
            for choice in event.get("choices", []):
                content += choice.get("delta", {}).get("content") or ""
            if len(content) >= 8:
                break
    finally:
        response.close()
    if len(content) < 8:
        raise RuntimeError("stream ended before meaningful generated content")
    return {"cancelled_after_generated_content": content, "stream_events": len(chunks)}


def start(args):
    process = subprocess.Popen([args.python, "-m", "serve.server", "--engine", "strata",
                                "--config", args.config, "--host", args.host,
                                "--port", str(args.port)], cwd=args.cwd)
    base = f"http://{args.host}:{args.port}"
    for _ in range(args.startup_seconds * 2):
        if process.poll() is not None:
            raise RuntimeError("server exited during startup")
        try:
            with urllib.request.urlopen(base + "/health", timeout=1) as response:
                if response.status == 200:
                    return process, base
        except Exception:
            time.sleep(0.5)
    process.terminate(); process.wait(timeout=10)
    raise RuntimeError("readiness timeout")


def stop(process):
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill(); process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", required=True, type=pathlib.Path)
    parser.add_argument("--python", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--startup-seconds", type=int, default=180)
    args = parser.parse_args()
    results = []
    process, base = start(args)
    try:
        tools = [{"type": "function", "function": {"name": "lookup_capital",
                 "description": "Look up the capital of a country.",
                 "parameters": {"type": "object", "properties": {"country": {"type": "string"}},
                                "required": ["country"], "additionalProperties": False}}}]
        messages = [{"role": "user", "content": "Use lookup_capital to get the capital of France."}]
        first = request(base, args.model, messages, tools=tools, tool_choice="required")
        assistant = first["choices"][0]["message"]
        calls = assistant.get("tool_calls", [])
        if len(calls) != 1 or calls[0]["function"]["name"] != "lookup_capital":
            raise RuntimeError("expected exactly one lookup_capital tool call")
        arguments = json.loads(calls[0]["function"]["arguments"])
        if arguments.get("country", "").lower() != "france":
            raise RuntimeError("tool arguments did not name France")
        messages += [assistant, {"role": "tool", "tool_call_id": calls[0]["id"],
                                "content": json.dumps({"capital": "Paris"})}]
        second = request(base, args.model, messages, tools=tools)
        if "Paris" not in (second["choices"][0]["message"].get("content") or ""):
            raise RuntimeError("assistant did not use the tool result")
        results.append("tool round-trip PASS")
        request(base, args.model, [{"role": "user", "content":
                "List the integers from one to one hundred, one per line."}], stream=True)
        results.append("stream cancellation PASS")
        recovery = request(base, args.model, [{"role": "user", "content": "Say exactly READY."}], max_tokens=16)
        if "READY" not in (recovery["choices"][0]["message"].get("content") or ""):
            raise RuntimeError("request after stream cancellation did not recover")
        results.append("post-cancel recovery PASS")
    finally:
        stop(process)
    process, base = start(args)
    try:
        restart = request(base, args.model, [{"role": "user", "content":
                "What is the capital of France? Answer briefly."}], max_tokens=16)
        if "Paris" not in (restart["choices"][0]["message"].get("content") or ""):
            raise RuntimeError("fresh process failed restart smoke")
        results.append("process restart PASS")
    finally:
        stop(process)
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
