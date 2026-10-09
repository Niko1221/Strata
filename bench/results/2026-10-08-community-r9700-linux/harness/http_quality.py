#!/usr/bin/env python3
"""Local HTTP task, long-context, multi-turn and cancellation regression fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

from bench_engine import save, load_tokenizer, ChatTemplate
from single_gpu import ROOT


def fixture_data(tokenizer):
    cases = [
        {"name": "json-extraction", "messages": [{"role": "user", "content":
         'Extract name and count from "Name: Mira; count: 17". Reply only with JSON keys name and count; count is an integer.'}],
         "expected": {"name": "Mira", "count": 17}},
        {"name": "arithmetic", "messages": [{"role": "user", "content":
         'Compute (37 * 19) - 28. Reply only with JSON {"result": integer}.'}], "expected": {"result": 675}},
        {"name": "python-code", "messages": [{"role": "user", "content":
         'Write only a Python function dedupe_keep_order(values). Input is a list of integers; return a new list with duplicates removed, preserving first occurrence order. Do not modify the input. Use no imports, decorators, helper functions or test code.'}]},
        {"name": "tool-call", "messages": [{"role": "user", "content":
         'Call get_weather for Hangzhou with units celsius. Do not guess the weather.'}],
         "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city",
         "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                        "required": ["city", "units"]}}}]},
        {"name": "multi-turn", "messages": [
         {"role": "user", "content": "Remember this project code: NIMBUS-7391. Reply OK."},
         {"role": "assistant", "content": "OK"},
         {"role": "user", "content": 'What was the project code? Reply only with JSON {"code": "..."}.'}],
         "expected": {"code": "NIMBUS-7391"}},
    ]
    tok = load_tokenizer(tokenizer)
    tpl = ChatTemplate(tokenizer / "chat_template.jinja")
    for length in (32768, 131072):
        filler = "\n".join(f"record {i:06d}: color=blue, value={i % 997};" for i in range(length // 8 + 100))
        body = tok.encode(filler)[:length - 256]
        snippets = {int(len(body) * .1): "\nThe access key ALPHA is 481927.\n",
                    int(len(body) * .5): "\nThe access key BETA is 602314.\n",
                    int(len(body) * .9): "\nThe access key GAMMA is 975086.\n"}
        parts, last = [], 0
        for offset, text in sorted(snippets.items()):
            parts.extend((tok.decode(body[last:offset]), text)); last = offset
        parts.append(tok.decode(body[last:]))
        prompt = f"Document {length}. Read and remember the access keys.\n" + "".join(parts)
        prompt += '\nReturn only JSON with integer values for the access keys ALPHA, BETA and GAMMA.'
        messages = [{"role": "user", "content": prompt}]
        ids = tok.encode(tpl.render(messages, enable_thinking=False), parse_special=True)
        cases.append({"name": f"recall-{length}", "messages": messages,
                      "expected": {"ALPHA": 481927, "BETA": 602314, "GAMMA": 975086},
                      "rendered_tokens": len(ids), "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
    return {"schema": 1, "scope": "small deterministic local regression suite, not a general quality benchmark", "cases": cases}


def unfence(text):
    match = re.fullmatch(r"\s*```(?:json|python)?\s*\n(.*?)\n```\s*", text, re.S)
    return match.group(1) if match else text.strip()


class Client:
    def __init__(self, url):
        if urllib.parse.urlparse(url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("private local test server required")
        self.url = url.rstrip("/")

    def request(self, path, body=None):
        request = urllib.request.Request(self.url + path, data=json.dumps(body).encode() if body is not None else None,
                                         headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(request, timeout=1800)

    def json(self, path, body=None):
        with self.request(path, body) as response:
            return json.load(response)

    def idle(self):
        for _ in range(100):
            data = self.json("/metrics")
            if data["live"]["state"] == "idle" and data["live"]["queued"] == 0:
                return data
            time.sleep(.2)
        raise RuntimeError("test server did not become idle")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--make-fixtures", action="store_true")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8097")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.make_fixtures:
        if not args.tokenizer or args.fixtures.exists():
            parser.error("--tokenizer and a new fixture filename are required")
        save(args.fixtures, fixture_data(args.tokenizer))
        return
    if not args.out:
        parser.error("--out required")
    args.out.mkdir(parents=True, exist_ok=False)
    client = Client(args.url)
    initial = client.idle()
    model = initial["engine"]["model"]
    base = {"model": model, "max_tokens": 512, "temperature": 0, "seed": 42,
            "top_k": 1, "top_p": 1, "min_p": 0, "chat_template_kwargs": {"enable_thinking": False}}
    results = {"initial_metrics": initial, "fixtures_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
               "url": args.url, "records": []}
    save(args.out / "results.json", results)
    for case in json.loads(args.fixtures.read_text())["cases"]:
        record = {"name": case["name"], "passed": False}
        try:
            client.idle()
            messages = [dict(m) for m in case["messages"]]
            if case["name"] == "multi-turn":
                first = client.json("/v1/chat/completions", {**base, "messages": case["messages"][:1]})
                record["first_response"] = first
                messages[1]["content"] = first["choices"][0]["message"]["content"]
                record["actual_messages"] = messages
            begin = time.perf_counter()
            response = client.json("/v1/chat/completions", {**base, "messages": messages,
                                                          **({"tools": case["tools"]} if "tools" in case else {})})
            record.update(response=response, wall_s=time.perf_counter() - begin)
            message = response["choices"][0]["message"]
            if case["name"] == "tool-call":
                calls = message.get("tool_calls", [])
                record["passed"] = len(calls) == 1 and calls[0]["function"]["name"] == "get_weather" and json.loads(calls[0]["function"]["arguments"]) == {"city": "Hangzhou", "units": "celsius"}
            elif case["name"] == "python-code":
                code = unfence(message.get("content") or "")
                checked = subprocess.run([sys.executable, "-I", str(ROOT / "tools/hip/r9700/check_generated_code.py")],
                                         input=json.dumps({"code": code}), text=True, capture_output=True, timeout=5)
                record.update(code_check_stdout=checked.stdout, code_check_stderr=checked.stderr)
                record["passed"] = checked.returncode == 0
            else:
                record["passed"] = json.loads(unfence(message.get("content") or "")) == case["expected"]
            if "rendered_tokens" in case:
                record["expected_prompt_tokens"] = case["rendered_tokens"]
                record["passed"] &= response["usage"]["prompt_tokens"] == case["rendered_tokens"]
            record["metrics_after"] = client.idle()
            if case["name"] == "multi-turn":
                # The followup must actually traverse the retained prefix, not just answer correctly from a cold prompt.
                record["passed"] &= record["metrics_after"]["requests"][0].get("reused", 0) > 0
        except Exception as exc:
            record["error"] = str(exc)
            record["passed"] = False
        results["records"].append(record)
        save(args.out / "results.json", results)
        print(f"{case['name']}: {'PASS' if record['passed'] else 'FAIL'}", flush=True)
    # A real SSE content event, disconnect, and immediate retry on the same private server.
    record = {"name": "sse-cancel-retry", "passed": False}
    try:
        client.idle()
        begin = time.perf_counter()
        request = {**base, "max_tokens": 4096, "stream": True,
                   "messages": [{"role": "user", "content": "Count from 1 to 1000, one number per line. Do not summarize."}]}
        with client.request("/v1/chat/completions", request) as response:
            for line in response:
                if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                    item = json.loads(line[6:])
                    choices = item.get("choices", [])
                    if choices and choices[0].get("delta", {}).get("content"):
                        record["first_content"] = item
                        record["http_sse_ttft_s"] = time.perf_counter() - begin
                        break
            else:
                raise RuntimeError("no SSE content")
        record["after_cancel"] = client.idle()
        last = record["after_cancel"]["requests"][0]
        if last["finish"] not in {"cancel", "disconnect"} or not 0 < last["output_tokens"] < 4096:
            raise RuntimeError("cancellation was not observed")
        retry = client.json("/v1/chat/completions", {**base, "max_tokens": 16,
                            "messages": [{"role": "user", "content": "What is 2 + 3? Reply only with the digit."}]})
        record["retry"] = retry
        record["passed"] = retry["choices"][0]["message"]["content"].strip() == "5"
    except Exception as exc:
        record["error"] = str(exc)
        record["passed"] = False
    results["records"].append(record)
    results["final_metrics"] = client.idle()
    results["passed"] = all(r["passed"] for r in results["records"])
    results["complete"] = True
    save(args.out / "results.json", results)
    print(f"sse-cancel-retry: {'PASS' if record['passed'] else 'FAIL'}", flush=True)
    if not results["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
