"""Probe native Responses against a running real-model server, without executing tools.

    python -m tools.responses_live_smoke --url http://127.0.0.1:8080/v1
If authentication is enabled, set STRATA_API_KEY. Model calls have small explicit token budgets.
"""
import argparse
import json
import os
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    headers = {"Content-Type": "application/json"}
    if os.environ.get("STRATA_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["STRATA_API_KEY"]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(body):
        req = urllib.request.Request(args.url.rstrip("/") + "/responses", data=json.dumps(body).encode(), headers=headers)
        try:
            with opener.open(req, timeout=args.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise SystemExit("The running server has no /v1/responses route. Restart it with the updated code first.")
            raise SystemExit(f"HTTP {error.code}: {error.read().decode()}")

    common = {"store": False, "reasoning": {"effort": "none"}}
    text = request({**common, "input": "Reply with exactly: strata-responses-ok", "max_output_tokens": 64})
    answer = "".join(part.get("text", "") for item in text["output"] if item["type"] == "message" for part in item["content"])
    if "strata-responses-ok" not in answer or text["status"] != "completed":
        raise SystemExit(f"Text following failed: status={text['status']}, text={answer!r}")
    print("real-model text: passed")
    body = {**common, "input": "Call echo exactly once with text set to strata-tool-ok. Do not answer in prose.",
            "max_output_tokens": 256, "tool_choice": "required", "parallel_tool_calls": False,
            "tools": [{"type": "function", "name": "echo", "parameters": {"type": "object",
                       "properties": {"text": {"type": "string"}}, "required": ["text"]}}]}
    response = request(body)
    calls = [item for item in response["output"] if item["type"] == "function_call"]
    if len(calls) != 1 or calls[0]["status"] != "completed" or json.loads(calls[0]["arguments"]) != {"text": "strata-tool-ok"}:
        raise SystemExit(f"Tool following failed: status={response['status']}, calls={calls}")
    print("real-model function generation: passed (tool was not executed)")
    body["input"] = [{"role": "user", "content": body["input"]}] + response["output"] + [
        {"type": "function_call_output", "call_id": calls[0]["call_id"], "output": "strata-tool-ok"},
        {"role": "user", "content": "Reply with exactly the echo tool result. Do not call another tool."}]
    body.update(tool_choice="none", max_output_tokens=64)
    response = request(body)
    answer = "".join(part.get("text", "") for item in response["output"] if item["type"] == "message" for part in item["content"])
    if response["status"] != "completed" or "strata-tool-ok" not in answer:
        raise SystemExit(f"Tool replay failed: status={response['status']}, text={answer!r}")
    print("real-model tool-result replay: passed")


if __name__ == "__main__":
    main()
