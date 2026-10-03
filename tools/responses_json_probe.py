"""Real-model Responses JSON qualification against an already running server.

Uses one existing server, never starts an engine or executes a model's tools.
The lookup result is a labeled mock owned by this test client. Model output is
not scripted. Pass the API key in the environment; it is never written to evidence.
"""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


TITLE = {"type": "json_schema", "name": "codex_output_schema", "strict": True,
         "schema": {"type": "object", "properties": {"title": {"type": "string", "minLength": 1, "maxLength": 36}},
                    "required": ["title"], "additionalProperties": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Server URL ending in /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="STRATA_API_KEY")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    report = {"scripted_model_output": False, "mocked_external_tool_result": True, "cases": [], "result": "running"}

    def save():
        (args.out / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def post(name, body, expected_status=200, terminal="completed"):
        number = len(report["cases"])
        req = {"model": args.model, "store": False, "reasoning": {"effort": "none"},
               "max_output_tokens": 512, "temperature": 0, **body}
        # None here means exercise the omitted field (as Codex's title task does).
        req = {k: v for k, v in req.items() if v is not None}
        (args.out / f"{number:02d}.request.json").write_text(json.dumps(req, indent=2, ensure_ascii=False), encoding="utf-8")
        request = urllib.request.Request(args.base_url.rstrip("/") + "/responses", data=json.dumps(req).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ[args.api_key_env]})
        start = time.monotonic()
        try:
            response = opener.open(request, timeout=300)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw, code, content_type = response.read().decode("utf-8"), response.status, response.headers.get("Content-Type", "")
        (args.out / f"{number:02d}.response.txt").write_text(raw, encoding="utf-8")
        row = {"name": name, "http_status": code, "content_type": content_type, "seconds": time.monotonic() - start}
        report["cases"].append(row)
        save()
        assert code == expected_status, (name, code, raw)
        if code != 200:
            assert "text/event-stream" not in content_type
            row["error"] = json.loads(raw)["error"]
            save()
            return None
        if req.get("stream"):
            events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
            assert [e["sequence_number"] for e in events] == list(range(len(events)))
            ends = [e for e in events if e["type"] in ("response.completed", "response.incomplete", "response.failed")]
            assert len(ends) == 1 and ends[0] is events[-1]
            result = ends[0]["response"]
            for index, item in enumerate(result["output"]):
                if item["type"] == "message":
                    deltas = [e for e in events if e["type"] == "response.output_text.delta" and e["output_index"] == index]
                    assert all(e["item_id"] == item["id"] and e["content_index"] == 0 for e in deltas)
                    assert "".join(e["delta"] for e in deltas) == item["content"][0]["text"]
                if item["type"] == "function_call":
                    deltas = [e for e in events if e["type"] == "response.function_call_arguments.delta" and e["output_index"] == index]
                    assert all(e["item_id"] == item["id"] for e in deltas)
                    assert "".join(e["delta"] for e in deltas) == item["arguments"]
            row["stream_reassembled"] = True
        else:
            result = json.loads(raw)
        row["status"] = result["status"]
        row["output_types"] = [x["type"] for x in result["output"]]
        save()
        assert result["status"] == terminal, (name, result)
        print(name, result["status"], round(row["seconds"], 3), flush=True)
        return result

    def answer(result):
        text = "".join(p["text"] for item in result["output"] if item["type"] == "message" for p in item["content"])
        return json.loads(text)

    try:
        first = post("Codex title schema with reasoning and omitted output limit", {
            "input": "Create a short title for fixing an addition function.", "text": {"format": TITLE},
            "reasoning": {"effort": "medium", "summary": "auto"}, "max_output_tokens": None})
        title = answer(first)
        assert set(title) == {"title"} and isinstance(title["title"], str) and 1 <= len(title["title"]) <= 36
        obj = post("arbitrary nested JSON object", {"input": 'Return {"data":[true,null,1.5,"caf\u00e9"]}.',
                                                    "text": {"format": {"type": "json_object"}}})
        assert answer(obj) == {"data": [True, None, 1.5, "caf\u00e9"]}
        fmt = {"type": "json_schema", "name": "nested", "strict": True, "schema": {
            "type": "object", "$defs": {"label": {"type": "string", "enum": ["caf\u00e9"]}},
            "properties": {"rows": {"type": "array", "items": {"$ref": "#/$defs/label"}, "minItems": 1, "maxItems": 1},
                           "ok": {"type": "boolean", "const": True}},
            "required": ["rows", "ok"], "additionalProperties": False}}
        result = post("nested schema, local ref, enum, Unicode, typed SSE", {
            "input": "Ignore JSON and write a paragraph about the sun.", "text": {"format": fmt}, "stream": True})
        assert answer(result) == {"rows": ["caf\u00e9"], "ok": True}
        post("output exhaustion stays incomplete", {"input": "Give a title.", "text": {"format": TITLE},
                                                    "max_output_tokens": 1, "stream": True}, terminal="incomplete")
        post("remote schema reference rejected before SSE", {"input": "Hello", "stream": True,
             "text": {"format": {**TITLE, "schema": {"$ref": "https://example.invalid/schema"}}}}, expected_status=400)
        tool = {"type": "function", "name": "lookup", "description": "Read the hidden integer. Call before answering.",
                "parameters": {"type": "object", "properties": {"key": {"type": "string", "enum": ["hidden"]}},
                               "required": ["key"], "additionalProperties": False}, "strict": True}
        fmt = {"type": "json_schema", "name": "lookup_result", "strict": True, "schema": {
            "type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"], "additionalProperties": False}}
        history = [{"role": "user", "content": "Call lookup with key hidden to read the hidden integer. Do not guess. Then give the value as JSON."}]
        result = post("real model function call before JSON answer", {"input": history, "tools": [tool],
             "text": {"format": fmt}, "reasoning": {"effort": "medium"}, "max_output_tokens": 3072, "stream": True})
        calls = [item for item in result["output"] if item["type"] == "function_call"]
        assert len(calls) == 1 and calls[0]["name"] == "lookup", result
        assert json.loads(calls[0]['arguments']) == {'key':'hidden'}
        history += result["output"]
        history.append({"type": "function_call_output", "call_id": calls[0]["call_id"],
                        "output": "MOCK EXTERNAL TOOL RESULT (not JSON): the hidden integer is 27. Other arbitrary text: purple."})
        result = post("reasoning, arbitrary mocked tool result, JSON final answer", {
            "input": history, "tools": [tool], "text": {"format": fmt}, "reasoning": {"effort": "medium"},
            "max_output_tokens": 3072, "stream": True})
        assert answer(result) == {"value": 27}
        report["result"] = "pass"
    except Exception as exc:
        report["result"] = "fail"
        report["failure"] = str(exc)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
