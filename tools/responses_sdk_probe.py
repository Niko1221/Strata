"""Official SDK probes against real Strata HTTP with explicitly synthetic MockEngine output.

The client executes a bounded read/write loop in its own disposable directory.
This is not a real-model or Codex task and must not unlock the R4/GBNF gate.
"""
from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import openai
from openai import OpenAI
from openai.types.responses import Response
from openai.types.responses import ResponseStreamEvent
from pydantic import TypeAdapter
from cryptography.fernet import Fernet
from serve.response_replay import ReplayCodec
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Server, Service, make_handler

MODEL = "qwen3.8-flash-next"
TOOLS = [{"type": "namespace", "name": "files", "description": "Client-owned file helpers.", "tools": [
         {"type": "function", "name": "read_file", "strict": False,
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
         {"type": "function", "name": "write_file", "strict": False,
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}]}]
SCRIPTS = ["Hello, SDK.", "Hello, SDK.",
           "Inspect the file first.</think><tool_call><function=files.read_file><parameter=path>example.txt</parameter></function></tool_call>",
           "The model planned to read the file.",
           "Replace the line after reading it.</think><tool_call><function=files.write_file><parameter=path>example.txt</parameter><parameter=content>after\n\n"
           "</parameter></function></tool_call>", "The model planned the requested edit.",
           "The client confirmed the write.</think>Edited example.txt.", "The model checked the client's write result."]


@contextlib.contextmanager
def fixture_server(scripts):
    tok = ByteTokenizer()
    engine = MockEngine(tok, scripts)
    svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    svc.experimental_responses = True
    svc.responses_replay = ReplayCodec(Fernet.generate_key())
    svc.api_key = "fixture-only"
    svc.api_monitor = True
    httpd = Server(("127.0.0.1", 0), make_handler(svc))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield svc, f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if openai.__version__ != "3.23.0":
        ap.error("use the pinned R0 Python environment (openai==3.23.0)")
    a.out.mkdir(parents=True, exist_ok=True)
    stream_events = []
    with tempfile.TemporaryDirectory(prefix="responses-sdk-") as temporary, fixture_server(SCRIPTS) as (svc, base):
        target = Path(temporary) / "example.txt"
        target.write_text("before\n", encoding="utf-8")
        with OpenAI(base_url=base, api_key="fixture-only", max_retries=0) as client:
            text = client.responses.create(model=MODEL, input="Hello", store=False)
            assert text.output_text == "Hello, SDK." and text.status == "completed"
            Response.model_validate(text.model_dump())
            with client.responses.stream(model=MODEL, input="Hello", store=False) as stream:
                for event in stream:
                    stream_events.append(event.model_dump(mode="json", exclude_unset=True))
                streamed = stream.get_final_response()
            assert streamed.output_text == text.output_text
            history = [{"role": "user", "content": "Inspect example.txt and replace before with after."}]
            client_effects = []
            for turn in range(3):
                kwargs = dict(model=MODEL, input=history, store=False, tools=TOOLS,
                              reasoning={"summary": "auto"}, include=["reasoning.encrypted_content"], max_output_tokens=1024)
                if turn == 1:
                    with client.responses.stream(**kwargs) as stream:
                        for event in stream:
                            stream_events.append(event.model_dump(mode="json", exclude_unset=True))
                        response = stream.get_final_response()
                else:
                    response = client.responses.create(**kwargs)
                Response.model_validate(response.model_dump())
                assert response.status == "completed"
                history.extend(item.model_dump(exclude_none=True) for item in response.output)
                for item in history:
                    if item.get("type") == "reasoning":
                        item.pop("content", None)  # exercise the actual encrypted-only continuation
                calls = [item for item in response.output if item.type == "function_call"]
                if turn < 2:
                    assert len(calls) == 1
                    # Strata must not perform the write itself when emitting the call.
                    assert target.read_text(encoding="utf-8") == "before\n"
                else:
                    assert not calls and response.output_text == "Edited example.txt."
                for call in calls:
                    assert call.id != call.call_id
                    assert call.namespace == "files"
                    arguments = json.loads(call.arguments)
                    assert arguments["path"] == "example.txt"  # bounded client authorization
                    if call.name == "read_file":
                        result = target.read_text(encoding="utf-8")
                    elif call.name == "write_file":
                        assert arguments["content"] == "after\n"
                        target.write_text(arguments["content"], encoding="utf-8")
                        result = "written"
                    else:
                        raise AssertionError("unknown client tool")
                    client_effects.append({"name": call.name, "path": "example.txt"})
                    history.append({"type": "function_call_output", "call_id": call.call_id, "output": result})
            assert target.read_text(encoding="utf-8") == "after\n"
        adapter = TypeAdapter(ResponseStreamEvent)
        for event in stream_events:
            adapter.validate_python(event)
        # Monitor input is the parsed request actually received from the official
        # SDK, not a handcrafted guess. No authorization headers are retained.
        exchanges = [{"request": json.loads(row["input"]), "response": json.loads(row["response"])}
                     for row in svc.api_requests]
        for exchange in exchanges:
            Response.model_validate(exchange["response"])  # validate actual wire data, independently of SDK parsing
    receipt = {"result": "pass", "client": "openai-python", "version": openai.__version__,
        "transport": "real loopback HTTP/SSE", "server": "Strata Service + MockEngine",
        "model_output_provenance": "synthetic scripts listed in tools/responses_sdk_probe.py",
        "native_model_test": False, "codex_test": False, "client_tool_effects": client_effects,
        "final_file": "after\n", "checks": ["text", "typed SSE", "SDK stream assembly", "SDK response model validation",
                                               "client-owned namespaced read/write/result loop", "manual full-item replay",
                                               "visible reasoning and genuine summaries", "authenticated encrypted-only replay",
                                               "strict SDK event model validation"],
        "exchanges": len(exchanges)}
    (a.out / "sdk-probes.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    (a.out / "sdk-loop.json").write_text(json.dumps(exchanges, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (a.out / "sdk-events.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in stream_events), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return a.out


if __name__ == "__main__":
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        output = main()
    messages = sorted(set(str(w.message) for w in observed))
    (output / "sdk-warnings.txt").write_text("\n\n".join(messages) + "\n", encoding="utf-8")
    print(f"SDK serialization warnings retained in sdk-warnings.txt: {len(messages)} distinct message(s)")
