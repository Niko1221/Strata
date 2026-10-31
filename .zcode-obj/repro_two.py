import sys, json
sys.path.insert(0, "F:/Strata-dev/tools")
import grammar_acceptance as G
import requests

SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
          "city": {"type": "string"}, "role": {"type": "string"}},
          "required": ["name", "age", "city", "role"], "additionalProperties": False}
FORMAT = {"type": "json_schema", "json_schema": {"name": "person", "schema": SCHEMA}}

proc, base = G.start_server(8299, 2, "F:/Strata-dev/.zcode-repro-serve.log")
try:
    
    G.warm(base)
    print("warm ok", flush=True)
    r = requests.post(base + "/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Tell me about a person."}],
        "response_format": FORMAT, "max_tokens": 2400,
        "reasoning_effort": "low", "temperature": 0.6}, timeout=900)
    print("HTTP", r.status_code, flush=True)
    body = r.json()
    ch = body.get("choices", [{}])[0]
    print("finish:", ch.get("finish_reason"), flush=True)
    print("CONTENT:", repr(ch.get("message", {}).get("content", ""))[:300], flush=True)
finally:
    proc.terminate()
    proc.wait(timeout=30)
    print("DONE", flush=True)
