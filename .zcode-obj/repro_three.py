import sys, requests
sys.path.insert(0, "F:/Strata-dev/tools")
import grammar_acceptance as G

proc, base = G.start_server(8299, 2, "F:/Strata-dev/.zcode-repro-serve.log")
try:
    G.warm(base)
    print("warm ok", flush=True)
    r = requests.post(base + "/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Tell me about a person."}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "person", "schema": {
            "type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                                             "city": {"type": "string"}, "role": {"type": "string"}},
            "required": ["name", "age", "city", "role"], "additionalProperties": False}}},
        "max_tokens": 2400, "reasoning_effort": "low", "temperature": 0.6}, timeout=900)
    print("HTTP", r.status_code, flush=True)
    body = r.json()
    ch = body.get("choices", [{}])[0]
    print("finish:", ch.get("finish_reason"), flush=True)
    msg = ch.get("message", {})
    print("CONTENT:", repr(msg.get("content", ""))[:400], flush=True)
    print("RC:", repr(msg.get("reasoning_content", ""))[-200:], flush=True)
    print("USAGE:", body.get("usage"), flush=True)
finally:
    proc.terminate()
    proc.wait(timeout=30)
    print("DONE", flush=True)
