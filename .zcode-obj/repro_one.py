import sys, json, requests
sys.path.insert(0, "F:/Strata-dev/tools")
import grammar_acceptance as G

srv, base = G.start_server(8299, 2, "F:/Strata-dev/.zcode-repro-serve.log")
try:
    G.warm(base)
    st = requests.get(base + "/v1/status", timeout=30).json()
    print("capability:", st.get("structured_output"), flush=True)
    schema = G.SCHEMAS["person"]
    r = requests.post(base + "/v1/chat/completions", stream=True, json={
        "messages": [{"role": "user",
                      "content": "Alice Nakamura is 34, lives in Osaka, and leads the platform team.\nAnswer with one JSON object."}],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "person", "schema": schema, "strict": True}},
        "max_tokens": 2400, "reasoning_effort": "low", "temperature": 0.6, "stream": True}, timeout=900)
    print("SSE", r.status_code, flush=True)
    if r.status_code != 200:
        print("BODY", r.text[:1200], flush=True)
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data:") and "[DONE]" not in line:
            try:
                d0 = json.loads(line[5:])
            except Exception:
                continue
            for ch0 in d0.get("choices") or []:
                de = ch0.get("delta") or {}
                if de.get("content"):
                    print("DELTA", repr(de["content"][:120]), flush=True)
                if de.get("reasoning_content"):
                    print("THINK", repr(de["reasoning_content"][:120]), flush=True)
                if ch0.get("finish_reason"):
                    print("FINISH", ch0["finish_reason"], flush=True)
finally:
    srv.terminate()
    srv.wait(timeout=60)
print("REPRO-DONE")
