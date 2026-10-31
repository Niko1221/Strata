import sys, json, requests
sys.path.insert(0, "F:/Strata-dev/tools")
import grammar_acceptance as G

srv, base = G.start_server(8301, 2, "F:/Strata-dev/.zcode-repro5-serve.log")
try:
    G.warm(base)
    prompt = "Tokyo today: 24 C, light breeze at 8 km/h, overcast."
    schema = G.SCHEMAS["weather"]
    r = requests.post(base + "/v1/chat/completions", stream=True, json={
        "messages": [{"role": "user", "content": prompt + "\nAnswer with one JSON object."}],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "weather", "schema": schema, "strict": True}},
        "max_tokens": 2400, "reasoning_effort": "low", "temperature": 0.6, "stream": True}, timeout=900)
    print("SSE", r.status_code, flush=True)
    think = []
    content = []
    finish = None
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data:") and "[DONE]" not in line:
            try:
                d0 = json.loads(line[5:])
            except Exception:
                continue
            for ch0 in d0.get("choices") or []:
                de = ch0.get("delta") or {}
                if de.get("reasoning_content"):
                    think.append(de["reasoning_content"])
                if de.get("content"):
                    content.append(de["content"])
                if ch0.get("finish_reason"):
                    finish = ch0["finish_reason"]
    t = "".join(think)
    c = "".join(content)
    print("FINISH", finish, flush=True)
    print("THINK-CHARS", len(t), "CONTENT-CHARS", len(c), flush=True)
    print("THINK-HEAD", repr(t[:200]), flush=True)
    print("THINK-TAIL", repr(t[-300:]), flush=True)
    print("CONTENT-HEAD", repr(c[:200]), flush=True)
    print("CONTENT-TAIL", repr(c[-400:]), flush=True)
finally:
    srv.terminate()
    srv.wait(timeout=60)
print("REPRO5-DONE")
