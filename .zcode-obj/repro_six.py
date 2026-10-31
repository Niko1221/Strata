import sys, json, requests
sys.path.insert(0, "F:/Strata-dev/tools")
import grammar_acceptance as G

srv, base = G.start_server(8302, 2, "F:/Strata-dev/.zcode-repro6-serve.log")
try:
    G.warm(base)
    prompt = "Tokyo today: 24 C, light breeze at 8 km/h, overcast."
    schema = G.SCHEMAS["weather"]
    for attempt in range(8):
        r = requests.post(base + "/v1/chat/completions", stream=True, json={
            "messages": [{"role": "user", "content": prompt + "\nAnswer with one JSON object."}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "weather", "schema": schema, "strict": True}},
            "max_tokens": 2400, "reasoning_effort": "low", "temperature": 0.6, "stream": True}, timeout=900)
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
        ok = finish == "stop" and len(c) < 500
        print(f"attempt {attempt}: FINISH {finish} think={len(t)} content={len(c)}", flush=True)
        if not ok:
            print("THINK-HEAD", repr(t[:250]), flush=True)
            print("THINK-TAIL", repr(t[-400:]), flush=True)
            print("CONTENT-HEAD", repr(c[:250]), flush=True)
            print("CONTENT-TAIL", repr(c[-400:]), flush=True)
            break
finally:
    srv.terminate()
    srv.wait(timeout=60)
print("REPRO6-DONE")
