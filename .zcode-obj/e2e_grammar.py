import json, time, urllib.request

KEY = "sk-ouzhiyingyyds666!"
BASE = "http://127.0.0.1:8081/v1/chat/completions"

SCHEMA = {"type": "object", "properties": {
    "city": {"type": "string"}, "temp_c": {"type": "number"}, "conditions": {"type": "string"}},
    "required": ["city", "temp_c", "conditions"], "additionalProperties": False}

def stream(body, timeout=900):
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    text, usage, finish, err = [], None, None, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            d = line[6:].strip()
            if d == "[DONE]":
                break
            try:
                obj = json.loads(d)
            except json.JSONDecodeError:
                continue
            if "error" in obj:
                err = obj["error"]; break
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    text.append(delta["content"])
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    return "".join(text), usage, finish, err, time.time() - t0

def json_ok(content):
    if not content.strip():
        return False, "empty"
    try:
        d = json.loads(content)
    except Exception as e:
        return False, f"unparseable ({e}) :: {content[:200]!r}"
    missing = [k for k in ("city", "temp_c", "conditions") if k not in d]
    if missing:
        return False, f"missing {missing} :: {content[:200]!r}"
    types = {"city": str, "temp_c": (int, float), "conditions": str}
    bad = [k for k, t in types.items() if not isinstance(d[k], t)]
    if bad:
        return False, f"wrong types {bad} :: {content[:200]!r}"
    return True, f"{content[:110]}"

def run(name, body, validate, budget=2000):
    body = {**body, "max_tokens": budget}
    content, usage, finish, err, dt = stream(body)
    toks = (usage or {}).get("completion_tokens") or 0
    if err:
        print(f"{name}: ERROR {err.get('code')} :: {err.get('message','')[:150]}")
        return
    ok, detail = validate(content)
    sp = f"{toks/dt:.1f} tok/s" if dt > 0 else "-"
    print(f"{name}: {'OK  ' if ok else 'FAIL'} {dt:5.1f}s tok={toks:4d} {sp:>11s} finish={finish}")
    print(f"        {detail}")

base = {"model": "Qwen3.8-Flash-Next", "stream": True}
prompt = [{"role": "user", "content": "What is the weather in Reykjavik right now? Answer."}]
fmt = {"type": "json_schema", "json_schema": {"name": "w", "schema": SCHEMA, "strict": True}}

run("grammar temp=0.6 (gated on)", {**base, "messages": prompt, "temperature": 0.6, "response_format": fmt}, json_ok)
run("grammar temp=0   (greedy)  ", {**base, "messages": prompt, "temperature": 0,   "response_format": fmt}, json_ok)
run("plain   temp=0.6          ", {**base, "messages": prompt, "temperature": 0.6},
    lambda c: (len(c.strip()) > 10, f"len={len(c)} text={c.strip()[:90]!r}"))
