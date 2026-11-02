"""Strict typed params (ours) vs boundary-only free text (llama.cpp's tagged path).
Same prompts, same server, grammar on. Measures speed + whether types are right.
"""
import json,urllib.request,time,statistics
KEY="sk-ouzhiyingyyds666!"; URL="http://127.0.0.1:8082/v1/chat/completions"
TYPED=[{"type":"function","function":{"name":"edit","description":"Patch a file",
  "parameters":{"type":"object","properties":{
    "path":{"type":"string"},
    "patch":{"type":"object","properties":{"old":{"type":"string"},"new":{"type":"string"}},
             "required":["old","new"]},
    "line":{"type":"integer"},
    "mode":{"type":"string","enum":["apply","dry"]}},
   "required":["path","patch","line","mode"],"additionalProperties":False}}}]
# llama.cpp style: every parameter a plain string, no的结构约束
FREE=[{"type":"function","function":{"name":"edit","description":"Patch a file",
  "parameters":{"type":"object","properties":{
    "path":{"type":"string"},"patch":{"type":"string"},
    "line":{"type":"string"},"mode":{"type":"string"}},
   "required":["path","patch","line","mode"],"additionalProperties":False}}}]
P="Patch config.xml: replace 'timeout=30' with 'timeout=60', at line 42. apply mode."
def run(tools, n=4):
    ts, toks, accs, types_ok = [], [], [], 0
    for i in range(n):
        body={"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":1200,"temperature":0.7,
          "messages":[{"role":"user","content":P}],"chat_template_kwargs":{"enable_thinking":False},
          "tools":tools,"tool_choice":"required"}
        req=urllib.request.Request(URL,data=json.dumps(body).encode(),
            headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
        t0=time.time()
        with urllib.request.urlopen(req,timeout=900) as r: o=json.loads(r.read().decode())
        ts.append(time.time()-t0)
        u=o.get("usage") or {}; toks.append(u.get("completion_tokens",0))
        tc=o["choices"][0]["message"].get("tool_calls") or []
        if not tc: continue
        try: a=json.loads(tc[0]["function"]["arguments"])
        except Exception: continue
        good=isinstance(a.get("patch"),dict) and isinstance(a.get("line"),int) and a.get("mode") in ("apply","dry")
        types_ok += good
        if i==0: sample=a
    return ts, toks, types_ok, sample if 'sample' in dir() else None
for label,tools in (("TYPED (ours)",TYPED),("FREE string (llama.cpp style)",FREE)):

    ts,toks,ok,sample=run(tools)
    print(f"{label:30s} median={statistics.median(ts):5.2f}s tokens={statistics.median(toks):4.0f} type_correct={ok}/4")
    print(f"   sample: {json.dumps(sample)[:150] if sample else 'none'}")
