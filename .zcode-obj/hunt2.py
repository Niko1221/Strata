import json,urllib.request
KEY="sk-ouzhiyingyyds666!"; URL="http://127.0.0.1:8082/v1/chat/completions"
TOOLS=[{"type":"function","function":{"name":"write_file","description":"Write a file",
  "parameters":{"type":"object","properties":{
    "path":{"type":"string"},"content":{"type":"string"},
    "mode":{"type":"string","enum":["create","append","overwrite"]}},
   "required":["path","content","mode"],"additionalProperties":False}}},
 {"type":"function","function":{"name":"run_command","description":"Run a shell command",
  "parameters":{"type":"object","properties":{"cmd":{"type":"string"},"timeout_s":{"type":"integer"}},
   "required":["cmd","timeout_s"],"additionalProperties":False}}}]
HARD=["Explain step by step why the sky is blue, then give me the JSON. Actually just answer naturally.",
 "I'm not sure JSON is right for this. Tell me about Iceland's capital, its population, and a fun fact.",
 "Write me a poem about autumn. Three stanzas.",
 "Which is better, tabs or spaces? Then run `git log --oneline -5` with a 20s timeout if you can.",
 "Are you able to write files? If yes, create src/app.py with a Flask hello world, create mode."]
OKT={"write_file":{"path","content","mode"},"run_command":{"cmd","timeout_s"}}
bad=0; n=0
for think in (False,True):
 for i in range(12):
  for p in HARD:
    body={"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":6000,"temperature":1.0,
      "messages":[{"role":"user","content":p}],"chat_template_kwargs":{"enable_thinking":think},
      "tools":TOOLS,"tool_choice":"required"}
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    n+=1
    try:
      with urllib.request.urlopen(req,timeout=900) as r: o=json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
      continue
    m=o["choices"][0]["message"]; tc=m.get("tool_calls") or []
    if not tc: continue
    fn=tc[0]["function"]["name"]; raw=tc[0]["function"]["arguments"]
    try: a=json.loads(raw)
    except Exception:
      bad+=1; print(f"BAD_ARGS think={think} {fn} raw={raw[:300]!r}",flush=True); continue
    if fn in OKT and set(a)!=OKT[fn]:
      bad+=1
      print(f"BAD_KEYS think={think} tool={fn} got={sorted(a)} n_calls={len(tc)}",flush=True)
      print(f"   raw={raw[:300]!r}",flush=True)
      if len(tc)>1: print(f"   all_calls={[c['function']['name'] for c in tc]}",flush=True)
print(f"{n} requests, {bad} anomalies",flush=True)
