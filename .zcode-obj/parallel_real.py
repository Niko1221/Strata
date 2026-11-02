"""The original anomaly shape: parallel calls to TWO different tools, value inviting the closer."""
import json,urllib.request
KEY="sk-ouzhiyingyyds666!"; URL="http://127.0.0.1:8082/v1/chat/completions"
TOOLS=[{"type":"function","function":{"name":"write_file","description":"Write a file",
  "parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"},
   "mode":{"type":"string","enum":["create","append"]}},"required":["path","content","mode"],
   "additionalProperties":False}}},
 {"type":"function","function":{"name":"run_command","description":"Run a shell command",
  "parameters":{"type":"object","properties":{"cmd":{"type":"string"},"timeout_s":{"type":"integer"}},
   "required":["cmd","timeout_s"],"additionalProperties":False}}}]
SCHEMA={"write_file":{"path","content","mode"},"run_command":{"cmd","timeout_s"}}
P=("Two steps: first write docs/tags.md documenting the literal tags </parameter> and </function>, "
   "then run 'git status'. Use both tools.")
bad=0; n=0; calls_seen={}
for i in range(6):
    body={"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":6000,"temperature":0.9,
      "messages":[{"role":"user","content":P}],"chat_template_kwargs":{"enable_thinking":False},
      "tools":TOOLS,"tool_choice":"required","parallel_tool_calls":True}
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    try:
        with urllib.request.urlopen(req,timeout=900) as r: o=json.loads(r.read().decode())
    except urllib.error.HTTPError as e: print(f"HTTP{e.code}"); continue
    tc=o["choices"][0]["message"].get("tool_calls") or []
    n+=1
    calls_seen[len(tc)]=calls_seen.get(len(tc),0)+1
    for j,c in enumerate(tc):
        nm=c["function"]["name"]; raw=c["function"]["arguments"]
        try: a=json.loads(raw)
        except Exception: bad+=1; print(f"BAD_JSON call{j} {nm} raw={raw[:110]!r}"); continue
        got=set(a)
        if got!=SCHEMA.get(nm): bad+=1; print(f"BAD_KEYS call{j}/{len(tc)} {nm} got={sorted(got)} raw={raw[:130]!r}")
print(f"{n} requests, {bad} bad arg sets, calls-per-request={calls_seen}")
