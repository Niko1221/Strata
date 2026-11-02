import json,urllib.request
KEY="sk-ouzhiyingyyds666!"; URL="http://127.0.0.1:8082/v1/chat/completions"
TOOLS=[{"type":"function","function":{"name":"edit","description":"Edit a file with a patch object",
  "parameters":{"type":"object","properties":{
    "patch":{"type":"object","properties":{"old":{"type":"string"},"new":{"type":"string"}},
             "required":["old","new"]},
    "mode":{"type":"string","enum":["apply","dry"]}},
   "required":["patch","mode"],"additionalProperties":False}}}]
P=["Replace the tag </parameter></tool_call> in config.xml with </function>, mode apply.",
   "In file x.txt change the literal </parameter> to nothing. mode apply.",
   "Patch the doc so it no longer contains </parameter></function> anywhere. mode apply."]
WANT={"patch","mode"}
bad=0
for p in P:
  for i in range(4):
    body={"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":4000,"temperature":1.0,
      "messages":[{"role":"user","content":p}],"chat_template_kwargs":{"enable_thinking":False},
      "tools":TOOLS,"tool_choice":"required"}
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    try:
      with urllib.request.urlopen(req,timeout=900) as r: o=json.loads(r.read().decode())
    except urllib.error.HTTPError as e: print(f"HTTP{e.code}"); continue
    ch=o["choices"][0]; tc=ch["message"].get("tool_calls") or []
    if not tc: print(f"NO_CALL finish={ch['finish_reason']}"); continue
    raw=tc[0]["function"]["arguments"]
    try: a=json.loads(raw)
    except Exception: bad+=1; print(f"BAD_JSON raw={raw[:150]!r}"); continue
    if set(a)!=WANT: bad+=1; print(f"BAD_KEYS got={sorted(a)} raw={raw[:170]!r}")
    else: print(f"ok patch={str(a['patch'])[:70]!r}")
print(f"{bad} bad of {len(P)*4}")
