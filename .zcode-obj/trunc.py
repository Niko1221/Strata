import json,urllib.request
KEY="sk-ouzhiyingyyds666!"; URL="http://127.0.0.1:8082/v1/chat/completions"
TOOLS=[{"type":"function","function":{"name":"write_file","description":"Write a file",
  "parameters":{"type":"object","properties":{
    "path":{"type":"string"},"content":{"type":"string"},
    "mode":{"type":"string","enum":["create","append","overwrite"]}},
   "required":["path","content","mode"],"additionalProperties":False}}}]
P="Create src/main.py with a long hello world script that prints 50 lines. Use create mode."
for mx in (40,60,90,140,300):
  for gram in (True,False):
    body={"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":mx,"temperature":0.0,
      "messages":[{"role":"user","content":P}],"chat_template_kwargs":{"enable_thinking":False},
      "tools":TOOLS,"tool_choice":"required"}
    if not gram: body["strata_grammar"]=False
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    try:
      with urllib.request.urlopen(req,timeout=600) as r: o=json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
      print(f"max={mx:4d} gram={'ON ' if gram else 'OFF'} HTTP{e.code}"); continue
    ch=o["choices"][0]; m=ch["message"]; tc=m.get("tool_calls") or []
    fr=ch.get("finish_reason")
    if not tc:
      print(f"max={mx:4d} gram={'ON ' if gram else 'OFF'} finish={fr} NO_CALL content={(m.get('content') or '')[:40]!r}")
      continue
    raw=tc[0]["function"]["arguments"]
    try: a=json.loads(raw); keys=sorted(a); ok="parsed"
    except Exception as e: keys=None; ok=f"PARSE_FAIL"
    print(f"max={mx:4d} gram={'ON ' if gram else 'OFF'} finish={fr} {ok} keys={keys} rawlen={len(raw)} tail={raw[-30:]!r}")
