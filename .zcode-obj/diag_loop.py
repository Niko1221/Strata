import json,urllib.request
KEY="sk-ouzhiyingyyds666!"; URL="http://127.0.0.1:8082/v1/chat/completions"
SCH={"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{
  "name":{"type":"string"},"year":{"type":"integer"},"complexity":{"type":"string"}},
  "required":["name","year","complexity"],"additionalProperties":False}}},"required":["items"],"additionalProperties":False}
P="Write me a poem about autumn. Three stanzas."
for i in range(5):
  body={"model":"Qwen3.8-Flash-Next","stream":True,"max_tokens":3000,"temperature":1.3,
    "messages":[{"role":"user","content":P}],"chat_template_kwargs":{"enable_thinking":True},
    "response_format":{"type":"json_schema","json_schema":{"name":"x","schema":SCH,"strict":True}}}
  req=urllib.request.Request(URL,data=json.dumps(body).encode(),
      headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
  reason=""; content=""; fr=None
  try:
    with urllib.request.urlopen(req,timeout=900) as r:
      for raw in r:
        line=raw.decode("utf-8","replace").strip()
        if not line.startswith("data: "): continue
        d=line[6:]
        if d=="[DONE]": break
        try: o=json.loads(d)
        except Exception: continue
        ch=(o.get("choices") or [{}])[0]; dl=ch.get("delta") or {}
        if dl.get("reasoning_content"): reason+=dl["reasoning_content"]
        if dl.get("content"): content+=dl["content"]
        if ch.get("finish_reason"): fr=ch["finish_reason"]
  except urllib.error.HTTPError as e:
    print(f"#{i+1} HTTP {e.code} reason={len(reason)} content={len(content)} tail={repr(content[-60:])}")
    continue
  lead=len(content)-len(content.lstrip())
  print(f"#{i+1} finish={fr} reason={len(reason)}ch content={len(content)}ch lead_ws={lead} fenced={content.lstrip().startswith('`')}")
