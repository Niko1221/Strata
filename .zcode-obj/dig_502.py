import json,urllib.request,time
KEY="sk-ouzhiyingyyds666!"; URL="http://127.0.0.1:8082/v1/chat/completions"
SCH={"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{
  "name":{"type":"string"},"year":{"type":"integer"},"complexity":{"type":"string"}},
  "required":["name","year","complexity"],"additionalProperties":False}}},"required":["items"],"additionalProperties":False}
HARD=["Explain step by step why the sky is blue, then give me the JSON.\n\nActually just answer naturally, I'm curious.",
 "I'm not sure JSON is right for this. Tell me about Iceland's capital, its population, and a fun fact.",
 "Write me a poem about autumn. Three stanzas.",
 "Are you able to write files? If yes, create src/app.py with a Flask hello world, create mode.",
 "Which is better, tabs or spaces? Then run `git log --oneline -5` with a 20s timeout if you can."]
for p in HARD:
  body={"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":3000,"temperature":1.3,
    "messages":[{"role":"user","content":p}],"chat_template_kwargs":{"enable_thinking":True},
    "response_format":{"type":"json_schema","json_schema":{"name":"x","schema":SCH,"strict":True}}}
  req=urllib.request.Request(URL,data=json.dumps(body).encode(),
      headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
  try:
    with urllib.request.urlopen(req,timeout=900) as r: o=json.loads(r.read().decode())
    c=o["choices"][0]["message"].get("content") or ""
    try: json.loads(c); print(f"  OK   {p[:34]!r}")
    except Exception: print(f"  DIRTY {p[:34]!r} head={c[:60]!r}")
  except urllib.error.HTTPError as e:
    print(f"  HTTP {e.code} {p[:34]!r}\n     {e.read().decode()[:400]}")
