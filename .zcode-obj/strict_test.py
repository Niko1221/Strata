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
for temp in (1.3,):
  for think in (True,False):
    ok=0
    for p in HARD:
      body={"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":3000,"temperature":temp,
        "messages":[{"role":"user","content":p}],"chat_template_kwargs":{"enable_thinking":think},
        "response_format":{"type":"json_schema","json_schema":{"name":"x","schema":SCH,"strict":True}}}
      req=urllib.request.Request(URL,data=json.dumps(body).encode(),
          headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
      t0=time.time()
      try:
        with urllib.request.urlopen(req,timeout=900) as r: o=json.loads(r.read().decode())
      except urllib.error.HTTPError as e:
        print(f"  think={think} HTTP {e.code}: {e.read().decode()[:80]}"); continue
      c=o["choices"][0]["message"].get("content") or ""
      fr=o["choices"][0]["finish_reason"]
      clean=False
      try: json.loads(c); clean=True
      except Exception: pass
      if clean: ok+=1
      else: print(f"  think={think} DIRTY finish={fr} len={len(c)} head={c[:55]!r} tail={c[-35:]!r} {time.time()-t0:.0f}s")
    print(f"STRICT think={think} temp={temp}: {ok}/5 pure JSON")
