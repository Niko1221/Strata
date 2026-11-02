import json,os,re,time,urllib.request
KEY="sk-ouzhiyingyyds666!"; URL="http://127.0.0.1:8082/v1/chat/completions"
LOG=r"F:\Strata-dev\.zcode-obj\engine-gated-stderr.log"
SCH={"type":"object","properties":{"items":{"type":"array","minItems":30,"items":{"type":"object","properties":{
  "name":{"type":"string"},"year":{"type":"integer"},"complexity":{"type":"string"},
  "basis":{"type":"string"},"note":{"type":"string"}},
  "required":["name","year","complexity","basis","note"],"additionalProperties":False}}},"required":["items"],"additionalProperties":False}
P="List 40 linear attention variants with year, complexity, basis and a one-sentence note. JSON only."
def run(tag, grammar, mx=4000, temp=0.0):
    body={"model":"Qwen3.8-Flash-Next","stream":True,"max_tokens":mx,"temperature":temp,
      "messages":[{"role":"user","content":P}],"chat_template_kwargs":{"enable_thinking":False}}
    if grammar: body["response_format"]={"type":"json_schema","json_schema":{"name":"x","schema":SCH,"strict":True}}
    since=os.path.getsize(LOG)
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    ans=[]
    with urllib.request.urlopen(req,timeout=1800) as r:
        for raw in r:
            l=raw.decode("utf-8","replace").strip()
            if not l.startswith("data: "): continue
            d=l[6:].strip()
            if d=="[DONE]": break
            try: o=json.loads(d)
            except: continue
            for ch in o.get("choices") or []:
                c=(ch.get("delta") or {}).get("content")
                if c: ans.append(c)
    time.sleep(1.5)
    with open(LOG,encoding="utf-8",errors="replace") as f:
        f.seek(since); tail=f.read()
    a="".join(ans); out=[]
    for ln in tail.splitlines():
        if "grammar chain:" in ln or "break row" in ln or "drafts accepted" in ln or "generated in" in ln:
            out.append(ln.split("strata serve: ")[-1])
    ok=""
    if grammar:
        try: ok=f"valid JSON items={len(json.loads(a)['items'])}"
        except Exception as e: ok=f"INVALID {e}"
    print(f"--- {tag}: {ok}")
    for o in out: print("    ",o)
run("grammar long", True)
run("plain   long", False)
