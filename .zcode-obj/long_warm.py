import json,os,re,time,urllib.request
KEY="sk-ouzhiyingyyds666!"; URL="http://127.0.0.1:8082/v1/chat/completions"
LOG=r"F:\Strata-dev\.zcode-obj\engine-gated-stderr.log"
SCH={"type":"object","properties":{"items":{"type":"array","minItems":35,"items":{"type":"object","properties":{
  "name":{"type":"string"},"year":{"type":"integer"},"complexity":{"type":"string"},
  "basis":{"type":"string"},"note":{"type":"string"}},
  "required":["name","year","complexity","basis","note"],"additionalProperties":False}}},"required":["items"],"additionalProperties":False}
P="List 40 linear attention variants with year, complexity, basis and a one-sentence note. JSON only."
def run(grammar, temp=0.0):
    body={"model":"Qwen3.8-Flash-Next","stream":True,"max_tokens":4000,"temperature":temp,
      "messages":[{"role":"user","content":P}],"chat_template_kwargs":{"enable_thinking":False}}
    if grammar: body["response_format"]={"type":"json_schema","json_schema":{"name":"x","schema":SCH,"strict":True}}
    since=os.path.getsize(LOG)
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    ans=[]
    with urllib.request.urlopen(req,timeout=1800) as r:
        for raw in r:
            l=raw.decode("utf-8","replace").strip()
            if l.startswith("data: "):
                d=l[6:].strip()
                if d=="[DONE]": break
                try: o=json.loads(d)
                except: continue
                for ch in o.get("choices") or []:
                    c=(ch.get("delta") or {}).get("content")
                    if c: ans.append(c)
    time.sleep(1.2)
    with open(LOG,encoding="utf-8",errors="replace") as f:
        f.seek(since); t=f.read()
    m=re.search(r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)", t)
    m2=re.search(r"drafts accepted (\d+) of (\d+)", t)
    br=[ln.split("break row:")[-1].strip() for ln in t.splitlines() if "break row" in ln]
    cw=[ln.split("grammar chain: ")[-1].split("strata")[0].strip() for ln in t.splitlines() if "grammar chain:" in ln]
    a="".join(ans)
    try: n_items=len(json.loads(a)["items"])
    except Exception as e: n_items=f"INVALID({e})"
    if m:
        print(f"  grammar={'Y' if grammar else 'N'}: {m.group(1)} tok {float(m.group(2))/1000.:.1f}s {m.group(3)} tok/s "
              f"accept={m2.group(1)}/{m2.group(2)} items={n_items}")
        if cw: print(f"      chain: {cw[0]}")
        if br: print(f"      breaks: {br[0]}")
    else: print(f"  grammar={grammar}: no telemetry")
print("== grammar long (warm)")
run(True); run(True)
print("== plain long")
run(False); run(False)
