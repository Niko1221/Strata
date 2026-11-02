import json, os, sys, time, urllib.request
KEY="sk-ouzhiyingyyds666!"
URL="http://127.0.0.1:8082/v1/chat/completions"
LOG=r"F:\Strata-dev\.zcode-obj\engine-gated-stderr.log"

SCHEMA={"type":"object","properties":{
  "paper":"string","comparison":{"type":"array","items":{
     "type":"object","properties":{
        "variant":{"type":"string"},"year":{"type":"integer"},
        "recurrence":{"type":"string"},"complexity_time":{"type":"string"},
        "complexity_mem":{"type":"string"},"improvement":{"type":"string"}},
     "required":["variant","year","recurrence","complexity_time","complexity_mem","improvement"],
     "additionalProperties":False}}},
 "required":["paper","comparison"],"additionalProperties":False}

PROMPT=("List at least 12 linear-attention variants with their year, recurrence form, "
        "time and memory complexity, and the key improvement. JSON only.")

def logsize():
    try: return os.path.getsize(LOG)
    except OSError: return 0

def newlines(since):
    with open(LOG, encoding="utf-8", errors="replace") as f:
        f.seek(since)
        return f.read()

def run(temp, label, grammar, mx=4000):
    body={"model":"Qwen3.8-Flash-Next","stream":True,"max_tokens":mx,"temperature":temp,
          "messages":[{"role":"user","content":PROMPT}],
          "chat_template_kwargs":{"enable_thinking":False}}
    if grammar:
        body["response_format"]={"type":"json_schema",
            "json_schema":{"name":"cmp","schema":SCHEMA,"strict":True}}
    since=logsize()
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    t0=time.time(); ans=[]; usage=None; finish=None; err=None
    with urllib.request.urlopen(req,timeout=1800) as r:
        for raw in r:
            l=raw.decode("utf-8","replace").strip()
            if not l.startswith("data: "): continue
            d=l[6:].strip()
            if d=="[DONE]": break
            try: o=json.loads(d)
            except: continue
            if "error" in o: err=o["error"]; break
            if o.get("usage"): usage=o["usage"]
            for ch in o.get("choices") or []:
                dl=ch.get("delta") or {}
                if dl.get("content"): ans.append(dl["content"])
                if ch.get("finish_reason"): finish=ch["finish_reason"]
    dt=time.time()-t0
    time.sleep(1.0)
    txt=newlines(since)
    tel=[ln for ln in txt.splitlines() if "generated in" in ln]
    sfx=[ln for ln in txt.splitlines() if "suffix drafts" in ln]
    a="".join(ans)
    print(f"\n=== {label}  ({dt:.1f}s)")
    if err:
        print(f"   ERROR {err.get('code')}: {err.get('message','')[:160]}"); return
    for ln in tel:
        gen=ln.split("generated in")[1]
        acc=ln.split("drafts accepted")[1].strip() if "drafts accepted" in ln else "?"
        print(f"   gen:{gen.strip()}")
        print(f"   drafts accepted:{acc}")
    for ln in sfx: print(f"   {ln.split('strata serve: ')[-1]}")
    if grammar:
        try:
            d=json.loads(a); n=len(d.get("comparison") or [])
            print(f"   GRAMMAR OK: {n} records")
        except Exception as e: print(f"   GRAMMAR FAIL: {e} :: {a[:160]!r}")
    else:
        print(f"   chars={len(a)}")

if __name__=="__main__":
    run(0.6,"PLAIN   temp=0.6", False)
    time.sleep(2)
    run(0.6,"GRAMMAR temp=0.6", True)
    time.sleep(2)
    run(0.6,"GRAMMAR temp=0.6 (repeat)", True)
