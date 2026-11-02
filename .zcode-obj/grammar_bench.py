import json, sys, time, urllib.request
KEY="sk-ouzhiyingyyds666!"
URL="http://127.0.0.1:8081/v1/chat/completions"

# a schema that forces thousands of tokens: a long array of structured records
SCHEMA={"type":"object","properties":{
  "paper":"string","comparison":{"type":"array","items":{
     "type":"object","properties":{
        "variant":{"type":"string"},
        "year":{"type":"integer"},
        "recurrence":{"type":"string"},
        "complexity_time":{"type":"string"},
        "complexity_mem":{"type":"string"},
        "improvement":{"type":"string"}},
     "required":["variant","year","recurrence","complexity_time","complexity_mem","improvement"],
     "additionalProperties":False}}},
 "required":["paper","comparison"],"additionalProperties":False}

PROMPT=("List at least 12 linear-attention variants with their year, recurrence form, "
        "time and memory complexity, and the key improvement. JSON only.")

def run(temp, label, grammar=True, mx=6000):
    body={"model":"Qwen3.8-Flash-Next","stream":True,"max_tokens":mx,"temperature":temp,
          "messages":[{"role":"user","content":PROMPT}]}
    if grammar:
        body["response_format"]={"type":"json_schema",
            "json_schema":{"name":"cmp","schema":SCHEMA,"strict":True}}
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    t0=time.time(); ans=[]; think=[]; usage=None; finish=None; err=None
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
                rc=dl.get("reasoning_content") or dl.get("reasoning")
                if rc: think.append(rc)
                if dl.get("content"): ans.append(dl["content"])
                if ch.get("finish_reason"): finish=ch["finish_reason"]
    dt=time.time()-t0
    u=usage or {}; out=u.get("completion_tokens") or 0
    a="".join(ans)
    print(f"{label}: out={out} tok wall={dt:.1f}s {out/dt:.1f} tok/s finish={finish}")
    if err:
        print(f"   ERROR {err.get('code')}: {err.get('message','')[:200]}"); return
    if grammar:
        try:
            d=json.loads(a)
            n=len(d.get("comparison") or [])
            print(f"   GRAMMAR OK: {n} variants, first={list((d.get('comparison') or [{}])[0])[:3]}")
        except Exception as e:
            print(f"   GRAMMAR FAIL: {e} :: {a[:200]!r}")
    else:
        print(f"   answer chars={len(a)}")

if __name__=="__main__":
    run(0.6,"grammar temp=0.6 (gated)", grammar=True)
    run(0.6,"plain   temp=0.6 (gated)", grammar=False)
