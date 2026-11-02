import json,os,time,urllib.request
KEY="sk-ouzhiyingyyds666!"; URL="http://127.0.0.1:8082/v1/chat/completions"
LOG=r"F:\Strata-dev\.zcode-obj\engine-gated-stderr.log"
SCHEMA={"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{
  "name":{"type":"string"},"year":{"type":"integer"},"complexity":{"type":"string"}},
  "required":["name","year","complexity"],"additionalProperties":False}}},"required":["items"],"additionalProperties":False}
P="List 10 linear attention variants with year and complexity. JSON only."

def run(temp, grammar, mx=1500):
    body={"model":"Qwen3.8-Flash-Next","stream":True,"max_tokens":mx,"temperature":temp,
      "messages":[{"role":"user","content":P}],"chat_template_kwargs":{"enable_thinking":False}}
    if grammar: body["response_format"]={"type":"json_schema","json_schema":{"name":"x","schema":SCHEMA,"strict":True}}
    since=os.path.getsize(LOG)
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    t0=time.time(); ans=[]; err=None
    with urllib.request.urlopen(req,timeout=1800) as r:
        for raw in r:
            l=raw.decode("utf-8","replace").strip()
            if not l.startswith("data: "): continue
            d=l[6:].strip()
            if d=="[DONE]": break
            try: o=json.loads(d)
            except: continue
            if "error" in o: err=o["error"]; break
            for ch in o.get("choices") or []:
                c=(ch.get("delta") or {}).get("content")
                if c: ans.append(c)
    dt=time.time()-t0; time.sleep(1.2)
    with open(LOG,encoding="utf-8",errors="replace") as f:
        f.seek(since); tail=f.read()
    import re
    tok=acc=off=0; eng_ms=0.0
    for ln in tail.splitlines():
        m=re.search(r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)", ln)
        if m:
            tok=int(m.group(1)); eng_ms=float(m.group(2))/1000.0
        m2=re.search(r"drafts accepted (\d+) of (\d+)", ln)
        if m2: acc=int(m2.group(1)); off=int(m2.group(2))
    a="".join(ans)
    lab=f"temp={temp} grammar={'Y' if grammar else 'N'}"
    if err: print(f"{lab}: ERROR {err.get('code')}"); return
    rate=f"{100.0*acc/off:.1f}%" if off else "-"
    sfx = [ln.split("suffix drafts: ")[-1] for ln in tail.splitlines() if "suffix drafts:" in ln]
    ok=""
    if grammar:
        try: ok=f"valid JSON items={len(json.loads(a)['items'])}"
        except Exception as e: ok=f"INVALID: {e}"
    sp=f"{tok/eng_ms:6.1f} tok/s (engine)" if eng_ms else "?"
    print(f"{lab}: wall={dt:5.1f}s tok={tok:4d} {sp}  accept={acc}/{off} ({rate})  {ok}")
    if sfx: print(f"        suffix: {sfx[0]}")

run(0.6, True); run(0.6, False)
run(0.0, True); run(0.0, False)
