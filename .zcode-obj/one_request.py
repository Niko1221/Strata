import json,os,sys,time,urllib.request
KEY="sk-ouzhiyingyyds666!"; URL="http://127.0.0.1:8082/v1/chat/completions"
LOG=r"F:\Strata-dev\.zcode-obj\engine-gated-stderr.log"
SCHEMA={"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{"n":{"type":"string"},"y":{"type":"integer"}},"required":["n","y"],"additionalProperties":False}}},"required":["items"],"additionalProperties":False}
mode=sys.argv[1]
body={"model":"Qwen3.8-Flash-Next","stream":True,"max_tokens":1200,"temperature":0.6,
      "messages":[{"role":"user","content":"List 12 linear attention variants with year. JSON only."}],
      "chat_template_kwargs":{"enable_thinking":False}}
if mode=="grammar":
    body["response_format"]={"type":"json_schema","json_schema":{"name":"x","schema":SCHEMA,"strict":True}}
since=os.path.getsize(LOG)
req=urllib.request.Request(URL,data=json.dumps(body).encode(),
    headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
t0=time.time(); ans=[]
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
dt=time.time()-t0
time.sleep(1.5)
with open(LOG,encoding="utf-8",errors="replace") as f:
    f.seek(since); tail=f.read()
a="".join(ans)
print(f"--- {mode}: {dt:.1f}s chars={len(a)}")
if mode=="grammar":
    try:
        d=json.loads(a); print("   GRAMMAR OK items=",len(d.get("items") or []))
    except Exception as e: print("   GRAMMAR FAIL",e,repr(a[:120]))
for ln in tail.splitlines():
    if any(k in ln for k in ("generated in","suffix drafts","lookup chain","accepted")):
        print("   ",ln.split("strata serve: ")[-1])
