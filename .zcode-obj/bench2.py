import json, sys, time, urllib.request
KEY="sk-ouzhiyingyyds666!"
URL="http://127.0.0.1:8081/v1/chat/completions"
PROMPT=open(".zcode-obj/prompt.txt",encoding="utf-8").read()

def run(temp, mx, label):
    body={"model":"Qwen3.8-Flash-Next","stream":True,"max_tokens":mx,"temperature":temp,
          "messages":[{"role":"user","content":PROMPT}]}
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    t0=time.time(); t_first=None; think=[]; ans=[]; usage=None; finish=None
    with urllib.request.urlopen(req,timeout=1800) as r:
        for raw in r:
            l=raw.decode("utf-8","replace").strip()
            if not l.startswith("data: "): continue
            d=l[6:].strip()
            if d=="[DONE]": break
            try: o=json.loads(d)
            except: continue
            if "error" in o:
                print(f"{label}: ERROR {o['error'].get('code')}: {o['error'].get('message','')[:200]}"); return
            if o.get("usage"): usage=o["usage"]
            for ch in o.get("choices") or []:
                dl=ch.get("delta") or {}
                rc=dl.get("reasoning_content") or dl.get("reasoning")
                if rc: think.append(rc)
                if dl.get("content"):
                    if t_first is None: t_first=time.time()
                    ans.append(dl["content"])
                if ch.get("finish_reason"): finish=ch["finish_reason"]
    dt=time.time()-t0
    u=usage or {}
    out=u.get("completion_tokens") or 0
    a="".join(ans); th="".join(think)
    print(f"{label}")
    print(f"   tokens: prompt={u.get('prompt_tokens')} out={out} (thinking chars={len(th)}, answer chars={len(a)})")
    print(f"   wall={dt:.1f}s  -> {out/dt:.1f} tok/s   finish={finish}")
    print(f"   answer head: {a[:180]!r}")

if __name__=="__main__":
    for s in sys.argv[1:]:
        t,m,l=s.split(":"); run(float(t),int(m),l)
