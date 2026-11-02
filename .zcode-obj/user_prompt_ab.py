import json,os,time,urllib.request
KEY="sk-ouzhiyingyyds666!"; URL="http://127.0.0.1:8082/v1/chat/completions"
LOG=r"F:\Strata-dev\.zcode-obj\engine-gated-stderr.log"
P=open(".zcode-obj/prompt.txt",encoding="utf-8").read()
import re
def run(think, mx, tag):
    body={"model":"Qwen3.8-Flash-Next","stream":True,"max_tokens":mx,"temperature":0.6,
      "messages":[{"role":"user","content":P}],"chat_template_kwargs":{"enable_thinking":think}}
    since=os.path.getsize(LOG)
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    t0=time.time(); ans=[]; think_c=[]
    with urllib.request.urlopen(req,timeout=1800) as r:
        for raw in r:
            l=raw.decode("utf-8","replace").strip()
            if not l.startswith("data: "): continue
            d=l[6:].strip()
            if d=="[DONE]": break
            try: o=json.loads(d)
            except: continue
            for ch in o.get("choices") or []:
                dl=ch.get("delta") or {}
                if dl.get("reasoning_content"): think_c.append(dl["reasoning_content"])
                if dl.get("content"): ans.append(dl["content"])
    dt=time.time()-t0; time.sleep(1.5)
    with open(LOG,encoding="utf-8",errors="replace") as f:
        f.seek(since); tail=f.read()
    m=re.search(r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)", tail)
    m2=re.search(r"drafts accepted (\d+) of (\d+)", tail)
    a="".join(ans); th="".join(think_c)
    if m:
        n,ms,sp=m.group(1),float(m.group(2))/1000,int(float(m.group(3)))
        acc=f"{m2.group(1)}/{m2.group(2)} ({100*int(m2.group(1))/int(m2.group(2)):.1f}%)" if m2 else "-"
        print(f"{tag}: {n} tok in {ms:.1f}s -> {sp} tok/s | drafts {acc} | thinking={len(th)}ch answer={len(a)}ch")
    else:
        print(f"{tag}: no telemetry (wall {dt:.1f}s, answer {len(a)}ch)")
run(False, 3000, "thinking OFF, 3000 tok")
run(True,  6000, "thinking ON,  6000 tok")
