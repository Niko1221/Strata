import json,os,re,time,urllib.request,statistics
KEY="sk-ouzhiyingyyds666!"; URL="http://127.0.0.1:8082/v1/chat/completions"
LOG=r"F:\Strata-dev\.zcode-obj\engine-gated-stderr.log"
SCH={"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{
  "name":{"type":"string"},"year":{"type":"integer"},"complexity":{"type":"string"}},
  "required":["name","year","complexity"],"additionalProperties":False}}},"required":["items"],"additionalProperties":False}
P="List 12 linear attention variants with year and complexity. JSON only."
def run(grammar, temp=0.0, mx=1500):
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
            if l.startswith("data: "):
                d=l[6:].strip()
                if d=="[DONE]": break
                try: o=json.loads(d)
                except: continue
                for ch in o.get("choices") or []:
                    c=(ch.get("delta") or {}).get("content")
                    if c: ans.append(c)
    time.sleep(0.8)
    with open(LOG,encoding="utf-8",errors="replace") as f:
        f.seek(since); t=f.read()
    m=re.search(r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)", t)
    m2=re.search(r"drafts accepted (\d+) of (\d+)", t)
    cw=[ln.split("grammar chain: ")[-1].split(" strata")[0].strip() for ln in t.splitlines() if "grammar chain:" in ln]
    ok=""
    if grammar:
        try: ok=f"items={len(json.loads(''.join(ans))['items'])}"
        except Exception as e: ok=f"INVALID({e})"
    if not m: return None
    return dict(sp=float(m.group(3)), acc=int(m2.group(1))/int(m2.group(2)) if m2 else 0,
                chain=cw[0] if cw else "-", ok=ok)
def bench(n, tag):
    g=[];p=[]
    for i in range(n):
        r=run(True); s=run(False)
        if r: g.append(r); p.append(s)
        print(f"  iter{i+1} grammar {r['sp']:.1f} (acc {r['acc']*100:.1f}%, {r['chain']}) | plain {s['sp']:.1f}")
    gm=statistics.median([x['sp'] for x in g]); pm=statistics.median([x['sp'] for x in p])
    print(f"\n{tag}: grammar median {gm:.1f} tok/s, plain median {pm:.1f} tok/s -> {100*(gm-pm)/pm:+.1f}%")
    print(f"  grammar accept median {100*statistics.median([x['acc'] for x in g]):.1f}% vs plain {100*statistics.median([x['acc'] for x in p]):.1f}%")
bench(5, "temp=0")
