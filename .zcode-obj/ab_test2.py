import json,os,re,sys,time,urllib.request,statistics
KEY="sk-ouzhiyingyyds666!"; URL="http://127.0.0.1:8082/v1/chat/completions"
LOG=r"F:\Strata-dev\.zcode-obj\engine-gated-stderr.log"
LABEL=sys.argv[1]
TOOLS=[{"type":"function","function":{"name":"write_file","description":"Write a file",
  "parameters":{"type":"object","properties":{
    "path":{"type":"string"},"content":{"type":"string"},
    "mode":{"type":"string","enum":["create","append","overwrite"]}},
   "required":["path","content","mode"],"additionalProperties":False}}},
 {"type":"function","function":{"name":"run_command","description":"Run a shell command",
  "parameters":{"type":"object","properties":{"cmd":{"type":"string"},"timeout_s":{"type":"integer"}},
   "required":["cmd","timeout_s"],"additionalProperties":False}}}]
SCH={"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{
  "name":{"type":"string"},"year":{"type":"integer"},"complexity":{"type":"string"},"note":{"type":"string"}},
  "required":["name","year","complexity","note"],"additionalProperties":False}}},"required":["items"],"additionalProperties":False}

def call(body, gram=True):
    if not gram: body = dict(body, strata_grammar=False)
    since=os.path.getsize(LOG)
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    t0=time.time()
    try:
        with urllib.request.urlopen(req,timeout=1500) as r: o=json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return dict(err=f"HTTP {e.code}")
    dt=time.time()-t0; time.sleep(0.6)
    with open(LOG,encoding="utf-8",errors="replace") as f:
        f.seek(since); t=f.read()
    mp=re.search(r"prompt (\d+) tokens", t)
    mg=re.search(r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)", t)
    ch=o["choices"][0]; msg=ch["message"]; tc=msg.get("tool_calls") or []
    active = "gdbg accept" in t or "grammar chain" in t
    # engine's own accounting: prompt + generated = what the engine actually ran
    p=int(mp.group(1)) if mp else -1; g=int(mg.group(1)) if mg else -1
    return dict(prompt=p,gen=g,ms=float(mg.group(2))/1000. if mg else dt,
                sp=float(mg.group(3)) if mg else 0.0, finish=ch.get("finish_reason"),
                tc=tc, content=(msg.get("content") or ""), dt=dt, gram_active=active)

SC={
 "tool_1call": lambda g: call({"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":500,"temperature":0.0,
   "messages":[{"role":"user","content":"Create a file at src/main.py containing a hello world script. Use create mode."}],
   "chat_template_kwargs":{"enable_thinking":False},"tools":TOOLS,"tool_choice":"auto"}, g),
 "json_10": lambda g: call({"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":2500,"temperature":0.0,
   "messages":[{"role":"user","content":"List 10 linear attention variants with year, complexity and a note. JSON only."}],
   "chat_template_kwargs":{"enable_thinking":False},
   "response_format":{"type":"json_schema","json_schema":{"name":"x","schema":SCH,"strict":True}}}, g),
 "json_think": lambda g: call({"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":4000,"temperature":0.0,
   "messages":[{"role":"user","content":"List 10 linear attention variants with year, complexity and a note. JSON only."}],
   "chat_template_kwargs":{"enable_thinking":True},
   "response_format":{"type":"json_schema","json_schema":{"name":"x","schema":SCH,"strict":True}}}, g),
 "chat_plain": lambda g: call({"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":800,"temperature":0.0,
   "messages":[{"role":"user","content":"Explain in 250 words why linear attention saves memory versus full attention."}],
   "chat_template_kwargs":{"enable_thinking":False}}, g),
}
N=int(sys.argv[2]) if len(sys.argv)>2 else 3
res={}
for name,fn in SC.items():
    for gram in (True, False):
      rows=[]
      tag = f"{name}[{'grammar' if gram else 'prompt '}]"
      for i in range(N):
        r=fn(gram)
        if r.get("err"): print(f"  {LABEL} {tag}: {r['err']}"); continue
        if name.startswith("tool"):
            r["valid"]=False
            if r["tc"]:
                try: json.loads(r["tc"][0]["function"]["arguments"]); r["valid"]=True
                except Exception: pass
            r["what"]=r["tc"][0]["function"]["name"] if r["tc"] else "NO TOOL CALL"
        elif name.startswith("json"):
            try: r["valid"]=True; r["what"]=f"items={len(json.loads(r['content'])['items'])}"
            except Exception: r["valid"]=False; r["what"]="BROKEN JSON"
        else:
            r["valid"]=None; r["what"]=f"{len(r['content'])}ch"
        rows.append(r)
        print(f"  {LABEL} {tag} #{i+1}: prompt={r['prompt']} gen={r['gen']} {r['sp']:6.1f}t/s eng_gram={r['gram_active']} {r['what']}")
      if rows:
        res[tag]=dict(gen=statistics.median([x['gen'] for x in rows]),
                       sp=statistics.median([x['sp'] for x in rows]),
                       ms=statistics.median([x['ms'] for x in rows]),
                       prompt=rows[0]['prompt'],
                       all_valid=all(x['valid'] for x in rows if x['valid'] is not None),
                       gram=rows[0]['gram_active'], what=rows[0]['what'])
json.dump(res,open(f".zcode-obj/ab2-{LABEL}.json","w",encoding="utf-8"),indent=1)
print(f"saved ab2-{LABEL}.json")
