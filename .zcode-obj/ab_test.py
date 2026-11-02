import json,os,re,sys,time,urllib.request
KEY="sk-ouzhiyingyyds666!"; URL="http://127.0.0.1:8082/v1/chat/completions"
LOG=r"F:\Strata-dev\.zcode-obj\engine-gated-stderr.log"
LABEL=sys.argv[1]
TOOLS=[{"type":"function","function":{"name":"write_file","description":"Write a file",
  "parameters":{"type":"object","properties":{
    "path":{"type":"string","description":"file path"},
    "content":{"type":"string","description":"file content"},
    "mode":{"type":"string","enum":["create","append","overwrite"]}},
   "required":["path","content","mode"],"additionalProperties":False}}}]
JSON_SCH={"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{
  "name":{"type":"string"},"year":{"type":"integer"},"complexity":{"type":"string"}},
  "required":["name","year","complexity"],"additionalProperties":False}}},"required":["items"],"additionalProperties":False}

def call(body):
    since=os.path.getsize(LOG)
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    t0=time.time()
    try:
        with urllib.request.urlopen(req,timeout=1200) as r: o=json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return dict(err=f"HTTP {e.code}: {e.read().decode()[:120]}")
    dt=time.time()-t0; time.sleep(0.7)
    with open(LOG,encoding="utf-8",errors="replace") as f:
        f.seek(since); t=f.read()
    mp=re.search(r"prompt (\d+) tokens", t)
    mg=re.search(r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)", t)
    ch=o["choices"][0]; msg=ch["message"]; tc=msg.get("tool_calls") or []
    return dict(prompt=int(mp.group(1)) if mp else -1,
                gen=int(mg.group(1)) if mg else -1,
                ms=float(mg.group(2))/1000. if mg else dt,
                sp=float(mg.group(3)) if mg else 0.0,
                finish=ch.get("finish_reason"), tc=tc,
                content=(msg.get("content") or ""), dt=dt)

def S_tool():
    r=call({"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":500,"temperature":0.0,
      "messages":[{"role":"user","content":"Create a file at src/main.py containing a hello world script. Use create mode."}],
      "chat_template_kwargs":{"enable_thinking":False},"tools":TOOLS,"tool_choice":"auto"})
    if r.get("err"): return r
    r["valid"]=False
    if r["tc"]:
        try: json.loads(r["tc"][0]["function"]["arguments"]); r["valid"]=True
        except Exception: pass
    r["what"]=r["tc"][0]["function"]["name"] if r["tc"] else "NO TOOL CALL"
    return r

def S_json():
    r=call({"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":1500,"temperature":0.0,
      "messages":[{"role":"user","content":"List 10 linear attention variants with year and complexity. JSON only."}],
      "chat_template_kwargs":{"enable_thinking":False},
      "response_format":{"type":"json_schema","json_schema":{"name":"x","schema":JSON_SCH,"strict":True}}})
    if r.get("err"): return r
    try: r["valid"]=True; r["what"]=f"items={len(json.loads(r['content'])['items'])}"
    except Exception: r["valid"]=False; r["what"]="BROKEN JSON"
    return r

def S_think():
    r=call({"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":4000,"temperature":0.0,
      "messages":[{"role":"user","content":"List 10 linear attention variants with year and complexity. JSON only."}],
      "chat_template_kwargs":{"enable_thinking":True},
      "response_format":{"type":"json_schema","json_schema":{"name":"x","schema":JSON_SCH,"strict":True}}})
    if r.get("err"): return r
    try: r["valid"]=True; r["what"]=f"items={len(json.loads(r['content'])['items'])}"
    except Exception: r["valid"]=False; r["what"]="BROKEN JSON"
    return r

def S_plain():
    return call({"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":800,"temperature":0.0,
      "messages":[{"role":"user","content":"Explain in 200 words why linear attention saves memory versus full attention."}],
      "chat_template_kwargs":{"enable_thinking":False}})

out={}
for name,fn in [("tool",S_tool),("json",S_json),("think",S_think),("plain",S_plain)]:
    r=fn(); r["scenario"]=name; out[name]=r
    if r.get("err"): print(f"{LABEL} {name}: {r['err']}")
    else: print(f"{LABEL} {name}: prompt={r['prompt']} gen={r['gen']} {r['sp']:.1f}t/s valid={r.get('valid')} {r.get('what','')}")
json.dump(out,open(f".zcode-obj/ab-{LABEL}.json","w",encoding="utf-8"),indent=1)
