import json,urllib.request,statistics,time
KEY="sk-ouzhiyingyyds666!"; URL="http://127.0.0.1:8082/v1/chat/completions"
TOOLS=[{"type":"function","function":{"name":"write_file","description":"Write a file",
  "parameters":{"type":"object","properties":{
    "path":{"type":"string"},"content":{"type":"string"},
    "mode":{"type":"string","enum":["create","append","overwrite"]}},
   "required":["path","content","mode"],"additionalProperties":False}}},
 {"type":"function","function":{"name":"run_command","description":"Run a shell command",
  "parameters":{"type":"object","properties":{"cmd":{"type":"string"},"timeout_s":{"type":"integer"}},
   "required":["cmd","timeout_s"],"additionalProperties":False}}}]
SCH={"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{
  "name":{"type":"string"},"year":{"type":"integer"},"complexity":{"type":"string"}},
  "required":["name","year","complexity"],"additionalProperties":False}}},"required":["items"],"additionalProperties":False}
# prompts that fight the required shape
HARD=["Explain step by step why the sky is blue, then give me the JSON. Actually just answer naturally.",
 "I'm not sure JSON is right for this. Tell me about Iceland's capital, its population, and a fun fact.",
 "Write me a poem about autumn. Three stanzas.",
 "Which is better, tabs or spaces? Then run `git log --oneline -5` with a 20s timeout if you can.",
 "Are you able to write files? If yes, create src/app.py with a Flask hello world, create mode."]
OKT={"write_file":{"path","content","mode"},"run_command":{"cmd","timeout_s"}}

def call(body,gram):
    if not gram: body=dict(body,strata_grammar=False)
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    t0=time.time()
    try:
        with urllib.request.urlopen(req,timeout=1200) as r:
            o=json.loads(r.read().decode()); return o,time.time()-t0,None
    except urllib.error.HTTPError as e:
        return None,time.time()-t0,f"HTTP{e.code}"

def jcase(p,gram,think,mx):
    o,dt,err=call({"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":mx,"temperature":1.0,
      "messages":[{"role":"user","content":p}],"chat_template_kwargs":{"enable_thinking":think},
      "response_format":{"type":"json_schema","json_schema":{"name":"x","schema":SCH,"strict":True}}},gram)
    if err: return err,None
    c=o["choices"][0]["message"].get("content") or ""
    try:
        d=json.loads(c)
        if not isinstance(d.get("items"),list): return "wrong_shape",None
        for it in d["items"]:
            if set(it)!={"name","year","complexity"} or not isinstance(it["year"],int): return "bad_field",None
        return "ok",len(c)
    except Exception:
        return ("fenced" if c.lstrip().startswith("`") else "not_json"),None

def tcase(p,gram,think,mx):
    o,dt,err=call({"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":mx,"temperature":1.0,
      "messages":[{"role":"user","content":p}],"chat_template_kwargs":{"enable_thinking":think},
      "tools":TOOLS,"tool_choice":"required"},gram)
    if err: return err,None
    ch=o["choices"][0]; m=ch["message"]; tc=m.get("tool_calls") or []
    if not tc: return "no_call",None
    fn=tc[0]["function"]["name"]
    if fn not in OKT: return f"unknown_tool",None
    try: a=json.loads(tc[0]["function"]["arguments"])
    except Exception: return "bad_args",None
    if set(a)!=OKT[fn]: return "bad_keys",None
    return "ok",len(tc[0]["function"]["arguments"])

N=6
for name,fn,mx,extra in [("JSON  think=off",jcase,1500,False),("JSON  think=on ",jcase,6000,True),
                          ("TOOL  think=off",tcase,900,False),("TOOL  think=on ",tcase,6000,True)]:
  print(f"--- {name}")
  for gram in (True,False):
    tally={}; sizes=[]
    for i in range(N):
      for p in HARD:
        r,sz=fn(p,gram,extra,mx); tally[r]=tally.get(r,0)+1
        if sz: sizes.append(sz)
    tot=sum(tally.values()); good=tally.get("ok",0)
    bad={k:v for k,v in tally.items() if k!="ok"}
    tag="ON " if gram else "OFF"
    print(f"    gram={tag}: {good}/{tot} ok"+(f"  FAILURES {bad}" if bad else "  (no failures)"))
