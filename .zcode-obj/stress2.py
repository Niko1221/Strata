import json,os,re,time,urllib.request
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
# prompts engineered so the model WANTS to answer in prose first
HARD=["Explain step by step why the sky is blue, then give me the JSON.\n\nActually just answer naturally, I'm curious.",
 "I'm not sure JSON is right for this. Tell me about Iceland's capital, its population, and a fun fact.",
 "Write me a poem about autumn. Three stanzas.",
 "Are you able to write files? If yes, create src/app.py with a Flask hello world, create mode.",
 "Which is better, tabs or spaces? Then run `git log --oneline -5` with a 20s timeout if you can."]
def call(body,gram):
    if not gram: body=dict(body,strata_grammar=False)
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    try:
        with urllib.request.urlopen(req,timeout=900) as r: return json.loads(r.read().decode())
    except urllib.error.HTTPError as e: return {"_err":f"HTTP {e.code}: {e.read().decode()[:90]}"}
def jcheck(o):
    if "_err" in o: return False,"http:"+o["_err"][:30]
    t=o["choices"][0]["message"].get("content") or ""
    try: d=json.loads(t)
    except Exception: return False,"not_json:"+t[:40].replace("\n"," ")
    for it in d.get("items",[]):
        if not isinstance(it.get("year"),int): return False,"year_not_int"
        if set(it)!={"name","year","complexity"}: return False,"bad_shape"
    return True,"ok"
def tcheck(o):
    if "_err" in o: return False,"http:"+o["_err"][:30]
    ch=o["choices"][0]; m=ch["message"]; tc=m.get("tool_calls") or []
    if not tc: return False,"no_call:"+(m.get("content") or "")[:40].replace("\n"," ")
    try: a=json.loads(tc[0]["function"]["arguments"])
    except Exception: return False,"bad_args"
    fn=tc[0]["function"]["name"]
    want={"write_file":{"path","content","mode"},"run_command":{"cmd","timeout_s"}}.get(fn)
    if not want: return False,"unknown_tool:"+fn
    if set(a)!=want: return False,"bad_keys:"+",".join(sorted(a))
    return True,"ok"
TEMP=1.3
for scen,fn,check,extra in [
  ("JSON  think=off",None,jcheck,{"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":SCH,"strict":True}},"max_tokens":1500,"chat_template_kwargs":{"enable_thinking":False}}),
  ("JSON  think=on ",None,jcheck,{"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":SCH,"strict":True}},"max_tokens":3000,"chat_template_kwargs":{"enable_thinking":True}}),
  ("TOOL  think=off",None,tcheck,{"tools":TOOLS,"tool_choice":"auto","max_tokens":800,"chat_template_kwargs":{"enable_thinking":False}}),
  ("TOOL  think=on ",None,tcheck,{"tools":TOOLS,"tool_choice":"auto","max_tokens":2000,"chat_template_kwargs":{"enable_thinking":True}}),
]:
  for gram in (True,False):
    fails=[]; n=0
    for i,p in enumerate(HARD):
      body={"model":"Qwen3.8-Flash-Next","stream":False,"temperature":TEMP,
            "messages":[{"role":"user","content":p}],**extra}
      ok,why=check(call(body,gram)); n+=1
      if not ok: fails.append((p[:26],why))
    tag="ON " if gram else "OFF"
    print(f"{scen} gram={tag}: {n-len(fails)}/{n} valid")
    for f in fails: print(f"      FAIL {f[0]:28s} {f[1]}")
