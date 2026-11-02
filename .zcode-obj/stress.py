import json,os,re,time,urllib.request
KEY="sk-ouzhiyingyyds666!"; URL="http://127.0.0.1:8082/v1/chat/completions"
LOG=r"F:\Strata-dev\.zcode-obj\engine-gated-stderr.log"
TOOLS=[{"type":"function","function":{"name":"write_file","description":"Write a file",
  "parameters":{"type":"object","properties":{
    "path":{"type":"string","description":"file path"},
    "content":{"type":"string","description":"file content"},
    "mode":{"type":"string","enum":["create","append","overwrite"]}},
   "required":["path","content","mode"],"additionalProperties":False}}},
 {"type":"function","function":{"name":"run_command","description":"Run a shell command",
  "parameters":{"type":"object","properties":{"cmd":{"type":"string"},"timeout_s":{"type":"integer"}},
   "required":["cmd","timeout_s"],"additionalProperties":False}}}]
SCH={"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{
  "name":{"type":"string"},"year":{"type":"integer"},"complexity":{"type":"string"},"note":{"type":"string"}},
  "required":["name","year","complexity","note"],"additionalProperties":False}}},"required":["items"],"additionalProperties":False}
# prompts that invite prose, forcing the model to choose between chatting and emitting the shape
PX=["What's the capital of Iceland?","Tell me a short joke.","Explain gravity in one sentence.",
    "Who wrote Pride and Prejudice?","Summarize what a transformer is."]
PT=["Create src/main.py with a hello world script, create mode.",
    "Run pytest with a 300 second timeout.","Write NOTES.md summarizing the project.",
    "Append 'v0.2.9 shipped' to CHANGELOG.md.","Run git status with a 30 second timeout."]

def call(body,gram):
    if not gram: body=dict(body, strata_grammar=False)
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    try:
        with urllib.request.urlopen(req,timeout=900) as r: return json.loads(r.read().decode())
    except urllib.error.HTTPError as e: return {"_err":f"HTTP {e.code}: {e.read().decode()[:100]}"}

def check_tool(o,tools_ok):
    if "_err" in o: return False,"http",o["_err"]
    ch=o["choices"][0]; msg=ch["message"]; tc=msg.get("tool_calls") or []
    if not tc:
        return False,"no_call",(msg.get("content") or "")[:60]
    try: args=json.loads(tc[0]["function"]["arguments"])
    except Exception as e: return False,"bad_args",str(e)[:60]
    fn=tc[0]["function"]["name"]
    if fn not in tools_ok: return False,"wrong_tool",fn
    spec=tools_ok[fn]
    if set(args)!=set(spec): return False,"bad_keys",str(sorted(args))
    return True,"ok",""

def check_json(o):
    if "_err" in o: return False,"http",o["_err"]
    txt=o["choices"][0]["message"].get("content") or ""
    try: d=json.loads(txt)
    except Exception: return False,"not_json",txt[:60].replace("\n"," ")
    if "items" not in d: return False,"no_items",str(list(d))[:60]
    for it in d["items"]:
        if set(it)!={"name","year","complexity","note"}: return False,"bad_shape",str(sorted(it))
        if not isinstance(it["year"],int): return False,"year_not_int",repr(it["year"])
    return True,"ok",f"items={len(d['items'])}"

TOOLS_OK={"write_file":{"path","content","mode"},"run_command":{"cmd","timeout_s"}}
for temp in (0.8,):
  for gram in (True,False):
    tf=jf=tn=jn=0; examples=[]
    for i,p in enumerate(PX):
      o=call({"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":1200,"temperature":temp,
        "messages":[{"role":"user","content":p}],"chat_template_kwargs":{"enable_thinking":False},
        "response_format":{"type":"json_schema","json_schema":{"name":"x","schema":SCH,"strict":True}}},gram)
      ok,why,det=check_json(o); jn+=1
      if not ok: jf+=1; examples.append(("json",p[:30],why,det))
    for i,p in enumerate(PT):
      o=call({"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":800,"temperature":temp,
        "messages":[{"role":"user","content":p}],"chat_template_kwargs":{"enable_thinking":False},
        "tools":TOOLS,"tool_choice":"auto"},gram)
      ok,why,det=check_tool(o,TOOLS_OK); tn+=1
      if not ok: tf+=1; examples.append(("tool",p[:30],why,det))
    tag="grammar ON " if gram else "grammar OFF"
    print(f"{tag} temp={temp}: JSON {jn-jf}/{jn} valid, TOOL {tn-tf}/{tn} valid")
    for e in examples[:6]: print(f"    FAIL {e[0]:5s} {e[1]:32s} {e[2]:10s} {e[3]}")
