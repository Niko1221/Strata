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
  "parameters":{"type":"object","properties":{
    "cmd":{"type":"string"},"timeout_s":{"type":"integer"}},
   "required":["cmd","timeout_s"],"additionalProperties":False}}}]
PROMPTS=[
 "Create a file at src/main.py containing a hello world script. Use create mode.",
 "Run the test suite with pytest and give it 300 seconds.",
 "Write NOTES.md with a one-line summary of the project.",
 "Append a changelog entry to CHANGELOG.md saying v0.2.9 shipped.",
 "Run `git status` with a 30 second timeout.",
]
def run(prompt, tools_on, temp=0.0, mx=400):
    body={"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":mx,"temperature":temp,
      "messages":[{"role":"user","content":prompt}],"chat_template_kwargs":{"enable_thinking":False}}
    if tools_on: body["tools"]=TOOLS; body["tool_choice"]="auto"
    since=os.path.getsize(LOG)
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    t0=time.time()
    try:
        with urllib.request.urlopen(req,timeout=900) as r: o=json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode()[:150]}"
    dt=time.time()-t0; time.sleep(0.7)
    with open(LOG,encoding="utf-8",errors="replace") as f:
        f.seek(since); t=f.read()
    m=re.search(r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)", t)
    msg=o["choices"][0]["message"]
    tc=msg.get("tool_calls") or []
    return dict(sp=float(m.group(3)) if m else 0, tc=tc, content=(msg.get("content") or ""),
                finish=o["choices"][0].get("finish_reason"), dt=dt), None
print("=== TOOL CALLS (grammar-constrained tool path)")
for p in PROMPTS:
    r,err=run(p,True)
    if err: print(f"  ERR {err}"); continue
    if r["tc"]:
        fn=r["tc"][0]["function"]
        try: args=json.loads(fn["arguments"]); shape=f"args OK keys={sorted(args)}"
        except Exception as e: shape=f"ARGS BROKEN ({e})"
        print(f"  {r['sp']:6.1f} t/s  tool={fn['name']}  {shape}  finish={r['finish']}")
    else:
        print(f"  {r['sp']:6.1f} t/s  NO TOOL CALL - text: {r['content'][:70]!r} finish={r['finish']}")
print("\n=== SAME PROMPTS, no tools (plain chat speed)")
for p in PROMPTS[:3]:
    r,err=run(p,False)
    if err: print(f"  ERR {err}"); continue
    print(f"  {r['sp']:6.1f} t/s  finish={r['finish']}")
