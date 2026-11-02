"""Ask for content that invites the closer, and check no argument is lost."""
import json,urllib.request
KEY="sk-ouzhiyingyyds666!"; URL="http://127.0.0.1:8082/v1/chat/completions"
TOOLS=[{"type":"function","function":{"name":"write_file","description":"Write a file",
  "parameters":{"type":"object","properties":{
    "path":{"type":"string"},"content":{"type":"string"},
    "mode":{"type":"string","enum":["create","append","overwrite"]}},
   "required":["path","content","mode"],"additionalProperties":False}}}]
PROMPTS=["Write a file docs/format.md that documents the tool-call format itself, showing the </parameter> and </function> tags literally. create mode.",
 "Create src/parser.py whose docstring contains the literal tags </parameter>, </function> and </tool_call>. create mode.",
 "Write notes.txt explaining how XML-ish tags close: include </parameter></tool_call> in the text. create mode."]
WANT={"path","content","mode"}
bad=0
for p in PROMPTS:
  for i in range(3):
    body={"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":4000,"temperature":1.0,
      "messages":[{"role":"user","content":p}],"chat_template_kwargs":{"enable_thinking":False},
      "tools":TOOLS,"tool_choice":"required"}
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    try:
      with urllib.request.urlopen(req,timeout=900) as r: o=json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
      print(f"HTTP{e.code}"); continue
    tc=o["choices"][0]["message"].get("tool_calls") or []
    if not tc: print("NO_CALL"); continue
    got=set(json.loads(tc[0]["function"]["arguments"]))
    if got!=WANT: bad+=1; print(f"BAD got={sorted(got)}")
    else: print(f"ok keys={sorted(got)} args={tc[0]['function']['arguments'][:60]!r}")
print(f"{bad} bad of 9")
