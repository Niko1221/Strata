import json,os,time,urllib.request
KEY="sk-ouzhiyingyyds666!"; URL="http://127.0.0.1:8082/v1/chat/completions"
LOG=r"F:\Strata-dev\.zcode-obj\engine-gated-stderr.log"
GR = 'root ::= "ZZZ-CONSTRAINED-OK"'
since=os.path.getsize(LOG)
body={"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":60,"temperature":0.7,
  "messages":[{"role":"user","content":"Say hello in English, nothing else."}],
  "chat_template_kwargs":{"enable_thinking":False},
  "grammar":GR}
req=urllib.request.Request(URL,data=json.dumps(body).encode(),
    headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
try:
    with urllib.request.urlopen(req,timeout=600) as r:
        o=json.loads(r.read().decode())
    txt=o["choices"][0]["message"].get("content")
    print("REPLY:",repr(txt))
    print("VERDICT:", "GRAMMAR ACTIVE" if txt and "ZZZ-CONSTRAINED-OK" in txt else "GRAMMAR NOT APPLIED")
except urllib.error.HTTPError as e:
    print("HTTP",e.code,e.read().decode()[:300])
time.sleep(1.5)
with open(LOG,encoding="utf-8",errors="replace") as f:
    f.seek(since); t=f.read()
for ln in t.splitlines():
    if "gdbg" in ln or "grammar" in ln.lower(): print("  ENG:",ln[:160])
