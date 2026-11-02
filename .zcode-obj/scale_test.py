import json,os,re,time,urllib.request
KEY="sk-ouzhiyingyyds666!"; URL="http://127.0.0.1:8082/v1/chat/completions"
LOG=r"F:\Strata-dev\.zcode-obj\engine-gated-stderr.log"
SCH={"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{
  "i":{"type":"integer"},"name":{"type":"string"},"note":{"type":"string"}},
  "required":["i","name","note"],"additionalProperties":False}}},"required":["items"],"additionalProperties":False}
def run(grammar, mx):
    P=f"Output exactly {mx//12} entries: linear attention variants with a note each. JSON only."
    body={"model":"Qwen3.8-Flash-Next","stream":True,"max_tokens":mx,"temperature":0.0,
      "messages":[{"role":"user","content":P}],"chat_template_kwargs":{"enable_thinking":False}}
    if grammar: body["response_format"]={"type":"json_schema","json_schema":{"name":"x","schema":SCH,"strict":True}}
    since=os.path.getsize(LOG)
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    with urllib.request.urlopen(req,timeout=1800) as r:
        for raw in r: pass
    time.sleep(0.9)
    with open(LOG,encoding="utf-8",errors="replace") as f:
        f.seek(since); t=f.read()
    m=re.search(r"prompt (\d+) tokens.*?(\d+) generated in (\d+) ms", t)
    if not m: return None
    pre_b=float(m.group(1)); n=int(m.group(2)); ms=float(m.group(3))/1000.
    return pre_b, n, ms, n/ms
for mx in (600, 2400, 4000):
    g=run(True,mx); p=run(False,mx)
    if g and p:
        print(f"max={mx:5d}  grammar {g[1]:5d} tok {g[2]:5.1f}s {g[3]:6.1f} t/s | plain {p[1]:5d} tok {p[2]:5.1f}s {p[3]:6.1f} t/s "
              f"| {100*(g[3]-p[3])/p[3]:+5.1f}%  (gap {g[2]-p[2]:+.2f}s)")
