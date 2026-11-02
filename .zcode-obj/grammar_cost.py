import json,os,re,time,urllib.request,sys
KEY="sk-ouzhiyingyyds666!"; URL="http://127.0.0.1:8082/v1/chat/completions"
LOG=r"F:\Strata-dev\.zcode-obj\engine-gated-stderr.log"
SCH={"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{
  "name":{"type":"string"},"year":{"type":"integer"},"note":{"type":"string"}},
  "required":["name","year","note"],"additionalProperties":False}}},"required":["items"],"additionalProperties":False}
def run(grammar, mx, temp=0.0):
    P=f"List {mx//60} linear attention variants with year and a note. JSON only."
    body={"model":"Qwen3.8-Flash-Next","stream":True,"max_tokens":mx,"temperature":temp,
      "messages":[{"role":"user","content":P}],"chat_template_kwargs":{"enable_thinking":False}}
    if grammar: body["response_format"]={"type":"json_schema","json_schema":{"name":"x","schema":SCH,"strict":True}}
    since=os.path.getsize(LOG)
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    with urllib.request.urlopen(req,timeout=1800) as r:
        for raw in r: pass
    time.sleep(1.0)
    with open(LOG,encoding="utf-8",errors="replace") as f:
        f.seek(since); tail=f.read()
    m=re.search(r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)", tail)
    m2=re.search(r"drafts accepted (\d+) of (\d+)", tail)
    if m:
        n,ms,sp=int(m.group(1)),float(m.group(2))/1000.,float(m.group(3))
        acc=f"{100*int(m2.group(1))/int(m2.group(2)):.1f}%" if m2 else "-"
        print(f"  max={mx:5d} grammar={'Y' if grammar else 'N'}: {n:5d} tok {ms:6.1f}s {sp:6.1f} tok/s accept={acc}")
        return sp
    print(f"  max={mx} grammar={grammar}: no telemetry"); return 0
print("== short (400)")
a=run(True,400); b=run(False,400)
print("== medium (1200)")
c=run(True,1200); d=run(False,1200)
print("== short again (for noise)")
e=run(True,400); f=run(False,400)
print(f"\ngrammar/plain ratio: short {a/b:.2f}x  medium {c/d:.2f}x  short2 {e/f:.2f}x")
