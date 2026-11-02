import json, urllib.request, sys
KEY="sk-ouzhiyingyyds666!"
PROMPT=open(".zcode-obj/prompt.txt",encoding="utf-8").read()
body={"model":"Qwen3.8-Flash-Next","stream":True,"max_tokens":int(sys.argv[1]),"temperature":0.6,
      "messages":[{"role":"user","content":PROMPT}]}
req=urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions",data=json.dumps(body).encode(),
    headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
buf=[]
with urllib.request.urlopen(req,timeout=1800) as r:
    for raw in r:
        l=raw.decode("utf-8","replace").strip()
        if not l.startswith("data: "): continue
        d=l[6:].strip()
        if d=="[DONE]": break
        try: o=json.loads(d)
        except: continue
        for ch in o.get("choices") or []:
            c=(ch.get("delta") or {}).get("content")
            if c: buf.append(c)
txt="".join(buf)
print("LEN",len(txt))
print(repr(txt[:600]))
print("...")
print(repr(txt[-400:]))
