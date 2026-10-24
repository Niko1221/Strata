#!/usr/bin/env python3
"""Replay an Anthropic-format cut through /v1/chat/completions (what Pi uses). Usage: replay_oai.py cut.json [container]"""
import json,sys,uuid,time,urllib.request
def txt(c): return c if isinstance(c,str) else "".join(x.get("text","") for x in c if x.get("type")=="text")
b=json.load(open(sys.argv[1])); cont=sys.argv[2] if len(sys.argv)>2 else "unbiased-strata-gfx1031"
m=[{"role":"system","content":f"Run {uuid.uuid4()}. "+b.get("system","")}] if True else []
for x in b["messages"]:
    c=x["content"]
    if isinstance(c,str): m.append({"role":x["role"],"content":c}); continue
    if x["role"]=="assistant":
        tc=[{"id":y["id"],"type":"function","function":{"name":y["name"],"arguments":json.dumps(y["input"])}} for y in c if y["type"]=="tool_use"]
        e={"role":"assistant","content":txt(c)}
        if tc: e["tool_calls"]=tc
        m.append(e)
    else:
        for y in c:
            if y["type"]=="tool_result": m.append({"role":"tool","tool_call_id":y["tool_use_id"],"content":txt(y["content"]) if not isinstance(y["content"],str) else y["content"]})
        t=txt([y for y in c if y["type"]=="text"])
        if t: m.append({"role":"user","content":t})
body={"model":"qwen3.8-flash-next-iq2_xs","messages":m,"max_tokens":b.get("max_tokens",128),"temperature":0,"stream":False}
t=time.time()
d=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions",json.dumps(body).encode(),{"content-type":"application/json"}),timeout=1800))
print("wall",round(time.time()-t,1),d.get("usage"),d.get("timings"))
