import json,urllib.request
KEY="sk-ouzhiyingyyds666!"; URL="http://127.0.0.1:8082/v1/chat/completions"
SCH={"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{
  "name":{"type":"string"},"year":{"type":"integer"},"complexity":{"type":"string"}},
  "required":["name","year","complexity"],"additionalProperties":False}}},"required":["items"],"additionalProperties":False}
P="Write me a poem about autumn. Three stanzas."
for gram in (True,False):
  body={"model":"Qwen3.8-Flash-Next","stream":False,"max_tokens":3000,"temperature":1.3,
    "messages":[{"role":"user","content":P}],"chat_template_kwargs":{"enable_thinking":True},
    "response_format":{"type":"json_schema","json_schema":{"name":"x","schema":SCH,"strict":True}}}
  if not gram: body["strata_grammar"]=False
  req=urllib.request.Request(URL,data=json.dumps(body).encode(),
      headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
  with urllib.request.urlopen(req,timeout=900) as r: o=json.loads(r.read().decode())
  m=o["choices"][0]["message"]
  print(f"===== grammar={'ON' if gram else 'OFF'} finish={o['choices'][0]['finish_reason']}")
  print(f"  reasoning_content: {len(m.get('reasoning_content') or '')} chars -> {repr((m.get('reasoning_content') or '')[:90])}")
  print(f"  content:           {len(m.get('content') or '')} chars -> {repr((m.get('content') or '')[:120])}")
  print(f"  content TAIL:      {repr((m.get('content') or '')[-60:])}")
