"""Deterministic chunk-boundary fuzz: split a well-formed call at every byte and compare."""
import json, sys; sys.path.insert(0,'.')
from serve.frontend import OutputParser
TOOLS=[{"name":"write_file","parameters":{"type":"object","properties":{
  "path":{"type":"string"},"content":{"type":"string"},
  "mode":{"type":"string","enum":["create","append","overwrite"]}},
  "required":["path","content","mode"]}}]
ONE=('<tool_call>\n<function=write_file>\n<parameter=path>\nautumn.md\n</parameter>\n'
     '<parameter=content>\n# The Season\'s Turning\n\nThe summer loosens, slow and sweet,\n</parameter>\n'
     '<parameter=mode>\ncreate\n</parameter>\n</function>\n</tool_call>')
TWO=ONE+"\n"+ONE.replace("autumn.md","b.md").replace("# The Season","# Second")

def parse(text, chunks):
    p=OutputParser(thinking=False, tools=TOOLS)
    calls=[]
    for piece in chunks:
        for ev in p.feed(piece):
            if ev.kind=="tool_call": calls.append(ev.call)
    for ev in p.finish("stop"):
        if ev.kind=="tool_call": calls.append(ev.call)
    return calls

def keys(c): return tuple(sorted((c.arguments or {}).keys()))

for name,text,want in [("one call",ONE,1),("two calls",TWO,2)]:
    ref=parse(text,[text])
    print(f"{name}: whole-text -> {len(ref)} calls, keys={[keys(c) for c in ref]}")
    bad=[]
    for cut in range(1,len(text)):
        calls=parse(text,[text[:cut],text[cut:]])
        if len(calls)!=want or any(keys(c)!=("content","mode","path") for c in calls):
            bad.append((cut,[keys(c) for c in calls],len(calls)))
    print(f"  split at every byte ({len(text)} cuts): {len(bad)} bad")
    for b in bad[:5]: print(f"    cut={b[0]} calls={b[2]} keys={b[1]}  ctx={text[max(0,b[0]-25):b[0]+25]!r}")
