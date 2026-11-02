"""Two DIFFERENT tools: does a param bleed from one call into the next?"""
import sys; sys.path.insert(0,'.')
from serve.frontend import OutputParser
TOOLS=[{"name":"write_file","parameters":{"type":"object","properties":{
  "path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"string"}},
  "required":["path","content","mode"]}},
 {"name":"run_command","parameters":{"type":"object","properties":{
  "cmd":{"type":"string"},"timeout_s":{"type":"integer"}},"required":["cmd","timeout_s"]}}]
C1=('<tool_call>\n<function=write_file>\n<parameter=path>\nautumn.md\n</parameter>\n'
    '<parameter=content>\nThe maples burn\n</parameter>\n'
    '<parameter=mode>\ncreate\n</parameter>\n</function>\n</tool_call>')
C2=('<tool_call>\n<function=run_command>\n<parameter=cmd>\ngit log --oneline -5\n</parameter>\n'
    '<parameter=timeout_s>\n20\n</parameter>\n</function>\n</tool_call>')
TEXT=C1+"\n"+C2
WANT=[("content","mode","path"),("cmd","timeout_s")]

def parse(chunks):
    p=OutputParser(thinking=False, tools=TOOLS)
    calls=[]
    for piece in chunks:
        for ev in p.feed(piece):
            if ev.kind=="tool_call": calls.append(ev.call)
    for ev in p.finish("stop"):
        if ev.kind=="tool_call": calls.append(ev.call)
    return calls

def keys(c): return tuple(sorted((c.arguments or {}).keys()))
ref=parse([TEXT])
print(f"whole: {len(ref)} calls keys={[keys(c) for c in ref]} want={WANT}")
bad=[]
for cut in range(1,len(TEXT)):
    calls=parse([TEXT[:cut],TEXT[cut:]])
    got=[keys(c) for c in calls]
    if len(calls)!=2 or got!=WANT: bad.append((cut,got))
print(f"2-way splits: {len(TEXT)-1} cuts, {len(bad)} bad")
for b in bad[:6]: print(f"   cut={b[0]} got={b[1]} ctx={TEXT[max(0,b[0]-20):b[0]+20]!r}")
# also 3-way around each cut to catch state that needs two boundaries
bad3=[]
for cut in range(1,len(TEXT)-1):
    for cut2 in range(cut+1,len(TEXT)):
        calls=parse([TEXT[:cut],TEXT[cut:cut2],TEXT[cut2:]])
        if [keys(c) for c in calls]!=WANT or len(calls)!=2:
            bad3.append((cut,cut2,[keys(c) for c in calls]))
print(f"3-way splits: {len(bad3)} bad")
for b in bad3[:6]: print(f"   cuts=({b[0]},{b[1]}) got={b[2]}")
