"""stream_tools=True (the server's config) with two DIFFERENT tools: is a call's argument object mutated?"""
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
WANT=[("write_file",("content","mode","path")),("run_command",("cmd","timeout_s"))]
def parse(chunks, **kw):
    p=OutputParser(thinking=False, tools=TOOLS, **kw)
    calls=[]
    for piece in chunks:
        for ev in p.feed(piece):
            if ev.kind=="tool_call": calls.append(ev.call)
    for ev in p.finish("stop"):
        if ev.kind=="tool_call": calls.append(ev.call)
    return calls
def sig(c): return (c.name, tuple(sorted((c.arguments or {}).keys())))
for kw in ({"stream_tools":False},{"stream_tools":True},{"stream_tools":True,"recover":True}):
    calls=parse([TEXT], **kw)
    got=[sig(c) for c in calls]
    print(f"{str(kw):48s} {len(calls)} calls {got} {'' if got==WANT else '<<< WRONG'}")
# now the same under every split, stream_tools=True + recover (server config)
bad=[]
for cut in range(1,len(TEXT)):
    calls=parse([TEXT[:cut],TEXT[cut:]], stream_tools=True, recover=True)
    if [sig(c) for c in calls]!=WANT: bad.append((cut,[sig(c) for c in calls]))
print(f"stream_tools+recover, 2-way splits: {len(TEXT)-1} cuts, {len(bad)} bad")
for b in bad[:6]: print(f"  cut={b[0]} got={b[1]} ctx={TEXT[max(0,b[0]-18):b[0]+18]!r}")
