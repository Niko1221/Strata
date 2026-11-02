"""Can a value hold </function> or </tool_call> WITHOUT the closer, and still break the parse?"""
import sys; sys.path.insert(0,'.')
from serve.frontend import OutputParser
TOOLS=[{"name":"write_file","parameters":{"type":"object","properties":{
  "path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"string"}},
  "required":["path","content","mode"]}}]
WANT=[("write_file",("content","mode","path"))]
def parse(body):
    out=[]
    for st in (False,True):
        p=OutputParser(thinking=False, tools=TOOLS, stream_tools=st, recover=True)
        calls=[]
        for ev in p.feed(body):
            if ev.kind=="tool_call": calls.append(ev.call)
        for ev in p.finish("stop"):
            if ev.kind=="tool_call": calls.append(ev.call)
        out.append([(c.name,tuple(sorted((c.arguments or {}).keys()))) for c in calls])
    return out
def build(val):
    return ('<tool_call>\n<function=write_file>\n<parameter=path>\na.md\n</parameter>\n'
            f'<parameter=content>\n{val}</parameter>\n'
            '<parameter=mode>\ncreate\n</parameter>\n</function>\n</tool_call>')
CASES=["doc: close a function with </function>",
       "doc: close the wrapper with </tool_call>",
       "doc: </function> then </tool_call>",
       "doc: <tool_call> starts one",
       "doc: </function>\n<function=run_command>",
      ]
for v in CASES:
    r=parse(build(v))
    ok = all(x==WANT for x in r)
    print(f"{'ok ' if ok else 'BAD'} val={v!r:46s} plain={r[0]} stream={r[1]}")
