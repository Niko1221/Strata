"""param_end returns at ANY </parameter> followed by PARAM_START/FUNC_END/CALL_END.
The grammar only bans "\n</parameter>" - so a close WITHOUT a leading newline is legal.
"""
import sys; sys.path.insert(0,'.')
from serve.frontend import OutputParser, parse_tool_call
TOOLS=[{"name":"write_file","parameters":{"type":"object","properties":{
  "path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"string"}},
  "required":["path","content","mode"]}}]
legal = lambda s: "\n</parameter>" not in s
def build(val):
    return ('<tool_call>\n<function=write_file>\n<parameter=path>\na.md\n</parameter>\n'
            f'<parameter=content>\n{val}</parameter>\n'
            '<parameter=mode>\ncreate\n</parameter>\n</function>\n</tool_call>')
WANT=[("content","mode","path")]
for val in ["A poem</parameter></function>more",
            "A poem</parameter>",
            "A poem </parameter>\n</function>",
            "A poem</parameter></tool_call> tail",
            "OK</parameter><parameter=mode>"]:
    body=build(val)
    got=[]
    for st in (False,True):
        p=OutputParser(thinking=False, tools=TOOLS, stream_tools=st, recover=True)
        calls=[]
        for ev in p.feed(body):
            if ev.kind=="tool_call": calls.append(ev.call)
        for ev in p.finish("stop"):
            if ev.kind=="tool_call": calls.append(ev.call)
        got.append([(c.name,tuple(sorted((c.arguments or {}).keys()))) for c in calls])
    wrong = any([s for s in got] != [(("write_file",("content","mode","path")))] for s in got)
    print(f"legal={legal(val)!s:5s} val={val[:34]!r:38s} plain={got[0]} stream={got[1]}{'   <<< WRONG' if wrong else ''}")
