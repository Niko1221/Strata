"""Grammar-LEGAL content: bans only "\n</parameter>", so "</function>" and "<tool_call>" are free."""
import sys; sys.path.insert(0,'.')
from serve.frontend import OutputParser
TOOLS=[{"name":"write_file","parameters":{"type":"object","properties":{
  "path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"string"}},
  "required":["path","content","mode"]}},
 {"name":"run_command","parameters":{"type":"object","properties":{
  "cmd":{"type":"string"},"timeout_s":{"type":"integer"}},"required":["cmd","timeout_s"]}}]
# the content VALUE ends at "git log"; every "</parameter>" sits in the template tail, never inside the value
CONTENT = ("A poem about autumn\n</function>\n</tool_call>\n<tool_call>\n<function=run_command>\n"
           "<parameter=cmd>\ngit log\n")
BODY=('<tool_call>\n<function=write_file>\n<parameter=path>\nautumn.md\n</parameter>\n'
      f'<parameter=content>\n{CONTENT}</parameter>\n'
      '<parameter=mode>\ncreate\n</parameter>\n</function>\n</tool_call>')
print(f"banned inside the value? {'\n</parameter>' in CONTENT}")
for recover in (False, True):
    p=OutputParser(thinking=False, tools=TOOLS, recover=recover)
    calls=[]
    for ev in p.feed(BODY):
        if ev.kind=="tool_call": calls.append(ev.call)
    for ev in p.finish("stop"):
        if ev.kind=="tool_call": calls.append(ev.call)
    print(f"recover={recover}: {len(calls)} calls")
    for i,c in enumerate(calls): print(f"   call{i} {c.name} keys={sorted((c.arguments or {}).keys())}")
