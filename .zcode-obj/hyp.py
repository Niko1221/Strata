import sys; sys.path.insert(0,'.')
from serve.frontend import OutputParser
TOOLS=[{"name":"write_file","parameters":{"type":"object","properties":{
  "path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"string"}},
  "required":["path","content","mode"]}},
 {"name":"run_command","parameters":{"type":"object","properties":{
  "cmd":{"type":"string"},"timeout_s":{"type":"integer"}},"required":["cmd","timeout_s"]}}]
# content VALUE carries a call-shaped sequence (grammar only bans "\n</parameter>")
BODY=('<tool_call>\n<function=write_file>\n<parameter=path>\nautumn.md\n</parameter>\n'
 '<parameter=content>\nA poem\n</function>\n</tool_call>\n<tool_call>\n<function=run_command>\n'
 '<parameter=cmd>\ngit log\n</parameter>\n<parameter=timeout_s>\n20\n</parameter>\n</function>\n</tool_call>\n'
 '</parameter>\n<parameter=mode>\ncreate\n</parameter>\n</function>\n</tool_call>')
p=OutputParser(thinking=False, tools=TOOLS, recover=True)   # recover is ON in the server since 0.2.8
calls=[]
for ev in p.feed(BODY):
    if ev.kind=="tool_call": calls.append(ev.call)
for ev in p.finish("stop"):
    if ev.kind=="tool_call": calls.append(ev.call)
print(f"{len(calls)} calls")
for i,c in enumerate(calls): print(f"  call{i} {c.name} keys={sorted((c.arguments or {}).keys())}")
