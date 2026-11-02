import sys; sys.path.insert(0,'.')
from serve.frontend import parse_tool_call
SCHEMA={"name":"write_file","parameters":{"type":"object","properties":{
  "path":{"type":"string"},"content":{"type":"string"},
  "mode":{"type":"string","enum":["create","append","overwrite"]}},
  "required":["path","content","mode"]}}
def call(path,content,mode):
    return (f'<function=write_file>\n<parameter=path>\n{path}\n</parameter>\n'
            f'<parameter=content>\n{content}\n</parameter>\n'
            f'<parameter=mode>\n{mode}\n</parameter>\n</function>')
cases={
 "normal": call("a.md","hello","create"),
 "content w/ stanza blanks": call("a.md","The maples burn\n\nThe light lies long\n\nA mist","create"),
 "content w/ tag-like line": call("a.md","x\n</parameter>\nline","create"),
 "content w/ function tag": call("a.md","a\n</function>\nb","create"),
 "content w/ tool_call tag": call("a.md","a\n</tool_call>\nb","create"),
 "content is one long line": call("autumn.md","# The Season's Turning\n\nThe summer loosens, slow and sweet,\nIts green withdrawn to gold and rust;","create"),
}
for name,body in cases.items():
    notes=[]
    try: c=parse_tool_call(body, SCHEMA, notes)
    except Exception as e:
        print(f"{name:28s} EXC {str(e)[:60]}"); continue
    keys=sorted(c.arguments or {})
    bad = "" if keys==["content","mode","path"] else "   <<< DROPPED"
    print(f"{name:28s} keys={keys} notes={notes}{bad}")
