import json
import subprocess
import sys
sys.path.insert(0, "F:/Strata-dev/serve")
sys.path.insert(0, "F:/Strata-dev")
from serve import grammar_tools

schema = {"type": "object", "properties": {
    "name": {"type": "string"}, "age": {"type": "integer"}, "city": {"type": "string"}},
    "required": ["name", "age", "city"]}
conv = subprocess.run(["F:/Strata-dev/build/grammar_schema.exe"], input=json.dumps(schema),
                      capture_output=True, text=True)
assert conv.returncode == 0, conv.stderr
text = grammar_tools.with_think_prologue(conv.stdout)
open("person_pro.txt", "w", encoding="utf-8", newline="\n").write(text)
print(text[:400])
