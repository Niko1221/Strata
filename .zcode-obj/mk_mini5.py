import json
import subprocess
import sys
sys.path.insert(0, "F:/Strata-dev")
from serve import grammar_tools

schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
conv = subprocess.run(["F:/Strata-dev/build/grammar_schema.exe"], input=json.dumps(schema),
                      capture_output=True, text=True)
assert conv.returncode == 0, conv.stderr
open("mini5.txt", "w", encoding="utf-8", newline="\n").write(grammar_tools.with_think_prologue(conv.stdout))
pieces = [{"p": "</think>", "c": 0, "e": 0}, {"p": "\n", "c": 0, "e": 0},
          {"p": " ", "c": 0, "e": 0}, {"p": "{", "c": 0, "e": 0},
          {"p": '"', "c": 0, "e": 0}, {"p": "name", "c": 0, "e": 0},
          {"p": ":", "c": 0, "e": 0}, {"p": "y", "c": 0, "e": 0},
          {"p": "}", "c": 0, "e": 0}, {"p": "```json", "c": 0, "e": 0},
          {"p": "```", "c": 0, "e": 0}, {"p": "<|eog|>", "c": 1, "e": 1}]
json.dump(pieces, open("mini5_pieces.json", "w", encoding="utf-8"), ensure_ascii=True)
print("bare: 0 1 3 4 5 6 2 4 7 4 8 11")
print("fence: 0 1 9 3 4 5 6 2 4 7 4 8 10 11")
