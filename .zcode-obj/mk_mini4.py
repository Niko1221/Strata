import json
import subprocess
import sys
sys.path.insert(0, "F:/Strata-dev")
from serve import grammar_tools

schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
conv = subprocess.run(["F:/Strata-dev/build/grammar_schema.exe"], input=json.dumps(schema),
                      capture_output=True, text=True)
assert conv.returncode == 0, conv.stderr
open("mini4.txt", "w", encoding="utf-8", newline="\n").write(grammar_tools.with_think_prologue(conv.stdout))
pieces = [{"p": "q", "c": 0, "e": 0}, {"p": "</think>", "c": 0, "e": 0},
          {"p": "\n", "c": 0, "e": 0}, {"p": " ", "c": 0, "e": 0},
          {"p": "{", "c": 0, "e": 0}, {"p": '"', "c": 0, "e": 0},
          {"p": "n", "c": 0, "e": 0}, {"p": "}", "c": 0, "e": 0},
          {"p": ":", "c": 0, "e": 0}, {"p": "`", "c": 0, "e": 0},
          {"p": "```json", "c": 0, "e": 0}, {"p": "<|eot|>", "c": 1, "e": 1}]
open("mini4_pieces.json", "w", encoding="utf-8").write(json.dumps(pieces))
print("ok")
