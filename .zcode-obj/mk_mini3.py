import json
g = ('root ::= "x\\n" answer\n'
     'answer ::= ws ( fenced | answer-root )\n'
     'ws ::= " " | "\\t" | "\\n"\n'
     'fenced ::= "```json" ws answer-root ws "```"\n'
     'answer-root ::= "{" answer-space "}"\n'
     'answer-space ::= | " " | "\\n"{1,2} [ \\t]{0,20}\n')
open("mini3.txt", "w", encoding="utf-8", newline="\n").write(g)
pieces = [{"p": "x", "c": 0, "e": 0}, {"p": "\n", "c": 0, "e": 0}, {"p": "z", "c": 0, "e": 0},
          {"p": "{", "c": 0, "e": 0}, {"p": " ", "c": 0, "e": 0}, {"p": "`", "c": 0, "e": 0},
          {"p": "```json", "c": 0, "e": 0}, {"p": "}", "c": 0, "e": 0},
          {"p": "<|eot|>", "c": 1, "e": 1}]
open("mini3_pieces.json", "w", encoding="utf-8").write(json.dumps(pieces))
print("ok")
