import json
pieces = [{"p": "y", "c": 0, "e": 0}, {"p": "x", "c": 0, "e": 0}, {"p": "\n", "c": 0, "e": 0},
          {"p": "z", "c": 0, "e": 0}, {"p": " ", "c": 0, "e": 0}, {"p": "<|eot|>", "c": 1, "e": 1}]
open("mini2_pieces.json", "w", encoding="utf-8").write(json.dumps(pieces))
print("ok")
