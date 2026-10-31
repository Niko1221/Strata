import json, subprocess, sys
sys.path.insert(0, "F:/Strata-dev")
from serve import grammar_tools as T
from serve import grammar as grammar_api

TOOLS = [{"name": "get_weather", "description": "d",
          "parameters": {"type": "object", "additionalProperties": False, "required": ["city", "unit"],
                         "properties": {"city": {"type": "string"},
                                        "unit": {"type": "string", "enum": ["c", "f"]},
                                        "urgent": {"type": "boolean"}}}},
         {"name": "send_sms", "description": "d",
          "parameters": {"type": "object", "additionalProperties": False, "required": ["to", "body"],
                         "properties": {"to": {"type": "string"}, "body": {"type": "string"},
                                        "urgent": {"type": "boolean"}}}}]
CONV = "F:/Strata-dev/build/grammar_schema.exe"
PROBE = "F:/Strata-dev/.zcode-obj/parse_probe.exe"


def convert(schema):
    p = subprocess.run([CONV], input=json.dumps(schema).encode(), capture_output=True)
    if p.returncode != 0:
        raise SystemExit("converter failed: " + p.stderr.decode()[:200])
    return p.stdout.decode()


CASES = {}
for kind, name in (("auto", None), ("required", None), ("named", "get_weather")):
    for thinking in (True, False):
        CASES[f"tools-{kind}-think{int(thinking)}"] = T.build_tool_grammar(
            TOOLS, kind, name, thinking, lambda sub: convert(sub))
CASES["json_object-prologue"] = T.with_think_prologue(grammar_api.JSON_OBJECT_GBNF)
CASES["json_object-bare"] = grammar_api.JSON_OBJECT_GBNF
PERSON = {"type": "object", "additionalProperties": False, "required": ["name", "age", "tags"],
          "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                         "tags": {"type": "array", "items": {"type": "string"}}}}
CASES["schema-person-prologue"] = T.with_think_prologue(convert(PERSON))
CASES["schema-person-bare"] = convert(PERSON)
CASES["bare-ab"] = 'root ::= "a" | "b"\n'

fails = 0
for k, text in CASES.items():
    path = f"F:/Strata-dev/.zcode-obj/probe_{k}.gbnf"
    open(path, "w", encoding="utf-8", newline="").write(text)
    p = subprocess.run([PROBE, path], capture_output=True)
    ok = p.returncode == 0
    fails += 0 if ok else 1
    print(f"{k:28s} {'PARSE OK' if ok else 'PARSE FAIL: ' + p.stderr.decode()[:160]}")
print("ALL-OK" if fails == 0 else f"{fails} FAILURES")
