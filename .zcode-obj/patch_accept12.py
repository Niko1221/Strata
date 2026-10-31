P = "F:/Strata-dev/tools/grammar_acceptance.py"
src = open(P, encoding="utf-8").read()

# 1. bounded arrays: an unbounded array (recipe.steps had only minItems) lets a degenerate model loop
#    elements forever ("One hundred and ninetieth: it's fine" x 50 -> budget canary); real schemas bound
old = '''    "recipe": {"type": "object", "additionalProperties": False,
               "properties": {"dish": {"type": "string"},
                              "steps": {"type": "array", "minItems": 4, "items": {"type": "string"}},
                              "vegetarian": {"type": "boolean"}},'''
new = '''    "recipe": {"type": "object", "additionalProperties": False,
               "properties": {"dish": {"type": "string"},
                              "steps": {"type": "array", "minItems": 4, "maxItems": 12,
                                        "items": {"type": "string"}},
                              "vegetarian": {"type": "boolean"}},'''
assert old in src
src = src.replace(old, new, 1)

old2 = '''                "properties": {"full_name": {"type": "string"}, "email": {"type": "string"},
                               "phone": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}},'''
new2 = '''                "properties": {"full_name": {"type": "string"}, "email": {"type": "string"},
                               "phone": {"type": "string"},
                               "tags": {"type": "array", "maxItems": 6, "items": {"type": "string"}}},'''
assert old2 in src
src = src.replace(old2, new2, 1)

# 2. the probe: a degenerate string at temp 0.8 canaried one slot of six twice in a row
old3 = '"max_tokens": 600, "reasoning_effort": "none", "temperature": 0.8, "seed": 100 + i}'
new3 = '"max_tokens": 600, "reasoning_effort": "none", "temperature": 0.6, "seed": 100 + i}'
assert old3 in src
src = src.replace(old3, new3, 1)

# 3. the engine-pid scan: ctypes returns szExeFile as bytes; compare decoded
old4 = '''            if name.lower() == exe_name.lower():
                return pid
            queue.append(pid)'''
new4 = '''            if (name.decode(errors="replace") if isinstance(name, bytes) else name).lower() == exe_name.lower():
                return pid
            queue.append(pid)'''
assert old4 in src
src = src.replace(old4, new4, 1)

old5 = '''                if e.szExeFile.lower() == "strata.exe":'''
new5 = '''                if (e.szExeFile.decode(errors="replace")
                        if isinstance(e.szExeFile, bytes) else e.szExeFile).lower() == "strata.exe":'''
assert old5 in src
src = src.replace(old5, new5, 1)

open(P, "w", encoding="utf-8").write(src)
print("patched12")
