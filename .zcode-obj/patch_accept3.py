import io, re
p = "F:/Strata-dev/tools/grammar_acceptance.py"
s = io.open(p, encoding="utf-8").read()

# suite-side timeout for all chat requests
s = s.replace('"timeout": 900', '"timeout": 420')
s = s.replace("timeout=900", "timeout=420")

# request progress + per-request seconds in the suite prints
old = '''    for j, (name, schema, prompt) in enumerate(items):
        r = post(base, {"messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema}},
                        "max_tokens": 2400, "reasoning_effort": "low", "temperature": 0.6})'''
new = '''    for j, (name, schema, prompt) in enumerate(items):
        print(f"  json {j + 1}/{len(items)} {name} ...", end=" ", flush=True)
        t0 = time.time()
        r = post(base, {"messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema}},
                        "max_tokens": 2400, "reasoning_effort": "low", "temperature": 0.6}, timeout=420)
        el = time.time() - t0'''
assert s.count(old) == 1
s = s.replace(old, new, 1)

# pass/fail lines gain elapsed
old = '''            print(f"  json {j + 1}/{len(items)} {name}: CANARY", flush=True)'''
new = '''            print(f"CANARY {el:.0f}s", flush=True)'''
assert s.count(old) == 1
s = s.replace(old, new, 1)
old = '''            print(f"  json {j + 1}/{len(items)} {name}: HTTP {r.status_code}", flush=True)'''
new = '''            print(f"HTTP {r.status_code} {el:.0f}s", flush=True)'''
assert s.count(old) == 1
s = s.replace(old, new, 1)

# the success print: find it
m = re.search(r'( +)print\(f"  json \{j \+ 1\}/\{len\(items\)\} \{name\}: \{[^}]*\}", flush=True\)', s)
print("success print:", m.group(0) if m else "NOT FOUND")

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
