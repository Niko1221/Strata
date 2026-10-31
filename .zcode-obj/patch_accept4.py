import io
p = "F:/Strata-dev/tools/grammar_acceptance.py"
s = io.open(p, encoding="utf-8").read()

old = '''    for j, (name, prompt) in enumerate(items):
        schema = SCHEMAS[name]
        r = post(base, {"messages": [{"role": "user", "content": prompt + "\\nAnswer with one JSON object."}],
                        "response_format": {"type": "json_schema",
                                            "json_schema": {"name": name, "schema": schema, "strict": True}},
                        "max_tokens": 2400, "reasoning_effort": "low", "temperature": 0.6})'''
new = '''    for j, (name, prompt) in enumerate(items):
        schema = SCHEMAS[name]
        print(f"  json {j + 1}/{len(items)} {name} ...", end=" ", flush=True)
        t0 = time.time()
        r = post(base, {"messages": [{"role": "user", "content": prompt + "\\nAnswer with one JSON object."}],
                        "response_format": {"type": "json_schema",
                                            "json_schema": {"name": name, "schema": schema, "strict": True}},
                        "max_tokens": 2400, "reasoning_effort": "low", "temperature": 0.6}, timeout=420)
        el = time.time() - t0'''
assert s.count(old) == 1, "json loop"
s = s.replace(old, new, 1)
s = s.replace('print(f"  json {j + 1}/{len(items)} {name}: CANARY", flush=True)',
              'print(f"CANARY {el:.0f}s", flush=True)')
s = s.replace('print(f"  json {j + 1}/{len(items)} {name}: HTTP {r.status_code}", flush=True)',
              'print(f"HTTP {r.status_code} {el:.0f}s", flush=True)')
s = s.replace('print(f"  json {j + 1}/{len(items)} {name}: {\'pass\' if not errs else \'FAIL\'}", flush=True)',
              'print(f"{\'pass\' if not errs else \'FAIL\'} {el:.0f}s" + (f" :: {fails[-1][2][:110]}" if errs else ""), flush=True)')

# tools suite: timeout + per-item timing print
old = '''        r = post(base, {"messages": [{"role": "user", "content": prompt}],
                        "tools": TOOLS, "tool_choice": "required",
                        "max_tokens": 1200, "reasoning_effort": "low", "temperature": 0.6})'''
new = '''        print(f"  tool {j + 1}/{len(TOOL_PROMPTS[:n])} ...", end=" ", flush=True)
        t0 = time.time()
        r = post(base, {"messages": [{"role": "user", "content": prompt}],
                        "tools": TOOLS, "tool_choice": "required",
                        "max_tokens": 1200, "reasoning_effort": "low", "temperature": 0.6}, timeout=420)
        el = time.time() - t0'''
assert s.count(old) == 1, "tools loop"
s = s.replace(old, new, 1)
old = '''        print(f"  tool {j + 1}/{len(TOOL_PROMPTS[:n])}: {'ok' if good else 'FAIL: ' + fails[-1][1][:140]}", flush=True)'''
new = '''        print(f"{'ok' if good else 'FAIL: ' + str(fails[-1][1])[:140]} {el:.0f}s", flush=True)'''
assert s.count(old) == 1, "tools print"
s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched4")
