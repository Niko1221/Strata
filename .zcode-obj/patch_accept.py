import io
p = "F:/Strata-dev/tools/grammar_acceptance.py"
s = io.open(p, encoding="utf-8").read()

# 1. 30 tool prompts (two weather cycles + flights)
old = '''] + [
    f"Find a one-way flight from {a} to {b} on 2026-11-{d}."
    for a, b, d in [("SFO", "NRT", "03"), ("LHR", "CDG", "05"), ("JFK", "LAX", "07"),
                    ("DXB", "BOM", "09"), ("SYD", "AKL", "11"), ("FRA", "ATH", "13"),
                    ("GRU", "MEX", "15"), ("SIN", "DPS", "17"), ("YYZ", "YVR", "19"),
                    ("CPT", "NBO", "21")]
]'''
new = '''] + [
    f"Find a one-way flight from {a} to {b} on 2026-11-{d}."
    for a, b, d in [("SFO", "NRT", "03"), ("LHR", "CDG", "05"), ("JFK", "LAX", "07"),
                    ("DXB", "BOM", "09"), ("SYD", "AKL", "11"), ("FRA", "ATH", "13"),
                    ("GRU", "MEX", "15"), ("SIN", "DPS", "17"), ("YYZ", "YVR", "19"),
                    ("CPT", "NBO", "21")]
] + [
    f"What is the weather in {c} right now?" for c in
    ["Lagos", "Bangkok", "Buenos Aires", "Dublin", "Nairobi", "Tel Aviv", "Vienna", "Bogota", "Manila", "Zurich"]
]'''
assert s.count(old) == 1
s = s.replace(old, new, 1)

# 2. json suite: bigger budget (thinking low rides 688 tokens; the answer needs room)
old = '''                        "max_tokens": 1200, "reasoning_effort": "low", "temperature": 0.6})
        if r.status_code == 502 and "structured_output_failed" in r.text:
            canary += 1
            fails.append((j, name, "canary 502: " + r.text[:200]))
            continue
        if r.status_code != 200:
            other += 1
            fails.append((j, name, f"http {r.status_code}: {r.text[:200]}"))
            continue'''
new = '''                        "max_tokens": 2400, "reasoning_effort": "low", "temperature": 0.6})
        if r.status_code == 502 and "structured_output_failed" in r.text:
            canary += 1
            fails.append((j, name, "canary 502: " + r.text[:200]))
            print(f"  json {j + 1}/{len(items)} {name}: CANARY", flush=True)
            continue
        if r.status_code != 200:
            other += 1
            fails.append((j, name, f"http {r.status_code}: {r.text[:200]}"))
            print(f"  json {j + 1}/{len(items)} {name}: HTTP {r.status_code}", flush=True)
            continue'''
assert s.count(old) == 1
s = s.replace(old, new, 1)

# 3. tools suite: per-fail print
old = '''        print(f"  tool {j + 1}/{len(TOOL_PROMPTS[:n])}: {'ok' if good else 'FAIL'}", flush=True)'''
new = '''        print(f"  tool {j + 1}/{len(TOOL_PROMPTS[:n])}: {'ok' if good else 'FAIL: ' + fails[-1][1][:140]}", flush=True)'''
assert s.count(old) == 1
s = s.replace(old, new, 1)

# 4. perf + probe: thinking OFF (measure the constrained DECODE, not the thinking budget)
old = '''    payload = {"messages": [{"role": "user", "content": PERF_PROMPT}],
               "max_tokens": 400, "reasoning_effort": "low", "temperature": 0.8, "seed": 7}'''
new = '''    payload = {"messages": [{"role": "user", "content": PERF_PROMPT}],
               "max_tokens": 400, "reasoning_effort": "none", "temperature": 0.8, "seed": 7}'''
assert s.count(old) == 1
s = s.replace(old, new, 1)
old = '''        payload = {"messages": [{"role": "user", "content": PROBE_PROMPT + f" (set {i})"}],
                   "max_tokens": 400, "reasoning_effort": "low", "temperature": 0.8, "seed": 100 + i}'''
new = '''        payload = {"messages": [{"role": "user", "content": PROBE_PROMPT + f" (set {i})"}],
                   "max_tokens": 400, "reasoning_effort": "none", "temperature": 0.8, "seed": 100 + i}'''
assert s.count(old) == 1
s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched")
