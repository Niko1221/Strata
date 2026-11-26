#!/usr/bin/env python3
"""Practical benchmark of the model serving on port 8080: is it right, and how long until the answer shows?

    bench_models.py LABEL

22 tasks with answers a program can check: 6 coding (the function is run against tests), 6 reasoning with one exact
answer, 4 tool calls (right tool and argument), 3 reading questions on a long document with planted facts, 3 format
tasks. Sent the way omp does (stream, reasoning_effort "high", up to 6000 tokens). For each task it records pass/fail,
seconds to the first visible answer text, total seconds, output tokens and how many of them were thinking.
Results -> tests/results-bench-LABEL.json and a summary line per group.
"""
import json, os, re, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "http://127.0.0.1:8080/v1/chat/completions"
label = sys.argv[1]

def ask(messages, tools=None, max_tokens=6000):
    body = {"model": "x", "messages": messages, "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": max_tokens, "reasoning_effort": "high", "temperature": 0}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; text = []; think = 0; calls = {}; usage = {}; finish = None
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                try:
                    d = json.loads(line[5:])
                except ValueError:
                    continue
                usage = d.get("usage") or usage
                for ch in d.get("choices") or []:
                    finish = ch.get("finish_reason") or finish
                    dl = ch.get("delta") or {}
                    if dl.get("reasoning_content"):
                        think += len(dl["reasoning_content"])
                    if dl.get("content"):
                        if first is None:
                            first = time.time() - t0
                        text.append(dl["content"])
                    for tc in dl.get("tool_calls") or []:
                        if first is None:
                            first = time.time() - t0
                        c = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                        f = tc.get("function") or {}
                        c["name"] += f.get("name") or ""; c["arguments"] += f.get("arguments") or ""
    except Exception as e:
        return {"error": repr(e)[:200], "total_s": round(time.time() - t0, 1), "content": "", "calls": []}
    return {"content": "".join(text), "calls": list(calls.values()), "first_s": None if first is None else round(first, 1),
            "total_s": round(time.time() - t0, 1), "out_tokens": usage.get("completion_tokens"), "think_chars": think,
            "finish": finish}

def run_code(reply, tests):
    m = re.findall(r"```(?:python)?\n(.*?)```", reply, re.S)
    code = max(m, key=len) if m else reply
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/t.py", "w").write(code + "\n\n" + tests + "\nprint('ALL_OK')\n")
        try:
            p = subprocess.run([sys.executable, "-I", "t.py"], cwd=d, capture_output=True, text=True, timeout=20)
            return "ALL_OK" in p.stdout
        except Exception:
            return False

CODE = [
 ("Write a Python function `merge_intervals(intervals)` that merges overlapping closed intervals given as [start, end] lists and returns them sorted. Only the code.",
  "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]])==[[1,5]]\nassert merge_intervals([])==[]\nassert merge_intervals([[5,7],[1,2]])==[[1,2],[5,7]]"),
 ("Write a Python function `roman_to_int(s)` that converts a Roman numeral string to an integer. Only the code.",
  "assert roman_to_int('MCMXCIV')==1994\nassert roman_to_int('LVIII')==58\nassert roman_to_int('IX')==9\nassert roman_to_int('MMXXVI')==2026"),
 ("Write a Python function `lru(capacity)` that returns an object with `get(key)` (returns -1 if missing) and `put(key, value)`, evicting the least recently used key when over capacity. Only the code.",
  "c=lru(2); c.put(1,1); c.put(2,2); assert c.get(1)==1; c.put(3,3); assert c.get(2)==-1; c.put(4,4); assert c.get(1)==-1; assert c.get(3)==3; assert c.get(4)==4"),
 ("Write a Python function `parse_duration(s)` that turns strings like '1h30m', '45s', '2h', '1h2m3s' into total seconds (int). Raise ValueError on anything else, including the empty string. Only the code.",
  "assert parse_duration('1h30m')==5400\nassert parse_duration('45s')==45\nassert parse_duration('2h')==7200\nassert parse_duration('1h2m3s')==3723\nfor b in ('', 'abc', '5x', '1m1h'):\n    try:\n        parse_duration(b); raise SystemExit('no error for '+b)\n    except ValueError:\n        pass"),
 ("Write a Python function `top_k_words(text, k)` that returns the k most frequent words, lowercased, ignoring punctuation, as a list of (word, count) tuples sorted by count descending and then alphabetically. Only the code.",
  "assert top_k_words('The cat. the dog! A cat, a CAT?',2)==[('cat',3),('a',2)]\nassert top_k_words('b a b a c',3)==[('a',2),('b',2),('c',1)]\nassert top_k_words('',2)==[]"),
 ("This function has a bug. Fix it and return only the corrected code.\n```python\ndef binary_search(a, x):\n    lo, hi = 0, len(a)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if a[mid] < x:\n            lo = mid\n        else:\n            hi = mid\n    return lo if lo < len(a) and a[lo] == x else -1\n```",
  "assert binary_search([1,3,5,7,9],7)==3\nassert binary_search([1,3,5,7,9],1)==0\nassert binary_search([1,3,5,7,9],9)==4\nassert binary_search([1,3,5,7,9],4)==-1\nassert binary_search([],1)==-1"),
]
REASON = [
 ("A train leaves at 14:35 and arrives at 17:10 the same day. It stops for 12 minutes on the way. How many minutes is it moving? End with 'Answer: <number>'.", "143"),
 ("I have 3 boxes. Box A has twice as many apples as box B. Box C has 5 more than box B. Together they hold 41 apples. How many are in box A? End with 'Answer: <number>'.", "18"),
 ("What is 17 * 23 + 144 / 12 - 9? End with 'Answer: <number>'.", "394"),
 ("Five people finished a race. Ana finished before Ben. Cal finished after Dee but before Ana. Eve finished last. Who finished second? End with 'Answer: <name>'.", "cal"),
 ("A password must be 4 characters, each a digit from 0-9, with no digit repeated, and the first digit not 0. How many passwords are possible? End with 'Answer: <number>'.", "4536"),
 ("A file is 3.2 GB. A link downloads 25 megabytes per second (1 GB = 1000 MB). How many seconds does the download take? End with 'Answer: <number>'.", "128"),
]
TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": p, "required": list(p)}}} for n, d, p in (
    ("read_file", "Read a file from the project", {"path": {"type": "string"}}),
    ("run_command", "Run a shell command and return its output", {"command": {"type": "string"}}),
    ("search_code", "Search the project for a text pattern", {"pattern": {"type": "string"}}),
    ("write_file", "Write text to a file", {"path": {"type": "string"}, "content": {"type": "string"}}))]
TOOLQ = [
 ("Show me what is in config/settings.yaml.", "read_file", "settings.yaml"),
 ("Find every place in the code that mentions parse_duration.", "search_code", "parse_duration"),
 ("How many lines does README.md have? Use the shell.", "run_command", "README.md"),
 ("Create a file notes/todo.txt containing exactly: buy milk", "write_file", "buy milk"),
]
FACTS = [("the maintenance window", "Thursday 03:15"), ("the port of the metrics exporter", "9417"), ("the on-call engineer for storage", "Priya Raman")]
filler = " ".join(f"Section {i}: the service keeps its usual limits and nothing changed in subsystem {i % 37}; operators should follow the standard runbook for item {i}." for i in range(380))
parts = filler.split("Section 120:")
DOC = (parts[0] + f"Note: {FACTS[0][0]} is {FACTS[0][1]}. Section 120:" + parts[1]).replace("Section 250:", f"Note: {FACTS[1][0]} is {FACTS[1][1]}. Section 250:").replace("Section 340:", f"Note: {FACTS[2][0]} is {FACTS[2][1]}. Section 340:")
FORMAT = [
 ("Return only a JSON object with keys \"name\" (string), \"tags\" (array of exactly 3 strings) and \"count\" (the number 7). No other text.",
  lambda t: (lambda o: isinstance(o.get("name"), str) and isinstance(o.get("tags"), list) and len(o["tags"]) == 3 and o.get("count") == 7)(json.loads(re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()))),
 ("List the first 6 prime numbers, comma-separated, on one line, nothing else.", lambda t: re.sub(r"\s", "", t.strip().rstrip(".")) == "2,3,5,7,11,13"),
 ("Reply with exactly the word DONE in capitals and nothing else.", lambda t: t.strip() == "DONE"),
]

rows = []
def rec(group, name, ok, r):
    rows.append({"group": group, "task": name, "ok": bool(ok), **{k: r.get(k) for k in ("first_s", "total_s", "out_tokens", "think_chars", "finish", "error")}})
    print(f"{group:8} {name[:34]:34} {'PASS' if ok else 'FAIL'}  first {r.get('first_s')} s  total {r.get('total_s')} s  out {r.get('out_tokens')}  {r.get('error') or ''}", flush=True)

ask([{"role": "user", "content": "Say hi."}], max_tokens=50)          # warm-up, not counted
for q, tests in CODE:
    r = ask([{"role": "user", "content": q}]); rec("code", q[24:60], run_code(r["content"], tests), r)
for q, want in REASON:
    r = ask([{"role": "user", "content": q}]); m = re.findall(r"Answer:\s*\**\s*([\w.,:]+)", r["content"])
    rec("reason", q[:34], bool(m) and m[-1].lower().replace(",", "").rstrip(".") == want, r)
for q, tool, needle in TOOLQ:
    r = ask([{"role": "user", "content": q}], tools=TOOLS)
    rec("tool", q[:34], any(c["name"] == tool and needle in c["arguments"] for c in r["calls"]), r)
for what, val in FACTS:
    r = ask([{"role": "user", "content": DOC + f"\n\nAccording to the text above, what is {what}? Answer in one short sentence."}])
    rec("reading", what, val.lower() in r["content"].lower(), r)
for q, check in FORMAT:
    r = ask([{"role": "user", "content": q}])
    try:
        ok = check(r["content"])
    except Exception:
        ok = False
    rec("format", q[:34], ok, r)

summary = {}
for g in ("code", "reason", "tool", "reading", "format"):
    rs = [x for x in rows if x["group"] == g]
    firsts = sorted(x["first_s"] for x in rs if x["first_s"] is not None); tot = sorted(x["total_s"] for x in rs)
    summary[g] = {"passed": sum(x["ok"] for x in rs), "of": len(rs), "median_first_s": firsts[len(firsts) // 2] if firsts else None,
                  "median_total_s": tot[len(tot) // 2], "worst_total_s": tot[-1], "no_answer": sum(1 for x in rs if x["first_s"] is None)}
summary["all"] = {"passed": sum(x["ok"] for x in rows), "of": len(rows), "total_minutes": round(sum(x["total_s"] for x in rows) / 60, 1)}
json.dump({"label": label, "summary": summary, "rows": rows}, open(f"{HERE}/results-bench-{label}.json", "w"), indent=1)
print("SUMMARY", label, json.dumps(summary), flush=True)
