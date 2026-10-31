import re

P = "F:/Strata-dev/tools/grammar_acceptance.py"
src = open(P, encoding="utf-8").read()

new_tools = '''def tools_suite(base, n, base_totals, out_dir=None):
    jl = open(out_dir / "tool-items.jsonl", "a", encoding="utf-8") if out_dir else None

    def rec(row):
        if jl:
            jl.write(json.dumps(row, ensure_ascii=False) + "\\n")
            jl.flush()

    ok, bad = 0, 0
    fails = []
    for j, prompt in enumerate(TOOL_PROMPTS[:n]):
        print(f"  tool {j + 1}/{len(TOOL_PROMPTS[:n])} ...", end=" ", flush=True)
        t0 = time.time()
        row = {"i": j}
        try:
            r = post(base, {"messages": [{"role": "user", "content": prompt}], "tools": TOOLS,
                            "tool_choice": "required", "max_tokens": 2400, "reasoning_effort": "low",
                            "temperature": 0.6}, timeout=420)
            el = time.time() - t0
            row["s"] = round(el, 1)
            if r.status_code != 200:
                bad += 1
                row["verdict"] = f"HTTP {r.status_code}"
                row["detail"] = r.text[:300]
                fails.append((j, f"http {r.status_code}: {r.text[:200]}"))
                print(f"HTTP {r.status_code} {el:.0f}s", flush=True)
            else:
                d = r.json()
                ch = d["choices"][0]
                calls = ch["message"].get("tool_calls") or []
                names = {t["function"]["name"] for t in TOOLS}
                good = ch.get("finish_reason") == "tool_calls" and len(calls) >= 1
                row["finish"] = ch.get("finish_reason")
                row["reasoning_len"] = len(ch["message"].get("reasoning_content") or "")
                row["calls"] = [(c.get("function") or {}).get("name") for c in calls]
                for c in calls:
                    fn = c.get("function") or {}
                    if fn.get("name") not in names:
                        good = False
                    try:
                        json.loads(fn.get("arguments") or "")
                    except Exception:
                        good = False
                if good:
                    ok += 1
                    row["verdict"] = "OK"
                    print(f"ok {el:.0f}s", flush=True)
                else:
                    bad += 1
                    row["verdict"] = "FAIL"
                    row["detail"] = f"finish={ch.get('finish_reason')} calls={str(calls)[:200]}"
                    fails.append((j, row["detail"]))
                    print(f"FAIL: {row['detail'][:140]} {el:.0f}s", flush=True)
        except Exception as e:
            el = time.time() - t0
            bad += 1
            row["s"] = round(el, 1)
            row["verdict"] = "EXC"
            row["detail"] = repr(e)[:400]
            fails.append((j, row["detail"]))
            print(f"EXC {el:.0f}s", flush=True)
        rec(row)
    if jl:
        jl.close()
    totals = requests.get(base + "/metrics", timeout=30).json()["totals"]
    rescued = totals.get("tool_calls_from_reasoning", 0) - base_totals.get("tool_calls_from_reasoning", 0)
    return {"n": len(TOOL_PROMPTS[:n]), "parsed": ok, "parse_failures": bad,
            "rescued_from_reasoning_delta": rescued, "fails": fails[:20]}


'''

start = src.index("def tools_suite(base, n, base_totals):")
end = src.index("# ---------------------------------------------------------------- phase 3")
src = src[:start] + new_tools + src[end:]

# ---- 3. probe_round: guard usage, collect per-slot errors
old_one = '''        r = post(base_url, payload)
        return r.json()["usage"]["completion_tokens"]
    cpu0, wall0 = cpu_seconds(eng), time.time()
    with cf.ThreadPoolExecutor(max_workers=slots) as ex:
        toks = sum(ex.map(one, range(slots)))
    dt, dcpu = time.time() - wall0, cpu_seconds(eng) - cpu0
    return {"tok_s": round(toks / dt, 1), "cpu_s": round(dcpu, 2), "cpu_ms_per_tok": round(1000 * dcpu / toks, 2),
            "tokens": toks}'''
new_one = '''        r = post(base_url, payload)
        d = r.json()
        u = ((d.get("usage") or {}).get("completion_tokens")) or 0
        if r.status_code != 200 or u == 0:
            raise RuntimeError(f"http {r.status_code}: {r.text[:200]}")
        return u
    errs = []

    def safe(i):
        try:
            return one(i)
        except Exception as e:
            errs.append(repr(e)[:250])
            return 0
    cpu0, wall0 = cpu_seconds(eng), time.time()
    with cf.ThreadPoolExecutor(max_workers=slots) as ex:
        toks = sum(ex.map(safe, range(slots)))
    dt, dcpu = time.time() - wall0, cpu_seconds(eng) - cpu0
    if toks == 0:
        return {"error": "; ".join(errs)[:400], "tok_s": 0, "cpu_s": round(dcpu, 2), "cpu_ms_per_tok": 0,
                "tokens": 0}
    return {"tok_s": round(toks / dt, 1), "cpu_s": round(dcpu, 2), "cpu_ms_per_tok": round(1000 * dcpu / toks, 2),
            "tokens": toks, "slot_errors": errs}'''
assert old_one in src, "probe anchor not found"
src = src.replace(old_one, new_one)

# ---- 4. main: pass out_dir into the suites + utf-8 stdout
src = src.replace('out["json_suite"] = json_suite(base, a.json_n)',
                  'out["json_suite"] = json_suite(base, a.json_n, out_dir)')
src = src.replace('out["tools_suite"] = tools_suite(base, a.tools_n, totals0)',
                  'out["tools_suite"] = tools_suite(base, a.tools_n, totals0, out_dir)')
anchor = '    a = ap.parse_args()\n'
assert anchor in src
src = src.replace(anchor, anchor + '''    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
''', 1)

open(P, "w", encoding="utf-8").write(src)
print("patched7")
