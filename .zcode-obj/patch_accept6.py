import re, sys

P = "F:/Strata-dev/tools/grammar_acceptance.py"
src = open(P, encoding="utf-8").read()

# ---- 1. rewrite json_suite: per-item JSONL, try/except, canary text, reasoning/content split
start = src.index("def json_suite(base, n):")
end = src.index("# ---------------------------------------------------------------- phase 2: tools suite")
new_json = '''def json_suite(base, n, out_dir=None):
    jl = open(out_dir / "json-items.jsonl", "a", encoding="utf-8") if out_dir else None

    def rec(row):
        if jl:
            jl.write(json.dumps(row, ensure_ascii=False) + "\\n")
            jl.flush()

    items = [(k, PROMPTS[k][i % len(PROMPTS[k])]) for k in SCHEMAS for i in range(n // len(SCHEMAS))]
    passed, canary, other = 0, 0, 0
    fails = []
    for j, (name, prompt) in enumerate(items):
        schema = SCHEMAS[name]
        print(f"  json {j + 1}/{len(items)} {name} ...", end=" ", flush=True)
        t0 = time.time()
        row = {"i": j, "name": name}
        try:
            r = post(base, {"messages": [{"role": "user", "content": prompt + "\\nAnswer with one JSON object."}],
                            "response_format": {"type": "json_schema",
                                                "json_schema": {"name": name, "schema": schema, "strict": True}},
                            "max_tokens": 2400, "reasoning_effort": "low", "temperature": 0.6}, timeout=420)
            el = time.time() - t0
            row["s"] = round(el, 1)
            if r.status_code == 502:
                canary += 1
                detail = r.text[:500]
                row["verdict"] = "CANARY"
                row["detail"] = detail
                fails.append((j, name, "canary 502: " + detail[:200]))
                print(f"CANARY {el:.0f}s :: {detail[:120]}", flush=True)
            elif r.status_code != 200:
                other += 1
                detail = r.text[:300]
                row["verdict"] = f"HTTP {r.status_code}"
                row["detail"] = detail
                fails.append((j, name, f"http {r.status_code}: {detail[:200]}"))
                print(f"HTTP {r.status_code} {el:.0f}s", flush=True)
            else:
                body = r.json()
                msg = body["choices"][0]["message"]
                row["finish"] = body["choices"][0].get("finish_reason")
                row["reasoning_len"] = len(msg.get("reasoning_content") or "")
                row["content_len"] = len(msg.get("content") or "")
                row["content_tail"] = (msg.get("content") or "")[-200:]
                try:
                    value = json.loads(msg.get("content") or "")
                except Exception as e:
                    other += 1
                    row["verdict"] = "NOT-JSON"
                    row["detail"] = str(e)[:200]
                    fails.append((j, name, f"not json: {e}: {str(msg.get('content'))[:160]}"))
                    print(f"FAIL {el:.0f}s", flush=True)
                else:
                    errs = validate(value, schema)
                    if errs:
                        other += 1
                        row["verdict"] = "SCHEMA-FAIL"
                        row["detail"] = "; ".join(errs[:3])
                        fails.append((j, name, row["detail"]))
                        print(f"FAIL {el:.0f}s :: {row['detail'][:110]}", flush=True)
                    else:
                        passed += 1
                        row["verdict"] = "PASS"
                        print(f"pass {el:.0f}s", flush=True)
        except Exception as e:
            el = time.time() - t0
            other += 1
            row["s"] = round(el, 1)
            row["verdict"] = "EXC"
            row["detail"] = repr(e)[:400]
            fails.append((j, name, row["detail"]))
            print(f"EXC {el:.0f}s", flush=True)
        rec(row)
    if jl:
        jl.close()
    return {"n": len(items), "passed": passed, "canary_502": canary, "other_failures": other,
            "fails": fails[:20]}


'''
src = src[:start] + new_json + src[end:]

# ---- 2. tools_suite: JSONL + budget 2400 + per-item try/except
start = src.index("def tools_suite(base, n, base_totals):")
end = src.index("# ---------------------------------------------------------------- phase 3")
m = re.search(r"def tools_suite\(base, n, base_totals\):.*?\n(?=# -+ phase 3)", src, re.S)
old = m.group(0)
print("OLD tools_suite length:", len(old))
# show the tail of the old function so the patch keeps its return shape
print(old[-400:])
open("F:/Strata-dev/.zcode-obj/tools_suite_old.txt", "w", encoding="utf-8").write(old)
open(P, "w", encoding="utf-8").write(src)
print("json_suite replaced; tools_suite saved for step 2")
