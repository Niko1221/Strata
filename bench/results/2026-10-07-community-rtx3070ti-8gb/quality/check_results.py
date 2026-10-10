"""Score a task-check record (written by strata_eval.py): run the generated expression evaluator against fixed cases,
parse and schema-check the JSON answer, check the poem's shape and the tool call, and print the answers that are
judged by reading. Usage: python check_results.py <task-check.json>"""
import json
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent / "eval_results.json"
r = {x["id"]: x for x in json.loads(path.read_text(encoding="utf-8"))}
print("record:", path.name, "| tasks:", len(r))

# code_expr: run the generated evaluate() against reference cases
code = re.search(r"```python\n(.*?)```", r["code_expr"]["content"], re.S).group(1)
ns = {}
exec(code, ns)
evaluate = ns["evaluate"]
cases = {
    "1 + 2 * 3": 7, "(1 + 2) * 3": 9, "10 / 4": 2.5, "-(2+3)": -5, "2 * -3": -6, "- -4": 4,
    "3 - 2 - 1": 0, "8 / 2 / 2": 2, "2 * (3 + 4) * 5": 70, " 1.5 + 2.25 ": 3.75, "((2))": 2,
    "-(1 + 2) * (3 - 5) / 4": 1.5, "100": 100, "1 - -1": 2, "2*3+4*5-6/2": 23,
}
ok = 0
for e, want in cases.items():
    try:
        got = evaluate(e)
        good = abs(got - want) < 1e-9
    except Exception as ex:
        got, good = repr(ex), False
    ok += good
    print(f"  expr {e!r:28} -> {got!r:10} want {want!r:6} {'ok' if good else 'FAIL'}")
errs = 0
for bad in ["(1 + 2", "1 +", "2 3", "abc"]:
    try:
        evaluate(bad)
        print(f"  malformed {bad!r}: no error (FAIL)")
    except Exception as ex:
        errs += 1
        print(f"  malformed {bad!r}: raised {type(ex).__name__} (ok)")
print(f"code_expr: {ok}/{len(cases)} cases correct, {errs}/4 malformed inputs rejected")

# json_strict
try:
    arr = json.loads(r["json_strict"]["content"])
    good = (isinstance(arr, list) and len(arr) == 3 and all(
        isinstance(p["name"], str) and isinstance(p["age"], int) and 20 <= p["age"] <= 60
        and isinstance(p["city"], str) and isinstance(p["skills"], list) and len(p["skills"]) == 2 for p in arr))
    print("json_strict: parses, schema ok =", good)
except Exception as ex:
    print("json_strict: FAIL", ex)

# poem: 4 lines x 7 Han characters
lines = [re.sub(r"[，。,.!！？?\s]", "", l) for l in r["poem"]["content"].strip().splitlines() if l.strip()]
print("poem: lines", len(lines), "chars per line", [len(l) for l in lines])

# tool call
tc = r["tool_call"].get("tool_calls") or []
print("tool_call:", [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in tc])

# fixed-answer checks
expect = {"math_zh_high": ["24/7", "frac{24}{7}"], "math_zh_none": ["24/7", "frac{24}{7}"],
          "logic_trap": ["1"], "logic_trap_high": ["1"], "needle": ["7391-ALPHA-蓝鲸"]}
for k, keys in expect.items():
    c = r[k]["content"] or ""
    print(f"{k}: contains expected answer = {any(s in c for s in keys)} | first line: {c.strip().splitlines()[0][:80]!r}")
print("code_bug: proposes 'lo = mid + 1' =", "lo = mid + 1" in (r["code_bug"]["content"] or ""))

# judged by reading: printed in full
for k in ("knowledge", "translate", "summary_zh"):
    print(f"--- {k} (judged by reading)\n{r[k]['content']}")
