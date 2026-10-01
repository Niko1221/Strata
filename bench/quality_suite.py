"""A small, checkable quality suite: every item has a verifiable right answer.

The point is a head-to-head between models/quantizations on the SAME box, so nothing
here depends on taste. Anything a grader must eyeball is deliberately excluded.
"""
import json, time, urllib.request, sys

PORT = 8090
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

def ask(prompt, max_tokens=200, effort="low"):
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0,
                       "reasoning_effort": effort}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    txt = (d["choices"][0]["message"].get("content") or "").strip()
    think = (d["choices"][0]["message"].get("reasoning_content") or "")
    return txt, think, d["usage"]["completion_tokens"], time.time() - t0

# (name, prompt, [acceptable answers], whether to allow reasoning tokens)
SUITE = [
    ("arith-2step",  "What is 17*23? Reply with only the number.", ["391"], False),
    ("arith-div",   "What is 144/7? Only the number.", ["20"], False),
    ("arith-pct",   "A shirt costs 80 and is 25% off. What is the sale price? Only the number.", ["60"], False),
    ("capital-fr",  "Capital of France? One word, lowercase.", ["paris"], False),
    ("capital-jp",  "Capital of Japan? One word, lowercase.", ["tokyo"], False),
    ("fact-date",   "In which year did the first Moon landing happen? Only the year.", ["1969"], True),
    ("know-planet", "Which is the largest planet in our solar system? One word, lowercase.", ["jupiter"], False),
    ("know-speed",  "Approximately how fast does light travel, in kilometres per second? Only the number.",
                    ["300000", "300,000", "300000000", "299792"], False),
    ("code-fib",    "Write Python: a function fib(n) returning the nth Fibonacci number, using a loop. Code only, "
                    "no explanation.", ["def"], True),
    ("code-sort",   "Write Python code to sort a list of integers. Code only.", ["sort", "sorted", "def"], True),
    ("code-bug",    "What is wrong with this Python: `for i in range(len(a)): print(a[i]` ? One sentence.",
                    ["bracket", "]", "missing", "unbalanced"], True),
    ("code-pydoc",  "In Python, what does a function decorated with @property do? One sentence.",
                    ["getter", "attribute", "property", "read-only"], True),
    ("instr-follow","Repeat this exact phrase and nothing else: HELLO WORLD TEST",
                    ["HELLO WORLD TEST"], False),
    ("instr-count", "How many letters are in the word BANANA? Only the number.", ["6"], False),
    # 15*4=60, 22/2=11, 60+11=71, 71-7=64.  (Checked: the Coder answered 64 and was right.)
    ("multi-calc",  "Compute (15*4) + (22/2) - 7. Only the number.", ["64"], True),
]

def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "model"
    passed = failed = 0
    fails = []
    for name, prompt, want, allow_think in SUITE:
        try:
            txt, think, ct, dt = ask(prompt, 220, "low" if allow_think else "none")
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            failed += 1; fails.append(name); continue
        blob = (txt + " " + think).lower() if allow_think else txt.lower()
        ok = any(w.lower() in blob for w in want)
        if ok: passed += 1
        else:
            failed += 1; fails.append(name)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:14s} ({ct:3d} tok {dt:5.1f}s)  {repr((txt or think)[:70])}")
    print(f"\n{label}: {passed}/{passed+failed} passed" + (f"  failed: {', '.join(fails)}" if fails else ""))

    # decode speed, warm
    ask("warm", 4, "none")
    _, _, ct, dt = ask("Count from 1 to 80 separated by commas. Output only the list.", 160, "none")
    print(f"{label}: decode {ct/dt:.1f} tok/s ({ct} tok / {dt:.2f}s)")

if __name__ == "__main__":
    main()
