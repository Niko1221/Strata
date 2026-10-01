"""Re-measure one cell with enough trials to see whether a surprising number is real.

131K came out SLOWER than 262K, which cannot be right: 131K has more expert slots
(5,050 vs 3,915), so it should decode faster. Either the effect is real and caused by
something other than the cache, or three trials is too few and one slow request moved
the median. This runs more trials and prints them all, so the spread is visible instead
of hidden behind a single number.
"""
import json, random, statistics, string, sys, time, urllib.request, pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
PORT = 8090
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

def ask(prompt, max_tokens=160):
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0,
                       "reasoning_effort": "none"}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    return (d["usage"]["completion_tokens"] / dt,
            (d["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            d["usage"]["completion_tokens"])

def main():
    label = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    ask("warm", 4)
    rates = []
    for i in range(n):
        tag = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(48))
        r, cached, ct = ask(f"[ref {tag}] Count from 1 to 80 separated by commas. Output only the list.")
        ok = "ok " if (ct > 40 and not cached) else "REJECTED"
        print(f"  trial {i+1}: {r:6.1f} tok/s  ({ct} tok, cached={cached}) {ok}", flush=True)
        if ct > 40 and not cached:
            rates.append(r)
    rates.sort()
    if rates:
        print(f"{label}: median {statistics.median(rates):.1f}  min {rates[0]:.1f}  "
              f"max {rates[-1]:.1f}  n={len(rates)}", flush=True)

if __name__ == "__main__":
    main()
