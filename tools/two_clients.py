"""Two concurrent clients against a running Strata server (mock or model): both replies must arrive whole.

Used to check the chunked path later: today it records the order replies finish in.
"""
import json, threading, time, urllib.request

URL = "http://127.0.0.1:8099/v1/chat/completions"


def ask(prompt, out, key):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": 32, "temperature": 0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    out[key] = {"wall": round(time.time() - t0, 2), "chars": len((d["choices"][0]["message"].get("content") or ""))}


def main():
    out = {}
    ts = [threading.Thread(target=ask, args=("hello %d" % i, out, i)) for i in range(2)]
    t0 = time.time()
    [t.start() for t in ts]
    [t.join() for t in ts]
    print("total", round(time.time() - t0, 2), "s")
    for k in sorted(out):
        print(" client", k, out[k])


if __name__ == "__main__":
    main()
