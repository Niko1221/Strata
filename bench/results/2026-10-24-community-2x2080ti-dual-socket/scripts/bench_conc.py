#!/usr/bin/env python3
"""Concurrent throughput against whichever Strata server runs on :8080 (old or new engine).

  python3 bench_conc.py <label> [concurrency]

Warms the prompt cache, then sends N concurrent 200-token completions and reports the
aggregate API throughput + GPU utilisation. Works on both engine versions unchanged.
"""
import json, statistics as st, subprocess, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8080/v1/chat/completions"
def _model():
    try:
        with urllib.request.urlopen(URL.replace("/chat/completions", "/models"), timeout=10) as r:
            return json.load(r)["data"][0]["id"]
    except Exception:
        return "qwen3.8-flash-next-unsloth-ud-iq4_xs"
MODEL = _model()
LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
FILLER = ("The quarterly logistics review covers warehouse throughput, inbound scheduling "
          "and dock allocation for every regional hub. ")
PROMPT = FILLER * 160 + "\n\nSummarise the review above in three bullet points."


class Util(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event(); self.s = {0: [], 1: []}
    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,power.draw",
                                      "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                for line in out.strip().splitlines():
                    f = [x.strip() for x in line.split(",")]
                    if len(f) >= 3:
                        self.s[int(f[0])].append((float(f[1]), float(f[2])))
            except Exception:
                pass
            self.stop.wait(0.25)


def send(i):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 200, "temperature": 0.0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return time.perf_counter() - t0, int(d.get("usage", {}).get("completion_tokens") or 0)


def main():
    with ThreadPoolExecutor(2) as ex:      # warm the prompt cache
        [f.result() for f in [ex.submit(send, i) for i in range(2)]]
    u = Util(); u.start()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(N) as ex:
        res = [f.result() for f in [ex.submit(send, i) for i in range(N)]]
    wall = time.perf_counter() - t0
    u.stop.set(); time.sleep(0.5)
    toks = sum(x[1] for x in res)
    u0 = [x[0] for x in u.s[0]] or [0]; u1 = [x[0] for x in u.s[1]] or [0]
    p0 = [x[1] for x in u.s[0]] or [0]; p1 = [x[1] for x in u.s[1]] or [0]
    r = {"label": LABEL, "n": N, "tokens": toks, "wall": wall, "agg": toks / wall if wall else 0,
         "per_req": [x[1] for x in res], "gpu0": st.median(u0), "gpu1": st.median(u1),
         "p0": st.median(p0), "p1": st.median(p1)}
    print(f"{LABEL}: 并发 {N}  聚合 {r['agg']:.1f} tok/s  每请求 {r['per_req']}  "
          f"GPU {r['gpu0']:.0f}/{r['gpu1']:.0f}%  功耗 {r['p0']:.0f}/{r['p1']:.0f}W", flush=True)
    import pathlib
    pathlib.Path(f"bench_conc_{LABEL}.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
