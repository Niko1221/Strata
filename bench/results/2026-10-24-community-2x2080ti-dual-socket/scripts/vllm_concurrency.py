#!/usr/bin/env python3
"""The fair half of the comparison: vLLM under the same concurrency gradient.

The Strata layer split measured 22.3 -> 60.4 -> 62.2 -> 64.1 aggregate tok/s at
1/2/4/8 concurrent requests (concurrency.py). But vLLM's 65.3 tok/s baseline was
measured one request at a time, so quoting it beside those numbers is unfair to vLLM.
This runs the SAME prompt, the SAME max_tokens and the SAME gradient against vLLM on
the same two cards, so "which way is smarter on this hardware" has one comparable table.
"""
import json, os, statistics as st, subprocess, sys, threading, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

VLLM = os.environ.get("VLLM_DIR", os.path.expanduser("~/vllm-2080ti"))
API = "http://127.0.0.1:8000"
FILLER = ("The quarterly logistics review covers warehouse throughput, inbound scheduling "
          "and dock allocation for every regional hub. ")
PROMPT = FILLER * 160 + "\n\nSummarise the review above in three bullet points."


def sh(cmd, cwd=VLLM, timeout=3600):
    return subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str),
                          capture_output=True, text=True, timeout=timeout)


def stop_strata():
    for pat in ("[e]ngine/strata", "[s]erve/" + "server.py", "[a]b_run.py"):
        subprocess.run(["pkill", "-f", pat], capture_output=True)
    time.sleep(4)


def model_id():
    with urllib.request.urlopen(f"{API}/v1/models", timeout=10) as r:
        d = json.load(r)
    return d["data"][0]["id"]


def send(mid, max_tokens=200, timeout=1800):
    body = {"model": mid, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens, "temperature": 0.0}
    req = urllib.request.Request(f"{API}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    # count real output tokens from the response, so vLLM needs no engine log
    n = d.get("usage", {}).get("completion_tokens")
    if n is None:
        n = len(d["choices"][0]["message"].get("content") or "") // 4
    return time.perf_counter() - t0, int(n)


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


def level(mid, n):
    u = Util(); u.start()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as ex:
        res = [f.result() for f in [ex.submit(send, mid) for _ in range(n)]]
    total = time.perf_counter() - t0
    u.stop.set(); time.sleep(0.4)
    gen = sum(n_ for _, n_ in res)
    u0 = [x[0] for x in u.s[0]] or [0]; u1 = [x[0] for x in u.s[1]] or [0]
    p0 = [x[1] for x in u.s[0]] or [0]; p1 = [x[1] for x in u.s[1]] or [0]
    return {"n": n, "gen": gen, "wall": total, "agg": gen / total if total else 0,
            "per_req_wall": [w for w, _ in res],
            "decode": [n_ / w for w, n_ in res],
            "u0": st.median(u0), "u1": st.median(u1),
            "w0": st.median(p0), "w1": st.median(p1)}


def main():
    stop_strata()
    print("=== 启动 vLLM ===", flush=True)
    r = sh(["./llm.sh", "start"])
    print("  start rc:", r.returncode, (r.stdout or r.stderr or "")[-300:], flush=True)
    for _ in range(120):
        try:
            mid = model_id(); print("  READY, model:", mid, flush=True); break
        except Exception:
            time.sleep(5)
    else:
        print("  vLLM 未就绪，放弃", flush=True); return

    rows = []
    for n in (1, 2, 4, 8):
        print(f"\n######## vLLM 并发 {n} ########", flush=True)
        try:
            x = level(mid, n)
        except Exception as e:
            print("  FAILED:", e, flush=True); continue
        rows.append(x)
        print(f"  生成 {x['gen']} tok / {x['wall']:.1f}s => 聚合 {x['agg']:.1f} tok/s", flush=True)
        print(f"  单请求 decode: {[round(v,1) for v in x['decode']]}", flush=True)
        print(f"  GPU0 {x['u0']:.0f}% / GPU1 {x['u1']:.0f}%   功耗 {x['w0']:.0f}/{x['w1']:.0f} W", flush=True)

    json.dump(rows, open(os.path.join(VLLM, "concurrency_vllm.json"), "w"), indent=1)
    print("\n==================== vLLM 汇总 ====================")
    base = rows[0]["agg"] if rows else 0
    print(f"{'并发':>4} {'聚合 tok/s':>11} {'相对1×':>8} {'单请求decode中位':>16} {'GPU0':>6} {'GPU1':>6} {'功耗W':>13}")
    for x in rows:
        print(f"{x['n']:>4} {x['agg']:>11.1f} {x['agg']/base if base else 0:>7.2f}x "
              f"{st.median(x['decode']):>16.1f} {x['u0']:>5.0f}% {x['u1']:>5.0f}% "
              f"{x['w0']:>5.0f}/{x['w1']:<5.0f}")


if __name__ == "__main__":
    main()
