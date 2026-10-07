"""Teacher-forced distributions, BF16 hyper-connection projections vs STRATA_HC_REQ8 (int8 + fp32/32).

The method of docs/UNSLOTH_Q4.md / bench/results/2026-10-03-v100-prompt-attn: a serve engine with
STRATA_LOGPOS=<file> STRATA_LOGPOS_TOPK=256 and --short-read covering the prompt, so every prompt token is read
through the verify windows and scored; fixed experts (--adapt-every 100000, --pcie-frac 0), greedy.  KL is
bf16 || other over the bf16 run's top 256 plus one bucket for the rest.  Control: a second bf16 run.

    python req8_kl.py run      # one server per arm, three texts each
    python req8_kl.py compare  # argmax agreement, top-10 overlap, mean/median/p99 KL, perplexity
"""
import json
import math
import os
import subprocess
import sys
import time
import urllib.request

S = os.path.dirname(os.path.abspath(__file__))
H = "/run/media/benjamin/BHOME/projects/hy3"
WT = H + "/Strata-check"                      # feat/hc-req8 (+ the local CUDA 12.4 build fix)
PY = H + "/Strata/.venv/bin/python"
D = H + "/Strata-data"
TEXTS = {"code": WT + "/src/kernels/cuda/native_rope.cu", "doc": WT + "/docs/DETAILS.md",
         "prose": WT + "/docs/BATCHING.md"}
ARMS = {"bf16": {}, "bf16-rerun": {}, "req8": {"STRATA_HC_REQ8": "1"}}
PORT = 18080


def text_of(name):
    t = open(TEXTS[name], encoding="utf-8").read()
    if name == "doc":
        t = t[len(t) // 3:]
    return t[:4000]          # ~1,000 tokens


def run():
    base = json.load(open(H + "/Strata/strata-q2_0.json"))
    for arm, extra in ARMS.items():
        logpos = f"{S}/{arm}.logpos"
        if os.path.exists(logpos):
            continue
        a = [x for x in base["args"]]
        for flag in ("--kv-resident", "--adapt-swaps", "--adapt-every", "--max-context"):
            if flag in a:
                i = a.index(flag)
                del a[i:i + 2]
        a += ["--max-context", "8192", "--short-read", "1600", "--adapt-every", "100000", "--pcie-frac", "0"]
        cfg = dict(base, exe=WT + "/build/strata", args=a, log=f"{S}/{arm}.engine.log", port=PORT,
                   env=dict(base.get("env", {}), STRATA_LOGPOS=logpos, STRATA_LOGPOS_TOPK="256", **extra))
        json.dump(cfg, open(f"{S}/{arm}.config.json", "w"), indent=1)
        unit = f"strata-kl-{arm}"
        subprocess.run(["systemd-run", "--user", "--collect", f"--unit={unit}", f"-pWorkingDirectory={WT}",
                        f"-pStandardOutput=file:{S}/{arm}.server.log", f"-pStandardError=file:{S}/{arm}.server.log",
                        PY, WT + "/serve/server.py", "--engine", "strata", "--config", f"{S}/{arm}.config.json",
                        "--port", str(PORT)], check=True, capture_output=True)
        for _ in range(180):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/status", timeout=3)
                if "ready" in open(f"{S}/{arm}.server.log").read():
                    break
            except Exception:
                pass
            time.sleep(5)
        for name in TEXTS:
            body = json.dumps({"model": "x", "messages": [{"role": "user", "content": text_of(name)}],
                               "max_tokens": 1, "temperature": 0, "reasoning_effort": "none"}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=900).read()
            with open(logpos, "a") as f:
                f.write(f"# end {name}\n")
            print(arm, name, "done", flush=True)
        subprocess.run(["systemctl", "--user", "stop", unit])
        time.sleep(3)


def rows(path):
    """{text: [(pos, target, target_lp, {id: lp} top-256)]}"""
    out, cur = {}, []
    for line in open(path):
        if line.startswith("# end "):
            out[line.split()[2]] = cur
            cur = []
            continue
        f = line.rstrip("\n").split("\t")
        top = {int(k): float(v) for k, v in (x.split(":") for x in f[8:])}
        cur.append((int(f[0]), int(f[1]), float(f[2]), top))
    return out


def compare():
    ref = rows(f"{S}/bf16.logpos")
    for other in ("bf16-rerun", "req8"):
        oth = rows(f"{S}/{other}.logpos")
        print(f"\nbf16 vs {other}")
        print(f"{'text':<6}{'pos':>6}{'argmax':>9}{'top-10':>8}{'mean KL':>10}{'median KL':>11}{'p99 KL':>9}"
              f"{'ppl bf16':>10}{'ppl other':>11}")
        allkl = []
        for name in TEXTS:
            a = {r[0]: r for r in ref[name]}
            b = {r[0]: r for r in oth[name]}
            common = sorted(set(a) & set(b))
            kls, same, ov, la, lb = [], 0, 0, [], []
            for p in common:
                ta, tb = a[p][3], b[p][3]
                pa = {k: math.exp(v) for k, v in ta.items()}
                qmin = math.exp(min(tb.values()))
                q = {k: (math.exp(tb[k]) if k in tb else qmin) for k in pa}
                prest, qrest = max(1e-12, 1 - sum(pa.values())), max(1e-12, 1 - sum(q.values()))
                kl = sum(p_ * math.log(p_ / max(q[k], 1e-12)) for k, p_ in pa.items() if p_ > 0)
                kl += prest * math.log(prest / qrest)
                kls.append(max(kl, 0.0))
                same += max(ta, key=ta.get) == max(tb, key=tb.get)
                ov += len(set(sorted(ta, key=ta.get, reverse=True)[:10]) & set(sorted(tb, key=tb.get, reverse=True)[:10]))
                la.append(a[p][2])
                lb.append(b[p][2])
            n = len(common)
            kls.sort()
            allkl += kls
            print(f"{name:<6}{n:>6}{100 * same / n:>8.1f}%{10 * ov / n:>7.1f}%{sum(kls) / n:>10.4f}{kls[n // 2]:>11.5f}"
                  f"{kls[int(n * 0.99)]:>9.3f}{math.exp(-sum(la) / n):>10.3f}{math.exp(-sum(lb) / n):>11.3f}")
        allkl.sort()
        n = len(allkl)
        print(f"all {n} positions: mean KL {sum(allkl) / n:.4f}, median {allkl[n // 2]:.5f}, "
              f"p99 {allkl[int(n * 0.99)]:.3f}, max {allkl[-1]:.3f}")


if __name__ == "__main__":
    {"run": run, "compare": compare}[sys.argv[1]]()
