#!/usr/bin/env python3
"""3-arm layer_split sweep on the host: A=20 (baseline), B=24, C=26 (only if B holds).
Self-contained: edits the run json via python, restarts strata-server, waits for load,
verifies placement + live args, runs a 240 s streaming bench, samples both GPUs,
parses engine log for expert hit rate, then leaves the server on the winning K.
"""
import json, os, re, shutil, subprocess, sys, threading, time, urllib.request, statistics

HOME = "~"
D = HOME + "/bench_sweep"
JSONF = HOME + "/Strata/strata-unsloth-ud-iq4_xs.json"
LOGF = HOME + "/Strata/strata-unsloth-ud-iq4_xs.log"
URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen3.8-flash-next-unsloth-ud-iq4_xs"
SEQ_S = 240.0

os.makedirs(D, exist_ok=True)
KEY = ""
for line in open(HOME + "/.config/<env-file>"):
    if line.startswith("STRATA_API_KEY="):
        KEY = line.split("=", 1)[1].strip()
assert len(KEY) > 10, "no strata key"
ASP = D + "/.asp"
open(ASP, "w").write("#!/bin/sh\necho <sudo-password>\n")
os.chmod(ASP, 0o700)
RESULTS = D + "/sweep_results.json"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def sh(cmd, sudo=False, timeout=90):
    env = dict(os.environ)
    if sudo:
        env["SUDO_ASKPASS"] = ASP
        cmd = ["sudo", "-A", "-p", ""] + cmd
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)

def models_loaded():
    try:
        req = urllib.request.Request("http://127.0.0.1:8000/v1/models",
                                     headers={"Authorization": f"Bearer {KEY}"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return '"value": "loaded"' in r.read().decode()
    except Exception:
        return False

# ---------- bench core (same workload as the 10-min test, for comparability) ----------
SENT = "서버 랙의 전원 분배 회로를 점검할 때는 부하 중심과 상쇄 선로 저항을 함께 고려해야 하며, 특히 급전 경로가 서로 다른 계통과 만나면 접지 전위차가 흘러 센서 신호에 오프셋이 생긴다. "

def make_messages(kind, r):
    if kind == "short":
        return [{"role": "user", "content": f"{r % 97}과 {r % 89}을 더한 값과 그 곱을 계산하고, 각 결과가 짝수인지 홀수인지 설명해줘."}], 200
    if kind == "medium":
        return [{"role": "user", "content": "아아 아래 문서의 핵심 주장을 세 줄로 요약해줘.\n\n" + f"[문서 {r}] " + SENT * 12}], 350
    return [{"role": "user", "content": "아아 위 로그에서 전원/접지 관련 경고만 골라 번호 목록으로 정리해줘.\n\n" + f"[로그 {r}] " + SENT * 150}], 250

def stream_chat(messages, max_tokens, rec=None):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0.3, "stream": True}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.time(); ttft = None; chunks = 0; chars = 0; err = None
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                p = line[5:].strip()
                if p == "[DONE]":
                    break
                try:
                    d = json.loads(p)["choices"][0]["delta"]
                except Exception:
                    continue
                c = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if c:
                    if ttft is None:
                        ttft = time.time() - t0
                    chunks += 1; chars += len(c)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    dur = time.time() - t0
    tok = max(chunks, round(chars / 3.2))
    gen = dur - (ttft or 0)
    out = rec if rec is not None else {}
    out.update(dur=round(dur, 2), ttft=round(ttft or dur, 2), tok=tok,
               tps=round(tok / gen, 1) if gen > 0.5 and not err else None, err=err)
    return out

def run_bench(path):
    plan = ["short", "short", "medium", "medium", "long"]
    stream_chat([{"role": "user", "content": " 인사 ."}], 30)  # warmup, not counted
    recs = []; t0 = time.time(); i = 0; last = 0
    while time.time() - t0 < SEQ_S:
        kind = plan[i % len(plan)]; i += 1
        msgs, mt = make_messages(kind, i * 137 + 11)
        r = stream_chat(msgs, mt); r["kind"] = kind
        recs.append(r)
        with open(path, "a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if time.time() - last > 40:
            last = time.time()
            ok = [x["tps"] for x in recs if x.get("tps")]
            log(f"  bench n={len(recs)} med={statistics.median(ok) if ok else '-'} err={sum(1 for x in recs if x['err'])}")
    return recs

# ---------- GPU sampler ----------
class Gpu(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.stop = False; self.samples = []
    def run(self):
        while not self.stop:
            out = sh(["nvidia-smi", "--query-gpu=index,power.draw,temperature.gpu,utilization.gpu",
                      "--format=csv,noheader,nounits"]).stdout
            for l in out.strip().splitlines():
                g, p, t, u = [x.strip() for x in l.split(",")]
                self.samples.append((int(g), float(p), float(t), float(u)))
            time.sleep(5)

def gpu_stats(g, samples):
    rows = [s for s in samples if s[0] == g]
    if not rows:
        return {}
    return {"power_max_w": round(max(r[1] for r in rows), 1), "power_avg_w": round(sum(r[1] for r in rows) / len(rows), 1),
            "temp_max_c": round(max(r[2] for r in rows), 1)}

# ---------- json control ----------
def set_k(k):
    cfg = json.load(open(JSONF))
    found = False
    for i, a in enumerate(cfg["args"]):
        if a == "--layer-split":
            cfg["args"][i + 1] = str(k); found = True
    assert found, "--layer-split missing in args"
    if "layer_split" in cfg:
        cfg["layer_split"] = str(k)
    tmp = JSONF + ".tmp"
    json.dump(cfg, open(tmp, "w"), ensure_ascii=False, indent=2)
    os.replace(tmp, JSONF)

def live_args_ok(k):
    out = sh(["bash", "-c", "ps -eo args | grep '[e]ngine/strata --serve' | head -1"]).stdout
    return f"--layer-split {k}" in out

def placement(log_from):
    with open(LOGF, "rb") as f:
        f.seek(log_from)
        txt = f.read().decode("utf8", "replace")
    m = re.findall(r"layer split: layers (\d+-\d+) \(CUDA0\), (\d+-\d+) \(CUDA1\)", txt)
    fr = re.findall(r"CUDA1: (\d+) MiB of VRAM free with everything loaded", txt)
    return {"placement": m[-1] if m else None, "titan_free_mib": int(fr[-1]) if fr else None}

def hit_rates(log_from):
    with open(LOGF, "rb") as f:
        f.seek(log_from)
        txt = f.read().decode("utf8", "replace")
    hr = [float(x) for x in re.findall(r"decode expert cache hit rate: (\d+\.\d)%", txt)]
    return round(statistics.median(hr), 2) if hr else None

def restart_wait():
    off = os.path.getsize(LOGF)
    sh(["systemctl", "restart", "strata-server"], sudo=True, timeout=120)
    t0 = time.time()
    while time.time() - t0 < 240:
        if models_loaded():
            time.sleep(3)
            return off
        time.sleep(4)
    raise RuntimeError("server did not load within 240 s")

def recover(reason):
    log(f"!! RECOVER: {reason} -> restoring original json and restarting")
    shutil.copy2(D + "/json.orig", JSONF)
    try:
        sh(["systemctl", "restart", "strata-server"], sudo=True, timeout=120)
    except Exception:
        pass
    for _ in range(60):
        if models_loaded():
            log("!! original config restored and serving")
            return
        time.sleep(4)
    log("!! RECOVER FAILED - server not serving!")

def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy2(JSONF, D + "/json.orig")
    shutil.copy2(JSONF, D + f"/json.bak-{ts}")
    arms = []
    current_k = 20

    def run_arm(k):
        nonlocal current_k
        log(f"=== ARM K={k}: setting json ===")
        set_k(k)
        path = D + f"/arm{k}.jsonl"
        open(path, "w").close()
        if k == current_k:
            log("K already live (no restart needed for baseline arm)" if k == 20 else "restart")
        try:
            if k != current_k:
                off = restart_wait()
            else:
                off = os.path.getsize(LOGF)
            if not live_args_ok(k):
                raise RuntimeError(f"live args missing --layer-split {k}")
            pl = placement(off)
            log(f"  placement={pl}")
            g = Gpu(); g.start()
            recs = run_bench(path)
            g.stop = True; time.sleep(6)
            ok = [r["tps"] for r in recs if r.get("tps")]
            tt = [r["ttft"] for r in recs]
            res = {"k": k, "n": len(recs), "errors": sum(1 for r in recs if r["err"]),
                   "tps_med": round(statistics.median(ok), 1) if ok else None,
                   "tps_min": round(min(ok), 1) if ok else None,
                   "ttft_med": round(statistics.median(tt), 2),
                   "hit_rate_med": hit_rates(off), "placement": pl.get("placement"),
                   "titan_free_mib": pl.get("titan_free_mib"),
                   "titan_gpu": gpu_stats(0, g.samples), "gpu3080": gpu_stats(1, g.samples)}
            arms.append(res)
            current_k = k
            log("ARM DONE " + json.dumps(res, ensure_ascii=False))
            return res
        except Exception as e:
            log(f"ARM K={k} FAILED: {e}")
            recover(f"arm K={k} failed")
            res = {"k": k, "failed": str(e)}
            arms.append(res)
            current_k = 20
            return res

    a = run_arm(20)
    json.dump({"arms": arms, "final_k": current_k}, open(RESULTS, "w"), ensure_ascii=False, indent=1)
    b = run_arm(24)
    json.dump({"arms": arms, "final_k": current_k}, open(RESULTS, "w"), ensure_ascii=False, indent=1)

    def holds(x, base):
        return x.get("tps_med") and base.get("tps_med") and x["tps_med"] >= 0.93 * base["tps_med"] and not x.get("failed")

    if holds(b, a) and (b.get("hit_rate_med") or 100) >= 95:
        c = run_arm(26)
    else:
        c = None
        log("K=26 arm skipped (K=24 did not hold up)")

    base = a["tps_med"] if a.get("tps_med") else None
    good = [x for x in arms if x.get("tps_med") and x["tps_med"] >= 0.95 * base and not x.get("failed")]
    winner = max([x["k"] for x in good], default=20)
    log(f"DECISION base={base} winner K={winner} (arms: "
        + ", ".join(f"{x['k']}:{x.get('tps_med')}" for x in arms) + ")")
    if current_k != winner:
        set_k(winner)
        try:
            restart_wait()
            log(f"final config live at K={winner}, args_ok={live_args_ok(winner)}")
        except Exception as e:
            recover(f"final switch to K={winner} restart failed: {e}")
    else:
        # make sure json is at winner too (e.g. all arms failed path)
        set_k(winner)
    json.dump({"arms": arms, "final_k": winner, "decision": "SWEEP DONE"},
              open(RESULTS, "w"), ensure_ascii=False, indent=1)
    log("SWEEP DONE")

try:
    main()
except Exception as e:
    log(f"FATAL {e}")
    recover(f"sweep fatal: {e}")
finally:
    try:
        os.remove(ASP)
    except OSError:
        pass
