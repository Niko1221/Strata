#!/usr/bin/env python3
"""Escalation sweep beyond K=26: arms 28,30,32,34,36 — stop as soon as an arm is
worse (tps_med < 0.99 x prev) or hit rate < 95% or it fails to serve. Winner = best
tps_med arm (baseline K=26 @ 108.6 from the 3-arm sweep, live now). Leaves the server
on the winner. Same 240 s streaming bench as before (comparable)."""
import json, os, re, shutil, subprocess, time, threading, urllib.request, statistics

HOME = "~"
D = HOME + "/bench_sweep"
JSONF = HOME + "/Strata/strata-unsloth-ud-iq4_xs.json"
LOGF = HOME + "/Strata/strata-unsloth-ud-iq4_xs.log"
URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen3.8-flash-next-unsloth-ud-iq4_xs"
SEQ_S = 240.0
ARMS = [28, 30, 32, 34, 36]
BASE_K, BASE_TPS = 26, 108.6

os.makedirs(D, exist_ok=True)
KEY = ""
for line in open(HOME + "/.config/<env-file>"):
    if line.startswith("STRATA_API_KEY="):
        KEY = line.split("=", 1)[1].strip()
assert len(KEY) > 10, "no strata key"
ASP = D + "/.asp"
open(ASP, "w").write("#!/bin/sh\necho <sudo-password>\n")
os.chmod(ASP, 0o700)
RESULTS = D + "/sweep2_results.json"

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

SENT = "서버 랙의 전원 분배 회로를 점검할 때는 부하 중심과 상쇄 선로 저항을 함께 고려해야 하며, 특히 급전 경로가 서로 다른 계통과 만나면 접지 전위차가 흘러 센서 신호에 오프셋이 생긴다. "

def make_messages(kind, r):
    if kind == "short":
        return [{"role": "user", "content": f"{r % 97}과 {r % 89}을 더한 값과 그 곱을 계산하고, 각 결과가 짝수인지 홀수인지 설명해줘."}], 200
    if kind == "medium":
        return [{"role": "user", "content": "아아 아래 문서의 핵심 주장을 세 줄로 요약해줘.\n\n" + f"[문서 {r}] " + SENT * 12}], 350
    return [{"role": "user", "content": "아아 위 로그에서 전원/접지 관련 경고만 골라 번호 목록으로 정리해줘.\n\n" + f"[로그 {r}] " + SENT * 150}], 250

def stream_chat(messages, max_tokens):
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
    return dict(dur=round(dur, 2), ttft=round(ttft or dur, 2), tok=tok,
                tps=round(tok / gen, 1) if gen > 0.5 and not err else None, err=err)

def run_bench(path):
    plan = ["short", "short", "medium", "medium", "long"]
    stream_chat([{"role": "user", "content": " 인사 ."}], 30)
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

def set_k(k):
    cfg = json.load(open(JSONF))
    i = cfg["args"].index("--layer-split")
    cfg["args"][i + 1] = str(k)
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
    return m[-1] if m else None

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
    log(f"!! RECOVER: {reason} -> restoring K={BASE_K} and restarting")
    shutil.copy2(D + "/json.orig2", JSONF)
    try:
        sh(["systemctl", "restart", "strata-server"], sudo=True, timeout=120)
    except Exception:
        pass
    for _ in range(60):
        if models_loaded():
            log("!! recovered to K=26 config and serving")
            return True
        time.sleep(4)
    log("!! RECOVER FAILED - server not serving!")
    return False

def save(arms, best_k, note):
    json.dump({"arms": arms, "best_k": best_k, "note": note}, open(RESULTS, "w"),
              ensure_ascii=False, indent=1)

def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy2(JSONF, D + "/json.orig2")
    shutil.copy2(JSONF, D + f"/json.bak2-{ts}")
    arms = []
    best_k, best_tps, prev_tps = BASE_K, BASE_TPS, BASE_TPS
    current_k = BASE_K
    note = None

    for k in ARMS:
        log(f"=== ARM K={k} (best so far: K={best_k} @ {best_tps}) ===")
        set_k(k)
        path = D + f"/arm{k}.jsonl"
        open(path, "w").close()
        try:
            off = restart_wait()
            if not live_args_ok(k):
                raise RuntimeError(f"live args missing --layer-split {k}")
            pl = placement(off)
            log(f"  placement={pl}")
            g = Gpu(); g.start()
            recs = run_bench(path)
            g.stop = True; time.sleep(6)
            ok = [r["tps"] for r in recs if r.get("tps")]
            res = {"k": k, "n": len(recs), "errors": sum(1 for r in recs if r["err"]),
                   "tps_med": round(statistics.median(ok), 1) if ok else None,
                   "tps_min": round(min(ok), 1) if ok else None,
                   "ttft_med": round(statistics.median([r["ttft"] for r in recs]), 2),
                   "hit_rate_med": hit_rates(off), "placement": pl,
                   "titan_gpu": gpu_stats(0, g.samples), "gpu3080": gpu_stats(1, g.samples)}
            arms.append(res); current_k = k
            log("ARM DONE " + json.dumps(res, ensure_ascii=False))
        except Exception as e:
            arms.append({"k": k, "failed": str(e)})
            note = f"arm K={k} failed: {e}"
            if not recover(note):
                save(arms, best_k, note + " + RECOVER FAILED")
                return
            current_k = BASE_K
            save(arms, best_k, note)
            break
        save(arms, best_k, None)
        if res["tps_med"] is None or (res["hit_rate_med"] or 0) < 95 or res["tps_med"] < 0.99 * prev_tps:
            note = (f"stop at K={k}: tps={res['tps_med']} hit={res['hit_rate_med']} "
                    f"(prev {prev_tps}) — no further escalation")
            log("STOP: " + note)
            break
        if res["tps_med"] > best_tps:
            best_tps, best_k = res["tps_med"], k
        prev_tps = res["tps_med"]

    log(f"DECISION winner K={best_k} @ {best_tps} t/s; live is K={current_k}")
    if current_k != best_k:
        set_k(best_k)
        try:
            restart_wait()
            log(f"final config live at K={best_k}, args_ok={live_args_ok(best_k)}")
        except Exception as e:
            recover(f"final switch failed: {e}")
            best_k = BASE_K
    else:
        set_k(best_k)
    save(arms, best_k, f"SWEEP2 DONE winner K={best_k} tps={best_tps}")
    log(f"SWEEP2 DONE winner K={best_k} tps={best_tps}")

try:
    main()
except Exception as e:
    log(f"FATAL {e}")
    recover(f"sweep2 fatal: {e}")
finally:
    try:
        os.remove(ASP)
    except OSError:
        pass
