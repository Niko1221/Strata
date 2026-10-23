#!/usr/bin/env python3
"""Endurance test at K=32: identical 10-min protocol to the morning baseline
(seq 540 s + 4-way burst 90 s + tail, same seed/prompts) so results compare 1:1
against the K=20 morning run (median 99.8, burst agg 53.4, tail 97.9, 0 errors).
Pre-registered decision: KEEP K=32 iff seq median >= 108 AND expert-hit median >= 93.0
AND 0 errors AND tail tps >= 90 — otherwise restore K=30 (json.orig3)."""
import json, os, re, shutil, subprocess, threading, time, urllib.request, statistics

HOME = "~"
D = HOME + "/bench_sweep"
JSONF = HOME + "/Strata/strata-unsloth-ud-iq4_xs.json"
LOGF = HOME + "/Strata/strata-unsloth-ud-iq4_xs.log"
URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen3.8-flash-next-unsloth-ud-iq4_xs"
SEQ_S, BURST_S, N = 540.0, 90.0, 4
K = 32

os.makedirs(D, exist_ok=True)
KEY = ""
for line in open(HOME + "/.config/<env-file>"):
    if line.startswith("STRATA_API_KEY="):
        KEY = line.split("=", 1)[1].strip()
assert len(KEY) > 10
ASP = D + "/.asp"
open(ASP, "w").write("#!/bin/sh\necho <sudo-password>\n")
os.chmod(ASP, 0o700)
RESULTS = D + "/sweep3_results.json"
OUT = D + "/endurance32.jsonl"

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
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
                if not line.startswith("data:"): continue
                p = line[5:].strip()
                if p == "[DONE]": break
                try: d = json.loads(p)["choices"][0]["delta"]
                except Exception: continue
                c = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if c:
                    if ttft is None: ttft = time.time() - t0
                    chunks += 1; chars += len(c)
    except Exception as e: err = f"{type(e).__name__}: {e}"
    dur = time.time() - t0
    tok = max(chunks, round(chars / 3.2)); gen = dur - (ttft or 0)
    return dict(dur=round(dur, 2), ttft=round(ttft or dur, 2), tok=tok,
                tps=round(tok / gen, 1) if gen > 0.5 and not err else None, err=err)

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
def gsum(gpu, gobj):
    rows = [s for s in gpu.samples if s[0] == gobj]
    if not rows: return {}
    n = len(rows)
    return {"power_max_w": round(max(r[1] for r in rows),1), "power_avg_w": round(sum(r[1] for r in rows)/n,1),
            "temp_max_c": round(max(r[2] for r in rows),1), "util_avg": round(sum(r[3] for r in rows)/n,1)}

def set_k(k):
    cfg = json.load(open(JSONF))
    i = cfg["args"].index("--layer-split"); cfg["args"][i+1] = str(k)
    if "layer_split" in cfg: cfg["layer_split"] = str(k)
    tmp = JSONF + ".tmp"; json.dump(cfg, open(tmp,"w"), ensure_ascii=False, indent=2); os.replace(tmp, JSONF)
def live_args_ok(k):
    return f"--layer-split {k}" in sh(["bash","-c","ps -eo args | grep '[e]ngine/strata --serve' | head -1"]).stdout
def restart_wait():
    off = os.path.getsize(LOGF)
    sh(["systemctl","restart","strata-server"], sudo=True, timeout=120)
    t0 = time.time()
    while time.time() - t0 < 240:
        if models_loaded():
            time.sleep(3); return off
        time.sleep(4)
    raise RuntimeError("load timeout")
def hit_rates(off):
    with open(LOGF,"rb") as f:
        f.seek(off); txt = f.read().decode("utf8","replace")
    hr = [float(x) for x in re.findall(r"decode expert cache hit rate: (\d+\.\d)%", txt)]
    return (round(statistics.median(hr),2), len(hr)) if hr else (None,0)

def restore_k30():
    log("!! restoring K=30 from json.orig3")
    shutil.copy2(D + "/json.orig3", JSONF)
    sh(["systemctl","restart","strata-server"], sudo=True, timeout=120)
    for _ in range(60):
        if models_loaded(): log("!! K=30 restored and serving"); return
        time.sleep(4)
    log("!! RESTORE FAILED")

def main():
    shutil.copy2(JSONF, D + "/json.orig3")
    log(f"=== ENDURANCE K={K}: setting json (restore point saved as K=30) ===")
    set_k(K)
    off = restart_wait()
    if not live_args_ok(K): raise RuntimeError("live args missing --layer-split 32")
    open(OUT, "w").close()
    gpu = Gpu(); gpu.start()
    t0 = time.time(); last = 0.0; seq = []
    plan = ["short","short","medium","medium","long"]; i = 0
    while time.time() - t0 < SEQ_S:
        kind = plan[i % len(plan)]; i += 1
        msgs, mt = make_messages(kind, i * 137 + 11)
        r = stream_chat(msgs, mt); r["kind"] = kind; seq.append(r)
        with open(OUT,"a") as f: f.write(json.dumps(r, ensure_ascii=False)+"\n")
        if time.time() - last > 45:
            last = time.time(); ok=[x["tps"] for x in seq if x.get("tps")]
            log(f"  seq n={len(seq)} med={round(statistics.median(ok),1) if ok else '-'} err={sum(1 for x in seq if x['err'])} elapsed={time.time()-t0:.0f}s")
    log(f"[PHASE] seq done n={len(seq)} -> burst {N}-way {BURST_S:.0f}s")
    burst = []; lock = threading.Lock(); bt0 = time.time()
    def worker(w):
        j = 0
        while time.time() - bt0 < BURST_S:
            j += 1
            msgs, mt = make_messages("short", (w+1)*1000 + j*7)
            r = stream_chat(msgs, 120); r["worker"] = w
            with lock: burst.append(r)
    ths = [threading.Thread(target=worker, args=(k,)) for k in range(N)]
    [t.start() for t in ths]; [t.join() for t in ths]
    bdur = time.time() - bt0
    with open(OUT,"a") as f:
        for r in burst: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    tail = stream_chat(*make_messages("short", 777))
    gpu.stop = True; time.sleep(6)
    ok = [r["tps"] for r in seq if r.get("tps")]; bok = [r["tps"] for r in burst if r.get("tps")]
    hr, hn = hit_rates(off)
    res = {"k": K, "seq_n": len(seq), "errors": sum(1 for r in seq if r["err"]) + sum(1 for r in burst if r["err"]),
           "seq_tps_med": round(statistics.median(ok),1) if ok else None,
           "seq_tps_min": round(min(ok),1) if ok else None,
           "seq_tps_last20": round(statistics.median([x["tps"] for x in seq[-20:] if x.get("tps")]),1),
           "ttft_med": round(statistics.median([r["ttft"] for r in seq]),2),
           "hit_med": hr, "hit_samples": hn,
           "burst_agg_tps": round(sum(r["tok"] for r in burst)/bdur,1) if bok else None,
           "tail_tps": tail.get("tps"), "tail_ttft": tail.get("ttft"),
           "titan": gsum(gpu,0), "g3080": gsum(gpu,1)}
    keep = (res["seq_tps_med"] or 0) >= 108 and (res["hit_med"] or 0) >= 93.0 and res["errors"] == 0 and (res["tail_tps"] or 0) >= 90
    res["decision"] = "KEEP_K32" if keep else "REVERT_K30"
    json.dump(res, open(RESULTS,"w"), ensure_ascii=False, indent=1)
    log("ARM DONE " + json.dumps(res, ensure_ascii=False))
    if keep:
        log("SWEEP3 DONE KEEP K=32 (endurance passed)")
    else:
        restore_k30()
        log("SWEEP3 DONE REVERTED K=30 (endurance failed)")

try:
    main()
except Exception as e:
    log(f"FATAL {e}")
    restore_k30()
    log("SWEEP3 DONE CRASHED-REVERTED K=30")
finally:
    try: os.remove(ASP)
    except OSError: pass
