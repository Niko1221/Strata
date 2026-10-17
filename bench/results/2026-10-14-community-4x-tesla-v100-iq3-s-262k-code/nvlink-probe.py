#!/usr/bin/env python3
# Проба: каким путём движок гоняет активации между четырьмя картами при --layer-split.
# Логика: NVLink-байты без root недоступны (DCGM profiling не грузится), но ЛЮБОЙ трафик,
# не пошедший по NVLink, обязан пройти по PCIe-линку каждой карты. Значит достаточно
# посчитать rxpci/txpci по фазам: молчаливый PCIe при длинном префиле = копии идут по NVLink.
# Фазы: покой -> декод -> покой -> префил 32K -> покой -> префил 128K -> покой.
import ast, json, math, random, subprocess, sys, time, urllib.request

HOME = "/home/dgbox"
BENCH = HOME + "/strata185/bin/strata-bench-community.py"
OUT = HOME + "/strata185/bench-community/nvlink-probe"
KEY = "sk-9f3a2b8c7d1e4f6a8b0c2d5e7f4a1b8c"
MODEL = "qwen3.8-flash-next-iq3_s"
URL = "http://127.0.0.1:8080/v1/chat/completions"
RATIO = 0.196                      # токенов на символ, измерено на этой машине

para = next(n.value for n in ast.parse(open(BENCH, encoding="utf-8").read()).body
            if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "PARA_CODE")
para = ast.literal_eval(para)


def prompt_for(target):
    n = max(1, int(target / RATIO / len(para)) + 1)
    return f"[метка {random.randint(10**8, 10**9-1)}]\n" + para * n


def send(prompt, max_tokens):
    body = json.dumps({"model": MODEL, "max_tokens": max_tokens, "temperature": 0,
                       "reasoning_effort": "none",
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Authorization": "Bearer " + KEY,
                                                         "Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return {"wall_s": round(time.time() - t0, 2),
            "prompt_n": d["usage"]["prompt_tokens"], "gen_n": d["usage"]["completion_tokens"]}


BUF = {"rows": []}
def dmon_tail():
    rows = BUF["rows"]
    acc = {i: [0.0, 0.0] for i in range(4)}
    for g, rx, tx in rows:
        acc[g][0] += rx; acc[g][1] += tx
    return {"samples": len(rows) // 4,
            "mb_rx": {g: round(v[0], 1) for g, v in acc.items()},
            "mb_tx": {g: round(v[1], 1) for g, v in acc.items()},
            "peak_rx_mb_s": {g: max([r[1] for r in rows if r[0] == g] or [0]) for g in range(4)},
            "peak_tx_mb_s": {g: max([r[2] for r in rows if r[0] == g] or [0]) for g in range(4)}}


def dmon_capture(seconds):
    """dmon в管道, строки копим в BUF; окно закрывается по времени."""
    BUF["rows"] = []
    p = subprocess.Popen(["nvidia-smi", "dmon", "-s", "t", "-d", "1", "-c", str(seconds)],
                        stdout=subprocess.PIPE, text=True)
    import threading
    def pump():
        for line in p.stdout:
            s = line.split()
            if len(s) == 3 and s[0].isdigit():
                BUF["rows"].append((int(s[0]), float(s[1]), float(s[2])))
    t = threading.Thread(target=pump, daemon=True); t.start()
    return p, t


RES = {}
import pathlib
pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)

def run_phase(name, dur, fn=None):
    print(f"--- {name}: окно {dur} с", flush=True)
    p, t = dmon_capture(dur)
    time.sleep(1)
    info = fn() if fn else {}
    p.wait(); t.join(timeout=5)
    m = dmon_tail(); m.update(info)
    RES[name] = m
    print("    " + json.dumps(m, ensure_ascii=False), flush=True)


run_phase("idle_1", 12)
run_phase("decode", 40, lambda: send(
    "Продолжи перечисление ровно построчно, без пояснений, начиная с 21: " +
    " ".join(str(i) for i in range(1, 21)), 1024))
run_phase("idle_2", 12)
run_phase("prefill_32k", 25, lambda: send(prompt_for(32768), 8))
run_phase("idle_3", 12)
run_phase("prefill_128k", 70, lambda: send(prompt_for(131072), 8))
run_phase("idle_4", 12)

json.dump(RES, open(OUT + "/phases.json", "w"), indent=1, ensure_ascii=False)
print("ИТОГ:", json.dumps({k: {"mb_rx": v["mb_rx"], "mb_tx": v["mb_tx"],
                               "prompt_n": v.get("prompt_n"), "gen_n": v.get("gen_n"),
                               "wall_s": v.get("wall_s")} for k, v in RES.items()},
                          ensure_ascii=False, indent=1))
print("ПРОБА_ЗАВЕРШЕНА")
