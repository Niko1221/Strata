#!/usr/bin/env python3
# Насыщен ли общий x8-аплинк при длинном префиле. Прежние числа были максимумами по каждой
# карте за окно - они не говорят, складываются ли пики в одну секунду. Здесь: сырой ряд
# nvidia-smi dmon -s t с шагом 1 с, и по нему - сумма по четырём картам в каждую секунду.
import ast, json, random, subprocess, sys, threading, time, urllib.request

HOME = "/home/dgbox"
BENCH = HOME + "/strata185/bin/strata-bench-community.py"
OUT = HOME + "/strata185/bench-community/nvlink-full/uplink-saturation.json"
KEY = "sk-9f3a2b8c7d1e4f6a8b0c2d5e7f4a1b8c"
MODEL = "qwen3.8-flash-next-iq3_s"
URL = "http://127.0.0.1:8080/v1/chat/completions"
RATIO = 0.196

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
    return {"wall_s": round(time.time() - t0, 2), "prompt_n": d["usage"]["prompt_tokens"],
            "gen_n": d["usage"]["completion_tokens"]}


rows = []
def pump(p):
    for line in p.stdout:
        s = line.split()
        if len(s) == 3 and s[0].isdigit():
            rows.append((int(s[0]), float(s[1]), float(s[2])))

print("--- окно 75 с: префил ~125K токенов", flush=True)
p = subprocess.Popen(["nvidia-smi", "dmon", "-s", "t", "-d", "1", "-c", "75"], stdout=subprocess.PIPE, text=True)
t = threading.Thread(target=pump, args=(p,), daemon=True); t.start()
time.sleep(1)
info = send(prompt_for(131072), 8)
p.wait(); t.join(timeout=5)

# склеиваем сэмплы в секунды: dmon выдаёт по 4 строки на интервал
secs, cur = [], []
for g, rx, tx in rows:
    cur.append((g, rx, tx))
    if len(cur) >= 4:
        secs.append(cur); cur = []
per_sec = [{"rx_sum": round(sum(r[1] for r in s) / 1000.0, 2),
            "tx_sum": round(sum(r[2] for r in s) / 1000.0, 2),
            "rx_max_card": round(max(r[1] for r in s) / 1000.0, 2)} for s in secs]
rx_sums = sorted(x["rx_sum"] for x in per_sec)
tx_sums = sorted(x["tx_sum"] for x in per_sec)
res = {
    "request": info,
    "samples_seconds": len(per_sec),
    "uplink_note": "Gen3 x8 = 7.88 GB/s; четыре карты делят один аплинк",
    "rx_sum_gbps": {"max": rx_sums[-1], "p95": rx_sums[int(len(rx_sums)*0.95)-1], "median": rx_sums[len(rx_sums)//2]},
    "tx_sum_gbps": {"max": tx_sums[-1], "p95": tx_sums[int(len(tx_sums)*0.95)-1], "median": tx_sums[len(tx_sums)//2]},
    "per_card_peak_gbps": {g: round(max([r[1] for r in rows if r[0] == g] or [0]) / 1000.0, 2) for g in range(4)},
    "seconds_above_6_gbps_sum": sum(1 for x in per_sec if x["rx_sum"] >= 6.0),
    "series": per_sec,
}
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k != "series"}, indent=1, ensure_ascii=False))
print("ЗАВЕРШЕНО")
