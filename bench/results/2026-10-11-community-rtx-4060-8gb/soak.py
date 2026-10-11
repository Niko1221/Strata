"""Strata 穩定性測試（第 6 步）：連續一筆接一筆送請求，時間到才停。

    python soak.py --url http://127.0.0.1:8080 --tag coder --minutes 60

請求組合（輪流）：短問答（非串流）、中長度摘要（串流）、JSON 輸出、每 10 筆一次「串流中途斷線」。
每 30 秒記一次 nvidia-smi（VRAM／溫度／功耗）＋ /metrics（RAM／狀態）＋ server 程序 RSS。
輸出：_strata_test/06_soak_<tag>.jsonl（每筆請求一行）、06_soak_<tag>_hw.jsonl、06_soak_<tag>_summary.json
通過條件：server 沒掛、沒有 5xx、RAM 沒有單調上升、首尾 10 分鐘的生成 tok/s 中位數差 < 15%。
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import threading
import time
from pathlib import Path

import requests

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8080")
ap.add_argument("--tag", default="run")
ap.add_argument("--minutes", type=float, default=60)
A = ap.parse_args()
URL = A.url.rstrip("/")
OUT = Path(r"E:\Strata\_strata_test")
req_log = OUT / f"06_soak_{A.tag}.jsonl"
hw_log = OUT / f"06_soak_{A.tag}_hw.jsonl"
for p in (req_log, hw_log):
    p.write_text("", encoding="utf-8")
stop = threading.Event()
DOC = Path(r"E:\Strata\docs\DETAILS.md").read_text(encoding="utf-8", errors="replace")

TOPICS = ["本益比", "自由現金流", "毛利率", "營業槓桿", "股東權益報酬率", "庫藏股", "可轉債", "存貨週轉率",
          "應收帳款天數", "EV/EBITDA", "股利殖利率", "資本支出", "研發費用資本化", "商譽減損", "毛利率擴張"]


def append(path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def server_rss_mb():
    """strata.exe＋python server 的 RSS 總和（MB）。"""
    try:
        import psutil
        tot = 0
        for p in psutil.process_iter(["name", "cmdline", "memory_info"]):
            n = (p.info["name"] or "").lower()
            cl = " ".join(p.info["cmdline"] or []).lower()
            if n.startswith("strata") or ("serve.server" in cl):
                tot += p.info["memory_info"].rss
        return round(tot / 2**20)
    except Exception:  # noqa: BLE001
        return None


def hw_sampler():
    while not stop.is_set():
        row = {"t": round(time.time(), 1)}
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,temperature.gpu,power.draw,utilization.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
            v = []
            for x in q.strip().split(","):
                try:
                    v.append(float(x))
                except ValueError:          # 這張 4060 的 power.draw 回 [N/A]
                    v.append(None)
            row.update(vram_mib=v[0], temp_c=v[1], power_w=v[2], util=v[3])
        except Exception as e:  # noqa: BLE001
            row["nvsmi_err"] = str(e)[:80]
        try:
            m = requests.get(URL + "/metrics", timeout=10).json()
            hw = m.get("hardware") or {}
            row.update(ram_used=hw.get("ram_used"), state=(m.get("live") or {}).get("state"))
        except Exception as e:  # noqa: BLE001
            row["metrics_err"] = str(e)[:80]
        row["server_rss_mb"] = server_rss_mb()
        append(hw_log, row)
        stop.wait(30)


def one(i):
    kind = "cancel" if i % 10 == 9 else ["short", "medium_stream", "json"][i % 3]
    t0 = time.time()
    rec = {"i": i, "kind": kind, "t": round(t0, 1)}
    try:
        if kind == "short":
            r = requests.post(URL + "/v1/chat/completions", timeout=1800, json={
                "model": "strata", "max_tokens": 200, "reasoning_effort": "none",
                "messages": [{"role": "user", "content": f"用三句話解釋「{random.choice(TOPICS)}」，繁體中文。"}]})
            rec["status"] = r.status_code
            j = r.json()
            rec.update(usage=j.get("usage"), timings=j.get("timings"),
                       ok=r.status_code == 200 and bool(j["choices"][0]["message"].get("content")))
        elif kind == "json":
            r = requests.post(URL + "/v1/chat/completions", timeout=1800, json={
                "model": "strata", "max_tokens": 200, "reasoning_effort": "none", "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": f"用 JSON 回答 {{\"term\": \"...\", \"one_line\": \"...\"}}：{random.choice(TOPICS)}"}]})
            rec["status"] = r.status_code
            j = r.json()
            txt = j["choices"][0]["message"].get("content") or ""
            json.loads(txt)
            rec.update(usage=j.get("usage"), timings=j.get("timings"), ok=r.status_code == 200)
        else:
            start = random.randint(0, max(0, len(DOC) - 9000))
            body = {"model": "strata", "max_tokens": 300 if kind != "cancel" else 2000, "reasoning_effort": "none",
                    "stream": True, "messages": [{"role": "user", "content": "Summarize in 3 bullets:\n\n" + DOC[start:start + 8000]}]}
            r = requests.post(URL + "/v1/chat/completions", json=body, timeout=1800, stream=True)
            rec["status"] = r.status_code
            n, last, ttft = 0, None, None
            for raw in r.iter_lines():
                if not raw or not raw.startswith(b"data:") or raw.endswith(b"[DONE]"):
                    continue
                n += 1
                if ttft is None:
                    ttft = time.time() - t0
                last = json.loads(raw[5:])
                if kind == "cancel" and n >= 20:
                    r.close()
                    break
            rec.update(chunks=n, ttft_s=round(ttft or -1, 2), ok=n > 0)
            if last and kind != "cancel":
                rec.update(usage=last.get("usage"), timings=last.get("timings"))
    except Exception as e:  # noqa: BLE001
        rec.update(ok=False, error=f"{type(e).__name__}: {e}"[:300])
    rec["seconds"] = round(time.time() - t0, 1)
    append(req_log, rec)
    return rec


threading.Thread(target=hw_sampler, daemon=True).start()
deadline = time.time() + A.minutes * 60
t_start = time.time()
i = 0
print(f"soak {A.tag}: {A.minutes} min @ {URL}", flush=True)
while time.time() < deadline:
    rec = one(i)
    tm = rec.get("timings") or {}
    print(f"{(time.time() - t_start) / 60:5.1f}m #{i:<4} {rec['kind']:13} ok={rec.get('ok')} st={rec.get('status')} "
          f"{rec['seconds']:6.1f}s dec={tm.get('predicted_per_second')} {rec.get('error', '')}", flush=True)
    i += 1
    if not rec.get("ok"):
        try:
            requests.get(URL + "/health", timeout=10)
        except Exception:  # noqa: BLE001
            print("server not answering /health — stopping", flush=True)
            break
stop.set()

# ---- 統計
reqs = [json.loads(x) for x in req_log.read_text(encoding="utf-8").splitlines() if x.strip()]
hws = [json.loads(x) for x in hw_log.read_text(encoding="utf-8").splitlines() if x.strip()]
t_end = time.time()
dur = t_end - t_start


def dec_in(lo, hi):
    return [r["timings"]["predicted_per_second"] for r in reqs if r.get("timings") and
            r["timings"].get("predicted_per_second") and lo <= r["t"] - t_start < hi]


first10, last10 = dec_in(0, 600), dec_in(max(0, dur - 600), dur + 1)
med = lambda xs: round(statistics.median(xs), 2) if xs else None  # noqa: E731
rss = [h["server_rss_mb"] for h in hws if h.get("server_rss_mb")]
ram = [h["ram_used"] for h in hws if h.get("ram_used")]
drift = None
if first10 and last10:
    drift = round((med(last10) - med(first10)) / med(first10) * 100, 1)
monotonic = None
if len(rss) >= 6:
    q = len(rss) // 4
    monotonic = statistics.mean(rss[-q:]) - statistics.mean(rss[:q])   # 最後 1/4 減最前 1/4 的平均（MB）
summary = {
    "tag": A.tag, "minutes": round(dur / 60, 1), "requests": len(reqs), "ok": sum(bool(r.get("ok")) for r in reqs),
    "http_5xx": sum(1 for r in reqs if (r.get("status") or 0) >= 500), "errors": [r for r in reqs if not r.get("ok")][:10],
    "decode_tok_s_median_first10": med(first10), "decode_tok_s_median_last10": med(last10), "decode_drift_pct": drift,
    "server_rss_mb_first_quarter_to_last_quarter_delta": round(monotonic, 1) if monotonic is not None else None,
    "server_rss_mb_min_max": [min(rss), max(rss)] if rss else None,
    "ram_used_gib_min_max": [round(min(ram) / 2**30, 1), round(max(ram) / 2**30, 1)] if ram else None,
    "gpu_temp_max_c": max((h.get("temp_c") or 0) for h in hws) if hws else None,
    "vram_mib_max": max((h.get("vram_mib") or 0) for h in hws) if hws else None,
}
summary["pass"] = bool(summary["requests"] and summary["ok"] == summary["requests"] and summary["http_5xx"] == 0
                       and (drift is None or abs(drift) < 15))
(OUT / f"06_soak_{A.tag}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print("== SOAK SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "errors"}, ensure_ascii=False), flush=True)
