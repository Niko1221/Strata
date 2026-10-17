"""Strata API 功能＋速度測試（第 4 步）。

    python api_suite.py --url http://127.0.0.1:8080 --tag coder --image <png> [--effort-max-tokens 2048] [--only a,b]

每項測試記 pass/fail、耗時、備註；速度項記 engine timings（prompt_per_second / predicted_per_second / cache_n）
＋用戶端 TTFT 與總時間；背景每 2 秒抓一次 nvidia-smi 與 /metrics 硬體數字。
輸出：_strata_test/04_api_<tag>.json（全部原始資料）＋終端機摘要。
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

OUT_DIR = Path(r"E:\Strata\_strata_test")
T_LONG = 1800  # 8 GB 卡很慢：單一請求最多等 30 分鐘

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8080")
ap.add_argument("--tag", default="run")
ap.add_argument("--image", default=None)
ap.add_argument("--effort-max-tokens", type=int, default=2048)
ap.add_argument("--speed-runs", type=int, default=3)
ap.add_argument("--long-chars", type=int, default=30000, help="長 prompt 的字元數（英文約 4 字元/token → ~8k token）")
ap.add_argument("--only", default="", help="逗號分隔，只跑這些測試名稱")
ap.add_argument("--skip", default="", help="逗號分隔，跳過這些測試名稱")
A = ap.parse_args()
URL = A.url.rstrip("/")
ONLY = {x for x in A.only.split(",") if x}
SKIP = {x for x in A.skip.split(",") if x}

results: list[dict] = []
samples: list[dict] = []
_stop = threading.Event()


# ------------------------------------------------------------------ 硬體取樣
def _num(x):
    """nvidia-smi 讀不到的欄位回 [N/A]（這張 4060 的 power.draw 就是）→ None。"""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def sampler():
    while not _stop.is_set():
        row = {"t": round(time.time(), 1)}
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,"
                                "power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                               timeout=10).stdout.strip().split(",")
            v = [_num(x) for x in q]
            row.update(vram_used_mib=v[0], vram_total_mib=v[1], gpu_util=v[2], gpu_temp_c=v[3], gpu_power_w=v[4])
        except Exception as e:  # noqa: BLE001
            row["nvsmi_err"] = str(e)[:80]
        try:
            m = requests.get(URL + "/metrics", timeout=5).json()
            hw = m.get("hardware") or {}
            row.update(ram_used=hw.get("ram_used"), ram_total=hw.get("ram_total"), cpu=hw.get("cpu"),
                       live_state=(m.get("live") or {}).get("state"), live_tok_s=(m.get("live") or {}).get("tok_s"))
        except Exception as e:  # noqa: BLE001
            row["metrics_err"] = str(e)[:80]
        samples.append(row)
        _stop.wait(2)


# ------------------------------------------------------------------ 工具
def oa(body, stream=False, timeout=T_LONG):
    body = {"model": "strata", **body}
    if stream:
        body["stream"] = True
    return requests.post(URL + "/v1/chat/completions", json=body, timeout=timeout, stream=stream)


def an(body, stream=False, timeout=T_LONG):
    body = {"model": "strata", **body}
    if stream:
        body["stream"] = True
    return requests.post(URL + "/v1/messages", json=body, timeout=timeout, stream=stream,
                         headers={"anthropic-version": "2023-06-01"})


def sse_lines(r):
    for raw in r.iter_lines(decode_unicode=False):
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace")
        if line.startswith("data:"):
            yield line[5:].strip()
        elif line.startswith("event:"):
            yield "EVENT " + line[6:].strip()


def oa_stream(body, timeout=T_LONG):
    """串流送出；回傳 (content, reasoning, ttft_s, total_s, last_chunk, n_chunks)。"""
    t0 = time.time()
    r = oa(body, stream=True, timeout=timeout)
    r.raise_for_status()
    content, reasoning, ttft, last, n = [], [], None, None, 0
    for d in sse_lines(r):
        if d.startswith("EVENT ") or d == "[DONE]":
            continue
        j = json.loads(d)
        n += 1
        last = j
        for ch in j.get("choices") or []:
            delta = ch.get("delta") or {}
            if (delta.get("content") or delta.get("reasoning_content")) and ttft is None:
                ttft = time.time() - t0
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or "")
    return "".join(content), "".join(reasoning), ttft, time.time() - t0, last, n


def choice0(r):
    """回傳 (json, choices[0])；伺服器回錯誤（沒有 choices）時把錯誤本文當失敗原因。"""
    j = r.json()
    if "choices" not in j:
        raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(j, ensure_ascii=False)[:300]}")
    return j, j["choices"][0]


def test(name):
    def deco(fn):
        if (ONLY and name not in ONLY) or name in SKIP:
            return fn
        t0 = time.time()
        rec = {"name": name}
        try:
            out = fn() or {}
            rec.update(out)
            rec.setdefault("ok", True)
        except Exception as e:  # noqa: BLE001
            rec.update(ok=False, error=f"{type(e).__name__}: {e}"[:600])
        rec["seconds"] = round(time.time() - t0, 1)
        results.append(rec)
        flag = "PASS" if rec.get("ok") else "FAIL"
        note = rec.get("note") or rec.get("error") or ""
        print(f"[{flag}] {name:28} {rec['seconds']:7.1f}s  {str(note)[:150]}", flush=True)
        (OUT_DIR / f"04_api_{A.tag}.json").write_text(json.dumps({"results": results, "samples": samples},
                                                                  ensure_ascii=False, indent=1), encoding="utf-8")
        return fn
    return deco


# ------------------------------------------------------------------ 開始
threading.Thread(target=sampler, daemon=True).start()
print(f"== Strata API suite @ {URL}  tag={A.tag}", flush=True)
HEALTH = requests.get(URL + "/health", timeout=30).json()
print("health:", HEALTH, flush=True)
MAXCTX = int(HEALTH.get("max_context") or 32768)


@test("health")
def _():
    h = requests.get(URL + "/health", timeout=30).json()
    return {"ok": h.get("status") == "ok", "note": h, "health": h}


@test("models")
def _():
    j = requests.get(URL + "/v1/models", timeout=30).json()
    ids = [m.get("id") for m in j.get("data", [])]
    return {"ok": bool(ids), "note": ids, "raw": j}


@test("metrics_json")
def _():
    j = requests.get(URL + "/metrics", timeout=30).json()
    return {"ok": "engine" in j and "hardware" in j, "note": {"engine": j.get("engine"), "hw_static": j.get("hardware_static")},
            "engine": j.get("engine"), "hardware_static": j.get("hardware_static")}


@test("metrics_prometheus")
def _():
    r = requests.get(URL + "/metrics", headers={"Accept": "text/plain"}, timeout=30)
    lines = [x for x in r.text.splitlines() if x and not x.startswith("#")]
    return {"ok": r.status_code == 200 and len(lines) > 5, "note": f"{len(lines)} metric lines"}


@test("oa_nonstream_zh")
def _():
    t0 = time.time()
    r = oa({"messages": [{"role": "user", "content": "用兩句話說明什麼是本益比（P/E），用繁體中文。"}],
            "max_tokens": 400, "reasoning_effort": "none"})
    j = r.json()
    msg = j["choices"][0]["message"]
    txt = msg.get("content") or ""
    return {"ok": r.status_code == 200 and len(txt) > 10, "note": txt[:120].replace("\n", " "),
            "usage": j.get("usage"), "timings": j.get("timings"), "finish": j["choices"][0].get("finish_reason"),
            "wall_s": round(time.time() - t0, 1), "content": txt}


@test("oa_stream_zh")
def _():
    c, rs, ttft, tot, last, n = oa_stream({"messages": [{"role": "user", "content": "列出三個評估成長股的指標，每個一行，繁體中文。"}],
                                           "max_tokens": 400, "reasoning_effort": "none"})
    return {"ok": len(c) > 10 and n > 3, "note": f"ttft={ttft and round(ttft, 1)}s chunks={n} | {c[:90]!r}",
            "ttft_s": ttft, "total_s": tot, "usage": (last or {}).get("usage"), "timings": (last or {}).get("timings"),
            "content": c}


EFFORT_Q = ("一檔股票先跌 50%，之後要漲多少 % 才能回到原價？再算：若之後每年漲 15%，大約要幾年？"
            "最後一行只寫『答案：X%，約 Y 年』。")
for eff in ("none", "low", "medium", "high"):
    @test(f"oa_effort_{eff}")
    def _(eff=eff):
        c, rs, ttft, tot, last, n = oa_stream({"messages": [{"role": "user", "content": EFFORT_Q}],
                                               "max_tokens": A.effort_max_tokens, "reasoning_effort": eff})
        fin = ((last or {}).get("choices") or [{}])[0].get("finish_reason")
        right = "100" in c
        return {"ok": bool(c.strip()) and right, "note": f"finish={fin} think_chars={len(rs)} ttft={ttft and round(ttft,1)}s "
                f"total={tot:.0f}s 100%? {right} | {c.strip()[-60:]!r}",
                "finish": fin, "reasoning_chars": len(rs), "ttft_s": ttft, "total_s": tot,
                "usage": (last or {}).get("usage"), "timings": (last or {}).get("timings"), "content": c,
                "reasoning": rs[:4000]}


@test("oa_json_object")
def _():
    r = oa({"messages": [
        {"role": "system", "content": "你是財經新聞分類器。只輸出 JSON：{\"sentiment\": \"positive|negative|neutral\", "
                                      "\"tickers\": [..], \"reason\": \"一句話\"}"},
        {"role": "user", "content": "Nvidia shares jump 6% after record data-center revenue; AMD slips 2%."}],
        "max_tokens": 300, "reasoning_effort": "none", "response_format": {"type": "json_object"}})
    j, ch = choice0(r)
    txt = ch["message"].get("content") or ""
    obj = json.loads(txt)
    return {"ok": isinstance(obj, dict) and "sentiment" in obj, "note": obj, "content": txt}


@test("oa_json_schema")
def _():
    schema = {"type": "object", "properties": {"ticker": {"type": "string"}, "action": {"type": "string",
              "enum": ["buy", "hold", "sell"]}, "confidence": {"type": "number"}},
              "required": ["ticker", "action", "confidence"], "additionalProperties": False}
    r = oa({"messages": [{"role": "user", "content": "TSM 營收連三季成長、毛利率創新高。給一個 JSON 判斷。"}],
            "max_tokens": 300, "reasoning_effort": "none",
            "response_format": {"type": "json_schema", "json_schema": {"name": "call", "schema": schema, "strict": True}}})
    j, ch = choice0(r)
    txt = ch["message"].get("content") or ""
    obj = json.loads(txt)
    ok = set(obj) == {"ticker", "action", "confidence"} and obj["action"] in ("buy", "hold", "sell")
    return {"ok": ok, "note": obj, "content": txt}


TOOLS = [{"type": "function", "function": {"name": "get_stock_price", "description": "取得美股即時股價",
          "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}}}]


@test("oa_tool_call")
def _():
    msgs = [{"role": "user", "content": "NVDA 現在股價多少？請用工具查。"}]
    r = oa({"messages": msgs, "tools": TOOLS, "max_tokens": 600, "reasoning_effort": "none"})
    j, ch = choice0(r)
    calls = ch["message"].get("tool_calls") or []
    if not calls:
        return {"ok": False, "note": f"no tool_calls, finish={ch.get('finish_reason')} | {ch['message'].get('content')!r:.120}"}
    args = json.loads(calls[0]["function"]["arguments"])
    msgs += [ch["message"], {"role": "tool", "tool_call_id": calls[0].get("id", "call_0"),
                             "content": json.dumps({"ticker": "NVDA", "price": 187.42, "currency": "USD"})}]
    r2 = oa({"messages": msgs, "tools": TOOLS, "max_tokens": 300, "reasoning_effort": "none"})
    final = choice0(r2)[1]["message"].get("content") or ""
    ok = calls[0]["function"]["name"] == "get_stock_price" and str(args.get("ticker", "")).upper() == "NVDA" and "187" in final
    return {"ok": ok, "note": f"finish={ch.get('finish_reason')} args={args} | final={final[:80]!r}", "final": final}


@test("oa_stream_cancel")
def _():
    body = {"messages": [{"role": "user", "content": "從 1 開始一行一個數字往上數到 2000。"}], "max_tokens": 4000,
            "reasoning_effort": "none"}
    r = oa(body, stream=True)
    n = 0
    for d in sse_lines(r):
        if d.startswith("{"):
            n += 1
        if n >= 15:
            break
    r.close()
    t0 = time.time()
    state = None
    while time.time() - t0 < 120:
        state = (requests.get(URL + "/metrics", timeout=10).json().get("live") or {}).get("state")
        if state == "idle":
            break
        time.sleep(0.5)
    dt = time.time() - t0
    return {"ok": state == "idle" and dt < 30, "note": f"closed after {n} chunks → idle in {dt:.1f}s", "idle_after_s": dt}


@test("an_nonstream_nothink")
def _():
    r = an({"max_tokens": 400, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "一句話：ETF 跟個股最大的差別是什麼？繁體中文。"}]})
    j = r.json()
    texts = [b.get("text", "") for b in j.get("content", []) if b.get("type") == "text"]
    kinds = [b.get("type") for b in j.get("content", [])]
    return {"ok": r.status_code == 200 and any(t.strip() for t in texts) and "thinking" not in kinds,
            "note": f"blocks={kinds} stop={j.get('stop_reason')} | {''.join(texts)[:90]!r}", "usage": j.get("usage")}


@test("an_stream_think_budget")
def _():
    t0 = time.time()
    r = an({"max_tokens": 2500, "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [{"role": "user", "content": "某公司 EPS 4 元、本益比 25 倍，若 EPS 成長 20% 且本益比不變，股價變多少？"}]},
           stream=True)
    events, thinking, text, ttft = [], [], [], None
    for d in sse_lines(r):
        if d.startswith("EVENT "):
            events.append(d[6:])
            continue
        j = json.loads(d)
        if j.get("type") == "content_block_delta":
            dl = j.get("delta") or {}
            if ttft is None:
                ttft = time.time() - t0
            thinking.append(dl.get("thinking") or "")
            text.append(dl.get("text") or "")
    txt = "".join(text)
    kinds = sorted(set(events))
    ok = "message_start" in kinds and "message_stop" in kinds and "120" in txt
    return {"ok": ok, "note": f"events={kinds} think_chars={len(''.join(thinking))} 120? {'120' in txt} | {txt.strip()[-70:]!r}",
            "ttft_s": ttft, "total_s": time.time() - t0, "content": txt}


@test("an_output_config_low")
def _():
    r = an({"max_tokens": 1500, "output_config": {"effort": "low"},
            "messages": [{"role": "user", "content": "1+2+...+100 等於多少？只回數字。"}]})
    j = r.json()
    texts = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
    return {"ok": "5050" in texts, "note": f"stop={j.get('stop_reason')} | {texts.strip()[:60]!r}", "usage": j.get("usage")}


@test("context_overflow_400")
def _():
    r = oa({"messages": [{"role": "user", "content": "hi"}], "max_tokens": MAXCTX + 100}, timeout=60)
    return {"ok": r.status_code == 400, "note": f"HTTP {r.status_code}: {r.text[:120]}"}


if A.image and HEALTH.get("images"):
    b64 = base64.b64encode(Path(A.image).read_bytes()).decode()
    VQ = "這張圖是什麼？列出你看得到的 3 個具體數字或文字，繁體中文。"

    @test("vision_openai")
    def _():
        t0 = time.time()
        r = oa({"max_tokens": 600, "reasoning_effort": "none", "messages": [{"role": "user", "content": [
            {"type": "text", "text": VQ}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]})
        j = r.json()
        txt = j["choices"][0]["message"].get("content") or "" if r.status_code == 200 else r.text
        return {"ok": r.status_code == 200 and len(txt) > 20, "note": txt[:160].replace("\n", " "), "content": txt,
                "usage": j.get("usage"), "timings": j.get("timings"), "wall_s": time.time() - t0}

    @test("vision_anthropic")
    def _():
        r = an({"max_tokens": 600, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": VQ}]}]})
        j = r.json()
        txt = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
        return {"ok": r.status_code == 200 and len(txt) > 20, "note": txt[:160].replace("\n", " "), "content": txt}
elif A.image:
    print("(vision skipped: /health says images=false)", flush=True)


# ------------------------------------------------------------------ 速度
SHORT = ("You are a careful equity analyst. In about 150 words, explain the difference between free cash flow and "
         "net income, why investors watch free cash flow for capital-intensive companies, and one common way "
         "companies make net income look better than cash flow. Use plain English. ") * 1
LONG_SRC = (Path(r"E:\Strata\docs\DETAILS.md").read_text(encoding="utf-8", errors="replace"))[:A.long_chars]


def speed_run(prompt, tag, i):
    nonce = f"[run {tag}-{i}-{random.randint(0, 10**9)}] "     # 開頭放亂數，避免對話快取讓第二次變假快
    c, rs, ttft, tot, last, n = oa_stream({"messages": [{"role": "user", "content": nonce + prompt}],
                                           "max_tokens": 256, "reasoning_effort": "none", "temperature": 0.6})
    tm = (last or {}).get("timings") or {}
    us = (last or {}).get("usage") or {}
    return {"i": i, "prompt_tokens": us.get("prompt_tokens"), "cached": (us.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "out_tokens": us.get("completion_tokens"), "prefill_tok_s": tm.get("prompt_per_second"),
            "decode_tok_s": tm.get("predicted_per_second"), "draft_n": tm.get("draft_n"),
            "draft_accepted": tm.get("draft_n_accepted"), "ttft_s": round(ttft or -1, 2), "total_s": round(tot, 1)}


for kind, prompt in (("short", SHORT), ("long", "Summarize the following technical document in 5 bullet points.\n\n"
                                                + LONG_SRC)):
    @test(f"speed_{kind}")
    def _(kind=kind, prompt=prompt):
        runs = [speed_run(prompt, kind, i) for i in range(A.speed_runs)]
        dec = [r["decode_tok_s"] for r in runs if r["decode_tok_s"]]
        pre = [r["prefill_tok_s"] for r in runs if r["prefill_tok_s"]]
        ttf = [r["ttft_s"] for r in runs if r["ttft_s"] and r["ttft_s"] > 0]
        note = (f"prompt≈{runs[0]['prompt_tokens']} tok | decode {min(dec, default=0)}–{max(dec, default=0)} tok/s | "
                f"prefill {min(pre, default=0)}–{max(pre, default=0)} tok/s | TTFT {min(ttf, default=0)}–{max(ttf, default=0)} s")
        return {"ok": len(dec) == len(runs), "note": note, "runs": runs}


@test("conversation_cache_reuse")
def _():
    base = "Summarize the following technical document in 5 bullet points.\n\n" + LONG_SRC[:12000]
    nonce = f"[cache-{random.randint(0, 10**9)}] "   # 兩次用同一個 nonce：第二次應命中對話快取
    out = []
    for i in range(2):
        c, rs, ttft, tot, last, n = oa_stream({"messages": [{"role": "user", "content": nonce + base}],
                                               "max_tokens": 64, "reasoning_effort": "none"})
        us = (last or {}).get("usage") or {}
        out.append({"ttft_s": round(ttft or -1, 2), "prompt_tokens": us.get("prompt_tokens"),
                    "cached": (us.get("prompt_tokens_details") or {}).get("cached_tokens")})
    ok = (out[1]["cached"] or 0) > 0 and out[1]["ttft_s"] < out[0]["ttft_s"]
    return {"ok": ok, "note": f"1st ttft={out[0]['ttft_s']}s cached={out[0]['cached']} → 2nd ttft={out[1]['ttft_s']}s "
                              f"cached={out[1]['cached']}", "runs": out}


_stop.set()
time.sleep(0.2)
peak_vram = max((s.get("vram_used_mib") or 0) for s in samples) if samples else None
peak_ram = max((s.get("ram_used") or 0) for s in samples) if samples else None
peak_temp = max((s.get("gpu_temp_c") or 0) for s in samples) if samples else None
summary = {"tag": A.tag, "url": URL, "health": HEALTH, "passed": sum(r.get("ok", False) for r in results),
           "total": len(results), "peak_vram_mib": peak_vram, "peak_ram": peak_ram, "peak_gpu_temp_c": peak_temp}
(OUT_DIR / f"04_api_{A.tag}.json").write_text(json.dumps({"summary": summary, "results": results, "samples": samples},
                                                          ensure_ascii=False, indent=1), encoding="utf-8")
print("== SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
