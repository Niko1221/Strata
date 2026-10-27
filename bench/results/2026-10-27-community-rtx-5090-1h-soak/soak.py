#!/usr/bin/env python3
"""A one-hour back-to-back soak against the running Strata server (docs/TEST_REQUESTS.md, issue #848).

Concurrency 1, one request at a time, no sleep between them: the point is an hour of continuous work, not a
peak-throughput run.  Every request records its own timings from the client's clock; every 30 s the GPU and the
server's /health are sampled next to it, so drift and heat can be read against the same timeline.

Writes, in --out:
  requests.jsonl  one line per request: started_at, ttft_s, total_s, completion_tokens, decode_tps, status, error
  gpu.jsonl       one line per 30 s: at, utilization, memory_used_mib, memory_total_mib, temperature_c, power_w
  health.jsonl    one line per 30 s: at, status, max_context, loaded
  summary.json    the aggregates printed at the end
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPT = ("You are a careful assistant. Explain in your own words why a mixture-of-experts language model "
          "can answer at a similar speed to a smaller dense model while using far more total parameters. "
          "Keep to the point and do not repeat yourself.")


def key(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def gpu_sample() -> dict | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,"
                              "temperature.gpu,power.draw,clocks.sm",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    except Exception:                                   # noqa: BLE001  a missing nvidia-smi is not the soak's business
        return None
    rows = [r.split(", ") for r in out.strip().split("\n") if r.strip()]
    for r in rows:
        if r and r[0] == "1":                           # the card the server runs on (see nvidia-smi: GPU1 = 5090)
            return {"index": int(r[0]), "utilization": float(r[1]), "memory_used_mib": float(r[2]),
                    "memory_total_mib": float(r[3]), "temperature_c": float(r[4]), "power_w": float(r[5]),
                    "clock_sm_mhz": float(r[6])}
    return None


def health(url: str, api_key: str) -> dict:
    try:
        req = urllib.request.Request(url + "/health", headers={"Authorization": "Bearer " + api_key})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read() or b"{}")
    except Exception as e:                              # noqa: BLE001
        return {"error": "%s: %s" % (type(e).__name__, e)}


def one_request(base: str, api_key: str, max_tokens: int) -> dict:
    body = json.dumps({"model": "strata", "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": max_tokens, "stream": True, "temperature": 0.7}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body, method="POST",
                                headers={"Content-Type": "application/json",
                                         "Authorization": "Bearer " + api_key})
    t0 = time.monotonic()
    ttft = None
    text = []
    usage = {}
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        if ttft is None:
                            ttft = time.monotonic() - t0
                        text.append(piece)
        total = time.monotonic() - t0
        gen = int(usage.get("completion_tokens") or 0)
        decode = (gen - 1) / (total - ttft) if (ttft is not None and gen > 1 and total > ttft) else None
        return {"status": "ok", "ttft_s": ttft, "total_s": total, "completion_tokens": gen,
                "prompt_tokens": int(usage.get("prompt_tokens") or 0), "decode_tps": decode,
                "chars": len("".join(text)), "error": None}
    except urllib.error.HTTPError as e:
        return {"status": "http_error", "code": e.code, "ttft_s": None, "total_s": time.monotonic() - t0,
                "completion_tokens": 0, "decode_tps": None, "error": e.read()[:300].decode("utf-8", "replace")}
    except Exception as e:                              # noqa: BLE001
        return {"status": "error", "ttft_s": None, "total_s": time.monotonic() - t0,
                "completion_tokens": 0, "decode_tps": None, "error": "%s: %s" % (type(e).__name__, e)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8899")
    ap.add_argument("--key-file", default="/workspace/Strata/logs/api-key.txt")
    ap.add_argument("--seconds", type=int, default=3600)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--sample-every", type=float, default=30.0)
    ap.add_argument("--out", default="/workspace/Strata/results-soak-1h")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    api_key = key(a.key_file)
    started = time.time()
    deadline = started + a.seconds
    req_f = (out / "requests.jsonl").open("w", encoding="utf-8")
    gpu_f = (out / "gpu.jsonl").open("w", encoding="utf-8")
    hl_f = (out / "health.jsonl").open("w", encoding="utf-8")
    n = errors = 0
    last_sample = 0.0
    print("soak: %ds, c=1, max_tokens=%d, base=%s, out=%s" % (a.seconds, a.max_tokens, a.base, out), flush=True)
    while time.time() < deadline:
        now = time.time()
        rec = {"at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)), "index": n,
               "elapsed_s": round(now - started, 1)}
        rec.update(one_request(a.base, api_key, a.max_tokens))
        rec["at_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        req_f.write(json.dumps(rec) + "\n")
        req_f.flush()
        n += 1
        if rec["status"] != "ok":
            errors += 1
            print("  #%d %s %s" % (n, rec["status"], str(rec.get("error"))[:120]), flush=True)
        elif n % 20 == 0:
            print("  #%d elapsed %4.1f min  ttft %.3f s  decode %.1f tok/s" % (
                n, (time.time() - started) / 60, rec["ttft_s"] or -1, rec["decode_tps"] or -1), flush=True)
        if time.monotonic() - last_sample >= a.sample_every:
            last_sample = time.monotonic()
            g = gpu_sample()
            if g:
                g["at"] = rec["at_end"]
                gpu_f.write(json.dumps(g) + "\n")
                gpu_f.flush()
            h = health(a.base, api_key)
            h["at"] = rec["at_end"]
            hl_f.write(json.dumps(h) + "\n")
            hl_f.flush()
    for f in (req_f, gpu_f, hl_f):
        f.close()

    rows = [json.loads(ln) for ln in (out / "requests.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    ok = [r for r in rows if r["status"] == "ok"]
    gpus = [json.loads(ln) for ln in (out / "gpu.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    tok = sum(r["completion_tokens"] for r in ok)
    hours = max((time.time() - started) / 3600.0, 1e-9)
    first = [r for r in ok if r["elapsed_s"] <= 600]
    last = [r for r in ok if r["elapsed_s"] >= max((r["elapsed_s"] for r in ok), default=0) - 600]
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None            # noqa: E731
    summary = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_s": round(time.time() - started, 1), "concurrency": 1, "max_tokens": a.max_tokens,
        "requests": len(rows), "ok": len(ok), "errors": len(rows) - len(ok),
        "generated_tokens": tok, "throughput_tok_s": round(tok / (time.time() - started), 2),
        "decode_tps_median": med([r["decode_tps"] for r in ok if r["decode_tps"]]),
        "ttft_s_median": med([r["ttft_s"] for r in ok if r["ttft_s"]]),
        "first_10min": {"n": len(first), "decode_median": med([r["decode_tps"] for r in first if r["decode_tps"]]),
                        "ttft_median": med([r["ttft_s"] for r in first if r["ttft_s"]])},
        "last_10min": {"n": len(last), "decode_median": med([r["decode_tps"] for r in last if r["decode_tps"]]),
                       "ttft_median": med([r["ttft_s"] for r in last if r["ttft_s"]])},
        "gpu": {"samples": len(gpus),
                "temp_max_c": max((g["temperature_c"] for g in gpus), default=None),
                "power_max_w": max((g["power_w"] for g in gpus), default=None),
                "memory_used_min_mib": min((g["memory_used_mib"] for g in gpus), default=None),
                "memory_used_max_mib": max((g["memory_used_mib"] for g in gpus), default=None),
                "utilization_median": med([g["utilization"] for g in gpus])},
        "errors_detail": [r for r in rows if r["status"] != "ok"][:20],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("SOAK DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
