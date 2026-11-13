#!/usr/bin/env python3
"""One hour of back-to-back requests against the running server, sampling the card as it goes.

Writes: requests.csv (one row per request), samples.csv (one row per sample), summary.json.
Standard library only, so it runs on the server with its own python3.
"""
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

OUT = os.environ.get("SOAK_OUT", "/workspace/Strata/soak-013")
DURATION_S = int(os.environ.get("SOAK_SECONDS", "3600"))
KEY = os.environ.get("SOAK_KEY", "")
URL = "http://127.0.0.1:8899/v1/chat/completions"
MODEL = "qwen3.8-flash-next-q2_0"
SAMPLE_EVERY = 10.0

PROMPTS = [
    ("short", "In one sentence, what is the difference between a stack and a queue?"),
    ("short", "Name three prime numbers greater than 50."),
    ("short", "What does HTTP status 429 mean? One sentence."),
    ("medium", "Explain how a hash table handles collisions, covering chaining and open addressing, in about 200 words."),
    ("medium", "Summarise the difference between a process and a thread, and when each is the better choice."),
    ("long", ("The following is a long passage. Read it and answer the question at the end.\n\n"
              + ("A distributed system is one in which components located on networked computers communicate and "
                 "coordinate their actions by passing messages. The components interact with each other in order to "
                 "achieve a common goal. Three significant characteristics of distributed systems are concurrency of "
                 "components, lack of a global clock, and independent failure of components. " * 40)
              + "\n\nQuestion: how many significant characteristics are named? Answer with the number only.")),
]
MAX_TOKENS = {"short": 120, "medium": 320, "long": 160}


def gpu_sample():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits", "-i", "1"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        parts = [p.strip() for p in out.split(",")]
        return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    except Exception:
        return -1, -1, -1, -1


def mem_sample():
    try:
        with open("/proc/meminfo") as fh:
            info = {}
            for line in fh:
                k, v = line.split(":", 1)
                info[k] = int(v.strip().split()[0])
        return info["MemTotal"], info["MemAvailable"]
    except Exception:
        return -1, -1


def health():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8899/health", timeout=8) as r:
            d = json.loads(r.read())
        return d.get("status"), d.get("loaded")
    except Exception as e:
        return "unreachable (%s)" % type(e).__name__, None


def ask(kind, prompt, max_tokens):
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + KEY})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    wall = time.time() - t0
    t = d.get("timings") or {}
    u = d.get("usage") or {}
    return dict(kind=kind, wall_s=round(wall, 3),
                prompt_n=t.get("prompt_n"), prompt_tps=round(t.get("prompt_per_second") or 0, 2),
                decode_n=t.get("predicted_n"), decode_tps=round(t.get("predicted_per_second") or 0, 2),
                drafts_n=t.get("draft_n"), drafts_ok=t.get("draft_n_accepted"),
                prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                status="ok")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    req_path = os.path.join(OUT, "requests.csv")
    smp_path = os.path.join(OUT, "samples.csv")
    with open(req_path, "w", encoding="utf-8") as fh:
        fh.write("i,kind,wall_s,prompt_n,prompt_tps,decode_n,decode_tps,drafts_n,drafts_ok,"
                 "prompt_tokens,completion_tokens,status,at\n")
    with open(smp_path, "w", encoding="utf-8") as fh:
        fh.write("t_s,gpu_used_mib,gpu_total_mib,gpu_util_pct,gpu_temp_c,"
                 "mem_total_kib,mem_avail_kib,health,loaded\n")

    start = time.time()
    deadline = start + DURATION_S
    next_sample = start
    rows, samples, failures = [], 0, 0
    print("soak: %d s, out=%s" % (DURATION_S, OUT), flush=True)

    i = 0
    while time.time() < deadline:
        i += 1
        kind, prompt = PROMPTS[(i - 1) % len(PROMPTS)]
        try:
            row = ask(kind, prompt, MAX_TOKENS[kind])
        except Exception as e:
            failures += 1
            row = dict(kind=kind, wall_s=0, prompt_n=None, prompt_tps=0, decode_n=None, decode_tps=0,
                       drafts_n=None, drafts_ok=None, prompt_tokens=None, completion_tokens=None,
                       status="%s: %s" % (type(e).__name__, str(e)[:60]))
        rows.append(row)
        with open(req_path, "a", encoding="utf-8") as fh:
            fh.write("%d,%s,%.3f,%s,%.2f,%s,%.2f,%s,%s,%s,%s,%s,%s\n" % (
                i, row["kind"], row["wall_s"], row["prompt_n"], row["prompt_tps"],
                row["decode_n"], row["decode_tps"], row["drafts_n"], row["drafts_ok"],
                row["prompt_tokens"], row["completion_tokens"],
                row["status"].replace(",", ";")[:80], time.strftime("%Y-%m-%dT%H:%M:%S")))
        # 采样
        now = time.time()
        if now >= next_sample:
            used, total, util, temp = gpu_sample()
            mt, ma = mem_sample()
            st, loaded = health()
            with open(smp_path, "a", encoding="utf-8") as fh:
                fh.write("%.1f,%d,%d,%d,%d,%d,%d,%s,%s\n" % (
                    now - start, used, total, util, temp, mt, ma, st, loaded))
            next_sample = now + SAMPLE_EVERY
        if i % 20 == 0:
            ok = [r for r in rows if r["status"] == "ok"]
            print("  %4d requests, %.0f%% of the hour, %d ok, %d failed, last decode %.1f tok/s"
                  % (i, 100.0 * (now - start) / DURATION_S, len(ok), failures,
                     ok[-1]["decode_tps"] if ok else 0), flush=True)

    ok = [r for r in rows if r["status"] == "ok"]
    def q(vals, p):
        vals = sorted(vals)
        return vals[min(len(vals) - 1, int(len(vals) * p))] if vals else None
    dec = [r["decode_tps"] for r in ok if r["decode_tps"]]
    prm = [r["prompt_tps"] for r in ok if r["prompt_tps"]]
    wall = [r["wall_s"] for r in ok]
    by_kind = {}
    for k in ("short", "medium", "long"):
        sub = [r for r in ok if r["kind"] == k]
        if sub:
            by_kind[k] = dict(n=len(sub),
                              decode_median=round(statistics.median([r["decode_tps"] for r in sub]), 2),
                              decode_min=min(r["decode_tps"] for r in sub),
                              decode_max=max(r["decode_tps"] for r in sub))

    # 前后半段对比（漂移）
    half = len(ok) // 2
    first_half = [r["decode_tps"] for r in ok[:half]] or [0]
    second_half = [r["decode_tps"] for r in ok[half:]] or [0]

    summary = dict(
        started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
        duration_s=round(time.time() - start, 1),
        engine=sys.version.split()[0],
        requests=len(rows), ok=len(ok), failed=failures,
        decode_tps=dict(median=round(statistics.median(dec), 2), p10=q(dec, 0.10), p90=q(dec, 0.90),
                        min=min(dec), max=max(dec)) if dec else None,
        prompt_tps=dict(median=round(statistics.median(prm), 2), p10=q(prm, 0.10), p90=q(prm, 0.90)) if prm else None,
        wall_s=dict(median=round(statistics.median(wall), 2), max=max(wall)) if wall else None,
        by_kind=by_kind,
        drift=dict(first_half_median=round(statistics.median(first_half), 2),
                   second_half_median=round(statistics.median(second_half), 2)) if dec else None,
        health_at_end=health(),
    )
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


raise SystemExit(main())
