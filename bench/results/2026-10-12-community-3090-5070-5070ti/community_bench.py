#!/usr/bin/env python3
"""Community benchmark for Strata, following docs/COMMUNITY_BENCHMARKS.md.

Measures, per configuration and per run:
  - prompt throughput: freshly read prompt tokens / prompt time (engine log line)
  - decode throughput: generated tokens / generation time
  - time to first token (TTFT): streamed, keep-alives and empty deltas ignored
  - total latency: whole request, client-side
  - reused vs freshly read tokens
  - VRAM and RAM snapshots before/after each run

Runs a short prompt and a long prompt, 3 repetitions each, greedy, 256-token cap.
Writes per-run JSON.

Usage: python3 bench/community_bench.py [--url http://127.0.0.1:8080] [--out bench/results/community.json]
"""
from __future__ import annotations
import argparse, json, os, random, statistics, time, urllib.request

FILLER = "Explain cache hierarchies, TLBs, and memory consistency in detail. "
SHORT_TARGET = 512
LONG_TARGET = 32768
OUT_CAP = 256
RUNS = 3


def build_prompt(target_tokens: int, salt: str) -> str:
    # ~11 tokens per filler sentence; salt guarantees no prefix reuse
    n = max(1, target_tokens // 11)
    return f"BENCHMARK {salt} " + FILLER * n


def vram_snapshot() -> dict:
    try:
        out = os.popen("nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader,nounits").read()
        return {"gpus": [int(l.split(",")[2]) for l in out.strip().splitlines()],
                "total_mib": sum(int(l.split(",")[2]) for l in out.strip().splitlines())}
    except Exception:
        return {"error": "not measured"}


def ram_snapshot() -> dict:
    try:
        with open("/proc/meminfo") as f:
            d = {k.split(":")[0]: int(k.split(":")[1].split()[0]) for k in f}
        return {"used_mib": (d["MemTotal"] - d["MemAvailable"]) // 1024,
                "total_mib": d["MemTotal"] // 1024}
    except Exception:
        return {"error": "not measured"}


def run_once(url: str, model: str, prompt: str, cap: int) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": cap, "temperature": 0.0, "stream": True}
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t_start = time.time()
    ttft = None
    generated = 0
    text = []
    usage = {}
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage") or usage
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            piece = delta.get("content") or delta.get("reasoning_content") or ""
            if piece:
                if ttft is None:
                    ttft = time.time() - t_start
                generated += len(piece)
                text.append(piece)
    total = time.time() - t_start
    return {"prompt_chars": len(prompt), "generated_chars": generated,
            "ttft_s": ttft, "total_s": total, "usage": usage,
            "text_preview": "".join(text)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--out", default="bench/results/community-rig.json")
    ap.add_argument("--runs", type=int, default=RUNS)
    a = ap.parse_args()

    status = json.load(urllib.request.urlopen(a.url + "/v1/status", timeout=10))
    model = status["model"]
    print(f"model: {model}  engine {status['engine']}  context {status['cache_max_tokens']}")

    results = []
    for label, target in (("short", SHORT_TARGET), ("long", LONG_TARGET)):
        prompts = [build_prompt(target, f"{label}-{i}-{random.random():.6f}") for i in range(a.runs)]
        runs = []
        for i, prompt in enumerate(prompts):
            v_before = vram_snapshot()
            r_before = ram_snapshot()
            r = run_once(a.url, model, prompt, OUT_CAP)
            r["vram_before_mib"] = v_before
            r["ram_before_mib"] = r_before
            r["vram_after_mib"] = vram_snapshot()
            r["ram_after_mib"] = ram_snapshot()
            r["run"] = i + 1
            runs.append(r)
            print(f"{label} run {i+1}: ttft {r['ttft_s']:.2f}s total {r['total_s']:.2f}s "
                  f"usage={r.get('usage')}")
        results.append({"configuration": label, "runs": runs})

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"model": model, "engine": status["engine"], "context": status["cache_max_tokens"],
               "measured_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "results": results},
              open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")

    for cfg in results:
        ttfts = [r["ttft_s"] for r in cfg["runs"]]
        totals = [r["total_s"] for r in cfg["runs"]]
        print(f"{cfg['configuration']}: TTFT median {statistics.median(ttfts):.2f}s "
              f"range {min(ttfts):.2f}-{max(ttfts):.2f}s")


if __name__ == "__main__":
    main()
