#!/usr/bin/env python3
"""Speed sweep for the 2026-10-08 community TITAN RTX report.

Streams chat completions against a running Strata server at three prompt
sizes (4K / 32K / 128K tokens), fixed max_tokens=256, and pairs every run
with the engine's own timing line from the serve log. Each run builds its
prompt from the same block corpus but with a different seed order and a
unique run header, so no run reuses another's prefix: every measured prefill
is fully fresh. The server's own sampling settings apply (temperature 1.0).

Usage (repo root, server running):
    python bench/results/2026-10-08-community-titan-rtx/bench.py \
        [--url http://127.0.0.1:8080] \
        [--log strata-iq3_s.log] [--model qwen3.8-flash-next-iq3_s] \
        [--out runs.json] [--warm]
"""
import argparse
import json
import random
import re
import subprocess
import sys
import time
import urllib.request

LINE_RE = re.compile(
    r"prompt (\d+) tokens = (\d+) reused \+ (\d+)(?: of (\d+))? read in (\d+) ms "
    r"\(([\d.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), "
    r"drafts accepted (\d+) of (\d+)"
)
HIT_RE = re.compile(r"decode expert cache hit rate: (\S+)")

CORPUS = [
    "The engine keeps the hottest experts in graphics memory and the full set in "
    "system RAM, swapping on demand as the router selects different specialists for "
    "each token, which is why free RAM changes decode speed more than clock speed does.",
    "A draft proposal window verifies several candidate tokens in one pass; when the "
    "big model accepts a draft the step costs about the same as a single token, so a "
    "high acceptance rate multiplies decode throughput without changing the answer.",
    "Prompt processing reads the conversation in fixed chunks and checkpoints between "
    "them, so a cancelled request still leaves its leading chunks reusable; the engine "
    "log separates reused prefix tokens from freshly read tokens for exactly this reason.",
    "Expert cache hit rate is the first number to look at when decode slows: a cold "
    "cache forces the graphics card to fetch experts over PCIe or from the CPU pool, "
    "and the rate climbs over the first minutes of use as the cache settles.",
    "Kernel launches cost several microseconds each, and small operations pay that "
    "launch overhead once per layer per token, which is why fusing gather, norm and "
    "activation steps into one kernel shows up as a measurable end-to-end gain.",
    "The router picks ten specialists for every routed layer and the top-k indices are "
    "identical regardless of which kernels serve them, so a kernel rewrite must prove "
    "bit-for-bit parity on the same routed windows before it can claim a speedup.",
    "Key and value tensors for a long conversation can stream between RAM and the "
    "card while attention reads the resident window; the resident window size is a "
    "budget knob traded against the space the expert cache still needs in VRAM.",
    "Quantisation size trades memory traffic against answer quality: smaller codebook "
    "sizes fit more experts in the card and in RAM at once, and the difference shows "
    "up almost entirely in long hard generations rather than in short chat replies.",
    "The key-value cache is quantised to int8 by default, halving both the memory the "
    "context needs and the traffic attention pays when it re-reads previous layers at "
    "long prompt lengths, with a small measurable quality cost at the far end.",
    "Calibration sweeps the fraction of experts kept reachable over the bus and the "
    "floor for draft acceptance, because the right split depends on the actual PCIe "
    "width, the CPU memory pool speed and how often the router leaves the hot set.",
    "Multi-GPU layer splits give each card its own copy of the dense weights and its "
    "share of the experts; the handoff between cards adds latency per token, so two "
    "slow cards do not necessarily beat one fast card on decode, only on context fit.",
    "A generation is only comparable to another generation if the prompt reuse, the "
    "expert cache state and the draft acceptance are all reported, since each of "
    "those three can move decode speed by double-digit percentages on the same box.",
]

QUESTION = (
    "Summarise the passage above in about two hundred words, then list five "
    "distinct items the passage mentions, each with a one-sentence gloss."
)


def build_prompt(seed, target_tokens):
    """Deterministic from (seed): shuffled blocks until ~target_tokens tokens."""
    rng = random.Random(seed)
    blocks = CORPUS * (target_tokens * 4 // sum(len(b) for b in CORPUS) + 2)
    rng.shuffle(blocks)
    header = f"Bench session {seed} of the TITAN RTX community report. "
    chars = target_tokens * 4
    body, total, i = [], len(header), 0
    while total < chars:
        b = f"[{seed}:{i}] {blocks[i % len(blocks)]}"
        body.append(b)
        total += len(b)
        i += 1
    return header + "\n".join(body) + "\n\n" + QUESTION


def stream_chat(url, model, prompt, max_tokens):
    """POST a streaming chat request; return (ttft_s, total_s, finish_reason)."""
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    ttft = None
    finish = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
                ttft = time.monotonic() - t0
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    return ttft, time.monotonic() - t0, finish


def sample_memory():
    vram = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip()
    mem = {}
    with open("/proc/meminfo") as f:
        for key in ("MemTotal", "MemAvailable"):
            mem[key] = int(next(l for l in f if l.startswith(key)).split()[1])
    return {"vram_used_mib": int(vram), "mem_total_kib": mem["MemTotal"],
            "mem_available_kib": mem["MemAvailable"]}




def drain_log(log):
    """Return new serve-log lines since the last read (handle tracks position)."""
    import os
    try:
        real = os.path.getsize(log.name)
    except OSError:
        real = log.tell()
    if real < log.tell():
        log.seek(0)
    new = log.read()
    return [l for l in new.splitlines() if "serve: prompt" in l or "hit rate" in l]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--log", default="strata-iq3_s.log")
    ap.add_argument("--model", default="qwen3.8-flash-next-iq3_s")
    ap.add_argument("--out", default="runs.json")
    ap.add_argument("--warm", action="store_true", help="one untimed 32K warm-up first")
    a = ap.parse_args()

    configs = [("4k", 4000), ("32k", 32000), ("128k", 128000)]
    seeds = [101, 102, 103]
    runs = []

    import io
    log = io.open(a.log, "r", errors="replace")
    drain_log(log)

    if a.warm:
        build = build_prompt(7, 32000)
        stream_chat(a.url, a.model, build, 256)
        drain_log(log)

    for name, target in configs:
        for seed in seeds:
            prompt = build_prompt(seed, target)
            ttft, total, finish = stream_chat(a.url, a.model, prompt, 256)
            lines = drain_log(log)
            timing = None
            hits = []
            for l in lines:
                m = LINE_RE.search(l)
                if m:
                    timing = m
                h = HIT_RE.search(l)
                if h:
                    hits.append(h.group(1))
            if not timing:
                print(f"!! no engine timing line for {name} seed {seed}", file=sys.stderr)
            runs.append({
                "config": name, "seed": seed,
                "prompt_chars": len(prompt),
                "ttft_s": round(ttft, 2) if ttft else None,
                "total_s": round(total, 1),
                "finish_reason": finish,
                "prompt_tokens": int(timing.group(1)) if timing else None,
                "reused_tokens": int(timing.group(2)) if timing else None,
                "fresh_read": int(timing.group(3)) if timing else None,
                "prefill_ms": int(timing.group(5)) if timing else None,
                "prompt_tok_s": float(timing.group(6)) if timing else None,
                "generated_tokens": int(timing.group(7)) if timing else None,
                "decode_ms": int(timing.group(8)) if timing else None,
                "decode_tok_s": float(timing.group(9)) if timing else None,
                "drafts": f"{timing.group(10)}/{timing.group(11)}" if timing else None,
                "expert_cache_hits": hits,
                "memory_after_run": sample_memory(),
            })
            print(json.dumps(runs[-1]))

    with open(a.out, "w") as f:
        json.dump(runs, f, indent=1)
    print(f"wrote {a.out} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
