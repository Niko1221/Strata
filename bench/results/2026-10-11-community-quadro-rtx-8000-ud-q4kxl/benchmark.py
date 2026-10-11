#!/usr/bin/env python3
"""Fresh-prompt speed sweep for the Quadro RTX 8000 UD-Q4_K_XL community report.

Measures prompt and decode throughput from the server's own `timings` (never by
dividing generated tokens by total request time), at a ladder of prompt lengths
with a fixed output cap, as docs/COMMUNITY_BENCHMARKS.md asks.

    python3 benchmark.py --lengths 4k --runs 3
    python3 benchmark.py --lengths 32k --runs 3
    python3 benchmark.py --lengths 128k --runs 3
    python3 benchmark.py --summary

Every run is appended to runs.jsonl as it finishes, so a chunked session (one
length per call) still aggregates in the end. Nothing here restarts the engine.

Why the integrity check exists: the machine also runs the agent that drives this
script, so its own chat requests share the one engine slot. A run is only trusted
when the server's newest request record matches this request's prompt token count
and the engine was idle before it started; otherwise the run is written with
"contaminated": true and excluded from the medians.
"""
import argparse
import json
import os
import random
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request

URL_DEFAULT = "http://127.0.0.1:8080"

# Varied filler, so the prompt is not one sentence repeated (repetitive text
# inflates prompt throughput and flattens expert routing, which is exactly what
# this engine caches). A nonce heads every prompt and sits in its middle, so the
# conversation prefix cannot be reused from a previous run.
FILLER = [
    "Sparse mixture-of-experts routing sends each token through a small subset of feed-forward experts.",
    "The gate projects hidden states into per-expert scores, applies a top-k mask and renormalizes the survivors.",
    "Recurrent delta layers maintain linear state across the sequence while sparse attention indexes a subset of keys.",
    "Quantized weights trade numerical precision for memory bandwidth at every matrix multiply.",
    "A page-locked complement holds the experts that do not fit in the graphics card's memory.",
    "The expert cache follows the conversation and swaps the least useful blocks back out again.",
    "Prompt ingestion is bounded by the host-to-device link when the experts are read over PCIe.",
    "Draft tokens from the multi-token predictor are checked in one pass and accepted from the left edge.",
    "Key and value tensors can be kept at eight-bit precision to leave room for a longer context.",
    "Decode speed falls when the memory of the card is completely full, not when the hit rate drops.",
    "A checkpoint of the conversation lets a later question read only the tokens that were added since.",
    "The host thread drains work while the pool workers compute the experts that stay in system memory.",
    "Greedy sampling with temperature zero makes two runs of the same prompt comparable token by token.",
    "Time to first token includes the queue wait, the prompt read and any model loading in front of it.",
    "A needle in a haystack measures recall on those inputs and says nothing about overall answer quality.",
    "Bandwidth per token, not FLOPs, sets the pace of a batch-one decode on a desktop card.",
    "The file tier reads through the operating system's page cache when RAM cannot hold every expert.",
    "Layer splitting across two cards moves whole transformer blocks rather than individual tensors.",
    "Calibration tunes the split between the GPU cache and the CPU expert pool for one machine.",
    "Measured numbers should name the unit, the timing boundary and the state of the cache.",
]

TASK = " Summarise the passage above in two sentences and give it a title."


def api(url, path, body=None, timeout=3600):
    req = urllib.request.Request(
        url + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read())


class GpuSampler(threading.Thread):
    """Peak VRAM, peak/mean utilisation, peak power and temperature during a run."""

    FIELDS = "utilization.gpu,memory.used,temperature.gpu,power.draw"

    def __init__(self, period=0.5):
        super().__init__(daemon=True)
        self.stop_at = threading.Event()
        self.samples = []
        self.period = period

    def run(self):
        while not self.stop_at.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0]
                u, m, t, p = (float(x) for x in out.split(","))
                self.samples.append((u, m, t, p))
            except Exception:
                pass
            self.stop_at.wait(self.period)

    def result(self):
        self.stop_at.set()
        self.join(timeout=3)
        if not self.samples:
            return None
        u = [s[0] for s in self.samples]
        return {
            "samples": len(self.samples),
            "vram_peak_mib": round(max(s[1] for s in self.samples)),
            "util_mean_pct": round(statistics.mean(u), 1),
            "util_max_pct": round(max(u)),
            "temp_max_c": round(max(s[2] for s in self.samples)),
            "power_max_w": round(max(s[3] for s in self.samples), 1),
            "power_mean_w": round(statistics.mean(s[3] for s in self.samples), 1),
        }


def corpus(rng, n_chars):
    parts, total = [], 0
    while total < n_chars:
        s = rng.choice(FILLER)
        parts.append(s)
        total += len(s) + 1
    return " ".join(parts)[:n_chars]


def make_prompt(rng, target_tokens, ratio, run_id):
    head = f"[run {run_id} nonce {rng.randrange(1 << 48):012x}] "
    body = corpus(rng, int(target_tokens * ratio))
    mid = len(body) // 2
    tail = f" [mid-marker {run_id}-{rng.randrange(1 << 32):08x}]"
    return head + body[:mid] + tail + body[mid:] + f" [end {run_id}]" + TASK


def parse_length(text):
    t = text.lower().strip()
    return int(float(t.rstrip("kmb")) * 1024) if t.endswith("k") else (
        int(float(t.rstrip("mb")) * 1024 * 1024) if t.endswith("m") else int(float(t)))


def median_block(values, digits=1):
    v = [round(x, digits) for x in values if isinstance(x, (int, float))]
    if not v:
        return {"runs": [], "median": None, "min": None, "max": None}
    return {"runs": v, "median": round(statistics.median(v), digits), "min": min(v), "max": max(v)}


def one_run(url, prompt, max_tokens, run_id, seed, target=None):
    before = api(url, "/status")
    metrics_before = api(url, "/metrics")
    newest_before = (metrics_before.get("requests") or [{}])[0].get("time")

    sampler = GpuSampler()
    sampler.start()
    t0 = time.monotonic()
    data = api(url, "/v1/chat/completions", {
        "model": "strata",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    })
    wall = time.monotonic() - t0
    gpu = sampler.result()

    tm, usage = data.get("timings") or {}, data.get("usage") or {}
    after = api(url, "/metrics")
    newest = (after.get("requests") or [{}])[0]

    # Integrity: the engine's newest request must be this one, and it must match
    # on prompt size, or another client (the agent driving this script) shared
    # the slot and the timings are not clean.
    ok = bool(tm) and newest.get("prompt_total") == usage.get("prompt_tokens")
    if newest_before == newest.get("time"):
        ok = False
    if (before.get("busy") or before.get("queued")):
        ok = False

    return {
        "run_id": run_id,
        "target_tokens": target,
        "seed": seed,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prompt_chars": len(prompt),
        "max_tokens": max_tokens,
        "prompt_n": tm.get("prompt_n"),
        "prompt_ms": tm.get("prompt_ms"),
        "prompt_tps": tm.get("prompt_per_second"),
        "gen_n": tm.get("predicted_n"),
        "gen_ms": tm.get("predicted_ms"),
        "gen_tps": tm.get("predicted_per_second"),
        "draft_n": tm.get("draft_n"),
        "draft_n_accepted": tm.get("draft_n_accepted"),
        "cache_n": tm.get("cache_n"),
        "usage_prompt": usage.get("prompt_tokens"),
        "usage_cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "finish": (data.get("choices") or [{}])[0].get("finish_reason"),
        "wall_s": round(wall, 2),
        "ttft_s_proxy": round(wall - (tm.get("predicted_ms") or 0) / 1000, 2),
        "engine_reused": newest.get("reused"),
        "engine_prompt_read": newest.get("prompt_read"),
        "engine_hit_rate": newest.get("hit_rate"),
        "engine_pcie_share": newest.get("pcie_share"),
        "engine_decode_tok_s": newest.get("decode_tok_s"),
        "gpu": gpu,
        "contaminated": not ok,
    }


def load_runs(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=URL_DEFAULT)
    ap.add_argument("--lengths", default="4k,32k,128k", help="comma-separated, k suffix allowed")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--settle", type=float, default=5.0, help="seconds to idle between runs")
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--summary", action="store_true", help="only aggregate runs.jsonl")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    runs_path = os.path.join(a.out, "runs.jsonl")
    ratio_path = os.path.join(a.out, "ratio.json")
    runs = load_runs(runs_path)

    if a.summary:
        report(runs, a.out)
        return 0

    ratio = json.load(open(ratio_path)).get("ratio", 3.6) if os.path.exists(ratio_path) else 3.6

    for text in a.lengths.split(","):
        target = parse_length(text)
        for i in range(a.runs):
            run_id = f"{text}-r{i + 1}-{int(time.time())}"
            rng = random.Random(a.seed + len(runs))
            prompt = make_prompt(rng, target, ratio, run_id)
            try:
                r = one_run(a.url, prompt, a.max_tokens, run_id, a.seed, target)
            except urllib.error.HTTPError as e:
                print(f"{text} run {i + 1}: HTTP {e.code} {e.read()[:200]}", flush=True)
                continue
            runs.append(r)
            with open(runs_path, "a") as f:
                f.write(json.dumps(r) + "\n")

            # Keep the next prompt near the target token count: the ratio between
            # characters and tokens is this tokenizer's, not a guess we should keep.
            if r.get("prompt_n"):
                ratio = max(1.5, min(8.0, ratio * target / r["prompt_n"]))
                json.dump({"ratio": ratio, "last_prompt_n": r["prompt_n"], "target": target},
                          open(ratio_path, "w"), indent=1)

            print(f"{text} run {i + 1}: prompt {r['prompt_n']} tok @ "
                  f"{(r['prompt_tps'] or 0):.1f} tok/s | decode {r['gen_n']} tok @ "
                  f"{(r['gen_tps'] or 0):.2f} tok/s | reused {r['engine_reused']} "
                  f"read {r['engine_prompt_read']} hit {(r['engine_hit_rate'] or 0) * 100:.1f}% "
                  f"| ttft~{r['ttft_s_proxy']}s wall {r['wall_s']}s "
                  f"| vram_peak {r['gpu']['vram_peak_mib'] if r['gpu'] else '?'} MiB "
                  f"| {'CONTAMINATED' if r['contaminated'] else 'clean'}", flush=True)
            if i + 1 < a.runs or True:
                time.sleep(a.settle)

    report(runs, a.out)
    return 0


def report(runs, out_dir):
    by = {}
    for r in runs:
        if r.get("contaminated"):
            continue
        # Bucket by the requested ladder length when known, else by the nearest
        # rung, else by thousands of prompt tokens.
        t = r.get("target_tokens") or r.get("prompt_n") or 0
        key = min((4096, 32768, 131072), key=lambda x: abs(x - t)) if t >= 2048 else t
        by.setdefault(int(key), []).append(r)

    summary = {}
    for bucket in sorted(by):
        rs = by[bucket]
        label = f"~{bucket / 1024:g}K prompt tokens" if bucket >= 2048 else f"~{bucket} prompt tokens"
        summary[label] = {
            "runs": len(rs),
            "prompt_tokens": [r["prompt_n"] for r in rs],
            "reused": [r["engine_reused"] for r in rs],
            "read": [r["engine_prompt_read"] for r in rs],
            "generated": [r["gen_n"] for r in rs],
            "prompt_tps": median_block([r["prompt_tps"] for r in rs]),
            "decode_tps": median_block([r["gen_tps"] for r in rs], 2),
            "ttft_s": median_block([r["ttft_s_proxy"] for r in rs], 2),
            "total_s": median_block([r["wall_s"] for r in rs], 2),
            "hit_rate_pct": median_block([(r["engine_hit_rate"] or 0) * 100 for r in rs], 1),
            "draft_accept_pct": median_block([
                100.0 * r["draft_n_accepted"] / r["draft_n"] if r.get("draft_n") else None for r in rs], 1),
            "vram_peak_mib": median_block([r["gpu"]["vram_peak_mib"] for r in rs if r.get("gpu")]),
            "power_max_w": median_block([r["gpu"]["power_max_w"] for r in rs if r.get("gpu")], 1),
            "temp_max_c": median_block([r["gpu"]["temp_max_c"] for r in rs if r.get("gpu")]),
        }

    out = os.path.join(out_dir, "results.json")
    json.dump({"summary": summary, "runs": runs}, open(out, "w"), indent=1)
    print("=== SUMMARY (clean runs only) ===")
    print(json.dumps(summary, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
