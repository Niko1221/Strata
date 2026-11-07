#!/usr/bin/env python3
"""House-protocol benchmark harness, in the shape bench/results/*/runs.csv uses.

4,096 / 32,768 / 128,000 target prompt tokens, 256-token output, temperature 0,
non-streaming, three runs per depth, a distinct revision marker per run so no run reuses
another's prefix, warm expert cache shared by design.  Decode rate comes from the engine's
own `generate: ... (N tok/s)` line, not from wall-clock division, so reasoning tokens are
counted the same way the vendor counts them.

  python3 house_bench.py <config.json> <config-name> [out-prefix]

Writes <out-prefix>-runs.csv, <out-prefix>-results.json and <out-prefix>-memory.json.
"""
import csv, json, os, re, statistics, subprocess, sys, time, urllib.request, urllib.error

URL = "http://127.0.0.1:8080/v1/chat/completions"
LOG = "/home/dom/Strata/strata-iq3_s.log"
OUTDIR = "/home/dom/strata-update-v0212-20261107"
TARGETS = [4096, 32768, 128000]
RUNS = 3
MAXTOK = 256
BYTES_PER_TOKEN = 4.75        # measured for THIS filler; re-derived per run from the warm-up reply

PARA = (
    "The router scores every expert against the token's hidden state, keeps the top few, and "
    "streams only those experts through the GPU while the rest wait in system memory; a miss is "
    "paid for in PCIe, a hit is paid for in patience, and the n-gram table guesses the next "
    "layer's set before the hidden state arrives so the transfer can start early. Expert affinity "
    "widens the window it keeps resident, and an update that changes which experts a token wants "
    "invalidates a stale affinity at once. "
)


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()


def key_of(cfg):
    return cfg.get("api_key", "")


def mem():
    v = [int(x) for x in re.findall(r"\s(\d{4,6})\s*$",
         sh("LC_ALL=C.UTF-8 rocm-smi --showmeminfo vram --json 2>/dev/null"), re.M)]
    rss = sh("for p in $(pgrep -x strata); do awk '/^VmRSS/{print $2}' /proc/$p/status 2>/dev/null; done | paste -sd+ | bc 2>/dev/null")
    return {"vram_mib_total": sum(v) // 1024 if v else None,
            "mem_available_mib": int(sh("awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo") or 0),
            "engine_rss_mib": int(int(rss or 0) // 1024)}


def corpus(target, marker, bpt=None):
    n = int(target * (bpt or BYTES_PER_TOKEN))
    body = (PARA * (n // len(PARA) + 2))[:n]
    return ("Document revision marker: %s\n\nThe following reference document is long and "
            "repetitive on purpose; read it and finish the last sentence.\n\n" % marker) + body


ENGRE = (r"prompt ([0-9]+) tokens = ([0-9]+) reused \+ ([0-9]+) read in ([0-9]+) ms \(([0-9.]+) tok/s\), "
         r"([0-9]+) generated in ([0-9]+) ms \(([0-9.]+) tok/s\), drafts accepted ([0-9]+) of ([0-9]+)")


def log_request():
    """The engine's own accounting line for the request that just completed (requests are sequential)."""
    out = sh("tail -c 4000000 " + LOG + " | grep -a 'strata serve: prompt ' | tail -1")
    m = re.search(ENGRE, out or "")
    if not m:
        return {}
    acc, drafts = int(m.group(9)), int(m.group(10))
    return {"engine_prompt_n": int(m.group(1)), "reused": int(m.group(2)), "read": int(m.group(3)),
            "prompt_ms": int(m.group(4)), "prompt_tps": float(m.group(5)),
            "out_n": int(m.group(6)), "gen_ms": int(m.group(7)), "decode_tps": float(m.group(8)),
            "accepted": acc, "drafts": drafts,
            "accept_pct": round(100.0 * acc / drafts, 1) if drafts else None}


def log_cache_hit():
    out = sh("tail -c 4000000 " + LOG + " | grep -aoE 'expert cache hit rate: [0-9.]+' | tail -1")
    m = re.search(r":\s*([0-9.]+)", out or "")
    return float(m.group(1)) if m else None


def mem():
    j = sh("LC_ALL=C.UTF-8 rocm-smi --showmeminfo vram --json 2>/dev/null")
    vram = None
    try:
        d = json.loads(j)
        vram = sum(int(c.get("VRAM Total Used Memory (B)", 0)) for c in d.values()) // 1048576
    except Exception:
        pass
    rss = sh("for p in $(pgrep -f 'engine/strata --serve'); do awk '/^VmRSS/{print $2}' /proc/$p/status 2>/dev/null; done | sort -n | tail -1")
    avail = sh("awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo")
    return {"vram_mib_total": vram, "mem_available_mib": int(avail or 0),
            "engine_rss_mib": int(int(rss or 0) // 1024)}


def call(prompt, key, model):
    payload = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                          "temperature": 0, "max_tokens": MAXTOK}).encode()
    req = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json",
                                                            "Authorization": "Bearer " + key})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read().decode())
    wall = time.time() - t0
    u = d.get("usage") or {}
    m = ((d.get("choices") or [{}])[0].get("message") or {})
    e = log_request()
    return {"prompt_n": u.get("prompt_tokens"), "cache_n": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "out_n": e.get("out_n") or u.get("completion_tokens"),
            "decode_tps": e.get("decode_tps"), "prompt_tps": e.get("prompt_tps"),
            "prompt_ms": e.get("prompt_ms"), "gen_ms": e.get("gen_ms"),
            "reused": e.get("reused"), "read": e.get("read"),
            "accept_pct": e.get("accept_pct"), "accepted": e.get("accepted"), "drafts": e.get("drafts"),
            "total_s": round(wall, 2), "reasoning": bool(m.get("reasoning_content")),
            "expert_cache_hit_pct": log_cache_hit(), "content_len": len(m.get("content") or "")}


def main():
    cfg_path = sys.argv[1]
    name = sys.argv[2]
    prefix = sys.argv[3] if len(sys.argv) > 3 else os.path.join(OUTDIR, "house-" + name)
    cfg = json.load(open(cfg_path))
    model = (cfg.get("model_name") or "qwen3.8-flash-next-iq3_s").lower()
    if "/" not in model:
        model = "amd/" + model
    key = key_of(cfg)
    # warm the expert page cache once; startup and warm-up are not timed
    wc = corpus(48000, "warmup-%d" % int(time.time()))
    warm = call(wc, key, model)
    # calibrate the filler to this tokenizer: bytes per token, from the engine's own count
    bpt = len(wc) / float(warm.get("prompt_n") or 1)
    print("warm-up: prompt=%s decode=%s  calibrated bytes/token=%.3f" %
          (warm.get("prompt_n"), warm.get("decode_tps"), bpt), flush=True)
    runs, fails = [], []
    for depth_i, target in enumerate(TARGETS):
        for run_i in range(1, RUNS + 1):
            marker = "rev-%d-%d-%d" % (depth_i, run_i, int(time.time()))
            try:
                r = call(corpus(target, marker, bpt), key, model)
                r.update({"config": name, "target_tokens": target, "run": run_i,
                          "revision_marker": marker})
                r.update(mem())
                runs.append(r)
                print("%-8s run %d  prompt_n=%-8s reused=%-7s cache=%-7s prompt=%-9s decode=%-6s accept=%-6s total=%-7s hit=%s" %
                      (target, run_i, r["prompt_n"], r.get("reused"), r["cache_n"], r["prompt_tps"],
                       r["decode_tps"], r["accept_pct"], r["total_s"], r["expert_cache_hit_pct"]), flush=True)
            except (urllib.error.URLError, TimeoutError, Exception) as e:
                fails.append({"target_tokens": target, "run": run_i, "error": repr(e)[:200]})
                print("%-8s run %d  FAILED %s" % (target, run_i, repr(e)[:120]), flush=True)
    cols = ["config", "target_tokens", "run", "prompt_n", "cache_n", "prompt_ms", "prompt_tps",
            "out_n", "decode_tps", "total_s", "draft_accept_pct", "expert_cache_hit_pct",
            "vram_mib_total", "mem_available_mib", "engine_rss_mib"]
    with open(prefix + "-runs.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in runs:
            w.writerow([r["config"], r["target_tokens"], r["run"], r["prompt_n"], r["cache_n"],
                        r.get("prompt_ms"), r["prompt_tps"], r["out_n"], r["decode_tps"], r["total_s"],
                        r["accept_pct"], r["expert_cache_hit_pct"], r["vram_mib_total"],
                        r["mem_available_mib"], r["engine_rss_mib"]])
    med, rng = {}, {}
    for target in TARGETS:
        sel = [r for r in runs if r["target_tokens"] == target and r.get("decode_tps")]
        if not sel:
            continue
        def q(k, f=float):
            vals = [f(r[k]) for r in sel if r.get(k) is not None]
            return round(f(statistics.median(vals)), 1) if vals else None
        def spread(k):
            vals = [float(r[k]) for r in sel if r.get(k) is not None]
            return [round(min(vals), 1), round(max(vals), 1)] if vals else None
        med[target] = {"decode_tps": q("decode_tps"), "total_s": q("total_s"),
                       "prompt_tps_spread": spread("prompt_tps"),
                       "accept_pct": q("accept_pct"),
                       "decode_tps_spread": spread("decode_tps"), "runs": len(sel)}
        # prompt rate median, computed properly
        pr = [float(r["prompt_tps"]) for r in sel if r.get("prompt_tps")]
        med[target]["prompt_tps"] = round(statistics.median(pr), 1) if pr else None
        rng[target] = {"prompt_tps": [min(pr), max(pr)] if pr else None,
                       "decode_tps": spread("decode_tps")}
    doc = {"config": name, "protocol": "house protocol: 4096/32768/128000 target tokens x %d runs, %d-token output, "
                                       "temperature 0, non-streaming, distinct revision marker per run"
                                       % (RUNS, MAXTOK),
           "engine_version": sh("tail -c 200000 /home/dom/Strata/strata-iq3_s.log | "
                                "grep -aoE 'strata serve v[0-9.]+ engine [0-9.]+' | tail -1"),
           "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "medians": med, "ranges": rng, "runs": runs, "failed": fails}
    open(prefix + "-results.json", "w").write(json.dumps(doc, indent=1) + "\n")
    print("\n== medians ==")
    for t, v in sorted(med.items()):
        print("  %-8s prompt %-9s decode %-6s (runs %d, decode spread %s)" %
              (t, v["prompt_tps"], v["decode_tps"], v["runs"], v["decode_tps_spread"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
