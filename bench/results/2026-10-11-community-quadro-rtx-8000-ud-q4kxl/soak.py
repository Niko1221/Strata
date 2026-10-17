#!/usr/bin/env python3
"""Long soak: back-to-back requests for a set time, to answer docs/TEST_REQUESTS.md's
"a long soak (an hour or more of back-to-back requests) on a split or resident setup".

Run this unattended, from the repository root:

    nohup python3 bench/results/2026-10-11-community-quadro-rtx-8000-ud-q4kxl/soak.py \
        --minutes 75 --out bench/results/2026-10-11-community-quadro-rtx-8000-ud-q4kxl/soak.jsonl \
        > soak.out 2>&1 &

Run it while nothing else uses the engine - including the chat and any agent talking
to this server: their requests share the one slot and would move the per-request
timings. The script never starts, stops or reconfigures the engine; if the engine
goes away it records the failure and keeps trying, which is itself the observation.

What it watches for, sampled every request and every --sample seconds:
  - decode throughput drift across the run (the first and last decile compared)
  - RAM growth (the resident expert complement and the conversation cache leaking)
  - VRAM and the engine's own "free MiB" figure (an over-sized cache paging shows up here)
  - errors and timeouts, and the finish reasons: with a fixed --max-tokens the expected
    reason is "length", so an early "stop", an "error" or a missing answer is the signal
"""
import argparse
import json
import statistics
import subprocess
import time
import urllib.error
import urllib.request

FILLER = (
    "Sparse mixture-of-experts routing sends each token through a small subset of feed-forward experts. "
    "The gate scores every expert, masks all but the top-k and renormalizes the survivors. "
    "A page-locked complement holds the experts that do not fit in the card's memory. "
)


def post(url, body, timeout=900):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read())


def get(url, path="/metrics", timeout=20):
    try:
        with urllib.request.urlopen(url + path, timeout=timeout) as f:
            return json.loads(f.read())
    except Exception as e:
        return {"_error": str(e)}


def nvidia():
    try:
        out = subprocess.run(["nvidia-smi",
                              "--query-gpu=memory.used,temperature.gpu,power.draw,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=10).stdout.strip().split("\n")[0]
        m, t, p, u = (float(x) for x in out.split(","))
        return {"vram_used_mib": round(m), "temp_c": t, "power_w": round(p, 1), "util_pct": u}
    except Exception as e:
        return {"_error": str(e)}


def ram():
    mi = dict(l.split(":", 1) for l in open("/proc/meminfo").read().splitlines() if ":" in l)
    return {"ram_available_gib": round(int(mi["MemAvailable"].split()[0]) / 1048576, 2),
            "ram_used_gib": round((int(mi["MemTotal"].split()[0]) - int(mi["MemAvailable"].split()[0]))
                                  / 1048576, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--minutes", type=float, default=75)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--prompt-tokens", type=int, default=1500, help="approx; each request is a fresh prompt")
    ap.add_argument("--sample", type=float, default=30, help="seconds between hardware samples")
    ap.add_argument("--out", default="soak.jsonl")
    a = ap.parse_args()

    end = time.time() + a.minutes * 60
    n = 0
    last_sample = 0.0
    with open(a.out, "a") as out:
        while time.time() < end:
            n += 1
            nonce = f"[soak {n} {int(time.time())} {n * 2654435761 % (1 << 40):x}] "
            body = {"model": "strata", "temperature": 0, "max_tokens": a.max_tokens, "stream": False,
                    "messages": [{"role": "user", "content":
                                  nonce + (FILLER * max(1, a.prompt_tokens // 60))[:a.prompt_tokens * 4]
                                  + " Summarise the above in two sentences."}]}
            t0 = time.monotonic()
            rec = {"n": n, "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            try:
                d = post(a.url, body)
                tm = d.get("timings") or {}
                rec.update({
                    "ok": True,
                    "prompt_n": tm.get("prompt_n"), "prompt_tps": tm.get("prompt_per_second"),
                    "gen_n": tm.get("predicted_n"), "decode_tps": tm.get("predicted_per_second"),
                    "finish": (d.get("choices") or [{}])[0].get("finish_reason"),
                    "wall_s": round(time.monotonic() - t0, 2),
                })
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
                rec.update({"ok": False, "error": f"{type(e).__name__}: {e}",
                            "wall_s": round(time.monotonic() - t0, 2)})
            if time.time() - last_sample >= a.sample:
                last_sample = time.time()
                m = get(a.url)
                rec.update(nvidia())
                rec.update(ram())
                eng = m.get("engine") or {}
                hw = m.get("hardware") or {}
                rec.update({"vram_free_mib": eng.get("vram_free_mib"),
                            "expert_slots": eng.get("expert_slots"),
                            "conv_cache_slots": eng.get("conversation_cache_slots"),
                            "conv_cache_mib": eng.get("conversation_cache_mib"),
                            "engine_tok_s": hw.get("tok_s")})
                rec["totals"] = m.get("totals")
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if n % 25 == 0:
                print(f"{n} requests, {max(0, end - time.time()) / 60:.1f} min left"
                      + (f", last decode {rec.get('decode_tps')} tok/s" if rec.get("ok") else
                         f", LAST FAILED: {rec.get('error')}"), flush=True)

    # Summary: drift and failures.
    runs = [json.loads(l) for l in open(a.out) if l.strip()]
    ok = [r for r in runs if r.get("ok") and r.get("decode_tps")]
    fails = [r for r in runs if not r.get("ok")]
    d = [r["decode_tps"] for r in ok]
    dec = lambda xs: round(statistics.median(xs), 2) if xs else None
    k = max(1, len(d) // 10)
    print(json.dumps({
        "requests": len(runs), "ok": len(ok), "failed": len(fails),
        "first_failure": fails[0]["ts"] if fails else None,
        "errors": sorted({str(r.get("error"))[:120] for r in fails})[:10],
        "finish_reasons": {r: sum(1 for x in ok if x.get("finish") == r)
                           for r in {x.get("finish") for x in ok}},
        "decode_tps_first_decile": dec(d[:k]), "decode_tps_last_decile": dec(d[-k:]),
        "decode_tps_median_all": dec(d),
        "prompt_tps_first_decile": dec([r["prompt_tps"] for r in ok[:k]]),
        "prompt_tps_last_decile": dec([r["prompt_tps"] for r in ok[-k:]]),
        "ram_used_gib_first": next((r.get("ram_used_gib") for r in runs if "ram_used_gib" in r), None),
        "ram_used_gib_last": next((r.get("ram_used_gib") for r in reversed(runs) if "ram_used_gib" in r), None),
        "vram_used_mib_max": max((r["vram_used_mib"] for r in runs if "vram_used_mib" in r), default=None),
        "vram_free_mib_min": min((r["vram_free_mib"] for r in runs if r.get("vram_free_mib") is not None),
                                 default=None),
        "temp_max_c": max((r["temp_c"] for r in runs if r.get("temp_c")), default=None),
        "conv_cache_slots_last": next((r.get("conv_cache_slots") for r in reversed(runs)
                                       if r.get("conv_cache_slots") is not None), None),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
