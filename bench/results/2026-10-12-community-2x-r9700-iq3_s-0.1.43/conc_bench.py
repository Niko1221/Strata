#!/usr/bin/env python3
"""Concurrent-request benchmark with the same workload at every concurrency level.

    python3 conc_bench.py --url http://127.0.0.1:8080/v1 --model qwen3.8-flash-next-iq3_s --name NAME [--conc 1,2,4]
        [--rounds 2] [--max-tokens 512] [--repeat 1] [--cache warm|cold] [--out conc-NAME.json]

The request set is fixed: the four prompts of decode_bench.py (prose, code, edit, reasoning), --repeat times, request
id i = prompt kind i % 4, seed 100 + i, temperature 0.7, top-p 0.95, top-k 20, the same --max-tokens. Every level and
every round sends this same set in the same order; level C keeps at most C requests in flight (C worker threads take the
next request id as soon as their previous one finishes), so C=1 sends the set one request after another.

Cache conditions (--cache):
  warm (default)  the prompt texts are identical at every level and round ("[req-N] " + prompt). One unmeasured priming
                  pass sends the whole set once before the first level, so every measured request is a repeat that a
                  prefix or conversation cache may serve. Each record keeps the server's usage.prompt_tokens_details.
                  cached_tokens, so the JSON shows what the cache actually reused. Priming is not retried: if any
                  priming request fails, its records go under "failures", the JSON is written with no measured round,
                  and the script exits 1 (a measured level must not start from a partly primed cache).
  cold            each request gets a fixed-length nonce unique to (level, round, id), so no cache can serve it. The
                  same scheme is used at every level; the prompt text (and so the sampled output) then differs
                  slightly between levels.

Every round (the priming pass included) has a unique "round_id" (0, 1, 2, ... in the order sent), and every request
record carries the round_id of its round. Levels must be distinct positive integers; --rounds, --repeat and
--max-tokens must be positive.

Per round: wall time = first send to last answer; aggregate tok/s = sum of completion tokens / wall. Per request:
decode tok/s, TTFT, tokens, finish reason, status (bench_stream.py validation). Every request yields a record, also if
the client code raised (status client_error). The summary is derived from the records by summarize(): a round with any
failed request is left out of the aggregate; failed requests are left out of the per-stream and TTFT medians and
counted. Writes --out (default results/NAME-conc.json), also when the run stops early.
Exit status: 0 all requests ok; 1 a request failed (priming or measured; the JSON lists it under "failures");
2 the warm-up request failed (the JSON is written with that record and nothing else).
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from bench_stream import failed, positive_int, public, stream  # noqa: E402
from decode_bench import PROMPTS  # noqa: E402

KINDS = list(PROMPTS)


def request_set(repeat):
    return [{"id": i, "kind": KINDS[i % len(KINDS)], "seed": 100 + i} for i in range(repeat * len(KINDS))]


def prompt_text(req, cache, level=None, rnd=None):
    if cache == "cold":
        tag = hashlib.sha256(f"{level}/{rnd}/{req['id']}/{time.time_ns()}".encode()).hexdigest()[:12]
    else:
        tag = f"req-{req['id']:02d}"
    return f"[{tag}] " + PROMPTS[req["kind"]]


def run_round(a, reqs, level, rnd, temp, measured=True, round_id=None):
    """Send the set with at most `level` requests in flight. Returns (round record, request records), one request
    record per request in `reqs` order: a request whose client code raised, or that no worker reached, gets a
    client_error record instead of None."""
    nxt = [0]
    lock = threading.Lock()
    recs = [None] * len(reqs)
    t0 = time.time()

    def record(j, slot, r, ts, te):
        q = reqs[j]
        out = public(r, prompt_id=q["id"], kind=q["kind"], round=rnd, round_id=round_id, concurrency=level, slot=slot,
                     start=round(ts - t0, 3), end=round(te - t0, 3))
        for k in ("t_start", "t_end"):
            out.pop(k, None)
        out.setdefault("seed", q["seed"])
        out.setdefault("max_tokens", a.max_tokens)
        out.setdefault("temperature", temp)
        return out

    def worker(slot):
        while True:
            with lock:
                if nxt[0] >= len(reqs):
                    return
                j = nxt[0]
                nxt[0] += 1
            ts = time.time()
            try:
                q = reqs[j]
                r = stream(a.url, a.model, [{"role": "user", "content": prompt_text(q, a.cache, level, rnd)}], temp,
                           a.max_tokens, seed=q["seed"], timeout=1800)
                recs[j] = record(j, slot, r, r["t_start"], r["t_end"])
            except Exception as e:                            # noqa: BLE001 - every request must leave a record
                recs[j] = record(j, slot, failed("client_error", f"{type(e).__name__}: {e}"), ts, time.time())

    th = [threading.Thread(target=worker, args=(s,)) for s in range(level)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    for j, r in enumerate(recs):
        if r is None:                                         # a worker died outside the per-request guard
            now = time.time()
            recs[j] = record(j, None, failed("client_error", "no record: the worker thread ended early"), now, now)
    wall = time.time() - t0
    return {"concurrency": level, "round": rnd, "round_id": round_id, "measured": measured, "wall_s": round(wall, 3),
            "started": time.strftime("%H:%M:%S", time.localtime(t0))}, recs


def _round_key(rec):
    """Join key of a round or request record: its round_id; files written before round_id existed (format 2, e.g.
    conc-D-batch2.json, which had no duplicate levels) fall back to (concurrency, round)."""
    rid = rec.get("round_id")
    return ("id", rid) if rid is not None else ("cr", rec["concurrency"], rec["round"])


def summarize(rounds, requests):
    """Derive the per-level summary from the stored records only (the README helper calls this too)."""
    out = {}
    for level in sorted({r["concurrency"] for r in rounds if r.get("measured", True)}):
        rs = [r for r in rounds if r["concurrency"] == level and r.get("measured", True)]
        aggs, failed_rounds = [], 0
        for r in rs:
            mine = [q for q in requests if _round_key(q) == _round_key(r)]
            if any(q["status"] != "ok" for q in mine):
                failed_rounds += 1
                continue
            aggs.append(sum(q["completion_tokens"] for q in mine) / r["wall_s"])
        keys = {_round_key(r) for r in rs}
        reqs = [q for q in requests if _round_key(q) in keys]
        ok = [q for q in reqs if q["status"] == "ok"]
        tps = [q["tps"] for q in ok if q["tps"] is not None]
        tt = [q["ttft"] for q in ok if q["ttft"] is not None]
        cached = [q["cached_tokens"] for q in ok if q["cached_tokens"] is not None]
        out[str(level)] = {
            "aggregate_tps": round(statistics.median(aggs), 1) if aggs else None,
            "aggregate_tps_rounds": [round(x, 1) for x in aggs],
            "per_stream_tps": round(statistics.median(tps), 1) if tps else None,
            "ttft_s": round(statistics.median(tt), 2) if tt else None,
            "completion_tokens": sum(q["completion_tokens"] for q in ok),
            "cached_tokens_median": statistics.median(cached) if cached else None,
            "requests_ok": len(ok), "requests_failed": len(reqs) - len(ok), "rounds_excluded": failed_rounds,
        }
    return out


def parse_levels(ap, text):
    try:
        levels = [int(x) for x in text.split(",")]
    except ValueError:
        ap.error(f"--conc: comma-separated integers expected, got {text!r}")
    if any(c <= 0 for c in levels):
        ap.error(f"--conc: levels must be positive, got {text!r}")
    if len(set(levels)) != len(levels):
        ap.error(f"--conc: each level once, got {text!r}")
    return levels


def write_out(a, out):
    path = a.out or os.path.join("results", a.name + "-conc.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Concurrent-request benchmark (see the module docstring).")
    ap.add_argument("--url", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8080/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="conc", help="run name (default output: results/NAME-conc.json)")
    ap.add_argument("--out", help="output JSON path")
    ap.add_argument("--conc", default="1,2,4", help="distinct positive concurrency levels, in the order run")
    ap.add_argument("--rounds", type=positive_int, default=2)
    ap.add_argument("--max-tokens", type=positive_int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--repeat", type=positive_int, default=1, help="the request set is the four prompts this many times")
    ap.add_argument("--cache", choices=("warm", "cold"), default="warm")
    a = ap.parse_args(argv)
    levels = parse_levels(ap, a.conc)
    reqs = request_set(a.repeat)
    rounds, records, failures = [], [], []
    out = {"script": "conc_bench.py", "format": 3, "name": a.name, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "model": a.model,
           "params": {"conc": levels, "rounds": a.rounds, "max_tokens": a.max_tokens, "temperature": a.temperature,
                      "top_p": 0.95, "top_k": 20, "repeat": a.repeat, "cache": a.cache,
                      "request_set": reqs, "scheduling": "at most C in flight; next request id on each completion"},
           "stopped": None}

    def finish(stopped=None):
        out.update({"stopped": stopped, "rounds": rounds, "requests": records, "failures": failures,
                    "summary": summarize(rounds, records)})
        path = write_out(a, out)
        print("CONC", a.name, json.dumps(out["summary"]), flush=True)
        if stopped:
            print(f"stopped before measurement: {stopped}; see 'failures' in {path}", flush=True)

    w = stream(a.url, a.model, [{"role": "user", "content": "Say OK."}], 0, 8)
    if w["status"] != "ok":
        print(f"warm-up request failed: {w['status']}: {w['error']}", flush=True)
        failures.append({"phase": "warm-up", "status": w["status"], "error": w["error"]})
        finish("the warm-up request failed")
        return 2
    if a.cache == "warm":
        rr, recs = run_round(a, reqs, 1, 0, a.temperature, measured=False, round_id=len(rounds))
        rounds.append(rr)
        records += recs
        bad = [q for q in recs if q["status"] != "ok"]
        print(f"  priming pass: {rr['wall_s']:.1f} s, {len(recs) - len(bad)} of {len(recs)} ok", flush=True)
        for q in bad:
            failures.append({"phase": "priming", "concurrency": 1, "round": 0, "round_id": rr["round_id"],
                             "prompt_id": q["prompt_id"], "status": q["status"], "error": q["error"]})
            print(f"    priming req {q['prompt_id']}: FAILED {q['status']}: {q['error']}", flush=True)
        if bad:
            finish(f"{len(bad)} of {len(recs)} priming requests failed (priming is not retried)")
            return 1
    for level in levels:
        for rnd in range(1, a.rounds + 1):
            rr, recs = run_round(a, reqs, level, rnd, a.temperature, round_id=len(rounds))
            rounds.append(rr)
            records += recs
            for q in recs:
                if q["status"] != "ok":
                    failures.append({"phase": "measured", "concurrency": level, "round": rnd,
                                     "round_id": rr["round_id"], "prompt_id": q["prompt_id"],
                                     "status": q["status"], "error": q["error"]})
                    print(f"    C={level} round {rnd} req {q['prompt_id']}: FAILED {q['status']}: {q['error']}", flush=True)
        s = summarize(rounds, records)[str(level)]
        print(f"  C={level}: aggregate {s['aggregate_tps']} tok/s, per stream {s['per_stream_tps']} tok/s, "
              f"ttft {s['ttft_s']} s, failed {s['requests_failed']}", flush=True)
    finish()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
