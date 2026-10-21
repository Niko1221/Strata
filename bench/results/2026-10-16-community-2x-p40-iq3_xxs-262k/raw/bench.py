#!/usr/bin/env python3
"""
flash-next-bench/bench.py — fixed-protocol throughput benchmark for llama.cpp-compatible
OpenAI chat endpoints (direct backend or through the smart proxy). Stdlib only.

Measures, per prompt size and concurrency level:
  - prefill tok/s and decode tok/s (from llama.cpp's `timings` block in non-stream responses)
  - TTFT (wall clock to first streamed delta, separate streaming pass)
  - aggregate decode tok/s at concurrency N (sum of predicted tokens / wall)
  - host RSS of the serving process(es), nvidia-smi, free, load, and the backend's
    `requests_processing` gauge before/after each run (so external load is on record)

Prompt caching is defeated two ways: a per-request nonce at the START of the prompt and
`cache_prompt: false`. `ignore_eos: true` makes every request generate exactly max_tokens.

Output: <out-dir>/<label>/<UTC stamp>/{config.json, runs.jsonl, summary.json, summary.md, env/*}
"""
import argparse, json, os, subprocess, sys, threading, time, urllib.request, uuid
from datetime import datetime, timezone

PARAGRAPHS = [
    "The maintenance crew arrived before dawn to inspect the pumping station. Each valve was logged, "
    "each gauge photographed, and the readings were compared against the previous quarter. Two of the "
    "pressure relief lines showed corrosion that had not been present in the spring survey.",
    "A library catalog is only useful if its entries are consistent. When three cataloguers describe the "
    "same pamphlet in three different ways, the search index fragments and patrons stop trusting it. "
    "The remedy is a written style guide, applied without exception, and reviewed once a year.",
    "The ferry schedule changes with the tides. On days when the low water falls at midday, the noon "
    "crossing is cancelled and the freight is held until the evening run. Regular passengers know this "
    "and plan around it, but visitors are often caught out and left waiting on the quay.",
    "Bread dough develops its structure through time as much as through kneading. A slow, cool rise "
    "lets enzymes break down starches into sugars, which feed the yeast and darken the crust. Rushing "
    "the process with extra yeast produces a loaf that looks right but tastes flat.",
    "The survey team mapped the old orchard by walking its boundaries with a measuring wheel. Where the "
    "stone wall had collapsed they used the line of surviving pear trees as a guide. The resulting plan "
    "differed from the deed by almost four metres on the northern edge.",
    "A good incident report answers four questions: what happened, when it was noticed, what was done, "
    "and what will prevent it next time. Reports that assign blame instead of describing the sequence "
    "of events are rarely read a second time and almost never change anything.",
]

def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def http_json(url, payload=None, timeout=600, method=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"))
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

_WRAPPER_TOKENS = {}

def tokenize_count(tokenize_url, text):
    """Body-only token count. llama.cpp: POST /tokenize. Strata (no /tokenize): POST /v1/messages/count_tokens, which
    counts the chat-templated message, minus the count of an empty message (the template wrapper) — same semantics."""
    if tokenize_url.rstrip("/").endswith("/count_tokens"):
        def count(s):
            return int(http_json(tokenize_url, {"model": "any", "messages": [{"role": "user", "content": s}]})["input_tokens"])
        if tokenize_url not in _WRAPPER_TOKENS:
            _WRAPPER_TOKENS[tokenize_url] = count("")
        return max(0, count(text) - _WRAPPER_TOKENS[tokenize_url])
    return len(http_json(tokenize_url, {"content": text})["tokens"])

def build_prompt_body(target_tokens, tokenize_url):
    """Deterministic filler sized to ~target_tokens (±3%) using the backend tokenizer."""
    if target_tokens <= 64:
        return "Explain in one paragraph why measurements should be dated."
    unit = " ".join(PARAGRAPHS)
    unit_tok = tokenize_count(tokenize_url, unit)
    n = max(1, int(target_tokens / unit_tok))
    parts = []
    for i in range(n):
        parts.append(f"Section {i+1}. " + unit)
    body = "\n\n".join(parts)
    tok = tokenize_count(tokenize_url, body)
    guard = 0
    while abs(tok - target_tokens) > 0.03 * target_tokens and guard < 60:
        guard += 1
        if tok > target_tokens:
            words = body.split(" ")
            drop = int(len(words) * (tok - target_tokens) / tok)
            body = " ".join(words[:max(1, len(words) - max(1, drop))])
        else:
            body += "\n\n" + PARAGRAPHS[guard % len(PARAGRAPHS)]
        tok = tokenize_count(tokenize_url, body)
    return body + "\n\nSummarize the material above in detail, section by section."

def snapshot_env(rss_pids, metrics_url, slots_url=None):
    snap = {"t": utc_stamp()}
    try:
        snap["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:
        snap["nvidia_smi"] = f"error: {e}"
    try:
        snap["free_m"] = subprocess.run(["free", "-m"], capture_output=True, text=True, timeout=5).stdout.strip()
        snap["loadavg"] = open("/proc/loadavg").read().strip()
    except Exception as e:
        snap["free_m"] = f"error: {e}"
    rss = {}
    for pid in rss_pids:
        try:
            for line in open(f"/proc/{pid}/status"):
                if line.startswith(("VmRSS", "VmLck", "VmSwap")):
                    k, v = line.split(":", 1)
                    rss.setdefault(str(pid), {})[k] = v.strip()
        except Exception as e:
            rss[str(pid)] = f"error: {e}"
    snap["rss"] = rss
    if metrics_url:
        try:
            req = urllib.request.Request(metrics_url)
            with urllib.request.urlopen(req, timeout=5) as r:
                for line in r.read().decode().splitlines():
                    if line.startswith("llamacpp:requests_processing"):
                        snap["backend_requests_processing"] = float(line.split()[1])
        except Exception as e:
            snap["backend_requests_processing"] = f"error: {e}"
    if slots_url and not isinstance(snap.get("backend_requests_processing"), float):
        # backends without the llamacpp:requests_processing metric (Strata): count slots that are processing
        try:
            slots = http_json(slots_url, timeout=5)
            snap["backend_requests_processing"] = float(sum(1 for s in slots if s.get("is_processing")))
        except Exception as e:
            snap["backend_requests_processing"] = f"error: {e}"
    return snap

SAMPLING = {"temperature": 0.0}   # --temperature: the request's sampling fields (none = omit, so the server's defaults apply)

def one_nonstream(url, model, prompt, max_tokens, timeout):
    nonce = uuid.uuid4().hex
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"[run {nonce}]\n{prompt}"}],
        "max_tokens": max_tokens, **SAMPLING, "stream": False,
        "cache_prompt": False, "ignore_eos": True,
    }
    t0 = time.monotonic()
    resp = http_json(url + "/v1/chat/completions", payload, timeout=timeout)
    wall = time.monotonic() - t0
    tm = resp.get("timings", {})
    usage = resp.get("usage", {})
    return {
        "nonce": nonce, "wall_s": round(wall, 3),
        "prompt_n": tm.get("prompt_n", usage.get("prompt_tokens")),
        "prompt_ms": tm.get("prompt_ms"), "prefill_tps": tm.get("prompt_per_second"),
        "predicted_n": tm.get("predicted_n", usage.get("completion_tokens")),
        "predicted_ms": tm.get("predicted_ms"), "decode_tps": tm.get("predicted_per_second"),
        "finish_reason": (resp.get("choices") or [{}])[0].get("finish_reason"),
        "timings_present": bool(tm),
        # speculative decoding (present only when the server runs a draft / MTP): drafted and accepted token counts
        "draft_n": tm.get("draft_n"), "draft_n_accepted": tm.get("draft_n_accepted"),
    }

def one_stream_ttft(url, model, prompt, max_tokens, timeout):
    nonce = uuid.uuid4().hex
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"[run {nonce}]\n{prompt}"}],
        "max_tokens": max_tokens, **SAMPLING, "stream": True,
        "cache_prompt": False, "ignore_eos": True,
    }
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    t0 = time.monotonic(); ttft = None; chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except Exception:
                continue
            delta = (obj.get("choices") or [{}])[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                chunks += 1
                if ttft is None:
                    ttft = time.monotonic() - t0
    return {"nonce": nonce, "ttft_s": None if ttft is None else round(ttft, 3),
            "wall_s": round(time.monotonic() - t0, 3), "content_chunks": chunks}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="e.g. http://localhost:8096 or http://localhost:4000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenize-url", required=True, help="backend /tokenize used only to size prompts")
    ap.add_argument("--metrics-url", default=None, help="backend /metrics for requests_processing")
    ap.add_argument("--slots-url", default=None, help="backend /slots; used for external load when /metrics lacks llamacpp:requests_processing (Strata)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--sizes", default="256:2,4096:2,16384:1", help="prompt_tokens:runs,...")
    ap.add_argument("--concurrency", default="1", help="comma list, e.g. 1,4")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--ttft-size", type=int, default=256, help="prompt size for the streaming TTFT pass")
    ap.add_argument("--rss-pids", default="", help="comma list of PIDs to sample RSS for")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--note", default="")
    ap.add_argument("--temperature", default="0.0", help="request temperature; 'none' omits the field so the server's configured sampling applies")
    args = ap.parse_args()
    SAMPLING.clear()
    if args.temperature != "none":
        SAMPLING["temperature"] = float(args.temperature)

    rss_pids = [int(p) for p in args.rss_pids.split(",") if p.strip()]
    stamp = utc_stamp()
    run_dir = os.path.join(args.out_dir, args.label, stamp)
    os.makedirs(os.path.join(run_dir, "env"), exist_ok=True)
    cfg = vars(args) | {"stamp": stamp, "argv": sys.argv}
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=2)

    # provenance
    for name, url in (("props.json", args.base_url + "/props"), ("models.json", args.base_url + "/v1/models")):
        try:
            json.dump(http_json(url, timeout=10), open(os.path.join(run_dir, "env", name), "w"), indent=2)
        except Exception as e:
            open(os.path.join(run_dir, "env", name), "w").write(json.dumps({"error": str(e)}))
    for pid in rss_pids:
        try:
            open(os.path.join(run_dir, "env", f"cmdline-{pid}.txt"), "w").write(
                open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode())
        except Exception:
            pass

    json.dump(snapshot_env(rss_pids, args.metrics_url, args.slots_url), open(os.path.join(run_dir, "env", "before_all.json"), "w"), indent=2)

    # warm-up (pages the model in; discarded)
    print("[bench] warm-up ...", flush=True)
    w = one_nonstream(args.base_url, args.model, "Say hello.", 16, args.timeout)
    print(f"[bench] warm-up: {w}", flush=True)
    json.dump(snapshot_env(rss_pids, args.metrics_url, args.slots_url), open(os.path.join(run_dir, "env", "after_warmup.json"), "w"), indent=2)

    sizes = [(int(s.split(":")[0]), int(s.split(":")[1]) if ":" in s else 1) for s in args.sizes.split(",")]
    concs = [int(c) for c in args.concurrency.split(",")]
    prompts = {}
    for sz, _ in sizes:
        prompts[sz] = build_prompt_body(sz, args.tokenize_url)
        print(f"[bench] prompt {sz}: {tokenize_count(args.tokenize_url, prompts[sz])} tokens (body only)", flush=True)

    runs_f = open(os.path.join(run_dir, "runs.jsonl"), "a")
    summary = {"label": args.label, "stamp": stamp, "model": args.model, "base_url": args.base_url,
               "max_tokens": args.max_tokens, "note": args.note, "results": []}

    for conc in concs:
        for sz, nruns in sizes:
            for rep in range(nruns):
                before = snapshot_env(rss_pids, args.metrics_url, args.slots_url)
                results = [None] * conc
                def worker(i):
                    try:
                        results[i] = one_nonstream(args.base_url, args.model, prompts[sz], args.max_tokens, args.timeout)
                    except Exception as e:
                        results[i] = {"error": str(e)}
                t0 = time.monotonic()
                threads = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
                [t.start() for t in threads]; [t.join() for t in threads]
                wall = time.monotonic() - t0
                after = snapshot_env(rss_pids, args.metrics_url, args.slots_url)
                ok = [r for r in results if r and "error" not in r]
                pred_total = sum((r.get("predicted_n") or 0) for r in ok)
                rec = {
                    "kind": "nonstream", "concurrency": conc, "prompt_target": sz, "rep": rep,
                    "wall_s": round(wall, 3), "streams": results,
                    "aggregate_decode_tps_wall": round(pred_total / wall, 3) if wall and pred_total else None,
                    "per_stream_prompt_n": [r.get("prompt_n") for r in ok],
                    "per_stream_decode_tps": [r.get("decode_tps") for r in ok],
                    "per_stream_prefill_tps": [r.get("prefill_tps") for r in ok],
                    "per_stream_draft_accept": [
                        (round(r["draft_n_accepted"] / r["draft_n"], 3) if r.get("draft_n") else None) for r in ok],
                    "ext_load_before": before.get("backend_requests_processing"),
                    "ext_load_after": after.get("backend_requests_processing"),
                    "load_before": before.get("loadavg"), "load_after": after.get("loadavg"),
                    "rss_after": after.get("rss"), "nvidia_after": after.get("nvidia_smi"),
                }
                runs_f.write(json.dumps(rec) + "\n"); runs_f.flush()
                summary["results"].append({k: rec[k] for k in rec if k not in ("streams",)} | {"errors": [r for r in results if r and "error" in r]})
                print(f"[bench] conc={conc} size={sz} rep={rep}: wall={wall:.1f}s "
                      f"prefill={rec['per_stream_prefill_tps']} decode={rec['per_stream_decode_tps']} "
                      f"agg={rec['aggregate_decode_tps_wall']} ext_load={rec['ext_load_before']}->{rec['ext_load_after']}"
                      f" draft_accept={rec['per_stream_draft_accept']}", flush=True)

    # TTFT streaming pass (single stream)
    ttft_prompt = prompts.get(args.ttft_size) or build_prompt_body(args.ttft_size, args.tokenize_url)
    ttfts = []
    for rep in range(2):
        try:
            r = one_stream_ttft(args.base_url, args.model, ttft_prompt, 64, args.timeout)
        except Exception as e:
            r = {"error": str(e)}
        r |= {"kind": "stream_ttft", "prompt_target": args.ttft_size, "rep": rep}
        ttfts.append(r); runs_f.write(json.dumps(r) + "\n"); runs_f.flush()
        print(f"[bench] ttft rep={rep}: {r}", flush=True)
    summary["ttft"] = ttfts
    json.dump(snapshot_env(rss_pids, args.metrics_url, args.slots_url), open(os.path.join(run_dir, "env", "after_all.json"), "w"), indent=2)
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)

    # markdown summary
    lines = [f"# {args.label} — {stamp}", "", f"model `{args.model}` via `{args.base_url}`; max_tokens {args.max_tokens}; "
             f"temperature {args.temperature}; cache_prompt false; ignore_eos true. {args.note}", "",
             "| conc | prompt | rep | prompt_n | prefill tok/s | decode tok/s (per stream) | aggregate decode tok/s (wall) | wall s | ext load before→after | draft accept |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["results"]:
        pn = ", ".join(str(x) for x in r.get("per_stream_prompt_n", []) if x is not None)
        da = ", ".join(f"{x:.3f}" for x in r.get("per_stream_draft_accept", []) if x is not None) or "—"
        lines.append(f"| {r['concurrency']} | {r['prompt_target']} | {r['rep']} | {pn} | "
                     f"{', '.join(f'{x:.1f}' for x in r['per_stream_prefill_tps'] if x is not None)} | "
                     f"{', '.join(f'{x:.2f}' for x in r['per_stream_decode_tps'] if x is not None)} | "
                     f"{r['aggregate_decode_tps_wall']} | {r['wall_s']} | {r['ext_load_before']}→{r['ext_load_after']} | {da} |")
    lines += ["", "TTFT (streaming, 64 tokens): " + ", ".join(
        f"{t.get('ttft_s')} s" if "error" not in t else f"error: {t['error']}" for t in ttfts)]
    open(os.path.join(run_dir, "summary.md"), "w").write("\n".join(lines) + "\n")
    print(f"[bench] done -> {run_dir}", flush=True)

if __name__ == "__main__":
    main()
