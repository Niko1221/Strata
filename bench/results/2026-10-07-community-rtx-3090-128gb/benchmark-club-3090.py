#!/usr/bin/env python3
"""Serial, fresh-prompt benchmark against a llama.cpp (club-3090) OpenAI server.

Same synthetic code-explanation prompts and per-request nonce as benchmark.py, but
built for llama.cpp: prompt sizing via /tokenize, prefill/decode tok/s from the
server's non-stream `timings` object, TTFT from a streaming request, and a
contamination guard based on /metrics `requests_processing` (llama.cpp has no
/v1/status). Run against a dedicated, loopback-bound container.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def post(url, body, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get_text(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def metrics(url):
    try:
        text = get_text(url + "/metrics")
    except urllib.error.HTTPError:
        return {}
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, val = line.partition(" ")
        try:
            out[name.split("{")[0]] = float(val)
        except ValueError:
            pass
    return out


def tokenize(url, text):
    try:
        r = post(url + "/tokenize", {"content": text, "add_special": False, "parse_special": True}, timeout=120)
        return len(r["tokens"]) if isinstance(r, dict) else len(r)
    except urllib.error.HTTPError:
        return max(1, int(len(text) / 3.2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--out", type=Path, default=Path("club3090"))
    ap.add_argument("--targets", default="4096,32768,128000")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    url = args.url.rstrip("/")

    props = json.loads(get_text(url + "/props"))
    (args.out / "props.json").write_text(json.dumps(props, indent=2) + "\n")
    n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx") or 0

    def request(content, maximum=256):
        return {"model": args.model, "messages": [{"role": "user", "content": content}],
                "temperature": 0, "max_tokens": maximum, "stream": False}

    filler = "\n".join(f"def task_{i:05d}(value: int) -> int: return (value * {(i % 97) + 1} + {i}) % 100003"
                       for i in range(12000))
    ending = ("\n\nWrite a detailed explanation of the code above. Discuss deterministic integer transforms, "
              "modulo arithmetic, testing, naming, complexity, and maintainability. Write at least 600 words.")

    rows = []

    def stream_ttft(label, req):
        body = dict(req, stream=True, stream_options={"include_usage": True})
        started = time.perf_counter()
        first, usage = None, {}
        wire = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(wire, timeout=1800) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("choices") and (chunk["choices"][0].get("delta") or {}).get("content"):
                    first = first if first is not None else time.perf_counter() - started
                if chunk.get("usage"):
                    usage = chunk["usage"]
        return first, time.perf_counter() - started, usage

    def perform(label, stream_req, timing_req):
        stream_body = json.dumps(stream_req).encode()
        timing_body = json.dumps(timing_req).encode()
        (args.out / (label + "-stream-request.json")).write_bytes(stream_body + b"\n")
        (args.out / (label + "-request.json")).write_bytes(timing_body + b"\n")
        before = metrics(url)
        ttft, elapsed, stream_usage = stream_ttft(label, stream_req)
        ns = post(url + "/v1/chat/completions", timing_req)
        after = metrics(url)
        timings = ns.get("timings") or {}
        usage = ns.get("usage") or stream_usage
        prompt_n = timings.get("prompt_n") or usage.get("prompt_tokens")
        gen_n = timings.get("predicted_n") or usage.get("completion_tokens")
        prompt_ms = timings.get("prompt_ms")
        predicted_ms = timings.get("predicted_ms")
        row = {"label": label, "stream_request_sha256": hashlib.sha256(stream_body).hexdigest(),
               "request_sha256": hashlib.sha256(timing_body).hexdigest(),
               "prompt_tokens": prompt_n, "generated_tokens": gen_n,
               "client_ttft_s": ttft, "client_elapsed_s": elapsed,
               "prompt_ms": prompt_ms, "predicted_ms": predicted_ms,
               "prefill_tok_s": timings.get("prompt_per_second") or ((prompt_n / (prompt_ms / 1000)) if prompt_ms else None),
               "decode_tok_s": timings.get("predicted_per_second") or ((gen_n / (predicted_ms / 1000)) if predicted_ms else None),
               "draft_n": timings.get("draft_n"), "draft_n_accepted": timings.get("draft_n_accepted"),
               "metrics_delta": {"tokens_evaluated": after.get("llamacpp:tokens_evaluated_total", 0) - before.get("llamacpp:tokens_evaluated_total", 0),
                                 "predictions": after.get("llamacpp:predictions_total", 0) - before.get("llamacpp:predictions_total", 0)},
               "text": (ns.get("choices") or [{}])[0].get("message", {}).get("content", "")[:400]}
        rows.append(row)
        (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        print("done", label, json.dumps({"prompt": prompt_n, "gen": gen_n, "ttft_s": ttft,
              "prefill": row["prefill_tok_s"], "decode": row["decode_tok_s"]}), flush=True)
        return row

    before_metrics = metrics(url)
    if "llamacpp:requests_processing" in before_metrics and before_metrics["llamacpp:requests_processing"]:
        raise SystemExit(f"server not idle before sweep: requests_processing={before_metrics['llamacpp:requests_processing']}")

    perform("warmup", request("Reply with exactly the word READY."), request("Reply with exactly the word GO."))
    for target in map(int, args.targets.split(",")):
        if n_ctx and target + 400 > n_ctx:
            print(f"skip target {target}: n_ctx {n_ctx}", flush=True)
            continue
        for run in range(1, args.runs + 1):
            prefix = f"Benchmark nonce: series-{target}-trial-{run}.\nReview this synthetic Python module:\n"
            lo, hi = 0, len(filler)
            while lo < hi:
                middle = (lo + hi + 1) // 2
                if tokenize(url, prefix + filler[:middle] + ending) <= target - 64:
                    lo = middle
                else:
                    hi = middle - 1
            body = filler[:lo] + ending
            stream_req = request(f"Benchmark nonce: series-{target}-trial-{run}-stream.\nReview this synthetic Python module:\n" + body)
            timing_req = request(f"Benchmark nonce: series-{target}-trial-{run}-timing.\nReview this synthetic Python module:\n" + body)
            perform(f"tokens-{target}-run-{run}", stream_req, timing_req)

    after_metrics = metrics(url)
    if "llamacpp:requests_processing" in after_metrics and after_metrics["llamacpp:requests_processing"]:
        raise SystemExit(f"server not idle after sweep: requests_processing={after_metrics['llamacpp:requests_processing']}")

    measured = [r for r in rows if r["label"] != "warmup"]
    groups = {}
    for target in map(int, args.targets.split(",")):
        subset = [r for r in measured if r["label"].startswith(f"tokens-{target}-")]
        if not subset:
            continue
        vals = {
            "prompt_tokens": [r["prompt_tokens"] for r in subset],
            "generated_tokens": [r["generated_tokens"] for r in subset],
            "client_ttft_s": [r["client_ttft_s"] for r in subset],
            "client_elapsed_s": [r["client_elapsed_s"] for r in subset],
            "prefill_tok_s": [r["prefill_tok_s"] for r in subset if r["prefill_tok_s"]],
            "decode_tok_s": [r["decode_tok_s"] for r in subset if r["decode_tok_s"]],
        }
        groups[str(target)] = {k: {"median": statistics.median(v), "min": min(v), "max": max(v)}
                               for k, v in vals.items() if v}
    (args.out / "summary.json").write_text(json.dumps(groups, indent=2) + "\n")
    print("summary", json.dumps(groups), flush=True)


if __name__ == "__main__":
    main()
