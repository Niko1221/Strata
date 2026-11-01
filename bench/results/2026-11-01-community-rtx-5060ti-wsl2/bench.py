#!/usr/bin/env python3
"""Strata benchmark: prefill (prompt) speed, decode speed, TTFT, concurrency.

Pure stdlib. Talks to an OpenAI-compatible /v1/chat/completions with SSE streaming.
"""
import argparse, json, statistics, sys, threading, time, urllib.request, uuid

BASE_TEXT = (
    "The quick brown fox jumps over the lazy dog while the engineer reads a long log file. "
    "Every token in this document is part of a synthetic corpus used to measure prompt reading speed. "
    "Strata keeps the model's experts in RAM and the busiest ones on the graphics card; the rest are computed "
    "on the processor in parallel with the GPU. This paragraph repeats with a unique marker so that the prompt "
    "cache never serves it twice. "
)


def make_prompt(target_tokens: int, nonce: str) -> str:
    # ~4 chars per token for this English corpus
    need = max(1, target_tokens * 4)
    body = (BASE_TEXT * (need // len(BASE_TEXT) + 1))[:need]
    return f"[run {nonce}]\n{body}\n\nSummarize the text above in one short sentence."


def stream_once(url, model, prompt, max_tokens, timeout=1800):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
    )
    t0 = time.time()
    ttft = None
    ntok = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        buf = b""
        while True:
            chunk = r.read(1)
            if not chunk:
                break
            buf += chunk
            if not buf.endswith(b"\n"):
                continue
            line = buf.strip()
            buf = b""
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                d = ch.get("delta") or {}
                for key in ("reasoning_content", "content"):
                    if d.get(key):
                        if ttft is None:
                            ttft = time.time() - t0
                        ntok += 1
                        break
    total = time.time() - t0
    return {
        "ttft_s": ttft if ttft is not None else total,
        "total_s": total,
        "completion_tokens": usage.get("completion_tokens", ntok),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "est_prompt_tokens": len(prompt) // 4,
    }


def one(args, target_tokens, idx):
    p = make_prompt(target_tokens, f"{uuid.uuid4().hex}-{idx}")
    r = stream_once(args.url, args.model, p, args.max_tokens)
    pt = r["prompt_tokens"] or r["est_prompt_tokens"]
    decode_s = max(1e-6, r["total_s"] - r["ttft_s"])
    return {
        "prompt_tokens": pt,
        "ttft_s": round(r["ttft_s"], 3),
        "total_s": round(r["total_s"], 3),
        "prefill_tps": round(pt / max(r["ttft_s"], 1e-6), 1),
        "decode_tps": round(max(0, r["completion_tokens"] - 1) / decode_s, 1),
        "completion_tokens": r["completion_tokens"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--model", default="qwen3.8-flash-next-ud-q4_k_xl")
    ap.add_argument("--prompt-tokens", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows, agg = [], []
    for rep in range(args.repeats):
        t0 = time.time()
        results = [None] * args.concurrency
        errs = []

        def work(i):
            try:
                results[i] = one(args, args.prompt_tokens, f"{rep}-{i}")
            except Exception as e:  # noqa
                errs.append(repr(e))

        ths = [threading.Thread(target=work, args=(i,)) for i in range(args.concurrency)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        wall = time.time() - t0
        ok = [r for r in results if r]
        if not ok:
            print(json.dumps({"label": args.label, "error": errs[:2]}), flush=True)
            continue
        row = {
            "label": args.label,
            "prompt_tokens": ok[0]["prompt_tokens"],
            "concurrency": args.concurrency,
            "rep": rep,
            "prefill_tps_median": statistics.median([r["prefill_tps"] for r in ok]),
            "decode_tps_median": statistics.median([r["decode_tps"] for r in ok]),
            "ttft_s_median": statistics.median([r["ttft_s"] for r in ok]),
            "aggregate_decode_tps": round(sum(r["completion_tokens"] for r in ok) / max(wall, 1e-6), 1),
            "errors": errs[:2],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    if rows and args.out:
        with open(args.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
