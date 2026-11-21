#!/usr/bin/env python3
"""Reproducible NON-STANDARD Strata Flash-Next long-context sweep.

Usage:
  python3 strata_long_context_bench.py --target 260000 --out PATH

Runs 3 synthetic speed+mid-needle requests (max 512 tokens) and
2 recall-only requests (10% and 90% needle; max 2048 tokens).
Uses EXACT sampling/thinking settings from Strata standard-v1 draft.1,
but the longer synthetic workloads are NON-STANDARD.
The official `strata bench run` is separate and not altered.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "http://127.0.0.1:18080"
MODEL = "qwen3.8-flash-next-q2_0"
EXPECTED = {32000: 32768, 64000: 65536, 128000: 131072, 260000: 262144}
PATTERN = "alpha beta gamma delta epsilon zeta eta theta"
VERSION = "strata-long-synthetic-thinking-xhigh-v3"
SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "reasoning_effort": "xhigh",
    "reasoning_budget_tokens": 0,
    "chat_template_kwargs": {
        "enable_thinking": True,
        "preserve_thinking": True,
    },
}
OUTPUT_MARGIN = 256


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def request_json(url: str, body=None, timeout=120):
    headers = {"Content-Type": "application/json"}
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(url, payload, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


_TOKENIZER = None
_TOKENIZER_PROVENANCE = {}


def init_tokenizer():
    """Load the bundled Strata tokenizer using the exact prepared model pack."""
    global _TOKENIZER, _TOKENIZER_PROVENANCE
    if _TOKENIZER is not None:
        return
    source = Path(os.environ.get("STRATA_TOKENIZER_SOURCE", "")).expanduser()
    folder = Path(os.environ.get("STRATA_TOKENIZER_DIR", "")).expanduser()
    if not source.is_file() or not folder.is_dir():
        raise RuntimeError("STOP: STRATA_TOKENIZER_SOURCE / STRATA_TOKENIZER_DIR missing; use shell launcher")
    spec = importlib.util.spec_from_file_location("strata_tokenizer", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("STOP: Cannot import bundled tokenizer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json")
    for name in files:
        if not (folder / name).is_file():
            raise RuntimeError(f"STOP: tokenizer pack file absent: {name}")
    vocab = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
    ids = [None] * len(vocab)
    for token, ident in vocab.items():
        if not isinstance(ident, int) or ident < 0 or ident >= len(ids) or ids[ident] is not None:
            raise RuntimeError("STOP: invalid tokenizer vocabulary mapping")
        ids[ident] = token
    if any(t is None for t in ids):
        raise RuntimeError("STOP: tokenizer vocabulary has missing token ids")
    cfg = json.loads((folder / "tokenizer.json").read_text(encoding="utf-8"))
    typ = json.loads((folder / "token_type.json").read_text(encoding="utf-8"))
    merges = (folder / "merges.txt").read_text(encoding="utf-8").splitlines()
    _TOKENIZER = module.Tokenizer(
        ids,
        merges,
        typ,
        pre=cfg.get("pre", "qwen35"),
        special_ids=cfg.get("special_ids") or {},
    )
    if len(ids) < 200000:
        raise RuntimeError(f"STOP: unexpected tokenizer vocab size: {len(ids)}")
    _TOKENIZER_PROVENANCE = {
        "vocabulary_size": len(ids),
        "pre_tokenizer": cfg.get("pre", "qwen35"),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "file_sha256": {
            name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in files
        },
    }
    print(
        f"LOCAL TOKENIZER: vocab={len(ids)}, pre={cfg.get('pre')}, source_sha256={_TOKENIZER_PROVENANCE['source_sha256'][:16]}...",
        flush=True,
    )


def tokenize(text: str) -> int:
    # Counts message content precisely with Strata's own BPE. The chat template
    # wrapper is accounted for separately using headroom + server-reported usage.
    init_tokenizer()
    return len(_TOKENIZER.encode(text))


def corpus(seed: int, rows: int, depth: int, secret: str) -> str:
    # A deterministic synthetic input. The per-trial seed prevents full-prompt
    # cache reuse while keeping the same semantic structure and tokenization.
    marker = f"Long context benchmark trial {seed}. Deterministic synthetic corpus.\n"
    cut = rows * depth // 100
    prefix = "".join(f"record-{seed:02d}-{i:06d}: {PATTERN}.\n" for i in range(cut))
    suffix = "".join(f"record-{seed:02d}-{i:06d}: {PATTERN}.\n" for i in range(cut, rows))
    needle = f"\nThe unique benchmark recall code is {secret}. Remember that exact code.\n"
    task = ("\nReport the unique benchmark recall code on the first line, without changing it. "
            "Then explain how to retrieve facts reliably from a long technical document "
            "in a concise paragraph. Do not invent other codes.")
    return marker + prefix + needle + suffix + task


def fit(seed: int, depth: int, secret: str, target: int, ctx: int, maxout: int):
    upper = min(target, ctx - maxout - OUTPUT_MARGIN)
    if upper < 1000:
        raise RuntimeError("STOP: configured context is insufficient")
    # Probe 512 records to estimate line token density, then binary search by
    # complete records. We preserve a margin for the server chat template.
    probe = corpus(seed, 512, depth, secret)
    per = max(1.0, (tokenize(probe) - 140) / 512)
    lo, hi = 1, min(150000, int(upper / per * 1.35 + 1000))
    best = None
    best_n = 0
    best_rows = 0
    while lo <= hi:
        rows = (lo + hi) // 2
        prompt = corpus(seed, rows, depth, secret)
        n = tokenize(prompt)
        if n <= upper:
            best, best_n, best_rows = prompt, n, rows
            lo = rows + 1
        else:
            hi = rows - 1
    if not best or best_n < upper * 0.97:
        raise RuntimeError(
            f"STOP: target sizing failed: tokenized={best_n}, target={target}, rows={best_rows}"
        )
    return best, best_n, best_rows


def stream_chat(prompt: str, cap: int, seed: int):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": cap,
        "seed": seed,
        **SAMPLING,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
    )
    first = None
    first_answer = None
    first_kind = None
    text = []
    reasoning_parts = 0
    usage, timings = None, None
    finish = None
    done = False
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}")
        for raw in resp:
            if not raw.startswith(b"data:"):
                continue
            chunk = raw[5:].strip()
            if chunk == b"[DONE]":
                done = True
                break
            event = json.loads(chunk)
            if "error" in event:
                raise RuntimeError(f"STREAM ERROR: {str(event['error'])[:300]}")
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            if isinstance(event.get("timings"), dict):
                timings = event["timings"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                r, c = delta.get("reasoning_content"), delta.get("content")
                if first is None and (r or c or delta.get("tool_calls")):
                    first = time.monotonic() - start
                    first_kind = "reasoning" if r else ("answer" if c else "tool_call")
                if isinstance(c, str) and c:
                    if first_answer is None:
                        first_answer = time.monotonic() - start
                    text.append(c)
                if isinstance(r, str):
                    reasoning_parts += len(r)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    wall = time.monotonic() - start
    if not done or finish not in ("stop", "length", "tool_calls"):
        raise RuntimeError(f"STOP: incomplete stream; done={done} finish_reason={finish}")
    usage = usage or {}
    timings = timings or {}
    cache = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {
        "usage": usage,
        "timings": timings,
        "finish_reason": finish,
        "ttft_s": round(first, 4) if first is not None else None,
        "first_answer_s": round(first_answer, 4) if first_answer is not None else None,
        "first_delta_kind": first_kind,
        "wall_s": round(wall, 4),
        "content": "".join(text),
        "reasoning_chars": reasoning_parts,
        "prompt_tok_s": timings.get("prompt_per_second"),
        "decode_tok_s": timings.get("predicted_per_second"),
        "cached_tokens": cache,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, required=True,
                    choices=sorted(EXPECTED), help="nominal input tokens")
    ap.add_argument("--out", type=Path, required=True, help="existing empty experiment directory")
    args = ap.parse_args()
    dest = args.out.resolve()
    if not dest.is_dir():
        raise SystemExit("STOP: --out directory does not exist")
    if any(dest.iterdir()):
        raise SystemExit("STOP: --out directory is not empty")
    target, expected_ctx = args.target, EXPECTED[args.target]
    metrics = request_json(BASE + "/metrics", timeout=30)
    engine = metrics.get("engine") or {}
    context = engine.get("max_context")
    if context != expected_ctx:
        raise SystemExit(f"STOP: engine context {context} != required {expected_ctx}")
    if engine.get("model") != MODEL or engine.get("drafting") != "off":
        raise SystemExit("STOP: expected Q2_0 with drafting OFF")
    print(f"Context={context} Target input={target} Drafting OFF; THINKING ON xhigh", flush=True)
    # Do not count the chat wrapper as exactly known without server usage;
    # use a conservative allowance and report both counts.
    n = tokenize("Tokenizer preflight " + PATTERN)
    if n < 3:
        raise SystemExit("STOP: tokenizer preflight failed")
    manifest = {
        "kind": VERSION,
        "status": "in_progress",
        "started_utc": utc(),
        "model": MODEL,
        "engine": engine,
        "tokenizer": _TOKENIZER_PROVENANCE,
        "target_input_tokens": target,
        "context_tokens": context,
        "backend": "CUDA",
        "gpu_unlock": "ON, checked by calling shell wrapper",
        "drafting": "off",
        "sampling": SAMPLING,
        "speed_output_cap": 512,
        "recall_output_cap": 2048,
        "speed_repeats": 3,
        "recall_depths": [10, 90],
        "note": ("Same thinking and sampling as Strata standard draft.1. "
                 "Different prompts/lengths; NONSTANDARD LONG-CONTEXT extension. "
                 "First-to-answer latency may be null if output cap is reached while reasoning."),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    rows = []
    output = dest / "long-results.jsonl"
    with output.open("x") as fp:
        for case, seed, depth, cap in [
            ("speed-r1", 1, 50, 512),
            ("speed-r2", 2, 50, 512),
            ("speed-r3", 3, 50, 512),
            ("recall-10", 4, 10, 2048),
            ("recall-90", 5, 90, 2048),
        ]:
            code = f"BENCH-{seed:02d}-{(700 + seed * 17)}-XR"
            effective_target = min(target, context - cap - OUTPUT_MARGIN)
            print(
                f"===== {case}: requested input={target}, maximum safe local input={effective_target}, "
                f"output cap={cap}, thinking=xhigh =====",
                flush=True,
            )
            prompt, estimated, count = fit(seed, depth, code, target, context, cap)
            print(f"Sending {case}: tokenized={estimated}, records={count}", flush=True)
            try:
                resp = stream_chat(prompt, cap, seed)
                used = resp["usage"].get("prompt_tokens")
                if (not isinstance(used, int) or used + cap > context - 64 or
                        abs(used - estimated) > 128 or
                        not effective_target * 0.97 <= used <= effective_target + 128):
                    raise RuntimeError(
                        f"STOP: engine/local token counts or context budget disagree: "
                        f"server={used}, local={estimated}, safe_target={effective_target}, "
                        f"output_cap={cap}, context={context}"
                    )
                answer = resp.pop("content")
                row = {
                    "case": case, "seed": seed, "depth_pct": depth,
                    "target_tokens": target, "effective_target_tokens": effective_target,
                    "tokenized_tokens": estimated, "actual_prompt_tokens": used,
                    "records": count, "output_cap": cap,
                    "correct_code": code,
                    "found": code in answer,
                    "final_answer": answer[:12000],
                    "answer_length_chars": len(answer),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    **resp,
                }
            except Exception as ex:
                row = {
                    "case": case, "seed": seed, "target_tokens": target,
                    "depth_pct": depth, "status": "failed", "error": str(ex)[:500],
                }
                fp.write(json.dumps(row) + "\n")
                fp.flush()
                print(f"STOP: {case} failed: {ex}", file=sys.stderr, flush=True)
                raise
            row["status"] = "complete"
            rows.append(row)
            fp.write(json.dumps(row) + "\n")
            fp.flush()
            print(
                f"{case}: read={row['actual_prompt_tokens']} tok, "
                f"prefill={row.get('prompt_tok_s')} tok/s, "
                f"decode={row.get('decode_tok_s')} tok/s, "
                f"TTFT={row.get('ttft_s')} s, "
                f"finish={row['finish_reason']}, "
                f"first_answer={row['first_answer_s']}, "
                f"code_found={'YES' if row['found'] else 'NO'}",
                flush=True
            )
    speeds = [r for r in rows if r["case"].startswith("speed")]
    recalls = [r for r in rows if r["case"].startswith("recall")]
    summary = {
        "kind": VERSION,
        "status": "complete",
        "context": context,
        "target": target,
        "cases": len(rows),
        "recall_passed": sum(r["found"] for r in recalls),
        "recall_total": len(recalls),
        "speed_cases_with_answer": sum(r["found"] for r in speeds),
        "speed_cases_capped": sum(r["finish_reason"] == "length" for r in speeds),
        "recall_cases_capped": sum(r["finish_reason"] == "length" for r in recalls),
        "sampling": SAMPLING,
        "speed_repetitions": 3,
        "median_prefill_tps": statistics.median(
            [x["prompt_tok_s"] for x in speeds if isinstance(x["prompt_tok_s"], (int, float))]
        ) if all(isinstance(x["prompt_tok_s"], (int, float)) for x in speeds) else None,
        "median_decode_tps": statistics.median(
            [x["decode_tok_s"] for x in speeds if isinstance(x["decode_tok_s"], (int, float))]
        ) if all(isinstance(x["decode_tok_s"], (int, float)) for x in speeds) else None,
        "median_ttft_s": statistics.median(
            [x["ttft_s"] for x in speeds if isinstance(x["ttft_s"], (int, float))]
        ) if all(isinstance(x["ttft_s"], (int, float)) for x in speeds) else None,
        "model": MODEL,
        "finished_utc": utc(),
        "note": ("Thinking xhigh; sampling as in official Strata draft.1. "
                 "NONSTANDARD prompts/lengths; speeds include reasoning tokens; "
                 "keep all runs and quality failures."),
    }
    (dest / "summary.json").write_text(json.dumps(summary, indent=2))
    manifest["status"] = "complete"
    manifest["finished_utc"] = utc()
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("===== LONG CONTEXT TEST COMPLETE =====")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("INTERRUPTED: partial evidence preserved", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
