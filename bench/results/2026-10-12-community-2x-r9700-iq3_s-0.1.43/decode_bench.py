#!/usr/bin/env python3
"""Decode-speed benchmark for any OpenAI-compatible server (llama-server, luce_server, Strata).

    python3 decode_bench.py --url http://127.0.0.1:8080/v1 --model qwen3.8-flash-next-iq3_s [--temps 0,1.0] [--runs 2]
        [--json decode.json]

For each prompt kind (prose, code, edit, reasoning) it streams one chat completion and reports
(completion tokens - 1) / (time of last streamed token - time of first), so prefill is excluded. --context-file (plain
or .gz) prepends a long document (same prefix every request, so servers with prefix caching read it once) to test
decode speed at long context. Thinking stays on (the model default); reasoning tokens count as generated tokens.

Every stream is validated by bench_stream.py (no error event, finish_reason stop/length, usage.completion_tokens,
[DONE]). Failed requests are printed and recorded under "failures" and never enter a median. Exit status 1 if any
request failed; 2 (usage error, nothing sent) if --runs or --max-tokens is not positive. PROMPTS is also imported by
tune_bench.py and conc_bench.py.
"""
import argparse
import gzip
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from bench_stream import positive_int, public, stream  # noqa: E402

EDIT_SOURCE = '''def parse_amount(text):
    """Parse '42.50 USD' into (Decimal, str)."""
    number, commodity = text.split(" ")
    if not commodity.isupper() or len(commodity) != 3:
        raise ValueError(f"bad commodity: {commodity!r}")
    return Decimal(number), commodity


def balance(postings):
    totals = {}
    for p in postings:
        totals[p.commodity] = totals.get(p.commodity, Decimal(0)) + p.amount
    return totals


def format_amount(amount, commodity):
    return f"{amount:.2f} {commodity}"


def report(postings):
    lines = []
    for commodity, total in sorted(balance(postings).items()):
        lines.append(f"TOTAL\\t{format_amount(total, commodity)}")
    return "\\n".join(lines)
'''

PROMPTS = {
    "prose": "Write a 400-word essay on why the printing press changed how knowledge spread in Europe.",
    "code": "Write a Python LRU cache class with get, put and a capacity limit, using an OrderedDict, "
            "with docstrings and a short unittest suite.",
    "edit": "Here is a Python module. Return the complete module with one change: rename `balance` to "
            "`commodity_totals` everywhere (definition and call sites). Output only the code.\n\n```python\n"
            + EDIT_SOURCE + "```",
    "reasoning": "A train leaves at 9:40 and travels 210 km at 84 km/h, then stops for 25 minutes, then "
                 "travels 96 km at 72 km/h. At what time does it arrive? Show your work briefly.",
}


CONTEXT = ""   # optional long document prepended to every prompt (--context-file)


def run(url, model, prompt, temp, max_tokens, seed=None):
    if CONTEXT:
        prompt = "Here is a reference document:\n\n" + CONTEXT + "\n\nIgnore the document for this task. " + prompt
    return stream(url, model, [{"role": "user", "content": prompt}], temp, max_tokens, seed=seed, timeout=1800)


def read_text(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser(description="Decode-speed benchmark (see the module docstring).")
    ap.add_argument("--url", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8080/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--temps", default="0,1.0")
    ap.add_argument("--runs", type=positive_int, default=2)
    ap.add_argument("--max-tokens", type=positive_int, default=2048)
    ap.add_argument("--seed", type=int, help="send seed SEED+run index with every request (default: no seed)")
    ap.add_argument("--kinds", default=",".join(PROMPTS))
    ap.add_argument("--context-file", help="prepend this text (plain or .gz) to every prompt (long-context decode test)")
    ap.add_argument("--context-chars", type=int, default=0, help="use only the first N characters of it")
    ap.add_argument("--json", help="write every request record and the summary to this file")
    a = ap.parse_args()
    global CONTEXT
    if a.context_file:
        CONTEXT = read_text(a.context_file)
        if a.context_chars:
            CONTEXT = CONTEXT[:a.context_chars]
    results, records, failures = {}, [], []
    for temp in [float(t) for t in a.temps.split(",")]:
        for kind in a.kinds.split(","):
            for i in range(a.runs):
                seed = a.seed + i if a.seed is not None else None
                r = run(a.url, a.model, PROMPTS[kind], temp, a.max_tokens, seed)
                label = f"{kind}@{temp} run {i + 1}"
                records.append(public(r, label=label, kind=kind, run=i + 1))
                if r["status"] == "ok" and r["tps"] is not None:
                    results.setdefault((kind, temp), []).append(r["tps"])
                    print(f"{kind:9s} temp {temp:3.1f} run {i + 1}: {r['completion_tokens']:5d} tokens, "
                          f"ttft {r['ttft']:5.2f} s, {r['tps']:6.1f} tok/s  [{r['finish_reason']}]", flush=True)
                else:
                    why = r["error"] or "no decode span (fewer than two streamed pieces)"
                    failures.append({"label": label, "status": r["status"], "error": why})
                    print(f"{kind:9s} temp {temp:3.1f} run {i + 1}: FAILED ({r['status']}): {why}", flush=True)
    summary = {f"{kind}@{temp}": round(statistics.median(v), 1) for (kind, temp), v in results.items()}
    print("\nsummary (median tok/s of the successful runs)")
    for (kind, temp), v in results.items():
        print(f"  {kind:9s} temp {temp:3.1f}: {statistics.median(v):6.1f}  ({len(v)} of {a.runs} runs)")
    if failures:
        print(f"{len(failures)} request(s) failed and are excluded from the summary", flush=True)
    if a.json:
        out = {"script": "decode_bench.py", "format": 2, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": a.model,
               "params": {"temps": a.temps, "runs": a.runs, "max_tokens": a.max_tokens, "seed": a.seed,
                          "kinds": a.kinds, "context_chars": len(CONTEXT)},
               "requests": records, "failures": failures, "summary": summary}
        d = os.path.dirname(os.path.abspath(a.json))
        os.makedirs(d, exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
