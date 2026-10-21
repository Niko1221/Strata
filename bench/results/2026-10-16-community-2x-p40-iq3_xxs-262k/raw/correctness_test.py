#!/usr/bin/env python3
"""Correctness set for the §5.4 gate (STRATA_FLASH_NEXT_PLAN.md): a FIXED set of 20 prompts at temperature 0, sent through the
smart proxy to one engine at a time, saved raw; then `--compare A B` lines the two engines' outputs up token for token (well,
character for character, plus a word count) and reports where they diverge.

Run per engine (the proxy must already have that stack active — this script never triggers a swap):
  python3 correctness_test.py --model flash-next-strata   --label strata
  python3 correctness_test.py --model flash-next-llamacpp --label llamacpp
Compare:
  python3 correctness_test.py --compare flash-next-correctness/<UTC>-strata flash-next-correctness/<UTC>-llamacpp

Both engines run the model with thinking on; the text compared is reasoning_content + a separator + content, so a divergence
inside the think block counts. max_tokens 400 keeps every answer under the llama.cpp stack's --reasoning-budget 1024, so the
budget never fires on that side. Stdlib only.
"""
import argparse, json, os, sys, time, urllib.request
from datetime import datetime, timezone

PROMPTS = [
    "What is 17 * 23? Answer with the number only.",
    "List the first ten prime numbers, comma-separated.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Explain the difference between TCP and UDP in three sentences.",
    "Translate to French: 'The measurement must be dated.'",
    "Sort these words alphabetically: pear, apple, fig, banana, cherry.",
    "A train leaves at 09:40 and arrives at 13:05. How long is the journey?",
    "Return a JSON object with keys name, port, healthy for a service called proxy on port 4000 that is healthy. JSON only.",
    "Write a bash one-liner that counts lines in all .log files in the current directory.",
    "What is the capital of Australia? One word.",
    "Summarize the plot of Hamlet in two sentences.",
    "Given the list [3, 1, 4, 1, 5, 9, 2, 6], what is the median?",
    "Write a SQL query that returns the five most recent orders from a table orders(id, created_at, total).",
    "Convert 72 degrees Fahrenheit to Celsius. Show the formula and the result.",
    "Name three differences between a process and a thread.",
    "Write a regular expression that matches an IPv4 address.",
    "Is 221 a prime number? Explain briefly.",
    "Rewrite in passive voice: 'The proxy routes every request to the active stack.'",
    "Return a JSON array of the days of the week starting with Monday. JSON only.",
    "Explain what a KV cache is in a transformer, in one paragraph.",
]

SEP = "\n<<<answer>>>\n"


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ask(base_url, model, prompt, max_tokens, timeout):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": False, "cache_prompt": False,
    }
    req = urllib.request.Request(base_url + "/v1/chat/completions", data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode())
    wall = time.monotonic() - t0
    msg = (resp.get("choices") or [{}])[0].get("message", {})
    return {
        "prompt": prompt,
        "reasoning_content": msg.get("reasoning_content") or "",
        "content": msg.get("content") or "",
        "finish_reason": (resp.get("choices") or [{}])[0].get("finish_reason"),
        "usage": resp.get("usage"), "timings": resp.get("timings"),
        "wall_s": round(wall, 3), "model_reported": resp.get("model"),
    }


def run(args):
    out = os.path.join(args.out_dir, "flash-next-correctness", f"{utc_stamp()}-{args.label}")
    os.makedirs(out, exist_ok=True)
    cfg = {"base_url": args.base_url, "model": args.model, "label": args.label, "max_tokens": args.max_tokens,
           "temperature": 0.0, "prompts": len(PROMPTS), "note": args.note, "argv": sys.argv}
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    with open(os.path.join(out, "outputs.jsonl"), "w") as f:
        for i, p in enumerate(PROMPTS, 1):
            try:
                rec = ask(args.base_url, args.model, p, args.max_tokens, args.timeout)
            except Exception as e:
                rec = {"prompt": p, "error": repr(e)}
            rec["idx"] = i
            f.write(json.dumps(rec) + "\n"); f.flush()
            ct = rec.get("usage", {}).get("completion_tokens") if rec.get("usage") else None
            print(f"[{i:2d}/{len(PROMPTS)}] {rec.get('finish_reason') or rec.get('error')} tokens={ct} wall={rec.get('wall_s')}s", flush=True)
    print("wrote", out)
    return out


def load(d):
    recs = {}
    for line in open(os.path.join(d, "outputs.jsonl")):
        r = json.loads(line)
        recs[r["idx"]] = r
    return recs


def text_of(r):
    return (r.get("reasoning_content") or "") + SEP + (r.get("content") or "")


def compare(a_dir, b_dir, out_dir):
    A, B = load(a_dir), load(b_dir)
    rows, identical = [], 0
    for i in sorted(A):
        ra, rb = A[i], B.get(i, {})
        ta, tb = text_of(ra), text_of(rb)
        n = 0
        while n < min(len(ta), len(tb)) and ta[n] == tb[n]:
            n += 1
        same = ta == tb
        identical += same
        words_common = len(ta[:n].split())
        rows.append({
            "idx": i, "identical": same, "len_a": len(ta), "len_b": len(tb), "common_prefix_chars": n,
            "common_prefix_words": words_common,
            "diverged_in": None if same else ("reasoning" if n < len(ra.get("reasoning_content") or "") else "answer"),
            "a_at_divergence": None if same else ta[max(0, n - 40):n + 60],
            "b_at_divergence": None if same else tb[max(0, n - 40):n + 60],
            "finish_a": ra.get("finish_reason"), "finish_b": rb.get("finish_reason"),
            "tokens_a": (ra.get("usage") or {}).get("completion_tokens"), "tokens_b": (rb.get("usage") or {}).get("completion_tokens"),
            "answer_identical": (ra.get("content") or "") == (rb.get("content") or ""),
        })
    rep = {"a": a_dir, "b": b_dir, "prompts": len(rows), "identical_full": identical,
           "identical_answer": sum(1 for r in rows if r["answer_identical"]), "rows": rows}
    stamp = utc_stamp()
    os.makedirs(os.path.join(out_dir, "flash-next-correctness"), exist_ok=True)
    path = os.path.join(out_dir, "flash-next-correctness", f"compare-{stamp}.json")
    json.dump(rep, open(path, "w"), indent=2)
    md = [f"# Correctness compare {stamp}", "", f"A = `{a_dir}`  ", f"B = `{b_dir}`", "",
          f"Full text (think + answer) identical: **{identical}/{len(rows)}**; final answer identical: "
          f"**{rep['identical_answer']}/{len(rows)}**", "",
          "| # | identical | answer same | diverged in | common prefix (chars / words) | len A / B | tokens A / B |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['idx']} | {r['identical']} | {r['answer_identical']} | {r['diverged_in'] or '-'} | "
                  f"{r['common_prefix_chars']} / {r['common_prefix_words']} | {r['len_a']} / {r['len_b']} | "
                  f"{r['tokens_a']} / {r['tokens_b']} |")
    md.append("")
    for r in rows:
        if not r["identical"]:
            md += [f"### Prompt {r['idx']} first divergence", "", "A: `" + (r["a_at_divergence"] or "").replace("\n", "\\n") + "`",
                   "", "B: `" + (r["b_at_divergence"] or "").replace("\n", "\\n") + "`", ""]
    open(path.replace(".json", ".md"), "w").write("\n".join(md))
    print("\n".join(md[:8]))
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:4000")
    ap.add_argument("--model")
    ap.add_argument("--label")
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--note", default="")
    ap.add_argument("--compare", nargs=2, metavar=("A_DIR", "B_DIR"))
    args = ap.parse_args()
    if args.compare:
        compare(args.compare[0], args.compare[1], args.out_dir)
    else:
        if not (args.model and args.label):
            ap.error("--model and --label are required to run the set")
        run(args)


if __name__ == "__main__":
    main()
