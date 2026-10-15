#!/usr/bin/env python3
"""--prompt-cache-policy: what a request that branches into the middle of a long conversation gets back from the
checkpoint chain, against a running Strata server (docs/DETAILS.md, "Which checkpoints stay").

Four greedy requests of 4 output tokens each, thinking off, the prompt text deterministic (seeded):

  r0_cold    system (~30K tokens) + a numbered document (~150K) + question A   read from token 0
  r1_retry   the same prompt with question B                                  the checkpoint before the question
  r2_branch  system + the document cut at --branch-frac + question A          whichever checkpoint is nearest
  r3_return  the full prompt with question C                                  what is left after the branch

Prints one JSON line per request (prompt tokens, tokens the engine reused, its prompt time) and a table.  Run it once
against a server started with the default retention and once with `--prompt-cache-policy log` in the engine's
arguments; the branch request is where the two differ.  No API key: start the server for this test without one, or
pass --key.

    python3 tools/checkpoint_branch_test.py --url http://127.0.0.1:8080 --label lru
"""
import argparse
import json
import random
import time
import urllib.request

WORDS = ("river stone lamp window garden paper cloud market bridge engine silver forest harbour letter signal "
         "meadow candle orbit valley copper ladder mirror tunnel saddle anchor beacon canvas ember fabric glacier "
         "hammer island jacket kernel lantern marble needle oyster pepper quartz ribbon timber velvet walnut").split()


def paragraph(i: int, rng: random.Random, n: int) -> str:
    return f"Paragraph {i}: " + " ".join(rng.choice(WORDS) for _ in range(n)) + "."


def build(seed: int, sys_paras: int, doc_paras: int, words: int):
    rng = random.Random(seed)
    system = ("You are a careful assistant. Reference material follows.\n\n"
              + "\n".join(paragraph(i, rng, words) for i in range(sys_paras)))
    doc = [paragraph(100000 + i, rng, words) for i in range(doc_paras)]
    return system, doc


def ask(url: str, key: str | None, system: str, doc: list, question: str, max_tokens: int = 4) -> dict:
    body = {"model": "strata", "messages": [{"role": "system", "content": system},
                                            {"role": "user", "content": "\n".join(doc) + "\n\n" + question}],
            "max_tokens": max_tokens, "temperature": 0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    wall = time.monotonic() - t0
    u = out.get("usage") or {}
    t = out.get("timings") or {}
    return {"prompt_tokens": u.get("prompt_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "prompt_ms": t.get("prompt_ms"), "wall_s": round(wall, 2),
            "text": (out.get("choices") or [{}])[0].get("message", {}).get("content")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--key", default=None, help="the server's API key, when it has one")
    ap.add_argument("--label", default="run", help="a name for this run in the output (lru, log, ...)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sys-paras", type=int, default=520, help="system prompt paragraphs (~58 tokens each)")
    ap.add_argument("--doc-paras", type=int, default=2900, help="document paragraphs (~51 tokens each)")
    ap.add_argument("--words", type=int, default=40)
    ap.add_argument("--branch-frac", type=float, default=0.42, help="where in the document the branch request cuts")
    ap.add_argument("--out", default=None, help="append each request's JSON line to this file")
    a = ap.parse_args()
    system, doc = build(a.seed, a.sys_paras, a.doc_paras, a.words)
    rows = []

    def run(name: str, d: list, q: str) -> None:
        r = ask(a.url, a.key, system, d, q)
        r.update(step=name, label=a.label)
        rows.append(r)
        print(json.dumps(r), flush=True)

    run("r0_cold", doc, "Reply with the single word OK.")
    run("r1_retry", doc, "Reply with the single word YES.")
    run("r2_branch", doc[:int(len(doc) * a.branch_frac)], "Reply with the single word OK.")
    run("r3_return", doc, "Reply with the single word NO.")
    print()
    print(f"{a.label}: step            prompt   reused    read   prompt_ms   wall_s")
    for r in rows:
        read = (r["prompt_tokens"] or 0) - (r["cached"] or 0)
        print(f"{a.label}: {r['step']:<14} {r['prompt_tokens']:>7} {r['cached']:>8} {read:>7}   {r['prompt_ms']!s:>9}   {r['wall_s']:>6}")
    if a.out:
        with open(a.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
