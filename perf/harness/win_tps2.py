#!/usr/bin/env python3
"""Pure decode throughput: one ~48K-token haystack, 8 different short suffixes, prefix cache warm for every sample.

For each suffix three requests are sent in a row: the first reads the suffix after the cached haystack, the next two
are warm.  `prompt_n` is printed per sample so a cache miss is visible instead of silently polluting the rate.

usage: STRATA_KEY=... python3 ab/win_tps2.py [--chars 160000] [--max-tokens 192] [--port 8089] [--label x]
"""
import argparse
import json
import os
import statistics
import time
import urllib.request

FILLER = (
    "The decode path of a mixture-of-experts transformer touches only the routed experts for each token. "
    "Effective bandwidth is therefore the product of the expert read size and the number of lookups per forward pass, "
    "divided by the wall clock of that pass. Grouping lookups by expert id turns scattered single-block reads into "
    "contiguous multi-block reads, which is what raises the achievable bandwidth on a GPU with a narrow compute budget. "
    "Speculative decoding changes the economics in the other direction: every accepted draft token is a token produced "
    "by a forward pass that was already paid for, so throughput scales with tokens per forward, not with bandwidth alone. "
    "A verification window of width w reads the union of the experts of its w candidates, so the marginal cost of an "
    "extra position is the extra experts that position introduces, not a full pass over the active set. "
    "The GDN recurrence carries a state that must be updated once per accepted token in order, while attention over the "
    "kv cache is a function of the sequence position and grows with context. "
)

SUFFIXES = [
    "\n\nSummarise section 3 of this document in one paragraph.",
    "\n\nList the three claims about bandwidth in this text.",
    "\n\nWhat does the text say about GDN recurrence?",
    "\n\nExplain the verification window in your own words.",
    "\n\nWhich sentence is the most technical? Quote it.",
    "\n\nContinue with more technical detail about expert caching.",
    "\n\nGive a short outline of the document.",
    "\n\nWhat trade-off does speculative decoding introduce here?",
]


def haystack(chars: int) -> str:
    out, i = [], 0
    while sum(len(x) for x in out) < chars:
        out.append("Section %d. %s" % (i, FILLER))
        i += 1
    return "".join(out)


def ask(port: int, key: str, prompt: str, max_tokens: int) -> dict:
    body = {"model": "strata", "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": True,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=2400) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=160000)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--label", default="")
    ap.add_argument("--per", type=int, default=3)
    a = ap.parse_args()
    key = os.environ["STRATA_KEY"]
    hay = haystack(a.chars)
    t0 = time.time()
    ask(a.port, key, hay + SUFFIXES[0], a.max_tokens)   # the prefill that fills the prefix cache
    print("prefill+first: %.1f s (%d chars)" % (time.time() - t0, len(hay)), flush=True)
    rows = []
    for s in SUFFIXES:
        for k in range(a.per):
            r = ask(a.port, key, hay + s, a.max_tokens)
            t = r.get("timings") or {}
            gen, da = t.get("predicted_n") or 0, t.get("draft_n_accepted")
            fwd = (gen - da) if da is not None and gen else None
            rows.append({"suffix": s.strip()[:28], "rep": k, "tok_s": t.get("predicted_per_second"),
                         "ms_fwd": (t["predicted_ms"] / fwd) if fwd else None,
                         "acc": (100.0 * da / t["draft_n"]) if (da is not None and t.get("draft_n")) else None,
                         "prompt_n": t.get("prompt_n")})
            print("  %-30s rep%d %6.1f tok/s  %6.2f ms/fwd  acc %5.1f%%  prompt_n %s" %
                  (rows[-1]["suffix"], k, rows[-1]["tok_s"], rows[-1]["ms_fwd"], rows[-1]["acc"] or 0,
                   rows[-1]["prompt_n"]), flush=True)
    warm = [x for x in rows if x["rep"] > 0]          # rep 0 pays the suffix's own prefill
    ts = [x["tok_s"] for x in warm]
    mf = [x["ms_fwd"] for x in warm if x["ms_fwd"]]
    ac = [x["acc"] for x in warm if x["acc"] is not None]
    miss = [x["prompt_n"] for x in warm if (x["prompt_n"] or 0) > 200]
    print(json.dumps({"label": a.label, "chars": a.chars, "n": len(warm), "tok_s_med": round(statistics.median(ts), 2),
                      "tok_s_min": min(ts), "tok_s_max": max(ts), "ms_fwd_med": round(statistics.median(mf), 2),
                      "acc_med": round(statistics.median(ac), 1), "cache_misses": len(miss)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
