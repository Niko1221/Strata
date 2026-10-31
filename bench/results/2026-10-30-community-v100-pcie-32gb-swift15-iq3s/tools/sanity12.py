#!/usr/bin/env python3
"""RES-20261029-01 (TASK-20261029-07): Strata's twelve sanity checks (docs/BENCHMARK_PROTOCOL.md, standard-v1), sent
from the frozen fixtures in bench/prompts/standard-v1/requests to a running server and scored as the protocol says:
exact text after stripping whitespace, strict JSON (one value, no prose or fences, duplicate keys rejected), one
structured tool call with exactly the expected name and arguments. Thinking on (the bundle's template kwargs), cap 2048
(reasoning included), seed 1. Deviation from the protocol: temperature 0 instead of the model card's sampling, so that
arms compare as a guard. Standard library only.
usage: sanity12.py --base URL --model ID --out DIR --arm NAME [--only I1,T3]"""
import argparse, glob, json, os, time, urllib.request

FIX = "<strata-dev>/bench/prompts/standard-v1"


def no_dupes(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate key")
    return dict(pairs)


def strict_json(s):
    s = s.strip()
    v = json.loads(s, object_pairs_hook=no_dupes)       # raises on prose, fences or trailing text
    return v


def same(a, b):                                          # values and types (bool is not int)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def score(exp, msg, finish):
    if finish == "length":
        return False, "truncated"
    content = (msg.get("content") or "")
    if exp["kind"] == "text":
        return content.strip() == exp["value"], content.strip()[:80]
    if exp["kind"] == "json":
        try:
            return same(strict_json(content), exp["value"]), content.strip()[:80]
        except ValueError as e:
            return False, f"not strict JSON ({e}): {content.strip()[:60]}"
    calls = msg.get("tool_calls") or []
    if len(calls) != 1:
        return False, f"{len(calls)} tool calls; content {content.strip()[:60]!r}"
    fn = calls[0].get("function") or {}
    try:
        args = strict_json(fn.get("arguments") or "")
    except ValueError as e:
        return False, f"arguments not strict JSON ({e})"
    return fn.get("name") == exp["name"] and same(args, exp["arguments"]), f"{fn.get('name')}({json.dumps(args)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--only", default="", help="comma-separated case ids (a script check)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    kw = json.load(open(f"{FIX}/build.json"))["template_kwargs"]
    rows, passed = [], 0
    for f in sorted(glob.glob(f"{FIX}/requests/sanity-*.json")):
        r = json.load(open(f))
        if a.only and r["case_id"] not in a.only.split(","):
            continue
        body = {"model": a.model, "messages": r["messages"], "max_tokens": r["max_output_tokens"], "temperature": 0,
                "seed": r["seed"], "chat_template_kwargs": kw, "reasoning_effort": kw.get("reasoning_effort")}
        if r.get("tools"):
            body["tools"] = r["tools"]; body["tool_choice"] = r["tool_choice"]
        req = urllib.request.Request(a.base + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t = time.time()
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=900))
            ch = resp["choices"][0]; ok, what = score(r["expected"], ch["message"], ch.get("finish_reason"))
            usage = resp.get("usage") or {}
        except Exception as e:                            # a failed request fails its case
            ch, ok, what, usage = {}, False, f"request failed: {e}", {}
        passed += ok
        row = {"arm": a.arm, "case": r["case_id"], "passed": ok, "seen": what, "finish": ch.get("finish_reason"),
               "completion_tokens": usage.get("completion_tokens"), "s": round(time.time() - t, 1)}
        rows.append(row); print(json.dumps(row), flush=True)
        open(os.path.join(a.out, f"sanity-{a.arm}-{r['case_id']}.json"), "w").write(json.dumps(resp if ch else {}, indent=1))
    with open(os.path.join(a.out, "sanity.jsonl"), "a") as fo:
        for row in rows:
            fo.write(json.dumps(row) + "\n")
    print(json.dumps({"arm": a.arm, "sanity_passed": passed, "of": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
