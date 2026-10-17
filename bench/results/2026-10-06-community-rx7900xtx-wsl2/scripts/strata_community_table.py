#!/usr/bin/env python3
"""strata_community_table.py ROOT: the results table of the community report from the runs.jsonl files of ROOT's configuration folders.
Rows: configuration x request kind; median and range over all iterations of the runs listed in CONFIGS. Markdown on stdout."""
import json, os, statistics as st, sys

root = sys.argv[1]
CONFIGS = [
    ("0.1.33 (aeb35be), defaults", ["strata-0.1.33", "strata-0.1.33-r2"]),
    ("0.1.40 (1735d64), defaults", ["strata-0.1.40", "strata-0.1.40-r2", "strata-0.1.40-r3"]),
    ("0.1.40, STRATA_PREFILL_STREAM_MIN=65536", ["strata-0.1.40-streammin65536"]),
]


def mr(vals, nd=1):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "not measured"
    return "%s (%s-%s)" % (round(st.median(vals), nd), round(min(vals), nd), round(max(vals), nd))


print("| Configuration | Request | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |")
print("| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |")
for name, dirs in CONFIGS:
    rows = []
    for d in dirs:
        p = os.path.join(root, d, "runs.jsonl")
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p)]
    for kind in ("short", "long", "followup"):
        r = [x for x in rows if x["kind"] == kind]
        if not r:
            continue
        pr = [x["prefill_tok_s"] for x in r] if kind != "short" else [None]
        print("| %s | %s | %s | %s | %s | %d | %s | %s | %s |" % (
            name, {"short": "short, cold", "long": "long, cold", "followup": "follow-up"}[kind],
            round(st.median([x["prompt_tokens"] for x in r])), round(st.median([x["cached_tokens"] for x in r])),
            round(st.median([x["completion_tokens"] for x in r])), len(r), mr(pr), mr([x["decode_tok_s"] for x in r]), mr([x["ttft_s"] for x in r])))
