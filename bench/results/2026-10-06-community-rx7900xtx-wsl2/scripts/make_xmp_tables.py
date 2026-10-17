#!/usr/bin/env python3
"""make_xmp_tables.py: markdown tables for the XMP (DDR4-3200) re-run and the prompt-length sweep, from runs-xmp/ and runs-sweep/ (run from this folder)."""
import json, statistics as st

CONFIGS = [("0.1.33 (aeb35be), defaults", "strata-0.1.33"), ("0.1.40 (1735d64), defaults", "strata-0.1.40"),
           ("0.1.40, STRATA_PREFILL_STREAM_MIN=65536", "strata-0.1.40-streammin65536")]


def rng(v, nd=1):
    v = [x for x in v if x is not None]
    return "not measured" if not v else f"{st.median(v):.{nd}f} ({min(v):.{nd}f}-{max(v):.{nd}f})"


print("### XMP community table\n")
print("| Configuration | Request | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |")
print("| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |")
for name, d in CONFIGS:
    rows = [json.loads(l) for l in open(f"runs-xmp/{d}/runs.jsonl")]
    for kind, label in (("short", "short, cold"), ("long", "long, cold"), ("followup", "follow-up")):
        r = [x for x in rows if x["kind"] == kind]
        pt = int(st.median([x["prompt_tokens"] for x in r])); ct = int(st.median([x["cached_tokens"] for x in r])); co = int(st.median([x["completion_tokens"] for x in r]))
        pf = "not measured" if kind == "short" else rng([x["prefill_tok_s"] for x in r])
        print(f"| {name} | {label} | {pt} | {ct} | {co} | {len(r)} | {pf} | {rng([x['decode_tok_s'] for x in r])} | {rng([x['ttft_s'] for x in r])} |")

print("\n### XMP needle table\n")
print("| Configuration | 32k depth 50% | 128k depth 50% |\n| --- | --- | --- |")
for name, d in CONFIGS:
    n = {x["length"]: x for x in json.load(open(f"runs-xmp/{d}/needles.json"))}
    cell = lambda x: f"{'found' if x['found'] else 'MISSED'}, {x['prompt_tokens']:,} tokens in {x['seconds']:.0f} s ({x['prompt_tokens']/x['seconds']:.0f} tok/s)"
    print(f"| {name} | {cell(n['32k'])} | {cell(n['128k'])} |")

sw = {d: [json.loads(l) for l in open(f"runs-sweep/{d}/sweep.jsonl")] for _, d in CONFIGS}
print("\n### Sweep tables (server -c 262144; every request cold, nonce prefix; values of the passes separated by ' / ')\n")
for name, d in CONFIGS:
    print(f"**{name}**, {max(x['iteration'] for x in sw[d])} pass(es)\n")
    print("| Length label | Actual prompt tokens | Prefill tok/s | Decode tok/s | TTFT seconds |\n| --- | ---: | ---: | ---: | ---: |")
    for k in (16, 32, 64, 96, 128, 192, 228, 256):
        r = sorted([x for x in sw[d] if x["length_k"] == k and not x["error"]], key=lambda x: x["iteration"])
        if not r:
            print(f"| {k}k | no result | | | |"); continue
        j = lambda f, nd: " / ".join(f"{x[f]:.{nd}f}" for x in r)
        print(f"| {k}k | {r[0]['prompt_tokens']:,} | {j('prefill_tok_s', 0)} | {j('decode_tok_s', 1)} | {j('ttft_s', 0)} |")
    print()
print("Errors/retries: " + str({d: sum(1 for x in sw[d] if x['error'] or x['attempt'] > 1) for _, d in CONFIGS}))
