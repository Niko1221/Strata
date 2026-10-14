#!/usr/bin/env python3
"""Summarize the engine's own timing lines in runs/engine-<arm>.txt, per phase.

  python3 scripts/engine_log_stats.py runs/engine-ours.txt runs/engine-stock.txt

Per phase (start-up, short, long, multi-turn; the '# ==== phase' markers in the excerpt):
- solo requests: 'strata serve: prompt' lines with more than 1 generated token (the solo path: the request ran alone,
  or was handed back from a batch slot). Their decode tok/s and MTP drafts accepted/proposed.
- batch admissions: lines with exactly 1 generated token (the request's first token, read on the solo path before
  the request moves to a batch slot; its later tokens are in the 'strata batch (pipelined ...)' summary line).
- fresh prompt reads: 0 reused tokens and more than 100 read; read ms and tok/s (median, min, max), and the first one
  of the phase (taken while no other request was decoding).
- follow-up reads: lines with reused tokens and more than 100 read tokens (conversation continued).
- 'strata batch: slot N takes M tokens (copied in X ms)': the admission copy into a batch slot (median, min, max).
- 'strata batch (pipelined, G groups of S)': rows (tokens) decoded in batch slots, ms, rows/s.
"""
import re
import statistics as st
import sys

PROMPT = re.compile(r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
                    r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")
COPY = re.compile(r"strata batch: slot \d+ takes \d+ tokens \(copied in ([\d.]+) ms\)")
BATCH = re.compile(r"strata batch \(pipelined, (\d+) groups of (\d+)\): (\d+) group-steps, (\d+) rows in (\d+) ms")


def med(xs):
    return st.median(xs) if xs else float("nan")


def summarize(path):
    phases, cur = {}, "start-up"
    for line in open(path):
        if line.startswith("# ==== phase: "):
            cur = line[len("# ==== phase: "):].split(" ")[0]
            continue
        phases.setdefault(cur, []).append(line.rstrip("\n"))
    print(f"== {path}")
    for name, lines in phases.items():
        reqs = [tuple(float(x) for x in m.groups()) for m in map(PROMPT.search, lines) if m]
        if not reqs:
            continue
        copies = [float(m.group(1)) for m in map(COPY.search, lines) if m]
        batches = [tuple(int(x) for x in m.groups()) for m in map(BATCH.search, lines) if m]
        solo = [r for r in reqs if r[5] > 1]
        adm = [r for r in reqs if r[5] == 1]
        fresh = [r for r in reqs if r[1] == 0 and r[2] > 100]
        follow = [r for r in reqs if r[1] > 0 and r[2] > 100]
        print(f"-- phase {name}: {len(reqs)} request lines: {len(solo)} solo (>1 token generated), "
              f"{len(adm)} batch admissions (1 token generated)")
        for r in solo:
            acc = f"{r[8]:.0f} of {r[9]:.0f} drafts accepted ({100 * r[8] / r[9]:.1f} %)" if r[9] else "no drafts"
            print(f"   solo: prompt {r[0]:.0f} ({r[1]:.0f} reused + {r[2]:.0f} read in {r[3]:.0f} ms), "
                  f"{r[5]:.0f} generated at {r[7]:.1f} tok/s, {acc}")
        print(f"   batch admissions with draft proposals: {sum(1 for r in adm if r[9] > 0)} of {len(adm)}")
        if fresh:
            print(f"   fresh reads: n={len(fresh)}, tokens median {med([r[2] for r in fresh]):.0f}, "
                  f"ms median {med([r[3] for r in fresh]):.0f} (min {min(r[3] for r in fresh):.0f}, "
                  f"max {max(r[3] for r in fresh):.0f}), tok/s median {med([r[4] for r in fresh]):.0f}; "
                  f"first: {fresh[0][2]:.0f} tokens in {fresh[0][3]:.0f} ms ({fresh[0][4]:.0f} tok/s)")
        if follow:
            print(f"   follow-up reads: n={len(follow)}, read tokens median {med([r[2] for r in follow]):.0f}, "
                  f"reused median {med([r[1] for r in follow]):.0f}, ms median {med([r[3] for r in follow]):.0f}, "
                  f"tok/s median {med([r[4] for r in follow]):.0f}")
        if copies:
            print(f"   slot copies: n={len(copies)}, median {med(copies):.1f} ms (min {min(copies):.1f}, "
                  f"max {max(copies):.1f})")
        for g, s, steps, rows, ms in batches:
            print(f"   batch: {g} groups of {s}, {steps} group-steps, {rows} rows in {ms} ms = "
                  f"{1000 * rows / ms:.1f} rows/s ({rows / steps:.2f} rows per group-step)")


for p in sys.argv[1:]:
    summarize(p)
