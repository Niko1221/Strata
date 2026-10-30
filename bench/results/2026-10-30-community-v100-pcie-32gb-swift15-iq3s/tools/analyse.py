#!/usr/bin/env python3
"""RES-20261029-01 (TASK-20261029-04 and -06): the derived tables of a decode-bench run, from raw/bench/results.jsonl.
Per (arm, ctx, task): n, median / min / max decode tok/s, prompt tok/s, VRAM share of routed entries, draft acceptance,
and whether the output hashes agree across repeats. Then per arm: how its medians compare with a reference arm's.
Standard library only; writes derived/summary.md and derived/cells.json, prints the markdown.
usage: analyse.py ARCHIVE_DIR [--ref ARM]"""
import collections, json, os, statistics, sys

a = sys.argv[1]
ref = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else None
rows = [json.loads(l) for l in open(os.path.join(a, "raw/bench/results.jsonl")) if l.strip()]
cells = collections.defaultdict(list)
for r in rows:
    cells[(r["arm"], r["ctx"], r["task"])].append(r)


def stat(xs):
    xs = [x for x in xs if x is not None]
    return (round(statistics.median(xs), 2), round(min(xs), 2), round(max(xs), 2)) if xs else (None, None, None)


out, table = [], {}
out.append("| arm | ctx | task | n | decode tok/s median (min-max) | range % | prompt tok/s median | VRAM share | drafts acc. | win ms | CPU-expert ms | hashes |")
out.append("|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|")
for (arm, ctx, task), rs in sorted(cells.items()):
    d = stat([r.get("decode_tok_s") for r in rs])
    p = stat([r.get("eng_prompt_tok_s") for r in rs])
    v = stat([r.get("vram_share_of_routed") for r in rs])
    acc = stat([r["drafts_accepted"] / r["drafts_offered"] if r.get("drafts_offered") else None for r in rs])
    w = stat([r.get("win_ms") for r in rs]); cpu = stat([r.get("win_cpu_experts_ms") for r in rs])
    hashes = {r["sha256"][:8] for r in rs}
    rng = round(100 * (d[2] - d[1]) / d[0], 1) if d[0] else None
    table[f"{arm}|{ctx}|{task}"] = {"n": len(rs), "decode": d, "range_pct": rng, "prompt": p, "vram_share": v,
                                    "draft_acc": acc, "win_ms": w, "cpu_ms": cpu, "hashes": sorted(hashes)}
    out.append(f"| {arm} | {ctx} | {task} | {len(rs)} | {d[0]} ({d[1]}-{d[2]}) | {rng} | {p[0]} | {v[0]} | {acc[0]} | "
               f"{w[0] if w[0] is not None else '-'} | {cpu[0] if cpu[0] is not None else '-'} | "
               f"{'same' if len(hashes) == 1 else 'DIFFER'} {','.join(sorted(hashes))} |")

if ref:
    out.append("")
    out.append(f"Against `{ref}` (median decode, and the larger of the two arms' ranges as the MDE):")
    out.append("")
    out.append("| arm | ctx | task | delta % | MDE % | reading | prompt delta % |")
    out.append("|---|---:|---|---:|---:|---|---:|")
    for k, c in sorted(table.items()):
        arm, ctx, task = k.split("|")
        rk = f"{ref}|{ctx}|{task}"
        if arm == ref or rk not in table or not c["decode"][0] or not table[rk]["decode"][0]:
            continue
        r = table[rk]
        delta = 100 * (c["decode"][0] / r["decode"][0] - 1)
        mde = max(c["range_pct"] or 0, r["range_pct"] or 0)
        reading = "faster" if delta > mde else "slower" if delta < -mde else "within noise"
        pd = round(100 * (c["prompt"][0] / r["prompt"][0] - 1), 1) if c["prompt"][0] and r["prompt"][0] else None
        out.append(f"| {arm} | {ctx} | {task} | {delta:+.1f} | {mde:.1f} | {reading} | {pd} |")

os.makedirs(os.path.join(a, "derived"), exist_ok=True)
json.dump(table, open(os.path.join(a, "derived/cells.json"), "w"), indent=1)
open(os.path.join(a, "derived/summary.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
