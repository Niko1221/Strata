"""A/B test of two strata-metal builds (the kernels in metal/patches/, docs/MACOS.md) on the real model.

    python metal/bench/ab.py --a build-metal-a/metal/strata-metal --b build-metal-b/metal/strata-metal \
        --gguf <shard 1> --prompts prompts.json [--rounds 5] [--max-new 200] [-- extra engine args]

Each round starts a fresh process of each build, in alternating order (ABBA...), and sends the same greedy GEN
requests.  Correctness: the token ids MUST be the same for both builds (a kernel change must not change an answer).
Speed: decode tok/s of every request (DONE's generated / decode ms); medians per build and a two-sided permutation
test on the per-round medians (exact, stdlib only).  Exit status 1 when the tokens differ.
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import subprocess
import sys


def run(exe, gguf, prompts, max_new, extra, log):
    extra = list(extra)
    p = subprocess.Popen([exe, "--serve", "--gguf", gguf, "--max-context", "8192", *extra], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=open(log, "a"), text=True, bufsize=1)
    for line in p.stdout:
        if line.startswith(("READY", "ERR")):
            break
    out = []
    for i, ids in enumerate(prompts):
        if i == 0:                                     # warm the Metal pipelines (not measured)
            p.stdin.write(f"GEN 2 {ids[0]}\n"); p.stdin.flush()
            for line in p.stdout:
                if line.startswith(("DONE", "ERR")):
                    break
        p.stdin.write(f"GEN {max_new} " + ",".join(map(str, ids)) + "\n"); p.stdin.flush()
        toks = []
        for line in p.stdout:
            if line.startswith("T "):
                toks.append(int(line[2:]))
            elif line.startswith(("DONE", "ERR")):
                f = line.split()
                if f[0] == "ERR":
                    sys.exit(f"{exe}: {line.strip()}")
                out.append((toks, int(f[1]) / (float(f[4]) / 1000.0)))
                break
    p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait(60)
    return out


def perm_p(xs, ys):
    """two-sided exact permutation test on the difference of means"""
    obs = abs(statistics.mean(xs) - statistics.mean(ys))
    pool, n, hits, total = xs + ys, len(xs), 0, 0
    for idx in itertools.combinations(range(len(pool)), n):
        a = [pool[i] for i in idx]; b = [pool[i] for i in range(len(pool)) if i not in idx]
        hits += abs(statistics.mean(a) - statistics.mean(b)) >= obs - 1e-12
        total += 1
    return hits / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--gguf", required=True); ap.add_argument("--prompts", required=True)
    ap.add_argument("--rounds", type=int, default=5); ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--log", default="ab-engine.log")
    ap.add_argument("--a-args", default="", help="engine arguments for A only (one string)")
    ap.add_argument("--b-args", default="", help="engine arguments for B only (one string)")
    ap.add_argument("extra", nargs="*")
    a = ap.parse_args()
    prompts = json.load(open(a.prompts))
    res = {"A": [], "B": []}
    for r in range(a.rounds):
        for v in (("A", "B") if r % 2 == 0 else ("B", "A")):
            own = (a.a_args if v == "A" else a.b_args).split()
            got = run(a.a if v == "A" else a.b, a.gguf, prompts, a.max_new, a.extra + own, a.log)
            res[v].append(got)
            print(f"round {r} {v}: " + "  ".join(f"{t:.2f}" for _, t in got) + " tok/s", flush=True)
    same = all(ra[i][0] == rb[i][0] for ra, rb in zip(res["A"], res["B"]) for i in range(len(prompts)))
    stable = all(x[i][0] == res["A"][0][i][0] for v in res for x in res[v] for i in range(len(prompts)))
    print(f"\ntokens: A == B {'yes' if same else 'NO'}, every run the same {'yes' if stable else 'NO'}")
    for i in range(len(prompts)):
        ma = [x[i][1] for x in res["A"]]; mb = [x[i][1] for x in res["B"]]
        print(f"prompt {i}: A median {statistics.median(ma):.2f} [{min(ma):.2f}-{max(ma):.2f}]  "
              f"B median {statistics.median(mb):.2f} [{min(mb):.2f}-{max(mb):.2f}]  "
              f"B/A {statistics.median(mb) / statistics.median(ma):.3f}  p={perm_p(ma, mb):.3f}")
    ma = [statistics.median(t for _, t in x) for x in res["A"]]; mb = [statistics.median(t for _, t in x) for x in res["B"]]
    print(f"all prompts (per-run medians): A {statistics.median(ma):.2f}  B {statistics.median(mb):.2f}  "
          f"B/A {statistics.median(mb) / statistics.median(ma):.3f}  p={perm_p(ma, mb):.3f}")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
