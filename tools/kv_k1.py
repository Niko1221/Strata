"""K1 of the ladder for the S3 arms: the per-position paired test, not the 8-segment one.

KL(arm) per position is KL(ref || arm) over ref's top-k tokens. The ladder's K1 is defined against
`int8`: the mean of KL(int8 || arm) over cwe-4k, its one-sided 95 % upper bound (+0.001 nats) and how
many of the 8 segments int8 leads. Another --ref is useful for the other columns - a `fp16` reference
is the honest unit for absolute quality, because KL(int8 || arm) also charges the arm for the errors
int8 makes itself - but `k1` is only evaluated when the reference is int8. With any other reference
the "segments the reference leads" column is not K1 and is reported as what it is: the segments in
which the arm is worse than the reference.

Usage: python tools/kv_k1.py --ref int8.tsv --run a.tsv [--run b.tsv ...] [--kl-topk 256]

`--block N` replaces the position-level normal approximation with a moving-block bootstrap:
consecutive positions of a segment are grouped N at a time (the last block of a segment is
shorter), and the blocks are resampled with replacement (2,000 resamples, seed 20261103). It is
what a single-segment run needs - positions of one document are not independent, so a
position-level SEM is too narrow. Off by default: without `--block` the numbers are the ones
every earlier K1 table printed.
"""

import argparse
import json
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quant_quality as qq  # noqa: E402

BOOT_RESAMPLES = 2000
BOOT_SEED = 20261103


def blocks_of(keyed: dict, width: int) -> list[list[float]]:
    """Consecutive positions within a segment, `width` at a time, in position order."""
    out = []
    for seg in sorted({s for s, _ in keyed}):
        vals = [keyed[(seg, p)] for p in sorted(p for s, p in keyed if s == seg)]
        out.extend(vals[i:i + width] for i in range(0, len(vals), width))
    return out


def block_bootstrap(blocks: list[list[float]]) -> dict:
    """Mean over resampled blocks: percentile 95 % CI and the one-sided 95 % upper bound."""
    rng = random.Random(BOOT_SEED)
    n = len(blocks)
    means = []
    for _ in range(BOOT_RESAMPLES):
        picked = [blocks[rng.randrange(n)] for _ in range(n)]
        flat = [x for b in picked for x in b]
        means.append(sum(flat) / len(flat))
    means.sort()
    return {"blocks": n, "resamples": BOOT_RESAMPLES, "seed": BOOT_SEED,
            "ci95_low": means[int(0.025 * BOOT_RESAMPLES)],
            "ci95_high": means[int(0.975 * BOOT_RESAMPLES) - 1],
            "upper95": means[int(0.95 * BOOT_RESAMPLES) - 1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--kl-topk", type=int, default=256)
    ap.add_argument("--block", type=int, default=0, metavar="N",
                    help=("moving-block bootstrap over blocks of N consecutive positions within a segment "
                          f"({BOOT_RESAMPLES} resamples, seed {BOOT_SEED}); 0 = off (the position-level normal "
                          "approximation, as in every earlier K1 table)"))
    ap.add_argument("--json")
    a = ap.parse_args()
    ref = qq.read_run(a.ref)
    is_int8_ref = "int8" in pathlib.Path(a.ref).stem
    out = {"ref": a.ref, "kl_topk": a.kl_topk, "runs": {}, "pairs": {}}
    print(f"reference {a.ref}, KL support: REF's {a.kl_topk} most likely tokens")
    if not is_int8_ref:
        print("K1 is not evaluated: its anchor is int8. The last column counts the segments in which "
              "the arm is worse than this reference.")
    print(f"{'run':34s} {'pos':>6s} {'mean KL':>9s} {'sd':>8s} {'t':>7s} {'95% up':>9s} {'seg arm worse':>14s}")
    per_pos = {}
    for run in a.run:
        r = qq.read_run(run)
        vals, seg_mean, keyed = [], {}, {}
        for seg in ref:
            if seg not in r:
                continue
            v = []
            for pos, kr in ref[seg].items():
                if pos not in r[seg]:
                    continue
                ka = r[seg][pos][4]
                k = qq.kl_between(kr[4], ka, a.kl_topk)
                keyed[(seg, pos)] = k
                v.append(k)
            if v:
                seg_mean[seg] = sum(v) / len(v)
                vals.extend(v)
        per_pos[pathlib.Path(run).name] = keyed
        n = len(vals)
        mean = sum(vals) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in vals) / (n - 1))
        sem = sd / math.sqrt(n)
        t = mean / sem
        # one-sided 95 % upper bound (normal approximation; n is 32,746, so t and z agree to 4 decimals)
        up = mean + 1.645 * sem
        worse = sum(1 for s in seg_mean if seg_mean[s] > 0)
        print(f"{pathlib.Path(run).name:34s} {n:6d} {mean:9.5f} {sd:8.4f} {t:7.2f} {up:9.5f} "
              f"{worse:>10d}/{len(seg_mean)}")
        rec = {"n": n, "mean_kl": mean, "sd": sd, "sem": sem, "t": t,
               "upper95": up, "segments_arm_worse": worse,
               "segment_mean_kl": seg_mean}
        if a.block:
            bs = block_bootstrap(blocks_of(keyed, a.block))
            bs["mean"] = mean
            rec["block_bootstrap"] = bs
            print(f"{'':34s} block bootstrap ({a.block}/block, {bs['blocks']} blocks): mean {mean:+.5f}  "
                  f"95% CI [{bs['ci95_low']:+.5f}, {bs['ci95_high']:+.5f}]  one-sided up {bs['upper95']:+.5f}")
        if is_int8_ref:
            # K1: mean <= 0, one-sided 95 % upper bound < +0.001 nats, int8 leads in at most 4 of 8
            rec["k1"] = bool(mean <= 0.0 and up < 0.001 and worse <= 4)
        out["runs"][pathlib.Path(run).name] = rec
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.json}")
    if len(a.run) > 1:
        print("\npaired per-position difference (same positions, so the text cancels):")
        names = [pathlib.Path(x).name for x in a.run]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ka, kb = per_pos[names[i]], per_pos[names[j]]
                keys = [k for k in ka if k in kb]
                d = [ka[k] - kb[k] for k in keys]
                n = len(d)
                mean = sum(d) / n
                sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
                sem = sd / math.sqrt(n)
                t = mean / sem
                up = mean + 1.645 * sem
                seg = {}
                for (s, _), x in ((k, ka[k] - kb[k]) for k in keys):
                    seg.setdefault(s, []).append(x)
                leads = sum(1 for s in seg if sum(seg[s]) / len(seg[s]) > 0)
                print(f"  {names[i]:30s} - {names[j]:30s} {mean:+.5f}  sd {sd:.4f}  t {t:+.2f}  "
                      f"95% CI [{mean - 1.96 * sem:+.5f}, {mean + 1.96 * sem:+.5f}]  "
                      f"{names[i]} worse in {leads}/{len(seg)} segments")
                out["pairs"][f"{names[i]} - {names[j]}"] = {
                    "n": n, "mean": mean, "sd": sd, "sem": sem, "t": t,
                    "ci95_low": mean - 1.96 * sem, "ci95_high": mean + 1.96 * sem,
                    "segments_a_worse": leads}
                if a.block:
                    diff = {k: ka[k] - kb[k] for k in keys}
                    bs = block_bootstrap(blocks_of(diff, a.block))
                    bs["mean"] = mean
                    out["pairs"][f"{names[i]} - {names[j]}"]["block_bootstrap"] = bs
                    print(f"    block bootstrap ({a.block}/block, {bs['blocks']} blocks): mean {mean:+.5f}  "
                          f"95% CI [{bs['ci95_low']:+.5f}, {bs['ci95_high']:+.5f}]  "
                          f"one-sided up {bs['upper95']:+.5f}")
        if a.json:
            pathlib.Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
