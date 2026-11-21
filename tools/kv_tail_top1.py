#!/usr/bin/env python
"""The tail question: does the calibration help where it matters?

The 4K mean KL prefers q4lm32w without the calibration (0.01689 against 0.01768); the p99 and the
top-1 agreement prefer it with the calibration (0.2263 against 0.2410; 98.97 % against 98.89 %).
The ladder's own criteria are not only the mean KL (K1), so the tail deserves a paired test
instead of two numbers side by side.

Two tests, both paired over the same 32,746 positions of cwe-4k, against the fp16 reference:
  * the tail: the positions where the *int8* arm's KL is largest (the 1 % worst positions).  The
    tail set is defined by the reference arm, so it is the same for every arm and no selection
    bias can enter; the difference on that set is then paired and tested.
  * the top-1: the disagreements between the two 16/16 arms on the argmax, with an exact McNemar
    test (the tool's areport uses the same idea for the answer region).

Usage:
  python tools/kv_tail_top1.py --ref <fp16.tsv> --int8 <int8.tsv> --a <q4lm32w.tsv> --b <q4lm32w+center.tsv> \
      [--ids <set.ids.json>] [--tail 0.01]
"""
import argparse
import importlib.util
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("quant_quality", HERE / "quant_quality.py")
qq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--ids")
    ap.add_argument("--tail", type=float, default=0.01)
    ap.add_argument("--kl-topk", type=int, default=qq.TOPK)
    args = ap.parse_args()

    ref = qq.read_run(args.ref)
    runs = {"int8": qq.read_run(args.int8), "A": qq.read_run(args.a), "B": qq.read_run(args.b)}
    series = {name: qq.kl_series(ref, r, args.kl_topk) for name, r in runs.items()}

    keys = sorted(set.intersection(*[set(s) for s in series.values()]))
    n = len(keys)
    print(f"positions: {n}")

    # the tail set: the worst 1 % by the int8 arm's own KL (arm-independent by construction)
    tail_n = max(1, int(round(n * args.tail)))
    order = sorted(keys, key=lambda k: series["int8"][k], reverse=True)
    tail = set(order[:tail_n])
    print(f"tail: {tail_n} positions ({args.tail:.1%}), the ones where int8's KL is largest\n")

    print(f"{'arm':>6} {'mean KL':>10} {'p99':>9} {'top1 vs fp16':>13} | "
          f"{'tail mean KL':>13} {'d vs int8 (tail)':>18} {'t':>7} {'95 % CI':>22}")
    print("-" * 112)
    for name in ("int8", "A", "B"):
        s = series[name]
        allv = sorted(s[k] for k in keys)
        p99 = allv[int(0.99 * (len(allv) - 1))]
        # the argmax: record[3] is the sampled token, record[4][0] the argmax of the top-k list
        top1 = 0
        for k in keys:
            seg, pos = k
            rr, rn = ref.get(seg, {}).get(pos), runs[name].get(seg, {}).get(pos)
            if rr and rn and rr[4][0][0] == rn[4][0][0]:
                top1 += 1
        tail_mean = sum(s[k] for k in tail) / len(tail)
        d = [s[k] - series["int8"][k] for k in sorted(tail)]
        m = sum(d) / len(d)
        sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
        sem = sd / math.sqrt(len(d))
        t = m / sem if sem else float("nan")
        lo, hi = m - 1.96 * sem, m + 1.96 * sem
        print(f"{name:>6} {sum(allv)/len(allv):10.5f} {p99:9.4f} {100.0*top1/n:12.2f} % | "
              f"{tail_mean:13.5f} {m:+18.5f} {t:+7.2f}   [{lo:+.5f}, {hi:+.5f}]")

    # McNemar on the top-1 disagreements between A and B
    both = [k for k in keys if ref.get(k[0], {}).get(k[1]) and runs["A"].get(k[0], {}).get(k[1])]
    a_only = sum(1 for k in both
                 if ref[k[0]][k[1]][4][0][0] == runs["A"][k[0]][k[1]][4][0][0]
                 and ref[k[0]][k[1]][4][0][0] != runs["B"][k[0]][k[1]][4][0][0])
    b_only = sum(1 for k in both
                 if ref[k[0]][k[1]][4][0][0] == runs["B"][k[0]][k[1]][4][0][0]
                 and ref[k[0]][k[1]][4][0][0] != runs["A"][k[0]][k[1]][4][0][0])
    print()
    print(f"top-1, A against B over the same positions: A right where B wrong: {a_only}, "
          f"B right where A wrong: {b_only}")
    if a_only + b_only:
        # exact binomial two-sided (McNemar), computed with integers so that a large n does not overflow
        nn = a_only + b_only
        k = min(a_only, b_only)
        tail = sum(math.comb(nn, i) for i in range(k + 1))
        p = min(1.0, (2 * tail) / (2 ** nn))
        print(f"  exact McNemar (two-sided): p = {p:.4f}  (n = {nn} discordant of {len(both)} positions)")
    else:
        print("  no discordant positions")


if __name__ == "__main__":
    main()
