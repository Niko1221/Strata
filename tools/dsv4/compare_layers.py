#!/usr/bin/env python3
# tools/dsv4/compare_layers.py — phase 1.3 per-layer golden comparison.
#
# Walks a dsv4ref dump tree (<out>/<pass>/*.bin) against the golden tree
# (/data/research/runs/202610-dsv4-golden/<pass>/*.npy) and prints, per stage:
# cosine similarity, absmax |diff|, relative error (absmax diff / golden absmax),
# std(golden)/std(ours), NaN counts.  Also checks the logits top-1 of the last row
# against the golden generated_ids chain (prefill -> 9544, decode1 -> 734, ...).
#
# usage: compare_layers.py <golden_dir> <dump_dir> [--passes prefill,decode1,...]
#        [--layers N] [--json out.json]
import argparse
import glob
import json
import os
import sys

import numpy as np

EXPECTED = {  # pass -> (golden next token for the last row)
    "prefill": 9544,
    "decode1": 734,
    "decode2": 260,
    "decode3": 4654,
}


def load_bin(path):
    return np.fromfile(path, dtype=np.float32)


def compare(name, a, b, stats_std):
    if a.size != b.size:
        return {"stage": name, "error": f"size {a.size} vs {b.size} (shape {b.shape})"}
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    diff = np.abs(a - b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0
    amax_g = float(np.abs(b).max()) if b.size else 0.0
    amax_d = float(diff.max()) if diff.size else 0.0
    # relative error against the golden std (per-layer tolerance anchor from stats.json)
    rel_std = amax_d / stats_std if stats_std else float("nan")
    return {
        "stage": name,
        "cosine": cos,
        "absmax_diff": amax_d,
        "rel_vs_golden_absmax": amax_d / amax_g if amax_g else float("nan"),
        "rel_vs_golden_std": rel_std,
        "std_golden": float(b.std()),
        "std_ours": float(a.std()),
        "nan_ours": int(np.isnan(a).sum()),
        "nan_golden": int(np.isnan(b).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden_dir")
    ap.add_argument("dump_dir")
    ap.add_argument("--passes", default=None)
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    passes = args.passes.split(",") if args.passes else None
    if passes is None:
        passes = sorted(
            d for d in os.listdir(args.dump_dir)
            if os.path.isdir(os.path.join(args.dump_dir, d)))
    report = {}
    worst = []
    for p in passes:
        gdir = os.path.join(args.golden_dir, p)
        ddir = os.path.join(args.dump_dir, p)
        if not os.path.isdir(gdir) or not os.path.isdir(ddir):
            print(f"[skip] {p}: missing dir ({gdir} or {ddir})")
            continue
        # per-layer std anchors from golden stats.json
        stats_path = os.path.join(gdir, "stats.json")
        std_anchor = {}
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                sj = json.load(f)
            for key, val in sj.items() if isinstance(sj, dict) else []:
                if isinstance(val, dict) and "std" in val:
                    std_anchor[key] = val["std"]
        rows = []
        for npy in sorted(glob.glob(os.path.join(gdir, "*.npy"))):
            stage = os.path.basename(npy)[:-4]
            # limit to requested layers
            if stage.startswith("L") and len(stage) >= 3:
                try:
                    if int(stage[1:3]) >= args.layers:
                        continue
                except ValueError:
                    pass
            binp = os.path.join(ddir, stage + ".bin")
            if not os.path.exists(binp):
                continue
            a = load_bin(binp)
            b = np.load(npy)
            b = np.asarray(b, dtype=np.float32)
            row = compare(stage, a, b, std_anchor.get(stage, 0.0))
            rows.append(row)
            if "error" not in row:
                rel = row["rel_vs_golden_std"]
                if rel != rel:  # nan
                    rel = row["rel_vs_golden_absmax"]
                worst.append((rel, p, row["stage"], row))
        report[p] = rows
        print(f"== {p} ==")
        hdr = f"{'stage':28s} {'cosine':>12s} {'absmax_d':>10s} {'rel_amax':>9s} {'rel_std':>9s} {'std g/o':>14s}"
        print(hdr)
        for r in rows:
            if "error" in r:
                print(f"{r['stage']:28s} ERROR {r['error']}")
                continue
            print(f"{r['stage']:28s} {r['cosine']:12.8f} {r['absmax_diff']:10.3e} "
                  f"{r['rel_vs_golden_absmax']:9.2e} {r['rel_vs_golden_std']:9.2e} "
                  f"{r['std_golden']:7.3f}/{r['std_ours']:<6.3f}")
        # logits top-1 check
        lp = os.path.join(ddir, "logits.bin")
        if p in EXPECTED and os.path.exists(lp):
            n_vocab = 129280
            logits = load_bin(lp).reshape(-1, n_vocab)
            top1 = int(np.argmax(logits[-1]))
            ok = top1 == EXPECTED[p]
            print(f"  logits last-row argmax = {top1} (golden {EXPECTED[p]}) "
                  f"{'MATCH' if ok else '*** MISMATCH ***'}")
            report[p + "_argmax"] = {"got": top1, "golden": EXPECTED[p], "ok": ok}
    if worst:
        worst.sort(reverse=True)
        print("\nworst by absmax_diff / golden std:")
        for w, p, _st, r in worst[:10]:
            print(f"  {p}/{r['stage']}: rel_std={w:.3e} cosine={r['cosine']:.8f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
