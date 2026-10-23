#!/usr/bin/env python3
# tools/dsv4/compare.py - stage-by-stage comparison of the dsv4ref skeleton dumps (raw FP32 .bin, row-major)
# against the golden oracle dumps (.npy) in /data/research/runs/202610-dsv4-golden/.
#
# usage: python3 tools/dsv4/compare.py <golden-dir> <dsv4ref-dump-dir>
#
# Per stage: shape check, cosine similarity, absmax of |a-b|, rel-err (absmax_diff / golden absmax), and
# std of both sides.  Only `embed` is a numerical comparison (the 43 trunk layers are identity stubs in
# dsv4ref, so hc_head_out / final_norm / logits have no golden counterpart to match - the golden tensors
# went through the real stack); for those the table records magnitudes only.  Verdicts:
#   embed cosine > 0.999  -> PASS  (Q8_0 vs BF16 source, expected)
#   embed cosine < 0.99   -> FAIL  (loading / dequant bug; stop and investigate)
import os
import sys

import numpy as np

STAGES = [
    # (stage, golden npy (relative to golden-dir), dsv4ref bin, shape)
    ("embed", "prefill/embed.npy", "embed.bin", (13, 4096)),
    ("hc_head_out", "prefill/hc_head_out.npy", "hc_head_out.bin", (13, 4096)),
    ("final_norm", "prefill/final_norm.npy", "final_norm.bin", (13, 4096)),
    ("logits", "prefill/logits.npy", "logits.bin", (13, 129280)),
]


def cosine(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    golden_dir, dump_dir = sys.argv[1], sys.argv[2]
    print(f"golden: {golden_dir}\ndsv4ref: {dump_dir}\n")
    print(f"{'stage':<13} {'shape':<14} {'cosine':>10} {'absmax|d|':>11} {'rel-err':>10} "
          f"{'std ours':>10} {'std golden':>11}  note")
    ok = True
    for stage, npy, binf, shape in STAGES:
        g = np.load(os.path.join(golden_dir, npy))
        a = np.fromfile(os.path.join(dump_dir, binf), dtype=np.float32)
        if a.size != int(np.prod(shape)):
            print(f"{stage:<13} SIZE MISMATCH: {a.size} floats, expected {shape}")
            ok = False
            continue
        a = a.reshape(shape)
        if g.shape != shape:
            print(f"{stage:<13} golden shape {g.shape} != {shape}")
            ok = False
            continue
        nan = int(np.isnan(a).sum())
        diff = np.abs(a.astype(np.float64) - g.astype(np.float64))
        cos = cosine(a, g)
        am = float(diff.max())
        rel = am / float(np.abs(g).max())
        note = ""
        if stage == "embed":
            if cos > 0.999:
                note = "PASS (Q8_0 vs BF16)"
            elif cos < 0.99:
                note = "FAIL - loading/dequant bug, investigate"
                ok = False
            else:
                note = "marginal - investigate"
                ok = False
        else:
            note = "magnitude only (identity layers; golden went through the real stack)"
        if nan:
            note += f"  NaN x{nan}!"
            ok = False
        print(f"{stage:<13} {str(shape):<14} {cos:>10.6f} {am:>11.4e} {rel:>10.3e} "
              f"{float(a.std()):>10.4e} {float(g.std()):>11.4e}  {note}")
    print("\noverall:", "OK" if ok else "PROBLEMS - see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
