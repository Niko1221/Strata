#!/usr/bin/env python3
"""compare_logits.py REF.bin GOT.bin VOCAB [tol]  -> prints max abs/rel diff and argmax agreement per position; exit 1 on failure."""
import sys, numpy as np
ref = np.fromfile(sys.argv[1], dtype="<f4"); got = np.fromfile(sys.argv[2], dtype="<f4"); v = int(sys.argv[3]); tol = float(sys.argv[4]) if len(sys.argv) > 4 else 2e-3
if ref.size != got.size: sys.exit("size mismatch: ref %d vs got %d" % (ref.size, got.size))
ref, got = ref.reshape(-1, v), got.reshape(-1, v)
d = np.abs(ref - got).max(1); scale = np.abs(ref).max(1)
am = (ref.argmax(1) == got.argmax(1))
print("positions %d | max|diff| %.3g | max rel %.3g | argmax agree %d/%d" % (len(ref), d.max(), (d / scale).max(), am.sum(), len(am)))
sys.exit(0 if (d / scale).max() < tol and am.all() else 1)
