"""Image rows of strata-ink2 (--dump-image-emb) against vision_oracle.py's .emb. usage: emb_cmp.py ORACLE.emb OURS.emb"""
import sys
import numpy as np

a = np.fromfile(sys.argv[1], dtype=np.float32).reshape(-1, 2304).astype(np.float64)
b = np.fromfile(sys.argv[2], dtype=np.float32).reshape(-1, 2304).astype(np.float64)
assert a.shape == b.shape, (a.shape, b.shape)
d = np.abs(a - b)
cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(a, axis=1)
print(f"rows {a.shape[0]}  max|d| {d.max():.3e} (row {d.max(1).argmax()})  max|ref| {np.abs(a).max():.3e}  "
      f"cos min {cos.min():.8f} mean {cos.mean():.8f}  rel-L2 max {rel.max():.3e} mean {rel.mean():.3e}")
