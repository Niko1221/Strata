#!/usr/bin/env python3
"""Compares two dumps written by `strata-qwen36 --selftest-dump N file` (e.g. Q36_GROUPED=0 vs Q36_GROUPED=1).
   python tests/compare_dumps.py a.bin b.bin"""
import struct, sys
import numpy as np

def load(p):
    raw = open(p, "rb").read()
    V, G = struct.unpack("ii", raw[:8])
    greedy = np.frombuffer(raw[8:8 + 4 * G], dtype=np.int32)
    lg = np.frombuffer(raw[8 + 4 * G:], dtype=np.float32).reshape(2, V).astype(np.float64)
    return greedy, lg

def stats(a, b):
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    pa = np.exp(a - a.max()); pa /= pa.sum(); pb = np.exp(b - b.max()); pb /= pb.sum()
    kl = float(np.sum(pa * (np.log(pa + 1e-300) - np.log(pb + 1e-300))))
    ta, tb = set(np.argsort(-a)[:20]), set(np.argsort(-b)[:20])
    return cos, kl, len(ta & tb), int(a.argmax() == b.argmax())

ga, la = load(sys.argv[1]); gb, lb = load(sys.argv[2])
ok = True
for name, i in (("after prefill", 0), ("after 16 decode steps", 1)):
    cos, kl, ov, am = stats(la[i], lb[i])
    print(f"{name:24s} cos {cos:.6f}  KL {kl:.3e}  top20 {ov}/20  argmax {'same' if am else 'DIFFERENT'}")
    ok &= cos > 0.995 and kl < 5e-2 and ov >= 17 and am
agree = int((ga == gb).sum())
print(f"greedy tokens equal: {agree}/{len(ga)}")
ok &= agree >= len(ga) - 4
print("COMPARE PASSED" if ok else "COMPARE FAILED")
sys.exit(0 if ok else 1)
