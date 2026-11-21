"""Compare strata-ink2ref dumps with the transformers oracle. usage: ref_cmp.py ORACLE.safetensors LAYERS.bin LOGITS.bin"""
import sys, numpy as np, torch
from safetensors.torch import load_file
o = load_file(sys.argv[1]); T = o["ids"].numel(); H = 2304
lay = np.fromfile(sys.argv[2], dtype=np.float32).reshape(T, 48, H)
lg = np.fromfile(sys.argv[3], dtype=np.float32).reshape(T, -1)
names = [f"L{l}" for l in range(6)] + [f"L{l}p{p}" for p in range(3) for l in range(6, 18)] + [f"L{l}" for l in range(18, 24)]
first_bad = None
for k in range(48):
    ref = o[f"call{k}.out"].numpy(); got = lay[:, k]
    rel = np.linalg.norm(got - ref, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)
    if rel.max() > 1e-3 and first_bad is None: first_bad = k
    print(f"call{k:2d} {names[k]:7s} rel_err max {rel.max():.2e} (pos {rel.argmax()})  |ref| {np.linalg.norm(ref[-1]):.3f}")
ol = o["logits"].double(); gl = torch.from_numpy(lg).double()
lp, lq = torch.log_softmax(ol, -1), torch.log_softmax(gl, -1)
kl = (lp.exp() * (lp - lq)).sum(-1)
print("logits max|d|", (ol - gl).abs().max().item(), " KL per pos", [round(x, 6) for x in kl.tolist()],
      " top1 agree", (ol.argmax(-1) == gl.argmax(-1)).float().mean().item())
print("FIRST-DIVERGING-CALL", first_bad)
