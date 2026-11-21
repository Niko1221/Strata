"""Per-position comparison of strata-ink2ref --dump-logits with a kl_run.py logits file. usage: cmp_logits.py ORACLE.pt REF.logits"""
import sys, numpy as np, torch
o = torch.load(sys.argv[1]); ol = o["logits"].double(); T, V = ol.shape
gl = torch.from_numpy(np.fromfile(sys.argv[2], dtype=np.float32).reshape(T, V)).double()
lp, lq = torch.log_softmax(ol, -1), torch.log_softmax(gl, -1)
kl = (lp.exp() * (lp - lq)).sum(-1); d = (ol - gl).abs().amax(-1)
agree = (ol.argmax(-1) == gl.argmax(-1))
print(f"T={T} KL max {kl.max().item():.2e} (pos {kl.argmax().item()})  max|dlogit| {d.max().item():.2e} (pos {d.argmax().item()})  top1 agree {agree.float().mean().item()*100:.2f}%")
for lo, hi in ((0, 1000), (1000, 1030), (1030, T)):
    if lo < T: print(f"  pos {lo}-{min(hi, T)-1}: max|dlogit| {d[lo:hi].max().item():.2e}  KL max {kl[lo:hi].max().item():.2e}")
bad = (~agree).nonzero().flatten().tolist()
for p in bad[:10]:
    top = torch.topk(ol[p], 2).values
    print(f"  top1 differs at {p}: oracle margin top1-top2 {(top[0]-top[1]).item():.2e}")
