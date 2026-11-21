"""strata-ink2 --dump-logits of a prompt with a picture against vision_kl.py's .pt (or a text kl_run.py .pt).
usage: vis_cmp.py ORACLE.pt OURS.logits [IMAGE_TOKEN]"""
import sys
import numpy as np
import torch

o = torch.load(sys.argv[1])
ref = o["logits"].double()
T, V = ref.shape
ours = torch.from_numpy(np.fromfile(sys.argv[2], dtype=np.float32).reshape(-1, V)).double()
assert ours.shape[0] == T, (ours.shape, T)
img = int(sys.argv[3]) if len(sys.argv) > 3 else 200054
ids = o["ids"]
lp, lq = torch.log_softmax(ref, -1), torch.log_softmax(ours, -1)
kl = (lp.exp() * (lp - lq)).sum(-1)
top1 = ref.argmax(-1) == ours.argmax(-1)
d = (ref - ours).abs().amax(-1)
isimg = ids == img
first = int(isimg.nonzero()[0]) if isimg.any() else T
last = int(isimg.nonzero()[-1]) if isimg.any() else -1
parts = {"all": torch.ones(T, dtype=torch.bool), "before picture": torch.arange(T) < first, "picture": isimg,
         "after picture": torch.arange(T) > last}
for name, m in parts.items():
    if m.sum() == 0:
        continue
    k = kl[m]
    print(f"{name:15s} T={int(m.sum()):5d}  KL mean {k.mean():.2e} median {k.median():.2e} max {k.max():.2e}  "
          f"top1 {top1[m].float().mean() * 100:.1f}%  max|dlogit| {d[m].max():.2e}")
