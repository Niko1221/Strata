"""KL(ref || other) per position, top-1 agreement, from kl_run.py outputs. usage: kl_cmp.py REF.pt OTHER.pt..."""
import sys, torch
ref = torch.load(sys.argv[1])
lp = torch.log_softmax(ref["logits"].double(), -1)
for path in sys.argv[2:]:
    o = torch.load(path)
    assert torch.equal(o["ids"], ref["ids"])
    lq = torch.log_softmax(o["logits"].double(), -1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    top1 = (lp.argmax(-1) == lq.argmax(-1)).float().mean().item()
    print(f"{path.split(chr(47))[-1]:16s} KL mean {kl.mean().item():.5f}  median {kl.median().item():.5f}  p99 {kl.quantile(0.99).item():.4f}  max {kl.max().item():.4f}  top1-agree {top1*100:.1f}%  (T={kl.numel()})")
