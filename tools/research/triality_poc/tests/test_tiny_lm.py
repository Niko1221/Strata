import torch
from triality_poc.tiny_lm import TinyLM, LMConfig, DataSplit, make_splits

def test_identity_and_causality_and_adapter_grad():
    torch.manual_seed(2);m=TinyLM(LMConfig()).double()
    x=torch.randint(1,24,(2,16));reference=m(x)
    m.use_adapter=True
    torch.testing.assert_close(m(x),reference,rtol=0,atol=0)
    altered=x.clone();altered[:,10:]=(altered[:,10:]+5)%24
    torch.testing.assert_close(m(altered)[:,:10],m(x)[:,:10],rtol=0,atol=0)
    m.set_codec('triality',2,None)
    out=m(x);(out*torch.randn_like(out)).sum().backward()
    assert sum(b.adapter.w_out.grad.abs().sum() for b in m.blocks)>0

def test_train_eval_partition_disjoint():
    splits=make_splits(123)
    seen=set()
    for name,data in splits.items():
        rows={tuple(r) for r in data.patterns.tolist()}
        assert rows.isdisjoint(seen)
        seen|=rows

def test_packed_ste_parity_and_frozen_base():
    torch.manual_seed(9);m=TinyLM(LMConfig()).float();m.set_codec('triality',2,3)
    x=torch.randint(1,24,(2,16))
    m.ste=False;a=m(x)
    m.ste=True;b=m(x)
    torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
    m.freeze_except_adapter()
    names=[n for n,p in m.named_parameters() if p.requires_grad]
    assert names and all('.adapter.' in n for n in names)
