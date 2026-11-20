"""Tiny causal GQA Transformer for a synthetic repeating-symbol language.

No external weights/data. This is NOT Bonsai, Strata, a natural-language LLM,
or a simulator of the 48 recurrent layers. Quantized training uses an explicitly
labelled straight-through estimator; exact packing/depacking is used in forward.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib
import math
import torch
from torch import Tensor,nn
from torch.nn import functional as F
from .core import QueryAdapter,make_views,build_cache,attention

@dataclass
class LMConfig:
    vocab:int=24
    heads:int=4
    kv_heads:int=2
    head_dim:int=16
    layers:int=3
    ff_mult:int=2
    adapter_rank:int=8

@dataclass
class DataSplit:
    patterns:Tensor
    x:Tensor
    y:Tensor
    sha256:str


def make_splits(seed:int,seq_len:int=32) -> dict[str,DataSplit]:
    """Unique 8-symbol patterns, repeated; no pattern can cross split boundaries."""
    g=torch.Generator().manual_seed(10000+seed);seen=set();out={}
    for name,size in [('pretrain',768),('train',256),('validation',64),('test',128)]:
        patterns=[]
        while len(patterns)<size:
            p=tuple(torch.randint(1,24,(8,),generator=g).tolist())
            if p not in seen:
                seen.add(p);patterns.append(p)
        p=torch.tensor(patterns,dtype=torch.long)
        seq=p.repeat(1,math.ceil((seq_len+1)/8))[:,:seq_len+1]
        digest=hashlib.sha256(p.numpy().tobytes()).hexdigest()
        out[name]=DataSplit(p,seq[:,:-1].clone(),seq[:,1:].clone(),digest)
    return out


def rope(x:Tensor) -> Tensor:
    """Ordinary interleaved RoPE for the tiny model, not qwen35 MRoPE."""
    d=x.shape[-1]
    if d%2:
        raise ValueError('Even head dimension required for tiny RoPE.')
    inv=10000.**(-torch.arange(0,d,2,device=x.device,dtype=x.dtype)/d)
    phase=torch.arange(x.shape[-2],device=x.device,dtype=x.dtype)[:,None]*inv
    co,si=phase.cos(),phase.sin()
    a,b=x[...,0::2],x[...,1::2]
    return torch.stack((a*co-b*si,a*si+b*co),-1).flatten(-2)


class TinyBlock(nn.Module):
    def __init__(self,cfg:LMConfig,index:int) -> None:
        super().__init__();self.cfg=cfg;h,d,kv=cfg.heads,cfg.head_dim,cfg.kv_heads;width=h*d
        self.norm1=nn.RMSNorm(width);self.norm2=nn.RMSNorm(width)
        self.q=nn.Linear(width,width,bias=False);self.k=nn.Linear(width,kv*d,bias=False)
        self.v=nn.Linear(width,kv*d,bias=False);self.gate=nn.Linear(width,width,bias=False)
        self.out=nn.Linear(width,width,bias=False)
        self.qnorm=nn.RMSNorm(d);self.knorm=nn.RMSNorm(d)
        self.ff=nn.Sequential(nn.Linear(width,width*cfg.ff_mult),nn.SiLU(),nn.Linear(width*cfg.ff_mult,width))
        self.adapter=QueryAdapter(h,d,cfg.adapter_rank)
        self.register_buffer('triality_views',make_views(d,600+index,kind='triality'))
        self.register_buffer('generic_views',make_views(d,600+index,kind='generic'))
        self.register_buffer('single_view',make_views(d,600+index,kind='hadamard'))

    def forward(self,x:Tensor,*,codec:str,k_bits:int|None,v_bits:int|None,use_adapter:bool,ste:bool) -> Tensor:
        b,t,w=x.shape;h,d,kv=self.cfg.heads,self.cfg.head_dim,self.cfg.kv_heads
        y=self.norm1(x)
        q=self.qnorm(self.q(y).view(b,t,h,d).transpose(1,2))
        k=self.knorm(self.k(y).view(b,t,kv,d).transpose(1,2))
        v=self.v(y).view(b,t,kv,d).transpose(1,2)
        if use_adapter:
            q=self.adapter(q)
        q,k=rope(q),rope(k)
        if codec=='exact':
            # Use the supported default SDPA dispatch, not an intentionally slow baseline.
            o=F.scaled_dot_product_attention(q,k.repeat_interleave(h//kv,1),v.repeat_interleave(h//kv,1),is_causal=True,dropout_p=0.)
        else:
            U=self.triality_views if codec=='triality' else self.generic_views if codec=='generic' else self.single_view
            cache=build_cache(k,v,U,k_bits,v_bits,beta=0,group=8,ste=ste)
            mask=torch.ones(t,t,device=x.device,dtype=torch.bool).tril()
            o=attention(q,cache,mask)
        o=o.transpose(1,2).reshape(b,t,w)*self.gate(y).sigmoid()
        x=x+self.out(o)
        return x+self.ff(self.norm2(x))


class TinyLM(nn.Module):
    def __init__(self,cfg:LMConfig=LMConfig()) -> None:
        super().__init__();self.cfg=cfg;d=cfg.heads*cfg.head_dim
        if cfg.heads%cfg.kv_heads:
            raise ValueError('Heads must be divisible by KV heads.')
        self.embedding=nn.Embedding(cfg.vocab,d)
        self.blocks=nn.ModuleList([TinyBlock(cfg,i) for i in range(cfg.layers)])
        self.norm=nn.RMSNorm(d);self.lm_head=nn.Linear(d,cfg.vocab,bias=False)
        self.use_adapter=False;self.codec='exact';self.k_bits=None;self.v_bits=None;self.ste=False

    def set_codec(self,kind:str,k_bits:int|None,v_bits:int|None) -> None:
        if kind not in ('exact','single','triality','generic'):
            raise ValueError('Unknown codec family')
        self.codec,self.k_bits,self.v_bits=kind,k_bits,v_bits

    def forward(self,x:Tensor,*,return_hidden:bool=False) -> Tensor|tuple[Tensor,list[Tensor]]:
        y=self.embedding(x);hidden=[]
        for block in self.blocks:
            y=block(y,codec=self.codec,k_bits=self.k_bits,v_bits=self.v_bits,use_adapter=self.use_adapter,ste=self.ste)
            hidden.append(y)
        z=self.lm_head(self.norm(y))
        return (z,hidden) if return_hidden else z

    def freeze_except_adapter(self) -> None:
        for name,p in self.named_parameters():
            p.requires_grad_('.adapter.' in name)
        self.use_adapter=True

    def adapter_state(self) -> dict[str,Tensor]:
        return {n:p.detach().cpu().clone() for n,p in self.state_dict().items() if '.adapter.' in n}
