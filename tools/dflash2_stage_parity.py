#!/usr/bin/env python3
"""Compare a trained DFlash2 block against an independent NumPy reference.

Run the engine with STRATA_DF_PARITY=DIR STRATA_DF_PARITY_CYCLE=0, then:
  python tools/dflash2_stage_parity.py --dflash draft-bf16.gguf --dir DIR

Checks every layer's norm, dynamic convolutions, Q/K/V, local attention, residuals,
MLP, final norm and predecessor-conditioned selector. Uses the actual exported
weights and dumped inputs/paged context. FP32 arithmetic with Strata's BF16
projection boundaries; not a claim of bit equality with a BF16 PyTorch backend.
"""
import argparse
from pathlib import Path
import struct
import numpy as np
from gguf_reader import GGUFFile
from dflash_stage_parity import Report, bf16, rmsnorm, read_bin


def check(path, directory):
    g=GGUFFile(path); m=g.metadata
    weights={}
    for t in g.tensors:
        if t.type_id != 30: raise ValueError('parity requires original BF16 weights')
        raw=np.memmap(path,mode='r',dtype='<u2',offset=g.data_start+t.offset,shape=(int(np.prod(t.shape)),))
        weights[t.name]=(raw.astype('<u4')<<16).view('<f4').reshape(tuple(reversed(t.shape)))
    W=weights
    H=int(m['dflash.embedding_length']);D=int(m['dflash.attention.key_length'])
    nq=int(m['dflash.attention.head_count']);nkv=int(m['dflash.attention.head_count_kv'])
    taps=int(m['dflash.conv_kernel_size']);group=int(m['dflash.conv_group_size'])
    rank=int(m['dflash.selector_rank']);top=int(m['dflash.selector_top_k'])
    window=int(m.get('dflash.attention.sliding_window',0));theta=float(m['dflash.rope.frequency_base'])
    pos,K,anchor,mask,page=struct.unpack('<5I',(Path(directory)/'meta.bin').read_bytes())
    report=Report()
    def dump(name):return read_bin(directory,name)
    def stage(name,ref):
        return report.stage(name,dump(name),ref)
    def project(x,name):return bf16(x) @ W[name].T
    def conv(x,dynamic,base,side):
        y=np.zeros_like(x)
        for tap in range(min(taps,K)):
            coeff=base[side,tap][None,:]+np.repeat(dynamic[tap:,side,tap],group,axis=-1)
            y[tap:]+=coeff*x[:K-tap]
        return y
    def dynamic(x,name):return project(x,name+'_proj.weight').reshape(K,2,taps,H//group)
    def rope(x, positions=None):
        inv=np.power(np.float32(theta**(-2.0/D)),np.arange(D//2,dtype=np.float32))
        if positions is None: positions=np.arange(pos,pos+K,dtype=np.float32)
        ang=np.asarray(positions,dtype=np.float32)[:,None,None]*inv
        c,s=np.cos(ang),np.sin(ang)
        a,b=x[...,:D//2],x[...,D//2:]
        return np.concatenate((a*c-b*s,b*c+a*s),axis=-1)
    # Fusion inputs can be overwritten by later add_context calls; each dump is self-contained.
    if (Path(directory)/'tapsin.bin').exists():
        x=dump('tapsin').reshape(-1,W['fc.weight'].shape[1])
        stage('ctx',rmsnorm(project(x,'fc.weight'),W['enc.output_norm.weight']))
    context=np.zeros((pos,H),np.float32)
    have=np.zeros(pos,bool)
    for chunk in Path(directory).glob('ctx_b*.bin'):
        p0=int(chunk.stem.removeprefix('ctx_b'))
        values=dump(chunk.stem).reshape(-1,H)
        count=min(len(values),pos-p0)
        if count>0:context[p0:p0+count]=values[:count];have[p0:p0+count]=True
    h=dump('emb').reshape(K,H)
    for l in range(int(m['dflash.block_count'])):
        pre=f'blk.{l}.'
        x=rmsnorm(h,W[pre+'attn_norm.weight']);stage(f'xn{l}',x)
        dyn=dynamic(x,pre+'attn_conv');x=conv(x,dyn,W[pre+'attn_conv_base'],0)
        q=project(x,pre+'attn_q.weight').reshape(K,nq,D);stage(f'qraw{l}',q)
        k=project(x,pre+'attn_k.weight').reshape(K,nkv,D)
        v=project(x,pre+'attn_v.weight').reshape(K,nkv,D)
        q=rmsnorm(q,W[pre+'attn_q_norm.weight']);stage(f'qnormed{l}',q)
        q=rope(q);k=rope(rmsnorm(k,W[pre+'attn_k_norm.weight']))
        stage(f'q{l}',q);stage(f'k{l}',k);stage(f'v{l}',v)
        # Attention oracle reads paged FP16 pools, independently of the GPU attention.
        pt=dump(f'pt{l}').view('<i4')
        cells=np.arange(pos+K);pr=pt[cells//page]*nkv*page+cells%page
        kp=dump(f'kpool{l}').reshape(-1,D);vp=dump(f'vpool{l}').reshape(-1,D)
        if have.all() and pos:
            expected_k=project(context,pre+'attn_k.weight').reshape(pos,nkv,D)
            expected_k=rope(rmsnorm(expected_k,W[pre+'attn_k_norm.weight']),np.arange(pos))
            expected_v=project(context,pre+'attn_v.weight').reshape(pos,nkv,D)
            pool_rows=pr[:pos,None]+np.arange(nkv)[None,:]*page
            report.stage(f'context_k{l}',kp[pool_rows],expected_k.astype(np.float16).astype(np.float32))
            report.stage(f'context_v{l}',vp[pool_rows],expected_v.astype(np.float16).astype(np.float32))
        attn=np.zeros((K,nq,D),np.float32)
        for r in range(K):
            visible=np.ones(pos+K,bool) if not window else np.abs(cells-(pos+r))<window
            for kv in range(nkv):
                hs=slice(kv*(nq//nkv),(kv+1)*(nq//nkv))
                keys=kp[pr[visible]+kv*page];values=vp[pr[visible]+kv*page]
                scores=q[r,hs]@keys.T/np.sqrt(np.float32(D))
                probs=np.exp(scores-scores.max(axis=-1,keepdims=True));probs/=probs.sum(axis=-1,keepdims=True)
                attn[r,hs]=probs@values
        stage(f'attn{l}',attn)
        output=project(attn.reshape(K,-1),pre+'attn_output.weight')
        h=h+conv(output,dyn,W[pre+'attn_conv_base'],1);stage(f'h_attn{l}',h)
        x=rmsnorm(h,W[pre+'ffn_norm.weight']);dyn=dynamic(x,pre+'ffn_conv')
        x=conv(x,dyn,W[pre+'ffn_conv_base'],0)
        gate=project(x,pre+'ffn_gate.weight');up=project(x,pre+'ffn_up.weight')
        output=project((gate/(1+np.exp(-gate)))*up,pre+'ffn_down.weight')
        h=h+conv(output,dyn,W[pre+'ffn_conv_base'],1);stage(f'h_mlp{l}',h)
    hidden=rmsnorm(h,W['output_norm.weight']);stage('final_norm',hidden)
    if (Path(directory)/'selector_hidden.bin').exists():
        projected=project(hidden,'selector_hidden.weight');stage('selector_hidden',projected)
        # Use engine logits: verifies selection independently of the target head's quantizer.
        logits=dump('head_logits').reshape(K,-1)
        candidates=np.argsort(-logits,axis=-1,kind='stable')[:,:top]
        ids=dump('selector_ids').view('<i4').reshape(K,top)
        if not np.array_equal(candidates,ids):raise AssertionError('top-k candidate IDs differ')
        prev=W['selector_predecessor.weight'];nxt=W['selector_successor.weight']
        picks=[];predecessor=anchor
        for r in range(1,K):
            scores=logits[r,ids[r]]+((prev[predecessor]*projected[r])[None,:]*nxt[ids[r]]).sum(axis=-1)
            predecessor=int(ids[r,np.argmax(scores)]);picks.append(predecessor)
        got=dump('head_picks').view('<i4')
        if not np.array_equal(picks,got):raise AssertionError(f'selector differs: {picks} vs {got.tolist()}')
        print('PASS: full-vocabulary top-k and conditioned selector IDs')
    report.print()
    return report.ok

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--dflash',required=True);ap.add_argument('--dir',required=True)
    a=ap.parse_args();raise SystemExit(0 if check(a.dflash,a.dir) else 1)
