"""CPU/CUDA operator benchmark, NOT an LLM throughput benchmark.

The exact arm uses PyTorch SDPA. Candidate arms include the QueryAdapter and
reference decode/score path. Cache construction is timed separately. CUDA-event
milliseconds and peak allocated memory are reported; no claimed speedup threshold.
"""
from __future__ import annotations
import argparse,json,math,os,platform,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch.nn import functional as F
from .core import resolve_device,QueryAdapter,make_views,build_cache,attention,tiled_attention

@torch.inference_mode()
def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cuda');p.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='float32')
    p.add_argument('--sequence',type=int,default=128);p.add_argument('--queries',type=int,default=8)
    p.add_argument('--heads',type=int,default=6);p.add_argument('--kv-heads',type=int,default=1)
    p.add_argument('--dim',type=int,default=32);p.add_argument('--repetitions',type=int,default=10)
    p.add_argument('--out',default='results/benchmark.json');a=p.parse_args()
    device=resolve_device(a.device);dtype=getattr(torch,a.dtype);torch.set_num_threads(1);torch.manual_seed(20261023)
    if a.heads%a.kv_heads or a.queries>a.sequence or min(a.sequence,a.queries,a.dim,a.repetitions)<1:
        raise ValueError('Invalid geometry/repetitions.')
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    q=torch.randn(1,a.heads,a.queries,a.dim,device=device,dtype=dtype)
    k=torch.randn(1,a.kv_heads,a.sequence,a.dim,device=device,dtype=dtype);v=torch.randn_like(k)
    mask=torch.arange(a.sequence,device=device)[None,:]<=torch.arange(a.sequence-a.queries,a.sequence,device=device)[:,None]
    ad=QueryAdapter(a.heads,a.dim).to(device=device,dtype=dtype)
    views=make_views(a.dim,3,device=device,dtype=dtype)
    def clock(fn):
        if device.type=='cuda':
            with torch.cuda.device(device):
                torch.cuda.synchronize(device);start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();out=fn();end.record();end.synchronize();return start.elapsed_time(end),out
        start=time.perf_counter();out=fn();return (time.perf_counter()-start)*1000,out
    pack_ms,cache=clock(lambda:build_cache(k,v,views,3,None,beta=0,group=8))
    single=build_cache(k,v,make_views(a.dim,3,kind='hadamard',device=device,dtype=dtype),6,None)
    kk=k.repeat_interleave(a.heads//a.kv_heads,1);vv=v.repeat_interleave(a.heads//a.kv_heads,1)
    fns={'exact_sdpa':lambda:F.scaled_dot_product_attention(q,kk,vv,attn_mask=mask,dropout_p=0.),
         'single6_reference':lambda:attention(q,single,mask),
         'triality3x3_with_adapter_dense':lambda:attention(ad(q),cache,mask),
         'triality3x3_with_adapter_tiled':lambda:tiled_attention(ad(q),cache,mask,q_tile=32,k_tile=64)}
    for _ in range(3):
        for fn in fns.values():clock(fn)
    records={name:[] for name in fns};peaks={name:None for name in fns}
    for rep in range(a.repetitions):
        order=list(fns) if rep%2==0 else list(reversed(fns))
        for name in order:
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            ms,_=clock(fns[name]);records[name].append(ms)
            if device.type=='cuda':peaks[name]=max(peaks[name] or 0,torch.cuda.max_memory_allocated(device))
    summary={n:{'p50_ms':float(np.median(r)),'p95_ms_exploratory':float(np.quantile(r,.95)),
                'samples_ms':r,'peak_allocated_bytes_including_live_tensors':peaks[n]} for n,r in records.items()}
    result={'scope':'operator only; candidate is not fused; no GPU/LLM speed claim from CPU timings',
            'environment':{'torch':torch.__version__,'device':str(device),'dtype':a.dtype,
                           'cuda_build':torch.version.cuda,'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None},
            'geometry':vars(a),'pack_ms':pack_ms,'payload_bytes':cache.payload_bytes,
            'profile_bytes':cache.profile_bytes,'arms':summary}
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
