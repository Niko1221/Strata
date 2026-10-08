#!/usr/bin/env python3
"""Independent full-vocabulary Q5_K/Q8_0/Q4_0 head + argmax check.
Uses dumped Q8_1 activations, including their stored sum (affine dot products
Q4_0 uses the original activation sum; Q5_K uses the reconstructed int8 sum).
"""
import argparse, json, pathlib, sys
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'third_party/llama.cpp/gguf-py'))
from gguf import GGMLQuantizationType as Q
from gguf.quants import dequantize
from gguf_reader import GGUFFile, BLOCK_GEOMETRY
from dflash_stage_parity import read_bin, row_stats

def check(head, directory):
    g=GGUFFile(head); t=next(t for t in g.tensors if t.name=='output.weight'); width,vocab=t.shape
    got=read_bin(directory,'head_logits').reshape(-1,vocab); k=got.shape[0]
    raw=read_bin(directory,'head_act_q8').view(np.uint8).reshape(k,width//32,36)
    ds=raw[:,:,:4].copy().view('<f2').reshape(k,width//32,2).astype(np.float32)
    x=(raw[:,:,4:].view(np.int8).astype(np.float32)*ds[:,:,0,None]).reshape(k,width)
    adjust=x.reshape(k,-1,32).sum(axis=2)-ds[:,:,1]
    ref=np.empty_like(got);block,bs=BLOCK_GEOMETRY[t.type_name];rb=width//block*bs
    with open(head,'rb') as f:
        f.seek(g.data_start+t.offset)
        for start in range(0,vocab,512):
            n=min(512,vocab-start);packed=np.frombuffer(f.read(n*rb),np.uint8).reshape(n,rb)
            weights=dequantize(packed,Q(t.type_id)); y=x@weights.T
            if t.type_name=='Q4_0':
                dw=packed.reshape(-1,18)[:,:2].copy().view('<f2').astype(np.float32).reshape(n,-1)
                y+=adjust@(8*dw).T
            elif t.type_name not in ('Q8_0', 'Q5_K'): raise ValueError('unsupported oracle head '+t.type_name)
            ref[:,start:start+n]=y
    stats=row_stats(got,ref); want=ref.argmax(axis=1); actual=got.argmax(axis=1)
    picks=read_bin(directory,'head_picks').view('<i4')
    rec={'type':t.type_name,'rows':[{'cosine':a,'max_abs':b,'mean_abs':c,'relative_l2':d} for a,b,c,d in stats],
         'reference_top1':want.tolist(),'gpu_top1':actual.tolist(),'argmax_picks':picks.tolist(),
         'top1_agreement':float(np.mean(want==actual)), 'picks_agree':bool(np.array_equal(picks,actual))}
    rec['passed']=rec['picks_agree'] and bool(np.array_equal(want,actual)) and all(a>=.999999 and d<=1e-4 for a,b,c,d in stats)
    print(json.dumps(rec,indent=2));return rec['passed']

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--head',required=True);p.add_argument('--dir',required=True);a=p.parse_args()
    sys.exit(0 if check(a.head,a.dir) else 1)
