"""Read SQ-10 token embeddings and unpack the S6_K output head."""
from pathlib import Path
import numpy as np
import mlx.core as mx

import os
base = Path(os.environ['INK_DFLASH_ASSETS'])
meta = {}
for line in (base/'layout.log').read_text().splitlines():
    if line.startswith('probe_tensor'):
        _, name, fmt, offset, rows, cols, valid, padded = line.split()
        meta[name] = (fmt, int(offset), int(rows), int(cols))
V = int(valid)

def mapped(name, dtype, shape, skip=0):
    return np.memmap(os.environ['INK_DFLASH_MODEL'], dtype, 'r', offset=meta['model.llm.'+name][1]+skip, shape=shape)

emb_sc = mapped('embed.weight','<f4',int(padded))
emb = mapped('embed.weight','i1',(int(padded),2304),4*int(padded))
emb_norm = mapped('embed_norm.weight','<f4',2304)

def embed(ids):
    x=emb[ids].astype('float32')*emb_sc[ids,None]
    return mx.array(x*(1/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6))*emb_norm)

def unpack_head(compress=False):
    assert meta['model.llm.unembed.weight'][0]=='S6_K'
    data=mapped('unembed.weight','u1',(V,9,198))
    # Per-256 half scale, four subblock scales, contiguous low nibbles and high bit pairs.
    chunks=[]
    for start in range(0,V,1024):
        blk=data[start:start+1024]; scale=blk[:,:,:2].copy().view('<f2').astype('float32')*blk[:,:,2:6]/255
        lo=blk[:,:,6:134,None]>>np.array([0,4],np.uint8)&15
        hi=blk[:,:,134:,None]>>np.array([0,2,4,6],np.uint8)&3
        levels=(lo.reshape(-1,9,256)+16*hi.reshape(-1,9,256)).astype('float32')-31.5
        tile=mx.array((levels.reshape(-1,9,4,64)*scale[:,:,:,None]).reshape(-1,2304))
        if compress:
            pair=mx.quantize(tile.astype(mx.bfloat16),mode='mxfp8'); mx.eval(*pair); chunks.append(pair)
        else: chunks.append(tile)
    if compress: return tuple(mx.concatenate(column) for column in zip(*chunks))
    out=mx.concatenate(chunks); mx.eval(out); return out

