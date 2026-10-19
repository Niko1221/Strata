#!/usr/bin/env python3
"""Explain a target-row mismatch from STRATA_DF_TARGET_LOGITS fixtures.

A greedy near-tie requires both choices to be the two leading tokens in both
rows and an absolute gap <= 0.05 logits in each row (odds ratio <= exp(.05)).
It is evidence about this row only, never an exemption for other mismatches.
"""
import argparse
import json
from pathlib import Path
import struct
import numpy as np


def records(path):
    with Path(path).open('rb') as f:
        while raw := f.read(48):
            if len(raw) != 48:
                raise ValueError('truncated fixture header')
            vocab, pos, window, row, index, pick = struct.unpack('<6q',raw)
            if not 1 <= vocab <= 1000000 or min(pos,window,row,index,pick) < 0 or pick >= vocab:
                raise ValueError('invalid fixture header')
            data = f.read(vocab*4)
            if len(data) != vocab*4:
                raise ValueError('truncated fixture logits')
            logits = np.frombuffer(data,'<f4')
            if not np.isfinite(logits).all():
                raise ValueError('nonfinite target logits')
            yield dict(position=pos,window=window,row=row,index=index,pick=pick),logits


def philox(seed, counter):
    mask = (1<<32)-1
    c0,c1,c2,c3 = counter&mask,counter>>32,seed&mask,seed>>32
    for i in range(10):
        a,b = 0x9E3779B9*c0,0xBB67AE85*c2
        c0,c1,c2,c3 = ((b>>32)^c1^i)&mask,b&mask,((a>>32)^c3)&mask,a&mask
    return (c0>>8)/(1<<24)


def sampled(logits, seed, position, temperature=.7, top_k=40, top_p=.9):
    # Same documented sampler order, independently evaluated on the host.
    ids = np.argsort(-logits,kind='stable')[:top_k]
    x = logits[ids].astype(np.float64)
    prob = np.exp(x-x[0]);prob /= prob.sum()
    keep = min(len(ids),int(np.searchsorted(np.cumsum(prob),top_p))+1)
    ids,x = ids[:keep],x[:keep]
    scaled = (x.astype(np.float32)*np.float32(1/temperature)).astype(np.float64)
    prob = np.exp(scaled-scaled.max());prob /= prob.sum()
    cdf = np.cumsum(prob);u = philox(seed,position)
    idx = min(len(ids)-1,int(np.searchsorted(cdf,u,side='right')))
    return dict(pick=int(ids[idx]),uniform=u,ids=ids.tolist(),probabilities=prob.tolist(),
                cdf_boundary_distance=float(np.min(np.abs(cdf-u))))


def explain(a,b,pick_a,pick_b,kind='greedy',position=0,seed=3456):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or a.ndim!=1 or len(a)<2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('expected two finite target logit rows of the same vocabulary')
    order_a=np.argsort(-a,kind='stable');order_b=np.argsort(-b,kind='stable')
    gap_a=abs(float(a[pick_a])-float(a[pick_b]));gap_b=abs(float(b[pick_a])-float(b[pick_b]))
    pair={pick_a,pick_b}
    tie=(pick_a!=pick_b and pair==set(order_a[:2])==set(order_b[:2]) and max(gap_a,gap_b)<=.05)
    result=dict(kind=kind,target_pick=pick_a,candidate_pick=pick_b,
                target_top2=order_a[:2].tolist(),candidate_top2=order_b[:2].tolist(),
                target_pair_gap=gap_a,candidate_pair_gap=gap_b,
                max_abs_logit_difference=float(np.max(np.abs(a.astype(float)-b))),
                target_pair_scores=[float(a[pick_a]),float(a[pick_b])],
                candidate_pair_scores=[float(b[pick_a]),float(b[pick_b])],
                documented_near_tie=False)
    if kind=='greedy':
        result['picks_match_argmax']=pick_a==int(order_a[0]) and pick_b==int(order_b[0])
        result['documented_near_tie']=bool(tie and result['picks_match_argmax'])
    else:
        sa,sb=sampled(a,seed,position),sampled(b,seed,position)
        result.update(target_sampler=sa,candidate_sampler=sb,
                      picks_match_sampler=sa['pick']==pick_a and sb['pick']==pick_b)
        # Sampling uses an inverse CDF over the probability-sorted list.
        # A near-equal ADJACENT pair can exchange order below the global leader.
        pair_present = all(v in s['ids'] for s in (sa,sb) for v in (pick_a,pick_b))
        adjacent_swap = False
        ranks = None
        if pair_present:
            ranks = [[s['ids'].index(pick_a),s['ids'].index(pick_b)] for s in (sa,sb)]
            lo = min(ranks[0])
            adjacent_swap = (abs(ranks[0][0]-ranks[0][1]) == 1 and
                             ranks[1] == list(reversed(ranks[0])) and
                             sa['ids'][:lo] == sb['ids'][:lo])
        result['selected_pair_ranks'] = ranks
        result['adjacent_order_swap'] = adjacent_swap
        result['documented_near_tie']=bool(pick_a!=pick_b and adjacent_swap and
                                           max(gap_a,gap_b)<=.05 and result['picks_match_sampler'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--record',type=int,default=0)
    p.add_argument('--kind',choices=['greedy','sampled'],default='greedy')
    p.add_argument('--output',type=Path,required=True)
    o=p.parse_args()
    ha,a=list(records(o.target))[o.record];hb,b=list(records(o.candidate))[o.record]
    if (ha['position'],ha['index'])!=(hb['position'],hb['index']):
        raise ValueError('fixtures are not predictions for the same position and output index')
    r=explain(a,b,ha['pick'],hb['pick'],o.kind,ha['position'])
    r.update(target_header=ha,candidate_header=hb)
    o.output.write_text(json.dumps(r,indent=2))
    print(json.dumps(r,indent=2))
    return 0 if r['documented_near_tie'] else 1


if __name__=='__main__':raise SystemExit(main())
