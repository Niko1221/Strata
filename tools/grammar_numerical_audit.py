"""Read bounded G5 raw-logit scratch files; publish numerical evidence, not a sampler.

Full raw arrays are local scratch, not a required public download. The report
retains their hashes, exact legal scores/masks/history/counters, max full-vocab
deviation and float64 diagnostic margins/CDF boundaries. Production selection
remains CUDA; approximate CPU probabilities are explicitly labelled.
"""
from __future__ import annotations
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import sys


def read(path):
    rows=[];ends=[]
    with path.open('rb') as f:
        while header:=f.readline():
            parts=header.decode('ascii').strip().split()
            if parts[0]=='G5END1':
                ends.append(list(map(int,parts[1:])));continue
            assert parts[0]=='G5ROW1' and len(parts)==19, parts
            request,T,row,kept,pos,pick,nv,nw,nh,seed,k=map(int,parts[1:12])
            top_p,min_p,temp,rep,freq,pres=map(float,parts[12:18]);greedy=int(parts[18])
            arrays=[]
            for code,n in [('f',nv),('i',nw),('i',nh),('i',T)]:
                value=array.array(code);value.frombytes(f.read(4*n))
                if sys.byteorder!='little':value.byteswap()
                assert len(value)==n;arrays.append(value)
            assert f.read(1)==b'\n'
            logits,mask,history,inputs=arrays
            rows.append(dict(request=request,T=T,row=row,kept=kept,counter=pos,selected=pick,
                seed=seed,top_k=k,top_p=top_p,min_p=min_p,temperature=temp,repeat=rep,frequency=freq,
                presence=pres,greedy=bool(greedy),logits=logits,mask=mask,history=list(history),inputs=list(inputs)))
    assert ends and all(dropped==0 for _,_,dropped in ends), ends
    assert sum(n for _,n,_ in ends)==len(rows)
    return rows


def uniform(seed,counter):
    c=[counter&0xffffffff,counter>>32,seed&0xffffffff,seed>>32]
    for i in range(10):
        a=0x9e3779b9*c[0];b=0xbb67ae85*c[2]
        c=[((b>>32)^c[1]^i)&0xffffffff,b&0xffffffff,((a>>32)^c[3])&0xffffffff,a&0xffffffff]
    return (c[0]>>8)/16777216


def diagnostic(row):
    legal=[i for i in range(len(row['logits'])) if row['mask'][i//32] & (1<<(i%32))]
    assert row['selected'] in legal
    assert all(math.isfinite(v) for v in row['logits']), 'nonfinite native head needs separate fault analysis'
    scores=[]
    for i in legal:
        raw=row['logits'][i];assert math.isfinite(raw)
        n=row['history'].count(i)
        value=(raw/row['repeat'] if raw>0 else raw*row['repeat'])-n*row['frequency']-row['presence'] if n else raw
        scores.append((value,i,raw))
    scores.sort(key=lambda v:(-v[0],v[1]))
    visible={k:row[k] for k in ('request','T','row','kept','counter','selected','seed','top_k','top_p','min_p',
                               'temperature','repeat','frequency','presence','greedy','history','inputs')}
    visible.update(mask_sha256=hashlib.sha256(row['mask'].tobytes()).hexdigest(),
                   raw_logits_sha256=hashlib.sha256(row['logits'].tobytes()).hexdigest(),
                   legal_scores=[dict(id=i,raw=raw,penalized_float64=v) for v,i,raw in scores],
                   greedy_margin_float64=scores[0][0]-scores[1][0] if len(scores)>1 else None)
    if not row['greedy']:
        candidates=scores[:max(1,min(row['top_k'] or 64,64))]
        mass=[math.exp(x[0]-candidates[0][0]) for x in candidates];total=sum(mass)
        upto=0;running=0
        for p in mass:
            upto+=1;running+=p/total
            if running>=row['top_p']:break
        candidates=candidates[:upto]
        candidates=[c for c in candidates if math.exp(c[0]-candidates[0][0])>=row['min_p']]
        mass=[math.exp((c[0]-candidates[0][0])/row['temperature']) for c in candidates]
        total=sum(mass);low=0;boundaries=[]
        for c,p in zip(candidates,mass):
            high=low+p/total;boundaries.append(dict(id=c[1],low=low,high=high));low=high
        u=uniform(row['seed'],row['counter'])
        visible.update(uniform=u,approximate_float64_cdf=boundaries,
                       nearest_cdf_boundary=min(abs(u-v) for b in boundaries for v in (b['low'],b['high'])))
    return visible


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True,help='contains numerical-<mode> directories')
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    modes=('target','mtp','coupled','suffix','suffix-coupled')
    data={mode:read(a.root/('numerical-'+mode)/'head-rows.bin') for mode in modes}
    reference={(r['request'],r['counter']):r for r in data['target'] if r['row']<r['kept']}
    report=dict(result='pass',source='actual native head logits',approximation='CDF and penalty math are offline float64 diagnostics, not exact CUDA distributions',modes={})
    for mode,rows in data.items():
        public=[diagnostic(r) for r in rows]
        (a.out/(mode+'-rows.json')).write_text(json.dumps(public,indent=2)+'\n',encoding='utf-8')
        comparisons=[];diverged=set()
        for row in rows:
            if row['row']>=row['kept'] or row['request'] in diverged:continue
            ref=reference.get((row['request'],row['counter']))
            if ref is None:continue
            assert row['mask']==ref['mask'] and row['history']==ref['history'], (mode,row['request'],row['counter'],'mask/history')
            assert row['seed']==ref['seed']
            differences=[abs(x-y) for x,y in zip(row['logits'],ref['logits'])]
            maximum=max(differences);at=differences.index(maximum)
            legal_deviation=max(d for i,d in enumerate(differences) if row['mask'][i//32] & (1<<(i%32)))
            comparisons.append(dict(request=row['request'],counter=row['counter'],T=row['T'],row=row['row'],
                selected=row['selected'],reference_selected=ref['selected'],max_abs_logit_deviation=maximum,
                max_legal_logit_deviation=legal_deviation,
                max_deviation_token=at,rmse=math.sqrt(sum(x*x for x in differences)/len(differences)),
                masks_histories_counters_equal=True))
            if row['selected']!=ref['selected']:diverged.add(row['request'])
        report['modes'][mode]=dict(rows=len(rows),compared_committed_rows=len(comparisons),
            first_divergent_requests=sorted(diverged),comparisons=comparisons,
            scratch_file_sha256=hashlib.sha256((a.root/('numerical-'+mode)/'head-rows.bin').read_bytes()).hexdigest())
    (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({m:{k:v for k,v in d.items() if k!='comparisons'} for m,d in report['modes'].items()},indent=2))


if __name__=='__main__':main()
