"""One causal selector implementation hypothesis; not a launch-parameter search.
Uses target tokens only after simulated verification, never as draft inputs.
No throughput claim. Native integration would still be required for a useful result.
"""
from pathlib import Path
from collections import defaultdict,Counter
import json,struct,numpy as np
root=Path.cwd();src=root/'../Strata-data/dflash2-acceptance-20261022';out=root/'../Strata-data/dflash2-context-audit-20261022';meta=json.loads((src/'lattice-capture.json').read_text());native=json.loads((src/'dense-complete.json').read_text());report={}
for task,m in meta.items():
 records={};gold=m['response']['tokens']
 with (src/'lattice.bin').open('rb') as f:
  f.seek(m['start'])
  while f.tell()<m['end']:
   magic,lo,hi,rows,k,anchor=struct.unpack('<6I',f.read(24));pos=lo+(hi<<32);ids=np.fromfile(f,'<i4',rows*k).reshape(rows,k);unary=np.fromfile(f,'<f4',rows*k).reshape(rows,k);scores=np.fromfile(f,'<f4',(rows-1)*k*k).reshape(rows-1,k,k);records[pos-m['prompt_tokens']]=(ids,scores)
 result={}
 for enabled in [False,True]:
  history=[gold[0]];cache=defaultdict(Counter);i=1;windows=1;accepted=offered=overrides=0
  while i<len(gold):
   ids,scores=records[i-1];prev_index=0;prev=history[-1];prevprev=history[-2] if len(history)>1 else None;proposal=[];prob=[]
   for row in range(1,8):
    s=scores[row-1,prev_index];p=np.exp(s-s.max());p/=p.sum();pick=int(np.argmax(s));confidence=float(p[pick]);counts=cache.get((prevprev,prev),{})
    if enabled and len(counts)==1:
     cached,n=next(iter(counts.items()));matches=np.flatnonzero(ids[row]==cached)
     if n>=2 and len(matches) and int(matches[0])!=pick:
      pick=int(matches[0]);confidence=(n+float(p[pick]))/(n+1);overrides+=1
    nxt=int(ids[row,pick]);proposal.append(nxt);prob.append(confidence);prev_index=pick;prevprev,prev=prev,nxt
   k=0
   while k<7 and prob[k]>=.7:k+=1
   a=0
   while a<k and i+a<len(gold) and proposal[a]==gold[i+a]:a+=1
   a=min(a,len(gold)-i-1)
   for token in gold[i:i+a+1]:
    if len(history)>=2:cache[(history[-2],history[-1])][token]+=1
    history.append(token)
   i+=a+1;windows+=1;accepted+=a;offered+=k
  result['causal_cache' if enabled else 'baseline']=dict(windows=windows,accepted=accepted,offered=offered,verified_rows=windows+offered,overrides=overrides)
  if not enabled:
   expected=native[task]['baseline']['response'];assert accepted==expected['accepted'] and offered==expected['offered']
 report[task]=result;print(task,result,flush=True)
(out/'causal-cache-replay.json').write_text(json.dumps(report,indent=2))
