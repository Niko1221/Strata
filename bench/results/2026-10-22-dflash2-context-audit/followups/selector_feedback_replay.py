"""Single fixed passive-aggressive selector-bias hypothesis, causal per request.
Only already verified prefix plus its emitted bonus may update the bias.
No launch-parameter search, cross-request state, training corpus, or speed claim.
"""
from pathlib import Path
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
  bias=np.zeros(248320,np.float32);i=1;windows=1;accepted=offered=updates=missing=0;events=[]
  while i<len(gold):
   ids,scores=records[i-1];prev_index=0;proposal=[];prob=[];chosen_scores=[];chosen_indices=[]
   for row in range(1,8):
    s=scores[row-1,prev_index]+bias[ids[row]];pick=int(np.argmax(s));p=np.exp(s-s[pick]);confidence=float(1/p.sum());proposal.append(int(ids[row,pick]));prob.append(confidence);chosen_scores.append(s.copy());chosen_indices.append(pick);prev_index=pick
   k=0
   while k<7 and prob[k]>=.7:k+=1
   a=0
   while a<k and i+a<len(gold) and proposal[a]==gold[i+a]:a+=1
   a=min(a,len(gold)-i-1)
   # a accepted proposals are known correct; only their bonus is a new valid
   # label. Nothing from the invalid suffix is used for adaptation.
   if enabled and a<7:
    actual=gold[i+a];pred=proposal[a];matches=np.flatnonzero(ids[a+1]==actual)
    if actual!=pred and len(matches):
     margin=chosen_scores[a][chosen_indices[a]]-chosen_scores[a][int(matches[0])]
     tau=np.float32(max(0,float(np.float32(1)+margin))/2)
     bias[actual]+=tau;bias[pred]-=tau;updates+=1;events.append(dict(anchor_output_index=i-1,bonus_output_index=i+a,accepted=a,pred=pred,actual=actual,step=float(tau)))
    elif actual!=pred:missing+=1
   i+=a+1;windows+=1;accepted+=a;offered+=k
  result['feedback' if enabled else 'baseline']=dict(windows=windows,accepted=accepted,offered=offered,verified_rows=windows+offered,updates=updates,missing_gold=missing,events=events)
  if not enabled:
   expected=native[task]['baseline']['response'];assert accepted==expected['accepted'] and offered==expected['offered']
 report[task]=result;print(task,{k:{a:b for a,b in v.items() if a!='events'} for k,v in result.items()},flush=True)
(out/'selector-feedback-replay.json').write_text(json.dumps(report,indent=2))
