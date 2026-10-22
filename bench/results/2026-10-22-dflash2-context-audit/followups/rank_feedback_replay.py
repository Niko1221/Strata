"""Fixed causal passive-aggressive rank correction, a diagnostic not speed evidence.
Original native scores are exact. The added FP32 matrix/edge math is NumPy and
would require native integration/parity before making an engine claim.
"""
from pathlib import Path
import json,struct,sys,numpy as np
root=Path.cwd();sys.path.insert(0,str(root/'tools'));from gguf_reader import GGUFFile
src=root/'../Strata-data/dflash2-acceptance-20261022';out=root/'../Strata-data/dflash2-context-audit-20261022';meta=json.loads((src/'lattice-capture.json').read_text());qm=json.loads((out/'rank/metadata.json').read_text());native=json.loads((src/'dense-complete.json').read_text());report={}
path=root/'../Strata-data/dflash2-official/dflash2-Q8_0.gguf';gg=GGUFFile(path)
def weights(name):
 t=next(t for t in gg.tensors if t.name==name);assert t.type_id==30
 x=np.memmap(path,mode='r',dtype='<u2',offset=gg.data_start+t.offset,shape=tuple(reversed(t.shape)))
 return (x.astype(np.uint32)<<16).view(np.float32)
A=weights('selector_predecessor.weight');B=weights('selector_successor.weight')
for task,m in meta.items():
 records={};qrecords={};anchors={};gold=m['response']['tokens']
 with (src/'lattice.bin').open('rb') as f:
  f.seek(m['start'])
  while f.tell()<m['end']:
   magic,lo,hi,rows,k,anchor=struct.unpack('<6I',f.read(24));pos=lo+(hi<<32);ids=np.fromfile(f,'<i4',rows*k).reshape(rows,k);unary=np.fromfile(f,'<f4',rows*k).reshape(rows,k);scores=np.fromfile(f,'<f4',(rows-1)*k*k).reshape(rows-1,k,k);records[pos-m['prompt_tokens']]=(ids,scores);anchors[pos-m['prompt_tokens']]=anchor
 with (out/'rank/features.bin').open('rb') as f:
  f.seek(qm[task]['start'])
  while f.tell()<qm[task]['end']:
   magic,lo,hi,block,rank,anchor=struct.unpack('<6I',f.read(24));pos=lo+(hi<<32);ids=np.fromfile(f,'<i4',block);q=np.fromfile(f,'<f4',block*rank).reshape(block,rank);qrecords[pos-m['prompt_tokens']]=q
 assert len(records)==len(qrecords)==95
 result={}
 for enabled in [False,True]:
  M=np.zeros((256,256),np.float32);i=1;windows=1;accepted=offered=updates=missing=improved=spoiled=0;events=[]
  while i<len(gold):
   ids,scores=records[i-1];q=qrecords[i-1];prev_index=0;prev=anchors[i-1];proposal=[];prob=[];chosen_scores=[];chosen_indices=[];predecessors=[]
   for row in range(1,8):
    s=scores[row-1,prev_index].copy()
    if enabled:
     delta=M@q[row-1];s+=np.sum(B[ids[row]]*(A[prev]*delta)[None,:],axis=1,dtype=np.float32)
    pick=int(np.argmax(s));p=np.exp(s-s[pick]);proposal.append(int(ids[row,pick]));prob.append(float(1/p.sum()));chosen_scores.append(s.copy());chosen_indices.append(pick);predecessors.append(prev);prev_index=pick;prev=proposal[-1]
   if enabled:
    old=int(ids[1,np.argmax(scores[0,0])]);improved+=old!=gold[i] and proposal[0]==gold[i];spoiled+=old==gold[i] and proposal[0]!=gold[i]
   k=0
   while k<7 and prob[k]>=.7:k+=1
   a=0
   while a<k and i+a<len(gold) and proposal[a]==gold[i+a]:a+=1
   a=min(a,len(gold)-i-1)
   if enabled and a<7:
    actual=gold[i+a];pred=proposal[a];matches=np.flatnonzero(ids[a+1]==actual)
    if actual!=pred and len(matches):
     v=A[predecessors[a]]*(B[actual]-B[pred]);x=q[a];den=float(np.dot(x,x))*float(np.dot(v,v));loss=float(np.float32(1)+chosen_scores[a][chosen_indices[a]]-chosen_scores[a][int(matches[0])]);tau=max(0,loss)/den if den>0 else 0
     assert np.isfinite(tau)
     M+=np.float32(tau)*np.outer(v,x);assert np.isfinite(M).all();updates+=1;events.append(dict(anchor_output_index=i-1,bonus_output_index=i+a,accepted=a,pred=pred,actual=actual,step=tau))
    elif actual!=pred:missing+=1
   i+=a+1;windows+=1;accepted+=a;offered+=k
  result['feedback' if enabled else 'baseline']=dict(windows=windows,accepted=accepted,offered=offered,verified_rows=windows+offered,updates=updates,missing_gold=missing,first_improved_at_visited_anchors=improved,first_spoiled_at_visited_anchors=spoiled,matrix_max_abs=float(np.abs(M).max()),events=events)
  if not enabled:
   expected=native[task]['baseline']['response'];assert accepted==expected['accepted'] and offered==expected['offered']
 report[task]=result;print(task,{k:{a:b for a,b in v.items() if a!='events'} for k,v in result.items()},flush=True)
(out/'rank-feedback-replay.json').write_text(json.dumps(report,indent=2))
