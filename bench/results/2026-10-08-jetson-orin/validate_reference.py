"""Compare last-prompt logits for identical explicit IDs, recording errors."""
import argparse,json,os,subprocess,time
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[3];base=Path('/home/calvin/models/strata-orin-validation')
ap=argparse.ArgumentParser();ap.add_argument('--ids',type=int,nargs='+');ap.add_argument('--name',default='reference');options=ap.parse_args()
ids=options.ids or [248045,9707,11,1917]
model=base/'IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf'
refout=base/f'{options.name}-cpu-logits.bin'
with (base/f'{options.name}-cpu.txt').open('w') as log:
 subprocess.run(['/tmp/strata-orin-reference/reference-logits',str(model),str(refout),*map(str,ids)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
h=np.fromfile(refout,dtype=np.int32,count=2);assert h.tolist()==[248320,len(ids)]
reference=np.fromfile(refout,dtype=np.float32,offset=8).reshape(len(ids),248320).astype(np.float64)
config=json.loads((base/'strata-orin-validation.json').read_text());reports=[]
cases=[(len(ids),1),(len(ids),64)] if options.ids else [(2,1),(4,1),(4,64)]
for count,prefill in cases:
 output=base/f'{options.name}-strata-logits-{count}-{prefill}.bin'
 args=list(config['args']);args[args.index('--prefill')+1]=str(prefill)
 env=os.environ.copy();env['STRATA_DUMP_FIRST_LOGITS']=str(output)
 start=time.monotonic()
 with (base/f'{options.name}-strata-{count}-{prefill}.txt').open('w') as log:
  subprocess.run([config['exe'],*args,'--tokens',','.join(map(str,ids[:count])),'--max-new','1'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
 a=reference[count-1];b=np.fromfile(output,dtype=np.float32).astype(np.float64)
 assert b.shape==a.shape and np.isfinite(a).all() and np.isfinite(b).all()
 def logprob(v):
  m=v.max();return v-m-np.log(np.exp(v-m).sum())
 la,lb=logprob(a),logprob(b);ia,ib=int(a.argmax()),int(b.argmax())
 reports.append(dict(ids=ids[:count],prefill=prefill,elapsed_s=time.monotonic()-start,reference_argmax=ia,strata_argmax=ib,max_abs_logit_difference=float(np.max(np.abs(a-b))),rms_logit_difference=float(np.sqrt(np.mean((a-b)**2))),reference_top_logprob=float(la[ia]),strata_reference_top_logprob=float(lb[ia]),kl_reference_to_strata=float(np.dot(np.exp(la),la-lb)),finite_logits=True))
 (base/f'{options.name}-comparison.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports[-1]),flush=True)
 assert ia==ib,(count,prefill,ia,ib)
