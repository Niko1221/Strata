"""Run the identical image through CPU and CUDA encoders, recording raw differences."""
import json,subprocess,time
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[3]
base=Path('/home/calvin/models/strata-orin-validation')
fixture=root/'bench/results/2026-10-07-jetson-orin/vision-fixture.png'
model=base/'IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf'
rows=[];vectors=[]
for kind in ['cpu','cuda']:
 out=base/f'vision-{kind}.bin'
 exe=root/'engine-cuda12'/('strata-vision-cpu' if kind=='cpu' else 'strata-vision')
 args=[str(exe),'--mmproj',str(base/'mmproj-Qwen3.8-Flash-Next-BF16.gguf'),'--model',str(model),'--min-tokens','16','--max-tokens','16','--threads','8']
 if kind=='cuda':args+=['--gpu']
 start=time.monotonic()
 r=subprocess.run(args,input=f'ENC {fixture} {out}\nQUIT\n',text=True,capture_output=True,timeout=600)
 (base/f'vision-{kind}.txt').write_text(r.stdout+'\n'+r.stderr)
 if r.returncode:raise RuntimeError((kind,r.returncode,r.stderr[-2000:]))
 h=np.fromfile(out,dtype=np.int32,count=5);v=np.fromfile(out,dtype=np.float32,offset=20)
 assert h[0]==0x31455653 and len(v)==int(h[1])*int(h[4]) and np.isfinite(v).all(),(h,len(v))
 rows.append(dict(kind=kind,elapsed_s=time.monotonic()-start,header=h.tolist(),stdout=r.stdout));vectors.append(v.astype(np.float64))
 assert rows[0]['header']==rows[-1]['header']
a,b=vectors;diff=a-b
report=dict(runs=rows,max_abs_difference=float(np.max(np.abs(diff))),rms_difference=float(np.sqrt(np.mean(diff**2))),relative_rms=float(np.linalg.norm(diff)/np.linalg.norm(a)),cosine=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))))
(base/'vision-comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
