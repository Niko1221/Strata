from pathlib import Path
import fcntl,json,os,subprocess,numpy as np,time
import argparse
ROOT=Path(__file__).resolve().parents[3]
ap=argparse.ArgumentParser(description='Teacher-force 128 outputs after a 16K Sage prompt; compare full logits.')
ap.add_argument('--model',type=Path,required=True)
ap.add_argument('--prompts',type=Path,required=True)
ap.add_argument('--llama-dump',type=Path,required=True)
ap.add_argument('--strata-bin',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args()
OUT=a.out.resolve();OUT.mkdir(parents=True,exist_ok=True)
MODEL=a.model.resolve()
prompts={p['name']:p['ids'] for p in json.loads(a.prompts.read_text())}
prompts['16384']=prompts['32768'][:16320]+prompts['32768'][-64:]
results={}
def compare(x,y):
 x=np.fromfile(x,dtype='<f4').reshape(-1,248320).astype(np.float64);y=np.fromfile(y,dtype='<f4').reshape(-1,248320).astype(np.float64)
 assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
 def lsm(z):
  z=z-z.max(axis=1,keepdims=True);return z-np.log(np.exp(z).sum(axis=1,keepdims=True))
 lx,ly=lsm(x),lsm(y);kl=(np.exp(lx)*(lx-ly)).sum(axis=1)
 return {'rows':len(x),'mean_kl':float(kl.mean()),'max_kl':float(kl.max()),'top1_matches':int((x.argmax(1)==y.argmax(1)).sum())}
with open('/tmp/strata-sage-benchmark.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for name in ['16384']:
  ids=','.join(map(str,prompts[name]));assert len(ids)<128000
  force=''
  kinds=['llama','strata','cpu'] if name=='short' else ['llama','strata']
  for kind in kinds:
   dump=OUT/f'quality-final-{name}-{kind}.bin'
   if kind in ['llama','cpu']:
    cmd=[str(a.llama_dump.resolve()),'--model',str(MODEL),'--tokens',ids,'--n','128','--ngl','0' if kind=='cpu' else '99','--dump-full-logits',str(dump)]
   else:
    cmd=[str(a.strata_bin.resolve()),'--native',str(MODEL),'--f16-weights','--kv','f16','--max-context','262144','--prefill','512','--pool-workers','15','--mtp','0','--no-warmup','--tokens',ids,'--force',force,'--n','128','--dump-full-logits',str(dump)]
   t=time.time()
   with (OUT/f'quality-final-{name}-{kind}.log').open('w') as f:
    subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,STRATA_DUMP_CTX='262144'),check=True)
   print(name,kind,round(time.time()-t,2),'s',flush=True)
   if kind=='llama':
    line=next(x for x in (OUT/f'quality-final-{name}-{kind}.log').read_text().splitlines() if x.startswith('gen:'))
    generated=list(map(int,line.split()[1:]));assert len(generated)==128
    force=','.join(map(str,generated[:-1]))
  results[name]={'llama_vs_strata':compare(OUT/f'quality-final-{name}-llama.bin',OUT/f'quality-final-{name}-strata.bin')}
  if name=='short':results[name]['llama_gpu_vs_cpu_floor']=compare(OUT/f'quality-final-{name}-llama.bin',OUT/f'quality-final-{name}-cpu.bin')
  (OUT/'quality-final-summary.json').write_text(json.dumps(results,indent=2)+'\n');print(name,results[name],flush=True)
  assert results[name]['llama_vs_strata']['mean_kl']<.02
