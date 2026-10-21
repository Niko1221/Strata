"""Compare release and candidate output bytes on the synthetic Q8 fixture.
Use make-fixture.py first. This test requires a CUDA GPU.
"""
from pathlib import Path
import argparse, fcntl, hashlib, json, subprocess, os, sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools/qwen35'))
from q36_harness import Eng
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--stock',type=Path,required=True)
ap.add_argument('--candidate',type=Path,required=True)
ap.add_argument('--fixture',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args()
out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
fixture=a.fixture.resolve()
results=[]
with open('/tmp/strata-sage-benchmark.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for mode in ['gpu','hybrid']:
  for prompt_n,chunk,mtp,fixed in [(n,32,m,True) for n in [3,32,128] for m in [0,2]]+[(512,256,0,True),(128,32,0,False)]:
   records=[]
   for kind,exe in [('stock',a.stock.resolve()),('patched',a.candidate.resolve())]:
    label=f'{mode}-{prompt_n}-mtp{mtp}-fixed{int(fixed)}-{kind}'
    args=['--native',str(fixture),'--max-context','1024' if prompt_n==512 else '512','--chunk',str(chunk),'--mtp',str(mtp),'--pool-workers','4','--no-warmup','--tokens',','.join(str(i%251+1) for i in range(prompt_n)),'--n','64','--dump-full-logits',str(out/(label+'.bin'))]
    if mode=='hybrid':args+=['--expert-slots','1']
    env=dict(os.environ);env.pop('STRATA_Q36_FIXED_SLOTS',None)
    if fixed:env['STRATA_Q36_FIXED_SLOTS']='1'
    with (out/(label+'.log')).open('w') as f:subprocess.run([str(exe),*args],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
    generated=next(x for x in (out/(label+'.log')).read_text().splitlines() if x.startswith('gen:'))
    data=(out/(label+'.bin')).read_bytes();assert len(data)==64*256*4
    records.append((generated,hashlib.sha256(data).hexdigest()))
   assert records[0]==records[1],(mode,prompt_n,mtp,fixed,records)
   results.append({'mode':mode,'prompt_n':prompt_n,'chunk':chunk,'mtp':mtp,'fixed_slots':fixed,'generated_n':64,'identical':True,'logits_sha256':records[0][1]})
   print(mode,prompt_n,chunk,mtp,fixed,'ids and complete logit bytes identical',flush=True)
 (out/'default-bytes.json').write_text(json.dumps(results,indent=2)+'\n')
 for mode in ['gpu','hybrid']:
  rows={}
  for mtp in [0,2]:
   for kind,exe in [('stock',a.stock.resolve()),('patched',a.candidate.resolve())]:
    label=f'protocol-{mode}-mtp{mtp}-{kind}'
    args=['--native',str(fixture),'--max-context','512','--chunk','32','--mtp',str(mtp),'--pool-workers','4']
    if mode=='hybrid':args+=['--expert-slots','1']
    eng=Eng(str(exe),args,env={'STRATA_Q36_FIXED_SLOTS':'1'},log=str(out/(label+'.log')))
    try:rows[f'{kind}-{mtp}']=[eng.solo(list(range(1,n+1)),64)[0] for n in [3,32,128]]
    finally:eng.close()
   assert rows[f'stock-{mtp}']==rows[f'patched-{mtp}'],(mode,mtp)
   if mtp:assert rows[f'patched-{mtp}']==rows['patched-0'],mode
  print(mode,'MTP protocol outputs identical',flush=True)
 print('PASS',flush=True)
