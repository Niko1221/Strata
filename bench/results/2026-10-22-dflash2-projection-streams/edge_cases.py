"""Run after performance arms, never concurrently with them. No performance claims."""
from pathlib import Path
import json,os,sys
root=Path.cwd();sys.path.insert(0,str(root/'tools'))
from dflash_sampling import Engine,replace,validate_residency
out=(root/'../Strata-data/dflash2-projection-streams-20261022').resolve()
config=json.loads((out/'serial/dflash2-il-command.json').read_text())
args=config['command'][1:]
prompts=json.loads((root/'../Strata-data/dflash2-diagnosis-20261022/prompts.json').read_text())
results={}
cases=[(f'block{k}',f'temperature=0 seed=1234 dflash_block={k} spec_min_p=0') for k in [1,3,7]]
cases += [('reference','temperature=0 seed=1234 dflash_block=7 spec_min_p=0 dflash2_reference=1'),('after-reference','temperature=0 seed=1234 dflash_block=7 spec_min_p=0'),('sampled','temperature=0.7 seed=1234 top_k=40 top_p=0.9'),('greedy-again','temperature=0 seed=1234 dflash_block=7 spec_min_p=0')]
for arm,mode in [('serial-edges','0'),('parallel-edges','1')]:
 env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')};env.update(config['env']);env['STRATA_DFLASH2_PARALLEL_PROJECTIONS']=mode
 exe=Path(config['command'][0]);run_args=args
 if mode=='1':
  run_args=['profile','--trace=cuda','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--output',str(out/'projection-trace'),str(exe),*args]
  exe=Path('/usr/local/bin/nsys')
 engine=Engine(exe,run_args,root,env,out/(arm+'.log'))
 records=[]
 try:
  validate_residency(engine.info,2900)
  for name,sampling in cases:
   r=engine.request(prompts['code_short'],sampling,32)
   assert 'error' not in r and len(r['tokens'])==32,r
   records.append(dict(case=name,sampling=sampling,**r))
   print(arm,name,r['accepted'],r['offered'],flush=True)
 finally:engine.close()
 results[arm]=records
 (out/'edge-cases.json').write_text(json.dumps(results,indent=2))
for a,b in zip(results['serial-edges'],results['parallel-edges']):
 for key in ['case','tokens','accepted','offered']:assert a[key]==b[key],(key,a,b)
print('PASS: identical tokens and draft counters, blocks 1/3/7, reference toggle, sampling and reset',flush=True)
