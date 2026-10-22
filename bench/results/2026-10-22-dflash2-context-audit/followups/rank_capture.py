from pathlib import Path
import sys,json,os,struct
import numpy as np
root=Path.cwd();out=(root/'../Strata-data/dflash2-context-audit-20261022/rank').resolve();out.mkdir(exist_ok=False);sys.path.insert(0,str(root/'tools'));from dflash_sampling import Engine
cfg=json.loads((out.parent/'page-control/dflash2-il-command.json').read_text());args=cfg['command'][1:-1]
env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')};env.update(cfg['env']);env['STRATA_DFLASH2_RANK_TRACE']=str(out/'features.bin')
prompts=json.loads((root/'../Strata-data/dflash2-diagnosis-20261022/prompts.json').read_text());reference=json.loads((root/'../Strata-data/dflash2-acceptance-20261022/dense-complete.json').read_text());result={};e=Engine(out.parent/'strata-rank-trace',args,root,env,out/'capture.log')
try:
 for task in ['code_short','code_long','cache_long']:
  start=(out/'features.bin').stat().st_size if (out/'features.bin').exists() else 0
  r=e.request(prompts[task],'temperature=0 seed=1234 spec_min_p=.7 dflash_block=7',96);end=(out/'features.bin').stat().st_size
  assert r['tokens']==reference[task]['dense']['response']['tokens'] and r['offered']==0 and r['accepted']==0
  records=[]
  with (out/'features.bin').open('rb') as f:
   f.seek(start)
   while f.tell()<end:
    magic,lo,hi,block,rank,anchor=struct.unpack('<6I',f.read(24));assert magic==0x31514644 and block==7 and rank==256
    pos=lo+(hi<<32);ids=np.fromfile(f,'<i4',block).tolist();q=np.fromfile(f,'<f4',block*rank).reshape(block,rank);assert np.isfinite(q).all()
    expected=reference[task]['dense']['traces'][len(records)];assert pos==expected['position'] and ids==expected['ids'];assert anchor==r['tokens'][pos-len(prompts[task])]
    records.append(dict(position=pos,ids=ids))
  assert len(records)==95
  result[task]=dict(start=start,end=end,prompt_tokens=len(prompts[task]),response=r,records=records);(out/'metadata.json').write_text(json.dumps(result));print(task,'PASS native rank trace',len(records),flush=True)
finally:e.close()
