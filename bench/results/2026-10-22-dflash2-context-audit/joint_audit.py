from pathlib import Path
import json,os,sys,struct,hashlib
import numpy as np
root=Path.cwd();out=(root/'../Strata-data/dflash2-context-audit-20261022/joint').resolve();out.mkdir(exist_ok=False);sys.path.insert(0,str(root/'tools'));from dflash_sampling import Engine,replace
cfg=json.loads((out.parent/'full/dflash2-il-command.json').read_text());args=cfg['command'][1:-1];replace(args,'--dflash',str((root/'../Strata-data/dflash2-official/dflash2-bf16.gguf').resolve()))
env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')};env.update(cfg['env']);env.update(STRATA_DFLASH2_AUDIT_TAPS=str(out/'taps.bin'),STRATA_DFLASH2_AUDIT_DENSE='1')
prompts=json.loads((root/'../Strata-data/dflash2-diagnosis-20261022/prompts.json').read_text());controls=json.loads((root/'../Strata-data/dflash2-diagnosis-20261022/control/mtp.json').read_text());log=out/'capture.log';result={};exe=out.parent/'strata-joint-audit'
(out/'command.json').write_text(json.dumps(dict(command=[str(exe),*args,'--serve'],env={k:v for k,v in env.items() if k.startswith('STRATA_')},sha256=hashlib.sha256(exe.read_bytes()).hexdigest()),indent=2));e=Engine(exe,args,root,env,log)
try:
 for task in ['code_short','code_long','cache_long']:
  start=(out/'taps.bin').stat().st_size;logstart=log.stat().st_size
  r=e.request(prompts[task],'temperature=0 seed=1234 spec_min_p=.7 dflash_block=7',96);end=(out/'taps.bin').stat().st_size
  gold=next(x['tokens'] for x in controls['rows'] if x['task']==task and x['control']);assert r['tokens']==gold and r['offered']==0 and r['accepted']==0,(task,r)
  length=len(prompts[task])+96;taps=np.empty((length,12800),np.float32);seen=np.zeros(length,np.uint8);records=[]
  with (out/'taps.bin').open('rb') as f:
   f.seek(start)
   while f.tell()<end:
    magic,source,pos,T,nt,ne,flags=struct.unpack('<7I',f.read(28));assert magic==0x31504644 and nt==5 and ne==2560
    x=np.fromfile(f,dtype='<f4',count=nt*T*ne).reshape(nt,T,ne).transpose(1,0,2).reshape(T,-1)
    assert not seen[pos:pos+T].any();assert source!=1 or T==1
    taps[pos:pos+T]=x;seen[pos:pos+T]=1;records.append([source,pos,T,flags])
  last=int(np.flatnonzero(seen)[-1]);assert seen[:last+1].all() and last==length-2
  traces=[]
  for line in log.read_text()[logstart:].splitlines():
   if line.startswith('dflash2 audit: '):
    pos,k,anchor,*ids=map(int,line.split()[2:]);assert k==7 and anchor==gold[pos-len(prompts[task])];traces.append(dict(position=pos,ids=ids))
  assert len(traces)==95 and [t['position'] for t in traces]==list(range(len(prompts[task]),len(prompts[task])+95))
  np.save(out/(task+'-taps.npy'),taps[:last+1]);(out/(task+'.json')).write_text(json.dumps(dict(prompt=prompts[task],output=gold,native=r,records=records)))
  result[task]=dict(response=r,traces=traces);(out/'native-dense.json').write_text(json.dumps(result));print(task,'PASS same-run capture',last+1,'rows',len(traces),'proposals',flush=True)
finally:e.close()
