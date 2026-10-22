from pathlib import Path
import subprocess
root=Path.cwd();out=(root/'../Strata-data/dflash2-projection-streams-20261022').resolve();out.mkdir(exist_ok=True)
base=(root/'../Strata-data/dflash2-diagnosis-20261022').resolve()
exe=out/'strata'
if not exe.exists():
 import shutil
 shutil.copyfile(root/'build-dflash2/strata',exe);exe.chmod(0o755)
common=['--engine',str(exe),'--prompts',str(base/'prompts.json'),'--tasks','code_short,code_long,cache_long','--expert-cache','2900','--reserve','480','--prefill','512','--short-read','64','--context','8192','--resident','8192','--spec','8','--prompt-cache','0','--only-selected','--reference',str(base/'control/mtp.json'),'--block','7','--floor','.7']
arms=[('serial','Q8_0',3,[]),('parallel','Q8_0',3,['STRATA_DFLASH2_PARALLEL_PROJECTIONS=1']),('parallel-eager','Q8_0',1,['STRATA_DFLASH2_PARALLEL_PROJECTIONS=1','STRATA_DFLASH_GRAPH=0']),('parallel-bf16','bf16',1,['STRATA_DFLASH2_PARALLEL_PROJECTIONS=1']),('parallel-q4','Q4_0',1,['STRATA_DFLASH2_PARALLEL_PROJECTIONS=1']),('parallel-combined','Q8_0',1,['STRATA_DFLASH2_PARALLEL_PROJECTIONS=1','STRATA_DFLASH2_FUSED_CONV=1','STRATA_DFLASH2_REGISTER_TOPK=1','STRATA_DFLASH2_LATTICE_SELECTOR=1'])]
for name,quant,repeats,env in arms:
 dest=out/name
 if (dest/'dflash2-il.json').exists():
  import json
  assert json.loads((dest/'dflash2-il.json').read_text()).get('validation','').startswith('common target-only')
  continue
 cmd=[str(root/'.venv/bin/python'),str(root/'bench/results/2026-10-21-dflash2-long-goal/run.py'),'dflash2-il','--output',str(dest),'--draft-quant',quant,'--repeats',str(repeats),*common]
 for item in env:cmd+=['--env',item]
 print('RUN',name,flush=True)
 with (out/(name+'.driver.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 print('DONE',name,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
