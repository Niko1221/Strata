from pathlib import Path
import subprocess
root=Path.cwd()
out=root/'../Strata-data/dflash2-parallel-20261022'
out.mkdir(exist_ok=True)
base=root/'../Strata-data/dflash2-diagnosis-20261022'
common=['--engine',str((root/'../Strata-data/dflash2-long-goal-20261022/strata-final-kernels').resolve()),'--prompts',str(base/'prompts.json'),'--tasks','code_long,cache_long','--expert-cache','2900','--reserve','480','--prefill','512','--short-read','64','--context','8192','--resident','8192','--spec','8','--prompt-cache','0','--only-selected','--repeats','3','--reference',str(base/'control/mtp.json')]
for name,env in [('baseline',[]),('side1',['STRATA_SIDE_MIXER=1']),('side2',['STRATA_SIDE_MIXER=2']),('branch',['STRATA_DF_BRANCH=1'])]:
 dest=out/name
 if (dest/'dflash2-il.json').exists():continue
 cmd=[str(root/'.venv/bin/python'),str(root/'bench/results/2026-10-21-dflash2-long-goal/run.py'),'dflash2-il','--output',str(dest),'--draft-quant','Q8_0','--block','7','--floor','.7',*common]
 for item in env:cmd+=['--env',item]
 print('RUN',name,flush=True)
 with (out/(name+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 print('DONE',name,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
