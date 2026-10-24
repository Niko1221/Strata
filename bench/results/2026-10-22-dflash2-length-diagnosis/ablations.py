from pathlib import Path
import subprocess,sys
root=Path.cwd();out=(root/'../Strata-data/dflash2-diagnosis-20261022').resolve();old=(root/'../Strata-data/dflash2-long-goal-20261022').resolve()
common=['--prompts',str(out/'prompts.json'),'--tasks','code_short,code_long','--expert-cache','2900','--reserve','480','--prefill','512','--short-read','64','--context','8192','--resident','8192','--spec','8','--prompt-cache','0','--only-selected','--repeats','1','--block','7','--floor','.7','--reference',str(out/'control/mtp.json')]
for name,exe,extra in [('full-span',old/'strata-context-span',['--draft-quant','Q8_0','--env','STRATA_DFLASH2_CONTEXT_SPAN=0']),('bf16',old/'strata-final-kernels',['--draft-quant','bf16'])]:
 print('RUN',name,flush=True)
 with (out/(name+'.driver.log')).open('w') as f:
  r=subprocess.run([str(root/'.venv/bin/python'),str(root/'bench/results/2026-10-21-dflash2-long-goal/run.py'),'dflash2-il','--output',str(out/name),'--engine',str(exe),*common,*extra],stdout=f,stderr=subprocess.STDOUT)
 print('DONE',name,r.returncode,flush=True)
 if r.returncode:sys.exit(r.returncode)
