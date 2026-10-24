from pathlib import Path
import subprocess,sys
root=Path.cwd();out=(root/'../Strata-data/dflash2-diagnosis-20261022').resolve();exe=(root/'../Strata-data/dflash2-long-goal-20261022/strata-final-kernels').resolve()
common=['--engine',str(exe),'--prompts',str(out/'prompts.json'),'--tasks','code_short,code_long,cache_short,cache_long','--expert-cache','2900','--reserve','480','--prefill','512','--short-read','64','--context','8192','--resident','8192','--spec','8','--prompt-cache','0','--only-selected','--repeats','3']
# All new DFlash2 execution optimizations and uneven verification are unset.
for name,mode,extra in [('control','mtp',['--target-only']),('mtp','mtp',['--mtp-max-t','4','--floor','.5','--reference',str(out/'control/mtp.json')]),('dflash2','dflash2-il',['--draft-quant','Q8_0','--block','7','--floor','.7','--reference',str(out/'control/mtp.json')])]:
 if name=='control' and (out/'control/mtp.json').exists():continue
 print('RUN',name,flush=True)
 with (out/(name+'.driver.log')).open('w') as f:
  r=subprocess.run([str(root/'.venv/bin/python'),str(root/'bench/results/2026-10-21-dflash2-long-goal/run.py'),mode,'--output',str(out/name),*common,*extra],stdout=f,stderr=subprocess.STDOUT)
 print('DONE',name,r.returncode,flush=True)
 if r.returncode:sys.exit(r.returncode)
