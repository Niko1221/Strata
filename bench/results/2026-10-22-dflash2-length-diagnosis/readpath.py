from pathlib import Path
import subprocess,sys
root=Path.cwd();out=(root/'../Strata-data/dflash2-diagnosis-20261022').resolve();exe=(root/'../Strata-data/dflash2-long-goal-20261022/strata-final-kernels').resolve()
args=['--engine',str(exe),'--prompts',str(out/'prompts.json'),'--tasks','code_short,code_long','--expert-cache','2900','--reserve','480','--prefill','512','--short-read','4096','--context','8192','--resident','8192','--spec','8','--prompt-cache','0','--only-selected','--repeats','1','--draft-quant','Q8_0','--block','7','--floor','.7']
with (out/'decode-readpath.driver.log').open('w') as f:
 r=subprocess.run([str(root/'.venv/bin/python'),str(root/'bench/results/2026-10-21-dflash2-long-goal/run.py'),'dflash2-il','--output',str(out/'decode-readpath'),*args],stdout=f,stderr=subprocess.STDOUT)
print('DONE decode-readpath',r.returncode,flush=True);sys.exit(r.returncode)
