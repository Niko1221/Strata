from pathlib import Path
import subprocess,shutil,json
root=Path.cwd();out=(root/'../Strata-data/dflash2-context-audit-20261022').resolve();base=(root/'../Strata-data/dflash2-diagnosis-20261022').resolve();exe=out/'strata-prompt-tail'
assert not exe.exists();shutil.copyfile(root/'build-dflash2/strata',exe);exe.chmod(0o755)
common=['--engine',str(exe),'--prompts',str(base/'prompts.json'),'--tasks','code_short,code_long,cache_long','--expert-cache','2900','--reserve','480','--prefill','512','--short-read','64','--context','8192','--resident','8192','--spec','8','--prompt-cache','0','--only-selected','--reference',str(base/'control/mtp.json'),'--block','7','--floor','.7','--draft-quant','Q8_0','--repeats','3']
for name,env in [('full',[]),('tail',['STRATA_DFLASH2_PROMPT_TAIL=1'])]:
 cmd=[str(root/'.venv/bin/python'),str(root/'bench/results/2026-10-21-dflash2-long-goal/run.py'),'dflash2-il','--output',str(out/name),*common]
 for value in env:cmd+=['--env',value]
 print('RUN',name,flush=True)
 with (out/(name+'.driver.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0,(name,r.returncode)
 print('DONE',name,flush=True)
