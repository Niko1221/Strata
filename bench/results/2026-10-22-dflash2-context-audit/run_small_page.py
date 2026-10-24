from pathlib import Path
import shutil,subprocess
root=Path.cwd();out=(root/'../Strata-data/dflash2-context-audit-20261022').resolve();base=(root/'../Strata-data/dflash2-diagnosis-20261022').resolve();exe=out/'strata-small-page'
assert not exe.exists();shutil.copyfile(root/'build-dflash2/strata',exe);exe.chmod(0o755)
common=['--engine',str(exe),'--prompts',str(base/'prompts.json'),'--tasks','code_short,code_long,cache_long','--expert-cache','2900','--reserve','480','--prefill','512','--short-read','64','--context','8192','--resident','8192','--spec','8','--prompt-cache','0','--only-selected','--reference',str(base/'control/mtp.json'),'--repeats','3']
for name,mode,extra in [('page-control','dflash2-il',['--draft-quant','Q8_0','--block','7','--floor','.7']),('page-on','dflash2-il',['--draft-quant','Q8_0','--block','7','--floor','.7','--env','STRATA_DFLASH2_SMALL_PAGE_ATTENTION=1']),('mtp','mtp',['--mtp-max-t','4','--floor','.5'])]:
 print('RUN',name,flush=True)
 with (out/(name+'.driver.log')).open('w') as f:r=subprocess.run([str(root/'.venv/bin/python'),str(root/'bench/results/2026-10-21-dflash2-long-goal/run.py'),mode,'--output',str(out/name),*common,*extra],stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0,(name,r.returncode)
 print('DONE',name,flush=True)
