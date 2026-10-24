from pathlib import Path
import shutil,subprocess
root=Path.cwd();out=(root/'../Strata-data/dflash2-context-audit-20261022').resolve();base=(root/'../Strata-data/dflash2-diagnosis-20261022').resolve();exe=out/'strata-stream-mean'
assert not exe.exists();shutil.copyfile(root/'build-dflash2/strata',exe);exe.chmod(0o755)
common=['--engine',str(exe),'--prompts',str(base/'prompts.json'),'--tasks','code_short,code_long,cache_long','--expert-cache','2900','--reserve','480','--prefill','512','--short-read','64','--context','8192','--resident','8192','--spec','8','--prompt-cache','0','--only-selected','--reference',str(base/'control/mtp.json'),'--repeats','1','--draft-quant','Q8_0','--block','7','--floor','.7','--env','STRATA_DFLASH2_BF16_STREAM_MEAN=1']
with (out/'mean-screen.driver.log').open('w') as f:
 r=subprocess.run([str(root/'.venv/bin/python'),str(root/'bench/results/2026-10-21-dflash2-long-goal/run.py'),'dflash2-il','--output',str(out/'mean-screen'),*common],stdout=f,stderr=subprocess.STDOUT)
assert r.returncode==0,r.returncode
