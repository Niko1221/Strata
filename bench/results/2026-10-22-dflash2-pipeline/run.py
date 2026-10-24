"""Fixed launch settings: compare only the new CPU scheduling implementation."""
from pathlib import Path
import subprocess,json,shutil,argparse
parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,default=Path('../Strata-data/dflash2-pipeline-20261022'));opts=parser.parse_args()
root=Path.cwd();out=opts.data_dir.resolve();out.mkdir(exist_ok=True);base=(root/'../Strata-data/dflash2-diagnosis-20261022').resolve();exe=out/'strata'
if not exe.exists():
 assert 'STRATA_CPU_EXPERT_PIPELINE' in (root/'src/kernels/cpu/pool.cpp').read_text(), 'Apply the recorded experimental source patch before rebuilding and running this benchmark'
 shutil.copyfile(root/'build-dflash2/strata',exe);exe.chmod(0o755)
common=['--engine',str(exe),'--prompts',str(base/'prompts.json'),'--tasks','code_short,code_long,cache_long','--expert-cache','2900','--reserve','480','--prefill','512','--short-read','64','--context','8192','--resident','8192','--spec','8','--prompt-cache','0','--only-selected','--repeats','3','--reference',str(base/'control/mtp.json')]
for name,mode,options in [('dflash-before','dflash2-il',['--draft-quant','Q8_0','--block','7','--floor','.7']),('mtp-before','mtp',['--mtp-max-t','4','--floor','.5']),('dflash-pipeline','dflash2-il',['--draft-quant','Q8_0','--block','7','--floor','.7','--env','STRATA_CPU_EXPERT_PIPELINE=1']),('mtp-pipeline','mtp',['--mtp-max-t','4','--floor','.5','--env','STRATA_CPU_EXPERT_PIPELINE=1'])]:
 dest=out/name
 if (dest/(mode+'.json')).exists():
  assert json.loads((dest/(mode+'.json')).read_text()).get('validation','').startswith('common target-only');continue
 print('RUN',name,flush=True)
 with (out/(name+'.driver.log')).open('w') as f:r=subprocess.run([str(root/'.venv/bin/python'),str(root/'bench/results/2026-10-21-dflash2-long-goal/run.py'),mode,'--output',str(dest),*common,*options],stdout=f,stderr=subprocess.STDOUT)
 print('DONE',name,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
