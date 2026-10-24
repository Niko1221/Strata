from pathlib import Path
import json,re,statistics,argparse
parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,default=Path('../Strata-data/dflash2-pipeline-20261022'));opts=parser.parse_args()
root=Path.cwd();out=opts.data_dir.resolve();summary={};data={}
pat=re.compile(r'strata decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window = verify ([\d.]+).*?CPU ([\d.]+)\] \+ stage ([\d.]+)\) \+ commit/emit ([\d.]+) \+ draft ([\d.]+);')
fields=['windows','average_T','tokens_per_window','ms_per_window','verify_ms_per_window','cpu_ms_per_window','stage_ms_per_window','commit_ms_per_window','draft_ms_per_window']
for name,tag in [('dflash-before','dflash2-il'),('mtp-before','mtp'),('dflash-pipeline','dflash2-il'),('mtp-pipeline','mtp')]:
 d=json.loads((out/name/(tag+'.json')).read_text());assert d['validation'].startswith('common target-only')
 profiles=[dict(zip(fields,map(float,m.groups()))) for m in pat.finditer((out/name/(tag+'.log')).read_text())]
 assert len(profiles)==len(d['rows'])
 for r,p in zip(d['rows'],profiles):r['profile']=p
 data[name]=d;summary[name]={}
 for task in ['code_short','code_long','cache_long']:
  rows=[r for r in d['rows'] if r['task']==task and r['rep']>=0];med=lambda fn:statistics.median(fn(r) for r in rows)
  summary[name][task]={'median_tok_s':med(lambda r:r['tok_s']),'individual_tok_s':[r['tok_s'] for r in rows],'accepted':med(lambda r:r['accepted']),'offered':med(lambda r:r['offered']),'windows':med(lambda r:r['profile']['windows']),'cpu_ms_per_token':med(lambda r:r['profile']['cpu_ms_per_window']*r['profile']['windows']/len(r['tokens']))}
for family in ['dflash','mtp']:
 before=data[family+'-before'];after=data[family+'-pipeline'];assert before['command']==after['command'] and before['sha256']==after['sha256']
 env=dict(after['env']);assert env.pop('STRATA_CPU_EXPERT_PIPELINE')=='1';assert env==before['env']
 for key in ['context','kv','kv_resident','expert_slots','expert_slots_primary','pool_workers','pcie_frac','spec']:
  assert before['info'][key]==after['info'][key],key
 for task in ['code_short','code_long','cache_long']:
  a=[r for r in before['rows'] if r['task']==task and r['rep']>=0];b=[r for r in after['rows'] if r['task']==task and r['rep']>=0]
  for x,y in zip(a,b):
   for k in ['tokens','accepted','offered']:assert x[k]==y[k],(family,task,k)
  old=summary[family+'-before'][task];new=summary[family+'-pipeline'][task]
  new['change_pct']=100*(new['median_tok_s']/old['median_tok_s']-1)
  new['cpu_change_pct']=100*(new['cpu_ms_per_token']/old['cpu_ms_per_token']-1)
summary['dflash_vs_mtp_pct']={mode:{task:100*(summary['dflash-'+mode][task]['median_tok_s']/summary['mtp-'+mode][task]['median_tok_s']-1) for task in ['code_short','code_long','cache_long']} for mode in ['before','pipeline']}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
