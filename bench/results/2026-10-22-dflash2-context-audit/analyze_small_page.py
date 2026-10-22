from pathlib import Path
import json,statistics,re
out=Path(__file__).resolve().parent;data={arm:json.loads((out/arm/('mtp.json' if arm=='mtp' else 'dflash2-il.json')).read_text()) for arm in ['page-control','page-on','mtp']};result={}
for task in ['code_short','code_long','cache_long']:
 row={}
 for arm,d in data.items():
  assert d['validation'].startswith('common target-only')
  rs=[r for r in d['rows'] if r['task']==task and r['rep']>=0 and not r['control']];assert len(rs)==3
  row[arm]=dict(tok_s=[r['tok_s'] for r in rs],median_tok_s=statistics.median(r['tok_s'] for r in rs),accepted=[r['accepted'] for r in rs],offered=[r['offered'] for r in rs])
 a=[r for r in data['page-control']['rows'] if r['task']==task and r['rep']>=0];b=[r for r in data['page-on']['rows'] if r['task']==task and r['rep']>=0]
 for x,y in zip(a,b):
  for k in ['tokens','accepted','offered']:assert x[k]==y[k],(task,k)
 row['change_pct']=100*(row['page-on']['median_tok_s']/row['page-control']['median_tok_s']-1)
 row['dflash_vs_mtp_pct']=100*(row['page-on']['median_tok_s']/row['mtp']['median_tok_s']-1)
 result[task]=row;print(task,row,flush=True)
(out/'small-page-summary.json').write_text(json.dumps(result,indent=2))
