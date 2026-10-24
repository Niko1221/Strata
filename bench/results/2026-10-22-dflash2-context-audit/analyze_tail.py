from pathlib import Path
import json,statistics
out=Path(__file__).resolve().parent;arms={name:json.loads((out/name/'dflash2-il.json').read_text()) for name in ['full','tail']};report={}
for task in ['code_short','code_long','cache_long']:
 data={}
 for arm,d in arms.items():
  rows=[r for r in d['rows'] if r['task']==task and r['rep']>=0 and not r['control']]
  assert len(rows)==3
  data[arm]=dict(decode_tok_s=[r['tok_s'] for r in rows],prompt_ms=[float(r['done'].split()[3]) for r in rows],accepted=[r['accepted'] for r in rows],offered=[r['offered'] for r in rows])
  data[arm]['median_decode_tok_s']=statistics.median(data[arm]['decode_tok_s']);data[arm]['median_prompt_ms']=statistics.median(data[arm]['prompt_ms'])
 a=[r for r in arms['full']['rows'] if r['task']==task and r['rep']>=0];b=[r for r in arms['tail']['rows'] if r['task']==task and r['rep']>=0]
 for x,y in zip(a,b):
  for key in ['tokens','accepted','offered']:assert x[key]==y[key],(task,key)
 data['prompt_change_pct']=100*(data['tail']['median_prompt_ms']/data['full']['median_prompt_ms']-1)
 data['decode_change_pct']=100*(data['tail']['median_decode_tok_s']/data['full']['median_decode_tok_s']-1)
 report[task]=data;print(task,data,flush=True)
(out/'tail-summary.json').write_text(json.dumps(report,indent=2))
