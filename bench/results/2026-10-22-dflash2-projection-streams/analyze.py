from pathlib import Path
import json,re,statistics,shutil
root=Path.cwd();src=(root/'../Strata-data/dflash2-projection-streams-20261022').resolve();dest=Path(__file__).resolve().parent
summary={};baseline=None
for name in ['serial','parallel','parallel-eager','parallel-bf16','parallel-q4','parallel-combined']:
 data=json.loads((src/name/'dflash2-il.json').read_text())
 assert data['validation'].startswith('common target-only'),(name,data.get('validation'))
 if baseline is None:baseline=data
 for task in ['code_short','code_long','cache_long']:
  rows=[r for r in data['rows'] if r['task']==task and r['rep']>=0]
  ref=next(r for r in baseline['rows'] if r['task']==task and r['rep']>=0)
  assert all(r['tokens']==ref['tokens'] for r in rows)
  if name in ['parallel','parallel-eager','parallel-combined']:
   assert all((r['accepted'],r['offered'])==(ref['accepted'],ref['offered']) for r in rows)
  values=[r['tok_s'] for r in rows]
  summary.setdefault(name,{})[task]={'median_tok_s':statistics.median(values),'individual_tok_s':values,'accepted':[r['accepted'] for r in rows],'offered':[r['offered'] for r in rows]}
 for suffix in ['.json','-command.json','.log']:
  text=(src/name/('dflash2-il'+suffix)).read_text().replace(str(root.parent),'<WORKSPACE>')
  (dest/(name+suffix)).write_text(text)
for task,row in summary['parallel'].items():row['change_pct']=100*(row['median_tok_s']/summary['serial'][task]['median_tok_s']-1)
(dest/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
