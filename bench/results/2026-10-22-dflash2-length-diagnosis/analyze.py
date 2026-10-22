from pathlib import Path
import json,re,statistics,argparse
parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,default=Path.cwd()/'../Strata-data/dflash2-diagnosis-20261022');out=parser.parse_args().data_dir.resolve()
pat=re.compile(r'strata decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window = verify ([\d.]+).*?CPU ([\d.]+)\] \+ stage ([\d.]+)\) \+ commit/emit ([\d.]+) \+ draft ([\d.]+);')
fields=['windows','average_T','tokens_per_window','ms_per_window','verify_ms_per_window','cpu_ms_per_window','stage_ms_per_window','commit_ms_per_window','draft_ms_per_window'];result={}
for name,tag in [('control','mtp'),('mtp','mtp'),('dflash2','dflash2-il')]:
 p=out/name/(tag+'.json')
 if not p.exists():continue
 d=json.loads(p.read_text());lines=(out/name/(tag+'.log')).read_text();profiles=[dict(zip(fields,map(float,m.groups()))) for m in pat.finditer(lines)]
 assert len(profiles)==len(d['rows']),(name,len(profiles),len(d['rows']))
 for row,profile in zip(d['rows'],profiles):
  # The final verifier window can accept beyond max_new; the server emits only the requested prefix.
  tail=profile['windows']+row['accepted']-len(row['tokens'])
  assert 0<=tail<=7,(name,row['task'],profile,tail)
  row['truncated_tail']=tail
  row['verified_decode_rows']=profile['windows']+row['offered']
  row['profile']=profile;z=row['done'].split();row['cache_hit_fraction']=int(z[9])/int(z[10]);row['cache_misses']=int(z[10])-int(z[9])
 result[name]={}
 for task in dict.fromkeys(r['task'] for r in d['rows']):
  rows=[r for r in d['rows'] if r['task']==task and r['rep']>=0]
  if not rows:continue
  med=lambda fn:statistics.median(fn(r) for r in rows)
  result[name][task]={'tok_s':med(lambda r:r['tok_s']),'accepted':med(lambda r:r['accepted']),'offered':med(lambda r:r['offered']),'cache_hit_fraction':med(lambda r:r['cache_hit_fraction']),'cache_misses':med(lambda r:r['cache_misses']),'verified_decode_rows':med(lambda r:r['verified_decode_rows']),'truncated_tail':med(lambda r:r['truncated_tail'])}
  result[name][task].update({k:med(lambda r:r['profile'][k]) for k in fields})
  for k in ['verify','cpu','draft']:result[name][task][k+'_ms_per_token']=med(lambda r:r['profile'][k+'_ms_per_window']*r['profile']['windows']/len(r['tokens']))
(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
