"""Rebuild summaries from saved SSE/engine measurements; no external dependencies."""
from pathlib import Path
import argparse,json,re,hashlib,statistics,csv
def load(p):return json.loads(p.read_text())
def write(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2))
def analyze(root,out):
 rows=[];runs=[]
 for i,arm in enumerate(['A','B','B','A']):
  d=root/'abba'/f'{i}-{arm}';env=load(d/'environment.json');fp=load(d/'fingerprint.json')
  runs.append({'run':d.name,'arm':arm,'environment':env,'fingerprint':fp})
  for case in load(d/'schedule.json'):
   p=d/(case['id']+'.json');j=load(p);t=j.get('timings',{});u=j.get('usage',{});log=j.get('engine_log','')
   hit=re.search(r'cache hit rate: [\d.]+% \((\d+) hits / (\d+) lookups\)',log)
   resident=re.search(r'resident RAM: ([\d.]+) GiB of experts in RAM, (\d+) exchanged with the VRAM tier, (\d+) blob reads',log)
   rot=re.search(r'exchange rotation: (\d+) blocks, (\d+) host memcpy bytes avoided',log)
   io=re.search(r'the OS read ([\d.]+) MB from storage \((\d+) major faults\)',log)
   grade=load(d/'repair-grade.json') if case['phase']=='repair' else None
   if grade:
    try:grade['evaluation']=json.loads(grade['stdout'])
    except (ValueError,KeyError):grade['evaluation']=None
   ok=j.get('http_status')==200 and j.get('sse_done') and not j.get('error') and bool(j.get('finish_reason'))
   n=t.get('predicted_n');ms=t.get('predicted_ms');valid=ok and isinstance(n,int) and n>0 and isinstance(ms,(int,float)) and ms>0
   row={'run':d.name,'arm':arm,'id':case['id'],'phase':case['phase'],'case_id':case['id'].split('-',1)[1],
    'completed_stream':ok,'finish_reason':j.get('finish_reason'),'error':j.get('error'),'max_tokens':case['max_tokens'],
    'input_sha256':j['input_sha256'],'output_sha256':hashlib.sha256(j['content'].encode()).hexdigest(),
    'prompt_tokens':u.get('prompt_tokens'),'output_tokens':u.get('completion_tokens'),'engine_decode_tokens':n,
    'cached_tokens':u.get('prompt_tokens_details',{}).get('cached_tokens'),
    'prompt_ms':t.get('prompt_ms'),'prompt_tps':t.get('prompt_per_second'),
    'decode_ms':ms,'decode_tps':1000*n/ms if valid else None,'tpot_ms':ms/n if valid else None,
    'ttft_s':j.get('ttft'),'e2e_s':j.get('e2e'),'draft_offered':t.get('draft_n'),'draft_accepted':t.get('draft_n_accepted'),
    'cpu_routed_entries_from_cache_log':int(hit[2])-int(hit[1]) if hit else None,
    'cache_hits':int(hit[1]) if hit else None,'cache_lookups':int(hit[2]) if hit else None,
    'resident_gib':float(resident[1]) if resident else None,'exchanges_cumulative':int(resident[2]) if resident else None,
    'file_blob_reads_cumulative':int(resident[3]) if resident else None,
    'rotation_blocks_cumulative':int(rot[1]) if rot else 0 if arm=='A' else None,
    'avoided_copy_bytes_cumulative':int(rot[2]) if rot else 0 if arm=='A' else None,
    'os_read_mb':float(io[1]) if io else None,'major_faults':int(io[2]) if io else None,
    'quality':grade,'raw_file':str(p.relative_to(root))}
   rows.append(row)
 summary=[]
 for cid in ['decode-0','decode-2','repair_a']:
  ss=[r for r in rows if r['phase']!='warmup' and r['case_id']==cid];arms={}
  for arm in ['A','B']:
   rr=[r for r in ss if r['arm']==arm];metrics={}
   for k in ['decode_tps','tpot_ms','prompt_tps','ttft_s','e2e_s']:
    vals=[r[k] for r in rr if r[k] is not None];metrics[k]={'median':statistics.median(vals) if vals else None,'min':min(vals) if vals else None,'max':max(vals) if vals else None,'n':len(vals)}
   metrics['aggregate_decode_tps']=sum(r['engine_decode_tokens'] for r in rr)/sum(r['decode_ms'] for r in rr)*1000
   metrics['output_lengths']=[r['output_tokens'] for r in rr];metrics['output_hashes']=sorted(set(r['output_sha256'] for r in rr))
   metrics['request_ids']=[r['run']+'/'+r['id'] for r in rr];arms[arm]=metrics
  changes={k:(arms['B'][k]['median']/arms['A'][k]['median']-1)*100 for k in ['decode_tps','tpot_ms','ttft_s','e2e_s']}
  summary.append({'case_id':cid,'arms':arms,'median_change_pct':changes,'output_hashes_equal':len({r['output_sha256'] for r in ss})==1,'ranges_overlap':not(arms['A']['tpot_ms']['max']<arms['B']['tpot_ms']['min'] or arms['B']['tpot_ms']['max']<arms['A']['tpot_ms']['min'])})
 before=load(root/'hardware-before.json');after=load(root/'hardware-after.json')
 old={l for l in before['dmesg'].splitlines() if 'NVRM: Xid' in l};new=[l for l in after['dmesg'].splitlines() if 'NVRM: Xid' in l and l not in old]
 audit={'requests':len(rows),'all_streams_complete':all(r['completed_stream'] for r in rows),'no_payload_cache_reuse':all(r['cached_tokens']==0 for r in rows),'one_binary':len({r['fingerprint']['engine_sha256'] for r in runs})==1,'one_initial_profile':len({r['fingerprint']['initial_profile_sha256'] for r in runs})==1,'fixed_shape':all(r['environment']['fixed_shape']==runs[0]['environment']['fixed_shape'] for r in runs),'source_unchanged':load(root/'source-before.json')==load(root/'source-after.json'),'original_files_unchanged':load(root/'original-hashes.json')==load(root/'original-hashes-after.json'),'selectors_unchanged':before['selectors']==after['selectors'],'new_xid_lines':new,'restored':load(root/'restored.json'),'official_tests':load(root/'tests.json')}
 for r in rows:
  if r['phase']!='warmup' and r['arm']=='B':assert r['rotation_blocks_cumulative'] is not None and r['avoided_copy_bytes_cumulative']>0
 for i in [0,3]:assert 'exchange buffer rotation enabled:' not in (root/'abba'/f'{i}-A'/'engine.log').read_text()
 for i in [1,2]:assert 'exchange buffer rotation enabled:' in (root/'abba'/f'{i}-B'/'engine.log').read_text()
 out.mkdir(exist_ok=True,parents=True);write(out/'results.json',{'format':'strata-custom-rotation-abba','standard_protocol':False,'engine_commit':'c62d5e3a15b40ea51c96aae9e8d36928965f53ab','runs':runs,'requests':rows,'summaries':summary,'audit':audit})
 flat=[{k:v for k,v in r.items() if k!='quality'} for r in rows]
 with (out/'requests.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 print(json.dumps({'audit':{k:v for k,v in audit.items() if k!='restored'},'summary':summary},ensure_ascii=False,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();analyze(a.root,a.out)
