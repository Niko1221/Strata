"""Offline verification of the sanitized report, raw records and summaries."""
from pathlib import Path
import json,hashlib,statistics,math,sys
R=Path(sys.argv[1] if len(sys.argv)>1 else '.').resolve()
load=lambda p:json.loads(p.read_text())
digest=lambda b:hashlib.sha256(b).hexdigest()
for line in (R/'SHA256SUMS').read_text().splitlines():
 expected,name=line.split('  ',1);p=(R/name).resolve();assert p.is_relative_to(R)
 assert digest(p.read_bytes())==expected,name
j=load(R/'results.json');rows=j['requests'];assert len(rows)==28
for r in rows:
 raw=load(R/r['raw_file']);t=raw['timings'];u=raw['usage']
 assert raw['http_status']==200 and raw['sse_done'] and raw['error'] is None
 assert digest(raw['content'].encode())==r['output_sha256']
 assert digest(json.dumps(raw['request']['messages'],ensure_ascii=False,sort_keys=True).encode())==r['input_sha256']
 assert u['completion_tokens']==r['output_tokens'] and u['prompt_tokens']==r['prompt_tokens']
 assert u['prompt_tokens_details']['cached_tokens']==0
 assert t['predicted_n']==r['engine_decode_tokens'] and t['predicted_ms']==r['decode_ms']
 assert math.isclose(1000*t['predicted_n']/t['predicted_ms'],r['decode_tps'],rel_tol=1e-12)
 assert math.isclose(t['predicted_ms']/t['predicted_n'],r['tpot_ms'],rel_tol=1e-12)
 if r['quality']:
  q=r['quality']['evaluation']
  if q:
   assert q['passed']==sum(x['passed'] for x in q['tests']) and q['total']==len(q['tests'])
for s in j['summaries']:
 for arm in ['A','B']:
  rr=[r for r in rows if r['phase']!='warmup' and r['case_id']==s['case_id'] and r['arm']==arm]
  for k in ['decode_tps','tpot_ms','prompt_tps','ttft_s','e2e_s']:
   vals=[r[k] for r in rr if r[k] is not None];x=s['arms'][arm][k]
   assert x=={'median':statistics.median(vals),'min':min(vals),'max':max(vals),'n':len(vals)}
  assert s['arms'][arm]['output_lengths']==[r['output_tokens'] for r in rr]
  assert s['arms'][arm]['output_hashes']==sorted(set(r['output_sha256'] for r in rr))
 for k in ['decode_tps','tpot_ms','ttft_s','e2e_s']:
  assert math.isclose(s['median_change_pct'][k],(s['arms']['B'][k]['median']/s['arms']['A'][k]['median']-1)*100,abs_tol=1e-10)
fps=[load(R/'abba'/f'{i}-{a}'/'fingerprint.json') for i,a in enumerate(['A','B','B','A'])]
assert len({f['engine_sha256'] for f in fps})==1 and len({f['initial_profile_sha256'] for f in fps})==1
envs=[{k:v for k,v in f['env'].items() if k!='STRATA_EXCHANGE_ROTATE'} for f in fps];assert all(e==envs[0] for e in envs)
assert [f['env']['STRATA_EXCHANGE_ROTATE'] for f in fps]==['0','1','1','0']
print('Verified: artifact hashes; 28 raw request records; engine-derived metrics; per-case summaries; repair score counts; same binary/profile and toggle-only environment.')
