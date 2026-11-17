"""Review these two saved pilot runs. No inference, downloads or service changes."""
from pathlib import Path
import hashlib,json,math,statistics,subprocess

HERE=Path(__file__).resolve().parent
REPORTS=[HERE.parent/'2026-11-17-standard-rtx5070ti-5600g-2c7257f4',HERE]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):return math.isclose(a,b,rel_tol=0,abs_tol=0.000002)
def strict(text):
 def pairs(items):
  out={}
  for k,v in items:
   if k in out:raise ValueError('duplicate JSON key')
   out[k]=v
  return out
 def constant(s):raise ValueError('non-JSON constant')
 return json.loads(text,object_pairs_hook=pairs,parse_constant=constant)
def equal(a,b):
 if type(a)!=type(b):return False
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
def score(spec,row,status):
 if status!='completed':return False
 e=spec['expected']
 if e['kind']=='text':return row['final_answer'].strip()==e['value']
 if e['kind']=='tool_call':
  calls=row['tool_calls']
  return len(calls)==1 and calls[0]['name']==e['name'] and equal(strict(calls[0]['arguments']),e['arguments'])
 return equal(strict(row['final_answer']),e['value'])
summary=[]
for root in REPORTS:
 manifest=read(root/'MANIFEST.json')
 for path,entry in manifest.items():
  p=root/path;assert p.is_file() and p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],path
 d=read(root/'results.json');raw={r['request_id']:r for r in map(json.loads,(root/'outputs.jsonl').read_text(encoding='utf-8').splitlines())}
 assert len(raw)==len(d['requests'])==22 and d['run']['status']=='complete'
 for artifact in d['artifacts']:assert sha(root/artifact['path'])==artifact['sha256'],artifact
 corpus={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (root/'provenance/corpus-manifest.sha256').read_text().splitlines()}
 assert sha(root/'provenance/corpus-manifest.sha256')==d['protocol']['corpus_sha256']
 verdicts=[]
 for request in d['requests']:
  ident=request['id'];row=raw[ident]
  file='requests/'+ident+'.json'
  payload=subprocess.check_output(['git','show',d['protocol']['git_commit']+':bench/prompts/standard-v1/'+file],cwd=HERE)
  assert hashlib.sha256(payload).hexdigest()==corpus[file],file
  spec=json.loads(payload)
  assert row['messages']==spec['messages'] and (row.get('tools') or [])==(spec.get('tools') or [])
  assert row['request']['max_tokens']==spec['max_output_tokens'] and row['request']['seed']==spec['seed']
  assert row['usage']['prompt_tokens']==request['tokens']['input_tokens']
  assert row['usage']['completion_tokens']==request['tokens']['output_tokens']
  assert row['usage']['prompt_tokens_details']['cached_tokens']==request['tokens']['cached_tokens']
  if request['phase']=='performance':assert request['tokens']['cached_tokens']==0
  if request['phase']=='sanity':
   verdict=score(spec,row,request['status']);assert verdict==request['evaluation']['passed'];verdicts.append(verdict)
 for s in d['summaries']:
  values=[]
  for ident in s['included_request_ids']:
   r=raw[ident];t=r['timings'];metric=s['metric']
   if metric=='prompt_tps':value=t['prompt_n']/(t['prompt_ms']/1000)
   elif metric=='decode_tps':value=t['predicted_n']/(t['predicted_ms']/1000)
   elif metric=='ttft_s':value=r['first_delta']['offset_s']-r['dispatch_offset_s']
   elif metric=='first_answer_s':value=r['first_answer_offset_s']-r['dispatch_offset_s']
   else:value=r['end_offset_s']-r['dispatch_offset_s']
   values.append(value)
  assert len(values)==s['n']
  if values:assert close(statistics.median(values),s['median']) and close(min(values),s['min']) and close(max(values),s['max']),(s,values)
  else:assert s['median'] is None and s['min'] is None and s['max'] is None
 assert sum(verdicts)==d['quality']['passed']==12 and len(verdicts)==12
 note=read(root/'provenance/operator.json');assert note['api_inference_request_delta']==22
 summary.append({'run_id':d['run']['id'],'status':d['run']['status'],'quality_rescored':'12/12','artifacts':'hashes match','summaries_recomputed':len(d['summaries']),'fixture_exact_requests':22})
a,b=[read(r/'results.json') for r in REPORTS]
assert {k:v for k,v in a['configuration'].items() if k!='track_notes'}=={k:v for k,v in b['configuration'].items() if k!='track_notes'}
assert sha(REPORTS[0]/'config.json')==sha(REPORTS[1]/'config.json')
print(json.dumps({'verdict':'pass','same_effective_configuration':True,'runs':summary},indent=2))
