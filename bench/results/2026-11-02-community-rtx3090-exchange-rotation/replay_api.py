#!/usr/bin/env python3
"""Single-attempt, serial chat benchmark. Secrets are never saved in request records."""
import argparse,os,json,time,urllib.request,datetime,hashlib
from pathlib import Path

def request(case,out,base_url,model,api_key=None,seed=20261015,reasoning='none',use_max_completion_tokens=False,before=None,after=None,log=None):
 payload={'model':model,'messages':case['messages'],'temperature':0,'top_p':0.95,'stream':True,'stream_options':{'include_usage':True},'max_completion_tokens' if use_max_completion_tokens else 'max_tokens':case['max_tokens']}
 if seed is not None:payload['seed']=seed
 if reasoning is not None:payload['reasoning_effort']=reasoning
 rec={'id':case['id'],'task_id':case.get('task_id',case['id']),'entry_point':case.get('entry_point'),'request':payload,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'time':time.time(),'content':'','reasoning':'','events':[],'sse_done':False,'error':None,'ttft':None,'before':before() if before else {}}
 offset=log.stat().st_size if log and log.exists() else 0
 headers={'Content-Type':'application/json'}
 if api_key:headers['Authorization']='Bearer '+api_key
 t=time.perf_counter()
 try:
  q=urllib.request.Request(base_url.rstrip('/')+'/chat/completions',data=json.dumps(payload).encode(),headers=headers)
  with urllib.request.urlopen(q,timeout=600) as r:
   rec['http_status']=r.status
   for line in r:
    if not line.startswith(b'data:'):continue
    raw=line[5:].strip()
    if raw==b'[DONE]':rec['sse_done']=True;break
    event=json.loads(raw);elapsed=time.perf_counter()-t;rec['events'].append({'seconds':elapsed,'data':event})
    if 'error' in event:rec['error']=event['error']
    for c in event.get('choices',[]):
     delta=c.get('delta') or {};s=delta.get('content') or '';rec['content']+=s;rec['reasoning']+=delta.get('reasoning_content') or delta.get('reasoning') or ''
     if s and rec['ttft'] is None:rec['ttft']=elapsed
     if c.get('finish_reason'):rec['finish_reason']=c['finish_reason']
    for k in ['usage','timings']:
     if k in event and event[k] is not None:rec[k]=event[k]
 except Exception as e:rec['error']=repr(e)
 rec['e2e']=time.perf_counter()-t;rec['after']=after() if after else {}
 if log:
  time.sleep(.15);rec['engine_log']=log.read_bytes()[offset:].decode(errors='replace') if log.exists() else ''
 rec['input_sha256']=hashlib.sha256(json.dumps(case['messages'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
 (out/(case['id']+'.json')).write_text(json.dumps(rec,ensure_ascii=False,indent=2));(out/(case['id']+'.txt')).write_text(rec['content'])
 print(json.dumps({k:rec.get(k) for k in ['id','finish_reason','ttft','e2e','error','usage','timings']},ensure_ascii=False),flush=True)
 return rec

def main():
 p=argparse.ArgumentParser();p.add_argument('--base-url',required=True);p.add_argument('--model',required=True);p.add_argument('--cases',default='cases.json');p.add_argument('--output',required=True);p.add_argument('--omit-seed',action='store_true');p.add_argument('--omit-reasoning',action='store_true');p.add_argument('--max-completion-tokens',action='store_true');a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);cases=json.loads(Path(a.cases).read_text())['cases'];seed=None if a.omit_seed else 20261015;reasoning=None if a.omit_reasoning else 'none'
 config={'base_url':a.base_url,'model':a.model,'seed':seed,'reasoning_effort':reasoning,'max_completion_tokens_flag':a.max_completion_tokens,'concurrency':1,'n':1,'task_count':len(cases),'cases_sha256':hashlib.sha256(Path(a.cases).read_bytes()).hexdigest()};(out/'run-config.json').write_text(json.dumps(config,indent=2))
 t=time.perf_counter()
 for case in cases:
  r=request(case,out,a.base_url,a.model,os.environ.get('BENCH_API_KEY'),seed,reasoning,a.max_completion_tokens)
  if r['error'] or not r['sse_done']:raise RuntimeError('Incomplete request; preserved record, no automatic retries: '+case['id'])
 (out/'completed.json').write_text(json.dumps({'task_count':len(cases),'suite_wall_s':time.perf_counter()-t},indent=2))
if __name__=='__main__':main()
