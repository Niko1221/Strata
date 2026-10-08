"""Opt-in real-engine Responses + disk-cache integration probe. Run in an isolated test directory."""
import argparse,json,os,pathlib,signal,subprocess,time,urllib.request,urllib.error,traceback
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workdir',type=pathlib.Path,required=True,help='Disposable test directory containing server.json, runtime.json and source/')
R=parser.parse_args().workdir.resolve();CFG=R/'server.json';runtime=json.loads((R/'runtime.json').read_text());env=os.environ.copy();env.update(runtime['env']);S=R/'source';URL='http://127.0.0.1:18210';proc=None;log=None
# Refuse production paths before starting a server or corrupting test artifacts.
cfg=json.loads(CFG.read_text())
for key in ('responses_store_path', 'slot_save_path', 'log'):
    value=pathlib.Path(cfg[key]).resolve()
    if not value.is_relative_to(R):
        raise ValueError(f'{key} must be inside the disposable workdir')
args=cfg['args'];spill_path=pathlib.Path(args[args.index('--conversation-cache-spill-dir')+1]).resolve()
if spill_path != R/'spill':
    raise ValueError('spill directory must be workdir/spill')
if cfg.get('host') != '127.0.0.1' or cfg.get('port') != 18210:
    raise ValueError('probe requires localhost:18210')
report={'started':time.time(),'checks':[],'requests':[],'host':subprocess.check_output(['hostname'],text=True).strip(),'source':subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()}
def persist(phase):
 report['phase']=phase;(R/'live-results.json').write_text(json.dumps(report,indent=2));print(phase,flush=True)
def check(name,ok,**detail):
 report['checks'].append(dict(name=name,passed=bool(ok),**detail));persist(name)
def http(method,path,body=None,timeout=900):
 req=urllib.request.Request(URL+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'},method=method)
 try:
  with urllib.request.urlopen(req,timeout=timeout) as resp:return resp.status,json.load(resp)
 except urllib.error.HTTPError as e:return e.code,json.loads(e.read())
def start(summaries=False):
 global proc,log
 cfg=json.loads(CFG.read_text());cfg['experimental_responses_summaries']=summaries;CFG.write_text(json.dumps(cfg,indent=2))
 log=(R/'server.log').open('a');proc=subprocess.Popen([runtime['python'],'-u','-m','serve.server','--engine','strata','--config',str(CFG),'--host','127.0.0.1','--port','18210'],cwd=S,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 deadline=time.time()+900
 while time.time()<deadline:
  if proc.poll() is not None:raise RuntimeError('Server startup failed; see server.log')
  try:
   if http('GET','/health',timeout=3)[0]==200:return
  except (OSError,ValueError):pass
  time.sleep(2)
 raise TimeoutError('Server startup')
def stop(crash=False):
 global proc,log
 if proc is not None and proc.poll() is None:
  if crash:os.killpg(proc.pid,signal.SIGKILL)
  else:proc.send_signal(signal.SIGTERM)
  try:proc.wait(timeout=90)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
 if log:log.close()
 proc=None

def text_of(j):return ''.join(c.get('text','') for o in j.get('output',[]) for c in o.get('content',[]) if c.get('type')=='output_text')
def ask(label,prompt,parent=None,**extra):
 body=dict(model='cache-integration-test',input=prompt,max_output_tokens=32,temperature=0,seed=42,reasoning={'effort':'none'})
 if parent:body['previous_response_id']=parent
 body.update(extra);before=time.monotonic();code,j=http('POST','/v1/responses',body)
 row=dict(label=label,status_code=code,wall_s=time.monotonic()-before,response=j);report['requests'].append(row);persist(label)
 if code!=200:raise RuntimeError(label+': '+str(j))
 check(label+' durable response',j.get('status') in ['completed','incomplete'] and http('GET','/v1/responses/'+j['id'])[0]==200)
 return j

def cached(j):return j.get('usage',{}).get('input_tokens_details',{}).get('cached_tokens',0)
def prompt(mark):
 return ('For this conversation the project marker is '+mark+'. Remember it exactly.\n'+('\n'.join(f'Record {i}: '+mark+' uses blue folders and checks each result before publishing.' for i in range(65)))+'\nReply only ACK; keep the marker for the next turn.')
def items(ident):return http('GET','/v1/responses/'+ident+'/input_items?limit=100&order=asc')
try:
 persist('loading');start(False)
 a=ask('A1',prompt('ALPHA731'));b=ask('B1',prompt('BETA492'));c=ask('C1',prompt('GAMMA916'))
 a2=ask('A2-after-B-C','What is the project marker? Reply with the marker only.',a['id']);check('A resumed from disk',cached(a2)>500 and 'ALPHA731' in text_of(a2),cached_tokens=cached(a2),marker_recalled='ALPHA731' in text_of(a2))
 a3=ask('A3-live','Repeat the marker only.',a2['id']);check('live continuation reused prefix',cached(a3)>cached(a2),cached_tokens=cached(a3))
 b2=ask('B2','What is the project marker? Reply with the marker only.',b['id']);check('B independent cache',cached(b2)>500 and 'BETA492' in text_of(b2),marker_recalled='BETA492' in text_of(b2))
 left=ask('left-branch','Remember the branch tag LEFT209. Reply ACK.',a2['id']);right=ask('right-branch','Remember the branch tag RIGHT806. Reply ACK.',a2['id'])
 _,il=items(left['id']);_,ir=items(right['id']);check('branch history isolation','LEFT209' in json.dumps(il) and 'RIGHT806' not in json.dumps(il) and 'RIGHT806' in json.dumps(ir) and 'LEFT209' not in json.dumps(ir))
 code,saved=http('POST','/slots/0?action=save',{'filename':'checkpoint.session'});check('explicit durable session save',code==200,result=saved)
 code,restored=http('POST','/slots/0?action=restore',{'filename':'checkpoint.session'});check('streaming session restore',code==200,result=restored)
 stop();start(False)
 after=ask('after-clean-restart','What is the project marker? Reply with the marker only.',b2['id']);check('history and KV survive restart',cached(after)>500 and 'BETA492' in text_of(after),marker_recalled='BETA492' in text_of(after))
 code,_=http('DELETE','/v1/responses/'+a2['id']);code_left,il=items(left['id']);check('parent deletion preserves existing descendants',code==200 and code_left==200 and 'ALPHA731' in json.dumps(il))
 # Private applications must bind their own immutable profile revision. Exercise the public analogue: changed instructions invalidate the prefix.
 changed=ask('changed-instructions','Reply CHANGE_OK only.',after['id'],instructions='You are a different test profile. Follow the current request exactly.')
 check('changed instructions avoid stale KV reuse',cached(changed)<100,cached_tokens=cached(changed))
 off=ask('summaries-off','Compute 17 * 19 and explain your arithmetic.',max_output_tokens=192,reasoning={'effort':'medium','summary':'concise'})
 check('summary generation disabled',all(not x.get('summary') for x in off.get('output',[])))
 stop();start(True)
 on=ask('summaries-on','Compute 17 * 19 and explain your arithmetic.',max_output_tokens=384,reasoning={'effort':'medium','summary':'concise'})
 reasoning=[x for x in on.get('output',[]) if x.get('type')=='reasoning'];check('summary second pass exercised',bool(reasoning) and any(x.get('summary') for x in reasoning))
 continued=ask('primary-after-summary','What product did you calculate? Reply with the number.',on['id']);check('primary history continues after summary',http('GET','/v1/responses/'+continued['id'])[0]==200 and cached(continued)>0 and '323' in text_of(continued),cached_tokens=cached(continued))
 # Disconnect a streaming request after generation begins, then verify the durable API state and server recovery.
 body=dict(model='cache-integration-test',input='Write a long numbered list of 100 simple Python programming tips.',max_output_tokens=1024,stream=True,reasoning={'effort':'none'})
 req=urllib.request.Request(URL+'/v1/responses',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 cancelled=None
 with urllib.request.urlopen(req,timeout=900) as stream:
  while True:
   line=stream.readline()
   if not line:break
   if not line.startswith(b'data: '):continue
   data=line[6:].strip()
   if data==b'[DONE]':break
   event=json.loads(data)
   if event.get('type')=='response.created':cancelled=event['response']['id']
   if event.get('type')=='response.output_text.delta':break
 for _ in range(60):
  time.sleep(1);code,state=http('GET','/v1/responses/'+cancelled)
  if state.get('status')!='in_progress':break
 check('stream disconnect settles response',state.get('status') in ['cancelled','failed'],status=state.get('status'))
 recovery=ask('after-disconnect','What is the project marker? Reply with the marker only.',c['id']);check('queue recovers after cancellation',cached(recovery)>500,cached_tokens=cached(recovery))
 stop(crash=True)
 # Only files in this disposable test spill directory are modified; API history is retained.
 spill=R/'spill';files=list(spill.glob('*.sess'))
 for p in files:
  if not p.resolve().is_relative_to(R):raise ValueError('spill file escapes disposable workdir')
  with p.open('r+b') as f:
   f.seek(-1,2);v=f.read(1);f.seek(-1,2);f.write(bytes([v[0]^1]))
 check('corruption fixture exists',bool(files),files=len(files))
 start(False)
 rebuilt=ask('corrupt-cache-fallback','What is the project marker? Reply with the marker only.',recovery['id']);check('corrupt KV falls back to saved history',cached(rebuilt)<100 and 'GAMMA916' in text_of(rebuilt),cached_tokens=cached(rebuilt),marker_recalled='GAMMA916' in text_of(rebuilt))
 stop()
 # Set a budget smaller than one snapshot: the engine must stay usable through cache eviction.
 cfg=json.loads(CFG.read_text());i=cfg['args'].index('--conversation-cache-disk-mib');cfg['args'][i+1]='1';CFG.write_text(json.dumps(cfg,indent=2));start(False)
 ev=ask('small-cache-budget','Repeat the project marker only.',rebuilt['id']);check('history survives cache-budget eviction',http('GET','/v1/responses/'+ev['id'])[0]==200,cached_tokens=cached(ev))
 stop();report['success']=all(x['passed'] for x in report['checks']);report['finished']=time.time();persist('complete')
except Exception as e:
 report['error']=repr(e);report['traceback']=traceback.format_exc();persist('failed');raise
finally:
 stop();(R/'live-results.json').write_text(json.dumps(report,indent=2))
