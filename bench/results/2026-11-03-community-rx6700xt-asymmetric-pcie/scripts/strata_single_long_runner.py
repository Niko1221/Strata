from pathlib import Path
import json,subprocess,time,urllib.request,sys
r=Path.home()/'LocalAI/experiments/strata-qwen-next-20261103';a=Path.home()/'ai-services/preparation-20261103-three-models';py=str(r/'control-0.1.56/python/bin/python3')
assert (r/'single-fast-short-summary.json').exists()
c=json.loads((r/'diag-single-fast.json').read_text());c['args'][c['args'].index('--max-context')+1]='262144';c['args'][c['args'].index('--prompt-cache')+1]='6';c['log']='single-long-cache-engine.log'
f=r/'single-long-cache.json';assert not f.exists();f.write_text(json.dumps(c,indent=2))
subprocess.run(['systemctl','--user','stop','strata-diag-single-fast-20261103'],check=True,timeout=60)
s=(a/'strata_start_baseline.py').read_text().replace('baseline-4k-profile.json','single-long-cache.json').replace("unit='strata-iq2xs-baseline-20261103'","unit='strata-single-long-cache-20261103'").replace('baseline-server.log','single-long-cache-server.log')
f=a/'strata_start_single_long.py';assert not f.exists();f.write_text(s);subprocess.run([py,str(f)],check=True)
end=time.monotonic()+900
while True:
 try:
  with urllib.request.urlopen('http://<IP>:8081/v1/models',timeout=5):break
 except Exception:pass
 if time.monotonic()>end:raise TimeoutError('single long startup')
 time.sleep(5)
sys.path.insert(0,str(r));from stream_bench import run
cg=subprocess.check_output(['systemctl','--user','show','strata-single-long-cache-20261103','--property=ControlGroup','--value'],text=True).strip();cp=Path('/sys/fs/cgroup')/cg.lstrip('/')
pid=next(int(p) for p in (cp/'cgroup.procs').read_text().split() if (Path('/proc')/p/'exe').resolve().name=='strata')
summary=[]
for window in [131072,262144]:
 source=r/'baseline-qwen-long' if window==131072 else r/'extended-context-cases'
 req=json.loads((source/f'{window}-r0/request.json').read_text());req['model']='Qwen3.8-Flash-Next-IQ2_XS-original';req['max_tokens']=128
 for turn in range(4):
  out=r/'single-long-cache'/f'{window}-turn{turn}';result=run('http://<IP>:8081',req,out,pid)
  summary.append({'window':window,'turn':turn,**{k:result.get(k) for k in ['status','usage','timings','ttft_any_s','content']}})
  (out/'cgroup-after.json').write_text(json.dumps({n:(cp/n).read_text() for n in ['memory.current','memory.events','memory.swap.current','memory.pressure']},indent=2));print(json.dumps(summary[-1]),flush=True)
  if result['status']!='complete':raise RuntimeError('Inspect failed single long test')
  if turn>=1:req['messages'] += [{'role':'assistant','content':result['content']},{'role':'user','content':'Using the same archive, return the same JSON object with the original START, MIDDLE and END codes and available units. Subtract reserved units.' if turn==1 else 'Return the same JSON object. Double-check that available is 137 minus 29.'}]
(r/'single-long-cache-summary.json').write_text(json.dumps(summary,indent=2))
