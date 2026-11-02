"""Custom ABBA rotation benchmark. No engine changes, retries, or published actions."""
from pathlib import Path
import os,sys,json,time,subprocess,signal,hashlib,threading,fcntl,urllib.request,traceback,re,datetime
BASE=Path('<BENCH_ROOT>')
O=BASE/'attempt-2'
S=BASE/'Strata-c62d5e3a15b40ea51c96aae9e8d36928965f53ab'
B=BASE/'build'
D=Path('<EXISTING_INSTALL>')
PY=D/'.venv/bin/python'
sys.path.insert(0,str(O));import replay_api as api
def save(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def get(path,port=8081):
 with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}',timeout=5) as f:return json.load(f)
def cmd(c,timeout=120):return subprocess.run(c,capture_output=True,text=True,timeout=timeout)
def child(pid):
 try:
  candidates=set()
  for task in Path(f'/proc/{pid}/task').iterdir():
   try:candidates.update(int(x) for x in (task/'children').read_text().split())
   except OSError:pass
  found=[c for c in candidates if Path(f'/proc/{c}/exe').resolve().name=='strata']
  assert len(found)==1,found
  return found[0]
 except (OSError,AssertionError):return None
def wait_loaded(p,port):
 start=time.monotonic()
 while time.monotonic()-start<600:
  if p.poll() is not None:raise RuntimeError('server exited before loaded')
  try:
   j=get('/health',port)
   if j.get('loaded'):return j
  except OSError:pass
  time.sleep(1)
 raise TimeoutError('model load timed out')
def no_gpu_apps():
 start=time.monotonic()
 while time.monotonic()-start<60:
  apps=cmd(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).stdout.strip()
  mem=int(cmd(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).stdout.strip())
  if not apps and mem==0:return
  time.sleep(1)
 raise RuntimeError(f'GPU admission failed: apps={apps}, used={mem}')
def stop(p):
 if p and p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def setarg(cfg,k,v):
 if k in cfg['args']:cfg['args'][cfg['args'].index(k)+1]=str(v)
 else:cfg['args'] += [k,str(v)]
def sample(out,pid,event):
 with out.open('w') as f:
  while not event.is_set():
   try:
    proc=Path('/proc')/str(pid)
    row={'time':time.time(),'status':(proc/'status').read_text(),'io':(proc/'io').read_text(),'meminfo':Path('/proc/meminfo').read_text(),'gpu':cmd(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current','--format=csv,noheader,nounits'],10).stdout.strip()}
    f.write(json.dumps(row)+'\n');f.flush()
   except Exception as e:f.write(json.dumps({'time':time.time(),'error':repr(e)})+'\n')
   event.wait(2)
def source_hashes():
 return {str(p.relative_to(S)):sha(p) for p in S.rglob('*') if p.is_file() and p.suffix in {'.cpp','.hpp','.cu','.cuh','.py','.h','.c','.jinja'} }
def main():
 assert (O/'build-completed.json').exists()
 run=O/'abba';run.mkdir(exist_ok=False)
 lock=Path('<EXPERIMENT_LOCK>/gpu-test.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert get('/metrics',8080)['live']['state']=='idle' and get('/metrics',8080)['live']['queued']==0
 apps=cmd(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader']).stdout.strip().splitlines()
 assert len(apps)==1 and str(D/'engine-cuda12/strata') in apps[0],apps
 owned=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   argv=(p/'cmdline').read_bytes().split(b'\0')
   if b'serve/server.py' in argv and b'8080' in argv and (p/'cwd').resolve()==D:owned.append(int(p.name))
  except (OSError,ProcessLookupError):pass
 assert len(owned)==1,owned
 original_files=[D/'engine-cuda12/strata',D/'start-optimized-3090.sh',D/'strata-optimized-3090.json',D/'src/core/expert_source.cpp',D/'src/program/generate.cpp',D/'include/strata/core/verify.hpp',D/'build-cuda12/libstrata_kernels.a']
 originals={str(p):sha(p) for p in original_files};save(O/'original-hashes.json',originals)
 selectors=cmd(['<HARDWARE_AUDIT>/selectors','--read']);assert selectors.returncode==0
 save(O/'hardware-before.json',{'selectors':selectors.stdout,'dmesg':cmd(['dmesg']).stdout,'gpu_csv':cmd(['nvidia-smi','--query-gpu=uuid,persistence_mode,power.limit','--format=csv,noheader']).stdout})
 save(O/'source-before.json',source_hashes())
 save(O/'host.json',{'gpu':cmd(['nvidia-smi','-q']).stdout,'cpu':cmd(['lscpu']).stdout,'memory':Path('/proc/meminfo').read_text(),'os':Path('/etc/os-release').read_text(),'uname':cmd(['uname','-a']).stdout,'storage':cmd(['lsblk','-o','NAME,TYPE,SIZE,MODEL,ROTA,TRAN']).stdout,'pip_freeze':cmd([str(PY),'-m','pip','freeze']).stdout,'gcc':cmd(['g++','--version']).stdout,'nvcc':cmd(['/usr/local/cuda-12.8/bin/nvcc','--version']).stdout,'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 cases=json.loads((O/'cases.json').read_text());persist=cmd(['nvidia-smi','--query-gpu=persistence_mode','--format=csv,noheader']).stdout.strip();stopped=False;p=None
 try:
  assert not any(x.get('is_processing') for x in get('/slots',8080))
  save(O/'daily-before.json',{'pid':owned[0],'engine_pid':child(owned[0]),'health':get('/health',8080),'metrics':get('/metrics',8080)})
  cp=child(owned[0]);stopped=True;os.kill(owned[0],signal.SIGTERM)
  if cp:
   try:os.kill(cp,signal.SIGTERM)
   except ProcessLookupError:pass
  pm=cmd(['nvidia-smi','-pm','0']);assert pm.returncode==0
  no_gpu_apps()
  # Validate the unchanged official rotation fixtures before model timing.
  tests=[]
  for name,c in [('ctest',[str(D/'.venv/bin/ctest'),'--test-dir',str(B),'-R','^(exchange_storage_test|file_expert_source_test)$','--output-on-failure']),('rotation-gpu',[str(B/'file_expert_source_test'),'--rotation-gpu'])]:
   t=cmd(c,180);(O/(name+'.log')).write_text(t.stdout+t.stderr);tests.append({'name':name,'cmd':c,'rc':t.returncode});save(O/'tests.json',tests);assert t.returncode==0,t.stderr
  no_gpu_apps()
  # Hash the actual reused weights and pack. Startup and hashing are excluded from request timing.
  base=json.loads((O/'base-config.json').read_text());roots=[Path(base['args'][base['args'].index(k)+1]) for k in ['--pack','--native','--ple-gguf','--mtp']]
  files=sorted({p for root in roots for p in (root.rglob('*') if root.is_dir() else [root]) if p.is_file()})
  previous=json.loads((BASE/'model-hashes.json').read_text())
  hashed_at=(BASE/'model-hashes.json').stat().st_mtime
  hashes={};reuse=[]
  for file in files:
   st=file.stat();old=previous.get(str(file));fresh=st.st_mtime<=hashed_at and st.st_ctime<=hashed_at
   assert old and old['bytes']==st.st_size and fresh,(file,old)
   hashes[str(file)]=old;reuse.append({'file':str(file),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'ctime_ns':st.st_ctime_ns})
   save(O/'model-hashes.json',hashes)
  save(O/'model-hashes-reuse.json',{'source':'../model-hashes.json from the completed pre-request hashing phase of attempt 1','source_sha256':sha(BASE/'model-hashes.json'),'files':reuse})
  shape=None
  for ix,arm in enumerate(['A','B','B','A']):
   assert all(Path(x['file']).stat().st_mtime_ns==x['mtime_ns'] and Path(x['file']).stat().st_ctime_ns==x['ctime_ns'] and Path(x['file']).stat().st_size==x['bytes'] for x in reuse)
   no_gpu_apps();out=run/f'{ix}-{arm}';out.mkdir(exist_ok=False)
   cfg=json.loads((O/'base-config.json').read_text());cfg.update(exe=str(B/'strata'),cwd=str(S),port=8081,host='127.0.0.1',parallel=1,open_browser=False,log=str(out/'engine.log'))
   profile=out/'expert-profile.bin';profile.write_bytes((O/'frozen-expert-profile.bin').read_bytes())
   for k,v in {'--expert-profile':profile,'--prefill':'2048','--expert-cache':'18432','--spec':'4','--pcie-frac':'1','--pcie-mode':'kernel','--adapt-every':'4','--adapt-swaps':'96','--adapt-decay':'0.7','--adapt-async':'0','--prompt-cache':'0','--conversation-cache-mib':'0','--ple-io':'ram'}.items():setarg(cfg,k,v)
   save(out/'config.json',cfg)
   env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')};env.update(STRATA_ROUTE_TAIL_SKIP='0',STRATA_PREFILL_CPU_SHARE='0',STRATA_EXCHANGE_ROTATE='1' if arm=='B' else '0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
   save(out/'fingerprint.json',{'arm':arm,'engine_sha256':sha(B/'strata'),'initial_profile_sha256':sha(profile),'env':{k:v for k,v in env.items() if k.startswith('STRATA_') or k.endswith('NUM_THREADS')}})
   event=threading.Event();th=None
   try:
    with (out/'server.out').open('w') as f:p=subprocess.Popen([str(PY),str(S/'serve/server.py'),'--engine','strata','--config',str(out/'config.json'),'--host','127.0.0.1','--port','8081'],cwd=S,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    start=time.monotonic();health=wait_loaded(p,8081);engine=get('/metrics')['engine'];log=(out/'engine.log').read_text()
    assert ('exchange buffer rotation enabled:' in log)==(arm=='B'),log[-5000:]
    actual={k:engine.get(k) for k in ['expert_slots','expert_cache_mib','arena_mib','spec','mtp_max','pool_workers','pcie_frac','context','kv']}
    if shape is None:shape=actual
    assert actual==shape,(actual,shape)
    assert engine.get('version')=='0.1.55',engine
    save(out/'environment.json',{'health':health,'engine':engine,'fixed_shape':actual,'load_s':time.monotonic()-start,'slots':get('/slots'),'proc_status':Path(f'/proc/{child(p.pid)}/status').read_text(),'meminfo':Path('/proc/meminfo').read_text()})
    th=threading.Thread(target=sample,args=(out/'resources.jsonl',child(p.pid),event),daemon=True);th.start()
    planned=[]
    for label in ['warm','r1','r2']:
     for c in cases['decode']:planned.append(dict(c,id=label+'-'+c['id'],max_tokens=512,phase='warmup' if label=='warm' else 'performance'))
    for c in cases['repair']:planned.append(dict(c,id='r1-'+c['id'],max_tokens=2048,phase='repair'))
    save(out/'schedule.json',planned)
    for case in planned:
     x=api.request(case,out,'http://127.0.0.1:8081/v1','qwen3.8-flash-next-iq2_xs',seed=20261013,reasoning='none',before=lambda:get('/metrics'),after=lambda:get('/metrics'),log=out/'engine.log')
     assert x.get('http_status')==200 and x.get('sse_done') and x.get('error') is None and x.get('finish_reason'),x.get('error')
     assert x['usage']['prompt_tokens']>0 and x['usage']['completion_tokens']>0
     assert x['timings']['predicted_ms']>0
     assert x['timings'].get('cache_n',0)==0
     if case['phase']=='repair':
      grade=subprocess.run([str(PY),str(O/'grade_code.py'),'-'],input=json.dumps(dict(x,case_id='code_retry')),capture_output=True,text=True,timeout=15)
      save(out/'repair-grade.json',{'rc':grade.returncode,'stdout':grade.stdout,'stderr':grade.stderr})
    save(out/'completed.json',{'time':time.time(),'requests':len(planned)})
   finally:
    event.set()
    if th:th.join(10)
    stop(p);p=None
   time.sleep(3)
  save(O/'completed.json',{'arms':['A','B','B','A'],'requests':28,'time':time.time()})
 except BaseException:
  save(O/'error.json',{'error':traceback.format_exc(),'time':time.time()});raise
 finally:
  stop(p)
  if stopped:
   no_gpu_apps()
   if persist=='Enabled':assert cmd(['nvidia-smi','-pm','1']).returncode==0
   with (O/'daily-restored.out').open('a') as f:p=subprocess.Popen([str(D/'start-optimized-3090.sh')],cwd=D,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   health=wait_loaded(p,8080)
   now={str(p):sha(p) for p in original_files};save(O/'original-hashes-after.json',now);assert originals==now
   save(O/'source-after.json',source_hashes());assert json.loads((O/'source-before.json').read_text())==json.loads((O/'source-after.json').read_text())
   final_selectors=cmd(['<HARDWARE_AUDIT>/selectors','--read']);assert final_selectors.stdout==selectors.stdout
   save(O/'hardware-after.json',{'selectors':final_selectors.stdout,'dmesg':cmd(['dmesg']).stdout})
   save(O/'restored.json',{'health':health,'slots':get('/slots',8080),'metrics':get('/metrics',8080),'hashes_equal':True,'selectors_equal':True,'time':time.time()})
if __name__=='__main__':main()
