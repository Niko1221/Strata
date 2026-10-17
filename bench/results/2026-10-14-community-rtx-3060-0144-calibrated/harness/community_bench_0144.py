"""Windows HTTP benchmark for Strata 0.1.44.1. No model downloads or engine updates.

Run with the Strata environment: python community_bench_0144.py --runtime PATH
  --data PATH --out PATH [--soak-seconds 900]
Fresh speed requests complete at a 512-token output cap; no sampled cancellation.
Only synthetic public prompts are used. Resource protection is recorded separately.
"""
import argparse, copy, ctypes, datetime as dt, hashlib, json, os, platform, shutil
import socket, subprocess, sys, threading, time, urllib.request
from ctypes import wintypes
from pathlib import Path
import psutil

SAMPLING=dict(temperature=1.0,top_p=.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0)
FLAGS=getattr(subprocess,'CREATE_NO_WINDOW',0)
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def kill(proc):
    if proc is None:return
    try:
        parent=psutil.Process(proc.pid);children=parent.children(recursive=True)
        for p in reversed(children):
            try:p.terminate()
            except psutil.NoSuchProcess:pass
        parent.terminate()
        _,alive=psutil.wait_procs(children+[parent],timeout=10)
        for p in alive:p.kill()
    except psutil.NoSuchProcess:pass
def worker(source,dest,base):
    body=json.loads(Path(source).read_text(encoding='utf-8'))
    start=time.monotonic();first=None;answer=[];thought=[];usage={};finish=None
    result={}
    try:
        req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=None) as response:
            for line in response:
                if not line.startswith(b'data: '):continue
                data=line[6:].strip()
                if data==b'[DONE]':break
                chunk=json.loads(data)
                if chunk.get('error'):raise RuntimeError(json.dumps(chunk['error']))
                if chunk.get('usage'):usage=chunk['usage']
                for choice in chunk.get('choices',[]):
                    delta=choice.get('delta') or {}
                    content=delta.get('content') or '';reasoning=delta.get('reasoning_content') or ''
                    if first is None and (content or reasoning):first=time.monotonic()-start
                    answer.append(content);thought.append(reasoning)
                    finish=choice.get('finish_reason') or finish
        result={'ok':True,'content':''.join(answer),'reasoning_content':''.join(thought),'usage':usage,
                'finish_reason':finish,'ttft_s':first,'client_elapsed_s':time.monotonic()-start}
    except Exception as e:
        result={'ok':False,'error_type':type(e).__name__,'error':str(e),'client_elapsed_s':time.monotonic()-start}
    write(Path(dest),result)

class ProtectionError(RuntimeError):pass
class Monitor:
    def __init__(self,runner):
        self.r=runner;self.done=threading.Event();self.flag=None;self.last={};self.low=None;self.high=None
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def loop(self):
        class PI(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD)]+[(x,ctypes.c_size_t) for x in
              ('CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache','KernelTotal',
               'KernelPaged','KernelNonpaged','PageSize')]+[('HandleCount',wintypes.DWORD),('ProcessCount',wintypes.DWORD),('ThreadCount',wintypes.DWORD)]
        with (self.r.out/'telemetry.jsonl').open('a',encoding='utf-8') as f:
            while not self.done.is_set():
                t=time.monotonic();mem=psutil.virtual_memory();pi=PI();pi.cb=ctypes.sizeof(pi)
                ok=ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi),ctypes.sizeof(pi))
                fraction=pi.CommitTotal/pi.CommitLimit if ok and pi.CommitLimit else None
                row={'utc':utc(),'epoch':time.time(),'session':self.r.session,'case':self.r.current,
                     'available_ram_bytes':mem.available,'used_ram_bytes':mem.used,'commit_fraction':fraction,
                     'commit_total_bytes':pi.CommitTotal*pi.PageSize if ok else None,'commit_limit_bytes':pi.CommitLimit*pi.PageSize if ok else None}
                rss=0
                if self.r.server and self.r.server.poll() is None:
                    try:
                        for p in [psutil.Process(self.r.server.pid)]+psutil.Process(self.r.server.pid).children(recursive=True):
                            try:rss+=p.memory_info().rss
                            except psutil.NoSuchProcess:pass
                    except psutil.NoSuchProcess:pass
                row['server_tree_rss_bytes']=rss
                q=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,clocks.current.graphics,clocks.current.memory,power.draw,power.limit,pcie.link.gen.current,pcie.link.width.current','--format=csv,noheader,nounits'],capture_output=True,text=True,creationflags=FLAGS)
                if q.returncode==0:
                    keys=('vram_mib','vram_total_mib','gpu_util_percent','temperature_c','graphics_mhz','memory_mhz','power_w','power_limit_w','pcie_gen','pcie_width')
                    row.update({k:float(v.strip()) if v.strip().replace('.','',1).isdigit() else None for k,v in zip(keys,q.stdout.strip().split(','))})
                io=psutil.disk_io_counters()
                if io:row.update(disk_read_bytes=io.read_bytes,disk_write_bytes=io.write_bytes)
                active=self.r.server is not None
                self.low=(self.low or t) if active and mem.available<1*1024**3 else None
                self.high=(self.high or t) if active and fraction is not None and fraction>=.95 else None
                if self.low is not None and t-self.low>=15:self.flag='available RAM <1 GiB for >=15 s'
                if self.high is not None and t-self.high>=30:self.flag='commit >=95% for >=30 s'
                if self.flag:row['protection_stop']=self.flag
                self.last=row;f.write(json.dumps(row)+'\n');f.flush();self.done.wait(max(.05,1-(time.monotonic()-t)))

class Runner:
    def __init__(self,a):
        self.a=a;self.runtime=a.runtime.resolve();self.data=a.data.resolve();self.out=a.out.resolve()
        self.server=None;self.active=None;self.log=None;self.session=None;self.current=None;self.rows=[]
        self.base='http://127.0.0.1:18080';self.number=0;self.monitor=Monitor(self)
        sys.path[:0]=[str(self.runtime),str(self.runtime/'tools')]
        from strata_tokenizer import Tokenizer
        from serve.frontend import ChatTemplate,openai_to_messages
        self.openai_to_messages=openai_to_messages
        directory=self.data/'packs/iq2_xs/tokenizer';vocab=json.loads((directory/'vocab.json').read_text(encoding='utf-8'))
        tokens=[None]*len(vocab)
        for text,n in vocab.items():tokens[n]=text
        self.tok=Tokenizer(tokens,(directory/'merges.txt').read_text(encoding='utf-8').splitlines(),json.loads((directory/'token_type.json').read_text()))
        self.template=ChatTemplate(directory/'chat_template.jinja')
        self.filler=self.tok.encode('\n'.join(f'def task_{i:05d}(value: int) -> int: return (value * {(i%97)+1} + {i}) % 100003' for i in range(18000)))
    def get(self,path):
        with urllib.request.urlopen(self.base+path,timeout=5) as f:return json.load(f)
    def check(self):
        if self.monitor.flag:raise ProtectionError(self.monitor.flag)
        if self.server and self.server.poll() is not None:raise RuntimeError('Owned server exited')
    def body(self,text,maximum=512):
        return dict(model=self.model,messages=[dict(role='user',content=text)],**SAMPLING,reasoning_effort='xhigh',
                    max_tokens=maximum,stream=True,stream_options={'include_usage':True},seed=2718)
    def count(self,body):
        messages,tools,kwargs=self.openai_to_messages(body)
        return len(self.tok.encode(self.template.render(messages,tools,**kwargs),parse_special=True))
    def make(self,target,label,maximum=512,depth=None):
        prefix=f'Unique community benchmark {label}.\n'
        ending='\nExplain this module in detail, including testing, arithmetic, complexity and naming. Write at least 600 words.'
        secret=f'amber-quartz-{target}-{depth}'
        needle=f'\n# The exact code word for this document is {secret}.\n' if depth is not None else ''
        if needle:ending='\nWhat is the exact code word hidden in this document? Reply only with the code word.'
        n=target-self.count(self.body(prefix+needle+ending,maximum));best=None
        for _ in range(12):
            cut=int(n*(depth or 0)/100)
            text=prefix+self.tok.decode(self.filler[:cut])+needle+self.tok.decode(self.filler[cut:n])+ending
            body=self.body(text,maximum);actual=self.count(body)
            if actual<=target and (best is None or actual>best[0]):best=(actual,body)
            if actual==target:break
            n+=target-actual
        assert best and target-best[0]<=20,(target,best and best[0])
        return best[1],secret if needle else None
    def start(self,model,context):
        assert self.server is None
        self.model=model;self.ctx=context;self.session=f'{model}-{context}';self.monitor.flag=None
        self.monitor.low=None;self.monitor.high=None
        cfg=copy.deepcopy(json.loads((self.out/'fixed-configs'/f'{model}.json').read_text(encoding='utf-8')))
        cfg['args'][cfg['args'].index('--max-context')+1]=str(context)
        cfg.update(model_name=model,host='127.0.0.1',port=18080,open_browser=False,parallel=1,lazy_load=False,
                   log=str(self.out/'servers'/f'{self.session}-engine.log'))
        env=os.environ.copy()
        for key in list(env):
            if key.startswith('STRATA_'):env.pop(key)
        if model=='iq3_xxs':
            env.update(STRATA_RESIDENT_HEADROOM_GIB='1',STRATA_UNBUFFERED_LOAD='0')
        path=self.out/'configs'/f'{self.session}.json';write(path,cfg)
        write(path.with_suffix('.environment.json'),{k:v for k,v in env.items() if k.startswith('STRATA_')})
        self.log=(self.out/'servers'/f'{self.session}-server.log').open('w',encoding='utf-8')
        self.server=subprocess.Popen([sys.executable,str(self.runtime/'serve/server.py'),'--engine','strata','--config',str(path),'--port','18080'],
                                    cwd=self.runtime,env=env,stdout=self.log,stderr=subprocess.STDOUT,creationflags=FLAGS)
        write(self.out/'owned-process.json',{'pid':self.server.pid,'created':psutil.Process(self.server.pid).create_time()})
        start=time.monotonic()
        while True:
            self.check()
            if time.monotonic()-start>600:raise RuntimeError('Model load did not complete within 10 minutes')
            try:health=self.get('/health')
            except (OSError,ValueError):time.sleep(1);continue
            if health.get('loaded'):break
            time.sleep(1)
        metrics=self.get('/metrics');assert metrics['engine']['version']=='0.1.44.1'
        assert metrics['engine']['max_context']==context
        write(self.out/'servers'/f'{self.session}-loaded.json',dict(load_seconds=time.monotonic()-start,metrics=metrics))
        print(json.dumps({'event':'server_ready','session':self.session}),flush=True)
        self.request('warmup',self.body('Explain briefly what a Python function is.',128),'warmup')
        prep,_=self.make(4096,f'preparation-{self.session}',128);self.request('preparation',prep,'warmup')
    def stop(self):
        kill(self.active);self.active=None;kill(self.server);self.server=None
        if self.log:self.log.close();self.log=None
        time.sleep(1)
    def request(self,label,body,group,target=None,rep=None,expected=None):
        self.number+=1;self.current=f'{self.session}:{label}';start_epoch=time.time();start=time.monotonic()
        base=self.out/'requests'/f'{self.number:04d}-{self.session}-{label}'
        expected_tokens=self.count(body);write(base.with_suffix('.json'),body)
        row=dict(number=self.number,session=self.session,model=self.model,context=self.ctx,label=label,group=group,target=target,
                 repetition=rep,started_epoch=start_epoch,expected_prompt_tokens=expected_tokens,
                 request_sha256=hashlib.sha256(json.dumps({k:v for k,v in body.items() if k!='model'},sort_keys=True).encode()).hexdigest())
        try:
            self.check()
            self.active=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(base.with_suffix('.json')),
                str(base.with_suffix('.response.json')),self.base],creationflags=FLAGS)
            last_print=start
            while self.active.poll() is None:
                self.check()
                if time.monotonic()-last_print>=30:
                    print(json.dumps({'event':'progress','session':self.session,'case':label,'seconds':round(time.monotonic()-start),
                                      'available_ram_gib':round(self.monitor.last.get('available_ram_bytes',0)/1024**3,1)}),flush=True);last_print=time.monotonic()
                time.sleep(.2)
            assert self.active.returncode==0,'Worker exited'
            result=json.loads(base.with_suffix('.response.json').read_text(encoding='utf-8'))
            if not result['ok']:raise RuntimeError(result['error'])
            until=time.monotonic()+30
            while True:
                metrics=self.get('/metrics')
                if metrics['live']['state']=='idle':break
                self.check();assert time.monotonic()<until,'Server did not return to idle';time.sleep(.1)
            write(base.with_suffix('.metrics.json'),metrics);engine=metrics['requests'][0]
            assert engine['prompt_tokens']==expected_tokens,(engine['prompt_tokens'],expected_tokens)
            assert engine['finish'] not in ('disconnect','error','cancelled'),engine
            if group in ('speed','needle','soak'):assert not engine.get('reused',0),'Fresh input unexpectedly reused'
            if not result['reasoning_content']:raise RuntimeError('Requested thinking output absent')
            row.update(ok=True,engine=engine,ttft_s=result['ttft_s'],client_elapsed_s=result['client_elapsed_s'],
                       output_file=base.with_suffix('.response.json').name,reasoning_chars=len(result['reasoning_content']),
                       answer_chars=len(result['content']),finish_reason=result['finish_reason'])
            if expected:
                row.update(expected_code_word=expected,recall_pass=expected in result['content'],answer=result['content'])
            if group=='conversation':row['answer']=result['content']
        except Exception as e:
            row.update(ok=False,error_type=type(e).__name__,error=str(e),client_elapsed_s=time.monotonic()-start)
            kill(self.active)
            if isinstance(e,ProtectionError):row['protection_stop']=True
        finally:
            self.active=None;row['ended_epoch']=time.time();self.rows.append(row)
            with (self.out/'rows.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            write(self.out/'status.json',{'utc':utc(),'session':self.session,'case':label,'finished_requests':len(self.rows),
                                        'failures':sum(not r['ok'] for r in self.rows),'soak_active':group=='soak'})
            print(json.dumps({'event':'request_done','session':self.session,'group':group,'label':label,'ok':row['ok'],
                'pp':round((row.get('engine',{}).get('prompt_tokens',0)-row.get('engine',{}).get('reused',0))*1000/max(row.get('engine',{}).get('prompt_ms',1),.001),1),
                'decode':row.get('engine',{}).get('decode_tok_s')}),flush=True)
        if row.get('protection_stop'):raise ProtectionError(row['error'])
        return row
    def conversation(self):
        base,_=self.make(32768,f'conversation-{self.ctx}',1024)
        text=base['messages'][0]['content'];base['messages'][0]['content']=text.rsplit('\nExplain this module',1)[0]+'\nBriefly acknowledge that you received the module. Do not summarize yet.'
        for turn in range(4):
            if turn:
                addition=self.tok.decode(self.filler[10000+turn*1024:10000+(turn+1)*1024])
                base['messages'].append({'role':'user','content':f'Additional source, turn {turn}:\n'+addition+'\nBriefly describe one relevant testing consideration in a single sentence.'})
            row=self.request(f'conversation-{turn}',base,'conversation',rep=turn)
            if not row['ok']:break
            if not row['answer'].strip():
                write(self.out/f'{self.session}-conversation-incomplete.json',{'turn':turn,'reason':'No answer before the 1024-token cap; later turns skipped'})
                break
            base['messages'].append({'role':'assistant','content':row['answer']})
    def soak(self):
        started=time.monotonic();start_epoch=time.time();i=0
        write(self.out/'soak-started.json',{'utc':utc(),'epoch':start_epoch,'seconds_requested':self.a.soak_seconds})
        while time.monotonic()-started<self.a.soak_seconds:
            target=(4096,32768,120000)[i%3];body,_=self.make(target,f'soak-{i}',512)
            body['seed']=9000+i;row=self.request(f'soak-{i}',body,'soak',target=target,rep=i)
            if not row['ok']:raise RuntimeError('Soak request failed; inspect retained row')
            i+=1
        write(self.out/'soak-completion.json',{'completed':True,'started_epoch':start_epoch,'ended_epoch':time.time(),
             'elapsed_seconds':time.monotonic()-started,'requests':i,'no_idle_delays':True})
    def run(self):
        sessions=[(model,ctx) for model in ('q2_0','iq2_xs','iq3_xxs') for ctx in (131072,262144)]
        completed=[]
        try:
            for model,ctx in sessions:
                try:
                    self.start(model,ctx)
                    targets=(4096,32768,120000) if ctx==131072 else (260000,)
                    for rep in range(3):
                        for target in targets if rep!=1 else reversed(targets):
                            body,_=self.make(target,f'speed-{ctx}-{target}-{rep}');body['seed']=2718+rep
                            row=self.request(f'speed-{target}-{rep}',body,'speed',target=target,rep=rep)
                            if not row['ok']:raise RuntimeError('Speed request failed; inspect retained row')
                    if ctx==262144:
                        body,code_word=self.make(260000,f'needle-{ctx}-50',2048,50)
                        row=self.request('needle-50',body,'needle',target=260000,rep=50,expected=code_word)
                        if not row['ok']:raise RuntimeError('Recall request failed; inspect retained row')
                    if model=='iq2_xs' and ctx==131072:
                        self.conversation()
                        self.soak()
                    completed.append(self.session)
                except Exception as e:
                    write(self.out/f'{model}-{ctx}-session-failure.json',{'type':type(e).__name__,'error':str(e),'utc':utc()})
                    print(json.dumps({'event':'session_failure','session':self.session,'type':type(e).__name__,'error':str(e)}),flush=True)
                finally:self.stop()
            write(self.out/'completion.json',{'finished':True,'sessions_completed':completed,'utc':utc(),
                  'requests':len(self.rows),'failed_requests':sum(not r['ok'] for r in self.rows)})
        finally:
            self.stop();self.monitor.done.set();self.monitor.thread.join(timeout=5)
            with socket.socket() as s:closed=s.connect_ex(('127.0.0.1',18080))!=0
            write(self.out/'cleanup.json',{'owned_server_stopped':self.server is None,'owned_worker_stopped':self.active is None,'port18080_closed':closed})

def main():
    if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(sys.argv[2],sys.argv[3],sys.argv[4]);return
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--runtime',type=Path,required=True);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--soak-seconds',type=int,default=900);a=ap.parse_args()
    assert os.name=='nt','This runner records Windows commit telemetry'
    assert not (a.out/'rows.jsonl').exists(),'Preserve prior measurements; choose a new output directory'
    assert a.soak_seconds>=900,'The reduced soak is at least 15 minutes'
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',18080))!=0,'Preserve service already using 18080'
    a.out.mkdir(exist_ok=True)
    for name in ('servers','configs','requests'):(a.out/name).mkdir(exist_ok=True)
    build=json.loads((a.runtime/'engine/BUILD.json').read_text());assert build['version']=='0.1.44.1'
    write(a.out/'protocol.json',{'date_utc':utc(),'engine':build,'engine_sha256':digest(a.runtime/'engine/strata.exe'),
        'runtime_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.runtime,text=True).strip(),
        'normal_config_sha256':digest(a.runtime/'strata-iq2_xs.json'),
        'sampling':SAMPLING,'reasoning_effort':'xhigh','speed_output_cap':512,'needle_output_cap':2048,'conversation_output_cap':1024,
        'repetitions':3,'target_lengths_128k':[4096,32768,120000],'target_lengths_262k':[260000],
        'placements':{'q2_0':'full native arena','iq2_xs':'full native arena','iq3_xxs':'resident 16 GiB; buffered reads; 1 GiB resident headroom'},
        'kv':'int8','kv_resident':32768,'no_request_deadline':True,
        'memory_guard':{'commit_fraction':.95,'commit_seconds':30,'available_ram_gib':1,'available_ram_seconds':15},
        'soak_seconds':a.soak_seconds,'speed_cache':'fresh prompts, zero reused tokens; retained expert cache',
        'order':[f'{model}-{ctx}' for model in ('q2_0','iq2_xs','iq3_xxs') for ctx in (131072,262144)],
        'recall':'one 260K case at 50% depth per model','conversation':'IQ2_XS 128K only, four turns',
        'warmup':'each session: short 128-token output, then 4K 128-token output; excluded',
        'session_policy':'one server per configuration; target order ascending, descending, ascending',
        'prompt_generator':'same synthetic Python functions and labels as 0.1.43; counts within 20 tokens',
        'telemetry':'whole-machine RAM/commit/GPU/system-wide disk I/O about every second; owned tree RSS',
        'old_engine_run':False,'calibration_reused':True})
    write(a.out/'hardware.json',{'os':platform.platform(),'python':sys.version,'physical_ram_bytes':psutil.virtual_memory().total,
        'logical_cpus':psutil.cpu_count(),'cpu_physical_cores':psutil.cpu_count(logical=False),
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version,power.limit,pcie.link.gen.max,pcie.link.width.max','--format=csv,noheader'],text=True).strip()})
    write(a.out/'runner-identity.json',{'runner_sha256':digest(__file__),'base_runner_sha256':digest(a.out/'previous-runner.py')})
    Runner(a).run()
if __name__=='__main__':main()
