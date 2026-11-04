"""Fresh-engine prefill comparison; only the resident budget differs."""
import argparse, hashlib, json, os, pathlib, random, secrets, socket, subprocess, sys, threading, time, urllib.request

def get(cfg,route,body=None):
    req=urllib.request.Request(f"http://127.0.0.1:{cfg['port']}{route}",
        headers={'Authorization':'Bearer '+cfg['api_key'],'Content-Type':'application/json'},
        data=None if body is None else json.dumps(body).encode())
    with urllib.request.urlopen(req,timeout=600) as r:return json.load(r)

def prompt(target,repeat):
    rng=random.Random(target*100+repeat); paragraphs=[];length=0
    while length < target*4.25:
        line=(f"Record {rng.randrange(10000,99999)}: The research team measured water flow through the northern valley. "
              f"On day {rng.randrange(1,366)}, station {rng.randrange(1,100)} reported {rng.randrange(1,900)} units. "
              "They checked the instruments and recorded observations before comparing them with the previous season.")
        paragraphs.append(line);length+=len(line)
    return f'Reference prefill-{target}-{repeat}.\n'+'\n'.join(paragraphs)+'\nSummarize the subject in one sentence.'

def run(a):
    cfg=json.loads(a.config.read_text(encoding='utf-8'))
    control=pathlib.Path(cfg['cwd']);sys.path.insert(0,str(control))
    from serve import lifeline
    from strata_control import tune_machine, user_args
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    label=a.condition+('_stress' if a.stress else '')
    cfg.pop('args_written',None)
    cfg['args']=user_args.flat([g for g in user_args.groups(cfg['args']) if g[0]!='--resident-budget-gib'])
    cfg['user_args']=user_args.flat([g for g in user_args.groups(cfg.get('user_args',[])) if g[0]!='--resident-budget-gib'])
    if a.condition=='resident14':cfg['args']+=['--resident-budget-gib','14']
    assert cfg['args'][cfg['args'].index('--max-context')+1]=='131072'
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    cfg.update(port=port,host='127.0.0.1',api_key=secrets.token_hex(16),open_browser=False,log=str(out/(label+'-engine.log')))
    path=out/(label+'-private-config.json');path.write_text(json.dumps(cfg,indent=2))
    workerlog=open(out/(label+'-worker.log'),'w',encoding='utf-8')
    proc=lifeline.popen([sys.executable,'-m','serve.server','--engine','strata','--config',str(path),'--host','127.0.0.1','--port',str(port)],
        'worker',cwd=str(control),env=dict(os.environ,STRATA_NO_PRESET='1',PYTHONUNBUFFERED='1'),
        stdout=workerlog,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    rows=[];samples=[];stop=threading.Event();ready=False
    def monitor():
        while not stop.wait(.5):samples.append({'monotonic':time.monotonic(),'available_mib':tune_machine.memory().get('available_mib'),'phase':'inference' if ready else 'startup'})
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        deadline=time.monotonic()+300
        while True:
            if proc.poll() is not None:raise RuntimeError('Engine worker exited during startup; see logs')
            try:get(cfg,'/health');break
            except Exception:
                if time.monotonic()>deadline:raise TimeoutError('Engine readiness timeout')
                time.sleep(1)
        ready=True;print('Ready: '+label,flush=True)
        model=get(cfg,'/v1/models')['data'][0]['id']
        sizes=(4096,126000) if a.stress else (4096,16384)
        repeats=1 if a.stress else a.repeats
        for size in sizes:
            for repeat in range(repeats):
                text=prompt(size,repeat)
                (out/f'prompt-{size}-{repeat}.txt').write_text(text,encoding='utf-8')
                t0=time.monotonic()
                data=get(cfg,'/v1/chat/completions',{'model':model,'messages':[{'role':'user','content':text}],
                    'max_tokens':8,'temperature':0,'stream':False,'strata_checkpoint':False})
                row={'condition':a.condition,'target':size,'repeat':repeat+1,'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),
                    'wall_s':round(time.monotonic()-t0,3),'usage':data.get('usage'),'timings':data.get('timings')}
                rows.append(row);(out/(label+'-results.json')).write_text(json.dumps(rows,indent=2))
                print(json.dumps(row),flush=True)
                assert row['timings']['cache_n']==0,'Request reused prompt tokens; do not treat as fresh prefill'
        if a.recall:
            lines=prompt(124000,99).splitlines()
            expected={'alpha':'MAPLE-4827','bravo':'ORBIT-9153','charlie':'CEDAR-2608'}
            for fraction,(name,code) in reversed(list(zip((.1,.5,.9),expected.items()))):
                lines.insert(int(len(lines)*fraction),f'Archive code for {name}: {code}.')
            text='Reference recall-20261104.\n'+'\n'.join(lines)+'\nReturn the archive codes for alpha, bravo and charlie, in that order. Output the codes only.'
            (out/'recall-prompt.txt').write_text(text,encoding='utf-8')
            t0=time.monotonic()
            data=get(cfg,'/v1/chat/completions',{'model':model,'messages':[{'role':'user','content':text}],
                'max_tokens':128,'reasoning_budget_tokens':32,'temperature':0,'stream':False,'strata_checkpoint':False})
            answer=data['choices'][0]['message'].get('content') or ''
            result={'expected':expected,'answer':answer,'matched':{k:v in answer for k,v in expected.items()},
                'usage':data.get('usage'),'timings':data.get('timings'),'wall_s':time.monotonic()-t0,
                'prompt_sha256':hashlib.sha256(text.encode()).hexdigest()}
            (out/'recall-result.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(result),flush=True)
    except Exception as exc:
        (out/(label+'-failure.json')).write_text(json.dumps({'error':str(exc),'completed_requests':len(rows)}))
        raise
    finally:
        stop.set();thread.join(timeout=2)
        (out/(label+'-memory.json')).write_text(json.dumps(samples,indent=2))
        for phase in ['startup','inference']:
            values=[s['available_mib'] for s in samples if s['phase']==phase and s['available_mib'] is not None]
            print(json.dumps({'phase':phase,'minimum_available_ram_mib':min(values) if values else None}),flush=True)
        proc.terminate()
        try:proc.wait(timeout=30)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
        workerlog.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True)
    p.add_argument('--condition',choices=['baseline','resident14'],required=True);p.add_argument('--repeats',type=int,default=3);p.add_argument('--stress',action='store_true');p.add_argument('--recall',action='store_true')
    run(p.parse_args())
