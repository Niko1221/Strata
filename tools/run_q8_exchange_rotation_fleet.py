#!/usr/bin/env python3
"""Q8/FP16 buffer ownership experiment, on the private llm-60 GPU queue."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time

H=Path.home(); R=Path(__file__).resolve().parents[1]
PY=H/'src/Strata/.venv/bin/python'
D=Path(os.environ.get('FLEET_EXCHANGE_OUTPUT',H/'fleet-downloads/rtxpro-q8-exchange-rotation-20261003'))
PRIOR=Path(os.environ.get('FLEET_EXCHANGE_AFTER',H/'fleet-downloads/rtxpro-q8-long-context-retrieval-20261003/status.json'))
CONTROL=H/'src/strata-q8-placement-080891d4'
MODES={'serial':('off',0),'mtp':('on',0),'ngram':('off',3),'mtp-ngram':('on',3)}

def save(p,j):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(j,indent=2)+'\n');tmp.replace(p)

def option(args,key,value):
    if key in args:args[args.index(key)+1]=str(value)
    else:args.extend([key,str(value)])

def first_difference(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),
                min(len(a),len(b)) if len(a)!=len(b) else None)

def worker(out):
    sys.path.insert(0,str(R/'tools'))
    from configure_rtxpro import configure
    out.mkdir(parents=True,exist_ok=False)
    state={'started':time.time(),'source':(R/'source-commit.txt').read_text().strip(),
        'kv':'fp16','weights':'Q8_0','records':[],
        'engine_sha256':hashlib.sha256((R/'build/strata').read_bytes()).hexdigest(),
        'scope':'One storage change; unchanged cache capacity, PCIe split, routing and adaptive waits.'}
    target=out/'matrix.json'; save(target,state)
    env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')}

    def command(label,args):
        state['current']=label;save(target,state)
        with (out/(label+'.log')).open('w') as log:
            p=subprocess.run(list(map(str,args)),env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        if p.returncode:raise RuntimeError(label+' failed with exit '+str(p.returncode))

    def case(label,mode,rotate,*,engine=R,prompt=65536,output=1024,gate=False,cases=('coding','editing')):
        cfg=configure({'weights':'Q8_0','kv':'fp16','load_projection':False},engine,H,16384 if gate else 73728)
        cfg['env'].update(STRATA_PLE_PREFAULT_THREADS='8',STRATA_ADAPT_NOWAIT='0',STRATA_EXCHANGE_ROTATE=str(rotate))
        option(cfg['args'],'--pcie-frac',-1)
        option(cfg['args'],'--expert-cache',16000 if gate else 16192 if MODES[mode][0]=='on' else 16400)
        config=out/(label+'-config.json');save(config,cfg)
        cmd=[PY,R/'tools/bench_mtp_modes.py','--config',config,'--output',out/label,
            '--input-tokens',str(prompt),'--output-tokens',str(output),'--mode',MODES[mode][0],
            '--suffix-draft',str(MODES[mode][1]),'--verify-window','8','--mtp-window','4',
            '--workload','long','--repetitions','1','--cases',*cases,
            '--source-commit',state['source'] if engine==R else '080891d445d1f28eb244d4ce4fbce22942dc0b29']
        started=time.time();command(label,cmd)
        data=json.loads((out/label/'result.json').read_text());run=data['runs'][0]
        log=(out/label/f'engine-mtp-{MODES[mode][0]}.log').read_text()
        counters=[tuple(map(int,m)) for m in re.findall(r'exchange rotation: (\d+) blocks, (\d+) host memcpy bytes avoided',log)]
        if rotate and (not counters or counters[-1][0]==0 or 'exchange buffer rotation enabled' not in log):
            raise RuntimeError(label+': rotation not exercised')
        rec={'label':label,'mode':mode,'rotate':rotate,'started':started,'finished':time.time(),
            'engine_info':run['engine_info'],'startup_seconds':run['startup_seconds'],'cases':run['cases'],
            'prompt_sha256':data['prompt_sha256'],'rotation_counters':counters,
            'diagnostics':[x for x in log.splitlines() if any(k in x for k in
                ('resident RAM:','expert tiers:','exchange rotation:','decode timing:'))]}
        assert run['engine_info']['kv']=='fp16'
        state['records'].append(rec);save(target,state);return rec

    def compare(a,b):
        assert a['prompt_sha256']==b['prompt_sha256']
        assert a['engine_info']['expert_slots']==b['engine_info']['expert_slots']
        rows=[]
        for x,y in zip(a['cases'],b['cases']):
            rows.append({'task':x['task'],'control_tps':x['decode_tps'],'rotation_tps':y['decode_tps'],
                'gain_pct':100*(y['decode_tps']/x['decode_tps']-1),
                'first_token_difference':first_difference(x['token_ids'],y['token_ids'])})
        state.setdefault('comparisons',[]).append({'control':a['label'],'candidate':b['label'],'cases':rows})
        save(target,state);return rows

    try:
        command('source-api-fixture',[R/'build/file_expert_source_test','--rotation-gpu'])
        command('source-api-memcheck',['compute-sanitizer','--tool','memcheck','--error-exitcode','71',
                R/'build/file_expert_source_test','--rotation-gpu'])
        state['source_fixture_and_gpu_memcheck_passed']=True;save(target,state)
        old=case('default-off-old','serial',0,engine=CONTROL,prompt=8192,output=256,gate=True,cases=('coding',))
        new=case('default-off-new','serial',0,prompt=8192,output=256,gate=True,cases=('coding',))
        state['default_off_exact_tokens']=old['cases'][0]['token_ids']==new['cases'][0]['token_ids'];save(target,state)
        if not state['default_off_exact_tokens']:raise RuntimeError('Default-off output regression')
        promising=[]
        for index,mode in enumerate(MODES):
            records={}
            for rotate in ((0,1) if index%2==0 else (1,0)):
                records[rotate]=case(f'{mode}-rotate{rotate}-r1',mode,rotate)
            rows=compare(records[0],records[1])
            if any(x['first_token_difference'] is not None for x in rows):
                state.setdefault('semantic_gate_failures',[]).append(mode);save(target,state)
                # Other decoding paths are independent; preserve evidence, do not bless this one.
                continue
            if max(x['gain_pct'] for x in rows)>=3:promising.append((mode,index))
        for mode,index in promising:
            records={}
            for rotate in ((1,0) if index%2==0 else (0,1)):
                records[rotate]=case(f'{mode}-rotate{rotate}-r2',mode,rotate)
            rows=compare(records[0],records[1])
            if any(x['first_token_difference'] is not None for x in rows):
                state.setdefault('semantic_gate_failures',[]).append(mode+'-repeat')
        state['completed']=True
    except BaseException as error:state['error']=repr(error);raise
    finally:state['finished']=time.time();save(target,state)

def main():
    if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(Path(sys.argv[2]));return
    sys.path.insert(0,str(H/'fleet-downloads'))
    import run_q4_trace_1c0c2bb as base
    source=(R/'source-commit.txt').read_text().strip()
    base.wh.CUTOFF=time.time()+24*3600
    r=base.CgroupRun(D,source)
    r.s.update(shutdown_afterwards=False,operational_guard_not_user_deadline=True,
        cutoff_utc=datetime.datetime.fromtimestamp(base.wh.CUTOFF,datetime.timezone.utc).isoformat(),
        current='waiting for previously queued Q8 tests',dependency=str(PRIOR));r.save()
    lock=(H/'fleet-downloads/.rtxpro-bandwidth.lock').open('a')
    try:
        while not PRIOR.exists() or not json.loads(PRIOR.read_text()).get('finished'):
            r.check_time();time.sleep(5)
        while True:
            try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:r.check_time();time.sleep(5)
        r.cpu('component-build',['g++','-std=c++17','-pthread','-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer',
            '-I'+str(R/'include'),R/'tests/core/exchange_storage_test.cpp','-o',R/'exchange-storage-sanitized'])
        r.cpu('component-sanitizers',[R/'exchange-storage-sanitized'])
        r.cpu('configure',['cmake','-S',R,'-B',R/'build','-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
            '-DSTRATA_ENABLE_CUDA=ON','-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc',
            '-DCUDAToolkit_ROOT=/usr/local/cuda','-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG',
            '-DCMAKE_CUDA_ARCHITECTURES=120','-DSTRATA_BUILD_TESTS=OFF',
            '-DSTRATA_MMQ_KQUANTS=ON','-DSTRATA_NATIVE_EXPERTS=ON','-DSTRATA_FLEET_CUDA_TRACE=ON',
            '-DSTRATA_GGML_DIR='+str(H/'src/Strata/third_party/llama.cpp')])
        r.cpu('build',['cmake','--build',R/'build','--target','strata','exchange_storage_test','file_expert_source_test','-j','8'])
        r.s['engine_sha256']=hashlib.sha256((R/'build/strata').read_bytes()).hexdigest();r.save()
        step=r.gpu('exchange-rotation',[PY,Path(__file__),'--worker',D/'{attempt}'],{},timeout=21600)
        r.s['matrix']=str(D/step['label']/'matrix.json');r.save()
    except BaseException as error:r.finish(error);raise
    else:r.finish()

if __name__=='__main__':main()
