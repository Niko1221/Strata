#!/usr/bin/env python3
"""Sequential Q8 FP16 trace and CPU/PCIe miss-placement screen on llm-60."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time

H=Path.home();R=Path(__file__).resolve().parents[1]
D=Path(os.environ.get('FLEET_Q8_PLACEMENT_OUTPUT',H/'fleet-downloads/rtxpro-q8-fp16-placement-20261003'))
PRIOR=Path(os.environ.get('FLEET_Q8_PLACEMENT_AFTER',H/'fleet-downloads/rtxpro-synthetic-q8-20261003/status.json'))
CONTROL=H/'src/strata-q8-parallel-ple-e359f44'
PY=H/'src/Strata/.venv/bin/python'
MODES={'serial':('off',0),'mtp':('on',0),'ngram':('off',3),'mtp-ngram':('on',3)}


def save(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)


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
    source=(R/'source-commit.txt').read_text().strip()
    state={'started':time.time(),'source':source,'kv':'fp16','records':[],
        'engine_sha256':hashlib.sha256((R/'build/strata').read_bytes()).hexdigest(),
        'scope':'Trace first; separate serial/MTP/ngram/combined placement paths; no unsafe adaptive wait bypass.'}
    target=out/'matrix.json';save(target,state)
    env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')}

    def case(label,mode,fraction,*,engine=R,prompt=65536,output=1024,
             profiling=False,gate=False,cases=('coding','editing')):
        state['current']=label;save(target,state)
        cfg=configure({'weights':'Q8_0','kv':'fp16','load_projection':False},engine,H,73728 if not gate else 16384)
        cfg['env'].update(STRATA_PLE_PREFAULT_THREADS='8',STRATA_ADAPT_NOWAIT='0')
        option(cfg['args'],'--pcie-frac',fraction)
        option(cfg['args'],'--expert-cache',16000 if gate else 16192 if MODES[mode][0]=='on' else 16400)
        if gate:option(cfg['args'],'--adapt-swaps',0)
        if profiling:cfg['env'].update(STRATA_FLEET_PROFILE='1',STRATA_FLEET_PROFILE_SKIP='32',STRATA_FLEET_PROFILE_COUNT='16')
        config=out/(label+'-config.json');save(config,cfg)
        cmd=[PY,R/'tools/bench_mtp_modes.py','--config',config,'--output',out/label,
            '--input-tokens',str(prompt),'--output-tokens',str(output),'--mode',MODES[mode][0],
            '--suffix-draft',str(MODES[mode][1]),'--verify-window','8','--mtp-window','4',
            '--workload','long','--repetitions','1','--cases',*cases,'--source-commit',
            source if engine==R else 'e359f44851e86f25162bf7fe2f70c2e387e0672b']
        if profiling:
            cmd=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
                '--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop',
                '--output='+str(out/(label+'-timeline')),*cmd]
        rec={'label':label,'mode':mode,'pcie_frac':fraction,'profiling':profiling,'started':time.time()}
        with (out/(label+'.log')).open('w') as log:
            res=subprocess.run(list(map(str,cmd)),env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        rec.update(finished=time.time(),exit=res.returncode)
        state['records'].append(rec);save(target,state)
        if res.returncode:
            if profiling:return None
            raise RuntimeError(label+' failed')
        data=json.loads((out/label/'result.json').read_text());run=data['runs'][0]
        assert run['engine_info']['kv']=='fp16'
        log=(out/label/f'engine-mtp-{MODES[mode][0]}.log').read_text()
        rec.update(engine_info=run['engine_info'],startup_seconds=run['startup_seconds'],
            cases=run['cases'],prompt_sha256=data['prompt_sha256'],
            diagnostics=[x for x in log.splitlines() if any(s in x for s in ('decode timing:','fleet profile','resident RAM:','resident RAM mode:'))])
        save(target,state)
        if profiling:
            rec['capture_complete']='fleet profile start' in log and 'fleet profile stop' in log
            with (out/(label+'-stats.csv')).open('w') as stats:
                p=subprocess.run(['nsys','stats','--report','cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum',
                    '--format','csv',str(out/(label+'-timeline.nsys-rep'))],stdout=stats,stderr=subprocess.STDOUT,timeout=180)
            rec['stats_exit']=p.returncode;save(target,state)
        return rec

    try:
        control=case('instrumentation-control','serial',0,engine=CONTROL,prompt=8192,output=128,gate=True,cases=('coding',))
        diagnostic=case('instrumentation-candidate','serial',0,prompt=8192,output=128,gate=True,cases=('coding',))
        state['instrumentation_default_off_exact_tokens']=control['cases'][0]['token_ids']==diagnostic['cases'][0]['token_ids']
        save(target,state)
        if not state['instrumentation_default_off_exact_tokens']:raise RuntimeError('Instrumentation control differs')
        for mode in MODES:
            before=case('trace-reference-'+mode,mode,-1,output=256,cases=('coding',))
            measured=case('trace-'+mode,mode,-1,output=256,profiling=True,cases=('coding',))
            if measured:
                measured['first_token_difference_from_unprofiled']=first_difference(before['cases'][0]['token_ids'],measured['cases'][0]['token_ids'])
                measured['instrumented_only']=True;save(target,state)
        results={}
        for mode in MODES:
            for frac in (-1,0,1):
                label=f'{mode}-pcie{frac}-r1'
                rec=case(label,mode,frac);results[(mode,frac)]=rec
                ref=results[(mode,-1)]
                assert rec['prompt_sha256']==ref['prompt_sha256']
                assert rec['engine_info']['expert_slots']==ref['engine_info']['expert_slots']
                rec['first_differences_by_task']={a['task']:first_difference(a['token_ids'],b['token_ids']) for a,b in zip(ref['cases'],rec['cases'])}
                save(target,state)
        # Follow each decoding path separately; repeat its strongest promising endpoint in reverse order.
        for mode in MODES:
            ref=results[(mode,-1)]
            def gain(frac):
                return math.prod(b['decode_tps']/a['decode_tps'] for a,b in zip(ref['cases'],results[(mode,frac)]['cases']))**0.5
            winner=max((0,1),key=gain)
            any_gain=max(b['decode_tps']/a['decode_tps'] for a,b in zip(ref['cases'],results[(mode,winner)]['cases']))
            if any_gain>=1.03:
                case(f'{mode}-pcie{winner}-r2',mode,winner)
                case(f'{mode}-pcie-1-r2',mode,-1)
        state['completed']=True
    except BaseException as error:
        state['error']=repr(error);raise
    finally:
        state['finished']=time.time();save(target,state)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(Path(sys.argv[2]));return
    sys.path.insert(0,str(H/'fleet-downloads'))
    import run_q4_trace_1c0c2bb as base
    source=(R/'source-commit.txt').read_text().strip()
    base.wh.CUTOFF=time.time()+18*3600
    runner=base.CgroupRun(D,source)
    runner.s.update(shutdown_afterwards=False,operational_guard_not_user_deadline=True,
        cutoff_utc=datetime.datetime.fromtimestamp(base.wh.CUTOFF,datetime.timezone.utc).isoformat(),
        current='waiting for synthetic geometry screen',dependency=str(PRIOR))
    runner.save();lock=(H/'fleet-downloads/.rtxpro-bandwidth.lock').open('a')
    try:
        while not PRIOR.exists() or not json.loads(PRIOR.read_text()).get('finished'):
            runner.check_time();time.sleep(5)
        runner.s['prior_completed']=bool(json.loads(PRIOR.read_text()).get('completed'));runner.save()
        while True:
            try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:runner.check_time();time.sleep(5)
        runner.cpu('configure',['cmake','-S',R,'-B',R/'build','-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
            '-DSTRATA_ENABLE_CUDA=ON','-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc',
            '-DCUDAToolkit_ROOT=/usr/local/cuda','-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG',
            '-DCMAKE_CUDA_ARCHITECTURES=120','-DSTRATA_BUILD_TESTS=OFF',
            '-DSTRATA_MMQ_KQUANTS=ON','-DSTRATA_NATIVE_EXPERTS=ON','-DSTRATA_FLEET_CUDA_TRACE=ON',
            '-DSTRATA_GGML_DIR='+str(H/'src/Strata/third_party/llama.cpp')])
        runner.cpu('build',['cmake','--build',R/'build','--target','strata','-j','8'])
        runner.s['engine_sha256']=hashlib.sha256((R/'build/strata').read_bytes()).hexdigest();runner.save()
        step=runner.gpu('placement',[PY,Path(__file__),'--worker',D/'{attempt}'],{},timeout=14400)
        runner.s['matrix']=str(D/step['label']/'matrix.json');runner.save()
    except BaseException as error:runner.finish(error);raise
    else:runner.finish()


if __name__=='__main__':main()
