#!/usr/bin/env python3
"""Isolate the earlier combined-mode CPU-miss result from preceding cache history.

Configuration-only follow-up on the frozen, ownership-rotation engine. Every
arm starts fresh and runs only editing; no preceding coding request. Native
RoPE, 64K input, 1K output, FP16 KV. Private fleet GPU queue only.
"""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import subprocess
import sys
import time

H=Path.home();R=Path(__file__).resolve().parents[1]
ENGINE=H/'src/strata-q8-exchange-rotation-1a50d913'
ENGINE_SHA='d14ed6b69a1814ce4b5c08932a47d6921a55fa0aa8dea50427ccf0782d1ad997'
PY=H/'src/Strata/.venv/bin/python'
D=H/'fleet-downloads/rtxpro-q8-fresh-edit-placement-20261003'
PRIOR=H/'fleet-downloads/rtxpro-q8-model-dram-20261003/status.json'


def save(p,j):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(j,indent=2)+'\n');tmp.replace(p)


def worker(out):
    sys.path.insert(0,str(ENGINE/'tools'))
    from configure_rtxpro import configure
    out.mkdir(parents=True,exist_ok=False)
    assert hashlib.sha256((ENGINE/'build/strata').read_bytes()).hexdigest()==ENGINE_SHA
    state={'started':time.time(),'engine_source':'1a50d913bf910a1f63fbc1a0788a7083e3ca5f8c',
        'engine_sha256':ENGINE_SHA,'weights':'Q8_0','kv':'fp16','rope':'native default',
        'input_tokens':65536,'output_budget':1024,'allocated_context':73728,
        'mode':'MTP + ngram','rotation':True,'records':[],'comparisons':[],
        'scope':'Fresh engine and editing-only request per arm; no preceding coding/cache-history confound.'}
    target=out/'matrix.json';save(target,state)

    def case(label,pcie):
        state['current']=label;save(target,state)
        cfg=configure({'weights':'Q8_0','kv':'fp16','load_projection':False},ENGINE,H,73728)
        cfg['env'].update(STRATA_PLE_PREFAULT_THREADS='8',STRATA_ADAPT_NOWAIT='0',STRATA_EXCHANGE_ROTATE='1')
        cfg['args'][cfg['args'].index('--pcie-frac')+1]=str(pcie)
        cfg['args'][cfg['args'].index('--expert-cache')+1]='16192'
        config=out/(label+'-config.json');save(config,cfg)
        cmd=[PY,ENGINE/'tools/bench_mtp_modes.py','--config',config,'--output',out/label,
            '--input-tokens','65536','--output-tokens','1024','--mode','on','--suffix-draft','3',
            '--verify-window','8','--mtp-window','4','--workload','long','--repetitions','1',
            '--cases','editing','--source-commit',state['engine_source']]
        started=time.time()
        with (out/(label+'.log')).open('w') as log:
            subprocess.run(list(map(str,cmd)),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
        data=json.loads((out/label/'result.json').read_text());run=data['runs'][0]
        log=(out/label/'engine-mtp-on.log').read_text()
        if 'exchange buffer rotation enabled' not in log:raise RuntimeError('Rotation did not activate')
        case=run['cases'][0]
        if case['input_tokens']!=65536 or not case['output_budget_reached']:
            raise RuntimeError('Request did not reach the specified input/output budget')
        rec={'label':label,'pcie':pcie,'started':started,'finished':time.time(),
            'engine_info':run['engine_info'],'startup_seconds':run['startup_seconds'],
            'prompt_sha256':data['prompt_sha256'],'case':case}
        state['records'].append(rec);save(target,state);return rec

    def compare(a,b):
        assert a['prompt_sha256']==b['prompt_sha256']
        x,y=a['case'],b['case'];aa,bb=x['token_ids'],y['token_ids']
        first=next((i for i,(v,w) in enumerate(zip(aa,bb)) if v!=w),
            min(len(aa),len(bb)) if len(aa)!=len(bb) else None)
        row={'control':a['label'],'candidate':b['label'],'first_token_difference':first,
            'control_tps':x['decode_tps'],'cpu_only_tps':y['decode_tps'],
            'decode_gain_pct':100*(y['decode_tps']/x['decode_tps']-1),
            'effective_gain_pct':100*(y['effective_output_tps']/x['effective_output_tps']-1)}
        state['comparisons'].append(row);save(target,state);return row

    try:
        a=case('auto-r1',-1);b=case('cpu-r1',0);row=compare(a,b)
        if row['first_token_difference'] is None and row['decode_gain_pct']>=3:
            b=case('cpu-r2',0);a=case('auto-r2',-1);compare(a,b)
        state['exact_pairs']=all(x['first_token_difference'] is None for x in state['comparisons'])
        state['repeatable_gain']=len(state['comparisons'])==2 and state['exact_pairs'] and all(
            x['decode_gain_pct']>=3 for x in state['comparisons'])
        state['completed']=True
    except BaseException as error:state['error']=repr(error);raise
    finally:state['finished']=time.time();save(target,state)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(Path(sys.argv[2]));return
    sys.path.insert(0,str(H/'fleet-downloads'))
    import run_q4_trace_1c0c2bb as base
    base.wh.CUTOFF=time.time()+24*3600
    source=(R/'source-commit.txt').read_text().strip();r=base.CgroupRun(D,source)
    r.s.update(shutdown_afterwards=False,current='waiting for queued DRAM measurements',
        dependency=str(PRIOR),operational_guard_not_user_deadline=True,
        cutoff_utc=datetime.datetime.fromtimestamp(base.wh.CUTOFF,datetime.timezone.utc).isoformat())
    r.save();lock=(H/'fleet-downloads/.rtxpro-bandwidth.lock').open('a')
    try:
        while not PRIOR.exists() or not json.loads(PRIOR.read_text()).get('finished'):r.check_time();time.sleep(5)
        rotation=H/'fleet-downloads/rtxpro-q8-exchange-rotation-20261003-r2/exchange-rotation-attempt-01/matrix.json'
        gate=json.loads(rotation.read_text())
        if not (gate.get('completed') and gate.get('source_fixture_and_gpu_memcheck_passed') and
                gate.get('default_off_exact_tokens')) or any(
                x.startswith('mtp-ngram') for x in gate.get('semantic_gate_failures',[])):
            raise RuntimeError('Combined rotation prerequisite not accepted; placement follow-up not started')
        while True:
            try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:r.check_time();time.sleep(5)
        step=r.gpu('fresh-edit',[PY,Path(__file__),'--worker',D/'{attempt}'],{},timeout=5400)
        r.s['matrix']=str(D/step['label']/'matrix.json');r.save()
    except BaseException as error:r.finish(error);raise
    else:r.finish()


if __name__=='__main__':main()
