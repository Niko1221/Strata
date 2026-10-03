#!/usr/bin/env python3
"""Trace failed suffix-mode gates with the existing frozen engine, no new kernels.

Copy/rotation/copy repetitions record window position and width, ordered expert
IDs, and first output divergence. The existing routing dump does not include
real route weights in serving mode; no logits or state parity claim is made.
All timings are instrumented and excluded from throughput evidence.
"""
from pathlib import Path
import fcntl
import hashlib
import json
import re
import struct
import subprocess
import sys
import time

H=Path.home();R=Path(__file__).resolve().parents[1]
ENGINE=H/'src/strata-q8-exchange-rotation-1a50d913'
PY=H/'src/Strata/.venv/bin/python'
D=H/'fleet-downloads/rtxpro-q8-exchange-policy-trace-20261003'
PRIOR=H/'fleet-downloads/rtxpro-q8-fresh-edit-placement-20261003/status.json'
GATE=H/'fleet-downloads/rtxpro-q8-exchange-rotation-20261003-r2/exchange-rotation-attempt-01/matrix.json'


def save(p,j):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(j,indent=2)+'\n');tmp.replace(p)


def difference(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),
        min(len(a),len(b)) if len(a)!=len(b) else None)


def routes(path,windows):
    data=path.read_bytes();offset=0;records=[]
    while offset<len(data):
        if offset+8>len(data):raise RuntimeError('Truncated routing header')
        layer,k=struct.unpack_from('<ii',data,offset);offset+=8
        if not (0<=layer<48 and 0<k<=64 and offset+8*k<=len(data)):
            raise RuntimeError('Invalid routing record')
        records.append((layer,struct.unpack_from('<'+'i'*k,data,offset)));offset+=8*k
    expected=48*sum(t for _,t in windows)
    if len(records)<expected:raise RuntimeError('Missing decode route records')
    prefix=len(records)-expected;tail=records[prefix:];index=0;decoded=[]
    for pos,t in windows:
        layers=[]
        for layer in range(48):
            group=tail[index:index+t];index+=t
            if len(group)!=t or any(l!=layer for l,_ in group):
                raise RuntimeError('Decode routing/window alignment failed')
            layers.append([list(ids) for _,ids in group])
        decoded.append({'position':pos,'T':t,'ordered_expert_ids':layers})
    return {'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),
        'records_before_decode':prefix,'windows':decoded,
        'limitation':'Serving trace stores placeholder route weights; only ordered expert IDs are compared.'}


def worker(out):
    sys.path.insert(0,str(ENGINE/'tools'))
    from configure_rtxpro import configure
    out.mkdir(parents=True,exist_ok=False)
    gate=json.loads(GATE.read_text())
    modes=[m for m in ('ngram','mtp-ngram') if any(x==m or x==m+'-repeat' for x in gate.get('semantic_gate_failures',[]))]
    state={'started':time.time(),'instrumented_only':True,'records':[],'comparisons':[],
        'scope':'First-divergence diagnosis; no throughput claim or automatic approval.',
        'engine_sha256':hashlib.sha256((ENGINE/'build/strata').read_bytes()).hexdigest()}
    if state['engine_sha256']!=gate['engine_sha256']:raise RuntimeError('Frozen engine changed')
    target=out/'matrix.json';save(target,state)

    def run(mode,rotate,rep):
        label=f'{mode}-rotate{rotate}-r{rep}';state['current']=label;save(target,state)
        mtp='on' if mode=='mtp-ngram' else 'off'
        cfg=configure({'weights':'Q8_0','kv':'fp16','load_projection':False},ENGINE,H,73728)
        cfg['env'].update(STRATA_PLE_PREFAULT_THREADS='8',STRATA_ADAPT_NOWAIT='0',
            STRATA_EXCHANGE_ROTATE=str(rotate),STRATA_TRACE='1')
        cfg['args'][cfg['args'].index('--expert-cache')+1]=str(16192 if mtp=='on' else 16400)
        route=out/(label+'-routes.bin');cfg['args']+=['--dump-routing',str(route)]
        config=out/(label+'-config.json');save(config,cfg)
        cmd=[PY,ENGINE/'tools/bench_mtp_modes.py','--config',config,'--output',out/label,
            '--input-tokens','65536','--output-tokens','512','--mode',mtp,'--suffix-draft','3',
            '--verify-window','8','--mtp-window','4','--workload','long','--repetitions','1',
            '--cases','coding','--source-commit',gate['source']]
        with (out/(label+'.log')).open('w') as log:
            subprocess.run(list(map(str,cmd)),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
        data=json.loads((out/label/'result.json').read_text());case=data['runs'][0]['cases'][0]
        log=(out/label/f'engine-mtp-{mtp}.log').read_text()
        windows=[list(map(int,x)) for x in re.findall(r'^strata trace: window (\d+) (\d+)$',log,re.M)]
        if not windows or windows[0][0]!=65535:raise RuntimeError('Missing expected decode window trace')
        rec={'label':label,'mode':mode,'rotate':rotate,'case':case,'windows':windows,
             'routing':routes(route,windows),'prompt_sha256':data['prompt_sha256']}
        save(out/(label+'-diagnosis.json'),rec)
        state['records'].append({k:v for k,v in rec.items() if k!='routing'});save(target,state);return rec

    def compare(a,b):
        assert a['prompt_sha256']==b['prompt_sha256']
        token=difference(a['case']['token_ids'],b['case']['token_ids'])
        win=difference(a['windows'],b['windows']);route_diff=None
        for i,(x,y) in enumerate(zip(a['routing']['windows'],b['routing']['windows'])):
            if (x['position'],x['T'])!=(y['position'],y['T']):break
            for l,(xx,yy) in enumerate(zip(x['ordered_expert_ids'],y['ordered_expert_ids'])):
                if xx!=yy:route_diff={'window':i,'position':x['position'],'layer':l,'T':x['T']};break
            if route_diff:break
        row={'a':a['label'],'b':b['label'],'first_output_token_difference':token,
            'first_window_schedule_difference':win,
            'first_route_id_difference_before_schedule_diverges':route_diff}
        if win is not None:
            row['window_a']=a['windows'][win] if win<len(a['windows']) else None
            row['window_b']=b['windows'][win] if win<len(b['windows']) else None
        state['comparisons'].append(row);save(target,state)

    try:
        for mode in modes:
            a=run(mode,0,1);b=run(mode,1,1);c=run(mode,0,2)
            compare(a,b);compare(a,c)
        state['completed']=True
    except BaseException as error:state['error']=repr(error);raise
    finally:state['finished']=time.time();save(target,state)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(Path(sys.argv[2]));return
    sys.path.insert(0,str(H/'fleet-downloads'));import run_q4_trace_1c0c2bb as base
    base.wh.CUTOFF=time.time()+24*3600;r=base.CgroupRun(D,(R/'source-commit.txt').read_text().strip())
    r.s.update(shutdown_afterwards=False,current='waiting for fresh editing placement',dependency=str(PRIOR),
        operational_guard_not_user_deadline=True);r.save()
    lock=(H/'fleet-downloads/.rtxpro-bandwidth.lock').open('a')
    try:
        while not PRIOR.exists() or not json.loads(PRIOR.read_text()).get('finished'):r.check_time();time.sleep(5)
        while True:
            try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:r.check_time();time.sleep(5)
        step=r.gpu('policy-trace',[PY,Path(__file__),'--worker',D/'{attempt}'],{},timeout=10800)
        r.s['matrix']=str(D/step['label']/'matrix.json');r.save()
    except BaseException as error:r.finish(error);raise
    else:r.finish()


if __name__=='__main__':main()
