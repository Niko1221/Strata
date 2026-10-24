#!/usr/bin/env python3
"""Repeatable single-user benchmark using Strata's own engine and tokenizer APIs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import statistics
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
from serve.server import StrataEngine, child_env, engine_args
import strata_tokenizer as ST
import psutil

def tokenizer(path):
    p=Path(path);v=json.loads((p/'vocab.json').read_text());tokens=[None]*len(v)
    for text,i in v.items():tokens[i]=text
    return ST.Tokenizer(tokens,(p/'merges.txt').read_text().split('\n'),json.loads((p/'token_type.json').read_text()))

def prompt_ids(tok,body,n,rep):
    head=tok.encode(f'<|im_start|>user\n[local-tune-length-{n}-repeat-{rep}] Read the reference below, then write a detailed technical explanation with examples of at least 400 words.\nREFERENCE:\n',parse_special=True)
    tail=tok.encode('\nEND REFERENCE. Explain its main engineering ideas and tradeoffs in detail.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',parse_special=True)
    need=n-len(head)-len(tail)
    assert need>0 and body
    ids=head+(body*((need+len(body)-1)//len(body)))[:need]+tail
    assert len(ids)==n
    return ids

def describe(values):
    assert values and all(x is not None for x in values)
    med=statistics.median(values)
    return {'median':med,'min':min(values),'max':max(values),
            'relative_range':(max(values)-min(values))/med if med else None,'n':len(values)}

def log_stats(text):
    result={}
    x=re.findall(r'EXL3 decode: ([\d.]+)% of the routes on the VRAM tier',text)
    if x:result['vram_route_hit_pct']=float(x[-1])
    x=re.findall(r'EXL3 miss share:.*',text)
    if x:result['exl3_miss_share']=x[-1]
    x=re.findall(r'ram_blobs[=: ]+(\d+)',text)
    result['ram_blobs']=int(x[-1]) if x else None
    result['ram_blobs_note']='Not emitted by this EXL3 engine' if not x else None
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True)
    ap.add_argument('--label',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--lengths',default='512,2048,8192,32768,longest')
    ap.add_argument('--runs',type=int,default=3)
    ap.add_argument('--max-new',type=int,default=128)
    ap.add_argument('--sweep',action='append',help='Per-request stock calibration key:values; repeat for separate one-knob phases')
    ap.add_argument('--warmup',type=int,default=1)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--stock-calibrate',action='store_true',help="Run Strata's Session/sweep draft-floor calibration without resetting the user's workers")
    ap.add_argument('--legacy-prompts',action='store_true',help='Use the earlier three comparison questions in stock calibration')
    a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text());tok=tokenizer(cfg['tokenizer'])
    corpus=ROOT/'tools/local_tuning/corpus.txt'
    if not corpus.exists():
        raise SystemExit('Missing frozen benchmark corpus: '+str(corpus))
    body=tok.encode(corpus.read_text(),parse_special=False)
    argv=cfg['args'];ctx=int(argv[argv.index('--max-context')+1]) if '--max-context' in argv else 65536
    longest=ctx-a.max_new-16
    lengths=[longest if x=='longest' else int(x) for x in a.lengths.split(',')]
    assert a.runs>=1 and max(lengths)+a.max_new+16<=ctx
    fixtures={(n,r):prompt_ids(tok,body,n,r) for n in lengths for r in range(a.runs)}
    if a.self_test:
        for (n,r),ids in fixtures.items():assert len(ids)==n and ids==prompt_ids(tok,body,n,r)
        assert len({tuple(ids[:32]) for ids in fixtures.values()})==len(fixtures)
        print('Prompt length, reproducibility and unique-prefix self-checks passed.');return
    for p in psutil.process_iter(['name']):
        if p.info['name']=='strata':raise SystemExit('Another Strata engine is running; stop it before tuning.')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise SystemExit('Refusing to overwrite '+str(out)+'; choose a new --out name.')
    cfg['log']=str(out.with_suffix('.engine.log'))
    Path(cfg['log']).write_text('')
    result={'label':a.label,'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'config':cfg,
            'protocol':{'lengths':lengths,'runs':a.runs,'max_new':a.max_new,'context':ctx,'temperature':0,
                        'sampling':'greedy; no speculative forced acceptance; configured suffix drafting retained',
                        'sweeps':a.sweep,
                        'ttft_definition':'elapsed time to first token from StrataEngine.generate; includes request work, excludes engine load',
                        'corpus_sha256':hashlib.sha256(corpus.read_bytes()).hexdigest(),
                        'fixture_sha256':{f'{n}:{r}':hashlib.sha256(json.dumps(ids).encode()).hexdigest() for (n,r),ids in fixtures.items()}},
            'runs':[],'memory_samples':[],'success':False}
    with open(cfg['exe'],'rb') as f:
        result['exe_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    result['harness_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    result['source_diff_sha256']=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=ROOT)).hexdigest()
    done=threading.Event();engine=None
    def save():out.write_text(json.dumps(result,indent=2)+'\n')
    def monitor():
        compilers={'rustc','cargo','cc1','cc1plus','nvcc','cicc','ptxas','ninja'}
        previous_cpu={}
        psutil.cpu_percent(interval=None)
        while not done.is_set():
            row={'time':time.time(),'available_ram_gib':psutil.virtual_memory().available/2**30,
                 'swap_used_gib':psutil.swap_memory().used/2**30,
                 'swap_in_bytes':psutil.swap_memory().sin,'swap_out_bytes':psutil.swap_memory().sout}
            row['system_cpu_busy_pct']=psutil.cpu_percent(interval=None)
            if engine is not None and engine.proc is not None:
                try:row['engine_rss_gib']=psutil.Process(engine.proc.pid).memory_info().rss/2**30
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            row['concurrent_compilers']=[]
            for proc in psutil.process_iter(['pid','name','cpu_times']):
                if proc.info['name'] in compilers:
                    info={'pid':proc.pid,'name':proc.info['name']}
                    ct=proc.info.get('cpu_times')
                    if ct is not None:
                        seconds=ct.user+ct.system
                        prev=previous_cpu.get(proc.pid)
                        if prev:info['cpu_pct']=100*(seconds-prev[1])/max(0.01,row['time']-prev[0])
                        previous_cpu[proc.pid]=(row['time'],seconds)
                    row['concurrent_compilers'].append(info)
            try:
                vals=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu,pcie.link.gen.current,pcie.link.width.current,memory.free,memory.reserved','--format=csv,noheader,nounits'],text=True,timeout=5).strip().split(',')
                row.update(zip(['vram_mib','gpu_util_pct','gpu_watts','gpu_temp_c','pcie_gen','pcie_width','free_vram_mib','driver_reserved_vram_mib'],map(float,vals)))
            except Exception:pass
            result['memory_samples'].append(row)
            if row['available_ram_gib']<1.25:
                result['memory_guard_triggered']=True;os.kill(os.getpid(),signal.SIGINT);return
            done.wait(2)
    worker=threading.Thread(target=monitor,daemon=True);worker.start()
    try:
        env=child_env(cfg);env.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
        start=time.perf_counter()
        engine=StrataEngine(cfg['exe'],engine_args(cfg),cwd=cfg.get('cwd'),log=cfg['log'],env=env)
        result['engine_load_seconds']=time.perf_counter()-start
        if a.stock_calibrate:
            from calibrate import Session,sweep,chat_ids,PROMPTS
            if a.legacy_prompts:
                from ab_engine import PROMPTS
            session=Session(engine,[chat_ids(tok,p) for p in PROMPTS])
            session.warm_up(1)
            result['stock_calibration']={}
            result['calibration_prompts']=list(PROMPTS)
            for phase in a.sweep or ['spec_min_p:0.5,0.6,0.7,0.8']:
                key,values=phase.split(':',1);values=[float(x) for x in values.split(',')]
                assert key in ('pcie_frac','spec_min_p','aux_cpus')
                result['stock_calibration'][key]=sweep(session,values,{},key,key,lambda s:print(s,flush=True))
                save()
            result['stock_calibration_note']='Stock Session/sweep, three rounds, current worker and tier config retained; automatic product-default resets and unrelated GGUF/multi-GPU sweeps excluded.'
            result['success']=True
            return
        def run(n,rep,tune=None,warm=False):
            ids=prompt_ids(tok,body,n,1000+rep) if warm else fixtures[n,rep]
            sampling={'temperature':0,'strata_checkpoint':False}
            if tune:sampling['strata_tune']=tune
            before=Path(cfg['log']).stat().st_size
            wall_started=time.time();t=time.perf_counter();first=None;generated=[]
            for token in engine.generate(ids,a.max_new if not warm else 32,sampling,threading.Event()):
                if token is not None:
                    if first is None:first=time.perf_counter()-t
                    generated.append(int(token))
            elapsed=time.perf_counter()-t
            metrics=dict(engine.last or {})
            if not generated or metrics.get('error'):raise RuntimeError('Generation failed: '+str(metrics))
            if metrics.get('prompt_read') is not None:assert metrics['prompt_read']==len(ids),metrics
            with open(cfg['log'],'r') as f:f.seek(before);delta=f.read()
            prompt_ms=metrics.get('prompt_ms') or 0;decode_ms=metrics.get('decode_ms') or 0
            row={'prompt_tokens':n,'repeat':rep,'tune':tune,'warmup':warm,'ttft_s':first,'wall_s':elapsed,
                 'started_at_unix':wall_started,'finished_at_unix':time.time(),
                 'prefill_tps':n*1000/prompt_ms if prompt_ms else None,
                 'decode_tps':len(generated)*1000/decode_ms if decode_ms else None,
                 'generated_tokens':len(generated),'output_ids':generated,'response':tok.decode(generated),
                 'engine_metrics':metrics,**log_stats(delta)}
            result['runs'].append(row);save()
            print(json.dumps({k:v for k,v in row.items() if k not in ['output_ids','response','engine_metrics']}),flush=True)
        for i in range(a.warmup):run(lengths[0],i,warm=True)
        if a.sweep:
            for phase in a.sweep:
                key,values=phase.split(':',1);values=[float(x) for x in values.split(',')]
                assert key in ('pcie_frac','spec_min_p','aux_cpus')
                for rep in range(a.runs):
                    for value in values if rep%2==0 else values[::-1]:
                        for n in lengths:run(n,rep,{key:value})
        else:
            for n in lengths:
                for rep in range(a.runs):run(n,rep)
        result['summary']={}
        for row in result['runs']:
            if row['warmup']:continue
            k=str(row['prompt_tokens'])+(':'+json.dumps(row['tune'],sort_keys=True) if row['tune'] else '')
            result['summary'].setdefault(k,[]).append(row)
        result['summary']={k:{m:describe([r[m] for r in rows]) for m in ['prefill_tps','ttft_s','decode_tps']}
                           for k,rows in result['summary'].items()}
        result['success']=True
    except BaseException as e:
        result['error']=repr(e);raise
    finally:
        if engine is not None:engine.close()
        done.set();worker.join(timeout=6)
        # The engine's stderr pump can lag DONE; attach diagnostics only after it drains.
        blocks=re.split(r'(?=strata serve: prompt \d+ tokens =)',Path(cfg['log']).read_text())[1:]
        if len(blocks)==len(result['runs']):
            for row,block in zip(result['runs'],blocks):row.update(log_stats(block))
        elif not a.stock_calibrate:result['log_telemetry_error']=f'{len(blocks)} request blocks for {len(result["runs"])} measurements'
        free=re.findall(r'(\d+) MiB of VRAM free',Path(cfg['log']).read_text())
        if free:result['minimum_logged_free_vram_mib']=min(map(int,free))
        if result['memory_samples']:
            result['peak_vram_mib']=max(r.get('vram_mib',0) for r in result['memory_samples'])
            result['minimum_available_ram_gib']=min(r['available_ram_gib'] for r in result['memory_samples'])
            result['compiler_activity_detected']=any(r.get('concurrent_compilers') for r in result['memory_samples'])
            free=[r['free_vram_mib'] for r in result['memory_samples'] if 'free_vram_mib' in r]
            if free:result['minimum_observed_free_vram_mib']=min(free)
            rss=[r['engine_rss_gib'] for r in result['memory_samples'] if 'engine_rss_gib' in r]
            if rss:result['peak_engine_rss_gib']=max(rss)
            for row in result['runs']:
                if 'started_at_unix' not in row:continue
                samples=[s for s in result['memory_samples'] if row['started_at_unix']<=s['time']<=row['finished_at_unix']]
                row['compiler_process_seen']=any(s.get('concurrent_compilers') for s in samples)
                row['compiler_cpu_pct_max']=max([sum(c.get('cpu_pct',0) for c in s.get('concurrent_compilers',[])) for s in samples] or [0])
        result['finished_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z');save()

if __name__=='__main__':main()
