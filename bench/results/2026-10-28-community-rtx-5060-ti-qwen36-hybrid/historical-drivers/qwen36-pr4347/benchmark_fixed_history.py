"""Twenty prospective fixed-history runs; restore daily model and downloads."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import psutil
from build_pair import ARMS, MODEL, OUT, REPO, ROOT, digest, stamp

sys.path.insert(0, str(REPO/'reports/tobeo-20261022/qwen36-context-curve'))
import run_curve as protocol

RUN = OUT/'fixed-history-01'
RUN.mkdir(exist_ok=False)
VALID = OUT/'validation-next/run-01'
CUDA = ROOT/'toolchains/cuda-13.2.0'
record = {'started': stamp(), 'phase': 'preparing', 'passed': False,
          'plan': 'fixed-history-plan.md', 'rows': [], 'fixtures': {},
          'downloads_suspended': [], 'service_transition': {}}


def save():
    tmp = RUN/'results.tmp'
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(RUN/'results.json')


def environment(keep=False):
    e = {k:v for k,v in os.environ.items() if not k.startswith('STRATA_')}
    e.update(STRATA_Q8_PACKED='1', STRATA_MMQ_KQ='1', STRATA_Q36_FIXED_SLOTS='1',
             CUDA_VISIBLE_DEVICES='0', CUDA_MODULE_LOADING='EAGER')
    if keep: e['STRATA_Q36_KEEP_PACKED']='1'
    e['PATH'] = os.pathsep.join([str(CUDA/'bin/x64'), str(CUDA/'bin'),
                                str(ARMS['pr4347']['build']), e.get('PATH','')])
    return e


def pause_downloads():
    for p in psutil.process_iter(['name','cmdline','create_time']):
        if 'python' not in (p.info['name'] or '').lower(): continue
        if not any(Path(arg).name=='download_deepseek.py' for arg in (p.info['cmdline'] or [])): continue
        if p.status()==psutil.STATUS_STOPPED: raise RuntimeError('Downloader already stopped externally')
        item={'pid':p.pid, 'created':p.create_time()}
        p.suspend()
        record['downloads_suspended'].append(item)
        save()
        if p.status()!=psutil.STATUS_STOPPED: raise RuntimeError('Download suspension failed')
    record['pause_at']=stamp();save();time.sleep(5)


def resume_downloads():
    rows=[]
    for item in record['downloads_suspended']:
        try:
            p=psutil.Process(item['pid'])
            if p.create_time()!=item['created']: raise RuntimeError('PID reused')
            p.resume()
            rows.append({'pid':p.pid,'resumed':p.status()!=psutil.STATUS_STOPPED})
        except psutil.NoSuchProcess: rows.append({'pid':item['pid'],'exited':True})
        except Exception as exc: rows.append({'pid':item['pid'],'error':repr(exc)})
    record['download_resumption']=rows;record['resume_at']=stamp();save()
    if any(row.get('error') or row.get('resumed') is False for row in rows):
        raise RuntimeError('Downloader restoration failed')


def bandwidth(label):
    exe=REPO/'reports/tobeo-20261024/pcie-root-cause/transfer_probe.exe'
    run=subprocess.run([str(exe)],env=environment(),capture_output=True,text=True,timeout=90)
    (RUN/(label+'.log')).write_text(run.stdout+'\n'+run.stderr,encoding='utf-8')
    match=re.search(r'"test":"pinned_H2D"[^\n]*"event_GBps":([0-9.]+)',run.stdout)
    result={'exit_code':run.returncode,'h2d_GBps':float(match.group(1)) if match else None}
    record[label]=result;save()
    if run.returncode or result['h2d_GBps'] is None or result['h2d_GBps']<15:
        raise RuntimeError('Transfer bandwidth gate failed')
    return result['h2d_GBps']


def run_one(pair,case,arm,exe):
    keep=arm=='A';tag=f'pair-{pair}-{case}-{arm}'
    fixture=record['fixtures'][case]
    for suffix in ['tokens','forced']:
        if digest(RUN/(case+'.'+suffix))!=fixture[suffix+'_file_sha256']:
            raise RuntimeError('Fixture changed')
    if digest(exe)!=record['adapter_sha256']:raise RuntimeError('Executable changed')
    args=[str(exe),'--native',str(MODEL),'--max-context','32768','--prefill','2048',
          '--pool-workers','7','--mtp','0','--tokens-file',str(RUN/(case+'.tokens')),
          '--force-file',str(RUN/(case+'.forced'))]
    row={'pair':pair,'case':case,'arm':arm,'started':stamp()}
    record['rows'].append(row);save();print('START '+tag,flush=True)
    logfile=RUN/(tag+'.log')
    with logfile.open('w',encoding='utf-8') as log:
        child=subprocess.Popen(args,env=environment(keep),stdout=log,stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW)
        try:code=child.wait(timeout=900)
        except BaseException:child.kill();child.wait();raise
    text=logfile.read_text(encoding='utf-8',errors='replace')
    slots=[int(x) for x in re.findall(r'hybrid experts: (\d+) of 256 per layer',text)]
    replacements=[int(x) for x in re.findall(r'(\d+) slot replacements',text)]
    gen=re.search(r'^gen: ([0-9 ]+)$',text,re.M)
    ids=[int(x) for x in gen.group(1).split()] if gen else []
    timing=re.search(r'prompt (\d+) tokens in ([\d.]+) ms \([\d.]+ tok/s\), (\d+) generated in ([\d.]+) ms',text)
    released='given back to the expert slots' in text
    row.update(exit_code=code,slots=slots,replacements=replacements,released=released,
               output_ids=ids,output_matches_prior=ids==fixture['expected_argmax'][arm],finished=stamp())
    if timing:
        row.update(prompt_tokens=int(timing[1]),prefill_ms=float(timing[2]),
                   generated_argmax=int(timing[3]),generation_ms=float(timing[4]),decode_steps=96)
        row['request_ms']=row['prefill_ms']+row['generation_ms']
    row['passed']=bool(code==0 and slots==[63 if keep else 89] and replacements and
                       not any(replacements) and released==(not keep) and row['output_matches_prior'] and
                       timing and row['prompt_tokens']==fixture['prompt_tokens'] and row['generated_argmax']==97)
    save();print(json.dumps(row|{'output_ids':'omitted'}),flush=True)
    if not row['passed']:raise RuntimeError('Per-run protocol gate failed: '+tag)


save();unloaded=False
try:
    validation=json.loads((OUT/'validation-next-summary.json').read_text(encoding='utf-8'))
    assert validation['passed'] and validation['numeric_passed'] and validation['task_passed']
    record['validation_sha256']=digest(OUT/'validation-next-summary.json')
    record['plan_sha256']=digest(OUT/'fixed-history-plan.md')
    record['script_sha256']=digest(__file__)
    adapter=json.loads((OUT/'validation-next/adapter-build.json').read_text(encoding='utf-8'))
    exe=Path(adapter['arms']['pr4347']['exe'])
    record['adapter_sha256']=digest(exe)
    assert record['adapter_sha256']==adapter['arms']['pr4347']['sha256']
    for path,sha in adapter['arms']['pr4347']['libraries'].items():assert digest(path)==sha
    config=REPO/'strata-iq3_s.json'
    record['daily_config_sha256_before']=digest(config)
    for case in ['r0-l1024','r0-l16384']:
        fixture={'expected_argmax':{}}
        for suffix in ['tokens','forced']:
            data=(VALID/(case+'.'+suffix)).read_bytes()
            (RUN/(case+'.'+suffix)).write_bytes(data)
            fixture[suffix+'_file_sha256']=digest(RUN/(case+'.'+suffix))
        fixture['prompt_tokens']=len((RUN/(case+'.tokens')).read_text().split(','))
        assert len((RUN/(case+'.forced')).read_text().split(','))==96
        for arm in ['A','B']:
            log=(VALID/(case+'-'+arm+'.log')).read_text(encoding='utf-8')
            fixture['expected_argmax'][arm]=[int(x) for x in re.search(r'^gen: ([0-9 ]+)$',log,re.M)[1].split()]
            assert len(fixture['expected_argmax'][arm])==97
        record['fixtures'][case]=fixture
    save()
    for name,arm in ARMS.items():
        assert subprocess.check_output(['git','-C',str(arm['source']),'rev-parse','HEAD'],text=True).strip()==arm['revision']
        assert not subprocess.check_output(['git','-C',str(arm['source']),'status','--porcelain','--untracked-files=no'],text=True).strip()
    state=record['service_transition'];state['health_before']=protocol.request('/health');save()
    state['unload']=protocol.request('/unload?wait=30',post=True)
    if state['unload'].get('status') not in ('unloaded','not loaded'):raise RuntimeError('Daily service busy')
    unloaded=True;save();pause_downloads()
    before=bandwidth('bandwidth-before')
    record['phase']='benchmark';save()
    for pair in range(5):
        cases=['r0-l1024','r0-l16384'] if pair%2==0 else ['r0-l16384','r0-l1024']
        for case in cases:
            for arm in (['A','B'] if pair%2==0 else ['B','A']):run_one(pair,case,arm,exe)
    after=bandwidth('bandwidth-after')
    record['bandwidth_relative_drift']=abs(after/before-1)
    if record['bandwidth_relative_drift']>0.15:raise RuntimeError('Transfer bandwidth drift exceeded 15%')
    record['passed']=True;record['phase']='complete';save()
except BaseException as exc:
    record['phase']='failed';record['error']=repr(exc);save();raise
finally:
    errors=[]
    try:resume_downloads()
    except BaseException as exc:errors.append(repr(exc))
    try:
        state=record['service_transition']
        if unloaded and state['health_before'].get('loaded'):
            state['load']=protocol.request('/load',post=True)
            state['health_after']=protocol.request('/health')
            if not state['health_after'].get('loaded'):raise RuntimeError('Daily service restore failed')
        if 'daily_config_sha256_before' in record:
            record['daily_config_sha256_after']=digest(REPO/'strata-iq3_s.json')
            if record['daily_config_sha256_after']!=record['daily_config_sha256_before']:raise RuntimeError('Daily config changed')
    except BaseException as exc:errors.append(repr(exc))
    if errors:record['restore_errors']=errors;record['passed']=False
    record['finished']=stamp();save()
    if errors:raise RuntimeError(str(errors))
