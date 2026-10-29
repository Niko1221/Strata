"""PR4335 numeric/task gates, then five fixed-history performance pairs."""
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
import psutil

OUT=Path(__file__).resolve().parent
PREVIOUS=OUT.parent/'qwen36-pr4347'
sys.path.insert(0,str(PREVIOUS))
from build_pair import ARMS,MODEL,REPO,ROOT,digest,stamp
from importlib.util import spec_from_file_location,module_from_spec
sys.path.insert(0,str(REPO/'reports/tobeo-20261022/qwen36-context-curve'))
import run_curve as protocol
sys.path.insert(0,str(ARMS['main']['source']/'tools/qwen35'))
from q36_harness import Eng
sys.path.insert(0,str(ARMS['main']['source']/'tools'))
from strata_tokenizer import Tokenizer

RUN=OUT/'execution-01';RUN.mkdir(exist_ok=False)
PR_SOURCE=ROOT/'src-pr4335-bdf0bce2'
PR_BUILD=ROOT/'build-pr4335-bdf0bce2-cuda132-sm120'
CUDA=ROOT/'toolchains/cuda-13.2.0'
REVISION='bdf0bce2e5520680524efd3a5cf79db984c3902c'
record={'started':stamp(),'phase':'preparing','passed':False,'runs':[],
        'numeric':{},'tasks':{},'rows':[],'fixtures':{},'downloads_suspended':[],
        'service_transition':{},'expected_slots':{}}


def save():
    tmp=RUN/'results.tmp'
    tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(RUN/'results.json')


def environment(source='pr',chunk=2048):
    e={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')}
    e.update(STRATA_Q8_PACKED='1',STRATA_MMQ_KQ='1',STRATA_Q36_FIXED_SLOTS='1',
             STRATA_Q36_PLACE_DEBUG='1',CUDA_VISIBLE_DEVICES='0',CUDA_MODULE_LOADING='EAGER')
    build=PR_BUILD if source=='pr' else ARMS['main']['build']
    e['PATH']=os.pathsep.join([str(CUDA/'bin/x64'),str(CUDA/'bin'),str(build),e.get('PATH','')])
    return e


def arguments(chunk):
    args=['--native',str(MODEL),'--max-context','32768','--pool-workers','7','--mtp','0']
    if chunk:args+=['--prefill',str(chunk)]
    return args


def definitions(path,names,namespace):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions}==set(names)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)


# Reuse only pure function definitions, never execute a previous experiment.
scorers=dict(np=np,re=re,subprocess=subprocess,sys=sys,OUT=PREVIOUS,json=json)
definitions(PREVIOUS/'validate_next.py',['compare','judge'],scorers)
compare=scorers['compare'];judge=scorers['judge']
definitions(PREVIOUS/'benchmark_fixed_history.py',
            ['pause_downloads','resume_downloads','bandwidth'],globals())


def parse_log(path):
    text=path.read_text(encoding='utf-8',errors='replace')
    slots=[int(x) for x in re.findall(r'hybrid experts: (\d+) of 256 per layer',text)]
    chunks=[int(x) for x in re.findall(r'context 32768 cells, prompt chunk (\d+)',text)]
    swaps=[int(x) for x in re.findall(r'(\d+) slot replacements',text)]
    assert len(slots)==len(chunks)==1,'Unexpected placement/chunk log'
    return {'slots':slots[0],'chunk':chunks[0],'replacement_counts':swaps,
            'released':'given back to the expert slots' in text,
            'rule_lines':[line for line in text.splitlines() if 'chunk rule:' in line]}


def native(label,source,chunk,case,dump=False,pair=None):
    exe=executables[source]
    assert digest(exe)==record['binaries'][source]['sha256']
    for suffix in ['tokens','forced']:
        assert digest(RUN/(case+'.'+suffix))==record['fixtures'][case][suffix+'_file_sha256']
    command=[str(exe),*arguments(chunk),'--tokens-file',str(RUN/(case+'.tokens')),
             '--force-file',str(RUN/(case+'.forced'))]
    if dump:command+=['--dump-full-logits',str(RUN/(label+'.f32'))]
    row={'label':label,'source':source,'requested_chunk':chunk,'case':case,'pair':pair,'started':stamp()}
    record['runs' if dump else 'rows'].append(row);save();print('START '+label,flush=True)
    log=RUN/(label+'.log')
    with log.open('w',encoding='utf-8') as output:
        child=subprocess.Popen(command,env=environment(source,chunk),stdout=output,stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW)
        try:code=child.wait(timeout=900)
        except BaseException:child.kill();child.wait();raise
    text=log.read_text(encoding='utf-8',errors='replace')
    row.update(exit_code=code,**parse_log(log),finished=stamp())
    gen=re.search(r'^gen: ([0-9 ]+)$',text,re.M)
    row['output_ids']=[int(x) for x in gen[1].split()] if gen else []
    if dump:row['dump_bytes']=(RUN/(label+'.f32')).stat().st_size
    else:
        match=re.search(r'prompt (\d+) tokens in ([\d.]+) ms \([\d.]+ tok/s\), (\d+) generated in ([\d.]+) ms',text)
        assert match,'Timing line missing'
        row.update(prompt_tokens=int(match[1]),prefill_ms=float(match[2]),
                   generated_argmax=int(match[3]),generation_ms=float(match[4]),decode_steps=96)
        row['request_ms']=row['prefill_ms']+row['generation_ms']
        row['output_matches_numeric']=row['output_ids']==expected[(case,chunk)]
    valid=(code==0 and row['chunk'] in [2048,4096] and (not chunk or row['chunk']==chunk)
           and not row['released'] and row['replacement_counts'] and not any(row['replacement_counts'])
           and len(row['output_ids'])==97)
    if dump:valid=valid and row['dump_bytes']==97*248320*4
    else:valid=(valid and row['output_matches_numeric'] and row['slots']==record['expected_slots'][str(chunk)]
                and row['generated_argmax']==97 and row['prompt_tokens']==record['fixtures'][case]['prompt_tokens'])
    row['passed']=bool(valid);save()
    if not row['passed']:raise RuntimeError('Native protocol gate failed: '+label)
    return row


def task_fixtures(tok):
    manifest=json.loads((PREVIOUS/'validation-next/run-01/task-manifest.json').read_text(encoding='utf-8'))
    def chat_ids(name,text):
        return tok.encode(f'<|im_start|>user\nCASE_{name}\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',parse_special=True)
    body=tok.encode(protocol.ledger(20261029));tasks=[]
    for task in manifest:
        task=dict(task)
        if task['name'].startswith('retrieval-'):
            length=int(task['name'].split('-')[1])
            prefix='The unique access code is NOVA-731. Remember it while reading this unrelated synthetic ledger.\n'
            suffix='\nReturn only JSON with the field "code" equal to the unique access code stated at the beginning.'
            excerpt=body[:length-len(chat_ids(task['name'],prefix+suffix))]
            while '\ufffd' in tok.decode(excerpt):excerpt=excerpt[:-1]
            ids=chat_ids(task['name'],prefix+tok.decode(excerpt)+suffix)
        else:ids=chat_ids(task['name'],task['text'])
        assert protocol.tokens_hash(ids)==task['input_sha256'] and len(ids)==task['prompt_tokens']
        task['ids']=ids;tasks.append(task)
    return tasks


def task_checks():
    tok=Tokenizer.from_gguf(MODEL);tasks=task_fixtures(tok)
    for chunk in [2048,4096]:
        engine=None;rows=[];record['tasks'][str(chunk)]=rows;save()
        log=RUN/f'tasks-{chunk}.log'
        try:
            engine=Eng(str(PR_BUILD/'strata-q36.exe'),arguments(chunk),env=environment(),log=str(log))
            placement=parse_log(log)
            assert placement['chunk']==chunk and placement['slots']==record['expected_slots'][str(chunk)]
            for task in tasks:
                result=protocol.measured_request(engine,task['ids'],task['cap'])
                output=tok.decode(result['output_ids'])
                row={'name':task['name'],'prompt_tokens':len(task['ids']),'output':output,
                     'output_ids':result['output_ids'],'reused_tokens':result['reused_tokens'],
                     'finish':result['finish'],'judgment':judge(task,output)}
                assert row['reused_tokens']==0
                rows.append(row);save();print('TASK',chunk,task['name'],row['judgment']['passed'],flush=True)
        finally:
            if engine:engine.close()
        placement=parse_log(log)
        assert placement['replacement_counts'] and not any(placement['replacement_counts'])
    record['task_passed']=all(len(rows)==12 and all(row['judgment']['passed'] for row in rows)
                              for rows in record['tasks'].values());save()
    if not record['task_passed']:raise RuntimeError('Task gate failed')


expected={};executables={};unloaded=False;save()
try:
    build=json.loads((OUT/'build-summary.json').read_text(encoding='utf-8'))
    adapters={source:json.loads(path.read_text(encoding='utf-8')) for source,path in {
        'main':PREVIOUS/'validation-next/adapter-build.json',
        'pr':OUT/'validation-next/adapter-build.json'}.items()}
    assert build['passed'] and all(v['passed'] for v in adapters.values())
    assert subprocess.check_output(['git','-C',str(PR_SOURCE),'rev-parse','HEAD'],text=True).strip()==REVISION
    assert not subprocess.check_output(['git','-C',str(PR_SOURCE),'status','--porcelain','--untracked-files=no'],text=True).strip()
    record['binaries']={}
    for source,key in [('main','main'),('pr','pr4335')]:
        info=adapters[source]['arms'][key];exe=Path(info['exe']);executables[source]=exe
        assert digest(exe)==info['sha256']
        for path,sha in info['libraries'].items():assert digest(path)==sha
        record['binaries'][source]={'sha256':info['sha256'],'original_exe_unchanged':info['original_exe_unchanged']}
        evidence=build if source=='pr' else json.loads((PREVIOUS/'build-summary.json').read_text(encoding='utf-8'))
        actual_build=PR_BUILD if source=='pr' else ARMS['main']['build']
        for filename,sha in evidence['arms'][key]['binaries'].items():assert digest(actual_build/filename)==sha
    record['source_revision']=REVISION
    record['plan_sha256']=digest(OUT/'experiment-plan.md');record['runner_sha256']=digest(__file__)
    record['helper_sha256']={p.name:digest(p) for p in [PREVIOUS/'validate_next.py',PREVIOUS/'benchmark_fixed_history.py']}
    record['daily_config_sha256_before']=digest(REPO/'strata-iq3_s.json')
    for case in ['r0-l1024','r0-l16384']:
        fixture={}
        for suffix in ['tokens','forced']:
            (RUN/(case+'.'+suffix)).write_bytes((PREVIOUS/'validation-next/run-01'/(case+'.'+suffix)).read_bytes())
            fixture[suffix+'_file_sha256']=digest(RUN/(case+'.'+suffix))
        fixture['prompt_tokens']=len((RUN/(case+'.tokens')).read_text().split(','))
        record['fixtures'][case]=fixture
    save();state=record['service_transition'];state['health_before']=protocol.request('/health')
    state['unload']=protocol.request('/unload?wait=30',post=True)
    assert state['unload'].get('status') in ('unloaded','not loaded')
    unloaded=True;record['phase']='numeric';save()
    for case in record['fixtures']:
        runs={}
        for label,source,chunk in [('main2048','main',2048),('main4096','main',4096),
                                   ('pr2048','pr',2048),('pr4096','pr',4096),('prauto','pr',0)]:
            runs[label]=native(case+'-'+label,source,chunk,case,dump=True)
        actual=runs['prauto']['chunk']
        assert record.setdefault('auto_chunk',actual)==actual,'Automatic selection drifted at the same configured context'
        results={label:compare(RUN/(case+'-'+a+'.f32'),RUN/(case+'-'+b+'.f32')) for label,a,b in
                 [('explicit2048','main2048','pr2048'),('explicit4096','main4096','pr4096'),
                  ('auto_selected','main'+str(actual),'prauto'),('cross_chunk','pr2048','pr4096')]}
        record['numeric'][case]=results;save()
        assert all(results[k]['bitwise_identical'] and results[k]['finite'] for k in ['explicit2048','explicit4096','auto_selected'])
        assert results['cross_chunk']['finite']
        for chunk in [2048,4096]:
            row=runs['pr'+str(chunk)];expected[(case,chunk)]=row['output_ids']
            previous=record['expected_slots'].setdefault(str(chunk),row['slots'])
            assert previous==row['slots']
    record['numeric_passed']=True;record['phase']='tasks';save();task_checks()
    pause_downloads();before=bandwidth('bandwidth-before')
    record['phase']='benchmark';save()
    for pair in range(5):
        cases=list(record['fixtures'])
        if pair%2:cases.reverse()
        for case in cases:
            for chunk in ([2048,4096] if pair%2==0 else [4096,2048]):
                native(f'pair-{pair}-{case}-{chunk}','pr',chunk,case,pair=pair)
    after=bandwidth('bandwidth-after');record['bandwidth_relative_drift']=abs(after/before-1)
    assert record['bandwidth_relative_drift']<=0.15
    record['passed']=True;record['phase']='complete';save()
except BaseException as exc:
    record['error']=repr(exc);record['phase']='failed';save();raise
finally:
    errors=[]
    try:resume_downloads()
    except BaseException as exc:errors.append(repr(exc))
    try:
        state=record['service_transition']
        if unloaded and state['health_before'].get('loaded'):
            state['load']=protocol.request('/load',post=True);state['health_after']=protocol.request('/health')
            assert state['health_after']['loaded']
        if 'daily_config_sha256_before' in record:
            record['daily_config_sha256_after']=digest(REPO/'strata-iq3_s.json')
            assert record['daily_config_sha256_after']==record['daily_config_sha256_before']
    except BaseException as exc:errors.append(repr(exc))
    if errors:record['restore_errors']=errors;record['passed']=False
    record['finished']=stamp();save()
    if errors:raise RuntimeError(str(errors))
