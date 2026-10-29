"""Prospective layout controls at 1K/16K, followed by twelve fixed task checks."""
import json,os,re,subprocess,sys
from pathlib import Path
import numpy as np
from build_pair import ARMS,MODEL,OUT,REPO,ROOT,digest,stamp
BASE=REPO/'reports/tobeo-20261022/qwen36-context-curve';sys.path.insert(0,str(BASE))
import run_curve as protocol
sys.path.insert(0,str(ARMS['main']['source']/'tools/qwen35'))
from q36_harness import Eng
sys.path.insert(0,str(ARMS['main']['source']/'tools'))
from strata_tokenizer import Tokenizer
DIR=OUT/'validation-next';RUN=DIR/'run-01';RUN.mkdir(exist_ok=False)
adapter=json.loads((DIR/'adapter-build.json').read_text());assert adapter['passed']
prior=json.loads((OUT/'execution-02/results.json').read_text())
report={'started':stamp(),'passed':False,'phase':'preparing','runs':[],'numeric':{},'tasks':{},'controls':{},
        'plan':'validation-next-plan.md','downloads_paused':False}
def save():
    tmp=RUN/'results.tmp';tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(RUN/'results.json')
def env(name,packed='1',keep=False):
    e={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')}
    e.update(STRATA_Q8_PACKED=packed,STRATA_MMQ_KQ='1',STRATA_Q36_FIXED_SLOTS='1',CUDA_VISIBLE_DEVICES='0',CUDA_MODULE_LOADING='EAGER')
    if keep:e['STRATA_Q36_KEEP_PACKED']='1'
    cuda=ROOT/'toolchains/cuda-13.2.0';e['PATH']=os.pathsep.join([str(cuda/'bin/x64'),str(cuda/'bin'),str(ARMS[name]['build']),e.get('PATH','')]);return e
def arguments(slots=None):
    args=['--native',str(MODEL),'--max-context','32768','--prefill','2048','--pool-workers','7','--mtp','0']
    if slots is not None:args+=['--expert-slots',str(slots),'--expert-vram','10']
    return args
def parse_slots(log):
    text=log.read_text(errors='replace');matches=re.findall(r'hybrid experts: (\d+) of 256 per layer',text)
    if len(matches)!=1:raise RuntimeError('Unexpected placement count')
    return int(matches[0])
def logits_run(label,name,case,packed='1',keep=False,slots=None):
    ids=json.loads((BASE/(case+'.ids.json')).read_text())
    forced=next(r for r in prior['rows'] if r.get('pair')==0 and r['case']==case and r['keep_packed'])['output_ids'][:96]
    tokenfile=RUN/(case+'.tokens');forcefile=RUN/(case+'.forced')
    tokenfile.write_text(','.join(map(str,ids)));forcefile.write_text(','.join(map(str,forced)))
    exe=Path(adapter['arms'][name]['exe'])
    if digest(exe)!=adapter['arms'][name]['sha256']:raise RuntimeError('Adapter changed')
    command=[str(exe),*arguments(slots),'--tokens-file',str(tokenfile),'--force-file',str(forcefile),'--dump-full-logits',str(RUN/(label+'.f32'))]
    log=RUN/(label+'.log');print('LOGITS '+label,flush=True)
    row={'label':label,'source':name,'case':case,'packed':packed,'keep':keep,'explicit_slots':slots,
         'input_sha256':protocol.tokens_hash(ids),'forced_sha256':protocol.tokens_hash(forced),'rows':97}
    report['runs'].append(row);save()
    with log.open('w',encoding='utf-8') as output:
        p=subprocess.Popen(command,env=env(name,packed,keep),stdout=output,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        try:code=p.wait(timeout=900)
        except BaseException:p.kill();p.wait();raise
    row.update(exit_code=code,slots=parse_slots(log),bytes=(RUN/(label+'.f32')).stat().st_size)
    text=log.read_text(errors='replace');row['replacement_counts']=[int(x) for x in re.findall(r'(\d+) slot replacements',text)]
    if code or row['bytes']!=97*248320*4 or any(row['replacement_counts']):raise RuntimeError('Logits run dimensions, exit or fixed-state check failed')
    save();return row
def compare(a,b):
    x=np.memmap(a,dtype=np.float32,mode='r').reshape(97,248320);y=np.memmap(b,dtype=np.float32,mode='r').reshape(97,248320)
    identical=bool(np.array_equal(x.view(np.uint32),y.view(np.uint32)));finite=bool(np.isfinite(x).all() and np.isfinite(y).all())
    kl=[];top=[];maximum=0.
    for start in range(0,97,8):
        p=np.array(x[start:start+8],dtype=np.float64);q=np.array(y[start:start+8],dtype=np.float64)
        maximum=max(maximum,float(np.max(np.abs(p-q))));top.extend((p.argmax(1)==q.argmax(1)).tolist())
        p-=p.max(1,keepdims=True);p-=np.log(np.exp(p).sum(1,keepdims=True))
        q-=q.max(1,keepdims=True);q-=np.log(np.exp(q).sum(1,keepdims=True))
        kl.extend(np.sum(np.exp(p)*(p-q),axis=1).tolist())
    return {'bitwise_identical':identical,'finite':finite,'top1_equal':sum(top),'rows':97,
            'mean_kl':float(np.mean(kl)),'max_kl':max(kl),'max_absolute_error':maximum}

tok=Tokenizer.from_gguf(MODEL)
def chat_ids(name,text):
    return tok.encode(f'<|im_start|>user\nCASE_{name}\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',parse_special=True)
tasks=[]
for i,(expr,value) in enumerate([('8+13',21),('47-19',28),('3+5+8',16),('7*9',63)]):
    tasks.append({'name':f'arithmetic-{i}','text':f'Calculate {expr}. Return only JSON with one key "value" containing the integer result.', 'kind':'json','expected':{'value':value},'cap':64})
tasks += [{'name':'chinese-literal','text':'请只输出四个汉字：缓存命中。不要标点，不要解释。','kind':'literal','expected':'缓存命中','cap':32},
          {'name':'chinese-json','text':'只返回一个JSON对象，字段“原因”的值必须是“容量不足”，不要添加其他字段或说明。','kind':'json','expected':{'原因':'容量不足'},'cap':64}]
for name,prompt,inputs,expected in [
    ('clamp','solve(x, lo, hi) returns x clamped to inclusive bounds lo and hi.', [[-2,0,5],[3,0,5],[9,0,5]],[0,3,5]),
    ('absdiff','solve(a, b) returns the absolute difference of a and b.',[[3,8],[8,3],[-4,-4]],[5,5,0]),
    ('squares','solve(xs) returns the sum of squares of the numbers in xs.',[[[1,2,3]],[[]],[[-2,3]]],[14,0,13]),
    ('reverse','solve(xs) returns a new list with the elements of xs reversed.',[[[1,2,3]],[[]],[['a','b']] ],[[3,2,1],[],['b','a']])]:
    tasks.append({'name':name,'text':'Write only one pure Python function, no imports and no explanation. '+prompt,'kind':'code','inputs':inputs,'expected':expected,'cap':192})
body=tok.encode(protocol.ledger(20261029))
for length in [1024,16384]:
    name=f'retrieval-{length}';prefix='The unique access code is NOVA-731. Remember it while reading this unrelated synthetic ledger.\n'
    suffix='\nReturn only JSON with the field "code" equal to the unique access code stated at the beginning.'
    overhead=len(chat_ids(name,prefix+suffix));excerpt=body[:length-overhead]
    while '\ufffd' in tok.decode(excerpt):excerpt=excerpt[:-1]
    tasks.append({'name':name,'ids':chat_ids(name,prefix+tok.decode(excerpt)+suffix),'kind':'json','expected':{'code':'NOVA-731'},'cap':64})
for task in tasks:
    if 'ids' not in task:task['ids']=chat_ids(task['name'],task['text'])
manifest=[{k:v for k,v in t.items() if k!='ids'}|{'prompt_tokens':len(t['ids']),'input_sha256':protocol.tokens_hash(t['ids'])} for t in tasks]
(RUN/'task-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

def judge(task,text):
    stripped=text.removesuffix('<|im_end|>').strip();fence=re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```',stripped,re.S)
    if fence:stripped=fence.group(1)
    if task['kind']=='literal':return {'passed':stripped==task['expected']}
    if task['kind']=='json':
        try:return {'passed':json.loads(stripped)==task['expected']}
        except ValueError:return {'passed':False,'error':'not one JSON object'}
    try:
        run=subprocess.run([sys.executable,str(OUT/'judge_code.py')],input=json.dumps({'text':stripped,'inputs':task['inputs'],'expected':task['expected']}),capture_output=True,text=True,timeout=3,creationflags=subprocess.CREATE_NO_WINDOW)
        return json.loads(run.stdout) if run.returncode==0 else {'passed':False,'error':run.stderr[:200]}
    except subprocess.TimeoutExpired:return {'passed':False,'error':'code timeout'}

save();state={'health_before':protocol.request('/health')};unloaded=False
try:
    state['unload']=protocol.request('/unload?wait=30',post=True)
    if state['unload'].get('status') not in ('unloaded','not loaded'):raise RuntimeError('Daily engine busy')
    unloaded=True;report['phase']='adapter-check';save()
    smoke=logits_run('adapter-smoke','main','r0-l1024',slots=63)
    report['adapter_check']=compare(RUN/'adapter-smoke.f32',OUT/'packed-logits-01/1.f32')
    if not report['adapter_check']['bitwise_identical']:raise RuntimeError('File adapter did not match original archived logits')
    report['phase']='numeric-controls';save()
    for case in ['r0-l1024','r0-l16384']:
        a=logits_run(case+'-A','pr4347',case,keep=True)
        b=logits_run(case+'-B','pr4347',case)
        report['controls'][case]={'keep_slots':a['slots'],'drop_slots':b['slots']}
        logits_run(case+'-C','main',case,packed='0',slots=a['slots'])
        logits_run(case+'-D','main',case,packed='0',slots=b['slots'])
        comparisons={key:compare(RUN/(case+'-'+x+'.f32'),RUN/(case+'-'+y+'.f32')) for key,x,y in [('A_C','A','C'),('B_D','B','D'),('A_B','A','B'),('C_D','C','D')]}
        report['numeric'][case]=comparisons;save()
        if not all(comparisons[k]['bitwise_identical'] and comparisons[k]['finite'] for k in ['A_C','B_D']):
            raise RuntimeError('Matched-placement numeric gate failed: '+case)
    report['numeric_passed']=True;report['phase']='task-checks';save()
    sizes=report['controls']['r0-l1024']
    configs=[('A','pr4347','1',True,None),('B','pr4347','1',False,None),('C','main','0',False,sizes['keep_slots']),('D','main','0',False,sizes['drop_slots'])]
    for label,name,packed,keep,slots in configs:
        print('TASKS '+label,flush=True);rows=[];report['tasks'][label]=rows;engine=None
        try:
            engine=Eng(str(ARMS[name]['build']/'strata-q36.exe'),arguments(slots),env=env(name,packed,keep),log=str(RUN/(label+'-tasks.log')))
            if parse_slots(RUN/(label+'-tasks.log'))!=(sizes['keep_slots'] if label in ['A','C'] else sizes['drop_slots']):raise RuntimeError('Task placement changed')
            for task in tasks:
                result=protocol.measured_request(engine,task['ids'],task['cap']);text=tok.decode(result['output_ids'])
                row={'name':task['name'],'prompt_tokens':len(task['ids']),'output':text,'output_ids':result['output_ids'],
                     'reused_tokens':result['reused_tokens'],'generated_tokens':result['generated_tokens'],'finish':result['finish'],'judgment':judge(task,text)}
                if result['reused_tokens']!=0:raise RuntimeError('Task reused a prefix')
                rows.append(row);save();print(label,task['name'],row['judgment']['passed'],flush=True)
        finally:
            if engine:engine.close()
    report['task_passed']=all(len(rows)==12 and all(r['judgment']['passed'] for r in rows) for rows in report['tasks'].values())
    report['passed']=report['numeric_passed'] and report['task_passed'];report['phase']='complete';save()
except BaseException as exc:report['error']=repr(exc);report['phase']='failed';save();raise
finally:
    try:
        if unloaded and state['health_before'].get('loaded'):
            state['load']=protocol.request('/load',post=True);state['health_after']=protocol.request('/health')
            if not state['health_after'].get('loaded'):raise RuntimeError('Daily service failed to restore')
    except BaseException as exc:state['restore_error']=repr(exc);report['passed']=False;raise
    finally:report['service_transition']=state;report['finished']=stamp();save()
