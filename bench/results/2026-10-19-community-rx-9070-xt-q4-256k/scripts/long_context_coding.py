"""Controlled local tool loop with a large synthetic repository-context bundle.

This is a distractor-heavy fixture, not a production repository or a Claude Code
long-context test. All model messages are retained; no summarization/truncation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import threading
import time

from benchmark import request, monitor, gpu_monitor

BASE = Path('.')
SOURCE = Path(__file__).resolve().parents[1]/'fixtures/fullstack'
WRITABLE = {'api.js','client.js','db.js','regression.test.js'}

def schema(name, description, properties, required):
    return {'type':'function','function':{'name':name,'description':description,
        'parameters':{'type':'object','properties':properties,'required':required,'additionalProperties':False}}}

TOOLS = [
    schema('read_file','Read a fixture source or test file.',{'path':{'type':'string'}},['path']),
    schema('write_file','Replace a source file or create regression.test.js. Original tests are immutable.',
           {'path':{'type':'string'},'content':{'type':'string'}},['path','content']),
    schema('run_tests','Run the local Node test suite.',{},[]),
]

def tests(folder):
    p=subprocess.run(['node','--test'],cwd=folder,
                     capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
    return {'exit_code':p.returncode,'output':(p.stdout+p.stderr)[-24000:]}

def tool(folder,name,args):
    if name=='run_tests':
        return tests(folder)
    path=args.get('path')
    if path not in WRITABLE | {'app.test.js','TASK.md','package.json'}:
        raise ValueError('File is outside the allowed fixture set')
    target=folder/path
    if name=='read_file':
        return {'content':target.read_text(encoding='utf-8')}
    if name=='write_file':
        if path not in WRITABLE or not isinstance(args.get('content'),str) or len(args['content'])>100000:
            raise ValueError('Write is not allowed')
        target.write_text(args['content'],encoding='utf-8')
        return {'written':path,'bytes':target.stat().st_size}
    raise ValueError('Unknown tool')

def bundle(count):
    rng=random.Random(7129)
    parts=['Synthetic repository context. Relevant files are api.js, db.js, client.js and app.test.js.\n',
           'FILE api.js\n'+(SOURCE/'api.js').read_text()+'\n']
    for i in range(count):
        if i==count//2:
            parts.append('FILE db.js\n'+(SOURCE/'db.js').read_text()+'\n')
        parts.append(f'FILE generated/module{i}.ts\nexport function item{i}(x:number):number {{return x*{rng.randrange(1,991)}+{rng.randrange(1,9999)};}}\n')
    for name in ['client.js','app.test.js','TASK.md']:
        parts.append(f'FILE {name}\n'+(SOURCE/name).read_text()+'\n')
    parts.append('Use the tools to repair the fixture, add regression tests, run tests and iterate. '
                 'Do not merely describe a patch. Only finish after the tests pass.')
    return ''.join(parts)

def stream(messages):
    payload={'model':'strata','messages':messages,'tools':TOOLS,'tool_choice':'auto',
             'temperature':0,'seed':42,'reasoning_effort':'none','max_tokens':4096,
             'stream':True,'stream_options':{'include_usage':True}}
    message={'role':'assistant','content':''}; calls={}; events=[]
    start=time.perf_counter(); first=None
    with request('/v1/chat/completions',payload) as response:
        for raw in response:
            if not raw.startswith(b'data: '):continue
            data=raw[6:].strip()
            if data==b'[DONE]':break
            event=json.loads(data);events.append(event)
            for choice in event.get('choices',[]):
                delta=choice.get('delta',{})
                if first is None and (delta.get('content') or delta.get('tool_calls')):
                    first=time.perf_counter()-start
                message['content']+=delta.get('content') or ''
                for part in delta.get('tool_calls',[]):
                    call=calls.setdefault(part['index'],{'id':'','type':'function','function':{'name':'','arguments':''}})
                    if part.get('id'):call['id']=part['id']
                    for key in ['name','arguments']:
                        call['function'][key]+=part.get('function',{}).get(key,'')
    if calls:message['tool_calls']=[calls[i] for i in sorted(calls)]
    result={'message':message,'wall_seconds':time.perf_counter()-start,
            'ttft_or_first_tool_seconds':first,'events':events}
    for event in events:
        for key in ['usage','timings']:
            if event.get(key):result[key]=event[key]
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--target-input',type=int,default=225280)
    parser.add_argument('--label',default='q4-coding-220k')
    options=parser.parse_args()
    if not options.label.replace('-','').isalnum():raise SystemExit('Invalid label')
    if not 8192 <= options.target_input <= 240000:
        raise SystemExit('Target must leave output and tool-history space within the 256K window')
    with request('/health') as response:
        health=json.load(response)
    if health.get('max_context',0) < options.target_input+16384:
        raise SystemExit('Server context does not leave enough room for tool history')
    with request('/metrics') as response:
        engine=json.load(response).get('engine')
    folder=BASE/'coding'/options.label
    if folder.exists():raise SystemExit('Fresh fixture required; refusing to overwrite')
    count=options.target_input//40
    for _ in range(5):
        prompt=bundle(count)
        with request('/v1/messages/count_tokens',{'model':'strata','max_tokens':4096,
                'messages':[{'role':'user','content':prompt}]}) as response:
            actual=json.load(response)['input_tokens']
        if abs(actual-options.target_input)<128:break
        count=max(1,int(count*options.target_input/actual))
    shutil.copytree(SOURCE,folder)
    report={'label':options.label,'target_input':options.target_input,'count_endpoint_tokens':actual,
            'server_health':health,'engine_configuration':engine,
            'limitation':'Synthetic distractor bundle; controlled tool loop, not Claude Code or a production repository.',
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'turns':[],'tools':[],
            'baseline_tests':tests(folder)}
    path=BASE/'results'/(options.label+'.json')
    (BASE/'results'/(options.label+'-prompt.txt')).write_text(prompt,encoding='utf-8')
    messages=[{'role':'user','content':prompt}]
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    stop=threading.Event();rows=[];gpu_rows=[]
    workers=[threading.Thread(target=monitor,args=(stop,rows),daemon=True),
             threading.Thread(target=gpu_monitor,args=(stop,gpu_rows),daemon=True)]
    for worker in workers:worker.start()
    start=time.perf_counter()
    print(json.dumps({'label':options.label,'counted_input':actual,'folder':str(folder)}),flush=True)
    try:
        for n in range(16):
            turn=stream(messages);report['turns'].append(turn);messages.append(turn['message'])
            print(json.dumps({'turn':n,'usage':turn.get('usage'),'seconds':turn['wall_seconds']}),flush=True)
            calls=turn['message'].get('tool_calls',[])
            if not calls:break
            for call in calls:
                name=call['function']['name']
                try:
                    args=json.loads(call['function']['arguments'])
                    result=tool(folder,name,args)
                except Exception as exc:
                    result={'error':str(exc)}
                report['tools'].append({'name':name,'arguments':call['function']['arguments'],'result':result})
                messages.append({'role':'tool','tool_call_id':call['id'],'content':json.dumps(result)})
                print(json.dumps({'tool':name,'error':result.get('error'),'exit_code':result.get('exit_code')}),flush=True)
            path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    except Exception as exc:
        report['error']=repr(exc)
    finally:
        report['wall_seconds']=time.perf_counter()-start
        stop.set()
        for worker in workers:worker.join(timeout=25)
        try:
            final_tests=tests(folder)
        except Exception as exc:
            final_tests={'exit_code':None,'error':repr(exc)}
        report.update(samples=rows,gpu_samples=gpu_rows,final_tests=final_tests,
            original_tests_intact=(folder/'app.test.js').read_bytes()==(SOURCE/'app.test.js').read_bytes(),
            regression_tests_added=(folder/'regression.test.js').is_file()
                and (folder/'regression.test.js').stat().st_size > 0,
            changed_source=[n for n in ['api.js','db.js','client.js']
                            if (folder/n).read_bytes()!=(SOURCE/n).read_bytes()])
        report['passed']=(not report.get('error') and report['original_tests_intact']
            and report['regression_tests_added']
            and report['final_tests']['exit_code']==0 and len(report['changed_source'])>=2
            and any(t['name']=='run_tests' and t['result'].get('exit_code')==0 for t in report['tools']))
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({k:report.get(k) for k in ['label','wall_seconds','passed','error','changed_source']}),flush=True)

if __name__=='__main__':main()
