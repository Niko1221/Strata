"""Bounded four-slot essay / tool-driven coding benchmark, native engine protocol.

Tool execution is an in-memory read/write/AST-check fixture, not a shell or a
DSH integration. Generated Python is compiled for syntax but never executed.
All output token IDs, receipt times, conversations and tool results are saved.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, OutputParser
from batch_test import tokenizer

TOOLS = [
    {"name": "read_file", "description": "Read a file in this agent's isolated workspace.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Replace module.py with complete Python source.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}},
    {"name": "run_checks", "description": "Check module.py syntax and required class, parser, serializer and unittest structure. Does not execute Python.",
     "parameters": {"type": "object", "properties": {}}},
]
DOMAINS = [("inventory", "Product", "Warehouse"), ("billing", "Invoice", "Payment"),
           ("shipping", "Shipment", "Carrier"), ("staff", "Employee", "Shift")]
ESSAYS = ["how cities can adapt to extreme heat without worsening inequality",
          "how scientific instruments changed the development of astronomy",
          "the tradeoffs between reliability, speed and cost in public transport",
          "how public libraries should respond to changes in technology and community needs"]

def fixture(domain, names):
    return (f"Implement the {domain} module with dataclasses {names[0]} and {names[1]}. "
            "Each class has exactly six fields: identifier: str, name: str, category: str, count: int, "
            "score: float, enabled: bool. Write explicit parse_<lowercase class name>(data) and "
            "serialize_<lowercase class name>(value) functions for each class. Require all fields. "
            "Reject wrong types and bool as a number, blank strings, negative count, nonfinite score, "
            "and score outside [0, 100]. Accept int or float score and convert to float. "
            "Serialize to a dictionary with all six fields. For each class include a unittest.TestCase "
            "with five methods covering valid input, a missing field, wrong type, invalid bound, "
            "and round trip. Write the full module including imports; no generic parser helper. "
            "Use only Python standard library imports. Write module.py, call run_checks, then report briefly. "
            "Do not print the source in your final message. Keep planning brief.")

def check_source(source, names):
    errors=[]
    try:
        tree=ast.parse(source)
        compile(tree, 'module.py', 'exec')  # compile only: no execution of generated code
    except (SyntaxError, ValueError, TypeError) as exc:
        return {'ok': False, 'errors': [str(exc)], 'scope': 'syntax/structure only'}
    classes={n.name:n for n in tree.body if isinstance(n,ast.ClassDef)}
    functions={n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    for name in names:
        if name not in classes:
            errors.append('missing class '+name)
        else:
            fields={n.target.id for n in classes[name].body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)}
            if fields != {'identifier','name','category','count','score','enabled'}:
                errors.append('incorrect fields in '+name)
            if not any(isinstance(d,ast.Name) and d.id=='dataclass' or isinstance(d,ast.Call) and isinstance(d.func,ast.Name) and d.func.id=='dataclass' for d in classes[name].decorator_list):
                errors.append('missing dataclass decorator on '+name)
        for prefix in ('parse_', 'serialize_'):
            if prefix+name.lower() not in functions: errors.append('missing '+prefix+name.lower())
    tests=sum(sum(isinstance(n,ast.FunctionDef) and n.name.startswith('test_') for n in c.body)
              for c in classes.values() if any(isinstance(b,ast.Attribute) and b.attr=='TestCase' or isinstance(b,ast.Name) and b.id=='TestCase' for b in c.bases))
    if tests<10: errors.append(f'only {tests} test methods; need at least 10')
    return {'ok':not errors,'errors':errors,'scope':'syntax/structure only; tests not executed'}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True)
    ap.add_argument('--exe',required=True)
    ap.add_argument('--report',required=True)
    ap.add_argument('--arm',choices=['baseline','candidate'],required=True)
    ap.add_argument('--workload',choices=['essay','agentic'],required=True)
    ap.add_argument('--seed',type=int,default=123)
    ap.add_argument('--tokens',type=int,default=2048)
    ap.add_argument('--turn-limit',type=int,default=6)
    ap.add_argument('--deadline',type=int,default=720)
    a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
    tok=tokenizer(cfg['tokenizer'])
    template=ChatTemplate(Path(cfg['tokenizer'])/'chat_template.jinja')
    tools=TOOLS if a.workload=='agentic' else []
    args=cfg['args']
    env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')}
    env.update(cfg.get('env',{}))
    env['PATH']=os.pathsep.join(cfg.get('lib_dirs',[])+[env.get('PATH','')])
    report=Path(a.report); report.parent.mkdir(parents=True,exist_ok=True)
    result={'harness_version':2,'arm':a.arm,'workload':a.workload,'exe_sha256':hashlib.sha256(Path(a.exe).read_bytes()).hexdigest(),
            'args':args,'env':cfg.get('env',{}),'seed':a.seed,'temperature':.85,'top_p':.95,'top_k':20,
            'thinking':False,'turn_limit':a.turn_limit,'token_limit_per_turn':a.tokens,
            'turns':[],'tools':[],'events':[],'agents':{},'ok':False}
    histories=[]; workspaces=[]
    for slot in range(4):
        if tools:
            domain,*names=DOMAINS[slot]
            histories.append([{'role':'system','content':'You are a coding agent with an isolated in-memory workspace. Use tools. Keep planning brief; do the work immediately.'},
                              {'role':'user','content':f'Complete the {domain} coding task. First read task.md, implement module.py, run_checks, fix any reported structure errors, and finish with a brief status.'}])
            workspaces.append({'task.md':fixture(domain,names),'module.py':''})
        else:
            histories.append([{'role':'user','content':f'Write an original, thoughtful 1800-word essay about {ESSAYS[slot]}. Develop a clear argument with concrete examples, counterarguments, and practical implications. Begin the essay immediately. Use connected prose, not an outline or bullet list.'}])
            workspaces.append({})
    result['initial_histories']=json.loads(json.dumps(histories))
    result['fixtures']=json.loads(json.dumps(workspaces))
    q=queue.Queue(); proc=None
    origin=time.monotonic(); deadline=origin+a.deadline
    with report.with_suffix('.stderr.log').open('w',encoding='utf8') as log:
        proc=subprocess.Popen([a.exe,'--serve',*args],cwd=cfg['cwd'],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
        def reader():
            with report.with_suffix('.protocol.log').open('w',encoding='utf8') as out:
                for line in proc.stdout:
                    stamp=time.monotonic(); out.write(line)
                    q.put((stamp,line.strip()))
            q.put((time.monotonic(),None))
        threading.Thread(target=reader,daemon=True).start()
        def get():
            remain=deadline-time.monotonic()
            if remain<=0: raise RuntimeError('run deadline exceeded')
            stamp,line=q.get(timeout=min(180,remain))
            if line is None or line.startswith(('ERR','FATAL')): raise RuntimeError('engine failure: '+str(line))
            return stamp,line
        def send(line):
            proc.stdin.write(line+'\n'); proc.stdin.flush()
        try:
            while not get()[1].startswith('READY'): pass
            result['startup_s']=time.monotonic()-origin
            print(a.arm,a.workload,'READY',flush=True)
            warm=tok.encode(template.render([{'role':'user','content':'Explain how a hash table works in detail.'}],enable_thinking=False),parse_special=True)
            send('GEN 256 temperature=0.85 top_p=0.95 top_k=20 seed=123 '+','.join(map(str,warm)))
            warm_tokens=[]
            while True:
                _,line=get()
                if line.startswith('T '): warm_tokens.append(int(line.split()[1]))
                if line.startswith('DONE '): break
            result['warmup_tokens']=warm_tokens
            t0=time.monotonic(); pending=list(range(4)); admitting=None; active=set(); done=set(); current={}; counts=[0]*4
            stable_s=0.; stable_tokens=0; prev=t0
            def finish(slot,reason,stamp):
                nonlocal admitting
                turn=current[slot]
                if 'finish' in turn: return
                active.discard(slot); turn['finish']=reason; turn['ended_s']=stamp-t0
                text=tok.decode(turn['ids']); turn['raw_text']=text
                # The native protocol includes terminal tokens. The HTTP server's
                # detokenizer suppresses them; they must not enter tool history.
                while text.endswith(('<|im_end|>', '<|endoftext|>')):
                    text=text[:text.rfind('<|')]
                turn['text']=text
                calls=[]; content=''
                if tools:
                    parser=OutputParser(thinking=False,tools=tools)
                    events=parser.feed(text)+parser.finish(reason)
                    calls=[e.call for e in events if e.kind=='tool_call']
                    content=''.join(e.text for e in events if e.kind=='content')
                    histories[slot].append({'role':'assistant','content':content,'tool_calls':[
                        {'id':c.id,'type':'function','function':{'name':c.name,'arguments':c.arguments}} for c in calls]})
                    for call in calls:
                        if call.name=='read_file':
                            value=workspaces[slot].get(call.arguments.get('path'), 'ERROR: allowed files are task.md and module.py')
                        elif call.name=='write_file':
                            if call.arguments.get('path')!='module.py' or not isinstance(call.arguments.get('content'),str): value={'error':'write_file accepts only module.py and string content'}
                            else:
                                workspaces[slot]['module.py']=call.arguments['content']; value={'written':'module.py','characters':len(call.arguments['content'])}
                        elif call.name=='run_checks':
                            value=check_source(workspaces[slot]['module.py'],DOMAINS[slot][1:])
                        else: value={'error':'unknown tool'}
                        result['tools'].append({'slot':slot,'turn':counts[slot],'name':call.name,'arguments':call.arguments,'result':value,'time_s':time.monotonic()-t0})
                        histories[slot].append({'role':'tool','tool_call_id':call.id,'name':call.name,'content':value if isinstance(value,str) else json.dumps(value)})
                if calls and counts[slot]<a.turn_limit: pending.append(slot)
                else:
                    done.add(slot)
                    result['agents'][str(slot)]={'turns':counts[slot],'finish':reason,'turn_limit_hit':bool(calls),
                        'checks':check_source(workspaces[slot]['module.py'],DOMAINS[slot][1:]) if tools else None}
                print(a.arm,a.workload,'slot',slot,'turn',counts[slot],'tokens',len(turn['ids']),'calls',[c.name for c in calls],reason,flush=True)
            while len(done)<4:
                now=time.monotonic()
                if admitting is None and pending:
                    if len(active)==4: stable_s+=now-prev
                    prev=now
                    slot=pending.pop(0); admitting=slot; counts[slot]+=1
                    ids=tok.encode(template.render(histories[slot],tools=tools,enable_thinking=False),parse_special=True)
                    turn={'slot':slot,'turn':counts[slot],'prompt_ids':ids,'requested_s':now-t0,'ids':[],'times_s':[]}
                    current[slot]=turn; result['turns'].append(turn)
                    send(f'BGEN {slot} {a.tokens} temperature=0.85 top_p=0.95 top_k=20 seed={a.seed} '+','.join(map(str,ids)))
                stamp,line=get(); f=line.split()
                # Receipt timestamps are monotonic; events queued during local tool checks can precede prev.
                dt=max(0.,stamp-prev)
                stable=admitting is None and len(active)==4
                if stable: stable_s+=dt
                prev=max(prev,stamp)
                if not f: continue
                if f[0] in ('T','BT'):
                    slot=admitting if f[0]=='T' else int(f[1])
                    if slot is None: raise RuntimeError('unowned solo token')
                    current[slot]['ids'].append(int(f[-1])); current[slot]['times_s'].append(stamp-t0)
                    if stable: stable_tokens+=1
                elif f[0]=='BADM':
                    slot=int(f[1]); result['events'].append({'line':line,'time_s':stamp-t0})
                    if f[2]=='1': active.add(slot)
                    else: finish(slot,current[slot].get('admission_reason','stop'),stamp)
                    admitting=None
                elif f[0]=='DONE' and admitting is not None:
                    current[admitting]['admission_reason']=f[5]
                elif f[0]=='BDONE':
                    result['events'].append({'line':line,'time_s':stamp-t0}); finish(int(f[1]),f[3],stamp)
            result['workflow_s']=time.monotonic()-t0
            result['c4_no_admission_s']=stable_s
            result['c4_no_admission_tokens']=stable_tokens
            result['c4_no_admission_tps']=stable_tokens/stable_s if stable_s else None
            result['output_tokens']=sum(len(t['ids']) for t in result['turns'])
            result['workflow_tps']=result['output_tokens']/result['workflow_s']
            all_times=[x for t in result['turns'] for x in t['times_s']]
            result['generation_span_tps']=result['output_tokens']/(max(all_times)-min(all_times))
            result['histories']=histories; result['workspaces']=workspaces; result['ok']=True
            send('QUIT'); proc.wait(timeout=60)
        except Exception as exc:
            result['error']=repr(exc); raise
        finally:
            if proc.poll() is None: proc.terminate(); proc.wait(timeout=30)
            result['exit_code']=proc.returncode
            report.write_text(json.dumps(result,indent=2),encoding='utf8')
    print('SAVED',report,round(result['workflow_tps'],2),flush=True)

if __name__=='__main__': main()
