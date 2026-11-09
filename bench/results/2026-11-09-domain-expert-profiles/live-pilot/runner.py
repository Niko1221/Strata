"""Prospective A/B/A coding pilot using existing Strata --expert-profile loading.

Copyright (c) 2026 Francisco Angulo de Lafuente. MIT licensed.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE=Path(__file__).resolve().parent
CASES=[
    dict(id='clamp',prompt='Return only Python 3 code for def clamp(x, lo, hi): which clips x to the inclusive interval [lo, hi]. Assume lo <= hi. No imports or explanation.'),
    dict(id='overlap',prompt='Return only Python 3 code for def overlap_length(a0, a1, b0, b1): which returns the length of the intersection of half-open intervals [a0,a1) and [b0,b1). Assume ordered endpoints. No imports or explanation.'),
    dict(id='sql',prompt='Return only one SQLite SELECT statement. Given transactions(category TEXT, amount REAL), return category and the sum of strictly positive amounts for each category having such a row, ordered by category. Do not include zero or negative amounts. No explanation.')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quality(case,text):
    code=text.strip()
    if code.startswith('```') and code.endswith('```'):
        code='\n'.join(code.splitlines()[1:-1]).strip()
    try:
        if case=='sql':
            if not code.lower().startswith('select'):
                return False
            with sqlite3.connect(':memory:') as db:
                db.execute('CREATE TABLE transactions(category TEXT,amount REAL)')
                db.executemany('INSERT INTO transactions VALUES(?,?)',[('a',5),('a',-2),('a',7),('b',10),('b',0),('c',-3),('z',0)])
                db.commit()
                def allowed(op,a,b,c,d):
                    if op==sqlite3.SQLITE_SELECT or op==sqlite3.SQLITE_READ and a=='transactions':
                        return sqlite3.SQLITE_OK
                    if op==sqlite3.SQLITE_FUNCTION and b.lower()=='sum':
                        return sqlite3.SQLITE_OK
                    return sqlite3.SQLITE_DENY
                db.set_authorizer(allowed)
                deadline=time.monotonic()+2
                db.set_progress_handler(lambda: int(time.monotonic()>deadline),1000)
                return db.execute(code).fetchall()==[('a',12.0),('b',10.0)]
        tree=ast.parse(code)
        allowed_nodes=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.If,ast.IfExp,ast.Compare,
            ast.Lt,ast.LtE,ast.Gt,ast.GtE,ast.Eq,ast.NotEq,ast.Name,ast.Load,ast.Store,ast.Call,ast.Constant,
            ast.BinOp,ast.Add,ast.Sub,ast.UnaryOp,ast.USub,ast.UAdd,ast.Assign,ast.Expr,ast.BoolOp,ast.And,ast.Or)
        if len(list(ast.walk(tree)))>150 or any(not isinstance(n,allowed_nodes) for n in ast.walk(tree)):
            return False
        if any(isinstance(n,ast.Call) and (not isinstance(n.func,ast.Name) or n.func.id not in ('min','max')) for n in ast.walk(tree)):
            return False
        if any(isinstance(n,ast.FunctionDef) and (n.decorator_list or n.args.defaults) for n in ast.walk(tree)):
            return False
        if any(isinstance(n,ast.Constant) and isinstance(n.value,(int,float)) and abs(n.value)>1000 for n in ast.walk(tree)):
            return False
        name='clamp' if case=='clamp' else 'overlap_length'
        if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef) or tree.body[0].name!=name:
            return False
        namespace={'__builtins__':{'min':min,'max':max,'int':int,'float':float}}
        exec(compile(tree,'<checked-model-function>','exec'),namespace)
        fn=namespace[name]
        checks=([((-5,0,10),0),((7,0,10),7),((15,0,10),10),((2.5,1,3),2.5),((-4,-3,-1),-3)]
            if case=='clamp' else [((0,2,3,5),0),((0,2,2,3),0),((0,10,2,7),5),((-5,-1,-3,2),2),((0,2.5,1,4),1.5)])
        return all(fn(*args)==want for args,want in checks)
    except Exception:
        return False


def request(port,prompt,tokens=96):
    data=json.dumps(dict(messages=[dict(role='user',content=prompt)],max_tokens=tokens,temperature=0,
                         chat_template_kwargs=dict(enable_thinking=False))).encode()
    req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=data,headers={'Content-Type':'application/json'})
    t0=time.monotonic()
    with urllib.request.urlopen(req,timeout=240) as r:
        out=json.load(r)
    return out,time.monotonic()-t0


def replace_arg(args,flag,value):
    args=list(args)
    while flag in args:
        i=args.index(flag)
        del args[i:i+2]
    return args+[flag,str(value)]


def stop_group(proc):
    """This group was created only for this leg, including any surviving descendants."""
    try:
        os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        proc.poll()
        try:
            os.killpg(proc.pid,0)
        except ProcessLookupError:
            break
        time.sleep(.1)
    else:
        try:
            os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def leg(label,profile,baseline,output):
    root=Path(baseline['cwd'])
    cfg=json.loads(json.dumps(baseline))
    cfg['args']=replace_arg(cfg['args'],'--expert-profile',profile)
    cfg['args']=replace_arg(cfg['args'],'--prompt-cache',0)
    cfg['args']=replace_arg(cfg['args'],'--pcie-frac',0)
    cfg['env'].update(STRATA_ROUTE_PRIOR='0',STRATA_ADAPT_NOWAIT='0',STRATA_ADAPT_LAG='1')
    cfg['port']=8199
    cfg['log']=str((output/f'{label}.engine.log').resolve())
    config=output/f'{label}.config.json'
    config.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    env=dict(os.environ)
    env.update(cfg['env'])
    result=dict(label=label,profile_sha256=sha(profile),requests=[])
    with socket.socket() as port_check:
        port_check.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        port_check.bind(('127.0.0.1',8199))
    t0=time.monotonic()
    with (output/f'{label}.server.log').open('w',encoding='utf-8') as log:
        proc=subprocess.Popen([str(root/'.venv/bin/python3'),str(root/'serve/server.py'),'--engine','strata',
            '--config',str(config.resolve()),'--port','8199'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        deadline=time.monotonic()+240
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f'server exited {proc.returncode}')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8199/v1/models',timeout=3) as r:
                    if r.status==200:
                        break
            except Exception:
                pass
            if time.monotonic()>deadline:
                raise TimeoutError('startup')
            time.sleep(2)
        warm,seconds=request(8199,'Reply with OK.',2)
        result['ready_and_warm_seconds']=time.monotonic()-t0
        result['warmup_wall_seconds']=seconds
        result['warmup']=warm
        text=Path(cfg['log']).read_text(encoding='utf-8')
        matches=re.findall(r'expert cache (\d+) slots',text)
        result['cache_slots']=int(matches[0]) if matches else None
        expected=json.loads((HERE/'capacity-plan.json').read_text(encoding='utf-8'))['expected_slots']
        result['capacity_match']=result['cache_slots']==expected
        if not result['capacity_match']:
            raise ValueError(f'cache size differs from frozen{expected}slots')
        for case in CASES:
            raw,seconds=request(8199,case['prompt'])
            text=raw['choices'][0]['message']['content']
            row=dict(case=case['id'],prompt=case['prompt'],wall_seconds=seconds,output=text,
                     quality_pass=quality(case['id'],text),raw=raw)
            result['requests'].append(row)
            print(label,case['id'],round(seconds,2),'quality',row['quality_pass'],flush=True)
        return result
    finally:
        stop_group(proc)
        (output/f'{label}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


def main():
    def interrupted(signum,frame):
        raise SystemExit(124)
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGALRM,interrupted)
    signal.alarm(24*60)
    out=(HERE/'out').resolve()
    out.mkdir(exist_ok=False)
    baseline=json.loads((HERE/'baseline.json').read_text(encoding='utf-8'))
    result=dict(author='Francisco Angulo de Lafuente',protocol_sha256=sha(HERE/'LIVE_PROTOCOL.md'),
        source_sha256=sha(__file__),engine_sha256=sha(baseline['exe']),kind='single_ABA_code_pilot',legs=[])
    try:
        for label,profile in [('A1','global.bin'),('B','code.bin'),('A2','global.bin')]:
            result['legs'].append(leg(label,(HERE/profile).resolve(),baseline,out))
            (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    except BaseException as exc:
        result['status']='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed'
        result['error_type']=type(exc).__name__
        (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        raise
    legs=result['legs']
    totals=[sum(r['wall_seconds'] for r in leg_['requests']) for leg_ in legs]
    same_outputs=all(legs[0]['requests'][i]['output']==legs[1]['requests'][i]['output']==legs[2]['requests'][i]['output']
        and len({r['requests'][i]['raw']['usage']['completion_tokens'] for r in legs})==1 for i in range(3))
    noise=abs(totals[0]-totals[2])/((totals[0]+totals[2])/2)
    quality_all=all(r['quality_pass'] for leg_ in legs for r in leg_['requests'])
    valid=quality_all and same_outputs and noise<=.1 and all(r['capacity_match'] for r in legs)
    result['assessment']=dict(request_totals_seconds=totals,baseline_relative_difference=noise,all_quality=quality_all,
        identical_outputs=same_outputs,comparison_valid=valid,
        improvement_gate=valid and totals[1]<=.95*min(totals[0],totals[2]),
        note='Singleinterleavedpilot; startup/warmup reported separately; no population or newdefault claim.')
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['assessment']),flush=True)


if __name__=='__main__':
    main()
