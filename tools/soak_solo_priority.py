"""Synthetic, multi-conversation HTTP soak. Never executes generated code or tools.

Requires an already-running server; does not restart or modify it. Credentials
come from STRATA_API_KEY. Each fixture has messages, expected (audit), and task.
python tools/soak_solo_priority.py --base http://127.0.0.1:8080 \
    --fixtures a.json b.json c.json --seconds 3600 --out /tmp/solo-soak
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import socket
import threading
import time
from urllib.parse import urlsplit

MANIFEST_TOOL={'type':'function','function':{'name':'record_patch_manifest','description':'Record a synthetic manifest only.',
    'parameters':{'type':'object','properties':{'session':{'type':'string'},'revision':{'type':'string'},
    'files':{'type':'array','items':{'type':'string'}}},'required':['session','revision','files']}}}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def parse_json(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    return json.loads(text)


def same_json(actual, expected):
    # Python's equality treats False == 0 and True == 1; these are different JSON values.
    return json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


class Client:
    """One connection, including socket-level cancellation before response headers."""
    def __init__(self, base, key):
        self.base, self.key = urlsplit(base), key
        self.conn = None
        self.sock = None
        self.first = threading.Event()
        self.cancelled = threading.Event()
        self.cancel_at = None

    def cancel(self):
        self.cancel_at = time.monotonic()
        self.cancelled.set()
        if self.sock is not None:
            try: self.sock.shutdown(socket.SHUT_RDWR)
            except OSError: pass

    def request(self, body, dialect='chat', label='request'):
        start = time.monotonic()
        row = {'label':label, 'dialect':dialect, 'started':time.time(), 'start':start}
        content, reasoning, calls, events = '', '', {}, []
        first, usage, finish, complete, completed_response = None, None, None, False, None
        try:
            cls = http.client.HTTPSConnection if self.base.scheme=='https' else http.client.HTTPConnection
            self.conn = cls(self.base.hostname, self.base.port, timeout=900)
            self.conn.connect(); self.sock = self.conn.sock
            if self.cancelled.is_set(): self.cancel(); raise OSError('Cancelled before sending')
            headers = {'Content-Type':'application/json'}
            if self.key: headers['Authorization']='Bearer '+self.key
            endpoint = '/v1/responses' if dialect=='responses' else '/v1/chat/completions'
            wire = {'temperature':0, 'seed':117, 'stream':True, **body}
            if dialect=='chat':
                wire.setdefault('stream_options', {'include_usage':True})
                wire.setdefault('reasoning_effort','none')
            else: wire.setdefault('reasoning',{'effort':'none'})
            self.conn.request('POST',self.base.path.rstrip('/')+endpoint,json.dumps(wire).encode(),headers)
            response = self.conn.getresponse()
            row['status'] = response.status
            if response.status != 200:
                raise RuntimeError('HTTP '+str(response.status)+': '+response.read().decode()[:1500])
            while True:
                line = response.readline()
                if not line: break
                if not line.startswith(b'data:'): continue
                raw = line[5:].strip()
                if raw==b'[DONE]': complete=True; break
                e = json.loads(raw)
                if e.get('error'): raise RuntimeError(str(e['error']))
                if dialect=='responses':
                    kind = e.get('type'); events.append(kind)
                    if kind in ('response.failed','response.incomplete'):
                        raise RuntimeError(json.dumps(e.get('response',{}))[:1500])
                    text = e.get('delta','') if kind in ('response.output_text.delta','response.function_call_arguments.delta','response.reasoning_summary_text.delta') else ''
                    if text and first is None: first=time.monotonic(); self.first.set()
                    if kind=='response.completed':
                        completed_response=e['response']; complete=True; usage=completed_response.get('usage'); finish='completed'
                        for item in completed_response.get('output',[]):
                            if item.get('type')=='function_call': calls[item['call_id']]={'id':item['call_id'],'name':item['name'],'arguments':item['arguments']}
                            if item.get('type')=='message': content+=''.join(x.get('text','') for x in item.get('content',[]))
                        break
                else:
                    usage=e.get('usage') or usage
                    for choice in e.get('choices',[]):
                        delta=choice.get('delta',{}); finish=choice.get('finish_reason') or finish
                        text=delta.get('content') or ''; thought=delta.get('reasoning_content') or ''
                        tc=delta.get('tool_calls') or []
                        if (text or thought or any(t.get('function',{}).get('arguments') for t in tc)) and first is None:
                            first=time.monotonic(); self.first.set()
                        content+=text; reasoning+=thought
                        for t in tc:
                            c=calls.setdefault(t.get('index',0),{'id':'','name':'','arguments':''})
                            if t.get('id'): c['id']=t['id']
                            for k in ('name','arguments'): c[k]+=t.get('function',{}).get(k) or ''
            if not complete: raise RuntimeError('Stream ended without a terminal success event')
            row['ok']=True
        except Exception as e:
            row.update(ok=False,error=type(e).__name__+': '+str(e))
        finally:
            if self.conn: self.conn.close()
        end=time.monotonic()
        row.update(end=end,wall_s=end-start,ttft_s=first-start if first is not None else None,
                   output_tokens=(usage or {}).get('completion_tokens',(usage or {}).get('output_tokens',0)),
                   usage=usage,finish=finish,content=content,reasoning_content=reasoning,tool_calls=list(calls.values()),
                   content_sha256=digest(content),cancel_requested=self.cancelled.is_set(),
                   cancel_return_s=end-self.cancel_at if self.cancel_at is not None else None,
                   response_events=events,response=completed_response)
        return row


class Harness:
    def __init__(self, base, out, cancel_interval=240):
        self.base=base; self.key=os.environ.get('STRATA_API_KEY',''); self.out=Path(out)
        self.out.mkdir(parents=True,exist_ok=False)
        self.lock=threading.Lock(); self.rows=[]; self.telemetry=[]; self.failures=[]
        self.stop=threading.Event(); self.active=set(); self.rollovers=[]
        self.cancel_interval=cancel_interval

    def get(self, path):
        u=urlsplit(self.base)
        cls=http.client.HTTPSConnection if u.scheme=='https' else http.client.HTTPConnection
        conn=cls(u.hostname,u.port,timeout=8)
        try:
            conn.request('GET',u.path.rstrip('/')+path,headers={'Authorization':'Bearer '+self.key})
            r=conn.getresponse()
            if r.status!=200: raise RuntimeError(str(r.status))
            return json.load(r)
        finally: conn.close()

    def snapshot(self):
        return {'at':time.time(),'metrics':self.get('/metrics'),'status':self.get('/v1/status')}

    def save(self, name, value):
        p=self.out/name; tmp=p.with_suffix(p.suffix+'.tmp')
        tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); tmp.replace(p)

    def record(self, row):
        with self.lock:
            self.rows.append(row)
            with (self.out/'requests.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')

    def request(self, label, body, dialect='chat', client=None):
        client=client or Client(self.base,self.key)
        with self.lock: self.active.add(client)
        try: row=client.request(body,dialect,label)
        finally:
            with self.lock: self.active.discard(client)
        self.record(row)
        return row

    def check(self, condition, label, detail):
        if not condition:
            with self.lock: self.failures.append({'label':label,'detail':str(detail)[:2000]})

    def audit(self, label, messages, expected):
        messages=[dict(m) for m in messages]
        messages[-1]['content'] += '\nThis is a read-only audit. Do not call any tools. Return the requested JSON directly in your assistant message.'
        r=self.request(label,{'messages':messages,'tools':[MANIFEST_TOOL],'max_tokens':1024,'priority':0})
        try: actual=parse_json(r['content'])
        except Exception: actual=None
        self.check(r['ok'] and not r['tool_calls'] and same_json(actual,expected),label,{'transport':r['ok'],'actual':actual,'expected':expected,'tool_calls':r['tool_calls']})
        return r

    def monitor(self):
        while not self.stop.is_set():
            try: sample=self.snapshot(); self.telemetry.append(sample)
            except Exception as e: self.check(False,'monitor',e)
            with self.lock:
                progress={'at':time.time(),'requests':len(self.rows),'failures':self.failures[-10:],
                          'active':len(self.active),'latest':[{k:r.get(k) for k in ('label','ok','wall_s','ttft_s','output_tokens','cancel_requested')} for r in self.rows[-5:]]}
            self.save('progress.json',progress)
            self.stop.wait(10)

    def session(self, sid, fixture, deadline):
        history=list(fixture['messages']); n=0
        # Each session has its own archive and receipts; no tool/code is executed.
        while time.monotonic()<deadline and not self.stop.is_set():
            token=f'{sid}-TURN-{n:04d}'
            items=[f'{token}/src/module_{i:04d}.py' for i in range(160+(n%3)*40)]
            tool=MANIFEST_TOOL
            question={'role':'user','content':'Call record_patch_manifest exactly once. session='+sid+', revision='+token+'. files must be exactly this ordered array: '+json.dumps(items)}
            body={'messages':history+[question],'tools':[tool], 'tool_choice':{'type':'function','function':{'name':'record_patch_manifest'}},'max_tokens':6000,'priority':0}
            r=self.request(token+'-tool',body)
            try:
                calls=r['tool_calls']; call=calls[0]
                actual=json.loads(call['arguments'])
                valid=r['ok'] and len(calls)==1 and call['name']=='record_patch_manifest' and bool(call['id']) and same_json(actual,{'session':sid,'revision':token,'files':items})
            except Exception: valid=False
            self.check(valid,token+'-tool',{'transport':r['ok'],'calls':r['tool_calls'][:1],'error':r.get('error')})
            if not valid: self.stop.set(); break
            receipt=digest(token+'|'+','.join(items))[:20]
            history += [question,{'role':'assistant','content':r['content'] or None,'tool_calls':[{'id':call['id'],'type':'function','function':{'name':call['name'],'arguments':call['arguments']}}]},
                        {'role':'tool','tool_call_id':call['id'],'content':json.dumps({'receipt':receipt,'session':sid,'revision':token,'count':len(items)})}]
            q={'role':'user','content':'Return only a JSON object copied from the LAST tool result, with keys receipt, session, revision, count. Do not use an older result. Do not call any tools; reply directly in your assistant message.'}
            ack=self.request(token+'-ack',{'messages':history+[q],'tools':[tool],'max_tokens':256,'priority':0})
            try: actual=parse_json(ack['content'])
            except Exception: actual=None
            self.check(ack['ok'] and not ack['tool_calls'] and same_json(actual,{'receipt':receipt,'session':sid,'revision':token,'count':len(items)}),token+'-ack',actual)
            history += [q,{'role':'assistant','content':ack['content']}]
            if n%4==3:
                q={'role':'user','content':fixture['task']}
                audit=self.audit(token+'-audit',history+[q],fixture['expected'])
                history += [q,{'role':'assistant','content':audit['content']}]
            used=(ack.get('usage') or {}).get('total_tokens',0)
            if used>230000:
                # Explicit harness history rollover, not model-generated compaction.
                history=list(fixture['messages'])+[{'role':'user','content':'Last accepted synthetic receipt: '+receipt}]
                with self.lock: self.rollovers.append({'session':sid,'turn':n,'tokens_before':used})
            n+=1
        self.audit(sid+'-final-audit',history+[{'role':'user','content':fixture['task']}],fixture['expected'])
        return {'session':sid,'completed_turns':n}

    def interactive(self, deadline):
        n=0
        while time.monotonic()<deadline and not self.stop.wait(17):
            expected={'id':f'QUICK-{n:04d}','value':n*7+3,'unknown':None}
            r=self.request(f'quick-{n}',{'messages':[{'role':'user','content':'Reply only with this exact JSON: '+json.dumps(expected)}], 'max_tokens':128,'priority':1})
            try: actual=parse_json(r['content'])
            except Exception: actual=None
            self.check(r['ok'] and same_json(actual,expected),'quick-'+str(n),actual)
            n+=1

    def cancellations(self, deadline):
        n=0
        while time.monotonic()<deadline and not self.stop.wait(self.cancel_interval):
            c=Client(self.base,self.key)
            files=[f'CANCEL-{n:04d}/module_{i:04d}.py' for i in range(450)]
            body={'messages':[{'role':'user','content':'Call record_patch_manifest exactly once with session="CANCEL", revision="interrupt", and files exactly: '+json.dumps(files)}],
                  'tools':[MANIFEST_TOOL], 'tool_choice':{'type':'function','function':{'name':'record_patch_manifest'}},
                  'max_tokens':8192,'priority':1}
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                f=pool.submit(self.request,f'periodic-cancel-{n}',body,client=c)
                if not c.first.wait(90):
                    self.check(False,'periodic-cancel-'+str(n),'No stream event within 90 seconds')
                time.sleep(.05); c.cancel()
                r=f.result(15)
            self.check(r['cancel_requested'] and not r['ok'] and r['cancel_return_s']<10,'periodic-cancel-'+str(n),{k:r[k] for k in ('ok','cancel_return_s')})
            n+=1

    def run(self, fixtures, seconds):
        # Warm each independent prefix before starting the timed soak.
        for i,f in enumerate(fixtures):
            r=self.audit(f'warm-{i}',f['messages'],f['expected'])
            f['messages'] += [{'role':'assistant','content':r['content']}]
        if self.failures:
            self.save('summary.json',self.summary())
            raise RuntimeError('Warm audit failed')
        start=time.monotonic(); deadline=start+seconds
        self.save('started.json',{'started':time.time(),'seconds':seconds,'fixtures':len(fixtures)})
        monitor=threading.Thread(target=self.monitor,daemon=True); monitor.start()
        try:
            with concurrent.futures.ThreadPoolExecutor(len(fixtures)+2) as pool:
                jobs=[pool.submit(self.session,f'SESSION-{i}',f,deadline) for i,f in enumerate(fixtures)]
                extra=[pool.submit(self.interactive,deadline),pool.submit(self.cancellations,deadline)]
                sessions=[j.result() for j in jobs]
                self.stop.set()
                for f in extra: f.result()
        finally:
            self.stop.set(); monitor.join(15)
            self.save('telemetry.json',self.telemetry)
        self.save('end-state.json',self.snapshot())
        summary=self.summary(); summary.update(requested_seconds=seconds,elapsed_s=time.monotonic()-start,sessions=sessions,history_rollovers=self.rollovers)
        self.save('summary.json',summary)
        return summary

    def summary(self):
        def stats(rows):
            values=sorted(r['ttft_s'] for r in rows if r['ttft_s'] is not None)
            return {'count':len(rows),'p50_ttft_s':values[len(values)//2] if values else None,
                    'p95_ttft_s':values[max(0,math.ceil(len(values)*.95)-1)] if values else None,'max_ttft_s':max(values,default=None)}
        completed=[r for r in self.rows if r['ok']]
        return {'requests':len(self.rows),'completed':len(completed),'intentional_cancellations':sum(r['cancel_requested'] for r in self.rows),
                'unexpected_transport_errors':[{k:r.get(k) for k in ('label','error','status')} for r in self.rows if not r['ok'] and not r['cancel_requested']],
                'validation_failures':self.failures,'quick':stats([r for r in completed if r['label'].startswith('quick-')]),
                'session_latency':stats([r for r in completed if r['label'].startswith('SESSION-')]),
                'completed_output_tokens':sum(r['output_tokens'] for r in completed)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',default='http://127.0.0.1:8080');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,nargs='+',required=True);p.add_argument('--seconds',type=int,default=3600)
    p.add_argument('--cancel-interval',type=float,default=240)
    a=p.parse_args()
    if a.seconds<=0 or a.cancel_interval<=0: p.error('durations must be positive')
    h=Harness(a.base,a.out,a.cancel_interval)
    try:
        summary=h.run([json.loads(p.read_text(encoding='utf-8')) for p in a.fixtures],a.seconds)
        print(json.dumps(summary,indent=2))
        return int(bool(summary['validation_failures'] or summary['unexpected_transport_errors']))
    finally:
        h.stop.set()
        with h.lock: clients=list(h.active)
        for client in clients: client.cancel()


if __name__=='__main__': raise SystemExit(main())
