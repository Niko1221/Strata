"""Cancellation and Responses tool round trips against an idle test server.

Requires solo_priority=true. Does not restart the server or execute model tools.
Credentials come from STRATA_API_KEY. Use a dedicated test instance, not one
serving users: these tests generate and cancel long synthetic requests.
"""
import argparse, concurrent.futures, json, time, traceback
from pathlib import Path
from soak_solo_priority import Client, Harness, parse_json


def wait(h,check,seconds=30):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        s=h.snapshot()
        if check(s):return s
        time.sleep(.2)
    raise RuntimeError('Timed out waiting for lifecycle state')

def sched(s):return s['metrics']['live']['solo_priority']
def idle(h):return wait(h,lambda s:not s['status']['activity']['in_flight'] and not sched(s)['waiting'] and not sched(s)['paused'])
def hold(priority=0):
    return {'messages':[{'role':'user','content':'Output exactly these labels, one per line, without omissions: '+','.join(f'ROW-{i:05d}' for i in range(2000))}],
            'max_tokens':8192,'priority':priority}
def quick(h,label,priority=0):
    expected={'check':label,'value':42,'unknown':None}
    r=h.request(label,{'messages':[{'role':'user','content':'Reply only this exact JSON: '+json.dumps(expected)}], 'priority':priority,'max_tokens':128})
    assert r['ok'] and parse_json(r['content'])==expected,r
    return r
def tool_body(label,priority=5,responses=False):
    values=[f'ITEM-{i:04d}' for i in range(450)]
    parameters={'type':'object','properties':{'session':{'type':'string'},'items':{'type':'array','items':{'type':'string'}}},'required':['session','items']}
    message={'role':'user','content':'Call record_items exactly once. Set session to '+label+' and items to this exact ordered array: '+json.dumps(values)}
    function={'name':'record_items','description':'Synthetic manifest, never executed.','parameters':parameters}
    if responses:
        body={'input':[message],'tools':[{'type':'function',**function}],'tool_choice':{'type':'function','name':'record_items'},'max_output_tokens':6000,'priority':priority}
    else:
        body={'messages':[message],'tools':[{'type':'function','function':function}], 'tool_choice':{'type':'function','function':{'name':'record_items'}}, 'max_tokens':6000,'priority':priority}
    return body,{'session':label,'items':values}
def valid_tool(row,expected):
    assert row['ok'] and len(row['tool_calls'])==1,row
    call=row['tool_calls'][0]
    assert call['name']=='record_items' and call['id'] and json.loads(call['arguments'])==expected,row
    return call

def cancellations(h):
    for case in ('queued','active','paused'):
        idle(h)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            victim=Client(h.base,h.key)
            if case=='queued':
                body,want=tool_body('QUEUED-SURVIVOR',5)
                survivor_client=Client(h.base,h.key)
                survivor=pool.submit(h.request,'queued-survivor',body,client=survivor_client)
                assert survivor_client.first.wait(90)
                doomed=pool.submit(h.request,'cancel-queued',hold(),client=victim)
                h.save('queued-before-cancel.json',wait(h,lambda s:sched(s)['waiting']>=1))
            else:
                doomed=pool.submit(h.request,'cancel-'+case,hold(),client=victim)
                assert victim.first.wait(90)
                if case=='active':
                    survivor=pool.submit(quick,h,'active-survivor',0)
                    h.save('active-before-cancel.json',wait(h,lambda s:sched(s)['waiting']>=1 and sched(s)['paused']==0))
                else:
                    body,want=tool_body('PAUSED-SURVIVOR',5)
                    survivor=pool.submit(h.request,'paused-survivor',body)
                    h.save('paused-before-cancel.json',wait(h,lambda s:sched(s)['paused']>=1))
            victim.cancel(); cancelled=doomed.result(15)
            assert cancelled['cancel_requested'] and not cancelled['ok'] and cancelled['cancel_return_s']<10,cancelled
            survived=survivor.result(120)
            if case!='active':valid_tool(survived,want)
        h.save(case+'-idle.json',idle(h));quick(h,case+'-recovered')

def responses(h):
    idle(h);before=sched(h.snapshot())['preemptions']
    body,want=tool_body('RESPONSES-ROUNDTRIP',0,True)
    client=Client(h.base,h.key)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        long=pool.submit(h.request,'responses-tool',body,'responses',client)
        assert client.first.wait(90);time.sleep(6)
        short=pool.submit(quick,h,'responses-interrupter',5)
        small=short.result(40);row=long.result(120)
    call=valid_tool(row,want)
    assert sched(idle(h))['preemptions']>before,'Responses call was not preempted'
    receipt={'receipt':'RESP-RECEIPT-731','count':450}
    follow={**body,'input':body['input']+[
        {'type':'function_call','call_id':call['id'],'name':call['name'],'arguments':call['arguments']},
        {'type':'function_call_output','call_id':call['id'],'output':json.dumps(receipt)},
        {'role':'user','content':'Reply only with JSON copied from the latest tool result. Do not call tools.'}],
        'tool_choice':'auto','max_output_tokens':256}
    r=h.request('responses-ack',follow,'responses')
    assert r['ok'] and not r['tool_calls'] and parse_json(r['content'])==receipt,r
    h.save('responses-check.json',{'one_call':True,'items_correct':450,'tool_result_recalled':True,'short_ttft_s':small['ttft_s']})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',default='http://127.0.0.1:8080')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    h=Harness(a.base,a.out)
    try:
        state=idle(h)
        assert sched(state)['enabled'], 'Enable solo_priority on the test server'
        cancellations(h)
        responses(h)
        h.save('summary.json',{'success':True,'cancel_cases':['queued','active','paused'],
                              'responses':True,'requests':h.summary()})
        return 0
    except Exception:
        with h.lock: clients=list(h.active)
        for client in clients: client.cancel()
        h.save('summary.json',{'success':False,'error':traceback.format_exc(),'requests':h.summary()})
        raise

if __name__=='__main__': raise SystemExit(main())
