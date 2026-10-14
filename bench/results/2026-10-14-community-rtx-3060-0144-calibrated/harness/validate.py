"""Validate saved HTTP benchmark evidence without running inference."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def artifact(folder, name):
    plain=folder/'requests'/name
    if plain.exists():return read(plain)
    with gzip.open(str(plain)+'.gz','rt',encoding='utf-8') as file:return json.load(file)

def validate(folder, runtime=None):
    rows=[json.loads(line) for line in (folder/'rows.jsonl').read_text(encoding='utf-8').splitlines()] if (folder/'rows.jsonl').exists() else read(folder/'per-request.json')
    protocol=read(folder/'protocol.json')
    complete=read(folder/'completion.json')
    cleanup=read(folder/'cleanup.json')
    expected=[f'{model}-{ctx}' for model in ('q2_0','iq2_xs','iq3_xxs') for ctx in (131072,262144)]
    assert complete['finished'] and complete['sessions_completed']==expected
    assert complete['failed_requests']==0 and all(row['ok'] for row in rows)
    assert all(cleanup[key] for key in ('owned_server_stopped','owned_worker_stopped','port18080_closed'))
    speeds=[row for row in rows if row['group']=='speed']
    assert len(speeds)==36
    for model in ('q2_0','iq2_xs','iq3_xxs'):
        for context,targets in ((131072,(4096,32768,120000)),(262144,(260000,))):
            for target in targets:
                group=[row for row in speeds if (row['model'],row['context'],row['target'])==(model,context,target)]
                assert sorted(row['repetition'] for row in group)==[0,1,2]
                assert all(target-20<=row['engine']['prompt_tokens']<=target for row in group)
    for row in rows:
        stem=row['output_file'].removesuffix('.response.json')
        body=artifact(folder,stem+'.json')
        output=artifact(folder,stem+'.response.json')
        request_hash=hashlib.sha256(json.dumps({key:value for key,value in body.items() if key!='model'},sort_keys=True).encode()).hexdigest()
        assert request_hash==row['request_sha256']
        assert output['ok'] and output['reasoning_content']
        assert row['engine']['prompt_tokens']==row['expected_prompt_tokens']
        assert 0<row['engine']['engine_generated']<=body['max_tokens']
        assert row['engine']['finish'] not in ('disconnect','error','cancelled')
        assert body['reasoning_effort']=='xhigh'
        assert all(body[key]==value for key,value in protocol['sampling'].items())
        if row['group'] in ('speed','needle','soak'):assert row['engine']['reused']==0
        if row['group']=='speed':assert body['seed']==2718+row['repetition'] and body['max_tokens']==512
        if row['group']=='needle':assert row['recall_pass']==(row['expected_code_word'] in output['content'])
    needles=[row for row in rows if row['group']=='needle']
    assert len(needles)==3 and all(row['context']==262144 and row['target']==260000 and row['repetition']==50 for row in needles)
    conversation=[row for row in rows if row['group']=='conversation']
    assert len(conversation)==4 and all(row['session']=='iq2_xs-131072' for row in conversation)
    assert all(row['engine']['reused']>0 for row in conversation[1:])
    soak=read(folder/'soak-completion.json')
    assert soak['completed'] and soak['elapsed_seconds']>=900 and soak['no_idle_delays']
    assert len([row for row in rows if row['group']=='soak'])==soak['requests']
    cross=[]
    for context,targets in ((131072,(4096,32768,120000)),(262144,(260000,))):
        for target in targets:
            for rep in range(3):
                hashes={row['request_sha256'] for row in speeds if (row['context'],row['target'],row['repetition'])==(context,target,rep)}
                assert len(hashes)==1
                cross.append(dict(context=context,target=target,rep=rep,hash=next(iter(hashes))))
    loaded=[]
    for session in expected:
        engine=read(folder/'servers'/f'{session}-loaded.json')['metrics']['engine']
        assert engine['version']=='0.1.44.1' and engine['context']==int(session.split('-')[-1])
        assert engine['kv']=='int8' and engine['kv_resident']==32768
        assert engine['pool_workers']==11
        if session.startswith('iq3_xxs'):assert 16000<=engine['arena_mib']<=16384
        else:assert engine['arena_mib']>30000
        loaded.append(dict(session=session,arena_mib=engine['arena_mib'],pool_workers=engine['pool_workers'],
            pcie_frac=engine['pcie_frac'],spec_min_p=engine['spec_min_p']))
    if runtime:
        assert hashlib.sha256((runtime/'engine/strata.exe').read_bytes()).hexdigest()==protocol['engine_sha256']
        assert hashlib.sha256((runtime/'strata-iq2_xs.json').read_bytes()).hexdigest()==protocol['normal_config_sha256']
    result=dict(speed_requests=36,recall_cases=3,recall_passes=sum(row['recall_pass'] for row in needles),
        conversation_turns=4,conversation_reuse_verified=True,soak_requests=soak['requests'],soak_seconds=soak['elapsed_seconds'],
        successful_requests=len(rows),all_outputs_checked=True,cross_model_input_hashes=cross,loaded=loaded,
        owned_processes_stopped=True,normal_config_checked=runtime is not None)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--runtime',type=Path)
    parser.add_argument('--out',type=Path,help='Optional new validation record; saved evidence is not overwritten by default')
    args=parser.parse_args()
    result=validate(args.folder,args.runtime)
    if args.out:args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:value for key,value in result.items() if key not in ('cross_model_input_hashes','loaded')},indent=2))
