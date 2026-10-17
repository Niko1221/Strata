import argparse, hashlib, json, math
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
args=p.parse_args()
d=args.directory
r=json.loads((d/'results.json').read_text())
assert r['status']=='complete'
assert len(r['cases'])==8
cases={}
for c in r['cases']:
    assert c['status']=='complete',c
    a=json.loads((d/(c['name']+'-arrivals.json')).read_text())
    ids=json.loads((d/(c['name']+'-ids.json')).read_text())
    assert len(a)==len(ids)==c['output_tokens']==c['timings']['generated']==1000
    assert c['requested_1000_completed']
    assert c['timings']['reused']==0
    assert c['timings']['prompt_read']==c['input_tokens']==c['input_target']
    assert len(json.loads((d/f'prompt-{c["input_target"]}-ids.json').read_text()))==c['input_target']
    assert a[0]==c['first_token_seconds']
    assert all(x<=y for x,y in zip(a,a[1:]))
    assert abs(a[-1]-c['wall_seconds'])<.01
    output=(d/(c['name']+'-output.txt')).read_text(encoding='utf-8')
    assert hashlib.sha256(output.encode()).hexdigest()==c['output_sha256']
    text=(d/f'prompt-{c["input_target"]}.txt').read_text(encoding='utf-8')
    assert hashlib.sha256(text.encode()).hexdigest()==c['prompt_sha256']
    assert math.isclose(c['decode_tps'],1000000/c['timings']['decode_ms'])
    if not c['mtp']:assert c['timings']['drafts_offered']==0
    cases[(c['input_target'],c['mtp'])]=(c,ids)
summary=[]
for length in [49152,65536,98304,131072]:
    off,off_ids=cases[(length,False)]
    on,on_ids=cases[(length,True)]
    assert off['prompt_sha256']==on['prompt_sha256']
    delta=on['wall_seconds']-off['wall_seconds']
    phase_delta=(on['timings']['prompt_ms']-off['timings']['prompt_ms']
                +on['timings']['decode_ms']-off['timings']['decode_ms'])/1000
    assert abs(delta-phase_delta)<.1
    s=dict(input_tokens=length,output_tokens=1000,off_seconds=off['wall_seconds'],on_seconds=on['wall_seconds'],
           seconds_saved_using_mtp_off=delta,
           off_reduction_vs_on_percent=100*delta/on['wall_seconds'],
           on_slowdown_vs_off_percent=100*delta/off['wall_seconds'],
           off_first_token_seconds=off['first_token_seconds'],on_first_token_seconds=on['first_token_seconds'],
           prefill_penalty_seconds=(on['timings']['prompt_ms']-off['timings']['prompt_ms'])/1000,
           decode_seconds_saved_using_mtp_on=(off['timings']['decode_ms']-on['timings']['decode_ms'])/1000,
           off_decode_tps=off['decode_tps'],on_decode_tps=on['decode_tps'],
           first_output_divergence_zero_based=next((i for i,(a,b) in enumerate(zip(off_ids,on_ids)) if a!=b),None))
    summary.append(s)
    print(json.dumps(s))
(d/'summary.json').write_text(json.dumps(summary,indent=2))
print('PASS: eight full 1,000-token runs, exact prompt lengths, matched pair inputs, hashes, timings and savings')
