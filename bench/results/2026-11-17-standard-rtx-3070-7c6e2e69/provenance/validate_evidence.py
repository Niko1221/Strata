"""Independently inspect standard evidence without sending inference requests."""
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

folder = Path(sys.argv[1])
doc = json.loads((folder / 'results.json').read_text(encoding='utf-8'))
assert doc['protocol']['id'] == 'strata-qwen38-thinking-v1-draft.1'
assert doc['protocol']['corpus_sha256'] == 'e75eff8b9feea04f745520b392c4c17d97aefc1ba185e222aa6740efa8f68407'
assert doc['configuration']['track'] == 'custom'
assert 'flash-next' in doc['model']['id'].lower()
assert doc['configuration']['sampling'] == {
    'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0,
    'presence_penalty': 0.0, 'repetition_penalty': 1.0, 'reasoning_effort': 'xhigh'}
assert doc['configuration']['chat_template_kwargs']['enable_thinking'] is True
assert doc['configuration']['chat_template_kwargs']['preserve_thinking'] is True
assert doc['extensions']['started_engine'] is True
assert doc['extensions']['server_requests_before_run'] == 0
assert hashlib.sha256((folder / 'provenance/corpus-manifest.sha256').read_bytes()).hexdigest() == doc['protocol']['corpus_sha256']
records = {r['id']: r for r in doc['requests']}
assert len(records) == 34
for a in doc['artifacts']:
    p = folder / a['path']
    assert p.resolve().is_relative_to(folder.resolve())
    assert hashlib.sha256(p.read_bytes()).hexdigest() == a['sha256'], p
evidence = {r['request_id']: r for r in map(json.loads, (folder / 'outputs.jsonl').read_text(encoding='utf-8').splitlines())}
assert records.keys() == evidence.keys()
for r in records.values():
    phase = r['phase']
    assert r['max_output_tokens'] == (128 if phase == 'warmup' else 2048 if phase == 'sanity' else 512)
    if phase == 'performance':
        assert r['seed'] == r['repetition']
        assert r['concurrency'] == 1
    raw = evidence[r['id']]
    assert raw['finish_reason'] == r['finish_reason']
    if r['status'] in ('completed', 'truncated'):
        assert raw['http_status'] == 200
    for metric, count, duration in [('prompt_tps','read_tokens','prompt_s'), ('decode_tps','decode_tokens','decode_s')]:
        actual = r['metrics'][metric]
        if actual is not None:
            assert math.isclose(actual, r['tokens'][count]/r['timings'][duration], abs_tol=0.02)
for summary in doc['summaries']:
    values = []
    for rid in summary['included_request_ids']:
        r = records[rid]
        metric = summary['metric']
        values.append(r['metrics'][metric] if metric.endswith('_tps') else r['timings'][metric])
    assert len(values) == summary['n']
    if values:
        for k, v in [('median', statistics.median(values)), ('min',min(values)), ('max',max(values))]:
            assert math.isclose(summary[k], v, abs_tol=0.00001), (summary,k)

def strict_json(text):
    def pairs(items):
        out = {}
        for k,v in items:
            if k in out: raise ValueError('duplicate key')
            out[k]=v
        return out
    return json.loads(text, object_pairs_hook=pairs, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))

def same(a,b):
    if type(a) is not type(b): return False
    if isinstance(a,dict): return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,list): return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

expected = {'I1': {'ok':True,'count':3}, 'I2':[-4,1,2,9], 'I3':{'project':'Orion','build':17},
            'I4':'READY', 'R1':'ORBIT-731', 'R2':'CEDAR-284', 'R3':'DELTA-906',
            'R4':{'north':'N-124','south':'S-875'},
            'T1':('lookup_record',{'id':'A-17'}), 'T2':('get_weather',{'city':'Zagreb','unit':'celsius'}),
            'T3':('convert_units',{'value':3,'from_unit':'km','to_unit':'m'}),
            'T4':('get_build_status',{'project':'Orion','build':17})}
scores = {}
for r in records.values():
    if r['phase'] != 'sanity': continue
    raw=evidence[r['id']]; target=expected[r['case_id']]; passed=False
    if r['status']=='completed':
        try:
            if isinstance(target,tuple):
                calls=raw['tool_calls']
                passed=len(calls)==1 and calls[0]['name']==target[0] and same(strict_json(calls[0]['arguments']),target[1])
            elif isinstance(target,(list,dict)):
                passed=same(strict_json(raw['final_answer']),target)
            else:
                passed=raw['final_answer'].strip(' \t\r\n\v\f')==target
        except (ValueError,KeyError,TypeError):
            passed=False
    assert passed is r['evaluation']['passed'], r['id']
    scores[r['case_id']]=passed
assert len(scores)==12 and sum(scores.values())==doc['quality']['passed']
checksum_file=folder/'SHA256SUMS.txt'
if checksum_file.exists():
    for line in checksum_file.read_text(encoding='utf-8-sig').splitlines():
        digest,name=line.split('  ',1)
        target=folder/name
        assert target.resolve().is_relative_to(folder.resolve())
        assert hashlib.sha256(target.read_bytes()).hexdigest()==digest, name
for p in folder.rglob('*'):
    if not p.is_file(): continue
    if p.suffix == '.py': continue  # scanner source contains the literal patterns it checks
    text=p.read_text(encoding='utf-8')
    assert not re.search(r'C:[\\/]+Users[\\/]+[^<\\/]|GPU-[0-9a-f]{8}|gho_[A-Za-z0-9]|github_pat_', text, re.I), p
print(json.dumps({'validated':True,'requests':len(records),'status':doc['run']['status'],
                  'independent_sanity':scores,'passed':sum(scores.values()),
                  'artifact_hashes':len(doc['artifacts'])},indent=2))
