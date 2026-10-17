"""Verify the published rule against raw measurements and output capacity."""
import ast
import json
import math
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
DATA=HERE/'data'
r=json.loads((DATA/'results.json').read_text())
model=json.loads((DATA/'decision-model.json').read_text())
cases={(c['input_target'],c['mtp']):c for c in r['cases']}
assert r['max_context']==model['max_context']==133120
assert r['kv_resident']==32768
assert len(model['rows'])==4
for row in model['rows']:
    length=row['input_tokens']
    off,on=cases[(length,False)],cases[(length,True)]
    p=(on['timings']['prompt_ms']-off['timings']['prompt_ms'])/1000
    saving=1/off['decode_tps']-1/on['decode_tps']
    boundary=p/saving
    assert math.isclose(boundary,row['projected_break_even_tokens'])
    assert math.isclose(boundary*saving-p,0,abs_tol=1e-9)
    first=row['minimum_projected_profitable_integer_output']
    assert first*saving-p>0 and (first-1)*saving-p<=0
    capacity=133120-length
    assert capacity==row['tested_output_capacity']
    assert row['projected_break_even_fits']==(boundary<=capacity)
    oa=json.loads((DATA/f'{length}-off-arrivals.json').read_text())
    na=json.loads((DATA/f'{length}-on-arrivals.json').read_text())
    for q in row['decisions']:
        n=q['output_tokens']
        expected=oa[n-1]-na[n-1] if n<=1000 else n*saving-p
        assert math.isclose(q['mtp_on_savings_seconds'],expected,abs_tol=1e-9)
        assert q['fits_tested_context']==(n<=capacity)
        if n<=1000:assert q['mode']=='off'
        if not q['fits_tested_context']:assert q['mode']=='outside tested capacity'
        if n>1000:
            assert math.isclose(q['projected_off_seconds']-q['projected_on_seconds'],expected,abs_tol=1e-9)
    for repeat in row['repeated_uncached_1000_calls']:
        assert math.isclose(repeat['seconds_saved_using_mtp_off'],
                            repeat['calls']*(on['wall_seconds']-off['wall_seconds']))
assert all(not s['decisions'][-1]['fits_tested_context'] for s in model['rows'])
assert model['rows'][-1]['tested_output_capacity']==2048
assert not model['rows'][-1]['projected_break_even_fits']
assert model['rows'][0]['decisions'][3]['mode']=='on candidate'
for path in HERE.rglob('*'):
    if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    if path.suffix in ['.log','.json']:
        text=path.read_text(encoding='utf-8')
        assert '/home/' not in text and 'C:/Users/' not in text,path
for target in re.findall(r'\]\(([^)]+)\)',(HERE/'README.md').read_text(encoding='utf-8')):
    if not target.startswith(('http:','https:','#')):
        assert (HERE/target.split('#',1)[0]).exists(),target
print('PASS: four projected crossings, prefix timings, long-output arithmetic, repeated-call totals, capacity gates, privacy and links')
