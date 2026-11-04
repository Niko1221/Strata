"""Offline checks; no inference or network calls."""
import hashlib,json,pathlib,statistics
root=pathlib.Path(__file__).resolve().parent
data={name:json.loads((root/(name+'-results.json')).read_text()) for name in ('baseline','resident14','resident14_stress')}
assert len(data['baseline'])==len(data['resident14'])==6
assert len(data['resident14_stress'])==2
for name,rows in data.items():
    for row in rows:
        t=row['timings'];u=row['usage']
        assert t['cache_n']==u['prompt_tokens_details']['cached_tokens']==0
        assert t['prompt_n']==u['prompt_tokens']
        assert u['completion_tokens']==t['predicted_n']==8
        assert abs(t['prompt_n']/t['prompt_ms']*1000-t['prompt_per_second'])<.2
        fixture=root/f"prompt-{row['target']}-{row['repeat']-1}.txt"
        assert hashlib.sha256(fixture.read_text(encoding='utf-8').encode()).hexdigest()==row['prompt_sha256']
for baseline,resident in zip(data['baseline'],data['resident14']):
    assert baseline['prompt_sha256']==resident['prompt_sha256']
    assert baseline['usage']['prompt_tokens']==resident['usage']['prompt_tokens']
base=json.loads((root/'baseline-config.json').read_text())
resident=json.loads((root/'resident14-config.json').read_text())
assert resident['args']==base['args']+['--resident-budget-gib','14']
assert resident['vision']==base['vision']
for cfg in (base,resident):
    assert cfg['args'][cfg['args'].index('--max-context')+1]=='131072'
    assert cfg['args'][cfg['args'].index('--kv')+1]=='int8'
for size in (4096,16384):
    a=statistics.median(r['timings']['prompt_per_second'] for r in data['baseline'] if r['target']==size)
    b=statistics.median(r['timings']['prompt_per_second'] for r in data['resident14'] if r['target']==size)
    print(f'{size}: {a:.1f} -> {b:.1f} tok/s; gain {(b/a-1)*100:.1f}%')
recall=json.loads((root/'recall-result.json').read_text())
assert hashlib.sha256((root/'recall-prompt.txt').read_text(encoding='utf-8').encode()).hexdigest()==recall['prompt_sha256']
assert 120000 <= data['resident14_stress'][-1]['usage']['prompt_tokens'] < 131072
print('PASS: 14 performance requests, matched input hashes/counts, zero cached tokens, timing arithmetic, context/KV and sole argument delta. Recall:',recall['matched'])
