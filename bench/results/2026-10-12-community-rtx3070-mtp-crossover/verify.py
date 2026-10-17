"""Check the published evidence independently of the plotting libraries."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
results = json.loads((DATA / 'results.json').read_text())
summary = json.loads((DATA / 'summary.json').read_text())
assert results['status'] == 'complete'
assert len(results['cases']) == 6
assert len(summary) == 3
cases = {}
for c in results['cases']:
    p, mode = c['input_target'], c['mtp']
    assert c['status'] == 'complete'
    a = json.loads((DATA / f'{c["name"]}-arrivals.json').read_text())
    ids = json.loads((DATA / f'{c["name"]}-ids.json').read_text())
    assert len(a) == len(ids) == c['output_tokens'] == c['timings']['generated']
    assert a[0] > 0 and all(left <= right for left, right in zip(a, a[1:]))
    assert abs(a[-1] - c['wall_seconds']) < .01
    assert a[0] == c['first_token_seconds']
    t = c['timings']
    assert t['prompt_tokens'] == t['prompt_read'] == p
    assert t['reused'] == 0
    if not mode:
        assert t['drafts_offered'] == t['drafts_accepted'] == 0
    assert math.isclose(c['prefill_tps'], p * 1000 / t['prompt_ms'])
    assert math.isclose(c['decode_tps'], len(ids) * 1000 / t['decode_ms'])
    output = (DATA / f'{c["name"]}-output.txt').read_text(encoding='utf-8')
    assert hashlib.sha256(output.encode()).hexdigest() == c['output_sha256']
    cases[(p, mode)] = (c, a, ids)

for s in summary:
    p = s['prompt']
    off, oa, oi = cases[(p, False)]
    on, na, ni = cases[(p, True)]
    common = min(len(oa), len(na))
    assert common == s['common_output_length']
    savings = [a-b for a,b in zip(oa,na)]
    wins = [i+1 for i,v in enumerate(savings) if v > 0]
    losses = [i+1 for i,v in enumerate(savings) if v <= 0]
    assert (wins[0] if wins else None) == s['first_observed_mtp_win']
    stable = max(losses, default=0) + 1
    assert (stable if stable <= common else None) == s['mtp_faster_after']
    divergence = next((i for i,(a,b) in enumerate(zip(oi,ni)) if a != b),None)
    assert divergence == s['first_output_divergence_zero_based']
    for q in s['checkpoints']:
        n = q['output']
        assert q['off_seconds'] == oa[n-1]
        assert q['on_seconds'] == na[n-1]
        assert math.isclose(q['mtp_saved_seconds'], oa[n-1]-na[n-1])
assert cases[(65536, False)][0]['output_tokens'] == 759
assert cases[(65536, False)][0]['timings']['finish'] == 'stop'
assert next(s for s in summary if s['prompt']==65536)['mtp_faster_after'] is None
assert results['commit'] == '61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406'
for path in HERE.rglob('*'):
    if path.suffix in ('.py','.md','.json','.log','.txt'):
        text = path.read_text(encoding='utf-8')
        if path.is_relative_to(DATA):
            assert '/home/' not in text and 'C:/Users/' not in text, path
        if path.suffix == '.py':
            ast.parse(text, filename=str(path))
for target in re.findall(r'\]\(([^)]+)\)', (HERE/'README.md').read_text()):
    if not target.startswith(('http:', 'https:', '#')):
        assert (HERE/target.split('#',1)[0]).exists(), target
assert (HERE/'mtp-crossover.png').stat().st_size > 10000
assert (HERE/'mtp-crossover.svg').stat().st_size > 10000
print('PASS: six runs, full arrivals/IDs, timing arithmetic, output hashes, crossovers, censoring, links and syntax')
