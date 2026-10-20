# Run from the repository root with existing local model artifacts.
import json, re, statistics
from pathlib import Path
import os, hashlib
out = Path(os.environ.get('STRATA_DRAFT_RESULTS', Path(__file__).resolve().parent))
summary = {}
allrows = {}
for tag in ['before', 'after', 'mtp']:
    data = json.loads((out / f'{tag}.json').read_text())
    rows = data['rows']
    allrows[tag] = rows
    profiles = re.findall('strata decode timing: (\\d+) windows,.*?([\\d.]+) ms/window = verify ([\\d.]+).*? \\+ draft ([\\d.]+);', (out / f'{tag}.log').read_text())
    assert len(profiles) == len(rows), (tag, len(profiles), len(rows))
    for row, profile in zip(rows, profiles):
        assert len(row['tokens']) == 96 and 'error' not in row
        row['profile'] = dict(zip(['windows', 'window_ms', 'verify_ms', 'draft_ms'], map(float, profile)))
    summary[tag] = {'expert_slots': data['info']['expert_slots'], 'tasks': {}}
    for task in ['code', 'long']:
        rr = [r for r in rows if r['task'] == task and 'rep' in r]
        assert len(rr) == 3
        speeds = [96000 / r['decode_ms'] for r in rr]
        summary[tag]['tasks'][task] = {'tok_s_median': statistics.median(speeds), 'tok_s_range': [min(speeds), max(speeds)], 'profile_medians': {k: statistics.median((r['profile'][k] for r in rr)) for k in rr[0]['profile']}, 'accepted_offered': [(r['accepted'], r['offered']) for r in rr]}
checks = {}
for task in ['code', 'long']:
    controls = [r for r in allrows['after'] if r['task'] == task and r.get('control')]
    assert len(controls) == 1
    control = controls[0]
    assert control['offered'] == control['accepted'] == 0
    checks[task] = all((r['tokens'] == control['tokens'] for rows in allrows.values() for r in rows if r['task'] == task))
    assert checks[task], task
checks['equal_before_after_expert_slots'] = summary['before']['expert_slots'] == summary['after']['expert_slots']
micro = {}
for tag in ['before', 'after']:
    for rep in range(3):
        for rows, end, ms, fingerprint in re.findall('attention rows=(\\d+) heads=32 end=(\\d+): ([\\d.]+) ms/pass bits=([a-f0-9]+)', (out / f'micro-{tag}-{rep}.log').read_text()):
            micro.setdefault(f'rows={rows},end={end}', {}).setdefault(tag, []).append(float(ms))
checks['attention_dump_bytes'] = len((out / 'attention-before-0.bits').read_bytes())
checks['attention_dump_all_bits_equal'] = all(((out / f'attention-before-{rep}.bits').read_bytes() == (out / f'attention-after-{rep}.bits').read_bytes() for rep in range(3)))
assert checks['attention_dump_bytes'] == 458752 and checks['attention_dump_all_bits_equal']
checks['attention_dump_sha256'] = hashlib.sha256((out / 'attention-after-0.bits').read_bytes()).hexdigest()
for shapes, tags in micro.items():
    for tag, values in tags.items():
        assert len(values) == 3
        tags[tag] = {'median_ms': statistics.median(values), 'range_ms': [min(values), max(values)]}
summary['micro'] = micro
assert checks['equal_before_after_expert_slots']
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(out / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
print(json.dumps(summary, indent=2))
print(checks)
