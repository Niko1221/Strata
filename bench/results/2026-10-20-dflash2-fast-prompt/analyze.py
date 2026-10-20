"""Summarize complete native runs, retaining controls and individual timings."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
o = p.parse_args()
summary = {}
for path in sorted(o.directory.rglob('*.json')):
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or 'rows' not in data:
        continue
    log = path.with_suffix('.log')
    if not log.exists():
        continue
    times = re.findall(r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms', log.read_text())
    rows = data['rows']
    assert len(times) == len(rows), (path, len(times), len(rows))
    measured = []
    for row, timing in zip(rows, times):
        assert len(row['tokens']) == 96
        assert 'error' not in row
        if row['rep'] >= 0 and not row['control']:
            measured.append(dict(task=row['task'], decode_tok_s=row['tok_s'], client_wall_s=row['wall_ms'] / 1000,
                                 prompt_s=int(timing[3]) / 1000, reused_tokens=int(timing[1]),
                                 accepted=row['accepted'], offered=row['offered']))
    controls = [r for r in rows if r['control']]
    entry = dict(executable_sha256=data['sha256'], info=data['info'],
                 total_requests=len(rows), measured=measured,
                 own_control_match=all(r['tokens'] == c['tokens'] for r in rows for c in controls if r['task'] == c['task']) if controls else None,
                 control_token_sha256=[hashlib.sha256(json.dumps(c['tokens']).encode()).hexdigest() for c in controls])
    if measured and len({r['task'] for r in measured}) == 1:
        entry['medians'] = {key: statistics.median(r[key] for r in measured)
                            for key in ['decode_tok_s', 'client_wall_s', 'prompt_s']}
        entry['ranges'] = {key: [min(r[key] for r in measured), max(r[key] for r in measured)]
                           for key in ['decode_tok_s', 'client_wall_s', 'prompt_s']}
    summary[str(path.relative_to(o.directory))] = entry
# MTP shares the target-only reference recorded in the corresponding DFlash process.
for path in sorted(o.directory.rglob('mtp.json')):
    reference = path.with_name('dflash2-il.json')
    if not reference.exists():
        continue
    mtp = json.loads(path.read_text())
    df = json.loads(reference.read_text())
    controls = [r['tokens'] for r in df['rows'] if r['control']]
    if controls:
        entry = summary[str(path.relative_to(o.directory))]
        entry['dflash_target_control_match'] = all(r['tokens'] == controls[0] for r in mtp['rows'])
        assert mtp['sha256'] == df['sha256']
        assert mtp['info']['expert_slots'] == df['info']['expert_slots']
(o.directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
for name, result in summary.items():
    if 'medians' in result:
        print(name, result['medians'], result['own_control_match'])
