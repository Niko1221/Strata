"""Check retained runs and print medians; use an output directory as argv[1]."""
import json
from pathlib import Path
import statistics
import sys

out = Path(sys.argv[1])
draft_tag = 'dflash2-il' if (out / 'dflash2-il.json').exists() else 'dflash2'
results = {tag: json.loads((out / f'{tag}.json').read_text()) for tag in [draft_tag, 'mtp']}
summary = {}
reference = {}
for row in results[draft_tag]['rows']:
    if row['control']:
        assert row['offered'] == row['accepted'] == 0
        reference[row['task']] = row['tokens']
assert set(reference) == {'code', 'long'}
for tag, result in results.items():
    assert all(r['tokens'] == reference[r['task']] for r in result['rows']), f'{tag} output differs'
    groups = {}
    for row in result['rows']:
        if row['rep'] < 0:
            continue
        block = next((int(k.split('=')[1]) for k in row['sampling'].split() if k.startswith('dflash_block=')), None)
        key = f"{row['task']}/block={block}/floor={row['floor']}"
        groups.setdefault(key, []).append(row)
    summary[tag] = {'info': result['info'], 'groups': {}}
    for key, rows in groups.items():
        rates = [r['tok_s'] for r in rows]
        summary[tag]['groups'][key] = dict(median=statistics.median(rates), min=min(rates), max=max(rates),
                                          samples=len(rates), rates=rates,
                                          accepted=[r['accepted'] for r in rows], offered=[r['offered'] for r in rows],
                                          prompt_ms=[float(r['done'].split()[3]) for r in rows])
        print(tag, key, f"{statistics.median(rates):.3f} [{min(rates):.3f}, {max(rates):.3f}]", len(rates))
summary['all_fixture_tokens_match_target_only'] = True
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
