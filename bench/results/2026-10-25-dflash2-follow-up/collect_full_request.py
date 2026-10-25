#!/usr/bin/env python3
"""Archive completed local studies and compute descriptive (not inferential) stats.

Usage: python3 collect_full_request.py /path/to/dflash2-opt-20261025
Raw token arrays and all samples are retained. Local absolute data paths become
${STRATA_DATA}; source hashes identify the original files before normalization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def encode(value):
    text = json.dumps(value, indent=2, ensure_ascii=False)
    return re.sub(r'\[\s*((?:-?\d+,\s*)+-?\d+)\s*\]',
                  lambda m: '[' + ' '.join(m[1].split()) + ']', text) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    destination = Path(__file__).resolve().parent / 'full-request'
    destination.mkdir(exist_ok=True)
    names = [
        ('control128', 'prompt-control128', 'mtp'),
        ('prompt-baseline128-a', 'prompt-baseline128-a', 'dflash2-il'),
        ('prompt-batch128-a', 'prompt-batch128-a', 'dflash2-il'),
        ('mtp128', 'prompt-mtp128', 'mtp'),
        ('control384', 'control384', 'mtp'),
    ]
    short = ['prompt-batch128-b-retry', 'prompt-baseline128-b',
             'grid-baseline128-a', 'grid128-a', 'grid-baseline128-b',
             'selector-baseline128-a', 'selector128-a', 'selector128-b',
             'selector-baseline128-b', 'pipeline128']
    topk = ['default384-a', 'topk384-a', 'register384-a', 'register-merge384-a',
            'register-merge384-b', 'register384-b', 'topk384-b', 'default384-b',
            'pipeline384-a', 'pipeline384-b', 'default384-c', 'vector384-smoke']
    names += [(name, 'session2/' + name, 'dflash2-il') for name in short + topk]
    controls = {}
    for count, directory in [(128, 'prompt-control128'), (384, 'control384')]:
        reference = json.loads((source / directory / 'mtp.json').read_text())
        expected = {}
        for row in reference['rows']:
            assert row['offered'] == 0 and row['accepted'] == 0
            assert row['tokens'] == expected.setdefault(row['task'], row['tokens'])
        controls[count] = expected
    manifest, summary, missing, timing = [], {}, [], {}
    for label, directory, mode in names:
        path = source / directory / (mode + '.json')
        if not path.exists():
            missing.append(label)
            continue
        raw = path.read_bytes()
        data = json.loads(raw)
        if 'validation' not in data:
            missing.append(label + ' (incomplete)')
            continue
        assert data['validation'].startswith(('common target-only controls passed', 'target-only run')), path
        for row in data['rows']:
            assert row['tokens'] == controls[data['count']][row['task']], (path, row['task'])
        measured = [row for row in data['rows'] if row['rep'] >= 0 and not row['control']]
        assert measured or data['validation'] == 'target-only run', path
        groups = {}
        for row in measured:
            task = row['task']
            group = groups.setdefault(task, {'prompt_ms': [], 'decode_ms': [],
                'tok_s': [], 'accepted_offered': [], 'output_tokens': []})
            group['prompt_ms'].append(float(row['done'].split()[3]) if 'done' in row else row.get('prompt_ms'))
            group['decode_ms'].append(row['decode_ms'])
            group['tok_s'].append(row['tok_s'])
            group['accepted_offered'].append([row['accepted'], row['offered']])
            group['output_tokens'].append(len(row['tokens']))
        for group in groups.values():
            group['median_tok_s'] = statistics.median(group['tok_s'])
            group['median_prompt_ms'] = statistics.median([v for v in group['prompt_ms'] if v is not None]) if any(v is not None for v in group['prompt_ms']) else None
            group['range_tok_s'] = [min(group['tok_s']), max(group['tok_s'])]
        summary[label] = groups
        normalized = json.loads(raw.decode().replace(str(source.parent), '${STRATA_DATA}'))
        (destination / (label + '.json')).write_text(encode(normalized))
        manifest.append({'label': label, 'source': str(path.relative_to(source)),
                         'source_sha256': hashlib.sha256(raw).hexdigest()})
        log_path = path.with_suffix('.log')
        if log_path.exists():
            log_bytes = log_path.read_bytes()
            lines = [line for line in log_bytes.decode(errors='replace').splitlines()
                     if line.startswith('strata decode timing:')]
            assert len(lines) == len(data['rows']), (log_path, len(lines), len(data['rows']))
            timing[label] = {'requests': [{key: row[key] for key in ('task', 'rep', 'control')} for row in data['rows']], 'log_sha256': hashlib.sha256(log_bytes).hexdigest(),
                             'lines': lines, 'request_rows': len(data['rows'])}

    (destination / 'decode-timing.json').write_text(encode(timing))
    (destination / 'summary.json').write_text(encode(summary))
    (destination / 'manifest.json').write_text(encode({'files': manifest, 'missing': missing}))
    print(f'Archived {len(manifest)} completed profiles; missing: {missing}')


if __name__ == '__main__':
    main()
