"""Archive completed host-codebook or capacity profiles with independent checks.

Usage: python3 collect_host_screen.py SOURCE [--capacity]
Local paths are normalized after hashing the original files. Every token sequence,
including warmups and same-process controls, must match its genuine target control.
"""
from pathlib import Path
import argparse
import hashlib
import json
import statistics
from collect_full_screen import encode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--capacity', action='store_true')
    args = parser.parse_args()
    source = args.source.resolve()
    base = source / 'capacity' if args.capacity else source
    manifest = base / 'manifest.json' if args.capacity else source / 'host-screen-manifest.json'
    records = json.loads(manifest.read_text())
    dst = Path(__file__).resolve().parent / ('capacity' if args.capacity else 'host-fixed')
    dst.mkdir(exist_ok=True)

    def normalize(text):
        for path, replacement in [
            (str(source.parent), '${STRATA_DATA}'),
            ('/home/jakub/.codex/worktrees/dflash2-opt-20261026/Strata', '${REPO}'),
            ('/home/jakub/workspace/deepseek-harness/harnes_workspace/Strata', '${REPO}'),
        ]:
            text = text.replace(path, replacement)
        return text

    control_paths = ({k: base / (k + '-control') / 'mtp.json' for k in ('mtp', 'gpu', 'host')}
                     if args.capacity else {'fixed': source / 'control/mtp.json'})
    references = {key: json.loads(path.read_text()) for key, path in control_paths.items()}
    tokens = {}
    for key, data in references.items():
        tokens[key] = {}
        for row in data['rows']:
            assert row['control'] and row['accepted'] == row['offered'] == 0
            assert row['tokens'] == tokens[key].setdefault(row['task'], row['tokens'])
    cross_placement_equal = all(value == next(iter(tokens.values())) for value in tokens.values())
    summary = {}; archived = []
    for record in records:
        if record['returncode'] != 0:
            continue
        label = record['label']
        kind = record['kind'] if args.capacity else 'fixed'
        mode = 'mtp' if args.capacity and (record['control'] or kind == 'mtp') else 'dflash2-il'
        path = base / label / (mode + '.json')
        raw = path.read_bytes(); data = json.loads(raw); reference = references[kind]
        for key in ('prompt_sha256', 'expert_profile_sha256'):
            assert data[key] == reference[key], (label, key)
        for key in ('context', 'kv', 'kv_resident', 'expert_slots', 'expert_slots_primary',
                    'pool_workers', 'spec', 'pcie_frac'):
            assert data['info'][key] == reference['info'][key], (label, key)
        for row in data['rows']:
            assert row['tokens'] == tokens[kind][row['task']], (label, row['task'])
        if args.capacity:
            assert min(data['post_warmup_free_mib'], data['final_free_mib']) >= 512
        (dst / (label + '.json')).write_text(encode(json.loads(normalize(raw.decode()))))
        log = path.with_suffix('.log')
        (dst / (label + '.log')).write_text(normalize(log.read_text()))
        groups = {}
        for row in data['rows']:
            if row['rep'] < 0 or row['control']:
                continue
            group = groups.setdefault(row['task'], {'tok_s': [], 'prompt_ms': [], 'wall_ms': [],
                                                    'accepted_offered': []})
            group['tok_s'].append(row['tok_s'])
            group['prompt_ms'].append(float(row['done'].split()[3]))
            group['wall_ms'].append(row['wall_ms'])
            group['accepted_offered'].append([row['accepted'], row['offered']])
        for group in groups.values():
            group['median_tok_s'] = statistics.median(group['tok_s'])
            group['range_tok_s'] = [min(group['tok_s']), max(group['tok_s'])]
        summary[label] = {'tasks': groups, 'startup_free_mib': data['startup_free_mib']}
        for key in ('post_warmup_free_mib', 'final_free_mib'):
            if key in data:
                summary[label][key] = data[key]
        item = dict(record)
        item['original_json_sha256'] = hashlib.sha256(raw).hexdigest()
        item['original_log_sha256'] = hashlib.sha256(log.read_bytes()).hexdigest()
        archived.append(json.loads(normalize(json.dumps(item))))
    (dst / 'summary.json').write_text(encode(summary))
    (dst / 'manifest.json').write_text(encode(archived))
    (dst / 'validation.json').write_text(encode({'profiles': len(archived),
        'all_sequences_match_own_placement_control': True,
        'tokens_equal_across_placement_controls': cross_placement_equal}))
    print('PASS', len(archived), 'profiles; cross-placement tokens equal:', cross_placement_equal)


if __name__ == '__main__':
    main()
