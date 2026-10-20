# Local native-protocol probe; run from the repository root. No downloads.
import collections, json, re, statistics, os
from pathlib import Path
out = Path(os.environ.get('STRATA_LONG_RESULTS', Path(__file__).resolve().parent))
summary = {}
for mode in ['dflash2', 'mtp']:
    path = out / f'confirm-{mode}.json'
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    groups = collections.defaultdict(list)
    timings = re.findall('strata decode timing: (\\d+) windows, avg T ([0-9.]+), ([0-9.]+) tokens/window, ([0-9.]+) ms/window = verify ([0-9.]+).*?CPU ([0-9.]+)\\].*?draft ([0-9.]+);', (out / f'confirm-{mode}.log').read_text())
    records = data['warmups'] + data['rows']
    assert len(timings) == len(records), (len(timings), len(records))
    for r, t in zip(records, timings):
        r = dict(r)
        r['profile'] = dict(zip(['windows', 'avg_t', 'tokens_per_window', 'window_ms', 'verify_ms', 'cpu_ms', 'draft_ms'], map(float, t)))
        if r['repeat'] < 0:
            continue
        floor = 'default' if r['floor'] is None else str(r['floor'])
        groups[f"{r['task']}/floor-{floor}/block-{r['block']}"].append(r)
    result = {'info': data['info'], 'groups': {}}
    for k, rs in groups.items():
        result['groups'][k] = {'n': len(rs), 'median_tok_s': statistics.median((r['tok_s'] for r in rs)), 'min_tok_s': min((r['tok_s'] for r in rs)), 'max_tok_s': max((r['tok_s'] for r in rs)), 'accepted': sorted(set((r['accepted'] for r in rs))), 'offered': sorted(set((r['offered'] for r in rs))), 'target_rows': sorted(set((96 - r['accepted'] + r['offered'] for r in rs))), 'median_profile': {key: statistics.median((r['profile'][key] for r in rs)) for key in rs[0]['profile']}}
    result['exploratory'] = [{k: v for k, v in r.items() if k not in ['tokens']} for r in data['rows'] if r['repeat'] == -2]
    result['all_threshold_outputs_identical'] = {task: all((r['tokens'] == [x for x in data['rows'] if x['task'] == task][0]['tokens'] for r in data['rows'] if r['task'] == task)) for task in ['code', 'long']}
    summary[mode] = result
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
# Require actual matching placement before checking target output IDs.
control_path = out / 'controls.json'
if control_path.exists():
    controls = json.loads(control_path.read_text())
    df = json.loads((out / 'confirm-dflash2.json').read_text())
    mtp = json.loads((out / 'confirm-mtp.json').read_text())
    assert len(df['rows']) == 15 and len(mtp['rows']) == 6 and (len(controls['rows']) == 4)
    slots = int(df['info']['expert_slots'])
    assert int(controls['info']['expert_slots']) == slots
    assert controls['info']['spec_min_p'] == '0.00' and df['info']['spec_min_p'] == '0.50'
    assert all((r['matches_confirmed_rows'] for r in controls['rows']))
    assert all((r['offered'] == r['accepted'] == 0 for r in controls['rows'] if r['mode'] == 'target-only'))
    for task in ['code', 'long']:
        target = next((r['tokens'] for r in controls['rows'] if r['task'] == task and r['mode'] == 'target-only'))
        assert len(target) == 96
        assert all((r['tokens'] == target for r in df['rows'] + df['warmups'] + mtp['rows'] + mtp['warmups'] if r['task'] == task))
    checks = dict(default_cli_floor=True, explicit_cli_zero=True, matching_actual_slots=slots, target_only_ids_matched=192, all_repeats_and_explorations_match_target=True, mtp_matches_same_fixture_ids=True)
    (out / 'checks.json').write_text(json.dumps(checks, indent=2))
