"""Promote the validated Gyro-S rc2 algebra (MMLU) result into the report.

The original rc2 full-battery run's MMLU slice failed with an HTTP 429 dataset
download and was recorded as unavailable. A guarded, offline algebra-only rerun
(`algebra-rc2-20261014e`) on the pinned Gyro-S rc2 model/engine produced a valid
scored result (79.0%, first 100 abstract_algebra, 5-shot). This importer validates
that run's provenance and promotes the score, keeping the legacy rc1 76% archived
separately. It does NOT rerun inference.

Usage: python scripts/import_gyro_s_algebra.py --source PATH [--apply]
"""
from pathlib import Path
import argparse, hashlib, json, re

ROOT = Path(__file__).resolve().parent.parent

# Pinned Gyro-S rc2 identity (must match the run manifest exactly).
PINNED = {
    'model_sha256': 'c6ea1b08ca312f99095b27588ba35976b53f948268e7f91e7761926aa01d9c87',
    'engine_sha256': '08d70e30b9176dee5273ba74bd67c4d4c435bef92d449323168d7a8a3497585f',
    'engine_commit': '9ff02c9fa9c86a88088370925b3c9131cc5d6c26',
    'context': 131072,
    'kv': 'int8',
}
# Offline MMLU cache built from the pinned cais/mmlu Parquet (see DATASET-RECEIPT.json).
CACHE_TEST_SHA = '61c1270f0bc5c1d3a1131b414b63bf3bf8ddcd166741972e09c122a63131de3c'
CACHE_DEV_SHA = 'a5a26ee63aa064136be38fc728ad549b7bb52cd3685339dd4bfa635ce3d9e586'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str):
        return re.sub(r'/home/[^/]+/', '<study-home>/', value)
    return value


def collect(source, data):
    results = source / 'results'
    base = source.parents[1]  # algebra-completion-20261014/
    model = data['models']['gyro-s']
    manifest = json.loads((results / 'manifest.json').read_text())
    readback = json.loads((base / 'controller' / 'runs' / manifest['campaign_id'] / 's-done-readback.json').read_text())

    # 1) Identity: run model/engine must equal the pinned Gyro-S rc2.
    for key, want in PINNED.items():
        assert manifest[key] == want, (key, manifest.get(key), want)
    assert manifest['model'] == 'gyro-s-rc2-ranking' and manifest['quant'] == 'TQ1_0'
    assert manifest['scope'] == 'algebra-only'

    # 2) Offline cache provenance: the run loaded the validated pinned cache, no 429.
    assert sha(base / 'data/mmlu/test.jsonl') == CACHE_TEST_SHA
    assert sha(base / 'data/mmlu/dev.jsonl') == CACHE_DEV_SHA
    plugin_log = (source / 'logs/plugin-mmlu.log').read_text()
    assert 'Loading MMLU from cache' in plugin_log and '14042 questions' in plugin_log
    assert '429' not in plugin_log

    # 3) Controller slice audit: the mmlu slice is valid.
    slices = readback['slice_audit']['slices']
    assert slices['plugin-mmlu']['valid'] is True, slices['plugin-mmlu']
    assert readback['slice_audit']['all_required_valid'] is True

    # 4) Scored report: exact score, denominator, protocol, model, completion.
    steps = [json.loads(l) for l in (results / 'steps.jsonl').read_text().splitlines() if l.strip()]
    step = next(x for x in reversed(steps) if x.get('name') == 'plugin-mmlu')
    assert step['exit_code'] == 0 and step['engine_event'] is False and step['server_down'] is False
    assert step['monitor_unknown'] is False and all(v == 0 for v in step['event_delta'].values())
    md = next(p for p in (source.parent / 'runs').rglob('*-mmlu.md') if manifest['campaign_id'] in p.name)
    text = md.read_text()
    acc = re.search(r'\*\*Accuracy:\*\*\s*([0-9.]+)%\s*\((\d+)/(\d+)\)', text)
    assert acc and acc.group(1) == '79.0' and acc.group(2) == '79' and acc.group(3) == '100', acc.groups() if acc else None
    assert '**Completion**: 100.0%' in text and 'gyro-s-rc2-ranking' in text and manifest['campaign_id'] in text
    assert '5-shot' in text
    dur = re.search(r'\*\*Duration:\*\*\s*([0-9.]+)s', text)
    toks = re.search(r'\*\*Tokens:\*\*\s*([\d,]+)', text)
    scored_duration_s = float(dur.group(1)); tokens = int(toks.group(1).replace(',', ''))

    metric = {
        'accuracy': 79.0,
        '_file': md.name,
        'engine_cohort': 'rc2',
        'run_id': manifest['campaign_id'],
        'sample_protocol': 'first 100 abstract_algebra, 5-shot (matches peer MMLU cells)',
        'scored_duration_s': scored_duration_s,
        'wall_s': step['wall_s'],
        'tokens': tokens,
    }
    provenance = {
        'run_id': manifest['campaign_id'],
        'identity': {k: manifest[k] for k in ['model', 'quant', 'model_sha256', 'engine_commit', 'engine_sha256', 'context', 'kv']},
        'offline_cache': {'test_sha256': CACHE_TEST_SHA, 'dev_sha256': CACHE_DEV_SHA, 'loaded_from_cache': True, 'http_429': False},
        'slice_audit': {'plugin-mmlu': slices['plugin-mmlu'], 'all_required_valid': readback['slice_audit']['all_required_valid']},
        'report': {'file': md.name, 'sha256': sha(md), 'accuracy': 79.0, 'correct': 79, 'total': 100, 'completion': '100.0%', 'shots': 5, 'scored_duration_s': scored_duration_s, 'wall_s': step['wall_s'], 'tokens': tokens},
        'guard': {'restored_by_guard': True, 'note': 'guarded algebra-only window; daily driver restored and verified healthy'},
        'legacy_rc1_preserved': 'rc1 76% remains archived under models/gyro-s/legacy_rc1_auxiliary; not substituted',
    }
    files = {'data/' + md.name: md}
    return metric, provenance, files


def apply(source, root=ROOT):
    data = json.loads((root / 'data/ALL-METRICS.json').read_text())
    metric, audit, files = collect(source, data)
    m = data['models']['gyro-s']
    # Preserve the rc1 76% archive (already stored under legacy_rc1_auxiliary).
    m['knowledge']['mmlu'] = metric
    m['auxiliary_source_cohorts']['mmlu'] = 'rc2'
    m['auxiliary_engine'] = 'rc2'
    audit['artifacts'] = {dest: {'original_sha256': sha(path)} for dest, path in files.items()}
    (root / 'data/gyro-s-rc2-algebra-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    for dest, path in files.items():
        content = re.sub(r'/home/[^/]+/', '<study-home>/', path.read_text())
        (root / dest).write_text(content)
        audit['artifacts'][dest]['packaged_sha256'] = sha(root / dest)
    (root / 'data/gyro-s-rc2-algebra-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    data['gyro_s_campaign']['algebra_audit'] = {
        'path': 'data/gyro-s-rc2-algebra-audit.json',
        'sha256': sha(root / 'data/gyro-s-rc2-algebra-audit.json'),
        'run_id': audit['run_id'],
        'accuracy': 79.0,
    }
    (root / 'data/ALL-METRICS.json').write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    return audit


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    if args.apply:
        a = apply(args.source)
    else:
        _, a, _ = collect(args.source, json.loads((ROOT / 'data/ALL-METRICS.json').read_text()))
    print(json.dumps(a, indent=2))
