#!/usr/bin/env python3
"""Replay the October 30 custom checks against an already running server. Paths/CLI adapted for publication; request bodies and evaluation are unchanged."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--source', type=Path, default=HERE.parents[2], help='0.2.8 checkout providing benchmark helpers')
ap.add_argument('--url', default='http://127.0.0.1:8080')
ap.add_argument('--config', type=Path, required=True, help='Private server config with your API key')
ap.add_argument('--out', type=Path, required=True, help='New output directory')
a = ap.parse_args()
sys.path.insert(0, str(a.source.resolve()))
from stream import chat
from strata_cli.bench_standard import Target, evaluate, load_corpus, stream_chat
from tools.strata_bench import digest, http_json, wait_ready


def main():
    cfg = json.loads(a.config.read_text())
    key = cfg['api_key']
    out = a.out
    out.mkdir(exist_ok=False)
    assert wait_ready(a.url, key, None, 900), 'Engine readiness timeout'
    health = http_json(a.url, '/health', key)
    assert health['max_context'] == 131072
    (out / 'health.json').write_text(json.dumps(health, indent=2))
    target = Target(a.url, key)
    corpus = load_corpus()
    quality = []
    for spec in corpus.specs:
        if spec['phase'] != 'sanity':
            continue
        raw = stream_chat(target, spec, 'coder390', timeout_s=600)
        status = ('failed' if raw['error'] else 'truncated' if raw['finish_reason'] == 'length'
                  else 'completed' if raw['finish_reason'] in ('stop', 'tool_calls') else 'failed')
        verdict = evaluate(spec, status, ''.join(raw['content']), list(raw['tool_calls'].values()))
        row = {'case': spec['id'], **verdict}
        quality.append(row)
        (out / (spec['id'] + '.json')).write_text(json.dumps({'raw': raw, 'verdict': row}, ensure_ascii=False, indent=2))
        print(json.dumps(row), flush=True)
    (out / 'quality.json').write_text(json.dumps({'passed': sum(r['passed'] for r in quality),
        'total': len(quality), 'corpus_sha256': corpus.sha256, 'rows': quality}, indent=2))
    assert all(r['passed'] for r in quality), 'Sanity case failed; inspect saved responses'
    question = ('Put the secret archive code from this document in the first Python comment. Then implement '
                'merge_intervals(intervals), preserving the input and merging overlapping or touching endpoints. '
                'Include assertions for edge cases. Provide Python code first, then explain the algorithm and its time complexity.')
    document = (HERE / 'code-context.txt').read_text()
    cases = [('short-code', 'Write a Python LRU cache using OrderedDict, then explain eviction and include example assertions.')]
    cases += [('long-code-' + str(i), document + '\n' + question) for i in range(2)]
    rows = []
    for name, prompt in cases:
        raw = chat(a.url, key, [{'role': 'user', 'content': prompt}], 256, timeout=1800)
        row = {'case': name, **digest(raw)}
        if name.startswith('long'):
            row['needle_ok'] = 'ZEBRA-4729' in raw['text']
        rows.append(row)
        (out / (name + '.json')).write_text(json.dumps({'summary': row, 'result': raw}, ensure_ascii=False, indent=2))
        print(json.dumps(row), flush=True)
        assert row['ok'] and row['generated_tokens'] >= 192, row
        if name.startswith('long'):
            assert row['needle_ok'] and row['prompt_tokens'] >= 130000, row
            assert row['prompt_tokens'] + row['generated_tokens'] <= 131072, row
    (out / 'summary.json').write_text(json.dumps({'quality_passed': len(quality), 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
