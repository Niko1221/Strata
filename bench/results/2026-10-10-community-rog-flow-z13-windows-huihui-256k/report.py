"""Offline verification/summary, config rendering, and explicit recorded-arm replay."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.request

HERE = Path(__file__).resolve().parent
WEIGHTS = {'code_greedy': .35, 'code_sampled': .35, 'xhigh_throughput': .25,
           'longcode8k': .25, 'longcode32k': .25, 'longcode128k': .25}


def read(name):
    return json.loads((HERE / name).read_text(encoding='utf-8'))


def runs():
    return [json.loads(line) for line in (HERE / 'runs.jsonl').read_text(encoding='utf-8').splitlines()]


def summary(records):
    rows, scores = [], {}
    for source in ['generalization-general-original.json', 'generalization-general-current.json']:
        medians = {}
        for case in WEIGHTS:
            group = [r for r in records if r['Source'] == source and r['Case'] == case]
            assert len(group) == 2, (source, case, len(group))
            ts = [r['Response']['timings'] for r in group]
            assert all(t['cache_n'] == 0 and t['predicted_n'] == 512 for t in ts)
            values = [t['predicted_per_second'] for t in ts]
            medians[case] = statistics.median(values)
            rows.append({'source': source, 'case': case, 'runs': len(group),
                'decode_median': medians[case], 'decode_range': [min(values), max(values)],
                'prompt_median': statistics.median(t['prompt_per_second'] for t in ts),
                'input_tokens': sorted({t['prompt_n'] + t['cache_n'] for t in ts})})
        scores[source] = sum(WEIGHTS.values()) / sum(WEIGHTS[c] / medians[c] for c in WEIGHTS)
    gain = scores['generalization-general-current.json'] / scores['generalization-general-original.json'] - 1
    return {'rows': rows, 'weighted_harmonic_scores': scores, 'gain_fraction': gain}


def verify():
    for line in (HERE / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        assert Path(name).name == name, name
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected, name
    records, fixtures, profiles = runs(), read('messages.json'), read('profiles.json')
    assert len(records) == 809 and sum(r['Formal'] for r in records) == 780
    assert len({r['Id'] for r in records}) == 809
    for stage, count in profiles['FormalStageCounts'].items():
        assert sum(r['Formal'] and r['Stage'] == stage for r in records) == count
    for key, messages in fixtures.items():
        encoded = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
        assert hashlib.sha256(encoded).hexdigest() == key, key
    for r in records:
        assert r['ConfigurationId'] in profiles['Configurations']
        assert r['Request']['MessagesRef'] in fixtures
    result = summary(records)
    assert abs(result['gain_fraction'] - .053059476782382875) < 1e-10
    print(json.dumps({'verified_files': len((HERE / 'SHA256SUMS').read_text().splitlines()),
        'records': len(records), 'formal': 780, 'gain_fraction': result['gain_fraction']}, indent=2))


def source_config(source):
    p = read('profiles.json')
    matches = [r for r in p['Runs'] if r['Source'] == source]
    if len(matches) != 1:
        raise ValueError('Source must identify exactly one recorded profile')
    return p['Configurations'][matches[0]['ConfigurationId']]


def replay(args):
    records = [r for r in runs() if r['Source'] == args.source]
    if not records or any(r['Stage'] == 'stability' for r in records):
        raise ValueError('Select one valid non-conversation source')
    if len({r['ConfigurationId'] for r in records}) != 1:
        raise ValueError('Replay requires one engine configuration')
    fixtures = read('messages.json')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def payload(r):
        request = copy.deepcopy(r['Request'])
        request['messages'] = fixtures[request.pop('MessagesRef')]
        return request

    def send(request):
        query = urllib.request.Request(args.url.rstrip('/') + '/v1/chat/completions',
            data=json.dumps(request, ensure_ascii=False).encode('utf-8'),
            headers={'Content-Type': 'application/json'})
        start = time.monotonic()
        with opener.open(query, timeout=1200) as response:
            result = json.load(response)
        return result, time.monotonic() - start

    if all(r['Formal'] for r in records):
        for case in ['code_greedy', 'xhigh_throughput']:
            candidate = next((r for r in records if r.get('Case') == case), None)
            if candidate:
                request = payload(candidate)
                request['max_tokens'] = 64
                request.setdefault('strata_tune', {})['spec_min_p'] = (0.0 if candidate['Stage'] == 'screen'
                                                                     else records[0]['Threshold'])
                send(request)
    # Exclusive creation prevents accidentally replacing a prior measurement.
    with Path(args.output).open('x', encoding='utf-8') as handle:
        for r in records:
            response, elapsed = send(payload(r))
            item = {'RecordedId': r['Id'], 'Source': r['Source'], 'Case': r.get('Case'),
                    'ElapsedSeconds': elapsed, 'Response': response}
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
            handle.flush()
            print(json.dumps({'case': r.get('Case'), 'timings': response.get('timings'),
                              'elapsed_seconds': elapsed}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    subs.add_parser('verify')
    subs.add_parser('summarize')
    render = subs.add_parser('render-config')
    render.add_argument('--workspace', required=True)
    render.add_argument('--source')
    render.add_argument('--output', required=True)
    replayer = subs.add_parser('replay')
    replayer.add_argument('--source', required=True)
    replayer.add_argument('--url', required=True)
    replayer.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.action == 'verify':
        verify()
    elif args.action == 'summarize':
        print(json.dumps(summary(runs()), indent=2))
    elif args.action == 'render-config':
        config = source_config(args.source) if args.source else read('config-selected.json')
        # Replace inside parsed string values, so workspace backslashes stay valid JSON.
        def replace(v):
            if isinstance(v, str):
                return v.replace('${WORKSPACE}', args.workspace)
            if isinstance(v, list):
                return [replace(x) for x in v]
            if isinstance(v, dict):
                return {k: replace(x) for k, x in v.items()}
            return v
        Path(args.output).write_text(json.dumps(replace(config), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    else:
        replay(args)


if __name__ == '__main__':
    main()
