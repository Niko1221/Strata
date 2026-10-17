#!/usr/bin/env python3
"""Repeat the original public requests against an already-running localhost server."""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

NONCE = 'match-20261014T162442Z'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strata', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.strata.resolve() / 'tools'))
    import strata_bench as bench
    import calibrate
    config = json.loads(args.config.read_text())
    if config.get('host') != '127.0.0.1' or config.get('port') != 8081:
        parser.error('This recorded protocol requires 127.0.0.1:8081')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = 'http://127.0.0.1:8081'
    prompt = 'Reply with the single word READY.'
    streamed = bench.chat(base, '', [{'role': 'user', 'content': prompt}], 16, timeout=120)
    body = json.dumps({'model': 'strata', 'messages': [{'role': 'user', 'content': prompt}],
                       'max_tokens': 16, 'temperature': 0, 'reasoning_effort': 'none',
                       'chat_template_kwargs': {'enable_thinking': False}}).encode()
    request = urllib.request.Request(base + '/v1/chat/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        regular = json.load(response)
    (output / 'smoke.json').write_text(json.dumps({'stream': streamed, 'regular': regular}, indent=2))
    if streamed.get('err') or not streamed.get('text') or not streamed.get('timings'):
        raise RuntimeError('Streaming smoke failed')
    if regular['choices'][0]['finish_reason'] not in ('stop', 'length'):
        raise RuntimeError('Non-streaming smoke failed')
    original = bench.Suite
    class MatchedSuite(original):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.nonce = NONCE
    bench.Suite = MatchedSuite
    code = bench.main(['--url', base, '--config', str(args.config.resolve()), '--runs', '3', '--long',
                       '--out', str(output / 'official')])
    if code:
        raise RuntimeError(f'Official suite failed: {code}')
    official = list((output / 'official').rglob('results.json'))
    if len(official) != 1:
        raise RuntimeError('Expected one complete official suite result')
    result = json.loads(official[0].read_text())['results']
    if result['failures'] or result['skipped']:
        raise RuntimeError('Failed/skipped official requests')
    with (output / 'calibration-probes.jsonl').open('x') as f:
        for repeat in range(3):
            for index, prompt in enumerate(calibrate.PROMPTS):
                text = f'[matched-{NONCE}-{repeat}-{index}] ' + prompt
                response = bench.chat(base, '', [{'role': 'user', 'content': text}], 128, timeout=180)
                row = {'repeat': repeat, 'prompt_index': index, 'prompt': text,
                       'response': response, 'metrics': bench.digest(response),
                       'answer_sha256': hashlib.sha256(response['text'].encode()).hexdigest()}
                f.write(json.dumps(row) + '\n')
                f.flush()
                if response.get('err') or not response.get('timings'):
                    raise RuntimeError('Calibration-prompt probe failed')

if __name__ == '__main__':
    main()
