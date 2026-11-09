"""Send exactly two retained synthetic requests; never execute model output."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_request(info, model=None):
    raw = (ROOT / info['file']).read_bytes()
    if sha(raw) != info['sha256']:
        raise ValueError('fixture hash mismatch: ' + info['file'])
    body = json.loads(raw)
    if model is not None:
        body['model'] = model
    wire = json.dumps(body).encode('utf-8')
    if model is None and sha(wire) != info['wire_sha256']:
        raise ValueError('original wire serialization differs')
    return wire


def endpoint(url):
    u = urllib.parse.urlsplit(url)
    if u.scheme not in ('http', 'https') or not u.hostname or u.username or u.password or u.query or u.fragment:
        raise ValueError('use an HTTP(S) base URL without credentials, query or fragment')
    base = url.rstrip('/')
    return base + ('/chat/completions' if u.path.rstrip('/').endswith('/v1') else '/v1/chat/completions')


def summarize(events, done, wire, elapsed):
    content, reasoning, choice_trace = [], [], []
    usage, timings, finish = {}, {}, None
    for event in events:
        if event.get('error'):
            raise RuntimeError('server returned an API error; inspect its local log')
        usage = event.get('usage') or usage
        timings = event.get('timings') or timings
        choice_trace.append(event.get('choices'))
        for choice in event.get('choices', []):
            d = choice.get('delta', {})
            content.append(d.get('content') or '')
            reasoning.append(d.get('reasoning_content') or '')
            finish = choice.get('finish_reason') or finish
    content, reasoning = ''.join(content), ''.join(reasoning)
    zero_run = max((len(m[0]) for m in re.finditer('0+', reasoning)), default=0)
    return {
        'wire_sha256': sha(wire), 'stream_done': done, 'finish_reason': finish,
        'usage': usage, 'timings': timings, 'wall_seconds': elapsed,
        'content_utf8_bytes': len(content.encode()), 'reasoning_utf8_bytes': len(reasoning.encode()),
        'content_sha256': sha(content.encode()), 'reasoning_sha256': sha(reasoning.encode()),
        'choice_delta_sequence_sha256': sha(json.dumps(choice_trace, sort_keys=True).encode()),
        'empty_visible_answer': not content.strip(), 'longest_zero_character_run_in_reasoning': zero_run,
        'contains_parse_ledger_definition': bool(re.search(r'\bdef\s+parse_ledger\s*\(', content)),
        'code_executed': False,
        'scope': 'decoded text and API deltas; zero characters are not a native token-ID trace',
    }


def post(url, key, wire, opener=urllib.request.urlopen):
    request = urllib.request.Request(endpoint(url), wire, headers={
        'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    started = time.monotonic()
    events, done, received = [], False, 0
    # Each socket read is bounded; the overall loop is also bounded. No retries.
    with opener(request, timeout=120) as response:
        for line in response:
            received += len(line)
            if time.monotonic() - started > 1200 or received > 16 * 1024 * 1024:
                raise RuntimeError('bounded response limit exceeded')
            if not line.startswith(b'data:'):
                continue
            raw = line[5:].strip()
            if raw == b'[DONE]':
                done = True
                break
            events.append(json.loads(raw))
    return summarize(events, done, wire, time.monotonic() - started)


def check_protocol(result, info):
    u = result['usage']
    count = u.get('completion_tokens')
    if not result['stream_done'] or result['finish_reason'] not in ('stop', 'length'):
        raise RuntimeError('response did not complete normally; do not continue the history')
    if u.get('prompt_tokens') != info['prompt_tokens'] or u.get('prompt_tokens_details', {}).get('cached_tokens') != 0:
        raise RuntimeError('prompt count/cache differs from the retained reproduction')
    if type(count) is not int or not 0 < count <= info['max_tokens'] or u.get('total_tokens') != info['prompt_tokens'] + count or u['total_tokens'] > 65536:
        raise RuntimeError('output/context accounting differs')
    if info['prompt_tokens'] == 64000 and count != 512:
        raise RuntimeError('first request did not consume the exact 512-output history')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', required=True, help='server base URL, with or without /v1')
    p.add_argument('--key-env', default='STRATA_API_KEY', help='environment variable containing the API key')
    p.add_argument('--model', help='optional serving alias override; changes wire hash, not message content')
    p.add_argument('--confirm-fresh-server', action='store_true', required=True)
    args = p.parse_args()
    endpoint(args.url)
    key = os.environ.get(args.key_env)
    if not key:
        p.error('the selected API-key environment variable is empty')
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for info in manifest['requests']:
        wire = load_request(info, args.model)
        result = post(args.url, key, wire)
        result['request'] = info['file']
        print(json.dumps(result, sort_keys=True), flush=True)
        check_protocol(result, info)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        # Do not print exception text that may contain a URL or authentication data.
        print(json.dumps({'error_type': type(error).__name__, 'result': 'incomplete; inspect server log locally'}), flush=True)
        raise SystemExit(2)
