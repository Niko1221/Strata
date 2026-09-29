"""Live cache correctness check. Requires an idle Strata server and a loaded model.

Run: python tools/check_cache_slots.py --url http://127.0.0.1:8080 --lines 1700
The default exercises a ~35k-token prompt with Qwen, beyond a 32k resident KV window.
All requests are synthetic; this replaces the contents of all four cache slots.
"""
import argparse
import json
import time
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--lines', type=int, default=1700)
    args = parser.parse_args()
    if args.lines < 2:
        parser.error('--lines must be at least 2')
    base = args.url.rstrip('/')
    with urllib.request.urlopen(base + '/health', timeout=10) as response:
        model = json.load(response)['model']
    nonce = uuid.uuid4().hex

    def request(body):
        req = urllib.request.Request(
            base + '/v1/chat/completions', data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'})
        return urllib.request.urlopen(req, timeout=600)

    def body(messages, slot):
        return dict(model=model, messages=messages, strata_cache_slot=slot,
                    temperature=0, top_k=1, max_tokens=80, stream=False,
                    chat_template_kwargs={'enable_thinking': False})

    def run(label, messages, slot, max_tokens=80):
        payload = body(messages, slot)
        payload['max_tokens'] = max_tokens
        start = time.monotonic()
        with request(payload) as response:
            result = json.load(response)
        usage = result['usage']
        row = dict(label=label, slot=slot, seconds=round(time.monotonic() - start, 3),
                   prompt_tokens=usage['prompt_tokens'],
                   cached_tokens=usage['prompt_tokens_details']['cached_tokens'],
                   text=result['choices'][0]['message']['content'])
        print(json.dumps(row), flush=True)
        return row

    positions = (args.lines // 3, args.lines - 1)

    def conversation(letter):
        records = '\n'.join(
            f'Record {i:05d}: {letter}-{(i * 7919) % 99991:05d} is an archival marker.'
            for i in range(args.lines))
        question = 'What are the exact markers for records ' + ' and '.join(
            f'{i:05d}' for i in positions) + '?'
        return [dict(role='system', content=nonce + letter + ' Return only the requested markers.'),
                dict(role='user', content=records + '\n' + question)]

    a, b = conversation('A'), conversation('B')
    cold_a = run('A cold', a, 0)
    cold_b = run('B cold', b, 1)
    warm_a = run('A restored', a, 0)
    warm_b = run('B restored', b, 1)
    independent_a = run('A independent cold', a, 2)
    assert cold_a['cached_tokens'] == cold_b['cached_tokens'] == independent_a['cached_tokens'] == 0
    assert warm_a['cached_tokens'] > 0 and warm_b['cached_tokens'] > 0
    assert cold_a['text'] == warm_a['text'] == independent_a['text'], 'A restoration changed the answer'
    assert cold_b['text'] == warm_b['text'], 'B restoration changed the answer'
    for letter, row in [('A', warm_a), ('B', warm_b)]:
        for position in positions:
            assert f'{letter}-{(position * 7919) % 99991:05d}' in row['text'], 'Record retrieval failed'

    counting = [dict(role='system', content=nonce + ' Count as requested.'),
                dict(role='user', content='List numbers 1 through 2000 separated by commas. No introduction.')]
    payload = body(counting, 3)
    payload.update(stream=True, max_tokens=8192)
    with request(payload) as response:
        for line in response:
            if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                continue
            event = json.loads(line[6:])
            if any(choice.get('delta', {}).get('content') for choice in event.get('choices', [])):
                break
        else:
            raise AssertionError('Stream ended without content; cancellation was not tested')
    time.sleep(1)  # allow disconnect/STOP processing before switching
    run('B after cancellation', b, 1)
    resumed = run('cancelled slot restored', counting, 3, max_tokens=32)
    cold = run('cancelled prompt independent cold', counting, 0, max_tokens=32)
    assert resumed['text'] == cold['text'], 'Interrupted cache changed the answer'
    assert resumed['cached_tokens'] > 0 and cold['cached_tokens'] == 0
    print('PASS: independent cache reuse, cold-output parity, record retrieval and interrupted-stream recovery')


if __name__ == '__main__':
    main()
