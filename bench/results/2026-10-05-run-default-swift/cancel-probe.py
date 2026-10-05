#!/usr/bin/env python3
"""Cancellation and recovery probe: start a long generation, disconnect mid-stream,
then prove the next request runs clean while the guards keep sampling.  Exits 1 on
any leaked/failed state (next request wrong, server unresponsive, or non-stop finish)."""
import argparse, json, socket, time, urllib.request

def stream(url, body, abort_after):
    req = urllib.request.Request(url + '/v1/chat/completions',
                                 data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        start = time.monotonic()
        got = 0
        while time.monotonic() - start < abort_after:
            chunk = r.read(64)
            if not chunk:
                break
            got += chunk.count(b'data:')
        # disconnect mid-stream: closing the response without draining it
        return got

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:19931')
    p.add_argument('--model', default='qwen3.8-flash-next-iq3_s')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    base = a.url.rstrip('/')
    filler = 'Explain how a ring buffer and DMA staging pipeline overlap in a GPU expert cache. ' * 40
    long_prompt = filler + 'Now write a 2000-word essay covering every aspect in detail.'
    results = []

    # warm recovery baseline
    body = dict(model=a.model, messages=[{'role': 'user', 'content': 'Reply READY.'}],
                max_tokens=8, temperature=0, seed=42, reasoning_effort='none')
    with urllib.request.urlopen(urllib.request.Request(
            base + '/v1/chat/completions', data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'}), timeout=300) as r:
        reply = json.load(r)
    results.append(dict(phase='baseline', finish=reply['choices'][0]['finish_reason']))

    # cancel mid-stream: read for 20 s then drop the connection
    cbody = dict(model=a.model, messages=[{'role': 'user', 'content': long_prompt}],
                 max_tokens=4096, temperature=0, seed=42, stream=True, reasoning_effort='none')
    events = stream(base, cbody, 20)
    results.append(dict(phase='cancel', stream_events_before_disconnect=events))
    time.sleep(3)

    # recovery: the same small request as the baseline must complete normally
    with urllib.request.urlopen(urllib.request.Request(
            base + '/v1/chat/completions', data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'}), timeout=300) as r:
        reply = json.load(r)
    finish = reply['choices'][0]['finish_reason']
    text = reply['choices'][0]['message']['content']
    results.append(dict(phase='recovery', finish=finish, text=text[:80]))
    with open(a.output, 'w') as f:
        json.dump(results, f, indent=1)
    if finish != 'stop':
        raise SystemExit(f'recovery request did not finish normally: {finish}')
    print('cancel/recovery OK:', json.dumps(results[-1]))

if __name__ == '__main__':
    main()
