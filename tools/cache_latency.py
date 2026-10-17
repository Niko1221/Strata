"""Serial, real-engine Responses latency probe; no application-specific wrapper.

Run once per source/build/cache mode. Use cache_latency_campaign.py for a matrix.
Every request and failure is recorded; a requested cache mode does not imply a hit.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
import urllib.error
import urllib.request


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def measure(url, body, service):
    request = urllib.request.Request(url + '/v1/responses',
        data=json.dumps(dict(body, stream=True)).encode(),
        headers={'Content-Type': 'application/json'})
    begin = time.perf_counter()
    first = None
    final = None
    text = ''
    error = None
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw in response:
                if not raw.startswith(b'data: '):
                    continue
                event = json.loads(raw[6:])
                if event.get('type') == 'response.output_text.delta' and event.get('delta'):
                    if first is None:
                        first = time.perf_counter()
                    text += event['delta']
                if event.get('type') in ('response.completed', 'response.incomplete', 'response.failed'):
                    final = event['response']
        if final is None or final.get('status') == 'failed' or first is None:
            error = str(final or 'stream ended without a terminal response')
    except (OSError, ValueError) as exc:
        error = repr(exc)
    return dict(ttft_s=None if first is None else first - begin,
                total_s=time.perf_counter() - begin, error=error,
                usage=(final or {}).get('usage'), status=(final or {}).get('status'),
                text=text, output_sha256=hashlib.sha256(text.encode()).hexdigest(),
                timings=copy.deepcopy(service.last_timings) if not error else None)


def slot_action(url, action, filename):
    request = urllib.request.Request(url + '/slots/0?action=' + action,
        data=json.dumps({'filename': filename}).encode(),
        headers={'Content-Type': 'application/json'})
    begin = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = dict(status=response.status, body=json.load(response))
    except urllib.error.HTTPError as exc:
        result = dict(status=exc.code, body=json.loads(exc.read()))
    except (OSError, ValueError) as exc:
        result = dict(status=0, body={'error': repr(exc)})
    result['elapsed_s'] = time.perf_counter() - begin
    return result


def load_runtime(source, cfg):
    sys.path[:0] = [str(source), str(source / 'tools')]
    from serve.server import Service, StrataEngine, Server, make_handler, engine_args, child_env
    from serve.frontend import ChatTemplate
    import strata_tokenizer as ST
    tpath = Path(cfg['tokenizer'])
    vocab = json.loads((tpath / 'vocab.json').read_text())
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    tok = ST.Tokenizer(tokens, (tpath / 'merges.txt').read_text().split('\n'),
                       json.loads((tpath / 'token_type.json').read_text()))
    engine = StrataEngine(cfg['exe'], engine_args(cfg), cwd=cfg['cwd'],
                          log=cfg['log'], env=child_env(cfg))
    svc = Service(engine, tok, ChatTemplate(tpath / 'chat_template.jinja'), model_name='cache-benchmark')
    server = Server(('127.0.0.1', 0), make_handler(svc))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return engine, svc, server


def corpus(svc, target, identity):
    """Construct >= target shared tokens, without trusting character counts."""
    from serve.responses import template_kwargs
    kw = template_kwargs({'reasoning': {'effort': 'none'}}, {})
    encode = lambda text: svc.encode_prompt([{'role': 'user', 'content': text}], [], kw)
    passage = identity + ' reference notes.\n' + ''.join(
        f'Record {i}: A Python queue validates inputs, commits state atomically, '
        'handles exceptions, and has tests for retries and empty inputs.\n' for i in range(target // 15 + 100))
    # Leave the pinned point within the immutable document, before the question.
    low, high = 0, len(passage)
    while low < high:
        mid = (low + high) // 2
        if len(encode(passage[:mid])) < target + 16:
            low = mid + 1
        else:
            high = mid
    doc = passage[:low] + '\n'
    def request(trial):
        question = (f'Question {trial:06d}: Write a complete runnable Python persistent queue class. '
                    'Include input validation, atomic file replacement, exception handling, '
                    'concurrency considerations, and tests for empty inputs and retries. '
                    'Start with code. Keep writing implementation and tests until the output limit. '
                    'Use the reference notes as background; do not summarize them.')
        text = doc + question
        return dict(model='cache-benchmark', input=text, max_output_tokens=128,
                    temperature=0, seed=42, reasoning={'effort': 'none'}, store=False,
                    strata_prefix={'tokens': target}), encode(text)
    left, right = request(0)[1], request(999999)[1]
    if left[:target] != right[:target]:
        raise ValueError('Corpus does not share the requested prefix')
    return request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--build-label', required=True)
    ap.add_argument('--source-commit', required=True)
    ap.add_argument('--mode', choices=['cold', 'pinned', 'ram_switch', 'disk_switch', 'disk_restore'], required=True)
    ap.add_argument('--targets', default='1024,2048,3072,4096,5120,6144,7168,8192')
    ap.add_argument('--samples', type=int, default=20)
    ap.add_argument('--warmups', type=int, default=3)
    ap.add_argument('--trial-start', type=int, default=0)
    ap.add_argument('--block', type=int, default=0)
    args = ap.parse_args()
    if args.samples < 1 or args.warmups < 1:
        ap.error('--samples and --warmups must be positive')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'requests.jsonl').exists():
        raise RuntimeError('Refusing to append a duplicate run; choose a new output directory')
    cfg = json.loads(args.config.read_text())
    cfg['log'] = str(out / 'engine.log')
    flags = cfg['args'] = list(cfg['args'])
    if any(x.startswith('--conversation-cache') or x == '--prompt-cache' for x in flags):
        raise ValueError('Supply a base config without cache flags; the probe owns them')
    flags += ['--prompt-cache', '0' if args.mode == 'cold' else '6']
    if args.mode == 'ram_switch':
        flags += ['--conversation-cache-mib', '4096', '--conversation-cache-slots', '4']
    elif args.mode == 'disk_switch':
        flags += ['--conversation-cache-disk-only', '--conversation-cache-spill-dir', str(out / 'kv'),
                  '--conversation-cache-disk-mib', '4096']
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    metadata = dict(build=args.build_label, source_commit=args.source_commit,
                    binary_sha256=fingerprint(cfg['exe']), mode=args.mode, block=args.block,
                    targets=args.targets, samples=args.samples, warmups=args.warmups, started=time.time(),
                    sampling={'temperature': 0, 'seed': 42, 'reasoning_effort': 'none'},
                    endpoint='/v1/responses', persistence=False)
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    engine = server = None
    try:
        engine, svc, server = load_runtime(args.source.resolve(), cfg)
        url = 'http://127.0.0.1:' + str(server.server_port)
        if args.mode == 'disk_restore':
            slots = out / 'slots'
            slots.mkdir(exist_ok=True)
            svc.slot_save_path = str(slots)
        targets = [int(n) for n in args.targets.split(',')]
        random.Random(730 + args.block).shuffle(targets)
        with (out / 'requests.jsonl').open('w', buffering=1) as sink:
            def record(req, ids, target, trial, phase):
                result = measure(url, req, svc)
                result.update(build=args.build_label, mode=args.mode, block=args.block,
                              prefix_tokens=target, expected_input_tokens=len(ids), trial=trial,
                              phase=phase, request_sha256=hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest())
                sink.write(json.dumps(result) + '\n')
                return result
            for target in targets:
                primary = corpus(svc, target, 'ALPHA')
                alternate = corpus(svc, target, 'BETA')
                # Warm-ups are recorded but never pooled into percentiles.
                for trial in range(-args.warmups, 0):
                    if args.mode.endswith('switch'):
                        req, ids = alternate(trial + 100000)
                        req['max_output_tokens'] = 8
                        record(req, ids, target, trial, 'alternate_warmup')
                    record(*primary(trial + 100000), target, trial, 'warmup')
                if args.mode == 'disk_restore':
                    filename = f'prefix-{target}.session'
                    saved = slot_action(url, 'save', filename)
                    for trial in range(args.trial_start, args.trial_start + args.samples):
                        req, ids = primary(trial)
                        if saved['status'] == 200:
                            # Replace the live conversation before reading back the saved session.
                            scratch = dict(req, input='A separate conversation. Reply READY.', max_output_tokens=8)
                            scratch.pop('strata_prefix')
                            record(scratch, [], target, trial, 'alternate')
                            restored = slot_action(url, 'restore', filename)
                        else:
                            restored = dict(status=0, elapsed_s=0, body={'error': 'save failed; restore not attempted'})
                        if restored['status'] == 200:
                            row = measure(url, req, svc)
                            row['generation_ttft_s'] = row['ttft_s']
                            row['generation_total_s'] = row['total_s']
                            if row['ttft_s'] is not None:
                                row['ttft_s'] += restored['elapsed_s']
                            row['total_s'] += restored['elapsed_s']
                        else:
                            row = dict(ttft_s=None, total_s=None, usage=None, status='failed',
                                       error='session save/restore failed', text='', output_sha256=None, timings=None)
                        row.update(build=args.build_label, mode=args.mode, block=args.block,
                                   prefix_tokens=target, expected_input_tokens=len(ids), trial=trial,
                                   phase='measured', save=saved, restore=restored,
                                   attempted=restored['status'] == 200,
                                   failure_stage=('save' if saved['status'] != 200 else
                                                  'restore' if restored['status'] != 200 else None),
                                   request_sha256=hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest())
                        sink.write(json.dumps(row)+'\n')
                        print(args.build_label, args.mode, target, trial, row['ttft_s'], row['usage'],
                              saved['status'], restored['status'], row['error'], flush=True)
                    path = slots / filename
                    if path.exists():
                        path.unlink()
                    continue
                for trial in range(args.trial_start, args.trial_start + args.samples):
                    if args.mode.endswith('switch'):
                        req, ids = alternate(trial)
                        req['max_output_tokens'] = 8
                        record(req, ids, target, trial, 'alternate')
                    row = record(*primary(trial), target, trial, 'measured')
                    print(args.build_label, args.mode, target, trial,
                          row['ttft_s'], row['usage'], row['error'], flush=True)
        metadata['finished'] = time.time()
    finally:
        if server:
            server.shutdown()
            server.server_close()
        if engine:
            engine.close()
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
