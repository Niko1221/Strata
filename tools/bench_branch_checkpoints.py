"""Reproduce plain-server prefix reuse, branching, bookmarks and deleted-cache replay.

Runs isolated loopback processes; deletes only checkpoint blocks in its scratch output.
Requires Linux, the model tokenizer and an existing native engine configuration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.request
AP = argparse.ArgumentParser(description='Isolated native checkpoint benchmark; no production services changed.')
AP.add_argument('--config', required=True, help='Existing native engine/tokenizer configuration')
AP.add_argument('--mtp', required=True, help='MTP runtime directory for speculation 4')
AP.add_argument('--output', required=True, help='New scratch output directory; its state-plain must not exist')
AP.add_argument('--tokens', type=int, default=32768)
AP.add_argument('--repeats', type=int, default=3)
AP.add_argument('--port', type=int, default=18219)
ARGS = AP.parse_args()
CORE = Path(__file__).resolve().parents[1]
ROOT = Path(ARGS.output).resolve()
PORT = ARGS.port
TARGET = ARGS.tokens
ROWS = []
AUDITS = []

def call(path, body=None):
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}' + path, data=None if body is None else json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)

def save():
    (ROOT / 'results.json').write_text(json.dumps({'rows': ROWS, 'audits': AUDITS}, indent=2))

def query(db, sql, values=()):
    with sqlite3.connect(db) as connection:
        return connection.execute(sql, values).fetchall()

def generate(mode, state, label, text, expected=(), **extra):
    body = dict(input=text, max_output_tokens=128, reasoning={'effort': 'none'}, temperature=0, stream=True)
    body.update(extra)
    catalog = state / 'responses/execution-cache/checkpoints.sqlite3'
    before = dict(query(catalog, 'SELECT id,last_used FROM successful_restores'))
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/responses', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    first = result = None
    output = ''
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            event = json.loads(line[6:])
            if event['type'] == 'response.output_text.delta':
                first = first or time.perf_counter()
                output += event['delta']
            if event['type'] in ('response.completed', 'response.incomplete', 'response.failed'):
                result = event['response']
    elapsed = time.perf_counter() - started
    after = dict(query(catalog, 'SELECT id,last_used FROM successful_restores'))
    row = dict(mode=mode, label=label, ttft_s=None if first is None else first - started, total_s=elapsed, text=output, expected=list(expected), correct=all((code in output for code in expected)), disk_restores=sum((before.get(sid) != used for sid, used in after.items())), response=result)
    row['cached_tokens'] = (result or {}).get('usage', {}).get('input_tokens_details', {}).get('cached_tokens', 0)
    row['owned_snapshots'] = query(catalog, 'SELECT count(*) FROM owners JOIN snapshots ON snapshots.id=owners.snapshot WHERE response=?', ((result or {}).get('id', ''),))[0][0]
    ROWS.append(row)
    save()
    print(mode, label, round(row['ttft_s'] or 0, 3), round(elapsed, 3), 'restores=' + str(row['disk_restores']), repr(output), flush=True)
    assert result and result['status'] == 'completed', row
    assert row['correct'], row
    return (result, row)

def stop(process):
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise RuntimeError('graceful shutdown timed out')
    assert process.returncode == 0, process.returncode

def run(mode):
    state = ROOT / ('state-' + mode)
    state.mkdir(exist_ok=False)
    cfg = json.loads(Path(ARGS.config).read_text())
    args = cfg['args']
    stripped = []
    i = 0
    while i < len(args):
        if args[i] in ('--conversation-cache-spill-dir', '--conversation-cache-disk-mib', '--conversation-cache-mib', '--mtp'):
            i += 2
        elif args[i] == '--conversation-cache-disk-only':
            i += 1
        else:
            stripped.append(args[i])
            i += 1
    stripped[stripped.index('--spec') + 1] = '4'
    stripped[stripped.index('--max-context') + 1] = str(TARGET + 4096)
    cfg['args'] = stripped + ['--conversation-cache-mib', '0', '--mtp', ARGS.mtp]
    cfg.update(host='127.0.0.1', port=PORT, api_key='', state=str(state), experimental_branch_checkpoints=True, responses_store_path=str(state / 'responses'), checkpoint_budget_mib=16384, checkpoint_max_snapshot_mib=4096, history_reserve_mib=4096, log=str(ROOT / ('engine-' + mode + '.log')))
    config = ROOT / ('config-' + mode + '.json')
    config.write_text(json.dumps(cfg, indent=2))
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    env.pop('STRATA_API_KEY', None)
    command = [sys.executable, '-m', 'serve.server', '--engine', 'strata', '--config', str(config), '--port', str(PORT)]
    with (ROOT / ('server-' + mode + '.log')).open('a') as log:

        def start():
            process = subprocess.Popen(command, cwd=CORE, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(900):
                if process.poll() is not None:
                    raise RuntimeError('server failed: ' + mode)
                try:
                    call('/v1/models')
                    return process
                except OSError:
                    time.sleep(1)
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise TimeoutError('startup')
        process = start()
        try:
            sys.path[:0] = [str(CORE), str(CORE / 'tools')]
            import strata_tokenizer as ST
            from serve.frontend import ChatTemplate
            from serve.responses import template_kwargs
            tp = Path(cfg['tokenizer'])
            vocab = json.loads((tp / 'vocab.json').read_text())
            tokens = [None] * len(vocab)
            for token, index in vocab.items():
                tokens[index] = token
            tok = ST.Tokenizer(tokens, (tp / 'merges.txt').read_text().split('\n'), json.loads((tp / 'token_type.json').read_text()))
            template = ChatTemplate(tp / 'chat_template.jinja')
            prefix = []
            kw = template_kwargs({'reasoning': {'effort': 'none'}}, {})
            kw['preserve_empty_reasoning'] = True

            def sized_prompt(rep):
                codes = [f'CEDAR-{100 + rep}', f'MARBLE-{200 + rep}', f'QUARTZ-{300 + rep}']

                def build(n):
                    text = f'Start verification code: {codes[0]}.\n' + ' apple' * (n // 2) + f'\nMiddle verification code: {codes[1]}.\n' + ' orange' * (n - n // 2) + f'\nEnd verification code: {codes[2]}.\nReply with all three verification codes, in order, separated by |. Nothing else.'
                    ids = prefix + tok.encode(template.render([{'role': 'user', 'content': text}], tools=None, **kw), parse_special=True)
                    return (text, ids)
                lo, hi = (0, TARGET)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if len(build(mid)[1]) <= TARGET:
                        lo = mid
                    else:
                        hi = mid - 1
                text, ids = build(lo)
                assert len(ids) == TARGET, len(ids)
                return (text, ids, codes)

            def distraction(label):
                return generate(mode, state, label, 'Reply with ready only.', **{})
            follow = 'Repeat all three verification codes from the document, in order, separated by |. Nothing else.'
            for rep in range(ARGS.repeats):
                prompt, ids, codes = sized_prompt(rep)
                fresh, row = generate(mode, state, f'fresh-{rep}', prompt, codes)
                saved = json.loads(query(state / 'responses/responses.sqlite3', 'SELECT prompt FROM execution_records WHERE response_id=?', (fresh['id'],))[0][0])
                assert saved == ids, ('actual execution differs from sized input', len(saved), len(ids))
                assert row['response']['usage']['input_tokens'] == TARGET, row
                AUDITS.append(dict(mode=mode, kind='exact-input', rep=rep, tokens=len(saved), profile_tokens=len(prefix), sha256=hashlib.sha256(prompt.encode()).hexdigest()))
                distraction(f'switch-{rep}')
                _, reused = generate(mode, state, f'reuse-{rep}', follow, codes, previous_response_id=fresh['id'])
                assert reused['disk_restores'] > 0, reused
                assert reused['cached_tokens'] > TARGET - 1024, reused
            original = call('/v1/responses/' + fresh['id'] + '/history')
            node = next((n for n in reversed(original['nodes']) if n['item'].get('role') == 'assistant'))
            distraction('switch-before-branch')
            protected, _ = generate(mode, state, 'branch-before-answer', [], codes, branch_from=dict(response_id=fresh['id'], node_id=node['id'], side='before'))
            assert call('/v1/responses/' + fresh['id'] + '/history') == original
            call('/v1/responses/' + protected['id'] + '/bookmark', {'protected': True})
            ordinary, _ = distraction('ordinary-before-stop')
            catalog = state / 'responses/execution-cache/checkpoints.sqlite3'
            assert query(catalog, 'SELECT count(*) FROM owners JOIN snapshots ON snapshots.id=owners.snapshot WHERE response=?', (protected['id'],))[0][0] > 0, 'selected checkpoint was evicted before shutdown'
            stop(process)
            owners = query(catalog, 'SELECT response FROM owners JOIN snapshots ON snapshots.id=owners.snapshot')
            assert (protected['id'],) in owners and (ordinary['id'],) not in owners, owners
            AUDITS.append(dict(mode=mode, kind='shutdown-cleanup', protected_retained=True, ordinary_removed=True))
            save()
            process = start()
            _, restored = generate(mode, state, 'protected-restart', follow, codes, previous_response_id=protected['id'])
            assert restored['disk_restores'] > 0, restored
            assert restored['cached_tokens'] > TARGET - 1024, restored
            stop(process)
            blocks = state / 'responses/execution-cache/blocks'
            assert blocks.resolve().is_relative_to(ROOT.resolve())
            deleted = 0
            for p in blocks.iterdir():
                if p.is_file() and len(p.name) == 64 and all((c in '0123456789abcdef' for c in p.name)):
                    p.unlink()
                    deleted += 1
            assert deleted > 0
            AUDITS.append(dict(mode=mode, kind='cache-deletion', blocks=deleted))
            save()
            process = start()
            _, replayed = generate(mode, state, 'deleted-cache-replay', follow, codes, previous_response_id=protected['id'])
            assert replayed['cached_tokens'] <= len(prefix), replayed
            assert replayed['disk_restores'] == 0, replayed
            save()
        finally:
            if process.poll() is None:
                stop(process)
if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    run('plain')
    (ROOT / 'COMPLETE').write_text('All plain-server assertions passed.\n')
    print('COMPLETE', flush=True)
