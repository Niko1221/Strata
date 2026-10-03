"""Small real-model GBNF HTTP demonstration. No scripted model output.

Runs the existing Service/StrataEngine on loopback, then stops its own resources.
The --manifest binds production sources and the native binary to the reviewed head.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request


PROMPT = 'Reply with exactly this sentence: The sky is purple and the number is 999.'
GRAMMAR = 'root ::= "color=" color ";count=" count "\\n"\ncolor ::= "red" | "green" | "blue"\ncount ::= [1-3]'
PATTERN = r'color=(red|green|blue);count=[1-3]\n'


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt('probe interrupted by signal ' + str(signum))
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    report = {'result': 'running', 'source_commit': manifest['source_commit'],
              'branch': manifest['branch'], 'scripted_model_output': False,
              'model': 'Coder IQ1_M', 'probe_pid': os.getpid(), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'prompt': PROMPT, 'grammar': GRAMMAR, 'modes': []}

    def persist():
        (args.out / 'result.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    def gpu_processes():
        return subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                        '--format=csv,noheader'], text=True).strip()

    try:
        for path, expected in manifest['source_hashes'].items():
            actual = hashlib.sha256((args.source / path).read_bytes()).hexdigest()
            assert actual == expected, 'source mismatch: ' + path
        report['verified_source_files'] = len(manifest['source_hashes'])
        report['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                                '--format=csv,noheader'], text=True).strip()
        report['gpu_before'] = gpu_processes()
        assert not report['gpu_before'], 'GPU has an existing compute process; refusing to start'
        sys.path[:0] = [str(args.source), str(args.source / 'tools')]
        from cryptography.fernet import Fernet
        from target_only_probe import tokenizer
        from serve.frontend import ChatTemplate
        from serve.grammar import CAPABILITY
        from serve.response_replay import ReplayCodec
        from serve.server import Service, StrataEngine, child_env, serve

        os.environ['STRATA_RESPONSES_REPLAY_KEY'] = Fernet.generate_key().decode()
        os.environ['STRATA_API_KEY'] = secrets.token_urlsafe(32)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        for config_path in args.config:
            config = json.loads(config_path.read_text(encoding='utf-8'))
            mode = 'target' if config['args'][config['args'].index('--spec') + 1] == '1' else 'mtp'
            folder = args.out / mode
            folder.mkdir()
            mode_record = {'mode': mode, 'config': config, 'cases': []}
            report['modes'].append(mode_record)
            binary_hash = hashlib.sha256(Path(config['exe']).read_bytes()).hexdigest()
            assert binary_hash == manifest['native_sha256'], 'native executable changed'
            mode_record['native_sha256'] = binary_hash
            assert not gpu_processes(), 'GPU became busy before starting ' + mode
            engine = httpd = None
            try:
                env = child_env(config)
                env.pop('STRATA_SPEC_COUPLED', None)
                env.pop('STRATA_GRAMMAR_LOGITS', None)
                env.update(STRATA_TRACE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                           HTTP_PROXY='http://127.0.0.1:9', HTTPS_PROXY='http://127.0.0.1:9',
                           ALL_PROXY='http://127.0.0.1:9', NO_PROXY='127.0.0.1,localhost')
                print('Starting real native ' + mode + ' decoder', flush=True)
                engine = StrataEngine(config['exe'], config['args'], cwd=config['cwd'],
                                      log=str(folder / 'engine.txt'), env=env)
                tok = tokenizer(config['tokenizer'])
                svc = Service(engine, tok, ChatTemplate(Path(config['tokenizer']) / 'chat_template.jinja'))
                svc.experimental_responses = True
                svc.responses_replay = ReplayCodec(os.environ['STRATA_RESPONSES_REPLAY_KEY'])
                svc.api_key = os.environ['STRATA_API_KEY']
                svc.start_telemetry = lambda: None
                httpd = serve(svc, host='127.0.0.1', port=0)
                mode_record['native_info'] = dict(engine.info)
                assert engine.info['grammar'] == CAPABILITY
                assert engine.info['decode_mode'] == ('target' if mode == 'target' else 'mtp')
                assert engine.info['mtp_loaded'] == int(mode == 'mtp')
                persist()

                def request(label, payload, api='responses'):
                    path = '/v1/responses' if api == 'responses' else '/v1/chat/completions'
                    prefix = folder / label
                    prefix.with_suffix('.request.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
                    before_history = len(svc.history)
                    before_native = dict(engine.last)
                    req = urllib.request.Request(f'http://127.0.0.1:{httpd.server_address[1]}' + path,
                           data=json.dumps(payload, ensure_ascii=False).encode(),
                           headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + svc.api_key})
                    started = time.perf_counter()
                    try:
                        with opener.open(req, timeout=180) as response:
                            status = response.status
                            content_type = response.headers['Content-Type']
                            raw = response.read().decode('utf-8')
                    except urllib.error.HTTPError as exc:
                        status = exc.code
                        content_type = exc.headers['Content-Type']
                        raw = exc.read().decode('utf-8')
                    record = {'name': label, 'path': path, 'request': payload, 'http_status': status,
                              'content_type': content_type, 'wall_seconds': round(time.perf_counter() - started, 4),
                              'native_done': dict(engine.last)}
                    if 'text/event-stream' in content_type:
                        events = []
                        for line in raw.splitlines():
                            if line.startswith('data:') and line[5:].strip() != '[DONE]':
                                events.append(json.loads(line[5:]))
                        prefix.with_suffix('.sse.txt').write_text(raw, encoding='utf-8')
                        assert api == 'responses'
                        assert [event['sequence_number'] for event in events] == list(range(len(events)))
                        terminal = [event for event in events if event['type'] in
                                    ('response.completed', 'response.incomplete', 'response.failed')]
                        assert len(terminal) == 1, 'terminal lifecycle must occur exactly once'
                        final = terminal[0]['response']
                        text = ''.join(event['delta'] for event in events if event['type'] == 'response.output_text.delta')
                        final_text = ''.join(part['text'] for item in final['output'] if item['type'] == 'message'
                                             for part in item['content'] if part['type'] == 'output_text')
                        assert text == final_text, 'SSE deltas do not reproduce final text'
                        assert events[-1]['type'] == terminal[0]['type'], 'output after terminal'
                        for event in events:
                            if event['type'] == 'response.output_text.delta':
                                assert final['output'][event['output_index']]['id'] == event['item_id']
                        record.update(text=text, response_status=final['status'], final=final,
                                      stream_events=len(events), stream_matches_final=True)
                    else:
                        data = json.loads(raw)
                        prefix.with_suffix('.response.json').write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
                        if status != 200:
                            record.update(error=data, generation_started=(len(svc.history) != before_history or dict(engine.last) != before_native))
                        elif api == 'responses':
                            text = ''.join(part['text'] for item in data['output'] if item['type'] == 'message'
                                           for part in item['content'] if part['type'] == 'output_text')
                            record.update(text=text, response_status=data['status'], final=data)
                        else:
                            record.update(text=data['choices'][0]['message']['content'],
                                          response_status=data['choices'][0]['finish_reason'], final=data)
                    mode_record['cases'].append(record)
                    persist()
                    print(mode + ' / ' + label + ': ' + json.dumps({key: record[key] for key in
                          ('http_status', 'response_status', 'text', 'wall_seconds') if key in record}, ensure_ascii=False), flush=True)
                    return record

                base = {'model': svc.model, 'input': PROMPT, 'store': False,
                        'reasoning': {'effort': 'none'}, 'temperature': 0, 'max_output_tokens': 64}
                plain = request('01-unconstrained', base)
                assert plain['http_status'] == 200 and plain['response_status'] == 'completed'
                assert not re.fullmatch(PATTERN, plain['text']), 'baseline did not demonstrate the differing language'
                constrained = {**base, 'grammar': GRAMMAR}
                result = request('02-gbnf-json', constrained)
                assert result['response_status'] == 'completed' and re.fullmatch(PATTERN, result['text'])
                result = request('03-gbnf-sse', {**constrained, 'stream': True})
                assert result['response_status'] == 'completed' and re.fullmatch(PATTERN, result['text'])
                result = request('04-unicode-sse', {**constrained, 'grammar': 'root ::= "café 🐈"', 'stream': True})
                assert result['response_status'] == 'completed' and result['text'] == 'café 🐈'
                result = request('05-token-limit', {**constrained, 'max_output_tokens': 1, 'stream': True})
                assert result['response_status'] == 'incomplete'
                assert result['final']['incomplete_details']['reason'] == 'max_output_tokens'
                result = request('06-invalid-grammar', {**constrained, 'grammar': 'root ::= missing', 'stream': True})
                assert result['http_status'] == 400 and 'application/json' in result['content_type']
                assert result['generation_started'] is False
                chat = {'model': svc.model, 'messages': [{'role': 'user', 'content': PROMPT}],
                        'reasoning_effort': 'none', 'temperature': 0, 'max_tokens': 64, 'grammar': GRAMMAR}
                result = request('07-chat-gbnf-json', chat, api='chat')
                assert result['response_status'] == 'stop' and re.fullmatch(PATTERN, result['text'])
                mode_record['result'] = 'pass'
            finally:
                if httpd is not None:
                    httpd.shutdown()
                    httpd.server_close()
                if engine is not None:
                    engine.close()
                    mode_record['engine_stopped'] = not engine.alive()
                mode_record['gpu_after'] = gpu_processes()
                persist()
            assert mode_record.get('engine_stopped') and not mode_record['gpu_after'], 'native cleanup incomplete'
        report['result'] = 'pass'
        print('PASS: 14 HTTP cases on real target-only and MTP native generation', flush=True)
    except BaseException:
        report['result'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        persist()


if __name__ == '__main__':
    main()
