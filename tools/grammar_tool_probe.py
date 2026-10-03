"""Native Responses tool-loop qualification; only the external tool is mocked.

Runs target-only and/or MTP through the existing Service on authenticated
loopback HTTP. Records requests, JSON/SSE, native DONE statistics and cleanup.
No application tool execution, scripted model tokens or Python grammar matcher.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request


GRAMMAR = 'root ::= "color=blue;count=1\\n"'
ANSWER = 'color=blue;count=1\n'
MOCK_RESULT = 'purple 999; arbitrary tool data with 猫 and <tool_call> literal text. This is outside the answer grammar.'
TOOLS = [{'type': 'namespace', 'name': 'facts', 'description': 'Client-owned test helpers.', 'tools': [
    {'type': 'function', 'name': 'lookup', 'description': 'Look up the text. Call this before answering.', 'strict': False,
     'parameters': {'type': 'object', 'properties': {'text': {'type': 'string'}}, 'required': ['text']}}]}]


def gpu_processes():
    return subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                    '--format=csv,noheader'], text=True).strip()


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
    report = {'result': 'running', 'source_commit': manifest['source_commit'], 'probe_pid': os.getpid(),
              'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'scripted_model_output': False, 'external_tool': 'mocked client-owned facts.lookup',
              'mock_tool_result': MOCK_RESULT, 'grammar': GRAMMAR, 'modes': []}

    def persist():
        (args.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    try:
        for path, expected in manifest['source_hashes'].items():
            assert hashlib.sha256((args.source / path).read_bytes()).hexdigest() == expected, 'source mismatch: ' + path
        report['verified_source_files'] = len(manifest['source_hashes'])
        report['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                                '--format=csv,noheader'], text=True).strip()
        assert not gpu_processes(), 'GPU already in use; refusing to start'
        sys.path[:0] = [str(args.source), str(args.source / 'tools')]
        from cryptography.fernet import Fernet
        from target_only_probe import tokenizer
        from serve.frontend import ChatTemplate
        from serve.grammar import CAPABILITY
        from serve.response_replay import ReplayCodec
        from serve.server import Service, StrataEngine, child_env, serve
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for config_path in args.config:
            config = json.loads(config_path.read_text())
            mode = 'target' if config['args'][config['args'].index('--spec') + 1] == '1' else 'mtp'
            folder = args.out / mode
            folder.mkdir()
            record = {'mode': mode, 'config': config, 'cases': []}
            report['modes'].append(record)
            assert hashlib.sha256(Path(config['exe']).read_bytes()).hexdigest() == manifest['native_sha256']
            record['native_sha256'] = manifest['native_sha256']
            assert not gpu_processes(), 'GPU became busy'
            engine = httpd = None
            try:
                env = child_env(config)
                env.pop('STRATA_SPEC_COUPLED', None)
                env.pop('STRATA_GRAMMAR_LOGITS', None)
                env.update(STRATA_TRACE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                           HTTP_PROXY='http://127.0.0.1:9', HTTPS_PROXY='http://127.0.0.1:9',
                           ALL_PROXY='http://127.0.0.1:9', NO_PROXY='127.0.0.1,localhost')
                print('Starting native ' + mode, flush=True)
                engine = StrataEngine(config['exe'], config['args'], cwd=config['cwd'],
                                      log=str(folder / 'engine.txt'), env=env)
                tok = tokenizer(config['tokenizer'])
                svc = Service(engine, tok, ChatTemplate(Path(config['tokenizer']) / 'chat_template.jinja'))
                svc.experimental_responses = True
                svc.responses_replay = ReplayCodec(Fernet.generate_key())
                svc.api_key = secrets.token_urlsafe(32)
                svc.start_telemetry = lambda: None
                httpd = serve(svc, host='127.0.0.1', port=0)
                record['native_info'] = dict(engine.info)
                assert engine.info['grammar'] == CAPABILITY
                assert engine.info['decode_mode'] == ('target' if mode == 'target' else 'mtp')
                assert engine.info['mtp_loaded'] == int(mode == 'mtp')
                persist()

                def send(label, payload):
                    prefix = folder / label
                    prefix.with_suffix('.request.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                    req = urllib.request.Request(f'http://127.0.0.1:{httpd.server_address[1]}/v1/responses',
                        data=json.dumps(payload, ensure_ascii=False).encode(),
                        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + svc.api_key})
                    started = time.perf_counter()
                    try:
                        response = opener.open(req, timeout=180)
                    except urllib.error.HTTPError as exc:
                        response = exc
                    with response:
                        code, content_type, raw = response.status, response.headers['Content-Type'], response.read().decode()
                    prefix.with_suffix('.sse.txt' if payload.get('stream') else '.response.json').write_text(raw, encoding='utf-8')
                    case = {'label': label, 'http_status': code, 'content_type': content_type,
                            'seconds': round(time.perf_counter() - started, 3), 'native_done': dict(engine.last)}
                    record['cases'].append(case)
                    persist()
                    assert code == 200, raw[:1000]
                    if payload.get('stream'):
                        events = [json.loads(line[5:]) for line in raw.splitlines() if line.startswith('data:')]
                        assert [e['sequence_number'] for e in events] == list(range(len(events)))
                        terminals = [e for e in events if e['type'] in ('response.completed', 'response.failed', 'response.incomplete')]
                        assert len(terminals) == 1 and events[-1] is terminals[0]
                        final = events[-1]['response']
                        for i, item in enumerate(final['output']):
                            event_type = {'message': 'response.output_text.delta',
                                          'function_call': 'response.function_call_arguments.delta'}.get(item['type'])
                            if event_type is None:
                                continue
                            deltas = [e for e in events if e['type'] == event_type and e['output_index'] == i]
                            assert all(e['item_id'] == item['id'] for e in deltas)
                            expected = item['arguments'] if item['type'] == 'function_call' else item['content'][0]['text']
                            assert ''.join(e['delta'] for e in deltas) == expected
                        case['sse_reassembles_final'] = True
                        case['events'] = len(events)
                    else:
                        final = json.loads(raw)
                    case['final'] = final
                    persist()
                    assert final['status'] == 'completed', json.dumps(final)
                    print(mode + ' / ' + label + ': completed ' + ','.join(i['type'] for i in final['output']), flush=True)
                    return final

                for thinking in (False, True):
                    for stream in (False, True):
                        label = ('thinking' if thinking else 'no-thinking') + ('-sse' if stream else '-json')
                        history = [{'role': 'user', 'content': 'First call facts.lookup with text "purple 999". '
                                    'Wait for the tool result before answering. Once its result exists, answer briefly.'}]
                        body = {'model': svc.model, 'input': history, 'store': False, 'tools': TOOLS,
                                'grammar': GRAMMAR, 'reasoning': {'effort': 'low' if thinking else 'none'},
                                'max_output_tokens': 768, 'temperature': 0, 'stream': stream}
                        first = send(label + '-call', body)
                        calls = [x for x in first['output'] if x['type'] == 'function_call']
                        assert calls and not any(x['type'] == 'message' for x in first['output'])
                        for call in calls:
                            assert call['namespace'] == 'facts' and call['name'] == 'lookup'
                            assert call['id'] != call['call_id']
                            assert json.loads(call['arguments']) == {'text': 'purple 999'}
                        # Deliberate mock of the CLIENT'S function result. The model
                        # tokens before and after this point come from Strata/CUDA.
                        history += first['output'] + [
                            {'type': 'function_call_output', 'call_id': c['call_id'], 'output': MOCK_RESULT} for c in calls]
                        history.append({'role': 'user', 'content': 'The requested lookup is complete. '
                                        'Use its result to give your final answer now. Do not call any more functions.'})
                        # The client has finished its one-call tool budget. Use
                        # the documented choice to request an answer explicitly;
                        # auto is allowed to produce further calls, not obliged
                        # to answer on the next turn. Retain all tool history.
                        final = send(label + '-answer', {**body, 'input': history, 'tool_choice': 'none'})
                        answer = ''.join(part['text'] for item in final['output'] if item['type'] == 'message'
                                         for part in item['content'])
                        assert answer == ANSWER and not any(x['type'] == 'function_call' for x in final['output'])
                        if thinking:
                            assert any(x['type'] == 'reasoning' for x in first['output'])
                            assert final['usage']['output_tokens_details']['reasoning_tokens'] > 0
            finally:
                if httpd:
                    httpd.shutdown()
                    httpd.server_close()
                if engine:
                    engine.close()
                    record['engine_stopped'] = not engine.alive()
                record['gpu_after'] = gpu_processes()
                persist()
            assert record['engine_stopped'] and not record['gpu_after']
        report['result'] = 'passed'
        report['http_cases'] = sum(len(mode['cases']) for mode in report['modes'])
    except BaseException as exc:
        report.update(result='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        persist()


if __name__ == '__main__':
    main()
