"""G3/G5 real-model HTTP/official-SDK qualification; one native process at a time.

Use an idle authorized GPU, private config and a fresh evidence directory. No
model output is scripted. This does not establish native-model Codex tool skill.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sys
import time
import traceback
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from cryptography.fernet import Fernet
import openai
from target_only_probe import tokenizer
from serve.frontend import ChatTemplate
from serve.grammar import CAPABILITY
from serve.response_replay import ReplayCodec
from serve.server import Service, StrataEngine, child_env, serve


MODEL = 'qwen3.8-flash-next'
PROMPT = 'Output the answer to 2+2 as one digit. No explanation.'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--unsupported-config', type=Path, action='append', default=[])
    ap.add_argument('--mode', choices=('target','mtp','coupled','suffix','suffix-coupled'), default='target')
    ap.add_argument('--speculation-benchmark', action='store_true')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.config.read_text(encoding='utf-8'))
    report = {'result': 'running', 'scripted_model_output': False, 'sdk_version': openai.__version__,
              'executable_sha256': hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest(),
              'args': cfg['args'], 'mode': a.mode, 'cases': []}
    benchmarks = []
    engine = httpd = client = svc = None

    def save(name, **data):
        report['cases'].append({'name': name, **data})
        (a.out / 'result.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(name, json.dumps(data, ensure_ascii=False)[:500], flush=True)

    def close():
        nonlocal engine, httpd, client
        if client is not None:
            client.close(); client = None
        if httpd is not None:
            httpd.shutdown(); httpd.server_close(); httpd = None
        if engine is not None:
            engine.close(); engine = None

    def start(config, label):
        nonlocal engine, httpd, client, svc
        env = child_env(config)
        env.pop('STRATA_SPEC_COUPLED', None)
        # Controlled timings: trace is off. Native G2 retained the cursor audit.
        env.pop('STRATA_TRACE', None)
        env.pop('STRATA_GRAMMAR_LOGITS', None)
        engine = StrataEngine(config['exe'], config['args'], cwd=config['cwd'],
                              log=str(a.out / (label + '-engine.txt')), env=env)
        tok = tokenizer(config['tokenizer'])
        svc = Service(engine, tok, ChatTemplate(Path(config['tokenizer']) / 'chat_template.jinja'))
        svc.experimental_responses = True
        svc.responses_replay = ReplayCodec(Fernet.generate_key())
        svc.api_key = secrets.token_urlsafe(24)  # memory only; never evidence
        svc.start_telemetry = lambda: None
        httpd = serve(svc, port=0)
        client = openai.OpenAI(base_url=f'http://127.0.0.1:{httpd.server_address[1]}/v1',
                               api_key=svc.api_key, timeout=180, max_retries=0)
        save(label + ' capability', info=engine.info)

    def args(api, grammar=None, cap=128, prompt=PROMPT, **extra):
        common = dict(model=MODEL, temperature=0)
        if grammar is not None:
            common['extra_body'] = {'grammar': grammar}
        if api == 'responses':
            return {**common, 'input': prompt, 'store': False, 'reasoning': {'effort': 'none'}, 'max_output_tokens': cap, **extra}
        return {**common, 'messages': [{'role': 'user', 'content': prompt}], 'reasoning_effort': 'none',
                'max_tokens': cap, **extra}

    def run(api, grammar=None, stream=False, **kw):
        started = time.perf_counter()
        first_delta = None
        call = client.responses.create if api == 'responses' else client.chat.completions.create
        result = call(**args(api, grammar, stream=stream, **kw))
        if not stream:
            text = result.output_text if api == 'responses' else result.choices[0].message.content
            status = result.status if api == 'responses' else result.choices[0].finish_reason
            wire = result.model_dump(mode='json')
        else:
            text, status, wire = '', None, []
            with result:
                for event in result:
                    wire.append(event.model_dump(mode='json'))
                    if api == 'responses':
                        if event.type == 'response.output_text.delta':
                            text += event.delta
                            if event.delta and first_delta is None: first_delta = time.perf_counter() - started
                        if event.type in ('response.completed', 'response.incomplete', 'response.failed'):
                            status = event.response.status
                            assert text == event.response.output_text
                    elif event.choices:
                        text += event.choices[0].delta.content or ''
                        if event.choices[0].delta.content and first_delta is None: first_delta = time.perf_counter() - started
                        status = event.choices[0].finish_reason or status
            if api == 'responses':
                assert [e['sequence_number'] for e in wire] == list(range(len(wire)))
        return dict(text=text, status=status, wire=wire, wall_s=time.perf_counter() - started,
                    first_text_delta_s=first_delta, native=dict(engine.last))

    def bad(api, extra):
        payload = args(api, 'root ::= "true"', stream=True)
        payload.update(payload.pop('extra_body'))
        payload.update(extra)
        path = '/v1/responses' if api == 'responses' else '/v1/chat/completions'
        url = f'http://127.0.0.1:{httpd.server_address[1]}' + path
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
               headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + svc.api_key})
        try:
            urllib.request.urlopen(req, timeout=180).close()
        except urllib.error.HTTPError as exc:
            data = exc.read().decode()
            assert exc.code == 400 and 'application/json' in exc.headers['Content-Type'], data
            return dict(status=exc.code, content_type=exc.headers['Content-Type'], error=json.loads(data))
        raise AssertionError('unsupported grammar request returned success')

    try:
        start(cfg, a.mode)
        assert engine.info['grammar'] == CAPABILITY
        assert engine.info['decode_mode'] == ('target' if a.mode == 'target' else 'mtp')
        assert engine.info['mtp_loaded'] == int(a.mode != 'target')
        assert bool(engine.info['lookup']) == a.mode.startswith('suffix')
        assert ('--coupled-draft' in cfg['args']) == a.mode.endswith('coupled')
        literal = '<think>literal</think>\n猫 café'
        grammar = 'root ::= ' + json.dumps(literal, ensure_ascii=False)
        # Multiline/quotes and command-looking comments are inert grammar data.
        grammar += '\n# STOP\n# GEN 4 99\n'
        for api in ('responses', 'chat'):
            for stream in (False, True):
                result = run(api, grammar, stream)
                assert result['text'] == literal and result['status'] in ('completed', 'stop'), result
                save(api + (' SSE' if stream else ' JSON') + ' literal markers and Unicode', **result)
            result = run(api, grammar, stream=True, temperature=0.8, top_p=0.85)
            assert result['text'] == literal and result['status'] in ('completed', 'stop')
            save(api + ' sampled SSE', **result)
            result = run(api, 'root ::= ""', stream=True)
            assert result['text'] == '' and result['status'] in ('completed', 'stop')
            save(api + ' epsilon completion', **result)
            result = run(api, 'root ::= "(" root ")" | "x"', prompt='Write ((x)). No explanation.')
            assert re.fullmatch(r'\(*x\)*', result['text']) and result['text'].count('(') == result['text'].count(')')
            assert result['status'] in ('completed', 'stop')
            save(api + ' recursive language', **result)
            for extra in ({'grammar': 'root ::= missing'}, {'grammar': 'root ::= root'}, {'grammar': 'root ::= "x"\0'},
                          {'grammar': 'root ::= "' + '猫' * 500 + '" invalid_reference'},
                          {'strata_mcp': True}, {'stop': 'END'},
                          {'stream_options': {'include_usage': True}},
                          {'reasoning': {'effort': 'invalid'}},
                          {'text': {'format': {'type': 'json_object'}}} if api == 'responses'
                          else {'response_format': {'type': 'json_object'}}):
                before = len(svc.history)
                save(api + ' rejection before SSE headers', request_extra=extra, **bad(api, extra))
                assert len(svc.history) == before, 'validation started generation'
        # Budget at every token boundary of the forced UTF-8 literal, including
        # tokenizer byte fragments. Final output is the concatenated stream.
        parameters = {'grammar': grammar, 'repetition_penalty': 1.2, 'penalty_last_n': 64,
                      'top_k': 3, 'min_p': 0.05, 'seed': 434}
        result = run('chat', grammar, stream=True, temperature=0.8, top_p=0.85,
                     presence_penalty=0.1, frequency_penalty=0.2, extra_body=parameters)
        assert result['text'] == literal and result['status'] == 'stop'
        save('Chat sampled penalties', parameters={**parameters, 'temperature': 0.8, 'top_p': 0.85,
                                                   'presence_penalty': 0.1, 'frequency_penalty': 0.2}, **result)
        source = 'root ::= "café 🐈"'
        whole = run('responses', source)
        for cap in range(1, whole['native']['generated']):
            result = run('responses', source, stream=True, cap=cap)
            assert result['status'] == 'incomplete' and 'café 🐈'.startswith(result['text'])
            assert '\ufffd' not in result['text'] and result['native']['finish'] == 'length'
            save('SSE UTF-8 budget ' + str(cap), **result)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(run, api, 'root ::= "' + value + '"')
                       for api, value in (('responses', 'alpha'), ('chat', 'beta'))]
            results = [f.result() for f in futures]
            assert [r['text'] for r in results] == ['alpha', 'beta']
            # Per-request wire outputs prove isolation. Shared diagnostic 'last'
            # is not used to attribute concurrent requests' native timing.
            save('concurrent adapters independent', outputs=[r['text'] for r in results])
        long_source = 'root ::= "' + 'a ' * 240 + '"'
        streamed = client.responses.create(**args('responses', long_source, stream=True, cap=512))
        with streamed:
            for event in streamed:
                if event.type == 'response.output_text.delta':
                    break
        clean = run('chat', 'root ::= "true"')
        assert clean['text'] == 'true' and clean['status'] == 'stop'
        assert any(h['finish'] in ('cancel', 'disconnect') for h in svc.history)
        save('HTTP disconnect drained and next request clean', text=clean['text'],
             history=[{k: h.get(k) for k in ('finish', 'output_tokens', 'engine_generated')} for h in svc.history][-2:])
        # Equal prompt/output distributions, warmed immutable grammar and prompt
        # caches, alternating order. No trace and no speculation in either row.
        plain = run('responses', cap=32)
        measured = 'root ::= ' + json.dumps(plain['text'], ensure_ascii=False)
        assert plain['status'] == 'completed'
        warm = run('responses', measured, cap=32)
        assert warm['text'] == plain['text']
        save('benchmark warmup', output=plain['text'], plain_native=plain['native'], grammar_native=warm['native'])
        for repeat in range(6):
            for constrained in ((False, True) if repeat % 2 == 0 else (True, False)):
                result = run('responses', measured if constrained else None, cap=32)
                assert result['text'] == plain['text'] and result['status'] == 'completed'
                last = result['native']
                benchmarks.append({'repeat': repeat, 'constrained': constrained, 'wall_s': result['wall_s'],
                    **{k: last[k] for k in ('generated', 'prompt_ms', 'decode_ms', 'reused', 'drafts_offered')}})
        with (a.out / 'target-only-benchmarks.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(benchmarks[0])); w.writeheader(); w.writerows(benchmarks)
        save('alternating benchmark passed', rows=len(benchmarks), output=plain['text'])
        if a.speculation_benchmark:
            rows = []
            for label, api, sampled, value, text_prompt in (
                ('digits', 'responses', False, '0 1 2 3 4 5 6 7 8 9',
                 'Write the digits 0 through 9 separated by spaces. No explanation.'),
                ('repeats', 'responses', False, 'alpha beta gamma delta ' * 8,
                 'Repeat this line exactly, eight times, without extra text:\nalpha beta gamma delta'),
                ('sampled-repeats', 'chat', True, 'alpha beta gamma delta ' * 8,
                 'Repeat this line exactly, eight times, without extra text:\nalpha beta gamma delta')):
                rule = 'root ::= ' + json.dumps(value)
                options = dict(prompt=text_prompt, cap=96)
                if sampled:
                    options.update(temperature=0.8, top_p=0.85, presence_penalty=0.05, frequency_penalty=0.1,
                        extra_body=dict(grammar=rule, seed=434, top_k=20, min_p=0.04,
                                        penalty_last_n=64, repetition_penalty=1.1))
                warm = run(api, rule, **options)
                assert warm['text'] == value and warm['status'] in ('completed', 'stop')
                for repeat in range(6):
                    for streamed in ((False,True) if repeat % 2 == 0 else (True,False)):
                        r = run(api, rule, stream=streamed, **options)
                        assert r['text'] == value and r['status'] in ('completed', 'stop')
                        rows.append(dict(mode=a.mode, workload=label, api=api, sampled=sampled, repeat=repeat, stream=streamed,
                            wall_s=r['wall_s'], first_text_delta_s=r['first_text_delta_s'],
                            **{k:r['native'][k] for k in ('generated','prompt_ms','decode_ms','reused','drafts_offered','drafts_accepted')}))
                with concurrent.futures.ThreadPoolExecutor(2) as pool:
                    pending = [pool.submit(run, api, rule, stream=streamed, **options)
                               for streamed in (False,True)]
                    concurrent_results = [f.result() for f in pending]
                assert all(r['text'] == value and r['status'] in ('completed', 'stop') for r in concurrent_results)
                save('concurrent benchmark ' + label, results=[{k:r[k] for k in ('wall_s','first_text_delta_s')} for r in concurrent_results])
            with (a.out/'speculation-benchmarks.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            save('constrained speculation benchmark', rows=len(rows))
        close()
        for i, path in enumerate(a.unsupported_config):
            rejected_cfg = json.loads(path.read_text(encoding='utf-8'))
            start(rejected_cfg, 'unsupported-' + str(i))
            assert engine.info.get('grammar') != CAPABILITY
            for api in ('responses', 'chat'):
                save(api + ' unqualified engine rejected', **bad(api, {}))
                assert not svc.history
            close()
        report['result'] = 'pass'
    except Exception:
        report['result'] = 'fail'
        report['failure'] = traceback.format_exc()
        raise
    finally:
        close()
        (a.out / 'result.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
