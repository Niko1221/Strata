"""G4 real-model client-state example plus native inspection/corpus qualification.

Only the example's in-memory task set changes. No task process is executed. Use
an idle authorized GPU, private config and fresh evidence directory.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import secrets
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from cryptography.fernet import Fernet
import openai
from grammar_state_contract import TaskState, StaleDecision, allowed_commands, derive_grammar, verify_candidate
from target_only_probe import tokenizer
from serve.frontend import ChatTemplate
from serve.response_replay import ReplayCodec
from serve.server import Service, StrataEngine, child_env, serve


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--inspector', type=Path, required=True)
    ap.add_argument('--matcher-tests', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.config.read_text(encoding='utf-8'))
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report = {'result': 'running', 'model_output': 'real native model, unscripted',
              'effects': 'in-memory toy application only; no jobs executed', 'sdk': openai.__version__,
              'engine_sha256': digest(cfg['exe']), 'inspector_sha256': digest(a.inspector),
              'matcher_test_sha256': digest(a.matcher_tests), 'trace': [], 'commands': []}
    engine = httpd = client = None
    corpus = []

    def trace(operation, occurrence, **data):
        report['trace'].append({'operation': operation, 'occurrence': occurrence, **data})
        (a.out / 'result.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(operation, occurrence, json.dumps(data, ensure_ascii=False)[:400], flush=True)

    def freeze(state, label):
        snapshot = state.snapshot('alice')
        trace('state.snapshot', label, state=asdict(snapshot), definition=snapshot.fingerprint())
        source = derive_grammar(snapshot)
        (a.out / (label + '.gbnf')).write_text(source, encoding='utf-8')
        corpus.append({'file': label + '.gbnf', 'accept': list(allowed_commands(snapshot)),
                       'reject': ['', 'WAIT\n', 'START unknown', 'START task-a\n']})
        trace('state.derive_language', label, grammar=source)
        return snapshot

    def propose(snapshot, label, cap=48):
        body = {'model': 'qwen3.8-flash-next', 'input': 'Choose START task-a if it is available. Otherwise choose WAIT.',
                'store': False, 'reasoning': {'effort': 'none'}, 'temperature': 0, 'max_output_tokens': cap,
                'grammar': derive_grammar(snapshot)}
        (a.out / (label + '-request.txt')).write_text('POST /v1/responses\n\n' +
              json.dumps(body, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        args = {k: v for k, v in body.items() if k != 'grammar'}
        response = client.responses.create(**args, extra_body={'grammar': body['grammar']})
        trace('state.request', label, response=response.model_dump(mode='json'), native=dict(engine.last))
        return response

    def candidate(snapshot, response, label):
        if response.status != 'completed':
            trace('state.validate', label, result='rejected', reason='generation did not complete', text=response.output_text)
            raise ValueError('generation did not complete')
        text = verify_candidate(snapshot, response.output_text)
        trace('state.validate', label, result='accepted', text=text)
        return text

    try:
        env = child_env(cfg)
        env.pop('STRATA_SPEC_COUPLED', None)
        engine = StrataEngine(cfg['exe'], cfg['args'], cwd=cfg['cwd'], log=str(a.out / 'engine.txt'), env=env)
        tok = tokenizer(cfg['tokenizer'])
        svc = Service(engine, tok, ChatTemplate(Path(cfg['tokenizer']) / 'chat_template.jinja'))
        svc.experimental_responses = True
        svc.responses_replay = ReplayCodec(Fernet.generate_key())
        svc.api_key = secrets.token_urlsafe(24)
        svc.start_telemetry = lambda: None
        httpd = serve(svc, port=0)
        client = openai.OpenAI(base_url=f'http://127.0.0.1:{httpd.server_address[1]}/v1',
                              api_key=svc.api_key, timeout=180, max_retries=0)
        report['engine_info'] = engine.info
        state = TaskState(('task-a', 'task-b'), {'alice': ('task-a', 'task-b'), 'bob': ('task-b',)})
        frozen = freeze(state, 'frozen')
        text = candidate(frozen, propose(frozen, 'stale'), 'stale')
        # Authorization changes AFTER real generation, BEFORE the atomic apply.
        state.replace_grant('alice', ('task-a',))
        current = state.snapshot('alice')
        try:
            state.apply('alice', frozen, text)
        except StaleDecision as exc:
            trace('state.check_revision', 'stale', result='rejected before effects', reason=str(exc))
        else:
            raise AssertionError('stale generation was applied')
        assert state.snapshot('alice') == current
        fresh = freeze(state, 'fresh')
        text = candidate(fresh, propose(fresh, 'fresh'), 'fresh')
        after = state.apply('alice', fresh, text)
        trace('state.apply', 'fresh', command=text, before=asdict(fresh), after=asdict(after))
        snapshot = freeze(state, 'after')
        incomplete = propose(snapshot, 'limited', cap=1)
        assert incomplete.status == 'incomplete'
        try:
            candidate(snapshot, incomplete, 'limited')
        except ValueError:
            pass
        else:
            raise AssertionError('incomplete generation authorized a transition')
        assert state.snapshot('alice') == snapshot
        cases = a.out / 'cases.json'
        cases.write_text(json.dumps(corpus, indent=2) + '\n', encoding='utf-8')
        # These are real native library executions, using the same sources the
        # client derived. No Python matcher, final regex or handoff toy parser.
        for label, command in (
                ('inspection', [str(a.inspector), cfg['tokenizer']]),
                ('derived-languages', [str(a.matcher_tests), str(cases), cfg['tokenizer'], str(a.out / 'vocab-audit.bin')])):
            report['commands'].append(command)
            with (a.out / (label + '.txt')).open('w', encoding='utf-8') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
            trace('constraint.native_test', label, result='passed')
        # The native test's required byte-table scratch output is reproducible
        # from the installed tokenizer. Retain its identity, not a model artifact.
        audit = a.out / 'vocab-audit.bin'
        report['vocab_audit_sha256'] = digest(audit)
        audit.unlink()
        report['result'] = 'pass'
    except Exception:
        report['result'] = 'fail'
        report['failure'] = traceback.format_exc()
        raise
    finally:
        if client is not None:
            client.close()
        if httpd is not None:
            httpd.shutdown(); httpd.server_close()
        if engine is not None:
            engine.close()
        (a.out / 'result.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        lines = ['Real G4 observations. Operation definitions and invocation occurrences are distinct.',
                 'Native inspection is a private matcher view, not a model checkpoint or ProgramModel.']
        for entry in report['trace']:
            lines.append(entry['operation'] + ' [' + entry['occurrence'] + '] ' +
                         json.dumps({k: v for k, v in entry.items() if k not in ('operation', 'occurrence')}, ensure_ascii=False))
        (a.out / 'state-trace.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
