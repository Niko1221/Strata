"""G3 adapter tests. Every model token here is synthetic, not native evidence."""
import concurrent.futures
import io
import json
import queue
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from cryptography.fernet import Fernet
from serve import server
from serve.frontend import ChatTemplate
from serve.grammar import CAPABILITY, GrammarConstraint, validate_grammar_request, vocabulary_identity
from serve.response_replay import ReplayCodec
from serve.responses import create_response, execute_response
from serve.test_responses import ROOT, MODEL, http, listening, normalized, request, service, sse_events


SOURCE = 'root ::= "<think>literal</think>\\n猫"'
SCRIPT = '<think>literal</think>\n猫'


class ByteTable(server.ByteTokenizer):
    token_types = [1] * 256 + [3] * len(server.ByteTokenizer.SPECIALS)

    def token_bytes(self, i):
        return bytes([i]) if i < 256 else self.SPECIALS[i - 256].encode()


class AdapterFixture(server.MockEngine):
    """Pretend only at the adapter boundary; this does not parse/enforce GBNF."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraints, self.checks = [], []

    def validate_constraint(self, constraint):
        self.checks.append(constraint)
        if constraint.source == 'root ::= missing':
            raise ValueError("synthetic compile error")
        return vocabulary_identity(self.tok, {257, 258})

    def generate(self, *args, constraint=None, **kwargs):
        if constraint is not None:
            assert 'grammar' not in args[2], 'grammar must be a typed argument, not a sampling setting'
        self.constraints.append(constraint)
        yield from super().generate(*args, **kwargs)


def fixture(script=SCRIPT, **kwargs):
    tok = ByteTable()
    svc = server.Service(AdapterFixture(tok, script, **kwargs), tok, ChatTemplate(ROOT / 'serve/chat_template.jinja'))
    svc.experimental_responses = True
    svc.responses_replay = ReplayCodec(Fernet.generate_key())
    return svc


def body(api, **extra):
    common = {'model': MODEL, 'grammar': SOURCE, 'temperature': 0}
    if api == 'responses':
        return {**common, 'input': 'hello', 'store': False, 'reasoning': {'effort': 'none'},
                'max_output_tokens': 128, **extra}
    return {**common, 'messages': [{'role': 'user', 'content': 'hello'}], 'reasoning_effort': 'none',
            'max_tokens': 128, **extra}


class Contract(unittest.TestCase):
    def test_utf8_byte_frame_and_inert_payload(self):
        source = 'root ::= "猫"\n# STOP\n# GEN 5 7\n'
        constraint = GrammarConstraint(source)
        raw = source.encode()
        self.assertEqual(constraint.frame('GEN 8 1,2'), b'GENG1 ' + str(len(raw)).encode() + b'\n' + raw + b'\nGEN 8 1,2\n')
        for command in ('QUIT', 'GEN 8 1\nSTOP', 'GEN 8 1\r', 'GEN 8 1\0'):
            with self.assertRaises(ValueError):
                constraint.frame(command)
        for source in (None, {}, '', 'a' * 8193, '\ud800', 'root ::= "x"\0'):
            with self.subTest(source=repr(source)[:50]), self.assertRaises(ValueError):
                GrammarConstraint(source)

    def test_conflicts(self):
        common = [dict(stop='END'), dict(strata_mcp=True),
                  dict(response_format={'type': 'json_object'}), dict(text={'format': {'type': 'json_schema'}}),
                  dict(reasoning={'effort': 'invalid'}), dict(reasoning_budget_tokens=16),
                  dict(temperature=float('nan')), dict(top_p=0),
                  dict(seed=-1), dict(top_k=65)]
        for api in ('responses', 'chat'):
            for extra in common:
                with self.subTest(api=api, extra=extra), self.assertRaises(ValueError):
                    validate_grammar_request(body(api, **extra), api)
            unset = body(api)
            unset.pop('reasoning' if api == 'responses' else 'reasoning_effort')
            self.assertIsNotNone(validate_grammar_request(unset, api))
        for extra in (dict(logit_bias={'7': -100}), dict(n=2), dict(logprobs=True), dict(modalities=['audio']),
                      dict(chat_template_kwargs={'enable_thinking': 'true'}), dict(n=True), dict(max_tokens=True),
                      dict(max_completion_tokens=8), dict(stream_options={'include_obfuscation': True}),
                      dict(stream_options={'include_usage': True}), dict(stream_options={'include_usage': False})):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_grammar_request(body('chat', **extra), 'chat')

    def test_mock_engine_does_not_claim_grammar(self):
        with self.assertRaisesRegex(ValueError, 'native grammar'):
            create_response(service(), body('responses'))

    def test_template_and_tokenizer_mismatch(self):
        svc = fixture()
        with mock.patch.object(svc.template, 'render', return_value='<|im_start|>assistant\n<think>\n'):
            with self.assertRaisesRegex(ValueError, 'template boundary'):
                create_response(svc, body('responses'))
        with mock.patch.object(svc.engine, 'validate_constraint', return_value='bytes-v1-fnv1a64:wrong'):
            with self.assertRaisesRegex(ValueError, 'byte tables differ'):
                create_response(svc, body('responses'))
        self.assertFalse(svc.engine.constraints)

    def test_partial_unicode_no_replacement_and_same_terminal(self):
        # Byte cap splits the first or second multibyte character. Only complete
        # UTF-8 is representable in JSON/SSE; withheld bytes still count as tokens.
        for cap in (1, 2, 3, 4, 5):
            svc = fixture('猫é')
            prepared = create_response(svc, body('responses', max_output_tokens=cap))
            events = list(execute_response(svc, prepared, threading.Event()))
            text = ''.join(e.get('delta', '') for e in events if e and e['type'] == 'response.output_text.delta')
            expected = '猫é'.encode()[:cap].decode('utf-8', errors='ignore')
            self.assertEqual(text, expected)
            self.assertNotIn('\ufffd', text)
            self.assertEqual(events[-1]['response']['status'], 'incomplete')
            self.assertEqual(events[-1]['response']['usage']['output_tokens'], cap)

    def test_invalid_utf8_or_incomplete_eos_fails_without_completion(self):
        for tokens in ([0xff, 257], [0xe7, 257]):
            svc = fixture()
            svc.engine.scripts = [tokens]
            svc.engine.script = tokens
            prepared = create_response(svc, body('responses'))
            events = list(execute_response(svc, prepared, threading.Event()))
            self.assertEqual(events[-1]['type'], 'response.failed')
            self.assertFalse(any(e.get('type') == 'response.completed' for e in events if e))
            self.assertFalse(svc.status['busy'])

    def test_close_releases_service_and_constraint_does_not_leak(self):
        svc = fixture([SCRIPT, 'clean'])
        prepared = create_response(svc, body('responses'))
        cancel = threading.Event()
        events = execute_response(svc, prepared, cancel)
        for event in events:
            if event and event['type'] == 'response.output_text.delta':
                break
        events.close()
        self.assertTrue(cancel.is_set())
        self.assertFalse(svc.status['busy'])
        plain = create_response(svc, request())
        list(execute_response(svc, plain, threading.Event()))
        self.assertIsNone(svc.engine.constraints[-1])
        self.assertEqual(plain.assembler.snapshot()['output'][0]['content'][0]['text'], 'clean')


class NativePipe(unittest.TestCase):
    def pipe(self, capability=CAPABILITY, lines=()):
        engine = object.__new__(server.StrataEngine)
        engine.info = {'grammar': capability}
        raw = io.BytesIO()
        engine.proc = SimpleNamespace(stdin=io.TextIOWrapper(raw, encoding='utf-8', newline='\r\n'))
        engine.lines = queue.Queue()
        for line in lines:
            engine.lines.put(line)
        engine.can_stop = True
        return engine, raw

    def test_missing_or_old_capability_writes_nothing(self):
        for cap in (None, 'none', 'gbnf-v1'):
            engine, raw = self.pipe(cap)
            with self.assertRaisesRegex(ValueError, 'gbnf-v2'):
                engine.validate_constraint(GrammarConstraint(SOURCE))
            self.assertEqual(raw.getvalue(), b'')

    def test_restart_does_not_reuse_old_capability(self):
        engine, _ = self.pipe()
        engine.spawn = ('unused', [], None, None, None)
        def older_init(self, *args):
            self.info = {'version': 'synthetic-older-engine'}
        with mock.patch.object(engine, 'close'), mock.patch.object(server.StrataEngine, '__init__', older_init):
            engine.restart()
        self.assertNotIn('grammar', engine.info)

    def test_preflight_and_generate_atomic_binary_frames(self):
        identity = 'bytes-v1-fnv1a64:1234'
        engine, raw = self.pipe(lines=['GRAMMAR_OK ' + identity + '\n', 'T 65\n', 'DONE 1 2 0 1 length\n'])
        constraint = GrammarConstraint(SOURCE)
        self.assertEqual(engine.validate_constraint(constraint), identity)
        self.assertEqual(list(engine.generate([1, 2], 1, {}, threading.Event(), constraint=constraint)), [65])
        self.assertEqual(raw.getvalue(), constraint.frame('CHECKG') + constraint.frame('GEN 1 1,2'))

    def test_penalties_reach_native_frame_once(self):
        engine, raw = self.pipe(lines=['DONE 0 2 0 1 length\n'])
        constraint = GrammarConstraint(SOURCE)
        values = {'temperature': 0.8, 'top_p': 0.85, 'top_k': 3, 'min_p': 0.05,
                  'repetition_penalty': 1.2, 'frequency_penalty': 0.2, 'presence_penalty': 0.1,
                  'penalty_last_n': 64, 'seed': 434}
        list(engine.generate([1, 2], 8, values, threading.Event(), constraint=constraint))
        command = raw.getvalue().splitlines()[-1].decode('ascii')
        for key in ('penalty_repeat=1.2', 'penalty_freq=0.2', 'penalty_present=0.1', 'penalty_last_n=64'):
            self.assertEqual(command.count(key), 1)

    def test_rejected_check_leaves_next_reply_intact(self):
        engine, _ = self.pipe(lines=['ERR invalid grammar\n', 'GRAMMAR_OK bytes-v1-fnv1a64:abcd\n'])
        with self.assertRaisesRegex(ValueError, 'invalid grammar'):
            engine.validate_constraint(GrammarConstraint(SOURCE))
        self.assertEqual(engine.validate_constraint(GrammarConstraint(SOURCE)), 'bytes-v1-fnv1a64:abcd')

    def test_unexpected_or_late_preflight_ends_desynchronized_process(self):
        engine, _ = self.pipe(lines=['T 999\n'])
        with mock.patch.object(engine, '_silent', return_value=server.EngineSilent('lost step')) as stop:
            with self.assertRaises(server.EngineSilent):
                engine.validate_constraint(GrammarConstraint(SOURCE))
            stop.assert_called_once()
        with mock.patch.object(engine.lines, 'get', side_effect=queue.Empty), \
                mock.patch.object(engine, '_silent', return_value=server.EngineSilent('late')) as stop:
            with self.assertRaises(server.EngineSilent):
                engine.validate_constraint(GrammarConstraint(SOURCE))
            stop.assert_called_once()


class HTTP(unittest.TestCase):
    def test_both_adapters_stream_and_final_preserve_literal_markers(self):
        svc = fixture()
        with listening(svc) as url:
            for api, path in (('responses', '/v1/responses'), ('chat', '/v1/chat/completions')):
                code, _, raw = http(url, body(api), path)
                self.assertEqual(code, 200, raw)
                final = json.loads(raw)
                code, _, raw = http(url, body(api, stream=True), path)
                self.assertEqual(code, 200, raw)
                if api == 'responses':
                    events, _ = sse_events(raw)
                    self.assertEqual(normalized(events[-1]['response']), normalized(final))
                    text = ''.join(e['delta'] for e in events if e['type'] == 'response.output_text.delta')
                    self.assertEqual(final['output'][0]['content'][0]['text'], SCRIPT)
                else:
                    chunks = [json.loads(x[6:]) for x in raw.decode().splitlines() if x.startswith('data: {')]
                    text = ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks if c.get('choices'))
                    self.assertEqual(final['choices'][0]['message']['content'], SCRIPT)
                    self.assertEqual(final['choices'][0]['finish_reason'], 'stop')
                self.assertEqual(text, SCRIPT)
        self.assertEqual(len(svc.engine.checks), 4)
        self.assertTrue(all(c.source == SOURCE for c in svc.engine.constraints))

    def test_errors_before_stream_headers_and_auth_cors(self):
        svc = fixture()
        with listening(svc) as url:
            for api, path in (('responses', '/v1/responses'), ('chat', '/v1/chat/completions')):
                for extra in (dict(grammar='root ::= missing'), dict(stop='END'), dict(reasoning={'effort': 'low'}),
                              dict(stream_options={'include_usage': True})):
                    code, headers, raw = http(url, body(api, stream=True, **extra), path)
                    self.assertEqual(code, 400, raw)
                    self.assertIn('application/json', headers['Content-Type'])
                code, _, _ = http(url, body(api), path, headers={'Origin': 'https://evil.example'})
                self.assertEqual(code, 403)
                svc.api_key = 'synthetic-test-key'
                code, _, _ = http(url, body(api), path)
                self.assertEqual(code, 401)
                svc.api_key = None
        self.assertFalse(svc.engine.constraints)

    def test_concurrent_preflight_and_generation_remain_request_local(self):
        svc = fixture('synthetic', delay_s=0.001)
        sources = ['root ::= "A"', 'root ::= "B"']
        with listening(svc) as url, concurrent.futures.ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(http, url, body('responses', grammar=source)) for source in sources]
            self.assertTrue(all(f.result()[0] == 200 for f in futures))
        self.assertCountEqual([c.source for c in svc.engine.checks], sources)
        self.assertCountEqual([c.source for c in svc.engine.constraints], sources)


if __name__ == '__main__':
    unittest.main()
