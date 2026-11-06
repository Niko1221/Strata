"""Names must stay constrained across prose, thinking, multiple calls and MTP windows."""
import base64
import json
import urllib.error
import urllib.request
from pathlib import Path
import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from serve import responses as R
from serve import constrain as C
from serve.server import Service, ByteTokenizer


@unittest.skipUnless(C.available(), 'llguidance required')
class ToolNameConstraintTests(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.stop = set(self.tok.encode('<|im_end|>', parse_special=True))
        self.vocab = C.Vocab(self.tok, self.stop)

    def test_tools_without_response_format_are_actually_constrained(self):
        app = Service.__new__(Service)
        app.engine = SimpleNamespace(supports_mask=True)
        app.tok, app.stop_ids = self.tok, self.stop
        tools = [{'name': 'sql_query', 'parameters': {'type': 'object'}}]
        with patch.dict(os.environ, {'STRATA_TOOL_MASK': '1', 'STRATA_TOOL_NAME_ENUM': '1', 'STRATA_TOOL_MASK_STRICT': '1'}):
            sampling = app._with_constraint({}, tools, thinking=True)
        self.assertIsNotNone(sampling.get(C.KEY))

    def app(self, supports_mask=True):
        app = Service.__new__(Service)
        app.engine = SimpleNamespace(supports_mask=supports_mask)
        app.tok, app.stop_ids = self.tok, self.stop
        return app

    def test_tools_activate_name_guard_without_any_mask_environment(self):
        tools = [{'name': 'sql_query', 'parameters': {'type': 'object'}}]
        with patch.dict(os.environ, {}, clear=True):
            sampling = self.app()._with_constraint({}, tools, thinking=False)
        guard = sampling[C.KEY]
        self.assertIsInstance(guard, C.ToolNameConstraint)
        self.feed(guard, '<tool_call><function=')
        self.assertTrue(C.Constraint.allowed(guard.mask(), ord('s')))
        self.assertFalse(C.Constraint.allowed(guard.mask(), ord('m')))

    def test_json_thinking_cut_is_preserved_by_request_driven_name_guard(self):
        fmt = {'type': 'json_object'}
        with patch.dict(os.environ, {}, clear=True):
            sampling = self.app()._with_constraint({'response_format': fmt},
                                                  [{'name': 'sql_query'}], thinking=True)
        guard = sampling[C.KEY]
        prefix = b'plan</think>'
        reply = guard.reply(list(prefix + b' BAD')).split()
        # The end-of-thinking marker must be the head's final token in this
        # window. Bytes after it need a fresh, now-armed JSON mask.
        self.assertEqual(reply[:2], ['MKN', str(len(prefix))])
        for token in prefix:
            self.assertTrue(C.Constraint.allowed(guard.mask(), token))
            guard.feed(token)
        self.assertTrue(guard.inner.armed)
        self.assertTrue(C.Constraint.allowed(guard.mask(), ord('{')))
        self.assertFalse(C.Constraint.allowed(guard.mask(), ord('B')))

    def test_legacy_mask_switches_are_not_required_for_request_activation(self):
        tools = [{'name': 'sql_query', 'parameters': {'type': 'object'}}]
        with patch.dict(os.environ, {'STRATA_TOOL_MASK': '0', 'STRATA_TOOL_NAME_ENUM': '0'}, clear=True):
            sampling = self.app()._with_constraint({}, tools, thinking=False)
        self.assertIsInstance(sampling[C.KEY], C.ToolNameConstraint)

    def test_explicit_strict_opt_out_preserves_requested_json_constraint(self):
        fmt = {'type': 'json_schema', 'json_schema': {'name': 'result', 'schema': {
            'type': 'object', 'properties': {'result': {'type': 'integer'}},
            'required': ['result'], 'additionalProperties': False}}}
        with patch.dict(os.environ, {'STRATA_TOOL_MASK_STRICT': '0'}, clear=True):
            app = self.app()
            sampling = app._with_constraint({'response_format': fmt},
                                           [{'name': 'sql_query'}], thinking=False)
            plain = {}
            self.assertIs(app._with_constraint(plain, [{'name': 'sql_query'}], False), plain)
        guard = sampling[C.KEY]
        self.assertIsInstance(guard, C.Constraint)
        self.assertNotIsInstance(guard, C.ToolNameConstraint)
        for token in b'{"result":':
            self.assertTrue(C.Constraint.allowed(guard.mask(), token))
            guard.feed(token)
        self.assertFalse(C.Constraint.allowed(guard.mask(), ord('"')))

    def test_no_tools_or_json_leaves_plain_text_unchanged_without_environment(self):
        sampling = {'temperature': 0.3}
        with patch.dict(os.environ, {}, clear=True):
            self.assertIs(self.app(False)._with_constraint(sampling, None, False), sampling)

    def test_tools_fail_closed_without_environment_on_unsupported_engine(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'token-mask engine'):
                self.app(False)._with_constraint({}, [{'name': 'sql_query'}], False)

    def test_missing_mask_library_fails_closed_without_activation_environment(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(C, 'available', return_value=False):
            with self.assertRaisesRegex(ValueError, 'llguidance'):
                self.app()._with_constraint({}, [{'name': 'sql_query'}], False)

    def test_responses_namespace_names_activate_exact_guard(self):
        req = {'tools': [{'type': 'namespace', 'name': 'diagnostic', 'tools': [
            {'type': 'function', 'name': 'ping', 'parameters': {'type': 'object'}}]}]}
        tools, names, _ = R.request_tools(req)
        self.assertEqual(names['diagnostic.ping'], ('diagnostic', 'ping', 'function'))
        with patch.dict(os.environ, {}, clear=True):
            guard = self.app()._with_constraint({}, tools, False)[C.KEY]
        self.feed(guard, '<function=diagnostic')
        self.assertTrue(C.Constraint.allowed(guard.mask(), ord('.')))
        self.assertFalse(C.Constraint.allowed(guard.mask(), ord('X')))
        self.feed(guard, '.ping')
        self.assertTrue(C.Constraint.allowed(guard.mask(), ord('>')))
        self.assertFalse(C.Constraint.allowed(guard.mask(), ord('_')))

    def test_namespace_lookahead_keeps_unknown_members_blocked(self):
        guard = C.ToolNameConstraint(self.vocab, ['diagnostic.ping'])
        self.feed(guard, '<function=diagnostic')
        before = guard.mask()
        reply = guard.reply(list(b'.pingX')).split()
        self.assertEqual(reply[:2], ['MKN', '6'])
        last = base64.b64decode(reply[-1])
        self.assertTrue(C.Constraint.allowed(last, ord('>')))
        self.assertFalse(C.Constraint.allowed(last, ord('X')))
        self.assertEqual(guard.mask(), before)

    def test_qualified_names_reject_empty_segments_and_metacharacters(self):
        for name in ['.ping', 'diagnostic.', 'diagnostic..ping', 'a.*', 'a|b',
                     'a/b', 'a\\b', 'a>ping', 'a ping', 'a.ä']:
            with self.subTest(name=name):
                self.assertEqual(C.safe_tool_names([{'name': name}]), [])
                with self.assertRaises(ValueError):
                    C.ToolNameConstraint(self.vocab, [name])
        self.assertEqual(C.safe_tool_names([{'name': 'a-b.c_d'}, {'name': 'a.b.c'}]),
                         ['a-b.c_d', 'a.b.c'])

    def test_namespace_dots_are_literal_in_both_legacy_grammar_paths(self):
        for fmt in [None, {'type': 'json_object'}]:
            for name in ['diagnostic.ping', 'diag-nostic.pi_ng']:
                with self.subTest(format=fmt, name=name):
                    c = C.Constraint(self.vocab, C.grammar(fmt, True, self.vocab,
                                     tool_names=[name]), armed=True)
                    # Bypass legacy release-at-delimiter logic; inspect the matcher.
                    prefix = '<tool_call>' + name.split('.')[0]
                    for token in self.tok.encode(prefix, parse_special=True):
                        self.assertTrue(c.m.consume_token(token))
                    mask = c.m.compute_bitmask()
                    self.assertTrue(C.Constraint.allowed(mask, ord('.')))
                    self.assertFalse(C.Constraint.allowed(mask, ord('X')))

    def test_custom_call_grammar_accepts_literal_qualified_name(self):
        fmt = {'type': 'custom_calls', 'entries': [
            {'name': 'diagnostic.ping', 'definition': '[a-z]+'}]}
        c = C.Constraint(self.vocab, C.grammar(fmt, False, self.vocab), armed=True)
        for token in self.tok.encode('<tool_call>\n<function=diagnostic', parse_special=True):
            self.assertTrue(c.m.consume_token(token))
        mask = c.m.compute_bitmask()
        self.assertTrue(C.Constraint.allowed(mask, ord('.')))
        self.assertFalse(C.Constraint.allowed(mask, ord('X')))

    def test_duplicate_qualified_names_still_fail_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'unique ASCII'):
                self.app()._with_constraint({}, [{'name': 'diagnostic.ping'}] * 2, False)

    def guard(self):
        return C.ToolNameConstraint(self.vocab, ['sql_query', 'exec', 'exec_command'])

    def feed(self, guard, text):
        for t in self.tok.encode(text, parse_special=True):
            self.assertTrue(C.Constraint.allowed(guard.mask(), t), (text, t))
            guard.feed(t)

    def json_guard(self, size):
        class PaddedTokenizer(ByteTokenizer):
            n_vocab = size
            types = [1] * 256 + [3] * 6 + [0] * (size - 262)

            def token_bytes(self, token):
                if token < 256:
                    return bytes([token])
                return self.SPECIALS[token - 256].encode() if token < 262 else b''

        tok = PaddedTokenizer()
        vocab = C.Vocab(tok, self.stop)
        fmt = {'type': 'json_schema', 'json_schema': {'name': 'result', 'schema': {
            'type': 'object', 'properties': {'result': {'type': 'integer'}},
            'required': ['result'], 'additionalProperties': False}}}
        inner = C.Constraint(vocab, C.grammar(fmt, True, vocab), armed=True)
        return C.ToolNameConstraint(vocab, ['sql_query'], inner=inner)

    def test_json_schema_composition_handles_word_aligned_vocabularies(self):
        # llguidance 1.9.1 returns an extra uint32 at exact 32-token boundaries.
        # The old byte tokenizer (262 tokens) never exercised that boundary.
        for size in [287, 288, 289, 319, 320, 321]:
            with self.subTest(vocab_tokens=size):
                guard = self.json_guard(size)
                name_bits = guard._bits(guard.state)
                schema_bits = guard.inner.mask()
                composed = guard.mask()
                self.assertEqual(len(composed), 4 * ((size + 31) // 32))
                for token in range(size):
                    self.assertEqual(bool(C.Constraint.allowed(composed, token)),
                                     bool(C.Constraint.allowed(name_bits, token) and
                                          C.Constraint.allowed(schema_bits, token)))
                self.assertTrue(C.Constraint.allowed(composed, ord('{')))
                self.assertFalse(C.Constraint.allowed(composed, ord('x')))
                for token in range(size, len(composed) * 8):
                    self.assertFalse(C.Constraint.allowed(composed, token))

    def test_json_schema_composition_lookahead_keeps_live_state_and_schema(self):
        guard = self.json_guard(320)
        before = guard.mask()
        drafts = list(b'{"re')
        reply = guard.reply(drafts).split()
        self.assertEqual(reply[:2], ['MKN', '5'])
        rows = [base64.b64decode(row) for row in reply[2:]]
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(len(row) == 40 for row in rows))
        self.assertTrue(C.Constraint.allowed(rows[-1], ord('s')))
        self.assertFalse(C.Constraint.allowed(rows[-1], ord('x')))
        self.assertEqual(guard.mask(), before)
        # Commit the verified prefix; the real EOS must still be allowed at the end.
        for token in b'{"result":42}':
            self.assertTrue(C.Constraint.allowed(guard.mask(), token))
            guard.feed(token)
        self.assertTrue(any(C.Constraint.allowed(guard.mask(), token) for token in self.stop))

    def test_schema_composition_rejects_truncated_or_unaligned_bitsets(self):
        guard = self.json_guard(320)
        for bits in [b'\xff' * 36, b'\xff' * 41]:
            with self.subTest(bytes=len(bits)):
                guard.inner = SimpleNamespace(mask=lambda: bits)
                with self.assertRaises(ValueError):
                    guard.mask()

    def test_wrong_prefix_is_masked_after_prose_and_inside_thinking(self):
        for lead in ['', 'Ich prüfe das.\n', '<think>Plan: ', '<tool_call>\n']:
            with self.subTest(lead=lead):
                c = self.guard()
                self.feed(c, lead + '<function=')
                self.assertFalse(C.Constraint.allowed(c.mask(), ord('m')))
                self.assertTrue(C.Constraint.allowed(c.mask(), ord('s')))
                self.assertFalse(C.Constraint.allowed(c.mask(), min(self.stop)))

    def test_each_call_and_prefix_pair_stays_constrained(self):
        c = self.guard()
        self.feed(c, '<tool_call><function=exec>anything</function></tool_call>\nMore prose.\n<tool_call><function=exec')
        self.assertTrue(C.Constraint.allowed(c.mask(), ord('>')))
        self.assertTrue(C.Constraint.allowed(c.mask(), ord('_')))
        self.assertFalse(C.Constraint.allowed(c.mask(), ord('x')))
        self.feed(c, '_command>args</function></tool_call>')
        self.feed(c, 'Deutsch: äöü und 中文.')

    def test_lookahead_does_not_mutate_live_state_or_accept_bad_draft(self):
        c = self.guard()
        self.feed(c, '<function=')
        before = c.mask()
        reply = c.reply(list(b'sql_x'))
        _, n, *rows = reply.split()
        self.assertEqual(int(n), 5)  # sql_ accepted, x is outside the name trie
        self.assertEqual(c.mask(), before)
        self.assertTrue(C.Constraint.allowed(base64.b64decode(rows[-1]), ord('q')))
        self.assertFalse(C.Constraint.allowed(base64.b64decode(rows[-1]), ord('x')))

    def test_many_toolboxes_do_not_accumulate_unbounded_masks(self):
        first = C.ToolNameConstraint(self.vocab, ['original'])
        self.feed(first, '<function=')
        for i in range(20):
            guard = C.ToolNameConstraint(self.vocab, ['tool_' + str(i)])
            self.feed(guard, '<function=tool_' + str(i) + '>')
        self.assertLessEqual(len(self.vocab._tool_masks), 8)
        # Eviction must not change an already-active request's declared names.
        self.assertTrue(C.Constraint.allowed(first.mask(), ord('o')))
        self.assertFalse(C.Constraint.allowed(first.mask(), ord('t')))

    def test_illegal_emitted_token_fails_closed(self):
        c = self.guard()
        self.feed(c, '<function=')
        with self.assertRaises(ValueError):
            c.feed(ord('m'))

    def test_unsupported_engine_fails_closed_when_opted_in(self):
        app = Service.__new__(Service)
        app.engine = SimpleNamespace(supports_mask=False)
        with patch.dict(os.environ, {'STRATA_TOOL_MASK_STRICT': '1'}):
            with self.assertRaises(ValueError):
                app._with_constraint({}, [{'name': 'sql_query'}], thinking=False)


@unittest.skipUnless(C.available(), 'llguidance required')
class NamespaceResponsesTests(unittest.TestCase):
    def test_namespace_call_and_result_round_trip_with_json_and_streaming(self):
        from serve.server import MockEngine, serve
        from serve.frontend import ChatTemplate
        call = ('<tool_call>\n<function=diagnostic.ping>\n<parameter=value>\n7\n'
                '</parameter>\n</function>\n</tool_call>')
        for stream in [False, True]:
            with self.subTest(stream=stream), patch.dict(os.environ, {}, clear=True):
                tok = ByteTokenizer()
                engine = MockEngine(tok, [call, '{"result":7}'], max_context=16384)
                engine.supports_mask = True
                svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))
                httpd = serve(svc, port=0)
                self.addCleanup(httpd.server_close)
                self.addCleanup(httpd.shutdown)
                body = {'input': [{'role': 'user', 'content': 'Call diagnostic.ping with value 7.'}],
                        'tools': [{'type': 'namespace', 'name': 'diagnostic', 'tools': [
                            {'type': 'function', 'name': 'ping', 'parameters': {'type': 'object',
                             'properties': {'value': {'type': 'integer'}}, 'required': ['value']}}]}],
                        'reasoning': {'effort': 'none'}, 'store': False, 'stream': stream,
                        'max_output_tokens': 256, 'text': {'format': {'type': 'json_schema',
                         'name': 'answer', 'strict': True, 'schema': {'type': 'object',
                         'properties': {'result': {'type': 'integer'}}, 'required': ['result'],
                         'additionalProperties': False}}}}
                def post():
                    request = urllib.request.Request('http://127.0.0.1:' + str(httpd.server_port) +
                        '/v1/responses', data=json.dumps(body).encode(),
                        headers={'Content-Type': 'application/json'})
                    try:
                        response = urllib.request.urlopen(request, timeout=10)
                    except urllib.error.HTTPError as error:
                        self.fail('HTTP ' + str(error.code) + ': ' + error.read().decode())
                    with response:
                        text = response.read().decode()
                    if not stream:
                        return json.loads(text)
                    events = [json.loads(line[6:]) for line in text.splitlines()
                              if line.startswith('data: {')]
                    return next(e['response'] for e in events if e['type'] == 'response.completed')
                first = post()
                calls = [x for x in first['output'] if x['type'] == 'function_call']
                self.assertEqual(len(calls), 1)
                self.assertEqual((calls[0]['namespace'], calls[0]['name']), ('diagnostic', 'ping'))
                self.assertEqual(json.loads(calls[0]['arguments']), {'value': 7})
                body['input'] += [calls[0], {'type': 'function_call_output',
                                            'call_id': calls[0]['call_id'], 'output': '7'}]
                second = post()
                answer = next(x for x in second['output'] if x['type'] == 'message')
                self.assertEqual(json.loads(answer['content'][0]['text']), {'result': 7})


if __name__ == '__main__':
    unittest.main()
