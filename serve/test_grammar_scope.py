"""Synthetic token-channel fixtures over the real Service, adapters and HTTP.

These mock scripts do NOT enforce GBNF. Native matcher/GPU tests establish
enforcement separately. External tool results here are deliberately arbitrary.
"""
import copy
import json
import threading
import unittest
from unittest import mock

from cryptography.fernet import Fernet
from serve import server
from serve.frontend import ChatTemplate
from serve.grammar import GrammarConstraint, GrammarToken, vocabulary_identity
from serve.response_replay import ReplayCodec
from serve.responses import create_response, execute_response
from serve import test_grammar
from serve.test_grammar import AdapterFixture, ByteTable
from serve.test_responses import (ROOT, MODEL, function_script, function_tool, http, listening,
                                  normalized, request, sse_events)


class ProtocolTable(ByteTable):
    SPECIALS = ByteTable.SPECIALS + ['</think>', '<tool_call>', '</tool_call>']
    token_types = [1] * 256 + [3] * len(SPECIALS)


class ScopedFixture(AdapterFixture):
    def __init__(self, tok, scripts):
        super().__init__(tok, '')
        self.regions, self.index, self.closed, self.inputs = scripts, 0, 0, []

    def validate_constraint(self, constraint):
        self.checks.append(constraint)
        return vocabulary_identity(self.tok, {257, 258}, constraint.scoped)

    def generate(self, ids, *args, constraint=None, **kwargs):
        self.inputs.append(self.tok.decode(ids))
        regions = self.regions[min(self.index, len(self.regions) - 1)]
        self.index += 1
        tokens = []
        for channel, text in regions:
            for token in self.tok.encode(text, parse_special=channel in ('control', 'tool')):
                tokens.append(GrammarToken(token, channel) if constraint and constraint.scoped else token)
        tokens.append(GrammarToken(257, 'control') if constraint and constraint.scoped else 257)
        self.scripts = [tokens]
        self.script = tokens
        try:
            yield from super().generate(ids, *args, constraint=constraint, **kwargs)
        finally:
            self.closed += 1


def fixture(scripts):
    tok = ProtocolTable()
    svc = server.Service(ScopedFixture(tok, scripts), tok, ChatTemplate(ROOT / 'serve/chat_template.jinja'))
    svc.experimental_responses = True
    svc.responses_replay = ReplayCodec(Fernet.generate_key())
    return svc


def payload(**kwargs):
    return request(grammar='root ::= "OK"', reasoning={'effort': 'low'}, max_output_tokens=2048, **kwargs)


def stream_final(test, base, body):
    code, headers, raw = http(base, {**body, 'stream': True})
    test.assertEqual(code, 200, raw)
    test.assertEqual(headers['Content-Type'], 'text/event-stream')
    events, _ = sse_events(raw)
    test.assertEqual([e['sequence_number'] for e in events], list(range(len(events))))
    test.assertEqual(sum(e['type'] in ('response.completed', 'response.failed', 'response.incomplete') for e in events), 1)
    final = events[-1]['response']
    for i, item in enumerate(final['output']):
        if item['type'] == 'function_call':
            deltas = [e for e in events if e['type'] == 'response.function_call_arguments.delta' and e['output_index'] == i]
            test.assertTrue(all(e['item_id'] == item['id'] for e in deltas))
            test.assertEqual(''.join(e['delta'] for e in deltas), item['arguments'])
        elif item['type'] == 'message':
            deltas = [e for e in events if e['type'] == 'response.output_text.delta' and e['output_index'] == i]
            test.assertEqual(''.join(e['delta'] for e in deltas), item['content'][0]['text'])
    return final


class ToolLoop(unittest.TestCase):
    def test_chat_tool_call_and_answer_views(self):
        call = [('tool', function_script('purple 999'))]
        svc = fixture([call, call, [('answer', 'OK')], [('answer', 'OK')]])
        definition = function_tool()
        definition.pop('type')
        body = {'model': MODEL, 'messages': [{'role': 'user', 'content': 'Use echo'}],
                'tools': [{'type': 'function', 'function': definition}], 'reasoning_effort': 'none',
                'max_tokens': 128, 'grammar': 'root ::= "OK"'}
        with listening(svc) as base:
            code, _, raw = http(base, body, '/v1/chat/completions')
            self.assertEqual(code, 200, raw)
            first = json.loads(raw)['choices'][0]
            self.assertEqual(first['finish_reason'], 'tool_calls')
            code, _, raw = http(base, {**body, 'stream': True}, '/v1/chat/completions')
            self.assertEqual(code, 200, raw)
            chunks = [json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith('data: {')]
            arguments = ''.join(call.get('function', {}).get('arguments', '') for c in chunks
                                for call in c['choices'][0]['delta'].get('tool_calls', []))
            self.assertEqual(arguments, first['message']['tool_calls'][0]['function']['arguments'])
            body['messages'] += [first['message'], {'role': 'tool',
                'tool_call_id': first['message']['tool_calls'][0]['id'], 'content': 'arbitrary result 猫 999'}]
            code, _, raw = http(base, body, '/v1/chat/completions')
            self.assertEqual(code, 200, raw)
            self.assertEqual(json.loads(raw)['choices'][0]['message']['content'], 'OK')
            code, _, raw = http(base, {**body, 'stream': True}, '/v1/chat/completions')
            self.assertEqual(code, 200, raw)
            chunks = [json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith('data: {')]
            self.assertEqual(''.join(c['choices'][0]['delta'].get('content', '') for c in chunks), 'OK')
            definition['strict'] = True
            before = svc.engine.index
            code, headers, _ = http(base, {**body, 'stream': True}, '/v1/chat/completions')
            self.assertEqual(code, 400)
            self.assertNotIn('text/event-stream', headers['Content-Type'])
            self.assertEqual(svc.engine.index, before)
            for malformed in (True, 42):
                code, headers, _ = http(base, {**body, 'tools': malformed, 'stream': True}, '/v1/chat/completions')
                self.assertEqual(code, 400)
                self.assertNotIn('text/event-stream', headers['Content-Type'])
                self.assertEqual(svc.engine.index, before)

    def test_parallel_namespaced_calls_mock_results_and_final_json_sse(self):
        tools = [function_tool(), {'type': 'namespace', 'name': 'files', 'description': 'Files', 'tools': [function_tool()]}]
        call = [('reasoning', 'Need both tools; thinking is not OK.'), ('control', '</think>\n\n'),
                ('tool', function_script('purple 999')), ('control', '\n'),
                ('tool', function_script('quote " and 猫', 'files.echo'))]
        answer = [('reasoning', 'Tool results are input, not grammar text.'), ('control', '</think>\n\n'), ('answer', 'OK')]
        svc = fixture([call, call, answer, answer])
        first = payload(input=[{'role': 'user', 'content': 'Use both echo tools.'}], tools=tools)
        original = copy.deepcopy(first)
        with listening(svc) as base:
            code, _, raw = http(base, first)
            self.assertEqual(code, 200, raw)
            final = json.loads(raw)
            self.assertEqual(normalized(stream_final(self, base, first)), normalized(final))
            calls = [x for x in final['output'] if x['type'] == 'function_call']
            self.assertEqual([(x.get('namespace'), x['name']) for x in calls], [(None, 'echo'), ('files', 'echo')])
            self.assertEqual(json.loads(calls[0]['arguments']), {'text': 'purple 999'})
            self.assertNotEqual(calls[0]['id'], calls[0]['call_id'])
            self.assertEqual(final['status'], 'completed')
            self.assertFalse(any(x['type'] == 'message' for x in final['output']))
            result_text = 'arbitrary tool data: 999 <tool_call> literal; not valid in root ::= "OK"; 猫'
            history = first['input'] + final['output'] + [
                {'type': 'function_call_output', 'call_id': c['call_id'], 'output': result_text} for c in reversed(calls)]
            follow = {**first, 'input': history}
            code, _, raw = http(base, follow)
            self.assertEqual(code, 200, raw)
            end = json.loads(raw)
            self.assertEqual(normalized(stream_final(self, base, follow)), normalized(end))
            # The native qualification uses this documented client decision
            # after its tool budget. It must disable new calls, not erase input.
            code, _, raw = http(base, {**follow, 'tool_choice': 'none'})
            self.assertEqual(code, 200, raw)
            self.assertEqual(json.loads(raw)['output'][-1]['content'][0]['text'], 'OK')
        self.assertEqual(end['status'], 'completed')
        self.assertEqual(end['output'][-1]['content'][0]['text'], 'OK')
        self.assertGreater(end['usage']['output_tokens_details']['reasoning_tokens'], 0)
        self.assertIn(result_text, svc.engine.inputs[-1])
        self.assertEqual(first, original)
        self.assertTrue(all(c.scoped and c.thinking and c.tools for c in svc.engine.constraints[:4]))
        self.assertTrue(svc.engine.constraints[-1].thinking)
        self.assertFalse(svc.engine.constraints[-1].tools)
        self.assertEqual(svc.engine.closed, 5)
        self.assertFalse(svc.status['busy'])

    def test_tools_without_thinking_and_literal_answer_bytes(self):
        answer = '\n<think>literal</think>\n猫'
        call = [('tool', function_script('arbitrary 999'))]
        svc = fixture([call, [('answer', answer)]])
        body = payload(tools=[function_tool()])
        body['reasoning'] = {'effort': 'none'}
        body['grammar'] = 'root ::= "\\n<think>literal</think>\\n猫"'
        with listening(svc) as base:
            first = stream_final(self, base, body)
            follow = {**body, 'input': [{'role': 'user', 'content': 'Do it'}, *first['output'],
                       {'type': 'function_call_output', 'call_id': first['output'][0]['call_id'], 'output': 'purple'}]}
            final = stream_final(self, base, follow)
        self.assertEqual(final['output'][0]['content'][0]['text'], answer)
        self.assertTrue(all(not c.thinking and c.tools for c in svc.engine.constraints))

    def test_summary_is_separate_from_answer_grammar(self):
        svc = fixture([[('reasoning', 'Reason about a number'), ('control', '</think>\n\n'), ('answer', 'OK')],
                       [('answer', 'A summary can contain any words.')]])
        body = payload()
        body['reasoning']['summary'] = 'concise'
        with listening(svc) as base:
            final = stream_final(self, base, body)
        self.assertEqual(final['status'], 'completed')
        self.assertEqual(final['output'][0]['summary'][0]['text'], 'A summary can contain any words.')
        self.assertEqual(final['output'][-1]['content'][0]['text'], 'OK')
        self.assertIsNone(svc.engine.constraints[-1])

    def test_budget_is_incomplete_and_disconnect_releases_owner(self):
        svc = fixture([[('tool', function_script('long value'))], [('answer', 'OK')]])
        body = payload(tools=[function_tool()])
        body['reasoning'] = {'effort': 'none'}
        body['max_output_tokens'] = 3
        with listening(svc) as base:
            final = stream_final(self, base, body)
        self.assertEqual(final['status'], 'incomplete')
        self.assertEqual(final['incomplete_details']['reason'], 'max_output_tokens')
        self.assertFalse(any(x['type'] == 'message' for x in final['output']))
        self.assertFalse(svc.status['busy'])
        body['max_output_tokens'] = 128
        p = create_response(svc, body)
        cancel = threading.Event()
        events = execute_response(svc, p, cancel)
        for e in events:
            if e and e['type'] == 'response.output_text.delta':
                break
        events.close()
        self.assertTrue(cancel.is_set())
        self.assertFalse(svc.status['busy'])
        self.assertEqual(svc.engine.closed, 2)

    def test_scope_requires_new_native_capability_before_writing(self):
        engine, wire = test_grammar.NativePipe().pipe('gbnf-v2')
        with self.assertRaisesRegex(ValueError, 'gbnf-v3'):
            engine.validate_constraint(GrammarConstraint('root ::= "OK"', thinking=True))
        self.assertEqual(wire.getvalue(), b'')

    def test_scoped_native_frame_and_channels(self):
        constraint = GrammarConstraint('root ::= "OK"', thinking=True, tools=True)
        engine, wire = test_grammar.NativePipe().pipe(lines=['TG reasoning 120\n', 'TG control 262\n',
                                               'TG answer 79\n', 'TG answer 75\n', 'DONE 4 2 0 1 length\n'])
        tokens = list(engine.generate([1, 2], 8, {}, threading.Event(), constraint=constraint))
        self.assertEqual([t.channel for t in tokens], ['reasoning', 'control', 'answer', 'answer'])
        self.assertEqual(wire.getvalue(), constraint.frame('GEN 8 1,2'))
        self.assertTrue(wire.getvalue().startswith(b'GENG2 13 1 1\n'))

    def test_configured_thinking_wrap_rejected_before_headers(self):
        svc = fixture([[('answer', 'OK')]])
        svc.reasoning_budget_tokens = 32
        with listening(svc) as base:
            code, headers, _ = http(base, {**payload(), 'stream': True})
        self.assertEqual(code, 400)
        self.assertNotIn('text/event-stream', headers['Content-Type'])
        self.assertEqual(svc.engine.index, 0)


if __name__ == '__main__':
    unittest.main()
