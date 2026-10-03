"""Synthetic protocol tests; native matcher/GPU receipts prove enforcement separately."""
import builtins
import copy
import json
import threading
import unittest
from unittest import mock

from serve.grammar import GrammarConstraint
from serve.responses import create_response, execute_response, RequestError
from serve.responses_json import prepare_json_output
from serve.test_grammar import fixture
from serve import test_grammar
from serve.test_grammar_scope import fixture as scoped_fixture
from serve.test_responses import http, listening, normalized, request, sse_events, function_tool, function_script

TITLE = {"type": "json_schema", "name": "codex_output_schema", "strict": True,
         "schema": {"type": "object", "properties": {"title": {"type": "string", "minLength": 1, "maxLength": 36}},
                    "required": ["title"], "additionalProperties": False}}


def body(fmt=TITLE, **extra):
    return request(text={"format": copy.deepcopy(fmt)}, **{"max_output_tokens": 256, **extra})


class JsonResponses(unittest.TestCase):
    def test_json_and_sse_are_exactly_the_same_text(self):
        answer = '{ "title" : "Fix addition" }'
        svc = fixture(answer)
        with listening(svc) as base:
            code, _, raw = http(base, body())
            self.assertEqual(code, 200, raw)
            final = json.loads(raw)
            self.assertEqual(final["status"], "completed")
            self.assertEqual(final["text"]["format"], TITLE)
            self.assertEqual(final["output"][0]["content"][0]["text"], answer)
            code, _, raw = http(base, body(stream=True))
            self.assertEqual(code, 200, raw)
            events, _ = sse_events(raw)
            self.assertEqual(normalized(events[-1]["response"]), normalized(final))
            self.assertEqual(''.join(e['delta'] for e in events if e['type']=='response.output_text.delta'), answer)
        self.assertTrue(all(c.json_schema for c in svc.engine.constraints))

    def test_no_invalid_json_or_invalid_schema_result_is_completed(self):
        for answer in ('not JSON', '{"title":""}', '{"title":3}', '{"title":"x","extra":1}',
                       '{"title":"' + 'x'*37 + '"}', '{"title":"x","title":"y"}', '{"title":NaN}'):
            with self.subTest(answer=answer):
                svc = fixture(answer)
                p = create_response(svc, body())
                events = list(execute_response(svc, p, threading.Event()))
                self.assertEqual(events[-1]['type'], 'response.failed')
                self.assertFalse(any(e['type']=='response.completed' for e in events))
                self.assertEqual(p.assembler.snapshot()['status'], 'failed')

    def test_schema_validation_is_mandatory_and_preflighted(self):
        original = builtins.__import__
        def missing(name, *args, **kwargs):
            if name == 'jsonschema': raise ImportError('fixture missing dependency')
            return original(name, *args, **kwargs)
        with mock.patch('builtins.__import__', side_effect=missing):
            with self.assertRaisesRegex(RequestError, 'no weaker validation'):
                create_response(fixture(), body())
        for schema in ({'type':'bad-type'}, {'$ref':'https://example.invalid/schema'}, {'maxLength':-1}):
            svc=fixture()
            with mock.patch.object(svc, 'load') as load, self.assertRaises(RequestError):
                create_response(svc, body({**TITLE, 'schema':schema}))
            load.assert_not_called()
        with self.assertRaises(RequestError):
            create_response(fixture(), body(grammar='root ::= "x"'))

    def test_object_mode_and_full_final_schema_validation(self):
        fmt = prepare_json_output({'type':'json_object'})
        fmt.validate('{"nested":[true,null,2.5,"?"]}')
        for value in ('[]','1','null'):
            with self.assertRaises(ValueError):fmt.validate(value)
        fmt = prepare_json_output({**TITLE, 'schema':{'type':'object','properties':{'n':{'type':'number','multipleOf':0.5}},'required':['n']}})
        fmt.validate('{"n":1.5}')
        with self.assertRaises(ValueError):fmt.validate('{"n":1.3}')

    def test_schema_properties_are_data_and_local_refs_work(self):
        schema={'type':'object','$defs':{'count':{'type':'integer','minimum':1}},
                'properties':{'$ref':{'type':'string'},'count':{'$ref':'#/$defs/count'}},
                'required':['$ref','count'],'additionalProperties':False}
        fmt=prepare_json_output({**TITLE,'schema':schema})
        fmt.validate('{"$ref":"literal property","count":2}')
        with self.assertRaises(ValueError):fmt.validate('{"$ref":"literal property","count":0}')
        with self.assertRaisesRegex(RequestError,'unknown JSON Schema keyword'):
            prepare_json_output({**TITLE,'schema':{'type':'string','minLenght':2}})

    def test_incomplete_json_remains_incomplete(self):
        svc=fixture('{"title":"not finished"}')
        p=create_response(svc, body(max_output_tokens=5))
        events=list(execute_response(svc,p,threading.Event()))
        self.assertEqual(events[-1]['type'],'response.incomplete')

    def test_reasoning_tools_and_native_budget_share_the_json_path(self):
        svc=scoped_fixture([[('reasoning','Need a tool.'),('tool',function_script('arbitrary text'))],
                            [('reasoning','Now answer.'),('answer','{"title":"Done"}')]])
        svc.reasoning_budget_tokens=2
        p=create_response(svc,body(reasoning={'effort':'medium'},tools=[function_tool()]))
        events=list(execute_response(svc,p,threading.Event()))
        self.assertEqual(events[-1]['type'],'response.completed')
        self.assertEqual(p.constraint.reasoning_tokens,2)
        self.assertTrue(p.constraint.json_schema)
        result=p.assembler.snapshot()
        call=next(x for x in result['output'] if x['type']=='function_call')
        history=[{'role':'user','content':'Start'},*result['output'],
                 {'type':'function_call_output','call_id':call['call_id'],'output':'not JSON; external tool data'}]
        p=create_response(svc,body(input=history,reasoning={'effort':'medium'},tools=[function_tool()]))
        events=list(execute_response(svc,p,threading.Event()))
        self.assertEqual(events[-1]['type'],'response.completed')

    def test_frame_and_old_engine_cannot_claim_json(self):
        c=GrammarConstraint('{"type":"object"}', True, False, True, 32)
        self.assertTrue(c.frame('CHECKG').startswith(b'GENG3 17 1 1 0 32\n'))
        engine,wire=test_grammar.NativePipe().pipe('gbnf-v3')
        with self.assertRaisesRegex(ValueError,'gbnf-v4'):engine.validate_constraint(c)
        self.assertEqual(wire.getvalue(),b'')


if __name__=='__main__':unittest.main()
