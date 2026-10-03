"""Actual pinned Codex definitions with explicitly synthetic engine/tool results.

No shell, image, agent or goal tool is executed by these transport tests.
"""
import copy
import json
import threading
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator
from serve.responses import create_response, execute_response, native_tools, validate_request, RequestError
from serve.test_responses import ROOT, service, request, listening, http, normalized, sse_events, function_tool, function_script
from serve.responses_json import native_schema, prepare_json_output
from serve.frontend import ChatTemplate, OutputParser, parse_tool_call

TOOLS = json.loads((ROOT/'docs/codex/tool-declarations-0.160.0.json').read_text(encoding='utf-8'))
SAMPLES = json.loads((ROOT/'docs/codex/tool-argument-examples-0.160.0.json').read_text(encoding='utf-8'))


def script(name, arguments):
    pieces = ['<tool_call>\n<function='+name+'>']
    for key, value in arguments.items():
        pieces.append('\n<parameter='+key+'>\n'+(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False))+'\n</parameter>')
    return ''.join(pieces)+'\n</function>\n</tool_call>'


class CapturedToolSchemas(unittest.TestCase):
    def test_zero_parameter_native_view_has_no_hidden_example(self):
        original = copy.deepcopy(TOOLS)
        tool = next(native for native, _, _ in native_tools(TOOLS) if native['name'] == 'get_goal')
        self.assertEqual(TOOLS, original)
        call = '<tool_call>\n<function=get_goal>\n</function>\n</tool_call>'
        original_tool = next(t for t in TOOLS if t.get('name') == 'get_goal')
        self.assertEqual(tool['description'], original_tool['description'])
        self.assertNotIn(call, tool['description'])
        parser = OutputParser(thinking=False, tools=[tool], stream_tools=True)
        events = parser.feed(call) + parser.finish()
        self.assertEqual(''.join(e.text for e in events if e.kind == 'tool_args'), '{}')
        self.assertEqual(next(e.call.arguments for e in events if e.kind == 'tool_call'), {})
        open_tool = {**original_tool, 'parameters': {**original_tool['parameters'], 'additionalProperties': True}}
        native = next(native_tools([open_tool]))[0]
        self.assertNotIn(call, native.get('description', ''))

    def test_xml_boolean_reaches_strict_response_without_rewriting_stream(self):
        tool = function_tool(name='fixture', strict=True, parameters={'type': 'object',
            'properties': {'value': {'type': 'boolean'}}, 'required': ['value'], 'additionalProperties': False})
        output = '<tool_call>\n<function=fixture>\n<parameter=value>\nFalse\n</parameter>\n</function>\n</tool_call>'
        svc = service([output, output])
        with listening(svc) as base:
            code, _, raw = http(base, request(tools=[tool]))
            self.assertEqual(code, 200, raw)
            final = json.loads(raw)
            self.assertEqual(final['status'], 'completed')
            self.assertEqual(json.loads(final['output'][0]['arguments']), {'value': False})
            code, _, raw = http(base, request(tools=[tool], stream=True))
            self.assertEqual(code, 200, raw)
            events, _ = sse_events(raw)
            self.assertEqual(normalized(events[-1]['response']), normalized(final))
            self.assertEqual(''.join(e['delta'] for e in events if e['type'] == 'response.function_call_arguments.delta'),
                             final['output'][0]['arguments'])

    def test_native_boolean_spellings_have_one_canonical_argument_value(self):
        for declared, raw, expected in (('boolean', 'False', False), ('boolean', 'True', True),
                ('boolean', ' false ', False), ('string', 'False', 'False'), (None, 'False', 'False'),
                ('boolean', 'not_a_boolean', 'not_a_boolean')):
            with self.subTest(declared=declared, raw=raw):
                properties = {'value': {'type': declared}} if declared else {}
                tool = {'name': 'fixture', 'parameters': {'type': 'object', 'properties': properties}}
                xml = '<function=fixture>\n<parameter=value>\n'+raw+'\n</parameter>\n</function>'
                self.assertEqual(parse_tool_call(xml, tool).arguments, {'value': expected})
                wrapped = '<tool_call>\n'+xml+'\n</tool_call>'
                # Every byte split, including inside the spelling and closing tag.
                for split in range(len(wrapped)+1):
                    parser = OutputParser(thinking=False, tools=[tool], stream_tools=True)
                    events = parser.feed(wrapped[:split]) + parser.feed(wrapped[split:]) + parser.finish()
                    arguments = ''.join(e.text for e in events if e.kind == 'tool_args')
                    call = next(e.call for e in events if e.kind == 'tool_call')
                    self.assertEqual(json.loads(arguments), {'value': expected})
                    self.assertEqual(call.arguments, {'value': expected})

    def test_all_twelve_strict_normalized_variants(self):
        tools=copy.deepcopy(TOOLS)
        for group in tools:
            for function in group['tools'] if group['type']=='namespace' else [group]:
                function.pop('strict')
        normalized_tools=validate_request(request(tools=tools),service())['tools']
        def fill(schema, sample=None):
            if schema.get('type')=='object':
                sample=sample or {}
                return {key:fill(value,sample.get(key)) for key,value in schema.get('properties',{}).items()}
            if sample is not None:
                if schema.get('type')=='array':return [fill(schema.get('items',{}),item) for item in sample]
                return sample
            if 'enum' in schema:return schema['enum'][0]
            return {'string':'fixture','integer':1,'number':1.0,'boolean':False,'array':[]}.get(schema.get('type'))
        for native,namespace,name in native_tools(normalized_tools):
            with self.subTest(tool=native['name']):
                sample=fill(native['parameters'],SAMPLES[native['name']])
                Draft202012Validator(native['parameters']).validate(sample)
                svc=service(script(native['name'],sample))
                prepared=create_response(svc,request(tools=tools,max_output_tokens=4096))
                events=list(execute_response(svc,prepared,threading.Event()))
                final=events[-1]['response']
                self.assertEqual(final['status'],'completed',final.get('error'))
                self.assertEqual(json.loads(final['output'][0]['arguments']),sample)
                self.assertTrue(any(e['type']=='response.function_call_arguments.done' for e in events))

    def test_all_twelve_schemas_json_sse_and_client_result_replay(self):
        expected = list(native_tools(TOOLS))
        self.assertEqual(len(expected),12)
        self.assertEqual({native['name'] for native,_,_ in expected},set(SAMPLES))
        for native, namespace, name in expected:
            with self.subTest(tool=native['name']):
                sample=SAMPLES[native['name']]
                Draft202012Validator(native['parameters']).validate(sample)
                svc=service([script(native['name'],sample)]*2+['arbitrary tool data received'])
                with listening(svc) as base:
                    body=request(tools=TOOLS,max_output_tokens=4096)
                    original=copy.deepcopy(body)
                    code,_,raw=http(base,body)
                    self.assertEqual(code,200,raw)
                    final=json.loads(raw)
                    self.assertEqual(final['status'],'completed',final)
                    self.assertEqual(body,original)
                    call=final['output'][0]
                    self.assertEqual((call.get('namespace'),call['name']),(namespace,name))
                    self.assertNotEqual(call['id'],call['call_id'])
                    self.assertEqual(json.loads(call['arguments']),sample)
                    code,_,raw=http(base,{**body,'stream':True})
                    self.assertEqual(code,200,raw)
                    events,_=sse_events(raw)
                    self.assertEqual(normalized(events[-1]['response']),normalized(final))
                    self.assertEqual(''.join(e['delta'] for e in events if e['type']=='response.function_call_arguments.delta'),call['arguments'])
                    history=[{'role':'user','content':'Call the fixture.'},*final['output'],
                             {'type':'function_call_output','call_id':call['call_id'],'output':'arbitrary non-JSON text; <tool_call> is data'}]
                    code,_,raw=http(base,{**body,'input':history})
                    self.assertEqual(code,200,raw)
                    self.assertEqual(json.loads(raw)['status'],'completed')

    def test_strict_calls_validate_before_done_and_results_remain_unconstrained(self):
        parameters={'type':'object','properties':{'text':{'type':'string','enum':['allowed']}},
                    'required':['text'],'additionalProperties':False}
        tool=function_tool(strict=True,parameters=parameters)
        for namespace in (None,'files'):
            tools=[tool] if namespace is None else [{'type':'namespace','name':namespace,'description':'fixture','tools':[tool]}]
            name='echo' if namespace is None else namespace+'.echo'
            for value, status in (('allowed','completed'),('invalid','failed')):
                with self.subTest(namespace=namespace,value=value):
                    svc=service(function_script(value,name))
                    p=create_response(svc,request(tools=tools))
                    events=list(execute_response(svc,p,threading.Event()))
                    self.assertEqual(events[-1]['response']['status'],status)
                    if status=='failed':
                        self.assertFalse(any(e['type']=='response.function_call_arguments.done' for e in events))
                        self.assertFalse(any(e['type']=='response.output_item.done' and e['item']['status']=='completed' for e in events))
                    else:
                        call=p.assembler.snapshot()['output'][0]
                        continuation=service('ack')
                        p=create_response(continuation,request(tools=tools,input=[{'role':'user','content':'Begin'},call,
                            {'type':'function_call_output','call_id':call['call_id'],'output':'This result deliberately does not match the argument schema.'}]))
                        self.assertEqual(list(execute_response(continuation,p,threading.Event()))[-1]['response']['status'],'completed')

    def test_omitted_strict_normalization_and_explicit_open_fallback(self):
        tool=function_tool();del tool['strict']
        original=copy.deepcopy(tool)
        req=validate_request(request(tools=[tool]),service())
        self.assertEqual(tool,original)
        result=req['tools'][0]
        self.assertTrue(result['strict'])
        self.assertEqual(result['parameters']['required'],['text'])
        self.assertFalse(result['parameters']['additionalProperties'])
        tool['parameters']['additionalProperties']=True
        result=validate_request(request(tools=[tool]),service())['tools'][0]
        self.assertFalse(result['strict'])
        self.assertTrue(result['parameters']['additionalProperties'])
        for required in (3,{},[['bad']]):
            with self.assertRaises(RequestError):
                create_response(service(),request(tools=[{**tool,'strict':True,'parameters':{'type':'object','required':required}}]))

    def test_optional_false_property_native_simplification_keeps_original_validation(self):
        schema={'type':'object','properties':{'anything':True,'forbidden':False},'required':['anything'],'additionalProperties':False}
        original=copy.deepcopy(schema)
        output=prepare_json_output({'type':'json_schema','name':'boolean_child','strict':True,'schema':schema})
        self.assertEqual(native_schema(schema)['properties'],{'anything':True})
        self.assertEqual(schema,original)
        output.validate('{"anything":[1,"x",null]}')
        with self.assertRaises(ValueError):output.validate('{"anything":1,"forbidden":2}')


if __name__=='__main__':unittest.main()
