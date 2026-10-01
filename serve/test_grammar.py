"""Grammar invariants, native pipe integration and OpenAI SDK compatibility."""
import io
import json
import queue
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate, OutputParser
from serve.grammar import GrammarDecoder
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve
from serve.structured import StructuredOutputError, prepare_format

SCHEMA = {"type": "object", "properties": {"voiceover": {"type": "boolean"}},
          "required": ["voiceover"], "additionalProperties": False}
FORMAT = {"type": "json_schema", "json_schema": {"name": "dialogue", "strict": True, "schema": SCHEMA}}


class Grammar(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.stops = (257, 258)

    def decoder(self, schema=SCHEMA, thinking=False):
        return GrammarDecoder(self.tok, self.stops, schema, thinking)

    def test_cannot_close_object_without_required_voiceover(self):
        dec = self.decoder()
        dec.accept(ord('{'))
        mask = dec.mask(262)
        self.assertFalse(mask[ord('}') // 8] & (1 << (ord('}') % 8)))
        self.assertFalse(mask[257 // 8] & (1 << (257 % 8)))
        self.assertTrue(dec.matcher.consume_tokens(self.tok.encode('"voiceover":false}')))
        self.assertTrue(dec.matcher.is_accepting())
        self.assertTrue(dec.mask(262)[257 // 8] & (1 << (257 % 8)))

    def test_reasoning_must_be_followed_by_schema(self):
        class ThinkingTokenizer(ByteTokenizer):
            SPECIALS=ByteTokenizer.SPECIALS+['</think>']
        tok=ThinkingTokenizer()
        dec = GrammarDecoder(tok,self.stops,SCHEMA,True)
        self.assertTrue(dec.matcher.consume_tokens(tok.encode('Consider it.</think>\n',parse_special=True)))
        self.assertFalse(dec.mask(263)[ord('x') // 8] & (1 << (ord('x') % 8)))
        self.assertTrue(dec.matcher.consume_tokens(tok.encode('{"voiceover":true}')))
        self.assertTrue(dec.matcher.is_accepting())

    def test_constraints_and_request_isolation(self):
        schema={"type":"object","properties":{"x":{"type":"integer","minimum":2,"maximum":4},
                "name":{"enum":["bela","babi"]}},"required":["x","name"],"additionalProperties":False}
        for text in ('{"x":1,"name":"bela"}', '{"x":3,"name":"other"}', '{"x":3,"name":"babi","extra":1}'):
            self.assertFalse(self.decoder(schema).matcher.consume_tokens(self.tok.encode(text)))
        self.assertTrue(self.decoder(schema).matcher.consume_tokens(self.tok.encode('{"x":3,"name":"babi"}')))
        first, second = self.decoder(), self.decoder()
        first.accept(ord('{'))
        self.assertNotEqual(first.mask(262), second.mask(262))
        with self.assertRaises(StructuredOutputError):
            first.mask(263)

    def test_unsupported_strict_schemas_are_request_errors(self):
        for schema in ({"type":"object"}, {**SCHEMA,"required":[]}, {**SCHEMA,"allOf":[SCHEMA]},
                       {**SCHEMA,"anyOf":[SCHEMA]}):
            with self.assertRaises(ValueError):
                prepare_format({**FORMAT,"json_schema":{**FORMAT['json_schema'],"schema":schema}},[])

    def test_pipe_mask_precedes_each_sample_and_no_capability_fallback(self):
        engine=StrataEngine.__new__(StrataEngine)
        engine.can_stop=True
        engine.can_grammar=True
        engine.lines=queue.Queue()
        engine.proc=mock.Mock()
        engine.proc.stdin=io.StringIO()
        for token in self.tok.encode('{"voiceover":false}')+[257]:
            engine.lines.put('MASK 262\n')
            engine.lines.put(f'T {token}\n')
        engine.lines.put('DONE 20 1 0 1 stop\n')
        dec=self.decoder()
        ids=list(engine.generate([256],64,{'_grammar':dec},threading.Event()))
        self.assertEqual(ids[-1],257)
        self.assertTrue(dec.matcher.is_accepting())
        lines=engine.proc.stdin.getvalue().splitlines()
        self.assertIn('grammar=1',lines[0])
        self.assertEqual(len(lines)-1,len(ids))
        self.assertTrue(all(len(line)==5+36*2 for line in lines[1:]))
        engine.can_grammar=False
        with self.assertRaises(ValueError):
            list(engine.generate([256],64,{'_grammar':self.decoder()},threading.Event()))

    def test_structured_strings_preserve_tool_tags(self):
        for thinking in (False,True):
            parser=OutputParser(thinking=thinking)
            parser.literal_content=True
            text='{"text":"<tool_call>literal</tool_call> and </think>"}'
            events=parser.feed(('Reason.</think>\n' if thinking else '')+text)+parser.finish()
            content=''.join(e.text for e in events if e.kind=='content')
            self.assertEqual(json.loads(content),json.loads(text))
            self.assertFalse(any(e.kind.startswith('tool') for e in events))

    def test_literal_control_tags_cannot_be_sampled_as_special_tokens(self):
        class LiteralTokenizer(ByteTokenizer):
            SPECIALS=ByteTokenizer.SPECIALS+['<tool_call>','</tool_call>','</think>']
        tok=LiteralTokenizer()
        text='<tool_call>literal</tool_call> and </think> with é'
        schema={'type':'object','properties':{'text':{'type':'string','const':text}},
                'required':['text'],'additionalProperties':False}
        dec=GrammarDecoder(tok,self.stops,schema)
        for token in tok.encode(json.dumps({'text':text},ensure_ascii=False,separators=(',',':'))):
            mask=dec.mask(265)
            for special in (262,263,264):
                self.assertFalse(mask[special//8] & (1 << (special%8)))
            self.assertTrue(mask[token//8] & (1 << (token%8)))
            dec.accept(token)
        self.assertTrue(dec.matcher.is_accepting())

    def test_official_sdk_parse_stream_and_token_limit(self):
        try:
            from openai import OpenAI, LengthFinishReasonError
            from pydantic import BaseModel
        except ImportError:
            self.skipTest('optional OpenAI SDK compatibility checks require openai')
        class Dialogue(BaseModel):
            voiceover: bool
        engine=MockEngine(self.tok,'{"voiceover":false}',max_context=4096)
        svc=Service(engine,self.tok,ChatTemplate(Path(__file__).parent/'chat_template.jinja'))
        httpd=serve(svc,port=0)
        try:
            client=OpenAI(base_url=f'http://127.0.0.1:{httpd.server_address[1]}/v1',api_key='local',max_retries=0)
            kw=dict(model=svc.model,messages=[{'role':'user','content':'Return a dialogue JSON object.'}],
                    reasoning_effort='none',response_format=Dialogue)
            parsed=client.chat.completions.parse(**kw)
            self.assertIs(parsed.choices[0].message.parsed.voiceover,False)
            with client.chat.completions.stream(**kw) as stream:
                deltas=[e for e in stream if e.type=='content.delta']
                self.assertGreater(len(deltas),1)
                final=stream.get_final_completion()
                self.assertIs(final.choices[0].message.parsed.voiceover,False)
            with self.assertRaises(LengthFinishReasonError):
                client.chat.completions.parse(**kw,max_completion_tokens=4)
            with self.assertRaises(LengthFinishReasonError):
                with client.chat.completions.stream(**kw,max_completion_tokens=4) as stream:
                    list(stream)
            client.close()
        finally:
            httpd.shutdown(); httpd.server_close()

if __name__=='__main__': unittest.main()
