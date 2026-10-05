"""Optional deepMoE protocol and API integration tests; no GPU or checkpoint needed."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import Mock, patch

from serve.deepmoe import DeepMoEEngine, DeepMoETemplate, backend_from_config
from serve.server import ByteTokenizer, EngineDied, Service, serve

FAKE = r'''#!/usr/bin/env python3
import json,sys,threading,time
lock=threading.Lock()
def emit(x):
 with lock:print(json.dumps(x),flush=True)
emit({'event':'ready','max_context':4096})
worker=None;cancel=threading.Event()
def run(req,stop):
 emit({'event':'prefill','done':len(req['prompt_ids']),'total':len(req['prompt_ids'])})
 text='Reason</think>Answer' if req['prompt_ids'][-7:]==list(b'<think>') else 'Answer'
 for t in list(text.encode())+[1]:
  if stop.is_set():break
  emit({'event':'token','id':t});time.sleep(.002)
 emit({'event':'done','generated':len(text)+1,'reused_tokens':0,'decode_ms':40,'prefill_ms':2})
for line in sys.stdin:
 req=json.loads(line)
 if req['op']=='generate':
  if worker:worker.join()
  cancel=threading.Event();worker=threading.Thread(target=run,args=(req,cancel));worker.start()
 elif req['op']=='cancel':cancel.set();emit({'event':'cancel'})
 elif req['op']=='quit':
  cancel.set()
  if worker:worker.join()
  break
'''


class Protocol(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.exe = Path(self.tmp.name) / 'fake'
        self.exe.write_text(FAKE)
        self.exe.chmod(0o755)
        self.engine = DeepMoEEngine(str(self.exe), [], model='unused', timeout=2, error_type=EngineDied)

    def tearDown(self):
        self.engine.close()
        self.tmp.cleanup()

    def ids(self, cancel=None):
        return [x for x in self.engine.generate([65], 30, {'temperature':0,'seed':7}, cancel or threading.Event())
                if x is not None]

    def test_handshake_and_sequential_turns(self):
        self.assertEqual(self.engine.max_context,4096)
        self.assertEqual(self.ids(),list(b'Answer')+[1])
        self.assertEqual(self.ids(),list(b'Answer')+[1])
        self.assertEqual(self.engine.last['decode_ms'],40)
        self.assertIsNone(self.engine.progress)

    def test_iterator_close_drains_before_next_turn(self):
        gen=self.engine.generate([65],30,{},threading.Event())
        self.assertEqual(next(gen),ord('A'))
        gen.close()
        self.assertEqual(self.ids(),list(b'Answer')+[1])

    def test_cancel_event_stops_turn_and_next_turn_recovers(self):
        cancel=threading.Event();gen=self.engine.generate([65],30,{},cancel)
        self.assertEqual(next(gen),ord('A'));cancel.set()
        self.assertEqual([x for x in gen if x is not None],[])
        self.assertEqual(self.ids(),list(b'Answer')+[1])

    def test_pre_cancelled_request_never_reaches_engine(self):
        cancel=threading.Event();cancel.set()
        self.assertEqual(self.ids(cancel),[])
        self.assertIsNone(self.engine.last)

    def test_dead_engine_reports_and_can_restart(self):
        self.engine.proc.kill();self.engine.proc.wait()
        with self.assertRaises(EngineDied):self.ids()
        self.engine.restart()
        self.assertEqual(self.ids(),list(b'Answer')+[1])

    def test_sampling_options_are_not_silently_ignored(self):
        for key,value in [('top_k',20),('min_p',.1),('repetition_penalty',1.1)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                list(self.engine.generate([65],5,{key:value},threading.Event()))
        self.assertIsNone(self.engine.last)

    def test_openai_and_anthropic_use_native_eos_and_reasoning(self):
        template=Mock()
        template.render.side_effect=lambda messages,**kw: 'q<think>' if kw.get('enable_thinking',True) else 'q</think>'
        svc=Service(self.engine,ByteTokenizer(),template,model_name='deepseek-v4.1-flash')
        self.assertEqual(svc.stop_ids,{1})
        httpd=serve(svc,port=0)
        try:
            base=f'http://127.0.0.1:{httpd.server_address[1]}'
            def post(path,body):
                req=urllib.request.Request(base+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=5) as r:return r.read().decode()
            msgs=[{'role':'user','content':'q'}]
            b=json.loads(post('/v1/chat/completions',{'model':'m','messages':msgs,'max_tokens':40,'top_k':0}))
            self.assertEqual(b['choices'][0]['message']['content'],'Answer')
            self.assertEqual(b['choices'][0]['message']['reasoning_content'],'Reason')
            self.assertEqual(b['choices'][0]['finish_reason'],'stop')
            stream=post('/v1/chat/completions',{'model':'m','messages':msgs,'max_tokens':40,'stream':True})
            self.assertIn('reasoning_content',stream);self.assertIn('data: [DONE]',stream)
            b=json.loads(post('/v1/messages',{'model':'m','messages':msgs,'max_tokens':40,'thinking':{'type':'disabled'}}))
            self.assertEqual(b['content'][0]['text'],'Answer')
        finally:
            httpd.shutdown();httpd.server_close()


class Template(unittest.TestCase):
    def setUp(self):
        self.tpl=object.__new__(DeepMoETemplate)
        self.tpl.encoding=Mock();self.tpl.encoding.encode_messages.return_value='native'

    def test_effort_maps_to_native_numeric_budget(self):
        for effort,n in [('low',50),('medium',75),('xhigh',100)]:
            self.assertEqual(self.tpl.render([{'role':'user','content':'q'}],reasoning_effort=effort),'native')
            self.assertEqual(self.tpl.encoding.encode_messages.call_args.kwargs['reasoning_effort'],n)
        self.tpl.render([],enable_thinking=False)
        self.assertEqual(self.tpl.encoding.encode_messages.call_args.kwargs['thinking_mode'],'chat')

    def test_unsupported_modalities_fail_before_inference(self):
        for messages,tools in [([], [{'name':'f'}]),([{'role':'tool','content':'x'}],None),
                               ([{'role':'user','content':[{'type':'image'}]}],None)]:
            with self.assertRaises(ValueError):self.tpl.render(messages,tools)
        self.tpl.encoding.encode_messages.assert_not_called()

    def test_invalid_config_does_not_start_engine(self):
        with patch('serve.deepmoe.DeepMoEEngine') as engine:
            for cfg in ({},{'exe':'x'},{'exe':'x','model':'x','parallel':2}):
                with self.assertRaises(ValueError):backend_from_config(cfg)
            engine.assert_not_called()


if __name__=='__main__':unittest.main()
