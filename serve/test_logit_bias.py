import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import urllib.error
import urllib.request

from serve.logit_bias import engine_key, normalize, validate_request
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve
from serve.frontend import ChatTemplate


class LogitBias(unittest.TestCase):
    def test_forms_and_transport(self):
        self.assertEqual(normalize({"7": -100, "9": 2.5}, 10), normalize([[7, False], [9, 2.5]], 10))
        self.assertEqual(engine_key([[9, 2.5], [7, False]]), " logit_bias=7:-100,9:2.5")
        self.assertEqual(StrataEngine.sampling_keys({"logit_bias": {"7": -100}}), " logit_bias=7:-100")
        for value in [None, {}, []]:
            self.assertEqual(engine_key(value), "")

    def test_invalid(self):
        bad = [True, 1, "x", [1], [[1]], [[1, 0, 2]], {"-1": 0}, {"1.0": 0}, {"10": 0},
               {"1": False}, [[1, True]], [[1.0, 0]], [[True, 0]], [[1, 0], ["1", 2]],
               {"1": 101}, {"1": -101}, {"1": float('nan')}, {"1": float('inf')}, {"1": 10**1000}]
        for value in bad:
            with self.subTest(value=str(value)[:50]), self.assertRaises(ValueError):
                normalize(value, 10)
        with self.assertRaises(ValueError):
            normalize({str(i): -100 for i in range(10)}, 10)

    def test_large_list(self):
        entries = [[i, False] for i in range(103215)]
        self.assertEqual(len(normalize(entries, 248320)), 103215)
        self.assertEqual(engine_key(entries).count(':-100'), 103215)

    def test_capability_and_batch(self):
        for engine in [SimpleNamespace(info={}), SimpleNamespace(info={'logit_bias': 1}, batch=2)]:
            with self.assertRaises(ValueError):
                validate_request({'logit_bias': {'1': -100}}, engine, 10)
            self.assertEqual(validate_request({}, engine, 10), {})
        self.assertEqual(validate_request({'logit_bias': {'1': -100}}, SimpleNamespace(info={'logit_bias': 1}), 10), {1: -100})

    def test_http_rejects_before_stream(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, 'ok', max_context=4096)
        service = Service(engine, tok, ChatTemplate(Path(__file__).parent/'chat_template.jinja'))
        httpd = serve(service, port=0)
        try:
            for bias in [True, {'1': -100}, [[1, True]]]:
                data = json.dumps({'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 2,
                                   'stream': True, 'logit_bias': bias}).encode()
                req = urllib.request.Request(f'http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions',
                                             data=data, headers={'Content-Type': 'application/json'})
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(req, timeout=10)
                with caught.exception as response:
                    self.assertEqual(response.code, 400)
                    self.assertIn('logit_bias', json.loads(response.read())['error']['message'])
        finally:
            httpd.shutdown()
            httpd.server_close()
