"""0.1.41 server hardening: a JSON body of the wrong shape is a 400 (not a dropped connection), an oversized body is a
413 before it is read, a bad Content-Length is a 400.

    python -m unittest serve.test_request_hardening -v
"""
import http.client
import json
import os
import unittest
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, body_limit, serve

ROOT = Path(__file__).resolve().parent.parent


class Hardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, "hello there", max_context=4096), tok,
                          ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        try:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            c.request("POST", path, body=data, headers={"Content-Type": "application/json", **(headers or {})})
            r = c.getresponse()
            return r.status, json.loads(r.read() or b"{}")
        finally:
            c.close()

    def test_wrong_shapes_are_400(self):
        for path, body in (("/v1/chat/completions", {"messages": 5}),
                           ("/v1/chat/completions", {"messages": [7]}),
                           ("/v1/chat/completions", {"messages": [{"role": "user", "content": 5}]}),
                           ("/v1/messages", {"messages": [None], "max_tokens": 4}),
                           ("/v1/messages", {"messages": "hi", "max_tokens": 4}),
                           ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "tools": 3})):
            with mock.patch("builtins.print"):
                code, out = self.post(path, body)
            self.assertEqual(code, 400, (path, body, out))
            self.assertIn("error", out)

    def test_not_json_is_400(self):
        with mock.patch("builtins.print"):
            code, out = self.post("/v1/chat/completions", b"{nope")
        self.assertEqual(code, 400, out)

    def test_a_good_request_still_answers(self):
        code, out = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
        self.assertEqual(code, 200, out)

    def test_oversized_content_length_is_413_unread(self):
        with mock.patch.dict(os.environ, {"STRATA_MAX_BODY_MIB": "1"}):
            self.assertEqual(body_limit(), 1 << 20)
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
            c.putrequest("POST", "/v1/chat/completions")
            c.putheader("Content-Type", "application/json")
            c.putheader("Content-Length", str(10 << 40))          # 10 TiB announced, nothing sent
            c.endheaders()
            r = c.getresponse()
            self.assertEqual(r.status, 413)
            self.assertIn("STRATA_MAX_BODY_MIB", json.loads(r.read())["error"]["message"])
            c.close()

    def test_negative_or_bad_content_length_is_400(self):
        for value in ("-5", "abc"):
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
            c.putrequest("POST", "/v1/chat/completions")
            c.putheader("Content-Type", "application/json")
            c.putheader("Content-Length", value)
            c.endheaders()
            r = c.getresponse()
            self.assertEqual(r.status, 400, value)
            c.close()

    def test_default_limit_is_generous(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATA_MAX_BODY_MIB", None)
            self.assertEqual(body_limit(), 256 << 20)
        with mock.patch.dict(os.environ, {"STRATA_MAX_BODY_MIB": "junk"}):
            self.assertEqual(body_limit(), 256 << 20)


if __name__ == "__main__":
    unittest.main()
