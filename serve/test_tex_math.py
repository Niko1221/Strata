"""Full TeX transport and endpoint failures without Docker or a GPU."""
import http.client
import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from serve import tex_math
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve


class Transport(unittest.TestCase):
    def test_disabled_and_invalid(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(tex_math.render({"source": "x", "display": False})["reason"], "unavailable")
        self.assertEqual(tex_math.render({"source": "x", "display": "yes"})["reason"], "invalid")
        self.assertEqual(tex_math.render({"source": "\ud800", "display": False})["reason"], "invalid")
        self.assertEqual(tex_math.render({"source": "α" * 9000, "display": True})["reason"], "limit")

    def test_source_never_enters_command_line(self):
        source = r"\input{/etc/passwd}; $(whoami)"
        with patch.dict(os.environ, {"STRATA_TEX_CONTAINER": "strata-math"}), patch.object(subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, b'{"ok":false,"reason":"invalid"}')
            self.assertFalse(tex_math.render({"source": source, "display": False})["ok"])
            self.assertNotIn(source, run.call_args.args[0])
            self.assertEqual(json.loads(run.call_args.kwargs["input"])["source"], source)

    def test_timeout_missing_and_bad_output(self):
        with patch.dict(os.environ, {"STRATA_TEX_CONTAINER": "strata-math"}), patch.object(subprocess, "run") as run:
            run.side_effect = subprocess.TimeoutExpired("docker", 7)
            self.assertEqual(tex_math.render({"source": "x", "display": False})["reason"], "limit")
            run.side_effect = FileNotFoundError()
            self.assertEqual(tex_math.render({"source": "x", "display": False})["reason"], "unavailable")
            run.side_effect = None
            run.return_value = subprocess.CompletedProcess([], 0, b'[]')
            self.assertEqual(tex_math.render({"source": "x", "display": False})["reason"], "unavailable")

    def test_busy(self):
        with patch.dict(os.environ, {"STRATA_TEX_CONTAINER": "strata-math"}):
            tex_math._lock.acquire()
            try:
                self.assertEqual(tex_math.render({"source": "x", "display": False})["reason"], "limit")
            finally:
                tex_math._lock.release()


class Endpoint(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(Path(__file__).with_name("chat_template.jinja")))
        self.server = serve(self.svc, port=0)

    def tearDown(self):
        self.server.shutdown(); self.server.server_close()

    def post(self, payload, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        conn.request("POST", "/web/math/render", payload, headers or {"Content-Type": "application/json"})
        response = conn.getresponse()
        status, body = response.status, json.loads(response.read())
        conn.close()
        return status, body

    def test_origin_content_type_and_authentication(self):
        with patch.object(tex_math, "render") as render:
            self.assertEqual(self.post(b'{}', {"Content-Type": "text/plain"})[0], 415)
            self.assertEqual(self.post(b'{}', {"Content-Type": "application/json", "Origin": "https://evil.example"})[0], 403)
            self.svc.api_key = "secret"
            self.assertEqual(self.post(b'{}')[0], 401)
            render.assert_not_called()

    def test_bad_input_and_limit(self):
        self.assertEqual(self.post(b'[]')[1]["reason"], "invalid")
        self.assertEqual(self.post(b'{')[1]["reason"], "invalid")
        self.assertEqual(self.post(b'{"source":"\\ud800","display":false}')[1]["reason"], "invalid")
        self.assertEqual(self.post(b'x' * 65537)[0], 413)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.post(b'{"source":"x","display":false}')[1]["reason"], "unavailable")


if __name__ == "__main__":
    unittest.main()
