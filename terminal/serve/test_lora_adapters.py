"""serve/test_lora_adapters.py - unit tests for GET/POST /lora-adapters and StrataEngine.lora_reload.

    python -m unittest terminal/serve/test_lora_adapters.py -v
"""
from __future__ import annotations

import io
import json
import queue
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (  # noqa: E402
    ByteTokenizer,
    MockEngine,
    Service,
    StrataEngine,
    format_lora_spec,
    lora_spec_from_args,
    serve,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeLoraEngine(MockEngine):
    def __init__(self, tokenizer, *, layers: int = 48, hot_reload: bool = True,
                 lora_path: str = "/models/a.gguf", lora_scale: float = 1.0):
        super().__init__(tokenizer, "</think>\n\nok", max_context=4096)
        self.info = {"lora": layers, "lora_reload": 1 if hot_reload else 0}
        self.lora_path = lora_path
        self.lora_scale = lora_scale
        self.lora_generation = 1 if layers > 0 else 0
        self.lora_digest = "0000000000000001" if layers > 0 else None
        self.reload_calls: list[str] = []
        self.reload_error: Exception | None = None
        self.last_sampling: dict = {}

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_sampling = dict(sampling or {})
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)

    def lora_reload(self, spec: str) -> dict:
        self.reload_calls.append(spec)
        if self.reload_error is not None:
            raise self.reload_error
        self.lora_generation += 1
        digest = f"{self.lora_generation:016x}"
        self.lora_digest = digest
        return {"generation": self.lora_generation, "digest": digest, "ms": 12.5, "dropped": 2}


class _HTTPMixin:
    base: str

    def request(self, method: str, path: str, body=None, headers: dict | None = None):
        hdrs = dict(headers or {})
        data = None
        if body is not None:
            if isinstance(body, (bytes, bytearray)):
                data = bytes(body)
            else:
                data = json.dumps(body).encode("utf-8")
                hdrs.setdefault("Content-Type", "application/json")
        req = urllib.request.Request(self.base + path, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())


class LoraSpecHelpersTest(unittest.TestCase):
    def test_lora_spec_from_args(self):
        self.assertEqual(lora_spec_from_args(["--native", "m.gguf"]), ("", 1.0))
        self.assertEqual(lora_spec_from_args(["--lora", "/tmp/a.gguf"]), ("/tmp/a.gguf", 1.0))
        self.assertEqual(lora_spec_from_args(["--lora-scaled", "/tmp/a.gguf:0.75"]), ("/tmp/a.gguf", 0.75))
        self.assertEqual(
            lora_spec_from_args(["--lora-scaled", "/tmp/a.gguf:0.5,/tmp/b.gguf:0.25"]),
            ("/tmp/a.gguf:0.5,/tmp/b.gguf:0.25", 1.0),
        )
        self.assertEqual(
            lora_spec_from_args(["--lora", r"C:\models\a.gguf"]),
            (r"C:\models\a.gguf", 1.0),
        )

    def test_format_lora_spec(self):
        self.assertEqual(format_lora_spec("/tmp/a.gguf"), ("/tmp/a.gguf", 1.0, "/tmp/a.gguf"))
        self.assertEqual(format_lora_spec("/tmp/a.gguf", 0.5), ("/tmp/a.gguf", 0.5, "/tmp/a.gguf:0.5"))
        self.assertEqual(format_lora_spec("/tmp/a.gguf:0.75"), ("/tmp/a.gguf", 0.75, "/tmp/a.gguf:0.75"))
        self.assertEqual(format_lora_spec("/tmp/a.gguf:0.75", 0.25), ("/tmp/a.gguf", 0.25, "/tmp/a.gguf:0.25"))
        self.assertEqual(
            format_lora_spec(r"C:\dir\a.gguf", 0.8),
            (r"C:\dir\a.gguf", 0.8, r"C:\dir\a.gguf:0.8"),
        )
        with self.assertRaises(ValueError):
            format_lora_spec("")
        with self.assertRaises(ValueError):
            format_lora_spec("/tmp/a.gguf\nQUIT")
        with self.assertRaises(ValueError):
            format_lora_spec("/tmp/a.gguf", True)
        with self.assertRaises(ValueError):
            format_lora_spec("/tmp/a.gguf", float("nan"))


class StrataEngineWireProtocolTest(unittest.TestCase):
    def _make_engine(self, responses: list[str]):
        eng = object.__new__(StrataEngine)
        eng.batch = 0
        eng.ended = False
        eng.lines = queue.Queue()
        for line in responses:
            eng.lines.put(line)
        stdin_buf = io.StringIO()

        class FakeProc:
            def __init__(self):
                self.stdin = stdin_buf

            def poll(self):
                return None

        eng.proc = FakeProc()
        eng.lora_generation = 1
        eng.lora_digest = None
        return eng, stdin_buf

    def test_ok_reply_parsed(self):
        eng, stdin_buf = self._make_engine(["OK lora_reload 2 deadbeef01234567 14.3 dropped=3\n"])
        res = eng.lora_reload("/tmp/new.gguf:0.8")
        self.assertEqual(stdin_buf.getvalue(), "LORA_RELOAD /tmp/new.gguf:0.8\n")
        self.assertEqual(res, {"generation": 2, "digest": "deadbeef01234567", "ms": 14.3, "dropped": 3})
        self.assertEqual(eng.lora_generation, 2)
        self.assertEqual(eng.lora_digest, "deadbeef01234567")

    def test_err_reply_raises_value_error(self):
        eng, _ = self._make_engine(["ERR lora_reload: layer 3 rank 64 exceeds allocated rank 32\n"])
        with self.assertRaisesRegex(ValueError, "exceeds allocated rank 32"):
            eng.lora_reload("/tmp/too_big.gguf")


class LoraAdaptersEndpointTest(_HTTPMixin, unittest.TestCase):
    def _start(self, engine, *, api_key: str = ""):
        tok = ByteTokenizer()
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.api_key = api_key
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.base = f"http://127.0.0.1:{httpd.server_address[1]}"
        return svc

    def test_get_none_loaded(self):
        tok = ByteTokenizer()
        eng = FakeLoraEngine(tok, layers=0, hot_reload=False, lora_path="")
        self._start(eng)
        status, body = self.request("GET", "/lora-adapters")
        self.assertEqual(status, 200)
        self.assertEqual(body, [])

    def test_get_loaded(self):
        tok = ByteTokenizer()
        eng = FakeLoraEngine(tok, layers=48, hot_reload=True, lora_path="/models/v1.gguf", lora_scale=0.8)
        self._start(eng)
        status, body = self.request("GET", "/lora-adapters")
        self.assertEqual(status, 200)
        self.assertEqual(body, [{
            "id": 0,
            "path": "/models/v1.gguf",
            "scale": 0.8,
            "enabled": True,
            "generation": 1,
            "layers": 48,
            "hot_reload": True,
        }])

    def test_post_dict_hot_reload_success_and_error(self):
        tok = ByteTokenizer()
        eng = FakeLoraEngine(tok, layers=48, hot_reload=True, lora_path="/models/v1.gguf", lora_scale=1.0)
        self._start(eng)

        # Success
        status, body = self.request("POST", "/lora-adapters", {"path": "/models/v2.gguf", "scale": 0.75})
        self.assertEqual(status, 200)
        self.assertEqual(eng.reload_calls, ["/models/v2.gguf:0.75"])
        self.assertEqual(body["path"], "/models/v2.gguf")
        self.assertEqual(body["scale"], 0.75)
        self.assertEqual(body["generation"], 2)
        self.assertEqual(body["dropped"], 2)
        self.assertTrue(body["enabled"])

        # Subsequent GET reflects the new adapter state
        status, adapters = self.request("GET", "/lora-adapters")
        self.assertEqual(status, 200)
        self.assertEqual(adapters[0]["path"], "/models/v2.gguf")
        self.assertEqual(adapters[0]["scale"], 0.75)
        self.assertEqual(adapters[0]["generation"], 2)

        # Engine reload error -> 400 and previous state preserved
        eng.reload_error = ValueError("layer 0 rank 64 exceeds allocated rank 16")
        status, err_body = self.request("POST", "/lora-adapters", {"path": "/models/bad.gguf", "scale": 1.0})
        self.assertEqual(status, 400)
        self.assertIn("exceeds allocated rank", err_body["error"]["message"])
        _, adapters_after = self.request("GET", "/lora-adapters")
        self.assertEqual(adapters_after[0]["path"], "/models/v2.gguf")
        self.assertEqual(adapters_after[0]["generation"], 2)

        # Bad input -> 400
        status, _ = self.request("POST", "/lora-adapters", {"path": "", "scale": 1.0})
        self.assertEqual(status, 400)
        status, _ = self.request("POST", "/lora-adapters", {"path": "/models/v3.gguf", "scale": "high"})
        self.assertEqual(status, 400)

    def test_post_dict_busy_and_unsupported_return_409(self):
        tok = ByteTokenizer()
        # Unsupported (no LoRA loaded at startup)
        eng_none = FakeLoraEngine(tok, layers=0, hot_reload=False, lora_path="")
        self._start(eng_none)
        status, _ = self.request("POST", "/lora-adapters", {"path": "/models/v2.gguf", "scale": 1.0})
        self.assertEqual(status, 409)

        # Unsupported (LoRA loaded, but hot_reload disabled)
        eng_no_hr = FakeLoraEngine(tok, layers=48, hot_reload=False, lora_path="/models/v1.gguf")
        self._start(eng_no_hr)
        status, _ = self.request("POST", "/lora-adapters", {"path": "/models/v2.gguf", "scale": 1.0})
        self.assertEqual(status, 409)

        # Busy (FIFO held by an active request)
        eng_busy = FakeLoraEngine(tok, layers=48, hot_reload=True, lora_path="/models/v1.gguf")
        svc = self._start(eng_busy)
        svc.fifo.acquire()
        try:
            status, body = self.request("POST", "/lora-adapters", {"path": "/models/v2.gguf", "scale": 1.0})
            self.assertEqual(status, 409)
            self.assertEqual(body["error"]["type"], "model_busy")
            self.assertEqual(eng_busy.reload_calls, [])
        finally:
            svc.fifo.release()

        # Busy reported by engine batch slots
        eng_busy.reload_error = ValueError("cannot reload LoRA while batch slots are actively decoding")
        status, _ = self.request("POST", "/lora-adapters", {"path": "/models/v2.gguf", "scale": 1.0})
        self.assertEqual(status, 409)

    def test_post_llamacpp_array_toggle_and_rescale(self):
        tok = ByteTokenizer()
        eng = FakeLoraEngine(tok, layers=48, hot_reload=True, lora_path="/models/v1.gguf", lora_scale=1.0)
        self._start(eng)

        # Disable via scale == 0 (no LORA_RELOAD sent)
        status, body = self.request("POST", "/lora-adapters", [{"id": 0, "scale": 0.0}])
        self.assertEqual(status, 200)
        self.assertEqual(eng.reload_calls, [])
        self.assertFalse(body[0]["enabled"])
        self.assertEqual(body[0]["scale"], 0.0)

        # Next completion sees lora=False in sampling defaults
        c_status, _ = self.request(
            "POST",
            "/v1/chat/completions",
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
        )
        self.assertEqual(c_status, 200)
        self.assertIs(eng.last_sampling.get("lora"), False)

        # Re-enable with same scale == 1.0 (no LORA_RELOAD needed)
        status, body = self.request("POST", "/lora-adapters", [{"id": 0, "scale": 1.0}])
        self.assertEqual(status, 200)
        self.assertEqual(eng.reload_calls, [])
        self.assertTrue(body[0]["enabled"])
        self.assertEqual(body[0]["scale"], 1.0)

        # Rescale to 0.5 on single-file spec triggers hot-reload
        status, body = self.request("POST", "/lora-adapters", [{"id": 0, "scale": 0.5}])
        self.assertEqual(status, 200)
        self.assertEqual(eng.reload_calls, ["/models/v1.gguf:0.5"])
        self.assertTrue(body[0]["enabled"])
        self.assertEqual(body[0]["scale"], 0.5)
        self.assertEqual(body[0]["generation"], 2)

        # Unknown adapter id -> 400
        status, _ = self.request("POST", "/lora-adapters", [{"id": 1, "scale": 1.0}])
        self.assertEqual(status, 400)

    def test_auth_and_origin_guards(self):
        tok = ByteTokenizer()
        eng = FakeLoraEngine(tok, layers=48, hot_reload=True, lora_path="/models/v1.gguf")
        self._start(eng, api_key="topsecret")

        # Missing or wrong API key -> 401 on GET and POST
        status, _ = self.request("GET", "/lora-adapters")
        self.assertEqual(status, 401)
        status, _ = self.request("POST", "/lora-adapters", {"path": "/models/v2.gguf", "scale": 1.0})
        self.assertEqual(status, 401)
        status, _ = self.request(
            "POST",
            "/lora-adapters",
            {"path": "/models/v2.gguf", "scale": 1.0},
            headers={"Authorization": "Bearer wrong"},
        )
        self.assertEqual(status, 401)

        # Foreign browser Origin -> 403 even with valid API key
        status, _ = self.request(
            "POST",
            "/lora-adapters",
            {"path": "/models/v2.gguf", "scale": 1.0},
            headers={"Authorization": "Bearer topsecret", "Origin": "https://evil.example"},
        )
        self.assertEqual(status, 403)

        # Non-JSON Content-Type -> 415
        status, _ = self.request(
            "POST",
            "/lora-adapters",
            b'{"path": "/models/v2.gguf"}',
            headers={"Authorization": "Bearer topsecret", "Content-Type": "text/plain"},
        )
        self.assertEqual(status, 415)

        # Valid API key succeeds on both GET and POST
        auth = {"Authorization": "Bearer topsecret"}
        status, body = self.request("GET", "/lora-adapters", headers=auth)
        self.assertEqual(status, 200)
        self.assertEqual(len(body), 1)
        status, body = self.request(
            "POST",
            "/lora-adapters",
            {"path": "/models/v2.gguf", "scale": 1.0},
            headers=auth,
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["generation"], 2)


if __name__ == "__main__":
    unittest.main()
