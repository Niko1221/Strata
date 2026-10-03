"""Local provider boundary tests against a real, fake HTTP upstream (no GPU)."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from urllib.parse import quote
from unittest import mock

from serve.frontend import ChatTemplate
from serve.providers import ProviderError, ProviderManager, dispatch, estimate_tokens, finish_native, local_base_url
from serve.server import ByteTokenizer, MockEngine, Service, make_handler

ROOT = Path(__file__).resolve().parents[1]
MODEL = "bonsai2:custom-q8_0"


class Upstream(BaseHTTPRequestHandler):
    mode = "normal"
    last = None
    releases = []
    release_pending = 0
    released = False
    started = threading.Event()
    release = threading.Event()

    def log_message(self, *args):
        pass

    def send_json(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            if self.release_pending:
                type(self).release_pending -= 1
            unloaded = self.released and not self.release_pending and self.mode != "release_never"
            self.send_json(200, {"object": "list", "data": [{"id": MODEL,
                "status": {"value": "unloaded" if unloaded else "loaded"}}, {"id": "qwen-custom"}]})
        elif self.path == "/redirect/models":
            self.send_response(302)
            self.send_header("Location", "http://example.com/v1/models")
            self.end_headers()
        else:
            self.send_json(404, {"error": {"message": "missing"}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        if self.path in ("/api/generate", "/models/unload"):
            type(self).releases.append((self.path, body))
            if self.mode != "release_error":
                type(self).released = True
                if self.path == "/models/unload":
                    type(self).release_pending = 2
            self.send_json(503 if self.mode == "release_error" else 200, {"done": True, "success": True})
            return
        type(self).last = body
        type(self).started.set()
        if self.mode == "wait_before_headers":
            self.release.wait(5)
            return
        if self.mode == "error":
            self.send_json(400, {"error": {"message": "private-upstream-detail"}})
            return
        if self.mode == "hold":
            self.release.wait(5)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {"choices": [{"delta": {"reasoning": "thinking"}}]},
                {"choices": [{"delta": {"content": "hello"}}]},
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-1", "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"}}]}}]},
            ]
            if self.mode != "unfinished":
                chunks.append({"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                               "usage": {"prompt_tokens": 10, "completion_tokens": 2}})
            for chunk in chunks:
                try:
                    self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                    if self.mode == "stalled_stream":
                        self.wfile.flush()
                        self.release.wait(5)
                except OSError:
                    return
            if self.mode != "unfinished":
                self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            answer = "memory" if body.get("max_tokens") == 512 else "hello"
            self.send_json(200, {"model": MODEL, "choices": [{"message": {"role": "assistant", "content": answer},
                                                           "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": 10, "completion_tokens": 2}})


class Providers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        cls.upstream.daemon_threads = True
        threading.Thread(target=cls.upstream.serve_forever, daemon=True).start()
        cls.upbase = f"http://127.0.0.1:{cls.upstream.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.upstream.shutdown()
        cls.upstream.server_close()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "providers.json"
        Upstream.mode = "normal"
        Upstream.last = None
        Upstream.releases = []
        Upstream.release_pending = 0
        Upstream.released = False
        Upstream.started.clear()
        Upstream.release.clear()
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "native", 4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.providers = ProviderManager(self.path)
        svc = self.svc

        class Handler(make_handler(svc)):
            def do_GET(inner):
                path = inner.path.split("?")[0].rstrip("/")
                if not inner._authorized():
                    return
                if dispatch(inner, svc, path, "GET"):
                    return
                super().do_GET()

            def do_POST(inner):
                path = inner.path.split("?")[0].rstrip("/")
                if not inner._authorized():
                    return
                if dispatch(inner, svc, path, "POST"):
                    return
                try:
                    super().do_POST()
                finally:
                    finish_native(inner, svc)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def tearDown(self):
        Upstream.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def call(self, path, body=None, headers=None, raw=False):
        request = urllib.request.Request(self.base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = self.opener.open(request, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read()
            return response.code, data if raw else json.loads(data)

    def register(self, **extra):
        code, result = self.call("/api/providers", {"name": "Custom Bonsai", "base_url": self.upbase,
            "model": MODEL, "context": 4096, "images": False, **extra})
        self.assertEqual(code, 200, result)
        return result["provider"]["id"]

    def activate(self):
        identifier = self.register()
        code, result = self.call("/api/providers/select", {"id": identifier})
        self.assertEqual(code, 200, result)
        return identifier

    def test_url_boundary(self):
        self.assertEqual(local_base_url("http://localhost:11434/v1/"), "http://127.0.0.1:11434/v1")
        self.assertEqual(local_base_url("http://[::1]:1234/v1"), "http://[::1]:1234/v1")
        for url in ("https://127.0.0.1/v1", "http://example.com/v1", "http://127.1/v1", "http://127.0.0.2/v1",
                    "http://user:secret@127.0.0.1/v1", "http://127.0.0.1/v1?token=secret", "http://127.0.0.1/v1#x",
                    "http://127.0.0.1:65536/v1", "http://127.0.0.1/%2e%2e/v1"):
            with self.subTest(url=url), self.assertRaises(ProviderError):
                local_base_url(url)

    def test_discovery_and_redirect_never_leaves_loopback(self):
        code, result = self.call("/api/provider-models?base_url=" + quote(self.upbase, safe=""))
        self.assertEqual(code, 200)
        self.assertEqual(result["models"][0]["id"], MODEL)
        redirect = self.upbase.replace("/v1", "/redirect")
        code, result = self.call("/api/provider-models?base_url=" + quote(redirect, safe=""))
        self.assertEqual(code, 502)

    def test_registration_persists_custom_model_without_ready_claim_after_restart(self):
        identifier = self.register()
        restored = ProviderManager(self.path)
        catalog = restored.catalog()
        self.assertIsNone(catalog["current"])
        self.assertEqual(catalog["providers"][0]["id"], identifier)
        self.assertEqual(catalog["providers"][0]["model"], MODEL)
        self.assertTrue(catalog["providers"][0]["prepared"])
        self.assertFalse(catalog["providers"][0]["ready"])

    def test_missing_model_not_saved_and_failed_probe_keeps_selection(self):
        identifier = self.activate()
        code, result = self.call("/api/providers", {"name": "Missing", "base_url": self.upbase,
            "model": "not-installed", "context": 4096})
        self.assertEqual(code, 404)
        self.assertEqual(self.svc.providers.current, identifier)
        self.assertEqual(len(self.svc.providers.profiles), 1)

    def test_tools_forward_and_engine_options_stripped(self):
        self.activate()
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
        messages = [{"role": "user", "content": "hello"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call-x", "type": "function",
                "function": {"name": "lookup", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call-x", "content": "result"}]
        code, result = self.call("/v1/chat/completions", {"model": MODEL, "messages": messages, "tools": tools,
            "strata_mcp": True, "experimental_speed_projection": True, "reasoning_effort": "medium", "top_k": 20})
        self.assertEqual(code, 200, result)
        self.assertEqual(Upstream.last["model"], MODEL)
        self.assertEqual(Upstream.last["messages"], messages)
        self.assertEqual(Upstream.last["tools"], tools)
        self.assertEqual(Upstream.last["reasoning_effort"], "medium")
        for key in ("strata_mcp", "experimental_speed_projection", "top_k"):
            self.assertNotIn(key, Upstream.last)
        code, _ = self.call("/v1/chat/completions", {"model": "wrong", "messages": messages})
        self.assertEqual(code, 404)

    def test_stable_strata_aliases_route_to_selected_backend(self):
        self.activate()
        messages = [{"role": "user", "content": "hello"}]
        for alias in ("active", "strata"):
            with self.subTest(alias=alias):
                code, result = self.call("/v1/chat/completions", {"model": alias, "messages": messages})
                self.assertEqual(code, 200, result)
                self.assertEqual(result["model"], MODEL)
                self.assertEqual(Upstream.last["model"], MODEL)
                code, result = self.call("/v1/chat/completions/count_tokens", {
                    "model": alias, "messages": messages,
                })
                self.assertEqual(code, 200, result)
                self.assertEqual(result["input_tokens"], estimate_tokens(messages))
                code, result = self.call("/v1/chat/compact", {
                    "model": alias, "messages": messages, "previous_summary": "",
                })
                self.assertEqual(code, 200, result)
                self.assertEqual(result["summary"], "memory")
                self.assertEqual(Upstream.last["model"], MODEL)

    def test_unknown_external_models_still_rejected_on_every_route(self):
        self.activate()
        for path in ("/v1/chat/completions", "/v1/chat/completions/count_tokens", "/v1/chat/compact"):
            with self.subTest(path=path):
                Upstream.last = None
                code, result = self.call(path, {
                    "model": "not-selected", "messages": [{"role": "user", "content": "hello"}],
                })
                self.assertEqual(code, 404, result)
                self.assertIsNone(Upstream.last)

    def test_stream_preserves_tools_usage_and_normalizes_reasoning(self):
        self.activate()
        code, data = self.call("/v1/chat/completions", {"model": MODEL,
            "messages": [{"role": "user", "content": "hi"}], "stream": True}, raw=True)
        self.assertEqual(code, 200)
        text = data.decode()
        self.assertIn('"reasoning_content": "thinking"', text)
        self.assertIn('"tool_calls"', text)
        self.assertIn('"usage"', text)
        self.assertIn("[DONE]", text)
        self.assertNotIn('"error"', text)

    def test_unfinished_stream_emits_error(self):
        self.activate()
        Upstream.mode = "unfinished"
        code, data = self.call("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
            "stream": True}, raw=True)
        self.assertEqual(code, 200)
        self.assertIn(b'"error"', data)
        self.assertNotIn(b"[DONE]", data)
        self.assertFalse(self.svc.status["busy"])

    def test_upstream_error_is_redacted_and_fifo_released(self):
        self.activate()
        Upstream.mode = "error"
        code, result = self.call("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(code, 502)
        self.assertNotIn("private-upstream-detail", json.dumps(result))
        self.assertFalse(self.svc.status["busy"])
        self.assertEqual(self.svc.status["queued"], 0)
        self.assertTrue(self.svc.fifo.acquire(blocking=False))
        self.svc.fifo.release()

    def test_switch_rejects_native_busy_and_external_active_request(self):
        identifier = self.register()
        with self.svc.status_lock:
            self.svc.status["busy"] = True
        code, _ = self.call("/api/providers/select", {"id": identifier})
        self.assertEqual(code, 409)
        with self.svc.status_lock:
            self.svc.status["busy"] = False
        self.call("/api/providers/select", {"id": identifier})
        Upstream.mode = "hold"
        results = []
        request = threading.Thread(target=lambda: results.append(self.call("/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}]})))
        request.start()
        self.assertTrue(Upstream.started.wait(3))
        code, _ = self.call("/api/providers/select", {"id": None})
        self.assertEqual(code, 409)
        self.assertEqual(self.svc.providers.current, identifier)
        Upstream.release.set()
        request.join(5)
        self.assertEqual(results[0][0], 200)
        self.assertEqual(self.call("/api/providers/select", {"id": None})[0], 200)

    def test_count_is_estimated_and_compaction_uses_same_provider(self):
        self.activate()
        messages = [{"role": "user", "content": "日本語と custom model"}]
        code, result = self.call("/v1/chat/completions/count_tokens", {"model": MODEL, "messages": messages})
        self.assertEqual(code, 200)
        self.assertTrue(result["estimated"])
        self.assertEqual(result["input_tokens"], estimate_tokens(messages))
        code, result = self.call("/v1/chat/compact", {"messages": messages, "previous_summary": ""})
        self.assertEqual(code, 200, result)
        self.assertEqual(result["summary"], "memory")
        self.assertEqual(Upstream.last["model"], MODEL)
        self.assertNotIn("tools", Upstream.last)

    def test_origin_guard_and_auth_apply_to_provider_management(self):
        code, _ = self.call("/api/providers", {"name": "X", "base_url": self.upbase, "model": MODEL},
                            headers={"Origin": "http://foreign.example"})
        self.assertEqual(code, 403)
        self.assertFalse(self.path.exists())
        self.svc.api_key = "local-test-key"
        code, _ = self.call("/api/providers")
        self.assertEqual(code, 401)
        code, _ = self.call("/api/providers", headers={"Authorization": "Bearer local-test-key"})
        self.assertEqual(code, 200)

    def test_backend_release_before_native_selection_and_failure_retains_current(self):
        for backend, path in (("ollama", "/api/generate"), ("llamacpp", "/models/unload")):
            with self.subTest(backend=backend):
                Upstream.released = False
                Upstream.release_pending = 0
                identifier = self.register(backend=backend)
                self.assertEqual(self.call("/api/providers/select", {"id": identifier})[0], 200)
                Upstream.mode = "release_error"
                self.assertEqual(self.call("/api/providers/select", {"id": None})[0], 503)
                self.assertEqual(self.svc.providers.current, identifier)
                Upstream.mode = "normal"
                self.assertEqual(self.call("/api/providers/select", {"id": None})[0], 200)
                self.assertEqual(Upstream.releases[-1][0], path)
                self.assertEqual(Upstream.releases[-1][1]["model"], MODEL)
                if backend == "ollama":
                    self.assertEqual(Upstream.releases[-1][1]["keep_alive"], 0)

    def test_llama_already_unloaded_can_return_to_native_without_unload_post(self):
        identifier = self.register(backend="llamacpp")
        self.assertEqual(self.call("/api/providers/select", {"id": identifier})[0], 200)
        Upstream.released = True
        Upstream.mode = "release_error"
        self.assertEqual(self.call("/api/providers/select", {"id": None})[0], 200)
        self.assertIsNone(self.svc.providers.current)
        self.assertEqual(Upstream.releases, [])

    def test_native_admission_prevents_dispatch_selection_race(self):
        identifier = self.register()

        class Request:
            provider_native_admitted = False
        handler = Request()
        self.assertFalse(dispatch(handler, self.svc, "/v1/chat/completions", "POST"))
        self.assertEqual(self.svc.providers.native_inflight, 1)
        with self.assertRaises(ProviderError) as caught:
            self.svc.providers.select(identifier, self.svc)
        self.assertEqual(caught.exception.code, 409)
        finish_native(handler, self.svc)
        self.assertEqual(self.svc.providers.native_inflight, 0)
        self.svc.providers.select(identifier, self.svc)

    def test_native_load_and_switch_controls_reserve_admission(self):
        identifier = self.register()
        for path in ("/load", "/v1/load", "/api/local-models/switch"):
            with self.subTest(path=path):
                class Request:
                    provider_native_admitted = False
                handler = Request()
                self.assertFalse(dispatch(handler, self.svc, path, "POST"))
                self.assertEqual(self.svc.providers.native_inflight, 1)
                with self.assertRaises(ProviderError) as caught:
                    self.svc.providers.select(identifier, self.svc)
                self.assertEqual(caught.exception.code, 409)
                finish_native(handler, self.svc)
                self.assertEqual(self.svc.providers.native_inflight, 0)

    def test_native_switch_worker_pending_blocks_provider_selection(self):
        identifier = self.register()

        class Switcher:
            lock = threading.Lock()
            state = "starting"
            def _state(self, svc):
                return {"status": self.state}
        self.svc.model_switcher = Switcher()
        for pending in ("starting", "restoring"):
            self.svc.model_switcher.state = pending
            with self.assertRaises(ProviderError) as caught:
                self.svc.providers.select(identifier, self.svc)
            self.assertEqual(caught.exception.code, 409)
            self.assertIsNone(self.svc.providers.current)
        self.svc.model_switcher.state = "ready"
        self.svc.providers.select(identifier, self.svc)

    def test_active_provider_blocks_native_switch_api(self):
        identifier = self.activate()
        code, _ = self.call("/api/local-models/switch", {"model": "original"})
        self.assertEqual(code, 409)
        self.assertEqual(self.svc.providers.current, identifier)

    def test_off_reasoning_maps_to_none(self):
        self.activate()
        code, _ = self.call("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                           "reasoning_effort": "off"})
        self.assertEqual(code, 200)
        self.assertEqual(Upstream.last["reasoning_effort"], "none")

    def test_profile_reasoning_mapping_is_explicit_and_persisted(self):
        identifier = self.register(reasoning_map={"high": "xhigh"})
        restored = ProviderManager(self.path)
        self.assertEqual(restored.profiles[identifier]["reasoning_map"], {"high": "xhigh"})
        self.assertEqual(self.call("/api/providers/select", {"id": identifier})[0], 200)
        messages = [{"role": "user", "content": "hi"}]
        self.assertEqual(self.call("/v1/chat/completions", {"messages": messages,
                        "reasoning_effort": "high"})[0], 200)
        self.assertEqual(Upstream.last["reasoning_effort"], "xhigh")
        self.assertEqual(self.call("/v1/chat/completions", {"messages": messages,
                        "reasoning_effort": "off"})[0], 200)
        self.assertEqual(Upstream.last["reasoning_effort"], "none")
        self.assertEqual(self.call("/api/providers/select", {"id": None})[0], 200)
        self.register(reasoning_map={})
        self.assertEqual(self.call("/api/providers/select", {"id": identifier})[0], 200)
        self.assertEqual(self.call("/v1/chat/completions", {"messages": messages,
                        "reasoning_effort": "high"})[0], 200)
        self.assertEqual(Upstream.last["reasoning_effort"], "high")

    def test_reasoning_mapping_rejects_non_identifier_values(self):
        for mapping in ({"custom": "xhigh"}, {"high": "xhigh\nother"}, {"high": 5},
                        {"high": "x" * 33}, ["high", "xhigh"]):
            with self.subTest(mapping=mapping):
                code, _ = self.call("/api/providers", {"name": "Test", "base_url": self.upbase,
                    "model": MODEL, "reasoning_map": mapping})
                self.assertEqual(code, 400)

    def test_client_disconnect_cancels_stalled_stream_and_releases_fifo(self):
        self.activate()
        Upstream.mode = "stalled_stream"
        request = urllib.request.Request(self.base + "/v1/chat/completions",
            data=json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True}).encode(),
            headers={"Content-Type": "application/json"})
        response = self.opener.open(request, timeout=5)
        self.assertTrue(response.readline().startswith(b"data:"))
        response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
        response.close()
        deadline = time.time() + 3
        while self.svc.status["busy"] and time.time() < deadline:
            time.sleep(0.05)
        self.assertFalse(self.svc.status["busy"])
        self.assertTrue(self.svc.fifo.acquire(blocking=False))
        self.svc.fifo.release()

    def test_client_disconnect_before_upstream_headers_releases_fifo(self):
        self.activate()
        Upstream.mode = "wait_before_headers"
        client = socket.create_connection(("127.0.0.1", self.server.server_port), timeout=5)
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True}).encode()
        request = (f"POST /v1/chat/completions HTTP/1.0\r\nHost: 127.0.0.1:{self.server.server_port}\r\n"
                   f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body
        client.sendall(request)
        self.assertTrue(Upstream.started.wait(3))
        client.shutdown(socket.SHUT_RDWR)
        client.close()
        deadline = time.time() + 3
        while self.svc.status["busy"] and time.time() < deadline:
            time.sleep(0.05)
        self.assertFalse(self.svc.status["busy"])
        self.assertEqual(self.svc.status["queued"], 0)
        self.assertTrue(self.svc.fifo.acquire(blocking=False))
        self.svc.fifo.release()
        self.assertFalse(Upstream.release.is_set())  # Upstream is still withholding headers.

    def test_llama_release_waits_for_unloaded_and_timeout_retains_selection(self):
        identifier = self.register(backend="llamacpp")
        self.assertEqual(self.call("/api/providers/select", {"id": identifier})[0], 200)
        Upstream.mode = "release_never"
        with mock.patch("serve.providers.RELEASE_WAIT_S", 0.1):
            self.assertEqual(self.call("/api/providers/select", {"id": None})[0], 503)
        self.assertEqual(self.svc.providers.current, identifier)
        Upstream.mode = "normal"
        self.assertEqual(self.call("/api/providers/select", {"id": None})[0], 200)
        self.assertEqual(Upstream.release_pending, 0)


if __name__ == "__main__":
    unittest.main()
