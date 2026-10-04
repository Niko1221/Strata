"""Browser template budgeting without model load, tool execution or image fetch."""
import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, CTX_SLACK, Service, serve
from serve.test_server import UnloadableEngine

ROOT = Path(__file__).resolve().parents[1]


class ChatContextHttp(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = UnloadableEngine(self.tok, "</think>\n\nok", max_context=4096)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body, headers=None):
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_count_matches_real_prompt_and_does_not_load(self):
        self.engine.unload()
        request = {"model": self.svc.model, "messages": [{"role": "user", "content": "how many?"}],
                   "reasoning_effort": "none", "max_tokens": 16}
        status, count = self.post("/v1/chat/count_tokens", request)
        self.assertEqual(status, 200)
        self.assertTrue(count["count_exact"])
        self.assertEqual(count["max_context"], 4096)
        self.assertEqual(count["context_slack"], CTX_SLACK)
        self.assertEqual(count["effective_max_tokens"], 16)
        self.assertFalse(self.engine.alive())
        self.assertEqual(self.engine.starts, 0)
        status, completion = self.post("/v1/chat/completions", request)
        self.assertEqual(status, 200)
        self.assertEqual(count["input_tokens"], completion["usage"]["prompt_tokens"])

    def test_shared_thinking_cap_and_explicit_override_match(self):
        self.svc.shared = {"max_tokens": 256, "reasoning_effort": "high"}
        request = {"messages": [{"role": "user", "content": "hello"}]}
        count = self.post("/v1/chat/count_tokens", request)[1]
        self.assertEqual(count["effective_max_tokens"], 256)
        completed = self.post("/v1/chat/completions", request)[1]
        self.assertEqual(count["input_tokens"], completed["usage"]["prompt_tokens"])
        explicit = self.post("/v1/chat/count_tokens", {**request, "max_completion_tokens": 32})[1]
        self.assertEqual(explicit["effective_max_tokens"], 32)

    def test_count_mcp_template_without_running_tools(self):
        tool = {"name": "search", "description": "Search the web", "parameters": {"type": "object"}}
        hub = mock.Mock()
        hub.template_tools.return_value = [tool]
        self.svc.mcp = hub
        request = {"messages": [{"role": "user", "content": "hello"}], "strata_mcp": True}
        status, count = self.post("/v1/chat/count_tokens", request)
        self.assertEqual(status, 200)
        hub.wait.assert_called_once_with(10)
        hub.template_tools.assert_called_once_with(exclude=set())
        hub.run.assert_not_called()
        ids, _, _ = self.svc.prepare(request["messages"], [tool], {}, 16)
        self.assertEqual(count["input_tokens"], len(ids))

    def test_count_authorization_and_foreign_origin_preserved(self):
        request = {"messages": [{"role": "user", "content": "hello"}]}
        self.svc.api_key = "fixture-key"
        self.assertEqual(self.post("/v1/chat/count_tokens", request)[0], 401)
        self.assertEqual(self.post("/v1/chat/count_tokens", request,
                                  {"Authorization": "Bearer fixture-key"})[0], 200)
        # Existing authenticated API clients may use foreign origins; an unkeyed browser page may not.
        self.assertEqual(self.post("/v1/chat/count_tokens", request,
                                  {"Authorization": "Bearer fixture-key", "Origin": "https://foreign.example"})[0], 200)
        self.svc.api_key = ""
        self.assertEqual(self.post("/v1/chat/count_tokens", request,
                                  {"Origin": "https://foreign.example"})[0], 403)

    def test_images_are_reserved_without_start_or_fetch(self):
        self.svc.vision = SimpleNamespace(spawn=(["--max-tokens", "768"], None, None))
        request = {"messages": [{"role": "user", "content": [{"type": "text", "text": "see"},
                   {"type": "image_url", "image_url": {"url": "https://example.invalid/image.png"}}]}]}
        with mock.patch("serve.server.urllib.request.urlopen", side_effect=AssertionError("image fetch")):
            # Call the normal HTTP helper outside the patched urllib client.
            from serve.frontend import openai_to_messages
            messages, tools, kw = openai_to_messages(request)
            self.assertEqual(len(self.svc.tok.encode(self.svc.template.render(messages, tools=tools, **kw),
                                                   parse_special=True)) > 0, True)
        status, count = self.post("/v1/chat/count_tokens", request)
        self.assertEqual(status, 200)
        self.assertFalse(count["count_exact"])
        self.assertGreaterEqual(count["input_tokens"], 768)

if __name__ == "__main__":
    unittest.main()
