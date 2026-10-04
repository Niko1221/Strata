"""Archive HTTP admission and UI transition over an isolated mock service."""
import json
import tempfile
import urllib.error
import urllib.request
import unittest
from pathlib import Path

from serve.chat_archive import ChatArchive
from serve.chat_memory import MemoryProvider
from serve.frontend import ChatTemplate
from serve.mcp import McpHub
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]


class ArchiveHttp(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "</think>\n\nok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.chat_archive = ChatArchive(Path(self.temp.name) / "archive.sqlite3")
        self.svc.mcp = McpHub({})
        self.svc.mcp.register_builtin(MemoryProvider(self.svc.chat_archive))
        self.server = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.svc.chat_archive.close()
        self.temp.cleanup()

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
            headers={**({"Content-Type": "application/json"} if body is not None else {}), **(headers or {})})
        try:
            reply = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            reply = error
        with reply:
            data = reply.read()
            return reply.status if hasattr(reply, "status") else reply.code, reply.headers, data

    def test_archive_requires_key_and_same_origin_even_with_wildcard_api_cors(self):
        self.svc.cors_origins = ["*"]
        self.svc.api_key = "fixture-key"
        self.assertEqual(self.request("/v1/chats")[0], 401)
        for headers in [{"Origin": "https://foreign.invalid"}, {"Sec-Fetch-Site": "cross-site"}]:
            status, response, _ = self.request("/v1/chats", headers={"Authorization": "Bearer fixture-key", **headers})
            self.assertEqual(status, 403)
            self.assertIsNone(response.get("Access-Control-Allow-Origin"))
        status, response, _ = self.request("/v1/chats", headers={"Authorization": "Bearer fixture-key", "Origin": self.base})
        self.assertEqual(status, 200)
        self.assertIsNone(response.get("Access-Control-Allow-Origin"))

    def test_import_atomic_reimport_and_active_archive_survive(self):
        session = {"id": "fixture", "title": "Review", "activeBranchId": "main", "branches": [{"id": "main", "title": "Main", "context": None,
            "messages": [{"role": "user", "text": "REVIEW-7421 exact path", "time": 1}]}]}
        self.assertEqual(self.request("/v1/chats/save", {"session": session, "activate": True})[0], 200)
        response = json.loads(self.request("/v1/chats/import", {"sessions": [session]})[2])
        self.assertEqual(response["skipped"], 1)
        status, _, data = self.request("/v1/chats")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["activeId"], "fixture")
        self.assertEqual(self.request("/v1/chats/import", {"sessions": [{"id": "broken"}]})[0], 400)
        self.assertEqual(len(json.loads(self.request("/v1/chats")[2])["sessions"]), 1)
        self.assertEqual(self.request("/v1/chats/save", {"session": session}, {"Origin": "https://foreign.invalid"})[0], 403)

    def test_tools_are_available_and_counted_without_archive_becoming_model_context(self):
        data = json.loads(self.request("/mcp")[2])
        self.assertEqual(data["tools"], 2)
        body = {"messages": [{"role": "user", "content": "hello"}], "reasoning_effort": "none"}
        plain = json.loads(self.request("/v1/chat/count_tokens", body)[2])
        offered = json.loads(self.request("/v1/chat/count_tokens", {**body, "strata_mcp": True})[2])
        self.assertGreater(offered["input_tokens"], plain["input_tokens"])

    def test_fresh_ui_alias_and_retirement_worker_preserve_records(self):
        status, headers, page = self.request("/strata")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertIn(b"Import old llama chats", page)
        status, headers, worker = self.request("/sw.js")
        self.assertEqual(status, 200)
        self.assertIn(b"unregister()", worker)
        self.assertNotIn(b"deleteDatabase", worker)
        self.assertNotIn(b"caches.delete", worker)
        self.assertNotIn(b"navigate(", worker)


if __name__ == "__main__":
    unittest.main()
