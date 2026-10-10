import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from serve import local_ui


class ConversationStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patch = mock.patch.object(local_ui, "CONVERSATIONS", self.root / "conversations")
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_round_trip_list_and_delete(self):
        meta = local_ui.save_conversation({
            "messages": [
                {"role": "user", "text": "hello from a persisted chat"},
                {"role": "assistant", "text": "hello"},
            ]
        })
        self.assertRegex(meta["id"], r"^[0-9a-f]{32}$")
        self.assertEqual(meta["title"], "hello from a persisted chat")
        self.assertEqual(meta["message_count"], 2)

        stored = local_ui.get_conversation(meta["id"])
        self.assertEqual(stored["messages"][1]["text"], "hello")
        self.assertEqual(local_ui.list_conversations()[0]["id"], meta["id"])

        result = local_ui.delete_conversation(meta["id"])
        self.assertEqual(result, {"id": meta["id"], "deleted": True})
        self.assertEqual(local_ui.list_conversations(), [])

    def test_existing_created_at_is_preserved(self):
        first = local_ui.save_conversation({"messages": [], "title": "first"})
        first_data = local_ui.get_conversation(first["id"])
        second = local_ui.save_conversation({"id": first["id"], "messages": [], "title": "second"})
        second_data = local_ui.get_conversation(second["id"])
        self.assertEqual(first_data["created_at"], second_data["created_at"])
        self.assertGreaterEqual(second_data["updated_at"], first_data["updated_at"])

    def test_id_cannot_escape_conversation_directory(self):
        with self.assertRaises(ValueError):
            local_ui.get_conversation("../outside")
        with self.assertRaises(ValueError):
            local_ui.delete_conversation("0" * 31 + "/")

    def test_size_limit_is_enforced(self):
        with mock.patch.object(local_ui, "MAX_CONVERSATION_BYTES", 128):
            with self.assertRaisesRegex(ValueError, "conversation is too large"):
                local_ui.save_conversation({"messages": [{"role": "user", "text": "x" * 500}]})


class McpConfigTests(unittest.TestCase):
    def test_http_upsert_and_url_validation(self):
        cfg = {}
        name, entry = local_ui.upsert_mcp_server(cfg, {
            "name": "local-tools", "transport": "http", "url": "http://127.0.0.1:8000/mcp"
        })
        self.assertEqual(name, "local-tools")
        self.assertEqual(entry, {"url": "http://127.0.0.1:8000/mcp"})
        self.assertEqual(cfg["mcp_servers"][name], entry)
        for bad in ("file:///tmp/mcp", "stdio://tool", "http://user:pass@host/mcp", "https://host/mcp#fragment"):
            with self.subTest(url=bad), self.assertRaises(ValueError):
                local_ui.upsert_mcp_server({}, {"name": "tools", "transport": "http", "url": bad})

    def test_status_view_lists_http_and_stdio_without_secret_values(self):
        view = local_ui.mcp_config_view({
            "mcp_servers": {
                "tools": {
                    "url": "https://example.invalid/mcp",
                    "headers": {"Authorization": "Bearer secret"},
                },
                "local": {
                    "command": "python", "args": ["-m", "demo"], "cwd": "C:/work",
                    "env": {"TOKEN": "super-secret", "MODE": "test"},
                },
            }
        })
        self.assertEqual(view, [
            {"name": "tools", "transport": "http", "url": "https://example.invalid/mcp", "has_headers": True},
            {"name": "local", "transport": "stdio", "command": "python", "args": ["-m", "demo"],
             "cwd": "C:/work", "has_env": True, "env_keys": ["MODE", "TOKEN"]},
        ])
        raw = json.dumps(view)
        self.assertNotIn("Bearer secret", raw)
        self.assertNotIn("super-secret", raw)

    def test_stdio_upsert_preserves_env_only_for_unchanged_launch(self):
        cfg = {"mcp_servers": {"local": {
            "command": "npx", "args": ["-y", "pkg-a"], "cwd": "C:/work", "env": {"TOKEN": "secret"}
        }}}
        _, same = local_ui.upsert_mcp_server(cfg, {
            "name": "local", "transport": "stdio", "command": "npx",
            "args": ["-y", "pkg-a"], "cwd": "C:/work",
        })
        self.assertEqual(same["env"], {"TOKEN": "secret"})

        _, changed = local_ui.upsert_mcp_server(cfg, {
            "name": "local", "transport": "stdio", "command": "npx",
            "args": ["-y", "pkg-b"], "cwd": "C:/work",
        })
        self.assertNotIn("env", changed)

        _, explicit = local_ui.upsert_mcp_server(cfg, {
            "name": "local", "transport": "stdio", "command": "npx",
            "args": ["-y", "pkg-b"], "cwd": "C:/work", "env": {"MODE": 1},
        })
        self.assertEqual(explicit["env"], {"MODE": "1"})

        for bad in (
            {"name": "x", "transport": "stdio", "command": ""},
            {"name": "x", "transport": "stdio", "command": "python", "args": {}},
            {"name": "x", "transport": "stdio", "command": "python", "env": []},
        ):
            with self.subTest(payload=bad), self.assertRaises(ValueError):
                local_ui.upsert_mcp_server({}, bad)

    def test_http_headers_preserved_only_for_same_url_and_legacy_override_removed(self):
        cfg = {
            "mcp_servers": {"tools": {"url": "https://one.invalid/mcp",
                                      "headers": {"Authorization": "Bearer secret"}}},
            "mcpServers": {"other": {"url": "https://other.invalid/mcp"}},
        }
        _, same = local_ui.upsert_mcp_server(cfg, {
            "name": "tools", "transport": "http", "url": "https://one.invalid/mcp",
        })
        self.assertEqual(same["headers"], {"Authorization": "Bearer secret"})
        _, changed = local_ui.upsert_mcp_server(cfg, {
            "name": "tools", "transport": "http", "url": "https://two.invalid/mcp",
        })
        self.assertNotIn("headers", changed)

        cfg["mcpServers"]["tools"] = {"url": "https://legacy.invalid/mcp"}
        _, edited = local_ui.upsert_mcp_server(cfg, {
            "name": "tools", "transport": "http", "url": "https://edited.invalid/mcp",
        })
        self.assertEqual(edited, {"url": "https://edited.invalid/mcp"})
        self.assertNotIn("tools", cfg["mcpServers"])

    def test_remove_mcp_server_removes_only_named_entry_from_both_blocks(self):
        cfg = {
            "mcp_servers": {
                "one": {"url": "http://127.0.0.1:8001/mcp"},
                "two": {"command": "python", "args": ["server.py"]},
            },
            "mcpServers": {
                "two": {"url": "http://127.0.0.1:8002/mcp"},
                "three": {"url": "http://127.0.0.1:8003/mcp"},
            },
        }
        self.assertTrue(local_ui.remove_mcp_server(cfg, "two"))
        self.assertEqual(set(cfg["mcp_servers"]), {"one"})
        self.assertEqual(set(cfg["mcpServers"]), {"three"})
        self.assertFalse(local_ui.remove_mcp_server(cfg, "two"))
        with self.assertRaises(ValueError):
            local_ui.remove_mcp_server(cfg, "../two")

if __name__ == "__main__":
    unittest.main()
