import copy
import json
import tempfile
import unittest
from pathlib import Path

from serve.mcp import hub_from_config


class McpConfigOverlay(unittest.TestCase):
    def load(self, cfg, overlay=None):
        if overlay is None:
            return hub_from_config(cfg)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mcp.json"
            path.write_text(json.dumps(overlay), encoding="utf-8")
            return hub_from_config(cfg, str(path))

    def test_file_can_disable_an_existing_server(self):
        cfg = {"mcp_servers": {"notes": {"command": "notes-server"},
                               "weather": {"command": "weather-server"}}}
        hub = self.load(cfg, {"mcpServers": {"notes": {"disabled": True}}})
        self.assertEqual(set(hub.servers), {"weather"})

    def test_disabling_the_last_server_returns_none(self):
        cfg = {"mcp_servers": {"notes": {"command": "notes-server"}}}
        self.assertIsNone(self.load(cfg, {"mcpServers": {"notes": {"disabled": True}}}))

    def test_later_inline_spelling_can_disable_earlier_entry(self):
        cfg = {"mcp_servers": {"notes": {"command": "notes-server"}},
               "mcpServers": {"notes": {"disabled": True}}}
        self.assertIsNone(self.load(cfg))

    def test_alternative_file_spelling_honors_disabled(self):
        cfg = {"mcpServers": {"notes": {"command": "notes-server"}}}
        self.assertIsNone(self.load(cfg, {"mcp_servers": {"notes": {"disabled": True}}}))

    def test_enabled_override_still_replaces_existing_entry(self):
        cfg = {"mcp_servers": {"notes": {"command": "old-server"}}}
        hub = self.load(cfg, {"mcpServers": {"notes": {"command": "new-server"}}})
        self.assertEqual(hub.servers["notes"].cfg["command"], "new-server")

    def test_later_file_can_reenable_disabled_entry(self):
        cfg = {"mcp_servers": {"notes": {"command": "old-server"}},
               "mcpServers": {"notes": {"disabled": True}}}
        hub = self.load(cfg, {"mcpServers": {"notes": {"command": "new-server"}}})
        self.assertEqual(hub.servers["notes"].cfg["command"], "new-server")

    def test_nonexistent_disable_preserves_other_servers(self):
        cfg = {"mcp_servers": {"weather": {"command": "weather-server"}}}
        hub = self.load(cfg, {"mcpServers": {"notes": {"disabled": True}}})
        self.assertEqual(set(hub.servers), {"weather"})

    def test_input_is_not_mutated(self):
        cfg = {"mcp_servers": {"notes": {"command": "notes-server"}}}
        overlay = {"mcpServers": {"notes": {"disabled": True}}}
        before = copy.deepcopy((cfg, overlay))
        self.load(cfg, overlay)
        self.assertEqual((cfg, overlay), before)

    def test_invalid_overlay_is_still_rejected(self):
        cfg = {"mcp_servers": {"notes": {"command": "notes-server"}}}
        for block in ([], {"notes": {}}, {"notes": {"command": "x", "args": "bad"}}):
            with self.subTest(block=block), self.assertRaises(SystemExit):
                self.load(cfg, {"mcpServers": block})

    def test_standalone_disabled_and_empty_configs_stay_empty(self):
        self.assertIsNone(self.load({"mcp_servers": {"notes": {"disabled": True}}}))
        self.assertIsNone(self.load({}))


if __name__ == "__main__":
    unittest.main()
