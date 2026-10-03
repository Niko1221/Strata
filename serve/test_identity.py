"""The loaded model's API identity over the real HTTP API, built exactly as serve/server.py builds it from
a strata-*.json config: model_name <- cfg["model_name"], svc.set_aliases(cfg.get("aliases")).  A --variant
config (as setup.py now writes it) must identify the variant in /v1/models and /v1/status, with the
canonical name served as an alias entry for old clients; a canonical config is unchanged.

    python -m unittest serve.test_identity -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# setup.py's own identity rules, so this test follows the source of truth for the config fields
import setup                                    # noqa: E402
from serve.frontend import ChatTemplate         # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096


class VariantIdentity(unittest.TestCase):
    """The exact wiring serve/server.py performs from a config's model_name + aliases fields."""

    def _httpd(self, model_name, aliases):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=CTX), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"), model_name=model_name)
        svc.set_aliases(aliases)
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        return httpd, base

    def get(self, base, path):
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return json.loads(r.read())

    def test_the_variant_identity_is_exposed_everywhere(self):
        httpd, base = self._httpd("qwen3.8-flash-next-q2_0-abliterated", ["qwen3.8-flash-next-q2_0"])
        try:
            models = self.get(base, "/v1/models")
            self.assertEqual(models["data"][0]["id"], "qwen3.8-flash-next-q2_0-abliterated")
            self.assertEqual(models["data"][0]["aliases"], ["qwen3.8-flash-next-q2_0"])
            # the canonical name comes back as an alias entry, marked as such - old clients keep working
            alias = next(m for m in models["data"] if m["id"] == "qwen3.8-flash-next-q2_0")
            self.assertEqual(alias["alias_of"], "qwen3.8-flash-next-q2_0-abliterated")
            self.assertEqual(self.get(base, "/v1/status")["model"], "qwen3.8-flash-next-q2_0-abliterated")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_the_canonical_identity_is_unchanged(self):
        httpd, base = self._httpd("qwen3.8-flash-next-q2_0", None)
        try:
            models = self.get(base, "/v1/models")
            self.assertEqual([m["id"] for m in models["data"]], ["qwen3.8-flash-next-q2_0"])
            self.assertNotIn("aliases", models["data"][0])
            self.assertNotIn("alias_of", models["data"][0])
            self.assertEqual(self.get(base, "/v1/status")["model"], "qwen3.8-flash-next-q2_0")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_a_config_setup_writes_for_a_variant_is_served_under_that_name(self):
        """The end-to-end contract: setup.api_identity -> a strata-*.json (as setup.prepare writes) ->
        serve/server.py reads -> /v1/models + /v1/status identify the variant."""
        model_name, aliases = setup.api_identity(setup.FAMILIES["qwen"], "Q2_0", "abliterated")
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "strata-q2_0-abliterated.json"
            cfg_path.write_text(json.dumps({"exe": "x", "args": [], "model_name": model_name,
                                            "aliases": list(aliases)}), encoding="utf-8")
            read = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
            httpd, base = self._httpd(read.get("model_name", "qwen3.8-flash-next"), read.get("aliases"))
            try:
                self.assertEqual(self.get(base, "/v1/models")["data"][0]["id"],
                                 "qwen3.8-flash-next-q2_0-abliterated")
                self.assertEqual(self.get(base, "/v1/status")["model"], "qwen3.8-flash-next-q2_0-abliterated")
            finally:
                httpd.shutdown()
                httpd.server_close()

    def test_a_config_without_a_name_still_defaults_to_the_original_model(self):
        # serve/server.py's default when a config has no model_name at all (an old/broken config)
        httpd, base = self._httpd("qwen3.8-flash-next", None)
        try:
            self.assertEqual(self.get(base, "/v1/models")["data"][0]["id"], "qwen3.8-flash-next")
            self.assertEqual(self.get(base, "/v1/status")["model"], "qwen3.8-flash-next")
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()