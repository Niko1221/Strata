"""Optional operator pricing in model discovery, through config startup and HTTP.

    python -m unittest serve.test_model_pricing -v
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve import server as S


CARD = {"prompt": "0.00000015", "completion": 4.7e-7, "input_cache_read": "0.000000016",
        "input_cache_write": "0.0000002", "request": "0"}


class PricingConfig(unittest.TestCase):
    def test_missing_null_and_empty(self):
        for cfg in ({}, {"pricing": None}, {"pricing": {}}):
            with self.subTest(cfg=cfg):
                self.assertEqual(S.pricing_from_config(cfg), {})

    def test_values_and_extensions_are_preserved(self):
        card = {**CARD, "cache_read": 0, "cache_write": "2e-7", "extension": {"unit": "example"}}
        result = S.pricing_from_config({"pricing": card})
        self.assertEqual(result, card)
        self.assertIsNot(result, card)
        self.assertIsInstance(result["prompt"], str)
        self.assertIsInstance(result["completion"], float)

    def test_invalid_block_types(self):
        for value in (False, True, 0, 1, "", "oops", [], [CARD]):
            with self.subTest(value=value), self.assertRaisesRegex(SystemExit, 'config pricing: expected an object'):
                S.pricing_from_config({"pricing": value})

    def test_invalid_rates(self):
        for key in (*CARD, "cache_read", "cache_write"):
            for value in (-1, "-0.1", True, False, None, [], {}, "", "oops", "NaN", "sNaN", "Infinity",
                          "-Infinity", float("nan"), float("inf"), float("-inf")):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(
                        SystemExit, rf'config pricing\.{key}: expected a finite, nonnegative'):
                    S.pricing_from_config({"pricing": {key: value}})

    def test_zero_and_scientific_notation(self):
        for value in (0, 0.0, "0", "0.0", "1.5e-7"):
            card = {key: value for key in CARD}
            self.assertEqual(S.pricing_from_config({"pricing": card}), card)


class _Served(Exception):
    pass


class PricingStartup(unittest.TestCase):
    def listings(self, cfg):
        bodies = {}
        original_serve = S.serve

        def capture(svc, host=None, port=None):
            httpd = original_serve(svc, host="127.0.0.1", port=0)
            try:
                for path in ("/models", "/v1/models"):
                    req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}{path}",
                                                 headers={"Authorization": "Bearer review-test-key"})
                    with urllib.request.urlopen(req, timeout=10) as r:
                        bodies[path] = r.read()
            finally:
                httpd.shutdown()
                httpd.server_close()
            raise _Served

        with tempfile.TemporaryDirectory() as d:
            args = ["server.py", "--engine", "mock", "--port", "0", "--tokenizer", str(Path(d) / "absent")]
            if cfg is not None:
                config = Path(d) / "config.json"
                config.write_text(json.dumps(cfg), encoding="utf-8")
                args += ["--config", str(config)]
            with mock.patch.object(sys, "argv", args), mock.patch.object(S, "serve", capture), \
                    mock.patch.object(S.Service, "start_telemetry"), \
                    mock.patch.dict(S.os.environ, {"STRATA_API_KEY": "review-test-key"}), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(_Served):
                    S.main()
        return bodies

    def test_existing_configs_keep_exact_listing(self):
        expected = json.dumps({"object": "list", "data": [
            {"id": "review-model", "object": "model", "status": {"value": "loaded"},
             "meta": {"n_ctx": 32768}, "architecture": {"input_modalities": ["text"],
                                                         "output_modalities": ["text"]}}]}).encode()
        for extra in ({}, {"pricing": None}, {"pricing": {}}):
            with self.subTest(extra=extra):
                for body in self.listings({"model_name": "review-model", **extra}).values():
                    self.assertEqual(body, expected)

    def test_startup_without_a_config(self):
        for body in self.listings(None).values():
            self.assertNotIn("pricing", json.loads(body)["data"][0])

    def test_configured_rates_on_both_endpoints_and_aliases(self):
        for card in (CARD, {"prompt": 0, "completion": "0"}, {"cache_read": "1.6e-8", "cache_write": 2e-7}):
            with self.subTest(card=card):
                bodies = self.listings({"model_name": "review-model", "aliases": ["local"], "pricing": card})
                self.assertEqual(bodies["/models"], bodies["/v1/models"])
                models = json.loads(bodies["/models"])["data"]
                self.assertEqual([m["id"] for m in models], ["review-model", "local"])
                self.assertTrue(all(m["pricing"] == card for m in models))

    def test_bad_config_stops_before_engine_start(self):
        with tempfile.TemporaryDirectory() as d:
            config = Path(d) / "config.json"
            config.write_text(json.dumps({"pricing": {"prompt": -1}}), encoding="utf-8")
            with mock.patch.object(sys, "argv", ["server.py", "--engine", "strata", "--config", str(config)]), \
                    mock.patch.object(S, "StrataEngine") as engine:
                with self.assertRaisesRegex(SystemExit, 'config pricing.prompt'):
                    S.main()
                engine.assert_not_called()


if __name__ == "__main__":
    unittest.main()
