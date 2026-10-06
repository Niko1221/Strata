"""serve/test_runconfig.py - #564: the web page's Settings view of the run config (GET / POST /config), against the
mock engine: only the listed keys change, every other key of the file stays, a bad value changes nothing, and only
Strata's own page may write it (JSON, no foreign Origin, the API key when one is set).

    python -m unittest serve.test_runconfig
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve import runconfig
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, sampling_defaults_from_config, serve

ROOT = Path(__file__).resolve().parents[1]
CFG = {"exe": "engine/strata.exe", "args": ["--pack", "data/packs/q2_0", "--kv", "int8", "--spec-min-p", "0.5"],
       "model_name": "qwen3.8-flash-next-q2_0", "port": 8080, "host": "0.0.0.0", "api_key": "",
       "sampling": {"temperature": 0.7, "presence_penalty": 1.5},
       "mcp_servers": {"files": {"command": "npx", "args": ["-y", "server-filesystem", "."]}},
       "cors_origins": ["http://localhost:3000"]}


class Apply(unittest.TestCase):
    def test_only_the_named_keys_change(self):
        new, changed = runconfig.apply(CFG, {"sampling.top_p": 0.95, "reasoning_budget_tokens": 4096,
                                             "vram_reserve_mib": 2048, "aliases": "qwen, local", "lazy_load": True})
        self.assertEqual(changed, ["sampling.top_p", "reasoning_budget_tokens", "vram_reserve_mib", "aliases",
                                   "lazy_load"])
        self.assertEqual(new["sampling"], {"temperature": 0.7, "presence_penalty": 1.5, "top_p": 0.95})
        self.assertEqual(new["args"][-2:], ["--vram-reserve-mib", "2048"])
        self.assertEqual(new["aliases"], ["qwen", "local"])
        for k in ("exe", "model_name", "port", "host", "api_key", "mcp_servers", "cors_origins"):
            self.assertEqual(new[k], CFG[k], k)
        self.assertEqual(CFG["sampling"], {"temperature": 0.7, "presence_penalty": 1.5})   # the input is left alone

    def test_null_is_the_default_again(self):
        cfg, _ = runconfig.apply(CFG, {"vram_reserve_mib": 2048, "fit_max_tokens": True})
        new, changed = runconfig.apply(cfg, {"vram_reserve_mib": None, "fit_max_tokens": None,
                                             "sampling.temperature": None, "idle_unload_s": None})
        self.assertEqual(changed, ["vram_reserve_mib", "fit_max_tokens", "sampling.temperature"])
        self.assertEqual(new["args"], CFG["args"])
        self.assertNotIn("fit_max_tokens", new)
        self.assertEqual(new["sampling"], {"presence_penalty": 1.5})
        new, _ = runconfig.apply({"sampling": {"top_k": 20}}, {"sampling.top_k": None})
        self.assertNotIn("sampling", new)                                 # an empty block goes

    def test_refused(self):
        for changes in ({"api_key": "x"}, {"host": "0.0.0.0"}, {"mcp_servers": {}}, {"before_load": "calc.exe"},
                        {"args": []}, {"exe": "evil.exe"}, {"trusted_origins": ["*"]}, {"allowed_hosts": ["*"]},
                        {"sampling.temperature": -1}, {"sampling.top_p": 0}, {"sampling.top_k": 65},
                        {"sampling.min_p": 2}, {"reasoning_budget_tokens": 1.5}, {"fit_max_tokens": "yes"},
                        {"effort_position": "middle"}, {"vram_reserve_mib": -5}, {"idle_unload_s": True},
                        {"aliases": [1]}, {}, None, ["x"]):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runconfig.apply(CFG, changes)
        with self.assertRaises(ValueError):                              # one bad value: nothing is applied
            runconfig.apply(CFG, {"fit_max_tokens": True, "sampling.top_k": 0})
        with self.assertRaises(ValueError):
            runconfig.apply({**CFG, "vision": {"gpu": True}}, {"lazy_load": True})

    def test_view(self):
        v = runconfig.view({**CFG, "args": CFG["args"] + ["--vram-reserve-mib", "1500"]}, "x/strata-q2_0.json")
        self.assertEqual(v["file"], "strata-q2_0.json")
        got = {k["key"]: k for k in v["keys"]}
        self.assertEqual(got["sampling.temperature"]["value"], 0.7)
        self.assertEqual(got["vram_reserve_mib"]["value"], 1500)
        self.assertIsNone(got["fit_max_tokens"]["value"])
        self.assertEqual(got["anthropic_thinking"]["choices"], ["model", "on_request"])
        self.assertNotIn("api_key", got)
        self.assertNotIn("mcp_servers", got)


class Presets(unittest.TestCase):
    """#1129: the two sampling sets the model card recommends - setup writes one of them (--thinking / --instruct),
    and the server's start line names it, so a start says out loud what a request that asks for none gets."""

    def test_the_cards_numbers_reach_the_server_as_they_are(self):
        for name, block in runconfig.SAMPLING_PRESETS.items():
            with self.subTest(name):
                self.assertEqual(sampling_defaults_from_config({"sampling": block}), block)

    def test_the_block_is_named(self):
        self.assertEqual(runconfig.preset_of(dict(runconfig.SAMPLING_PRESETS["thinking"])), "thinking")
        self.assertEqual(runconfig.preset_of({**runconfig.SAMPLING_PRESETS["instruct"], "top_p": 0.80, "top_k": 20.0}),
                         "instruct")                          # 0.80 and 20.0, as a JSON file writes them
        for own in (None, {},                                      # no block, an empty one
                    {"temperature": 1.0, "top_p": 0.95, "top_k": 20},      # the shorter block of an earlier setup
                    {"temperature": 0.0}, {"temperature": "warm"}):        # their own numbers, a value not a number
            self.assertIsNone(runconfig.preset_of(own), own)

    def test_the_one_line_a_start_prints(self):
        self.assertEqual(runconfig.sampling_summary(runconfig.SAMPLING_PRESETS["instruct"]),
                         "temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, "
                         "repetition_penalty=1.0")
        self.assertEqual(runconfig.sampling_summary({"seed": 7, "top_p": 0.5}), "top_p=0.5, seed=7")   # own keys last
        self.assertEqual(runconfig.sampling_summary(None), "")


class Http(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "strata-q2_0.json"
        self.path.write_text(json.dumps(CFG, indent=1), encoding="utf-8")
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.config_path = str(self.path)
        self.httpd = serve(self.svc, port=0)
        self.host = f"127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.dir.cleanup()

    def call(self, method, body=None, headers=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        req = urllib.request.Request(f"http://{self.host}/config", method=method, headers=h,
                                     data=None if body is None else (body if isinstance(body, bytes)
                                                                     else json.dumps(body).encode()))
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def saved(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_read_and_write_from_the_page(self):
        status, v = self.call("GET")
        self.assertEqual(status, 200, v)
        self.assertEqual({k["key"]: k["value"] for k in v["keys"]}["sampling.temperature"], 0.7)
        with contextlib.redirect_stdout(io.StringIO()):
            status, v = self.call("POST", {"set": {"sampling.temperature": 1.0, "open_browser": False}},
                                  {"Origin": f"http://{self.host}"})
        self.assertEqual(status, 200, v)
        self.assertEqual(v["changed"], ["sampling.temperature", "open_browser"])
        cfg = self.saved()
        self.assertEqual(cfg["sampling"]["temperature"], 1.0)
        self.assertIs(cfg["open_browser"], False)
        for k in ("mcp_servers", "cors_origins", "host", "args", "exe"):
            self.assertEqual(cfg[k], CFG[k], k)
        self.assertEqual(json.loads((self.path.parent / "strata-q2_0.json.bak").read_text(encoding="utf-8")), CFG)
        status, v = self.call("POST", {"set": {"open_browser": False}})   # no Origin (a script on this PC): allowed
        self.assertEqual((status, v["changed"]), (200, []))

    def test_only_from_strata_s_own_page(self):
        before = self.path.read_bytes()
        status, _ = self.call("POST", {"set": {"fit_max_tokens": True}}, {"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        status, _ = self.call("POST", b'{"set": {"fit_max_tokens": true}}', {"Content-Type": "text/plain"})
        self.assertEqual(status, 415)
        for body in ({"set": {"api_key": ""}}, {"set": {"mcp_servers": {"x": {"command": "calc"}}}},
                     {"set": {"sampling.top_k": 0}}, {"nothing": 1}, b"not json"):
            status, b = self.call("POST", body, {"Origin": f"http://{self.host}"})
            self.assertEqual(status, 400, (body, b))
        self.assertEqual(self.path.read_bytes(), before)                 # nothing was written
        self.assertFalse((self.path.parent / "strata-q2_0.json.bak").exists())

    def test_the_api_key(self):
        self.svc.api_key = "secret"
        self.assertEqual(self.call("GET")[0], 401)
        self.assertEqual(self.call("POST", {"set": {"fit_max_tokens": True}})[0], 401)
        with contextlib.redirect_stdout(io.StringIO()):
            status, _ = self.call("POST", {"set": {"fit_max_tokens": True}}, {"Authorization": "Bearer secret"})
        self.assertEqual(status, 200)
        self.assertIs(self.saved()["fit_max_tokens"], True)

    def test_without_a_run_config(self):
        self.svc.config_path = None
        self.assertEqual(self.call("GET")[0], 404)
        self.assertEqual(self.call("POST", {"set": {"fit_max_tokens": True}})[0], 404)

    def test_the_page_has_the_view(self):
        web = ROOT / "serve" / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("cfg-card", "cfg-form", "cfg-save", "cfg-msg"):
            self.assertIn(f'id="{el}"', html)
        self.assertIn('fetch("config", {method: "POST", headers: headers(true)', js)


if __name__ == "__main__":
    unittest.main()
