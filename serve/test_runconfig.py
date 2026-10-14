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
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve import runconfig
from serve import engine_options
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve, engine_args, child_env

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


class Engine(unittest.TestCase):
    """The Engine tab: the catalog, its env/args keys in the run config, and vendor/arch detection."""

    def test_catalog_sane(self):
        self.assertTrue(engine_options.OPTIONS and engine_options.ARGS)
        keys = [o["key"] for o in engine_options.OPTIONS + engine_options.ARGS]
        self.assertEqual(len(keys), len(set(keys)))
        for o in engine_options.OPTIONS:
            self.assertTrue(o["key"].startswith("env.STRATA_"), o["key"])
            self.assertIn(o["vendor"], engine_options.VENDORS)
            self.assertIn(o["tier"], engine_options.TIERS)
            self.assertTrue(o["desc"], o["key"])
        for o in engine_options.ARGS:
            self.assertTrue(o["key"].split(".", 1)[1].startswith("--"), o["key"])
            self.assertIn(o["kind"], ("bool", "int", "number", "str", "enum"))
        amd = engine_options.catalog("amd")
        self.assertTrue(any(o["vendor"] == "amd" for o in amd))
        self.assertTrue(any(o["vendor"] == "all" for o in amd))
        self.assertFalse(any(o["vendor"] in ("nvidia", "intel") for o in amd))
        all_gpu = engine_options.catalog("all")
        all_keys = [o["key"] for o in all_gpu]
        self.assertEqual(len(all_keys), len(set(all_keys)))       # unknown backend must not duplicate common options
        self.assertEqual(next(o["vendor"] for o in engine_options.ARGS if o["flag"] == "--pipeline-windows"), "all")
        self.assertEqual(sum(o["key"] == "arg.--pipeline-windows" for o in amd), 1)
        apple = engine_options.catalog("apple")
        self.assertTrue(any(o["key"] == "env.STRATA_METAL_ARENA_GB" for o in apple))
        self.assertFalse(any(o["vendor"] in ("nvidia", "intel", "amd") for o in apple))
        self.assertFalse(any(o["key"] == "argflag.--vram-elastic" for o in apple))   # NVIDIA-only stays hidden on a Mac
        apple_keys = [o["key"] for o in apple]
        self.assertEqual(len(apple_keys), len(set(apple_keys)))   # vendor + common options, no duplicates

    def test_vendor_and_arch(self):
        self.assertEqual(engine_options.detect_vendor({"backend": "hip"}), "amd")
        self.assertEqual(engine_options.detect_vendor({"backend": "cuda"}), "nvidia")
        self.assertEqual(engine_options.detect_vendor({"backend": "sycl"}), "intel")
        self.assertEqual(engine_options.detect_vendor({"backend": "metal"}), "apple")
        self.assertEqual(engine_options.detect_vendor({}), "all")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "engine.log"
            p.write_text("strata generate: GPU 0: AMD Radeon AI PRO R9700 (gfx1201)\n", encoding="utf-8")
            self.assertEqual(engine_options.detect_arch(p), "gfx1201")
            p.write_text("GPU 0: NVIDIA GeForce RTX 3060, compute capability 8.6, 12.0 GiB\n", encoding="utf-8")
            self.assertEqual(engine_options.detect_arch(p), "sm_86")
            p.write_text("GPU 0: NVIDIA RTX 5090, compute capability 12.0\n", encoding="utf-8")
            self.assertEqual(engine_options.detect_arch(p), "sm_120")
            p.write_text("GPU 0: Intel Arc A770 Graphics\n", encoding="utf-8")
            self.assertEqual(engine_options.detect_arch(p), "Arc A770")
            p.write_text("strata metal: decode window on Apple M4 Pro: swift-1.5-iq3_s wrapped (23.0 GB), "
                         "262144 cells, PLE on, ready in 4200 ms\n", encoding="utf-8")
            self.assertEqual(engine_options.detect_arch(p), "Apple M4 Pro")
            p.write_text("no gpu here\n", encoding="utf-8")
            self.assertEqual(engine_options.detect_arch(p), "")
            # a banner older than the 4 MiB tail is not read; the newest one in the tail wins
            p.write_bytes(b"strata generate: GPU 0: old (gfx900)\n" + b"x" * (5 << 20) +
                          b"\nstrata generate: GPU 0: AMD Radeon AI PRO R9700 (gfx1201)\n")
            self.assertEqual(engine_options.detect_arch(p), "gfx1201")
            p.write_bytes(b"strata generate: GPU 0: old (gfx900)\n" + b"x" * (1 << 20) +
                          b"\nstrata metal: decode window on Apple M4 Pro: pack wrapped, ready\n")
            self.assertEqual(engine_options.detect_arch(p), "Apple M4 Pro")
        self.assertEqual(engine_options.detect_arch(Path("/nonexistent/engine.log")), "")
        self.assertEqual(engine_options.vendor_of_arch("Apple M4 Max"), "apple")
        self.assertEqual(engine_options.vendor_of_arch("gfx1201"), "amd")
        self.assertEqual(engine_options.vendor_of_arch(""), "")

    def test_multi_gpu_union(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "engine.log"
            p.write_text("strata generate: GPU 0: NVIDIA GeForce RTX 3060 (sm_86)\n"
                         "strata generate: GPU 1: AMD Radeon RX 7600 (gfx1100)\n"
                         "layer split: card order 0,1\n", encoding="utf-8")
            self.assertEqual(engine_options.detect_archs(p), ["sm_86", "gfx1100"])
            self.assertEqual(engine_options.detect_arch(p), "gfx1100")     # the last one, as before
            self.assertEqual(engine_options.detect_vendors({"backend": "cuda"}, p), ["nvidia", "amd"])
            self.assertEqual(engine_options.detect_vendors({}, p), ["nvidia", "amd"])   # no backend: the log decides
            self.assertEqual(engine_options.detect_vendors({"backend": "hip"}), ["amd"])
            self.assertEqual(engine_options.detect_vendors({}), ["all"])
        mixed = engine_options.catalog(["nvidia", "amd"])
        keys = [o["key"] for o in mixed]
        self.assertEqual(len(keys), len(set(keys)))                        # no key twice
        self.assertIn("env.STRATA_DF_PDL", keys)                           # NVIDIA's own
        self.assertIn("env.STRATA_DENSE_MMQ", keys)                        # AMD's own
        self.assertIn("env.STRATA_GR_V3", keys)                            # common
        self.assertNotIn("env.STRATA_METAL_ARENA_GB", keys)                # Apple's, not in play
        self.assertEqual([o["key"] for o in engine_options.catalog("amd")],
                         [o["key"] for o in engine_options.catalog(["amd"])])   # str == one-element list

    def test_env_and_arg_keys_round_trip(self):
        new, changed = runconfig.apply(CFG, {"env.STRATA_ATTN_PRE75": "1", "arg.--pipeline-windows": 2,
                                             "argflag.--vram-elastic": True, "argenum.--kv": "k8v4",
                                             "argstr.--prefill": "auto:32768", "arg.--spec-min-p": 0.65,
                                             "arg.--adapt-min-gain": 1.75})
        self.assertEqual(new["env"], {"STRATA_ATTN_PRE75": "1"})
        self.assertEqual(new["args"][new["args"].index("--kv") + 1], "k8v4")   # the existing --kv is replaced
        self.assertEqual(new["args"][new["args"].index("--prefill") + 1], "auto:32768")
        self.assertEqual(new["args"][new["args"].index("--spec-min-p") + 1], "0.65")
        self.assertIn("--vram-elastic", new["args"])
        self.assertEqual(len(changed), 7)
        for k, v in (("env.STRATA_ATTN_PRE75", "1"), ("arg.--pipeline-windows", 2),
                     ("argflag.--vram-elastic", True), ("argenum.--kv", "k8v4"),
                     ("argstr.--prefill", "auto:32768"), ("arg.--spec-min-p", 0.65), ("arg.--adapt-min-gain", 1.75)):
            self.assertEqual(runconfig.value_of(new, k), v)
        self.assertIsNone(runconfig.value_of(CFG, "env.STRATA_ATTN_PRE75"))
        back, _ = runconfig.apply(new, {"env.STRATA_ATTN_PRE75": None, "arg.--pipeline-windows": None,
                                        "argflag.--vram-elastic": False, "argenum.--kv": None,
                                        "argstr.--prefill": None, "arg.--spec-min-p": None,
                                        "arg.--adapt-min-gain": None})
        self.assertNotIn("env", back)
        for flag in ("--pipeline-windows", "--vram-elastic", "--kv", "--prefill"):
            self.assertNotIn(flag, back["args"])
        self.assertEqual(back["args"], ["--pack", "data/packs/q2_0"])   # None removes explicit overrides: engine defaults apply

    def test_env_values_refused(self):
        for changes in ({"env.NOT_A_SWITCH": "1"}, {"env.STRATA_ATTN_PRE75": "a\nb"}, {"env.STRATA_ATTN_PRE75": ""},
                        {"env.STRATA_ATTN_PRE75": "x" * 600}, {"env.STRATA_ATTN_PRE75": []},
                        {"arg.--pipeline-windows": -1}, {"arg.--pipeline-windows": 3}, {"arg.--pipeline-windows": 1.5},
                        {"arg.--batch": 1}, {"arg.--batch": 9}, {"arg.--adapt-async": 2},
                        {"arg.--spec-min-p": 1.1}, {"arg.--adapt-min-gain": -0.1},
                        {"argflag.--vram-elastic": "yes"}, {"argenum.--kv": "q2_0"}, {"argstr.--prefill": {}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runconfig.apply(CFG, changes)

    def test_the_page_has_the_engine_tab(self):
        web = ROOT / "serve" / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("tab-btn-engine", "view-engine", "eng-list", "eng-save", "eng-search", "eng-ab", "eng-scope"):
            self.assertIn(f'id="{el}"', html)
        self.assertIn('fetch("engine-options"', js)
        self.assertIn('"engine"', js)


class EngineHttp(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "strata-q2_0.json"
        log = Path(self.dir.name) / "engine.log"
        log.write_text("strata generate: GPU 0: AMD Radeon AI PRO R9700 (gfx1201)\n", encoding="utf-8")
        self.path.write_text(json.dumps({**CFG, "backend": "hip", "log": str(log)}, indent=1), encoding="utf-8")
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.config_path = str(self.path)
        self.httpd = serve(self.svc, port=0)
        self.host = f"127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.dir.cleanup()

    def call(self, method, path, body=None, headers=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        req = urllib.request.Request(f"http://{self.host}{path}", method=method, headers=h,
                                     data=None if body is None else json.dumps(body).encode())
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_options_listed_for_the_vendor(self):
        status, d = self.call("GET", "/engine-options")
        self.assertEqual(status, 200, d)
        self.assertEqual(d["vendor"], "amd")
        self.assertEqual(d["arch"], "gfx1201")
        by_key = {o["key"]: o for o in d["options"]}
        self.assertIn("env.STRATA_DENSE_MMQ", by_key)          # AMD's own
        self.assertIn("env.STRATA_GR_V3", by_key)              # all-vendor
        self.assertNotIn("argflag.--vram-elastic", by_key)     # NVIDIA's, not listed for AMD
        self.assertTrue(all(o["vendor"] in ("amd", "all") for o in d["options"]))

    def test_mixed_vendor_split_lists_both(self):
        log = Path(self.dir.name) / "engine.log"
        log.write_text("strata generate: GPU 0: NVIDIA GeForce RTX 3060 (sm_86)\n"
                       "strata generate: GPU 1: AMD Radeon RX 7600 (gfx1100)\n", encoding="utf-8")
        self.path.write_text(json.dumps({**CFG, "backend": "cuda", "log": str(log)}, indent=1), encoding="utf-8")
        status, d = self.call("GET", "/engine-options")
        self.assertEqual(status, 200, d)
        self.assertEqual(d["vendor"], "nvidia + amd")
        self.assertEqual(d["arch"], "sm_86, gfx1100")
        by_key = {o["key"]: o for o in d["options"]}
        self.assertIn("env.STRATA_DF_PDL", by_key)         # NVIDIA's own: the split spans both cards
        self.assertIn("env.STRATA_DENSE_MMQ", by_key)      # AMD's own
        self.assertNotIn("env.STRATA_METAL_ARENA_GB", by_key)

    def test_apple_metal_options_and_config_save(self):
        log = Path(self.dir.name) / "engine.log"
        log.write_text("strata metal: decode window on Apple M4 Pro: pack wrapped (23.0 GB), ready\n", encoding="utf-8")
        apple_cfg = {**CFG, "backend": "metal", "log": str(log)}
        self.path.write_text(json.dumps(apple_cfg, indent=1), encoding="utf-8")
        status, d = self.call("GET", "/engine-options")
        self.assertEqual(status, 200, d)
        self.assertEqual(d["vendor"], "apple")
        self.assertEqual(d["arch"], "Apple M4 Pro")
        by_key = {o["key"]: o for o in d["options"]}
        self.assertIn("env.STRATA_METAL_ARENA_GB", by_key)
        self.assertIn("env.STRATA_HC_Q8", by_key)                 # common Metal-aware setting
        self.assertNotIn("env.STRATA_DENSE_MMQ", by_key)          # AMD-only
        self.assertNotIn("env.STRATA_METAL_ALLOC_LOG", by_key)    # debug-only inventory entry
        self.assertNotIn("argflag.--vram-elastic", by_key)        # NVIDIA-only
        with contextlib.redirect_stdout(io.StringIO()):
            status, saved = self.call("POST", "/config", {"set": {"env.STRATA_METAL_ARENA_GB": "64"}},
                                      {"Origin": f"http://{self.host}"})
        self.assertEqual(status, 200, saved)
        updated = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(updated["env"]["STRATA_METAL_ARENA_GB"], "64")
        self.assertEqual(json.loads(Path(str(self.path) + ".bak").read_text(encoding="utf-8"))["backend"], "metal")
        updated.pop("backend")       # unknown/old configs can still infer Apple from the Metal startup line
        self.path.write_text(json.dumps(updated, indent=1), encoding="utf-8")
        status, d = self.call("GET", "/engine-options")
        self.assertEqual(status, 200, d)
        self.assertEqual(d["vendor"], "apple")

    def test_write_an_env_switch(self):
        with contextlib.redirect_stdout(io.StringIO()):
            status, v = self.call("POST", "/config", {"set": {"env.STRATA_ATTN_PRE75": "1", "arg.--pipeline-windows": 2}},
                                  {"Origin": f"http://{self.host}"})
        self.assertEqual(status, 200, v)
        self.assertEqual(v["changed"], ["env.STRATA_ATTN_PRE75", "arg.--pipeline-windows"])
        cfg = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(cfg["env"], {"STRATA_ATTN_PRE75": "1"})
        self.assertEqual(cfg["args"][-2:], ["--pipeline-windows", "2"])
        status, d = self.call("GET", "/engine-options")
        self.assertEqual({o["key"]: o["value"] for o in d["options"]}["env.STRATA_ATTN_PRE75"], "1")

    def test_restart_is_guarded(self):
        status, b = self.call("POST", "/restart", {}, {"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        status, b = self.call("POST", "/restart", {})       # own page: the mock engine cannot respawn from a config
        self.assertEqual(status, 400)
        self.assertIn("restarted", b["error"]["message"])
        status, d = self.call("GET", "/engine-options")
        self.assertIs(d["restart_pending"], False)          # a mock engine: nothing to compare, nothing pending


class FakeEngine:
    """Enough of StrataEngine for the restart path: spawn, unload, restart, alive."""
    def __init__(self, spawn):
        self.spawn, self.unloaded, self.restarts = spawn, True, 0
    def alive(self):
        return not self.unloaded
    def unload(self):
        self.unloaded = True
    def restart(self):
        self.restarts += 1
        self.unloaded = False
    def exit_code(self):
        return None


class Restart(unittest.TestCase):
    """POST /restart: the engine respawns from the run config as it is NOW (new env/args), not from the old spawn."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "strata-q2_0.json"
        self.path.write_text(json.dumps(CFG, indent=1), encoding="utf-8")
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok", max_context=4096), tok,
                           ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.config_path = str(self.path)
        self.svc.engine = FakeEngine((CFG["exe"], engine_args(CFG), None, None, child_env(CFG), False, None))

    def tearDown(self):
        self.dir.cleanup()

    def test_pending_then_apply(self):
        self.assertFalse(self.svc.restart_pending(CFG))
        new, _ = runconfig.apply(CFG, {"env.STRATA_ATTN_PRE75": "1"})
        self.path.write_text(json.dumps(new, indent=1), encoding="utf-8")
        self.assertTrue(self.svc.restart_pending(new))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.svc.apply_config_and_restart(), "restarting")
        deadline = time.time() + 15
        while self.svc.engine.restarts == 0 and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(self.svc.engine.restarts, 1)                    # the thread loaded the new engine
        self.assertEqual(self.svc.engine.spawn[4]["STRATA_ATTN_PRE75"], "1")   # respawned with the new env

    def test_metal_settings_are_used_for_restart_spawn(self):
        cfg = {**CFG, "backend": "metal"}
        self.path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        self.svc.engine = FakeEngine(self.svc._spawn_from(cfg))
        new, _ = runconfig.apply(cfg, {"env.STRATA_METAL_ARENA_GB": "64"})
        self.path.write_text(json.dumps(new, indent=1), encoding="utf-8")
        self.assertTrue(self.svc.restart_pending(new))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.svc.apply_config_and_restart(), "restarting")
        deadline = time.time() + 15
        while self.svc.engine.restarts == 0 and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(self.svc.engine.restarts, 1)
        self.assertEqual(self.svc.engine.spawn[4]["STRATA_METAL_ARENA_GB"], "64")
        self.assertFalse(self.svc.restart_pending(new))

    def test_busy_changes_nothing(self):
        self.svc.engine.unloaded = False                                 # the engine is up
        self.svc.status["busy"] = 1                                      # and a request is running
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.svc.apply_config_and_restart(), "busy")
        self.assertEqual(self.svc.engine.restarts, 0)
        self.assertFalse(self.svc.engine.unloaded)                       # it was not stopped
        self.assertNotIn("STRATA_ATTN_PRE75", self.svc.engine.spawn[4])


if __name__ == "__main__":
    unittest.main()
