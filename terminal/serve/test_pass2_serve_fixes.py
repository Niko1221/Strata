"""Regression tests for Pass 2 serve fixes:
- #6686: /v1/models and /v1/models/<id> include OpenAI-required `created` (int) and `owned_by` ("strata")
- #6680: Hy4-Flash `xhigh` -> `max` in shared settings & built-in /ui effort selector (Hy4 Off/High/Max, Inkling hidden)
- #6621 / test_hy4_serve_e2e: family base alias matching in Service.check_model
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

TERMINAL = Path(__file__).resolve().parents[1]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (  # noqa: E402
    ByteTokenizer,
    MockEngine,
    Service,
    clean_shared_defaults,
    load_shared_settings,
    serve,
)

WEB = Path(__file__).resolve().parent / "web"


class TestModelsOpenAISchema6686(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(
            MockEngine(tok, "</think>\n\nok", max_context=8192),
            tok,
            ChatTemplate(TERMINAL / "serve" / "chat_template.jinja"),
            model_name="qwen3.8-flash-next-iq3_xxs",
        )
        cls.svc.set_aliases(["qwen"])
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=15) as r:
            return json.loads(r.read().decode("utf-8"))

    def test_models_list_and_retrieve_include_created_and_owned_by(self):
        for path in ("/v1/models", "/models"):
            body = self._get(path)
            self.assertEqual(body["object"], "list")
            self.assertEqual(len(body["data"]), 2)
            for item in body["data"]:
                self.assertIsInstance(item["created"], int)
                self.assertGreater(item["created"], 1_700_000_000)
                self.assertEqual(item["owned_by"], "strata")
                self.assertEqual(item["object"], "model")
                self.assertEqual(item["status"]["value"], "loaded")
                self.assertEqual(item["max_model_len"], 8192)
                self.assertIn("architecture", item)

        for path in ("/v1/models/qwen3.8-flash-next-iq3_xxs", "/v1/models/qwen", "/models/qwen"):
            item = self._get(path)
            self.assertIsInstance(item["created"], int)
            self.assertEqual(item["owned_by"], "strata")


class TestCheckModelFamilyAlias(unittest.TestCase):
    def _svc(self, model_name: str, *, hy4: bool = False, inkling: bool = False, aliases=None):
        tok = ByteTokenizer()
        eng = MockEngine(tok, "ok", max_context=4096)
        eng.hy4_engine = hy4
        eng.inkling_engine = inkling
        svc = Service(eng, tok, ChatTemplate(TERMINAL / "serve" / "chat_template.jinja"), model_name=model_name)
        if aliases:
            svc.set_aliases(aliases)
        return svc

    def test_family_base_alias_accepted_while_cross_family_and_wrong_quant_rejected(self):
        hy4 = self._svc("hy4-flash-sq-64", hy4=True)
        self.assertIsNone(hy4.check_model({"model": "hy4-flash-sq-64"}))
        self.assertIsNone(hy4.check_model({"model": "hy4-flash"}))
        self.assertIsNone(hy4.check_model({"model": "claude-sonnet-4-5-20250929"}))
        self.assertIsNotNone(hy4.check_model({"model": "hy4-flash-sq-32"}))
        self.assertIsNotNone(hy4.check_model({"model": "inkling-2-mini-sq-20"}))

        ink = self._svc("inkling-2-mini-sq-10", inkling=True)
        self.assertIsNone(ink.check_model({"model": "inkling-2-mini"}))
        self.assertIsNotNone(ink.check_model({"model": "inkling-2-mini-sq-20"}))
        self.assertIsNotNone(ink.check_model({"model": "hy4-flash"}))

        # strict_model rejects family base aliases too
        hy4.strict_model = True
        self.assertIsNotNone(hy4.check_model({"model": "hy4-flash"}))
        self.assertIsNone(hy4.check_model({"model": "hy4-flash-sq-64"}))


class TestHy4AndInklingEffortAndWebUI6680(unittest.TestCase):
    def test_clean_and_load_shared_settings_hy4_xhigh_and_max(self):
        self.assertEqual(clean_shared_defaults({"reasoning_effort": "xhigh"}, hy4=True), {"reasoning_effort": "max"})
        self.assertEqual(clean_shared_defaults({"reasoning_effort": "max"}, hy4=True), {"reasoning_effort": "max"})
        self.assertEqual(clean_shared_defaults({"reasoning_effort": "no_think"}, hy4=True), {"reasoning_effort": "none"})
        with self.assertRaises(ValueError):
            clean_shared_defaults({"reasoning_effort": "xhigh"}, hy4=False)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "shared.json"
            p.write_text(json.dumps({"reasoning_effort": "max", "temperature": 0.9}), encoding="utf-8")
            self.assertEqual(load_shared_settings(p, hy4=True), {"reasoning_effort": "max", "temperature": 0.9})

        tok = ByteTokenizer()
        eng = MockEngine(tok, "ok", max_context=4096)
        eng.hy4_engine = True
        svc = Service(eng, tok, ChatTemplate(TERMINAL / "serve" / "chat_template.jinja"), model_name="hy4-flash-sq-64")
        svc.set_family("hy4")
        self.addCleanup(lambda: svc.set_family(None))
        self.assertEqual(svc.set_shared({"reasoning_effort": "max"}), {"reasoning_effort": "max"})
        self.assertEqual(svc.set_shared({"reasoning_effort": "xhigh"}), {"reasoning_effort": "max"})

        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        with urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/health", timeout=15) as r:
            health = json.loads(r.read().decode("utf-8"))
        self.assertEqual(health["family"], "hy4")

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_web_ui_thinking_selector_for_hy4_inkling_and_qwen(self):
        html = (WEB / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="thinking-field"', html)
        self.assertIn('id="show-thinking-row"', html)
        self.assertIn('data-v="max"', html)

        src = (WEB / "app.js").read_text(encoding="utf-8")
        start = src.index("let health =")
        end = src.index("// the texts that depend", start)
        helper_js = src[start:end]
        shared_start = src.index("function sharedDefaults")
        shared_end = src.index("async function saveShared", shared_start)
        shared_js = src[shared_start:shared_end]

        script = (
            "function samplingUntouched() { return true; }\n"
            "function projectionLoaded() { return false; }\n"
            "const els = {\n"
            "  'thinking-field': {hidden: false},\n"
            "  'show-thinking-row': {hidden: false},\n"
            "  's-thinking': {\n"
            "    classes: new Set(),\n"
            "    classList: {toggle(c, on) { if (on) this._s.add(c); else this._s.delete(c); }, _s: null},\n"
            "    children: ['none','low','medium','high','max'].map(v => ({dataset:{v}, hidden: v==='max'})),\n"
            "  },\n"
            "};\n"
            "els['s-thinking'].classList._s = els['s-thinking'].classes;\n"
            "const $ = (id) => els[id];\n"
            + helper_js + "\n" + shared_js + "\n"
            "const out = {};\n"
            "health = {model:'hy4-flash-sq-64', family:'hy4'};\n"
            "syncThinkingUI();\n"
            "out.hy4 = {\n"
            "  fieldHidden: els['thinking-field'].hidden,\n"
            "  showHidden: els['show-thinking-row'].hidden,\n"
            "  seg3: els['s-thinking'].classes.has('seg--3'),\n"
            "  visible: els['s-thinking'].children.filter(b => !b.hidden).map(b => b.dataset.v),\n"
            "  map: ['none','low','medium','high','xhigh','max'].map(v => thinkingForFamily(v)),\n"
            "  sharedMax: sharedDefaults({thinking:'max', temperature:1, top_p:0.95, top_k:20}),\n"
            "  sharedXhigh: sharedDefaults({thinking:'xhigh', temperature:1, top_p:0.95, top_k:20}),\n"
            "};\n"
            "health = {model:'inkling-2-mini-sq-20', family:'inkling2'};\n"
            "syncThinkingUI();\n"
            "out.inkling = {\n"
            "  fieldHidden: els['thinking-field'].hidden,\n"
            "  showHidden: els['show-thinking-row'].hidden,\n"
            "  effort: thinkingForFamily('high'),\n"
            "  shared: sharedDefaults({thinking:'high', temperature:1, top_p:0.95, top_k:20}),\n"
            "};\n"
            "health = {model:'qwen3.8-flash-next', family:null};\n"
            "syncThinkingUI();\n"
            "out.qwen = {\n"
            "  fieldHidden: els['thinking-field'].hidden,\n"
            "  seg3: els['s-thinking'].classes.has('seg--3'),\n"
            "  visible: els['s-thinking'].children.filter(b => !b.hidden).map(b => b.dataset.v),\n"
            "  fromMax: thinkingForFamily('max'),\n"
            "};\n"
            "console.log(JSON.stringify(out));\n"
        )
        res = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30)
        self.assertEqual(res.returncode, 0, res.stderr)
        got = json.loads(res.stdout)
        self.assertFalse(got["hy4"]["fieldHidden"])
        self.assertFalse(got["hy4"]["showHidden"])
        self.assertTrue(got["hy4"]["seg3"])
        self.assertEqual(got["hy4"]["visible"], ["none", "high", "max"])
        self.assertEqual(got["hy4"]["map"], ["none", "high", "high", "high", "max", "max"])
        self.assertEqual(got["hy4"]["sharedMax"], {"reasoning_effort": "max"})
        self.assertEqual(got["hy4"]["sharedXhigh"], {"reasoning_effort": "max"})

        # Inkling-2-Mini has its own thinking levels (none = 0, low 0.2, medium 0.7, high 0.9): the selector is shown, like Qwen's
        self.assertFalse(got["inkling"]["fieldHidden"])
        self.assertFalse(got["inkling"]["showHidden"])
        self.assertEqual(got["inkling"]["effort"], "high")
        self.assertEqual(got["inkling"]["shared"], {"reasoning_effort": "high"})

        self.assertFalse(got["qwen"]["fieldHidden"])
        self.assertFalse(got["qwen"]["seg3"])
        self.assertEqual(got["qwen"]["visible"], ["none", "low", "medium", "high"])
        self.assertEqual(got["qwen"]["fromMax"], "high")


if __name__ == "__main__":
    unittest.main()
