"""Automatic strata_prefix through Service.prepare. Mock engine, no GPU.

Two requests share a long document and neither one sends strata_prefix. The second prompt's
ids match a prepare() that never pins, and the request carries the pin the trie found.

    python -m unittest serve.test_auto_prefix -v
"""
from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace

from serve.frontend import ChatTemplate
from serve.pin_registry import AutoPrefix, PinRegistry
from serve.prefix_trie import PrefixTrie
from serve.server import ByteTokenizer, MockEngine, Service

ROOT = Path(__file__).resolve().parents[1]
MIN = 300          # above the chat template's own header, so only the document is worth a pin


def detector():
    return AutoPrefix(PrefixTrie(ttl_s=None, idle_inserts=None),
                      PinRegistry(min_tokens=MIN, max_pins=2), enabled=True)


class AutoPrefixPrepare(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.template = ChatTemplate(ROOT / "serve/chat_template.jinja")

    def service(self, enabled=True):
        engine = MockEngine(self.tok, "ok", max_context=4096)
        svc = Service(engine, self.tok, self.template)
        svc.auto_prefix = detector() if enabled else AutoPrefix(enabled=False)
        return svc

    def msgs(self, doc, question):
        return [{"role": "user", "content": doc + question}]

    def test_second_request_pins_and_ids_match_the_unpinned_prepare(self):
        doc = "DOC " * 80
        on = self.service()
        off = self.service(enabled=False)
        first = self.msgs(doc, " Which word repeats?")
        second = self.msgs(doc, " How many times?")
        req = {}
        ids1 = on.prepare(first, None, {}, 16, req=req)[0]
        self.assertNotIn("strata_prefix", req)
        self.assertEqual(ids1, off.prepare(first, None, {}, 16)[0])
        buf = io.StringIO()
        req2 = {}
        with contextlib.redirect_stdout(buf):
            ids2 = on.prepare(second, None, {}, 16, req=req2)[0]
        self.assertEqual(ids2, off.prepare(second, None, {}, 16)[0])
        n = req2["strata_prefix"]["tokens"]
        self.assertGreaterEqual(n, MIN)
        self.assertLess(n, len(ids2) - 1)
        self.assertEqual(ids2[:n], ids1[:n])
        log = buf.getvalue()
        self.assertIn("strata_prefix auto:", log)
        self.assertIn("client did not set strata_prefix", log)

    def test_disabled_leaves_the_request_unchanged(self):
        svc = self.service(enabled=False)
        req = {"temperature": 0}
        ids = svc.prepare(self.msgs("DOC " * 80, " q"), None, {}, 16, req=req)[0]
        self.assertEqual(req, {"temperature": 0})
        self.assertEqual(ids, svc.prepare(self.msgs("DOC " * 80, " q"), None, {}, 16)[0])

    def test_client_prefix_is_not_replaced(self):
        svc = self.service()
        doc = "DOC " * 80
        svc.prepare(self.msgs(doc, " one"), None, {}, 16, req={})
        req = {"strata_prefix": {"tokens": 10}}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            svc.prepare(self.msgs(doc, " two"), None, {}, 16, req=req)
        self.assertEqual(req["strata_prefix"], {"tokens": 10})
        self.assertNotIn("strata_prefix auto:", buf.getvalue())

    def test_budget_refuses_a_third_distinct_pin(self):
        svc = self.service()
        docs = ("DOC " * 80, "ALT " * 80, "ZZZ " * 80)
        pinned = []
        for doc in docs:
            svc.prepare(self.msgs(doc, " first"), None, {}, 16, req={})
            req = {}
            with contextlib.redirect_stdout(io.StringIO()):
                svc.prepare(self.msgs(doc, " second"), None, {}, 16, req=req)
            pinned.append(req.get("strata_prefix"))
        self.assertIn("tokens", pinned[0])
        self.assertIn("tokens", pinned[1])
        self.assertIsNone(pinned[2])

    def test_a_new_engine_process_is_not_matched(self):
        svc = self.service()
        doc = "DOC " * 80
        svc.prepare(self.msgs(doc, " one"), None, {}, 16, req={})
        with contextlib.redirect_stdout(io.StringIO()):
            svc.prepare(self.msgs(doc, " two"), None, {}, 16, req={})
        svc.engine.proc = SimpleNamespace(pid=1)          # the process the trie was built against is gone
        req = {}
        with contextlib.redirect_stdout(io.StringIO()):
            svc.prepare(self.msgs(doc, " three"), None, {}, 16, req=req)
        self.assertNotIn("strata_prefix", req)


if __name__ == "__main__":
    unittest.main()
