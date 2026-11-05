"""serve/test_jev_context.py - the opt-in request `context` field (mock engine, no GPU).

    python -m unittest serve.test_jev_context -v

`context` (string or list of strings) renders verbatim between the state and the question
block in EVERY field's prompt - per-field, V-mode branches, and the consolidated rows path.
Absent or empty: the render is byte-identical to the donor's frozen shape (regression guard
below). Wrong type: 422 with field "context" (fail closed, like every other request field).
"""
from __future__ import annotations

import json
import os as _os
# JEV_TEST_MODES_BOOTSTRAP: the suite exercises a deployment that has OPTED IN
# to decisions (chat-only is the default posture); set before any serve.jev_calib
# import resolves MODES at module load.
_os.environ.setdefault("STRATA_JEV_MODES", "chat,decision")

import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import jev  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service, serve  # noqa: E402
from serve.test_systemone import DecisionEngine, post  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

STATE = {"utterance": "I need to order a new card, mine is about to expire."}
CATALOGUE = "Request groups: card, top_up, transfer, cash, other."
TWO_FIELDS = {
    "is_card": {"type": "noul", "instructions": "Does this customer request concern a card?"},
    "group": {"type": "choice", "instructions": "Which group does this request belong to?",
              "criteria": {"card": None, "top_up": None, "transfer": None}},
}


def start_server():
    tok = ByteTokenizer()
    engine = DecisionEngine(tok, "A")
    svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    httpd = serve(svc, port=0)
    return engine, httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def prompts_text(engine) -> list[str]:
    """Every prompt the engine was handed, decoded (ByteTokenizer: id = byte value, specials >= 256)."""
    tok = ByteTokenizer()
    def dec(ids):
        out = []
        for t in ids:
            out.append(tok.SPECIALS[t - 256] if t >= 256 else chr(t))
        return "".join(out)
    return [dec(p) for p in engine.prompts]


class SystemOneLoadGate(unittest.TestCase):
    """B4 (2026-11-05, expert review blocker 4): the route calls svc.load() like every other
    engine path - after an idle unload, systemone restarts the engine and answers 200 instead
    of wedging forever (and last_request_at moves, so the idle timer stops firing under
    steady decision traffic).  Mock-based: alive() is the unload flag, ensure_loaded is the
    restart the real Service would do."""

    def test_unload_then_systemone_answers_200(self):
        tok = ByteTokenizer()
        engine = DecisionEngine(tok, "A")
        engine.unloaded = False

        def alive():
            return not engine.unloaded

        engine.alive = alive                     # loaded() now reports the unload state
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        restarts = {"n": 0}

        def ensure_loaded():
            restarts["n"] += 1
            engine.unloaded = False              # what a real restart achieves

        svc.ensure_loaded = ensure_loaded
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            engine.unloaded = True               # the idle-unload timer fired
            svc.last_request_at = None
            st, body = post(base, "/v1/systemone",
                            {"state": {"k": 1},
                             "questions": {"q": {"type": "noul", "instructions": "on?"}}})
            self.assertEqual(st, 200, body)
            self.assertEqual(restarts["n"], 1)   # the route restarted the engine itself
            self.assertIsNotNone(svc.last_request_at)   # and counted this as activity
        finally:
            httpd.shutdown()
            httpd.server_close()


class ReadoutGates(unittest.TestCase):
    """F5 (B5+M6, 2026-11-05): verify-then-gate.  Source verification first: the only batch
    check near server.py:6977 is inside prompt_progress() - it suppresses PROGRESS REPORTING
    under --batch; nothing on the /v1/systemone route, jev.handle, or between them gates on
    batch, multi or jevpos (grep: zero 'batch' occurrences in serve/jev.py before this fix).
    So the gates are new: 503 with a clear reason on batch/multi, and on jevpos != 1."""

    def _server(self):
        tok = ByteTokenizer()
        engine = DecisionEngine(tok, "A")
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.shutdown)
        self.addCleanup(httpd.server_close)
        return engine, f"http://127.0.0.1:{httpd.server_address[1]}"

    def _req(self):
        return {"state": {"k": 1},
                "questions": {"q": {"type": "noul", "instructions": "on?"}}}

    def test_batch_engine_is_refused_before_any_prompt(self):
        engine, base = self._server()
        engine.batch = 4                       # --batch: slot windows, no main-verifier rows
        engine.prompts.clear()
        st, body = post(base, "/v1/systemone", self._req())
        self.assertEqual(st, 503, body)
        self.assertIn("--batch", body["error"]["message"])
        self.assertEqual(engine.prompts, [])   # refused before a prompt was sent

    def test_multi_engine_is_refused(self):
        engine, base = self._server()
        engine.multi = 2                       # DeepSeek --parallel (server sets batch=multi)
        st, body = post(base, "/v1/systemone", self._req())
        self.assertEqual(st, 503, body)
        self.assertIn("parallel", body["error"]["message"])

    def test_engine_without_jevpos_is_refused(self):
        engine, base = self._server()
        engine.info.pop("jevpos", None)        # started without --jev-brow
        engine.prompts.clear()
        st, body = post(base, "/v1/systemone", self._req())
        self.assertEqual(st, 503, body)
        self.assertIn("--jev-brow", body["error"]["message"])
        self.assertEqual(engine.prompts, [])

    def test_jevpos_one_is_served(self):
        engine, base = self._server()          # DecisionEngine advertises jevpos=1
        st, body = post(base, "/v1/systemone", self._req())
        self.assertEqual(st, 200, body)
        self.assertIn("q", body["answers"])


class ContextDefault(unittest.TestCase):
    """No `context`: the render must be exactly the donor's frozen shape."""

    @classmethod
    def setUpClass(cls):
        cls.engine, cls.httpd, cls.base = start_server()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_absent_context_render_unchanged(self):
        st, body = post(self.base, "/v1/systemone",
                        {"state": STATE, "questions": TWO_FIELDS})
        self.assertEqual(st, 200)
        state_text = jev.render_state(STATE)
        for p in prompts_text(self.engine):
            self.assertIn(state_text + "\n\nQuestion:", p)      # state butts the question: no gap inserted
            self.assertNotIn(CATALOGUE, p)

    def test_empty_string_is_absent(self):
        self.engine.prompts.clear()
        st, body = post(self.base, "/v1/systemone",
                        {"state": STATE, "questions": TWO_FIELDS, "context": ""})
        self.assertEqual(st, 200)
        for p in prompts_text(self.engine):
            self.assertIn(jev.render_state(STATE) + "\n\nQuestion:", p)

    def test_wrong_type_422(self):
        for bad in ({"a": 1}, 17, [CATALOGUE, 3], [None]):
            st, body = post(self.base, "/v1/systemone",
                            {"state": STATE, "questions": TWO_FIELDS, "context": bad})
            self.assertEqual(st, 422, bad)
            self.assertEqual(body["error"]["field"], "context", bad)


class ContextRender(unittest.TestCase):
    """`context` present: verbatim, between the state and the question, in every field."""

    @classmethod
    def setUpClass(cls):
        cls.engine, cls.httpd, cls.base = start_server()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_string_between_state_and_question_every_field(self):
        self.engine.prompts.clear()
        st, body = post(self.base, "/v1/systemone",
                        {"state": STATE, "questions": TWO_FIELDS, "context": CATALOGUE})
        self.assertEqual(st, 200)
        self.assertEqual(set(body["answers"]), {"is_card", "group"})
        ps = prompts_text(self.engine)
        self.assertGreaterEqual(len(ps), 2)                     # one prompt per field (per-field path)
        state_text = jev.render_state(STATE)
        for p in ps:
            self.assertIn(state_text + "\n\n" + CATALOGUE + "\n\nQuestion:", p)

    def test_list_joined_with_newlines(self):
        self.engine.prompts.clear()
        st, body = post(self.base, "/v1/systemone",
                        {"state": STATE, "questions": TWO_FIELDS,
                         "context": ["First line.", "Second line."]})
        self.assertEqual(st, 200)
        for p in prompts_text(self.engine):
            self.assertIn("First line.\nSecond line.", p)

    def test_blank_lines_dropped(self):
        self.engine.prompts.clear()
        st, body = post(self.base, "/v1/systemone",
                        {"state": STATE, "questions": TWO_FIELDS, "context": ["  ", ""]})
        self.assertEqual(st, 200)
        for p in prompts_text(self.engine):
            self.assertIn(jev.render_state(STATE) + "\n\nQuestion:", p)   # all-blank == absent

    def test_vmode_branches_carry_context(self):
        self.engine.prompts.clear()
        st, body = post(self.base, "/v1/systemone",
                        {"state": STATE, "questions": TWO_FIELDS, "context": CATALOGUE},
                        headers={"X-Jev-Mode": "V"})
        self.assertEqual(st, 200)
        ps = prompts_text(self.engine)
        self.assertTrue(ps)
        for p in ps:
            if "Proposed answer:" in p:                          # a V branch prompt
                self.assertIn(CATALOGUE + "\n\nQuestion:", p)


class ContextConsolidated(unittest.TestCase):
    """STRATA_JEV_ROWS=1: one prompt for the whole request - the context rides unit 0."""

    @classmethod
    def setUpClass(cls):
        cls.saved = _os.environ.pop("STRATA_JEV_ROWS", None)
        _os.environ["STRATA_JEV_ROWS"] = "1"
        cls.engine, cls.httpd, cls.base = start_server()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        if cls.saved is None:                                    # env hygiene: never leak into later suites
            _os.environ.pop("STRATA_JEV_ROWS", None)
        else:
            _os.environ["STRATA_JEV_ROWS"] = cls.saved

    def test_consolidated_prompt_carries_context(self):
        st, body = post(self.base, "/v1/systemone",
                        {"state": STATE, "questions": TWO_FIELDS, "context": CATALOGUE})
        self.assertEqual(st, 200)
        ps = prompts_text(self.engine)
        self.assertTrue(ps)
        p0 = ps[0]
        self.assertIn(jev.render_state(STATE) + "\n\n" + CATALOGUE + "\n\nQuestion:", p0)


if __name__ == "__main__":
    unittest.main()
