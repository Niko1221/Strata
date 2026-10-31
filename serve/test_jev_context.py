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
