"""serve/test_jev_golden.py - the golden battery against /v1/systemone (mock engine here; the
live-engine variant is the same battery over a --jev-brow engine, run on the rig).

    python3 -m unittest serve.test_jev_golden -v

Asserts only INVARIANTS (type correctness, probability sums, bounds, determinism on repeat) -
never model-specific answers; exact probabilities are pinned by the live golden run instead.
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service, serve  # noqa: E402
from serve.jev_golden import check_invariants, load_battery  # noqa: E402
from serve.test_systemone import DecisionEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class GoldenBattery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        engine = DecisionEngine(tok, "A")
        cls.svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_battery_invariants(self):
        items = load_battery()
        self.assertEqual(len(items), 12)
        for i, item in enumerate(items):
            with self.subTest(item=i):
                st, body = post(self.base, "/v1/systemone", item)
                self.assertEqual(st, 200, f"item {i}: {body}")
                bad = check_invariants(item, body.get("answers", {}))
                self.assertEqual(bad, [], f"item {i} violations: {bad}")

    def test_deterministic_on_repeat(self):
        items = load_battery()
        for i, item in enumerate(items):
            with self.subTest(item=i):
                _, a = post(self.base, "/v1/systemone", item)
                _, b = post(self.base, "/v1/systemone", item)
                self.assertEqual(a.get("answers"), b.get("answers"),
                                 f"item {i}: repeat gave different answers")


if __name__ == "__main__":
    unittest.main()


class NoulPromptDescriptions(unittest.TestCase):
    """2026-10-15: the image suite caught Yes showing No's meaning (a<=0 Platt fit).
    Pin the EXACT noul question-block text with criteria present."""

    def test_yes_means_true_no_means_false(self):
        from serve import jev as J
        qdefs, _ = J.parse_questions({"q": {
            "type": "noul", "instructions": "Is there a large green circle?",
            "criteria": {"true": "a large green circle is visible",
                         "false": "no large green circle is visible"}}})
        block = J.question_block(qdefs[0])
        self.assertIn("Yes means: a large green circle is visible\n", block)
        self.assertIn("No means: no large green circle is visible\n", block)
        self.assertNotIn(": :", block)          # the doubled colon is gone
        i_yes = block.index("Yes means")
        i_no = block.index("No means")
        self.assertLess(i_yes, i_no)            # Yes first (donor order)

    def test_choice_descriptions_keep_colon_shape(self):
        from serve import jev as J
        qdefs, _ = J.parse_questions({"q": {
            "type": "choice", "instructions": "Pick",
            "criteria": {"a": "the a option", "b": None}}})
        block = J.question_block(qdefs[0])
        self.assertIn("A. a: the a option\n", block)
        self.assertIn("B. b\n", block)


class LmEvalSmokeInvariants(unittest.TestCase):
    """lm-evaluation-harness guardrails (survey 2026-10-15): per-choice readouts
    must be distinct; identical values are the signature of a broken
    context/continuation split (the BOS-double-count class of bug). Catches
    wiring breaks that argmax tests pass through. Runs the golden battery live
    against the mock engine and inspects every answer."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        engine = DecisionEngine(tok, "A")
        cls.svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_battery_answers_have_distinct_normalized_probabilities(self):
        for i, item in enumerate(load_battery()):
            st, body = post(self.base, "/v1/systemone", item)
            self.assertEqual(st, 200)
            for qid, a in body.get("answers", {}).items():
                pr = a.get("probabilities") or [1 - a["noul"], a["noul"]]
                self.assertTrue(any(p != pr[0] for p in pr),
                                f"item {i}/{qid}: all probabilities identical — readout split broken")
                self.assertTrue(all(0.0 < p < 1.0 for p in pr),
                                f"item {i}/{qid}: probability at 0 or 1 — degenerate readout")
                self.assertAlmostEqual(sum(pr), 1.0, places=5,
                                       msg=f"item {i}/{qid}: probabilities do not sum to 1")
                self.assertIn("margin", a, f"item {i}/{qid}: margin missing")
                self.assertGreaterEqual(a["margin"], 0.0)
