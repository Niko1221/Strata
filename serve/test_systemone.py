"""serve/test_systemone.py - POST /v1/systemone: jev decision endpoint, M3.0 (mock engine, no GPU).

    python -m unittest serve.test_systemone -v

The endpoint renders the frozen jev system+state prompt (SYSTEM_V1), appends each question's
suffix, decodes once per branch, reads the final-row logits, and answers via the jev readout
math (log_softmax_masked -> label_logscore -> softmax).
"""
from __future__ import annotations

import json
import math
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# jevcpp's frozen decision prompt (DECISIONS.md D3; do not edit).
SYSTEM_V1 = ("You are a precise decision engine. Read the state, then answer the question "
             "with a single label and nothing else.")


# ---- the jev readout math (jevref/jmath.py, M1 identity) ----

def log_softmax_masked(logits, usable=None):
    if usable is None:
        usable = [True] * len(logits)
    mx = -math.inf
    for z, u in zip(logits, usable):
        if u and z > mx:
            mx = z
    acc = 0.0
    for z, u in zip(logits, usable):
        if u:
            acc += math.exp(z - mx)
    lse = math.log(acc) + mx
    return [z - lse if u else -math.inf for z, u in zip(logits, usable)], lse


def label_logscore(logps, variants):
    best = -math.inf
    for vs in variants:
        for t in vs:
            if logps[t] > best:
                best = logps[t]
    acc = 0.0
    for vs in variants:
        for t in vs:
            acc += math.exp(logps[t] - best)
    return math.log(acc) + best


def softmax(scores):
    mx = max(scores)
    e = [math.exp(x - mx) for x in scores]
    t = sum(e)
    return [x / t for x in e]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def confidence(p):
    k = len(p)
    if k < 2:
        return 0.0
    v = (k * max(p) - 1.0) / (k - 1)
    return min(1.0, max(0.0, v))


class DecisionEngine(MockEngine):
    """MockEngine plus a scripted final-row logits table {label_token: logit}.  In the real
    engine the row is the final position's full-vocab logits (Verifier::head_logits_, the
    window_logprobs path, served by BROW); here a dict over the label tokens stands in, so
    the endpoint's readout math is exercised exactly.  ByteTokenizer: id = byte value, so
    "No"->[78], "Yes"->[89], "A"->[65], "0"->[48]."""

    ROW = {78: 0.0, 89: 4.0,          # noul: No, Yes
           65: 0.0, 66: 4.0, 67: -1.0,   # choice: A, B, C
           48: 0.0, 49: 4.0, 50: -1.0}   # score: 0, 1, 2

    def __init__(self, tokenizer, script, max_context=32768):
        super().__init__(tokenizer, script, max_context=max_context)
        self.prompts: list[list[int]] = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)

    def final_row_logits(self, prompt):
        return dict(self.ROW)


def start_server():
    tok = ByteTokenizer()
    engine = DecisionEngine(tok, "A")
    svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    httpd = serve(svc, port=0)
    return engine, httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


from serve.server import StrataEngine  # noqa: E402
from serve.test_slots import FakeEngine  # noqa: E402


class BrowWire(unittest.TestCase):
    """StrataEngine.final_row_logits: BROW <tokens> -> BROWD <tok:logit>..., between requests."""

    @classmethod
    def setUpClass(cls):
        eng = FakeEngine()
        eng.script = ["BROWD 2 78:0.5 89:4.25\n"]
        cls.eng = eng
        cls.svc = Service(eng, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_brow_roundtrip(self):
        row = self.eng.final_row_logits([1, 2, 3], [78, 89])
        self.assertEqual(row, {78: 0.5, 89: 4.25})
        self.assertEqual(self.eng.proc.sent[-1], "BROW 78 89\n")

    def test_brow_answer_beyond_requested_tokens_fails_closed(self):
        self.eng.script = ["BROWD 2 78:0.5 999:4.0\n"]      # 999 was never asked for
        try:
            self.eng.final_row_logits([1, 2, 3], [78])
            self.fail("expected EngineDied")
        except Exception as e:
            self.assertIn("BROWD", str(e))

    def test_brow_nonfinite_fails_closed(self):
        self.eng.script = ["BROWD 1 78:nan\n"]
        try:
            self.eng.final_row_logits([1], [78])
            self.fail("expected EngineDied")
        except Exception as e:
            self.assertIn("BROWD", str(e))


class SystemOneValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine, cls.httpd, cls.base = start_server()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_missing_state_422(self):
        st, body = post(self.base, "/v1/systemone",
                        {"questions": {"q1": {"type": "noul", "instructions": "Is it on?"}}})
        self.assertEqual(st, 422)
        self.assertEqual(body["error"]["field"], "state")

    def test_missing_questions_422(self):
        st, body = post(self.base, "/v1/systemone", {"state": {"k": 1}})
        self.assertEqual(st, 422)
        self.assertEqual(body["error"]["field"], "questions")

    def test_bad_qtype_422(self):
        st, body = post(self.base, "/v1/systemone", {"state": {"k": 1}, "questions": {
            "q1": {"type": "essay", "instructions": "Write."}}})
        self.assertEqual(st, 422)
        self.assertIn("type", body["error"]["field"])

    def test_choice_needs_two_options_422(self):
        st, body = post(self.base, "/v1/systemone", {"state": {"k": 1}, "questions": {
            "q1": {"type": "choice", "instructions": "Pick.", "criteria": {"only": 1}}}})
        self.assertEqual(st, 422)
        self.assertEqual(body["error"]["field"], "questions.q1.criteria")

    def test_empty_questions_422(self):
        st, body = post(self.base, "/v1/systemone", {"state": {"k": 1}, "questions": {}})
        self.assertEqual(st, 422)
        self.assertEqual(body["error"]["field"], "questions")


class SystemOneAnswers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine, cls.httpd, cls.base = start_server()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_noul_answer(self):
        st, body = post(self.base, "/v1/systemone", {"state": {"k": 1}, "questions": {
            "q1": {"type": "noul", "instructions": "Is it on?"}}})
        self.assertEqual(st, 200)
        ans = body["answers"]["q1"]
        self.assertEqual(ans["type"], "noul")
        logps, _ = log_softmax_masked([0.0, 4.0, -1.0])
        exp_d = logps[1] - logps[0]
        self.assertAlmostEqual(ans["noul"], sigmoid(exp_d), places=6)

    def test_choice_answer(self):
        st, body = post(self.base, "/v1/systemone", {"state": {"k": 1}, "questions": {
            "q1": {"type": "choice", "instructions": "Pick one.",
                   "criteria": {"alpha": 1, "beta": 2, "gamma": 3}}}})
        self.assertEqual(st, 200)
        ans = body["answers"]["q1"]
        self.assertEqual(ans["type"], "choice")
        logps, _ = log_softmax_masked([0.0, 4.0, -1.0])
        p = softmax([logps[i] for i in range(3)])
        self.assertEqual(ans["choice"], 1)
        for got, exp in zip(ans["probabilities"], p):
            self.assertAlmostEqual(got, exp, places=6)
        self.assertAlmostEqual(ans["confidence"], confidence(p), places=6)

    def test_score_answer(self):
        st, body = post(self.base, "/v1/systemone", {"state": {"k": 1}, "questions": {
            "q1": {"type": "score", "instructions": "Rate.",
                   "criteria": ["low", "mid", "high"]}}})
        self.assertEqual(st, 200)
        ans = body["answers"]["q1"]
        self.assertEqual(ans["type"], "score")
        logps, _ = log_softmax_masked([0.0, 4.0, -1.0])
        p3 = softmax([logps[i] for i in range(3)])
        sc = sum(i * pi for i, pi in enumerate(p3))
        self.assertAlmostEqual(ans["score"], sc, places=6)

    def test_render_uses_frozen_system_prompt(self):
        post(self.base, "/v1/systemone", {"state": {"k": 1}, "questions": {
            "q1": {"type": "noul", "instructions": "Is it on?"}}})
        self.assertTrue(self.engine.prompts)
        text = self.engine.tok.decode(self.engine.prompts[-1])
        self.assertIn(SYSTEM_V1, text)
        self.assertIn("Is it on?", text)

    def test_same_state_questions_share_prefix(self):
        post(self.base, "/v1/systemone", {"state": {"hp": 7}, "questions": {
            "q1": {"type": "noul", "instructions": "On?"},
            "q2": {"type": "noul", "instructions": "Hot?"}}})
        self.assertGreaterEqual(len(self.engine.prompts), 2)
        p1, p2 = self.engine.prompts[-2], self.engine.prompts[-1]
        common = 0
        for a, b in zip(p1, p2):
            if a != b:
                break
            common += 1
        self.assertGreater(common, 4, "same-state questions must share a rendered prefix")


if __name__ == "__main__":
    unittest.main()
