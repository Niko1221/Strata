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
# JEV_TEST_MODES_BOOTSTRAP: the suite exercises a deployment that has OPTED IN
# to decisions (chat-only is the default posture since the flip); set before
# any serve.jev_calib import resolves MODES at module load.
import os as _os
_os.environ.setdefault("STRATA_JEV_MODES", "chat,decision")
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
           48: 0.0, 49: 4.0, 50: -1.0, 51: -2.0}   # score: 0, 1, 2, 3

    def __init__(self, tokenizer, script, max_context=32768):
        super().__init__(tokenizer, script, max_context=max_context)
        self.prompts: list[list[int]] = []
        self.pins: list[dict] = []           # the sampling dict of each generate (M3.1 pin)
        self.rows_read: list[int] = []       # the prompt positions BROWAT was asked for (fix B)
        self.info = {"jevpos": 1}            # the engine build advertises row capture (fix B)

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        self.pins.append(dict(sampling or {}))
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)

    def final_row_logits(self, ids, tokens):
        # the real engine answers only the asked tokens (BROW); the mock mirrors that
        return {t: self.ROW[t] for t in tokens if t in self.ROW}

    def row_logits_at(self, row, tokens):
        # fix B (jevpos): the row captured at that prompt position.  This table is row-independent, so the
        # mock proves the WIRING (one read, one row per field, same math) - the row's real value is live.
        self.rows_read.append(int(row))
        return {t: self.ROW[t] for t in tokens if t in self.ROW}


def start_server():
    tok = ByteTokenizer()
    engine = DecisionEngine(tok, "A")
    svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    httpd = serve(svc, port=0)
    return engine, httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def post(base, path, body, headers=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


from serve.server import StrataEngine  # noqa: E402
from serve.test_slots import FakeEngine, FakeProc  # noqa: E402


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

    def tearDown(self):
        """A fail-closed case ends the engine, and FakeProc.kill() is permanent (killed stays True for the
        life of that stand-in), so hand the next test a live one and an empty line queue - otherwise the
        test that follows a kill waits for a reply the scripted player refuses to send."""
        self.eng.proc = FakeProc(self.eng)
        self.eng.ended = False
        while not self.eng.lines.empty():
            self.eng.lines.get_nowait()

    def test_brow_roundtrip(self):
        self.eng.script = ["BROWD 2 78:0.5 89:4.25\n"]      # earlier tests replaced it
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

    # ---- M3.1: BROWAT reads any row of the same window, not only the final one ----

    def test_browat_roundtrip_names_the_row(self):
        self.eng.script = ["BROWATD 2 78:0.5 89:4.25\n"]
        row = self.eng.row_logits_at(37, [78, 89])
        self.assertEqual(row, {78: 0.5, 89: 4.25})
        self.assertEqual(self.eng.proc.sent[-1], "BROWAT 37 78 89\n")

    def test_browat_row_zero_is_sent_verbatim(self):
        # row 0 is a real row (the window's first position), not a sentinel: it must reach the engine
        # as "BROWAT 0 ..." and never be confused with "the final row" (row 0 of a 1-token window is
        # both, which is why the wire carries the number rather than a flag)
        self.eng.script = ["BROWATD 1 78:1.0\n"]
        self.assertEqual(self.eng.row_logits_at(0, [78]), {78: 1.0})
        self.assertEqual(self.eng.proc.sent[-1], "BROWAT 0 78\n")

    def test_browat_row_outside_the_window_fails_closed(self):
        # the engine answers `ERR BROWAT: row 4000 is outside the last window (98 rows)`; the server
        # must not silently take a neighbouring row - it ends the engine (EngineDied) instead
        self.eng.script = ["ERR BROWAT: row 4000 is outside the last window (98 rows)\n"]
        try:
            self.eng.row_logits_at(4000, [78])
            self.fail("expected EngineDied")
        except Exception as e:
            self.assertIn("BROWAT", str(e))

    def test_browat_malformed_answer_fails_closed(self):
        self.eng.script = ["BROWATD 2 78:0.5\n"]            # says 2, carries 1
        try:
            self.eng.row_logits_at(37, [78, 89])
            self.fail("expected EngineDied")
        except Exception as e:
            self.assertIn("BROWATD", str(e))

    def test_browrows_reports_the_last_window(self):
        self.eng.script = ["BROWROWSD 98\n"]
        self.assertEqual(self.eng.brow_rows(), 98)
        self.assertEqual(self.eng.proc.sent[-1], "BROWROWS\n")

    def test_browrows_malformed_answer_fails_closed(self):
        self.eng.script = ["BROWROWSD 98 99\n"]             # one integer, or nothing
        try:
            self.eng.brow_rows()
            self.fail("expected EngineDied")
        except Exception as e:
            self.assertIn("BROWROWSD", str(e))


class StatePin(unittest.TestCase):
    """M3.1 fix C (state parking): N fields on one state send pin=<shared prefix>, so the engine reads the
    state once and every later field reads only its own question turn.

    The mock engine has no cache, so what these tests prove is the CONTRACT: the key is sent, its value is
    the real shared token prefix of the rendered prompts, one-field requests never send it, the env switch
    turns it off, and the answers are identical with and without it.  What the pin BUYS is a live
    measurement (jevport.log 'request prompt P cached C'), not a unit test.
    """

    @classmethod
    def setUpClass(cls):
        cls.engine, cls.httpd, cls.base = start_server()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self._old = _os.environ.get("STRATA_JEV_PIN")
        _os.environ.pop("STRATA_JEV_PIN", None)          # default: pinning is on
        self.engine.prompts.clear()
        self.engine.pins.clear()

    def tearDown(self):
        if self._old is None:
            _os.environ.pop("STRATA_JEV_PIN", None)
        else:
            _os.environ["STRATA_JEV_PIN"] = self._old

    THREE = {"q1": {"type": "noul", "instructions": "Is it on?"},
             "q2": {"type": "choice", "instructions": "Pick one.",
                    "criteria": {"alpha": 1, "beta": 2}},
             "q3": {"type": "score", "instructions": "Rate it.", "criteria": ["low", "mid", "high"]}}

    def _post(self, questions, headers=None):
        return post(self.base, "/v1/systemone", {"state": {"hp": 7}, "questions": questions}, headers)

    def test_multi_field_pins_the_shared_prefix(self):
        st, _ = self._post(self.THREE)
        self.assertEqual(st, 200)
        self.assertEqual(len(self.engine.prompts), 3, "still one request per field")
        pins = [p.get("pin") for p in self.engine.pins]
        self.assertTrue(all(isinstance(x, int) and x > 0 for x in pins), pins)
        self.assertEqual(len(set(pins)), 1, "one shared prefix, so one pin number")
        pin = pins[0]
        for ids in self.engine.prompts:
            self.assertLessEqual(pin, len(ids), "a pin longer than the prompt would be a lie")
        self.assertEqual(self.engine.prompts[0][:pin], self.engine.prompts[1][:pin],
                         "the pin must name the same tokens in every field's prompt")
        self.assertEqual(self.engine.prompts[0][:pin], self.engine.prompts[2][:pin])

    def test_answers_are_unchanged_by_the_pin(self):
        _, on = self._post(self.THREE)
        _os.environ["STRATA_JEV_PIN"] = "0"
        _, off = self._post(self.THREE)
        self.assertEqual(on["answers"], off["answers"],
                         "a caching hint must not move a single answer")

    def test_one_field_sends_no_pin(self):
        self._post({"q1": {"type": "noul", "instructions": "Is it on?"}})
        self.assertNotIn("pin", self.engine.pins[-1])

    def test_env_off_sends_no_pin(self):
        _os.environ["STRATA_JEV_PIN"] = "0"
        self._post(self.THREE)
        self.assertFalse(any("pin" in p for p in self.engine.pins), self.engine.pins)

    def test_v_mode_pins_the_option_head(self):
        body = {"q1": {"type": "choice", "instructions": "Pick one.",
                       "criteria": {"alpha": 1, "beta": 2, "gamma": 3}}}
        st, _ = self._post(body, headers={"X-Jev-Mode": "V"})
        self.assertEqual(st, 200)
        pins = [p.get("pin") for p in self.engine.pins]
        self.assertEqual(len(pins), 3, "one branch per option")
        self.assertTrue(all(isinstance(x, int) and x > 0 for x in pins), pins)
        pin = pins[0]
        self.assertEqual(self.engine.prompts[0][:pin], self.engine.prompts[1][:pin],
                         "the option branches share the state + head prefix")


class RowReadout(unittest.TestCase):
    """M3.1 fix B (jevpos): one prompt read, one head row kept per field.

    The mock's logits are row-independent, so these tests prove the CONTRACT - one generate, the cue
    positions named in it, one row read per field, the same answers - not what the rows contain.  The
    latency and the accuracy shift are live measurements (rowreadout_probe.py).
    """

    @classmethod
    def setUpClass(cls):
        cls.engine, cls.httpd, cls.base = start_server()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self._old = _os.environ.get("STRATA_JEV_ROWS")
        _os.environ["STRATA_JEV_ROWS"] = "1"
        self._old_info = dict(self.engine.info)
        self.engine.prompts.clear()
        self.engine.pins.clear()
        self.engine.rows_read.clear()

    def tearDown(self):
        if self._old is None:
            _os.environ.pop("STRATA_JEV_ROWS", None)
        else:
            _os.environ["STRATA_JEV_ROWS"] = self._old
        self.engine.info.clear()
        self.engine.info.update(self._old_info)

    TWO = {"q1": {"type": "noul", "instructions": "Is it on?"},
           "q2": {"type": "choice", "instructions": "Pick one.", "criteria": {"alpha": 1, "beta": 2}}}

    def _post(self, questions):
        return post(self.base, "/v1/systemone", {"state": {"hp": 7}, "questions": questions})

    def test_two_fields_cost_one_read(self):
        st, body = self._post(self.TWO)
        self.assertEqual(st, 200)
        self.assertEqual(len(self.engine.prompts), 1, "one prompt, not one per field")
        self.assertEqual(body["usage"]["prompt_tokens"], len(self.engine.prompts[0]))
        rows = self.engine.pins[0].get("jevpos")
        self.assertIsInstance(rows, list)
        self.assertEqual(len(rows), 2, "one cue position per field")
        self.assertLess(rows[0], rows[1])
        for r in rows:
            self.assertLess(r, len(self.engine.prompts[0]), "a row must be inside the prompt that ran")
        self.assertEqual(self.engine.rows_read, rows, "each named row is the one read")
        self.assertIs(self.engine.pins[0].get("strata_checkpoint"), False,
                      "one-shot: nothing extends this turn, so no checkpoint")

    def test_answers_match_the_per_question_path(self):
        _, on = self._post(self.TWO)
        _os.environ.pop("STRATA_JEV_ROWS", None)
        _, off = self._post(self.TWO)
        self.assertEqual(on["answers"], off["answers"],
                         "the same row must produce the same answer on either path")

    def test_env_off_keeps_the_per_question_path(self):
        _os.environ.pop("STRATA_JEV_ROWS", None)
        st, _ = self._post(self.TWO)
        self.assertEqual(st, 200)
        self.assertEqual(len(self.engine.prompts), 2)
        self.assertFalse(any("jevpos" in p for p in self.engine.pins))

    def test_an_engine_without_jevpos_is_not_asked_for_rows(self):
        self.engine.info.pop("jevpos", None)
        st, _ = self._post(self.TWO)
        self.assertEqual(st, 200)
        self.assertEqual(len(self.engine.prompts), 2, "no capability, no one-prompt path")
        self.assertFalse(any("jevpos" in p for p in self.engine.pins))

    def test_one_field_never_takes_the_one_prompt_path(self):
        st, _ = self._post({"q1": {"type": "noul", "instructions": "Is it on?"}})
        self.assertEqual(st, 200)
        self.assertFalse(any("jevpos" in p for p in self.engine.pins))


class SamplingKeys(unittest.TestCase):
    """The wire keys jev builds: a real position list only, never a malformed one."""

    def test_jevpos_list(self):
        self.assertEqual(StrataEngine.sampling_keys({"jevpos": [3, 40, 97]}), " jevpos=3,40,97")

    def test_jevpos_rejects_junk(self):
        for bad in ([], [-1], ["3"], [3, "x"], [True], None):
            self.assertEqual(StrataEngine.sampling_keys({"jevpos": bad}), "", bad)

    def test_pin_key(self):
        self.assertEqual(StrataEngine.sampling_keys({"pin": 63}), " pin=63")
        self.assertEqual(StrataEngine.sampling_keys({"pin": 0}), "")


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
        p = softmax([logps[0], logps[1]])
        # M3.2 donor order: round 4dp, renormalize (server.cpp:1134-1142)
        r = [round(x * 10000.0) / 10000.0 for x in p]
        t = sum(r)
        self.assertAlmostEqual(ans["noul"], r[1] / t, places=6)
        self.assertEqual(ans.get("calib"), "identity")

    def test_choice_answer(self):
        st, body = post(self.base, "/v1/systemone", {"state": {"k": 1}, "questions": {
            "q1": {"type": "choice", "instructions": "Pick one.",
                   "criteria": {"alpha": 1, "beta": 2, "gamma": 3}}}})
        self.assertEqual(st, 200)
        ans = body["answers"]["q1"]
        self.assertEqual(ans["type"], "choice")
        logps, _ = log_softmax_masked([0.0, 4.0, -1.0])
        p = softmax([logps[i] for i in range(3)])
        r = [round(x * 10000.0) / 10000.0 for x in p]     # M3.2: donor 4dp rounding
        t = sum(r)
        e = [x / t for x in r]
        self.assertEqual(ans["choice"], 1)
        for got, exp in zip(ans["probabilities"], e):
            self.assertAlmostEqual(got, exp, places=6)
        self.assertAlmostEqual(ans["confidence"], confidence(e), places=6)

    def test_score_answer(self):
        st, body = post(self.base, "/v1/systemone", {"state": {"k": 1}, "questions": {
            "q1": {"type": "score", "instructions": "Rate.",
                   "criteria": ["low", "mid", "high"]}}})
        self.assertEqual(st, 200)
        ans = body["answers"]["q1"]
        self.assertEqual(ans["type"], "score")
        logps, _ = log_softmax_masked([0.0, 4.0, -1.0])
        p3 = softmax([logps[i] for i in range(3)])
        r = [round(x * 10000.0) / 10000.0 for x in p3]    # M3.2: donor 4dp rounding
        t = sum(r)
        e = [x / t for x in r]
        sc = sum(i * pi for i, pi in enumerate(e))
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
