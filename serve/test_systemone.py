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
from serve import jev  # noqa: E402
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
        self.max_news: list[int] = []        # the max_new of each generate
        self.info = {"jevpos": 1}            # the engine build advertises row capture (fix B)
        self.window_rows = None              # None: the last prompt was read in ONE window
        self.capture_ok = True               # False: the engine kept no rows (a dropped jevpos list)
        self.final_reads = 0                 # how often the final-row (BROW) read was used

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        self.pins.append(dict(sampling or {}))
        self.max_news.append(int(max_new))
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)

    def final_row_logits(self, ids, tokens):
        # the real engine answers only the asked tokens (BROW); the mock mirrors that
        self.final_reads += 1
        return {t: self.ROW[t] for t in tokens if t in self.ROW}

    def captured_rows(self):
        """BROWCAP: how many head rows the last request kept for the positions it named."""
        named = (self.pins[-1].get("jevpos") or []) if self.pins else []
        return len(named) if self.capture_ok else 0

    def brow_rows(self):
        """BROWROWS: the row count of the LAST window.  A prompt the engine chunked is read in more
        than one window, so this is NOT the prompt length then - tests set `window_rows` to say so."""
        if self.window_rows is not None:
            return int(self.window_rows)
        return len(self.prompts[-1]) if self.prompts else 0

    def row_logits_at(self, row, tokens):
        # fix B (jevpos): the row captured at that prompt position.  This table is row-independent, so the
        # mock proves the WIRING (one read, one row per field, same math) - the row's real value is live.
        self.rows_read.append(int(row))
        return {t: self.ROW[t] for t in tokens if t in self.ROW}


class _SeamTok:
    def encode(self, text, parse_special=True):
        return [ord(c) for c in text]


class _BreakingTemplate:
    """Renders a prompt whose FIRST bytes change once the conversation grows - exactly the seam move the
    prefix check exists to catch (a BPE merge or a template quirk shifts every later position by a token)."""

    def render(self, messages, add_generation_prompt=True, enable_thinking=False):
        body = "".join(m["role"] + ":" + m["content"] + "|" for m in messages)
        return ("<<" if len(messages) > 3 else "") + body


class _FakeSvc:
    def __init__(self, template):
        self.tok = _SeamTok()
        self.template = template


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

    # ---- M3.1: BROWCAP reports how many rows the last request kept (the guard the row path verifies) ----

    def test_browcap_roundtrip(self):
        self.eng.script = ["BROWCAPD 2\n"]
        self.assertEqual(self.eng.captured_rows(), 2)
        self.assertEqual(self.eng.proc.sent[-1], "BROWCAP\n")

    def test_browcap_zero_means_no_capture(self):
        self.eng.script = ["BROWCAPD 0\n"]
        self.assertEqual(self.eng.captured_rows(), 0)

    def test_browcap_unexpected_answer_fails_closed(self):
        self.eng.script = ["BROWD 1 78:0.5\n"]           # an answer to a different op
        try:
            self.eng.captured_rows()
            self.fail("expected EngineDied")
        except Exception as e:
            self.assertIn("BROWCAP", str(e))

    def test_browcap_malformed_answer_fails_closed(self):
        self.eng.script = ["BROWCAPD two\n"]
        try:
            self.eng.captured_rows()
            self.fail("expected EngineDied")
        except Exception as e:
            self.assertIn("BROWCAPD", str(e))

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


class ConsolidatedSeam(unittest.TestCase):
    """consolidated_rows() must refuse to hand back row numbers when the later prompt does not extend the
    earlier one byte-for-byte: a row read one token off answers the wrong question while looking fine."""

    Q = [{"id": "q1", "type": "noul", "instructions": "Is it on?", "labels": ["No", "Yes"],
          "descriptions": ["", ""], "render_labels": []},
         {"id": "q2", "type": "noul", "instructions": "Is it off?", "labels": ["No", "Yes"],
          "descriptions": ["", ""], "render_labels": []}]

    def test_a_stable_seam_yields_one_row_per_field(self):
        ids, rows = jev.consolidated_rows(_FakeSvc(_BreakingTemplate()), "STATE", self.Q[:1])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], len(ids) - 1)

    def test_a_moving_seam_returns_none(self):
        self.assertIsNone(jev.consolidated_rows(_FakeSvc(_BreakingTemplate()), "STATE", self.Q),
                          "a seam that moves must not produce row numbers")


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

    def test_later_fields_do_not_park_a_checkpoint(self):
        """A field that resumes from the pinned checkpoint has nothing to park, and parking costs real work:
        measured 2026-10-23 on the 3090, five fields took 1527 ms with the park and 1232 ms without it, on
        byte-identical prompts and identical answers.  The FIRST field must still park - it is the one the
        pin marks, so suppressing it there would leave nothing for the later fields to resume from."""
        self._post(self.THREE)
        keys = self.engine.pins
        self.assertEqual(len(keys), 3)
        self.assertNotIn("strata_checkpoint", keys[0], "field 1 creates the checkpoint the pin marks")
        for k in keys[1:]:
            self.assertIs(k.get("strata_checkpoint"), False, keys)

    def test_no_park_suppression_without_a_pin(self):
        """Without a pin every field is a candidate to be resumed from, so none of them may skip its park."""
        _os.environ["STRATA_JEV_PIN"] = "0"
        self._post(self.THREE)
        self.assertFalse(any("strata_checkpoint" in p for p in self.engine.pins), self.engine.pins)

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
        self.engine.window_rows = None
        self.engine.final_reads = 0
        self.engine.capture_ok = True

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

    def test_field_order_is_canonical(self):
        """The caller's key order must not change the prompt.

        One consolidated prompt lets each later field see the earlier ones, so a caller who sent q2 first used to
        get different answers for the same set: measured live on 200 banking77 items, reversing the two fields
        flipped 52 one way against 1 the other (McNemar p=0.0008).  The fix renders the fields in a fixed order,
        so this pins that the same questions make the same prompt and name the same rows either way.
        """
        _, forward = self._post(self.TWO)
        forward_prompt = list(self.engine.prompts[0])
        forward_rows = list(self.engine.pins[0].get("jevpos") or [])
        self.engine.prompts.clear()
        self.engine.pins.clear()
        self.engine.rows_read.clear()

        _, reversed_body = self._post({"q2": self.TWO["q2"], "q1": self.TWO["q1"]})
        self.assertEqual(self.engine.prompts[0], forward_prompt,
                         "the same questions must render the same prompt whoever asks them in either order")
        self.assertEqual(self.engine.pins[0].get("jevpos"), forward_rows,
                         "and name the same cue rows, so the same rows are read")
        self.assertEqual(set(forward["answers"]), {"q1", "q2"})
        self.assertEqual(set(reversed_body["answers"]), {"q1", "q2"},
                         "answers stay keyed by qid, so the reordering is invisible to the caller")

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



    def test_rows_path_falls_back_when_the_engine_kept_no_rows(self):
        """BROWAT addresses the last WINDOW when nothing was captured, so a prompt position would be read
        as a window index - a neighbouring row, silently.  A request whose rows were not kept must fall
        back instead of reading them."""
        self.engine.capture_ok = False
        st, body = self._post(self.TWO)
        self.assertEqual(st, 200)
        self.assertEqual(len(self.engine.prompts), 3, "one consolidated read + one per field")
        self.assertEqual(self.engine.rows_read, [], "no absolute position was read off window addressing")
        self.assertEqual(self.engine.final_reads, 2)
        self.assertEqual(len(body["answers"]), 2)

    def test_a_label_that_is_not_one_token_is_refused_at_the_http_surface(self):
        """#3643's contract: a multi-token label must be a 422 on the question, not a score read off a
        truncated encoding.  The readout encodes each label and takes [:1], so a two-token label would
        otherwise be scored on its first token and look fine."""
        from unittest import mock
        real = ByteTokenizer.encode

        def two_tokens(self, text, parse_special=True):
            ids = real(self, text, parse_special)
            return ids * 2 if text == "A" else ids

        with mock.patch.object(ByteTokenizer, "encode", two_tokens):
            st, body = self._post({"q": {"type": "choice", "instructions": "Pick one.",
                                         "criteria": {"alpha": 1, "beta": 2}}})
        self.assertEqual(st, 422)
        self.assertEqual(body["error"]["field"], "questions.q.criteria")
        self.assertIn("one token", body["error"]["message"])

    def test_a_score_label_that_is_not_one_token_is_refused_too(self):
        """score labels are rendered "0".."9" and read the same way (encode, [:1]), so they need the same
        single-token guarantee as choice; the refusal used to skip them."""
        from unittest import mock
        real = ByteTokenizer.encode

        def two_tokens(self, text, parse_special=True):
            ids = real(self, text, parse_special)
            return ids * 2 if text == "0" else ids

        with mock.patch.object(ByteTokenizer, "encode", two_tokens):
            st, body = self._post({"q": {"type": "score", "instructions": "Rate it.",
                                         "criteria": ["low", "mid", "high"]}})
        self.assertEqual(st, 422)
        self.assertEqual(body["error"]["field"], "questions.q.criteria")
        self.assertIn("one token", body["error"]["message"])

    def test_more_than_32_fields_never_take_the_row_path(self):
        """The engine drops a jevpos list longer than 32, and a dropped list means no capture: the path must
        not run at all rather than answer from window rows."""
        many = {f"q{i}": {"type": "noul", "instructions": f"Is {i} set?"} for i in range(33)}
        st, body = self._post(many)
        self.assertEqual(st, 200)
        self.assertEqual(len(self.engine.prompts), 33, "one prompt per field, not one consolidated read")
        self.assertFalse(any("jevpos" in p for p in self.engine.pins))
        self.assertEqual(len(body["answers"]), 33)

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


class Hybrid(unittest.TestCase):
    """M3.1 hybrid: per-field prompts + pinned prefix + the cue row captured during the read (no decode).

    What the mock can prove: the keys ride together (pin and jevpos in one request), the row read is the
    cue, and the answers are unchanged.  What it cannot prove is the readout itself - that is the live A/B
    (latency, agreement, confidence).  Measured live: the cue row is produced by the request's first decode
    window, so the hybrid cannot skip the decode and buys ~1 ms of wire round-trip - which is why C stays
    the default and this stays opt-in.
    """

    @classmethod
    def setUpClass(cls):
        cls.engine, cls.httpd, cls.base = start_server()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self._old = _os.environ.get("STRATA_JEV_HYBRID")
        _os.environ["STRATA_JEV_HYBRID"] = "1"
        self.engine.prompts.clear()
        self.engine.pins.clear()
        self.engine.rows_read.clear()
        self.engine.window_rows = None
        self.engine.final_reads = 0
        self.engine.capture_ok = True
        self.engine.max_news.clear()

    def tearDown(self):
        if self._old is None:
            _os.environ.pop("STRATA_JEV_HYBRID", None)
        else:
            _os.environ["STRATA_JEV_HYBRID"] = self._old

    THREE = {"q1": {"type": "noul", "instructions": "Is it on?"},
             "q2": {"type": "choice", "instructions": "Pick one.", "criteria": {"alpha": 1, "beta": 2}},
             "q3": {"type": "score", "instructions": "Rate it.", "criteria": ["low", "mid", "high"]}}

    def _post(self, questions):
        return post(self.base, "/v1/systemone", {"state": {"hp": 7}, "questions": questions})

    def test_pin_and_jevpos_ride_together_and_skip_the_decode(self):
        st, _ = self._post(self.THREE)
        self.assertEqual(st, 200)
        self.assertEqual(len(self.engine.prompts), 3, "per-field prompts: nothing contaminates the next field")
        for ids, keys, mn, cue_read in zip(self.engine.prompts, self.engine.pins,
                                           self.engine.max_news, self.engine.rows_read):
            self.assertGreater(keys.get("pin", 0), 0, "the shared prefix is still pinned")
            self.assertEqual(keys.get("jevpos"), [len(ids) - 1], "the cue is the prompt's last token")
            self.assertEqual(cue_read, len(ids) - 1, "the row read is that cue")
            self.assertEqual(mn, 1, "max_new stays 1: the cue row IS the first decode window's row "
                                    "(max_new=0 is refused by the engine, and would compute no row)")
        self.assertFalse(self.engine.pins[0].get("strata_checkpoint") is False,
                         "the first field keeps the checkpoint the pin marks")
        self.assertIs(self.engine.pins[1].get("strata_checkpoint"), False,
                      "later fields resume from it and park nothing")


    def test_a_dropped_capture_reads_the_final_row(self):
        """The cue only exists if the engine kept it.  No capture -> BROWAT would address the last window,
        whose rows start at that window's first token; read the final row instead."""
        self.engine.capture_ok = False
        st, body = self._post(self.THREE)
        self.assertEqual(st, 200)
        self.assertEqual(self.engine.rows_read, [])
        self.assertEqual(self.engine.final_reads, 3)
        self.assertEqual(len(body["answers"]), 3)

    def test_answers_are_unchanged(self):
        _, on = self._post(self.THREE)
        _os.environ.pop("STRATA_JEV_HYBRID", None)
        _, off = self._post(self.THREE)
        self.assertEqual(on["answers"], off["answers"])

    def test_env_off_decodes_as_before(self):
        _os.environ.pop("STRATA_JEV_HYBRID", None)
        self._post(self.THREE)
        self.assertTrue(all(mn == 1 for mn in self.engine.max_news), self.engine.max_news)
        self.assertFalse(any("jevpos" in p for p in self.engine.pins))


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



class ChoiceLabelScheme(unittest.TestCase):
    """#3643 label scheme (leonardomurillo): 0-9 up to ten options, a-z to twenty-six, refuse >26,
    and every label must be ONE token or the request is refused (never scored on a truncated path).

    "upper" (A-Z) stays the default so the default path is byte-identical to the last release and the
    calibrators on disk - all fitted against A-Z - stay valid.  STRATA_JEV_LABELS=digit-lower opts in.
    """

    def setUp(self):
        self._scheme = jev.LABEL_SCHEME

    def tearDown(self):
        jev.LABEL_SCHEME = self._scheme

    def test_default_scheme_is_unchanged(self):
        jev.LABEL_SCHEME = "upper"
        self.assertEqual(jev.choice_labels(4), ["A", "B", "C", "D"])

    def test_digit_lower_scheme(self):
        jev.LABEL_SCHEME = "digit-lower"
        self.assertEqual(jev.choice_labels(4), ["0", "1", "2", "3"])
        self.assertEqual(jev.choice_labels(10)[-1], "9")
        self.assertEqual(jev.choice_labels(11)[-1], "a")
        # 26 options = ten digits + sixteen letters, so the range ends at "p" (his scheme caps at 26)
        self.assertEqual(jev.choice_labels(26)[-1], "p")
        self.assertEqual(len(jev.choice_labels(26)), 26)
        self.assertEqual(len(set(jev.choice_labels(26))), 26)

    def test_range_is_2_to_26(self):
        for bad in (1, 27):
            with self.assertRaises(ValueError):
                jev.choice_labels(bad)

    def test_question_block_renders_the_scheme(self):
        jev.LABEL_SCHEME = "digit-lower"
        qdefs, _ = jev.parse_questions({"q": {"type": "choice", "instructions": "pick one",
                                             "criteria": ["red", "green", "blue"]}})
        block = jev.question_block(qdefs[0])
        self.assertIn("0. red", block)
        self.assertIn("2. blue", block)
        self.assertNotIn("A. red", block)

    def test_multi_token_label_is_refused(self):
        """ByteTokenizer maps one byte to one token, so a two-character label is two tokens."""
        class _Svc:
            tok = ByteTokenizer()
        qdefs, _ = jev.parse_questions({"q": {"type": "choice", "instructions": "pick",
                                             "criteria": ["x", "y"]}})
        self.assertIsNone(jev._label_token_problem(_Svc(), qdefs))
        qdefs[0]["render_labels"] = ["ab", "c"]          # force a multi-token label
        self.assertEqual(jev._label_token_problem(_Svc(), qdefs), ("q", "ab", 2))

    def test_noul_and_score_are_unaffected(self):
        jev.LABEL_SCHEME = "digit-lower"
        qdefs, _ = jev.parse_questions({"q": {"type": "noul", "instructions": "yes or no?"},
                                        "s": {"type": "score", "instructions": "how many?",
                                              "criteria": ["zero", "one", "two"]}})
        self.assertEqual(qdefs[0]["render_labels"], ["No", "Yes"])
        self.assertEqual(qdefs[1]["render_labels"], ["0", "1", "2"])


if __name__ == "__main__":
    unittest.main()
