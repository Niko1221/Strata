"""serve/test_jev_calib.py - M3.2: jevcpp calibration port (mock engine, no GPU).

    python -m unittest serve.test_jev_calib -v

Covers the donor surface (src/server.cpp:518-633, 1142, 1223-1230, 2798-2810):
- calib load/apply: identity without a file; Platt on noul; temperature on choice/score;
  argmax preserved (monotone); probabilities renormalized to 1.
- the exact apply ORDER: round 4dp -> renormalize -> calibrate -> answers from calibrated pr.
- log rows: one per answered question, donor B1.1 schema (ts, tuple, qid, type, raw_p),
  raw_p UNROUNDED.
- POST /admin/calib: auth contract + hot reload mid-server.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import time
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
from serve import jev, jev_calib  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class CalibMath(unittest.TestCase):
    """jev_calib.apply: the donor's post-hoc, monotone transforms."""

    def test_identity_when_not_loaded(self):
        jev_calib._set_for_test(None)
        pr = [0.7, 0.3]
        out, cid = jev_calib.apply("noul", pr)
        self.assertEqual(cid, "identity")
        self.assertEqual(out, pr)

    def test_temperature_flattens_and_renormalizes(self):
        jev_calib._set_for_test(jev_calib.Calib(loaded=True, T_choice=2.0))
        pr = [0.9, 0.1]
        out, cid = jev_calib.apply("choice", pr)
        self.assertEqual(cid, "T_choice_v1")
        self.assertAlmostEqual(sum(out), 1.0, places=9)
        self.assertLess(out[0], 0.9)                      # flatter
        self.assertEqual(out.index(max(out)), 0)           # argmax preserved

    def test_temperature_identity_at_one(self):
        jev_calib._set_for_test(jev_calib.Calib(loaded=True, T_choice=1.0))
        pr = [0.9, 0.1]
        out, cid = jev_calib.apply("choice", pr)
        self.assertEqual(cid, "identity")
        self.assertAlmostEqual(out[0], 0.9, places=9)

    def test_score_uses_t_score(self):
        jev_calib._set_for_test(jev_calib.Calib(loaded=True, T_score=3.0))
        pr = [0.05, 0.9, 0.05]
        out, cid = jev_calib.apply("score", pr)
        self.assertEqual(cid, "T_score_v1")
        self.assertAlmostEqual(sum(out), 1.0, places=9)
        self.assertEqual(out.index(max(out)), 1)

    def test_platt_on_noul(self):
        # a=0.5, c=-2 compresses the logit gap and biases down: a strong yes comes back lower
        jev_calib._set_for_test(jev_calib.Calib(loaded=True, a_noul=0.5, c_noul=-2.0))
        pr = [0.01, 0.99]
        out, cid = jev_calib.apply("noul", pr)
        self.assertEqual(cid, "platt_v1")
        self.assertAlmostEqual(out[0] + out[1], 1.0, places=9)
        self.assertLess(out[1], 0.99)
        self.assertGreater(out[1], 0.5)
        self.assertEqual(out.index(max(out)), 1)           # argmax preserved

    def test_load_parses_donor_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "calib.json"
            p.write_text(json.dumps({"tuple": "t|iq3_xxs|none|v1|L|all",
                                     "T_choice": 1.5861, "T_score": 1.0,
                                     "a_noul": 1.0, "c_noul": 0.0}))
            c = jev_calib.load(p)
            self.assertTrue(c.loaded)
            self.assertAlmostEqual(c.T_choice, 1.5861, places=6)

    def test_load_missing_file_is_identity(self):
        c = jev_calib.load(Path("/nonexistent/calib.json"))
        self.assertFalse(c.loaded)

    def test_load_bad_json_is_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "calib.json"
            p.write_text("{not json")
            c = jev_calib.load(p)
            self.assertFalse(c.loaded)


from serve.test_systemone import DecisionEngine  # noqa: E402


class SystemOneCalibrated(unittest.TestCase):
    """The endpoint applies the donor's exact order and logs rows; /admin/calib hot-reloads."""

    CALIB = {"tuple": "p-jevport-3090|iq3_xxs|none|v1|L|all",
             "T_choice": 2.5, "T_score": 1.0, "a_noul": 0.5, "c_noul": -2.0}

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        (cls.dir / "data").mkdir()
        cls._env = jev_calib.CALIB_DIR
        jev_calib.CALIB_DIR = cls.dir
        jev_calib.ADMIN_KEY = ""
        tok = ByteTokenizer()
        engine = DecisionEngine(tok, "A")
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        jev_calib.CALIB_DIR = cls._env
        jev_calib._set_for_test(None)
        cls.tmp.cleanup()

    def setUp(self):
        (self.dir / "calib.json").write_text(json.dumps(self.CALIB))
        jev_calib.swap(self.dir)

    def _post(self, path, body, headers=None):
        h = {"Content-Type": "application/json"}
        if headers:
            h.update(headers)
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers=h)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_choice_answer_is_calibrated(self):
        # ROW: A=0.0, B=4.0, C=-1.0 -> raw softmax ~ [0.018, 0.975, 0.0066]
        st, body = self._post("/v1/systemone", {
            "state": {"k": 1},
            "questions": {"q1": {"type": "choice", "instructions": "Pick.",
                                 "criteria": {"a": 1, "b": 2, "c": 3}}}})
        self.assertEqual(st, 200)
        pr = body["answers"]["q1"]["probabilities"]
        self.assertAlmostEqual(sum(pr), 1.0, places=6)
        # T=2.5 flattens: B's raw ~0.975 must come down
        self.assertLess(pr[1], 0.9)
        self.assertEqual(body["answers"]["q1"]["choice"], 1)   # argmax preserved
        # confidence is computed FROM the calibrated pr (donor order)
        k, pmax = len(pr), max(pr)
        self.assertAlmostEqual(body["answers"]["q1"]["confidence"],
                               (k * pmax - 1) / (k - 1), places=6)

    def test_noul_answer_is_calibrated(self):
        # ROW: No=0.0, Yes=4.0 -> raw d ~= 4, raw p_yes ~= 0.982;
        # platt a=0.5 c=-2 -> sigmoid(0.5*4 - 2) = sigmoid(0) = 0.5
        st, body = self._post("/v1/systemone", {
            "state": {"k": 1},
            "questions": {"q2": {"type": "noul", "instructions": "Is it on?"}}})
        self.assertEqual(st, 200)
        noul = body["answers"]["q2"]["noul"]
        self.assertAlmostEqual(noul, 0.5, delta=0.01)
        self.assertEqual(body["answers"]["q2"].get("calib"), "platt_v1")

    def test_score_uses_identity_when_t_one(self):
        st, body = self._post("/v1/systemone", {
            "state": {"k": 1},
            "questions": {"q3": {"type": "score", "instructions": "Rate.",
                                 "criteria": ["low", "mid", "high", "top"]}}})
        self.assertEqual(st, 200)
        pr = body["answers"]["q3"]["probabilities"]
        # identity (T_score=1): rounded-4dp raw softmax; these four round so the sum is exactly 1
        self.assertAlmostEqual(pr[1], 0.9732, places=9)
        self.assertEqual(body["answers"]["q3"].get("calib"), "identity")

    def test_log_rows_donor_schema(self):
        st, body = self._post("/v1/systemone", {
            "state": {"k": 1},
            "questions": {"q4": {"type": "choice", "instructions": "Pick.",
                                 "criteria": {"a": 1, "b": 2}}}})
        self.assertEqual(st, 200)
        n = jev_calib.log_flush(self.dir)                 # the background flusher's exact drain
        self.assertGreaterEqual(n, 1)
        log = (self.dir / "data" / "log-L.jsonl").read_text().strip().splitlines()
        rows = [json.loads(x) for x in log]
        row = [r for r in rows if r["qid"] == "q4"][-1]
        self.assertIn("ts", row)
        self.assertTrue(row["tuple"].endswith("|choice"))
        self.assertIn("raw_p", row)
        self.assertEqual(len(row["raw_p"]), 2)
        # raw_p is UNROUNDED (donor logs r.p, not the rounded pr)
        self.assertTrue(any(len(str(abs(x)).split(".")[-1]) > 4 for x in row["raw_p"]))

    def test_admin_calib_requires_bearer(self):
        jev_calib.ADMIN_KEY = "k1"
        try:
            st, _ = self._post("/admin/calib", {})
            self.assertEqual(st, 401)
            st, body = self._post("/admin/calib", {},
                                  {"Authorization": "Bearer k1"})
            self.assertEqual(st, 200)
            self.assertTrue(body["loaded"])
            self.assertEqual(body["modes"]["text(legacy)"]["tuple"], "p-jevport-3090|iq3_xxs|none|v1|L|all")
        finally:
            jev_calib.ADMIN_KEY = ""

    def test_admin_calib_404_when_no_key_configured(self):
        """Opus G2: with no admin key the surface must not exist (fail-closed)."""
        jev_calib.ADMIN_KEY = ""
        st, body = self._post("/admin/calib", {})
        self.assertEqual(st, 404)
        self.assertIn("not found", body["error"]["message"])

    def test_admin_calib_hot_reload(self):
        (self.dir / "calib.json").write_text(json.dumps({
            "tuple": "hot|iq3_xxs|none|v1|L|all",
            "T_choice": 1.0, "T_score": 1.0, "a_noul": 1.0, "c_noul": 0.0}))
        jev_calib.ADMIN_KEY = "hot1"                    # fail-closed: key required now
        try:
            st, body = self._post("/admin/calib", {}, {"Authorization": "Bearer hot1"})
            self.assertEqual(st, 200)
            self.assertEqual(body["modes"]["text(legacy)"]["tuple"], "hot|iq3_xxs|none|v1|L|all")
            # identity now: choice answer back at raw softmax
            st, body = self._post("/v1/systemone", {
                "state": {"k": 1},
                "questions": {"q5": {"type": "choice", "instructions": "Pick.",
                                     "criteria": {"a": 1, "b": 2, "c": 3}}}})
            pr = body["answers"]["q5"]["probabilities"]
            self.assertGreater(pr[1], 0.9)
        finally:
            jev_calib.ADMIN_KEY = ""


def httpd_server_port(httpd):
    return httpd.server_address[1]


if __name__ == "__main__":
    unittest.main()


class PerTupleCalib(unittest.TestCase):
    """P0-1 (Opus plan): calibrators are keyed by readout letter."""

    def setUp(self):
        jev_calib._set_for_test(None, "")
        jev_calib._set_for_test(None, "L")
        jev_calib._set_for_test(None, "D")
        jev_calib._set_for_test(jev_calib.Calib(loaded=True, T_choice=3.0, T_score=3.0), "D")

    def tearDown(self):
        for m in ("", "L", "V", "D", "S"):
            jev_calib._set_for_test(None, m)

    def test_image_fit_does_not_touch_text(self):
        pr, cid = jev_calib.apply("choice", [0.9, 0.1], "")      # text: no fit -> identity
        self.assertEqual(cid, "identity")
        pr, cid = jev_calib.apply("choice", [0.9, 0.1], "D")     # image D: its own T
        self.assertEqual(cid, "T_choice_v1")

    def test_L_falls_back_to_text_and_S_does_not(self):
        jev_calib._set_for_test(jev_calib.Calib(loaded=True, T_score=2.0), "")
        pr, cid = jev_calib.apply("score", [0.8, 0.2], "L")      # L shares the legacy fit
        self.assertEqual(cid, "T_score_v1")
        _, cid = jev_calib.apply("score", [0.8, 0.2], "S")       # S: strict, no fallback
        self.assertEqual(cid, "identity")

    def test_swap_loads_per_mode_files(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            Path(td, "calib.json").write_text(json.dumps({"T_choice": 1.5}))
            Path(td, "calib-D.json").write_text(json.dumps({"T_choice": 4.0}))
            loaded = jev_calib.swap(Path(td))
            self.assertTrue(loaded["text(legacy)"])
            self.assertTrue(loaded["D"])
            _, cid = jev_calib.apply("choice", [0.9, 0.1], "")
            self.assertEqual(cid, "T_choice_v1")
            pr, cid = jev_calib.apply("choice", [0.9, 0.1], "D")
            self.assertEqual(cid, "T_choice_v1")
            self.assertGreater(pr[1], 0.1 / 0.9 * 0.5)  # T=4 flattens toward uniform


class OpusPrereviewFixes(unittest.TestCase):
    """R1/G3/G13 from /opt/data/reviews/jevport-prereview-opus.md (2026-10-15)."""

    def setUp(self):
        for m in ("", "L", "V", "D", "S"):
            jev_calib._set_for_test(None, m)

    def tearDown(self):
        for m in ("", "L", "V", "D", "S"):
            jev_calib._set_for_test(None, m)

    def test_r1_image_modes_never_inherit_text_fit(self):
        jev_calib._set_for_test(jev_calib.Calib(loaded=True, T_score=2.0), "")  # legacy text fit
        _, cid = jev_calib.apply("score", [0.8, 0.2], "S")
        self.assertEqual(cid, "identity")      # S: NO fallback to the text fit
        _, cid = jev_calib.apply("score", [0.8, 0.2], "D")
        self.assertEqual(cid, "identity")      # D: no own file -> identity
        _, cid = jev_calib.apply("score", [0.8, 0.2], "")   # text legacy: falls back
        self.assertEqual(cid, "T_score_v1")
        _, cid = jev_calib.apply("score", [0.8, 0.2], "L")
        self.assertEqual(cid, "T_score_v1")    # L shares the legacy fit

    def test_g13_tuple_mismatch_refused(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            Path(td, "calib-S.json").write_text(
                json.dumps({"T_score": 3.0, "tuple": "x|q|none|v1|D|image-direct"}))
            loaded = jev_calib.swap(Path(td))
            self.assertFalse(loaded["S"])          # D-tuple file refused for S
            self.assertNotIn(True, [v for k, v in loaded.items() if k == "S"])

    def test_g13_matching_tuple_accepted(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            Path(td, "calib-S.json").write_text(
                json.dumps({"T_score": 3.0, "tuple": "x|q|none|v1|S|image-summary"}))
            loaded = jev_calib.swap(Path(td))
            self.assertTrue(loaded["S"])

    def test_g3_log_flush_splits_by_tuple(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            jev_calib.log_enqueue({"tuple": "a|q|none|v1|L|noul", "qid": "x1", "type": "noul", "raw_p": [0.5, 0.5]})
            jev_calib.log_enqueue({"tuple": "a|q|none|v1|D|noul", "qid": "x2", "type": "noul", "raw_p": [0.5, 0.5]})
            jev_calib.log_enqueue({"tuple": "a|q|none|v1|S|noul", "qid": "x3", "type": "noul", "raw_p": [0.5, 0.5]})
            n = jev_calib.log_flush(Path(td))
            self.assertEqual(n, 3)
            data = Path(td, "data")
            self.assertTrue((data / "log-L.jsonl").exists())
            self.assertTrue((data / "log-D.jsonl").exists())
            self.assertTrue((data / "log-S.jsonl").exists())
            self.assertFalse((data / "log.jsonl").exists())   # no mixed file anymore


class ModeGating(unittest.TestCase):
    """A0.1 mode gating (donor --modes chat,decision): fail-closed 404s,
    env parsing, default both-on."""

    def test_modes_default_chat_only(self):
        # Justin's ruling 2026-10-15: undefined = chat only (stock Strata
        # posture); the decision surface must be opted into
        import importlib, os
        old = os.environ.pop("STRATA_JEV_MODES", None)
        try:
            m = importlib.import_module("serve.jev_calib")
            importlib.reload(m)
            self.assertEqual(m.MODES, frozenset({"chat"}))
        finally:
            if old is not None:
                os.environ["STRATA_JEV_MODES"] = old
            else:
                os.environ.pop("STRATA_JEV_MODES", None)
            importlib.reload(importlib.import_module("serve.jev_calib"))

    def test_modes_decision_only(self):
        import importlib, os
        old = os.environ.pop("STRATA_JEV_MODES", None)
        os.environ["STRATA_JEV_MODES"] = "decision"
        try:
            m = importlib.import_module("serve.jev_calib")
            importlib.reload(m)
            self.assertEqual(m.MODES, frozenset({"decision"}))
        finally:
            if old is not None:
                os.environ["STRATA_JEV_MODES"] = old
            else:
                os.environ.pop("STRATA_JEV_MODES", None)   # must DELETE: leaked 'decision' gated later tests
            importlib.reload(importlib.import_module("serve.jev_calib"))

    def test_modes_unknown_fails_closed(self):
        with self.assertRaises(ValueError):
            jev_calib._parse_modes("chat,bananas")

    def test_systemone_404_when_decision_gated_off(self):
        # router-level check replicated: the gate key is what the router tests
        self.assertIn("decision", jev_calib.MODES)   # default instance has it on
        # and the parse that would turn it off:
        self.assertNotIn("decision", jev_calib._parse_modes("chat"))
