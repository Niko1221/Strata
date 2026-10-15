"""serve/test_jev_vmode.py - M3.1: V-mode choice readout (X-Jev-Mode: V), mock engine.

    python -m unittest serve.test_jev_vmode -v

Pins the donor contract (server.cpp A2.6):
- V applies to CHOICE only; noul/score fall through to the L path untouched.
- Per option: a branch prompt "Proposed answer: <name>\n...Answer:" whose readout
  is a yes/no pair; d_i = s_yes - s_no; p = softmax(d) (4dp, renormalized).
- Response: probabilities keyed by OPTION NAME, "choice" = the name string.
- Log row: tuple ...|V|choice with raw_p and v_d.
- The branch prompts share the state+head prefix (KV prefix-mount shape).
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
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
from serve.server import ByteTokenizer, Service, serve  # noqa: E402
from serve import jev, jev_calib  # noqa: E402
from serve.test_systemone import DecisionEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class VEngine(DecisionEngine):
    """DecisionEngine whose row depends on the LAST branch prompt: yes for the option
    named 'mountains', no otherwise (ByteTokenizer ids: No=78, Yes=89).  ROW also
    covers A-D letters so the L-mode fallthrough (this class inherits it) works."""

    ROW = {78: 0.0, 89: 4.0,                     # noul/V yes branch
           65: 0.0, 66: 4.0, 67: -1.0, 68: -2.0}  # choice letters A..D

    def final_row_logits(self, ids, tokens):
        prompt = self.tok.decode(self.prompts[-1]) if self.prompts else ""
        yes = "mountains" in prompt.split("Proposed answer:")[-1].split("\n")[0]
        row = ({78: 0.0, 89: 4.0} if yes else {78: 4.0, 89: 0.0}) \
            if "Proposed answer:" in prompt else self.ROW
        return {t: row[t] for t in tokens if t in row}


def start(vmode_header):
    tok = ByteTokenizer()
    engine = VEngine(tok, "A")
    svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    httpd = serve(svc, port=0)

    def _post(body):
        h = {"Content-Type": "application/json"}
        if vmode_header:
            h["X-Jev-Mode"] = vmode_header
        req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/systemone",
                                     data=json.dumps(body).encode(), headers=h)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    post = staticmethod(_post)
    return engine, httpd, post


REQ = {"state": {"k": 1}, "questions": {
    "q1": {"type": "choice", "instructions": "Pick.",
           "criteria": {"forests": None, "oceans": None, "prairies": None, "mountains": None}}}}


class VMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        jev_calib.CALIB_DIR = Path(cls.tmp.name)
        jev_calib._set_for_test(None)
        cls.engine, cls.httpd, cls.post = start("V")

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.tmp.cleanup()

    def test_choice_is_name_keyed_softmax_of_d(self):
        st, body = self.post(REQ)
        self.assertEqual(st, 200)
        a = body["answers"]["q1"]
        self.assertEqual(a["choice"], "mountains")
        self.assertEqual(set(a["probabilities"].keys()),
                         {"forests", "oceans", "prairies", "mountains"})
        # one yes branch (d=+4) among three no (d=-4): softmax over gap 8
        self.assertAlmostEqual(a["probabilities"]["mountains"], 0.9991, delta=0.002)
        self.assertAlmostEqual(a["probabilities"]["forests"], 0.0003, delta=0.002)
        self.assertAlmostEqual(sum(a["probabilities"].values()), 1.0, places=6)
        self.assertGreater(a["confidence"], 0.9)

    def test_one_branch_prompt_per_option_with_shared_head(self):
        st, _ = self.post(REQ)
        self.assertEqual(st, 200)
        prompts = [self.engine.tok.decode(p) for p in self.engine.prompts[-4:]]
        self.assertEqual(len(prompts), 4)
        for i, key in enumerate(("forests", "oceans", "prairies", "mountains")):
            self.assertIn(f"Proposed answer: {key}\n", prompts[i])
            self.assertIn("Is the proposed answer the best option? Reply Yes or No.\nAnswer:", prompts[i])
            self.assertIn("Question: Pick.\nOptions:\n- forests\n- oceans\n- prairies\n- mountains\n",
                          prompts[i])
            self.assertNotIn("A. forests", prompts[i])      # L-mode letter render must not leak
        self.assertIn('"k": 1', prompts[0])                 # the state rides along (JSON for objects)

    def test_readout_count_is_per_option(self):
        st, body = self.post(REQ)
        self.assertEqual(st, 200)
        self.assertEqual(body["usage"]["readout_tokens"], 4)  # one BROW per branch

    def test_log_row_v_tuple_with_v_d(self):
        st, _ = self.post(REQ)
        self.assertEqual(st, 200)
        n = jev_calib.log_flush(jev_calib.CALIB_DIR)
        self.assertGreaterEqual(n, 1)
        rows = [json.loads(x) for x in
                (jev_calib.CALIB_DIR / "data" / "log-V.jsonl").read_text().splitlines()]
        row = [r for r in rows if r["qid"] == "q1"][-1]
        self.assertTrue(row["tuple"].endswith("|V|choice"))
        self.assertEqual(len(row["v_d"]), 4)
        self.assertAlmostEqual(row["v_d"][3], 4.0, places=6)   # mountains = yes branch
        self.assertAlmostEqual(row["v_d"][0], -4.0, places=6)


class LModeUnchanged(unittest.TestCase):
    """No header (or L): the choice path stays byte-L-mode (letters, array probs)."""

    @classmethod
    def setUpClass(cls):
        cls.engine, cls.httpd, cls.post = start(None)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_l_mode_still_letters_and_array(self):
        st, body = self.post(REQ)
        self.assertEqual(st, 200)
        a = body["answers"]["q1"]
        self.assertEqual(a["choice"], 1)                     # index, not name
        self.assertIsInstance(a["probabilities"], list)
        self.assertEqual(len(a["probabilities"]), 4)
        prompts = [self.engine.tok.decode(p) for p in self.engine.prompts]
        self.assertIn("A. forests", prompts[-1])

    def test_v_header_leaves_noul_alone(self):
        engine, httpd, post = start("V")
        try:
            st, body = post({"state": {"k": 1}, "questions": {
                "q2": {"type": "noul", "instructions": "Is it on?"}}})
            self.assertEqual(st, 200)
            a = body["answers"]["q2"]
            self.assertEqual(a["type"], "noul")
            self.assertIn("noul", a)
            self.assertNotIn("probabilities", a)
            prompts = [engine.tok.decode(p) for p in engine.prompts]
            self.assertIn("Reply Yes or No.", prompts[-1])  # L-render noul
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
