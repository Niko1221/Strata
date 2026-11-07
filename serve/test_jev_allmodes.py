"""serve/test_jev_allmodes.py - the full mode matrix on ONE engine load (mock engine, no GPU).

    python -m unittest serve.test_jev_allmodes -v

One Service + one engine for the whole chain, in order:
  1  chat                       (/v1/chat/completions)
  2  text decision: choice
  3  text decision: noul
  4  text decision: score
  5  text decision + then.generate
  6  MIXED MULTIPLE questions: choice + noul + score in ONE request
  7  image + decision (image_mode summary, the two-call default)
  8  image + decision (image_mode direct, raw image tokens decide)
  9  image + decision + then.generate
  10 chat again                  (same load, the model stayed running)

The one-load invariant is asserted twice: the engine INSTANCE is the same object at
every step (no swap), and every step took at least one engine turn on it (nothing
silently fell back to another engine or a cold path).
"""
from __future__ import annotations

import base64
import json
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
from serve.server import ByteTokenizer, Service, serve  # noqa: E402
from serve.test_systemone import DecisionEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

IMG_B64 = base64.b64encode(b"x" * 128).decode()


class ChainEngine(DecisionEngine):
    """DecisionEngine plus: any turn that carries image embeddings yields the scripted
    summary (the vision stage).  Order-independent, unlike SummaryEngine's first-turn
    trick, so the chain can run chat and text decisions BEFORE the image steps."""

    SUMMARY = "A clearing with a small pond, two sheep and a red barn."

    def __init__(self, tokenizer, script):
        super().__init__(tokenizer, script)
        self.turns_seen: list[list[int]] = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.turns_seen.append(list(ids))
        if embeddings is not None:                     # the vision summary turn
            for t in self.tok.encode(self.SUMMARY, parse_special=False):
                yield t
            return
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class _FakeVision:
    """Stands in for Vision: encode returns (path, n_pads); the path must exist
    (prepare() stats it for the combined-embeddings size)."""

    def __init__(self):
        self.encoded: list[str] = []
        import tempfile
        self.dir = Path(tempfile.mkdtemp(prefix="allmodes-vision-"))

    def encode(self, source, keep=()):
        import tempfile
        self.encoded.append(source[:40])
        f = tempfile.NamedTemporaryFile(suffix=".img", delete=False)
        f.write(b"0" * 64)
        f.close()
        return Path(f.name), 4


class AllModesOneLoad(unittest.TestCase):
    """Every decision mode, mixed/multiple questions, both image paths, and chat -
    against a single engine instance that is never swapped or re-created."""

    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = ChainEngine(cls.tok, "A")
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.svc.vision = _FakeVision()
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    # ---- helpers ----

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def step(self, name, path, body):
        """One chain step: same engine object, and the step used it (>= 1 new turn)."""
        before = len(self.engine.turns_seen)
        st, out = self.post(path, body)
        took = len(self.engine.turns_seen) - before
        self.assertEqual(st, 200, f"{name}: {st} {json.dumps(out)[:300]}")
        self.assertIs(self.svc.engine, self.engine, f"{name}: the engine was swapped")
        self.assertGreaterEqual(took, 1, f"{name}: the step took no engine turn")
        return out

    # ---- the chain ----

    def test_all_modes_one_engine_load(self):
        # 1. chat only
        out = self.step("1 chat", "/v1/chat/completions",
                        {"messages": [{"role": "user", "content": "Say ready"}], "max_tokens": 8})
        self.assertIn("choices", out)

        # 2. text decision: choice  (A/B/C map to the mock ROW tokens 65/66/67)
        out = self.step("2 text choice", "/v1/systemone",
                        {"state": {"utterance": "I need to order a new card."},
                         "questions": {"q": {"type": "choice",
                                             "instructions": "Which group?",
                                             "criteria": {"A": "card", "B": "top up", "C": "other"}}}})
        a = out["answers"]["q"]
        self.assertEqual(a["type"], "choice")
        self.assertEqual(a["choice"], 1)                       # B has the top logit
        self.assertEqual(a["label"], "B")
        self.assertEqual(len(a["probabilities"]), 3)
        self.assertAlmostEqual(sum(a["probabilities"]), 1.0, places=5)
        self.assertIn("margin", a)
        self.assertIn("calib", a)

        # 3. text decision: noul  (No/Yes -> tokens 78/89, Yes wins)
        out = self.step("3 text noul", "/v1/systemone",
                        {"state": {"utterance": "The screen shows a login prompt."},
                         "questions": {"q": {"type": "noul", "instructions": "Is text visible?"}}})
        a = out["answers"]["q"]
        self.assertEqual(a["type"], "noul")
        self.assertGreater(a["noul"], 0.5)                     # Yes
        self.assertTrue(0.0 <= a["noul"] <= 1.0)
        self.assertIn("margin", a)

        # 4. text decision: score  (labels 0..3 -> tokens 48..51, "1" wins)
        out = self.step("4 text score", "/v1/systemone",
                        {"state": {"utterance": "Two of the five lights are on."},
                         "questions": {"q": {"type": "score", "instructions": "How full is it?",
                                             "criteria": ["0", "1", "2", "3"]}}})
        a = out["answers"]["q"]
        self.assertEqual(a["type"], "score")
        self.assertGreater(a["score"], 0.5)                    # mass on label "1"
        self.assertLess(a["score"], 2.5)
        self.assertEqual(len(a["probabilities"]), 4)

        # 5. text decision + then.generate
        out = self.step("5 text choice+then", "/v1/systemone",
                        {"state": {"utterance": "Health is low and an enemy is near."},
                         "questions": {"q": {"type": "choice", "instructions": "Next action?",
                                             "criteria": {"A": "fight", "B": "drink", "C": "flee"}}},
                         "then": {"generate": {"messages": [{"role": "user", "content": "Narrate the turn."}],
                                               "max_tokens": 16}}})
        self.assertEqual(out["answers"]["q"]["type"], "choice")
        self.assertIn("then", out)
        self.assertIsInstance(out["then"].get("text"), str)
        self.assertIn("finish_reason", out["then"])

        # 6. MIXED MULTIPLE questions: choice + noul + score in ONE request
        out = self.step("6 mixed multiple", "/v1/systemone",
                        {"state": {"utterance": "Card expiring, text visible, two lights on."},
                         "questions": {
                             "qc": {"type": "choice", "instructions": "Which group?",
                                    "criteria": {"A": "card", "B": "top up", "C": "other"}},
                             "qn": {"type": "noul", "instructions": "Is text visible?"},
                             "qs": {"type": "score", "instructions": "How full?",
                                    "criteria": ["0", "1", "2", "3"]}}})
        self.assertEqual(set(out["answers"]), {"qc", "qn", "qs"})
        self.assertEqual(out["answers"]["qc"]["type"], "choice")
        self.assertEqual(out["answers"]["qc"]["label"], "B")
        self.assertGreater(out["answers"]["qn"]["noul"], 0.5)
        self.assertEqual(out["answers"]["qs"]["type"], "score")

        # 7. image + decision, image_mode summary (two-call default)
        out = self.step("7 image summary", "/v1/systemone",
                        {"image_b64": IMG_B64,
                         "questions": {"q": {"type": "noul", "instructions": "Is there water?"}}})
        self.assertEqual(out["stages_used"], ["vision", "decision"])
        self.assertIn("pond", out["image_summary"])
        self.assertGreater(out["answers"]["q"]["noul"], 0.5)

        # 8. image + decision, image_mode direct (raw image tokens decide)
        out = self.step("8 image direct", "/v1/systemone",
                        {"image_b64": IMG_B64, "image_mode": "direct",
                         "questions": {"q": {"type": "choice", "instructions": "Which scene?",
                                             "criteria": {"A": "field", "B": "pond", "C": "street"}}}})
        a = out["answers"]["q"]
        self.assertEqual(a["type"], "choice")
        self.assertEqual(a["label"], "B")                      # pond: the row readout decides

        # 9. image + decision + then.generate
        out = self.step("9 image+then", "/v1/systemone",
                        {"image_b64": IMG_B64,
                         "questions": {"q": {"type": "noul", "instructions": "Is there water?"}},
                         "then": {"generate": {"messages": [{"role": "user", "content": "Describe it in one sentence."}],
                                               "max_tokens": 16}}})
        self.assertGreater(out["answers"]["q"]["noul"], 0.5)
        self.assertIsInstance(out.get("then", {}).get("text"), str)

        # 10. chat again - the SAME load served everything; the model stayed running
        out = self.step("10 chat again", "/v1/chat/completions",
                        {"messages": [{"role": "user", "content": "Say done"}], "max_tokens": 8})
        self.assertIn("choices", out)

        # the one-load invariant, stated once more at the end
        self.assertIs(self.svc.engine, self.engine)
        self.assertGreaterEqual(len(self.engine.turns_seen), 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
