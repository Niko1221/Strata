"""POST /v1/score: teacher-forced log-probabilities - the engine line and its parsing, then the route's shapes.

A training loop reads three things from this endpoint and nothing else: the score of every position of a sequence it
chose, the same figure again at different weights, and the assurance that two runs are comparable.  The tests below
pin those, not the arithmetic of the model (which the engine's own tests cover)."""
import io
import json
import queue
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, Service, StrataEngine, serve
from serve.test_lifecycle import ResidentEngine


class ScoreEngine(ResidentEngine):
    """The resident test engine, plus the SCORE command: `map` over the ids, position t scored against ids[t+1]."""
    def __init__(self, tok):
        super().__init__(tok)
        self.calls, self.lora_calls, self.fail = [], [], None

    def score(self, ids, timeout=21600.0, lora=0):
        self.calls.append(list(ids))
        self.lora_calls.append(lora)
        if self.fail is not None:
            raise self.fail
        lp = [float(i) - float(len(ids)) for i in range(len(ids) - 1)]   # distinct per position, order visible
        return {"logprobs": lp, "sum": sum(lp), "n": len(lp), "ms": 12.5, "head_ms": 1.5, "rows": 1024}


class EngineCommand(unittest.TestCase):
    def engine(self, replies):
        e = StrataEngine("missing-executable", [], lazy=True)
        e.proc = mock.Mock()
        e.proc.poll.return_value = None
        e.proc.stdin = io.StringIO()
        e.ended = False
        e.lines = queue.Queue()
        for r in replies:
            e.lines.put(r)
        return e

    def test_the_line_and_the_answer(self):
        e = self.engine(["SL 0 -1.881845\n", "SL 1 -17.509408\n", "SSUM -19.391253 2 264.2 5.3 1024\n"])
        out = e.score([1, 2, 220])
        self.assertEqual(e.proc.stdin.getvalue(), "SCORE 1,2,220\n")
        self.assertEqual(out["logprobs"], [-1.881845, -17.509408])
        self.assertEqual((out["sum"], out["n"], out["ms"], out["head_ms"], out["rows"]), (-19.391253, 2, 264.2, 5.3, 1024))

    def test_the_sign_reaches_the_engine(self):
        # theta +- eps is one sequence read twice with the sign taken off the adapter: the line carries it, so the
        # two reads are one command apart and no table moves between them.
        e = self.engine(["SL 0 -1.0\n", "SSUM -1.0 1 5.0 0.5 1024\n"])
        e.score([1, 2], lora=-1)
        self.assertEqual(e.proc.stdin.getvalue(), "SCORE - 1,2\n")
        e = self.engine(["SL 0 -1.0\n", "SSUM -1.0 1 5.0 0.5 1024\n"])
        e.score([1, 2], lora=1)
        self.assertEqual(e.proc.stdin.getvalue(), "SCORE + 1,2\n")
        e = self.engine(["SL 0 -1.0\n", "SSUM -1.0 1 5.0 0.5 1024\n"])
        e.score([1, 2])
        self.assertEqual(e.proc.stdin.getvalue(), "SCORE 1,2\n")

    def test_a_nan_position_survives(self):                  # a target of -1 comes back as nan: kept, not dropped
        e = self.engine(["SL 0 -1.0\n", "SL 1 nan\n", "SSUM -1.0 2 5.0 0.5 1024\n"])
        out = e.score([1, 2, 3])
        self.assertEqual(out["logprobs"][0], -1.0)
        self.assertNotEqual(out["logprobs"][1], out["logprobs"][1])

    def test_refused_and_died(self):
        e = self.engine(["ERR score: two ids at least\n"])
        with self.assertRaisesRegex(ValueError, "two ids at least"):
            e.score([1])
        e = self.engine([None])
        with self.assertRaises(EngineDied):
            e.score([1, 2])
        e = self.engine([])                                  # silent: the deadline ends it
        with self.assertRaises(EngineDied):
            e.score([1, 2], timeout=0.2)


class Route(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = ScoreEngine(tok)
        self.engine.loaded = True
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read().decode())

    def test_positions_line_up_with_the_tokens(self):
        code, body = self.request("/v1/score", {"tokens": [1, 2, 220]})
        self.assertEqual(code, 200, body)
        self.assertEqual(body["object"], "score")
        self.assertEqual(body["tokens"], [1, 2, 220])
        self.assertEqual(body["n"], 3)
        self.assertEqual(body["scored"], 2)
        # one figure per token, the last one null: a caller zipping logprobs with tokens never slides by one
        self.assertEqual(len(body["logprobs"]), 3)
        self.assertIsNone(body["logprobs"][-1])
        self.assertEqual(body["logprobs"][:2], [-3.0, -2.0])
        self.assertEqual(body["sum_logprob"], -5.0)
        self.assertEqual(body["mean_logprob"], -2.5)
        self.assertEqual(body["nll"], 5.0)
        self.assertAlmostEqual(body["perplexity"], pow(2.718281828459045, 2.5), places=6)
        self.assertEqual(body["engine"]["head_ms"], 1.5)

    def test_a_prompt_is_tokenized_here(self):
        code, body = self.request("/v1/score", {"prompt": "ab"})
        self.assertEqual(code, 200, body)
        self.assertEqual(body["tokens"], [97, 98])
        self.assertEqual(self.engine.calls, [[97, 98]])

    def test_the_lora_sign_reaches_the_engine(self):
        code, body = self.request("/v1/score", {"tokens": [1, 2, 3], "lora": -1})
        self.assertEqual(code, 200, body)
        self.assertEqual(self.engine.lora_calls, [-1])
        code, body = self.request("/v1/score", {"tokens": [1, 2, 3]})
        self.assertEqual(code, 200, body)
        self.assertEqual(self.engine.lora_calls, [-1, 0])    # absent: whatever the adapter's setting is

    def test_messages_are_rendered_without_a_generation_prompt(self):
        code, body = self.request("/v1/score", {"messages": [{"role": "user", "content": "ab"}]})
        self.assertEqual(code, 200, body)
        self.assertEqual(self.engine.calls[-1], body["tokens"])
        text = "".join(chr(t) for t in body["tokens"] if t < 128)
        self.assertIn("ab", text)
        self.assertFalse(text.rstrip().endswith("<|im_start|>assistant"), text)   # scoring a sequence, not asking

    def test_bad_input(self):
        for bad in ({"tokens": "1,2"}, {"tokens": [1]}, {"tokens": [1, -2]}, {"tokens": [1, True]},
                    {"tokens": [1, 2.5]}, {"messages": 5}, {"prompt": 7}, {}):
            code, body = self.request("/v1/score", bad)
            self.assertEqual(code, 400, bad)
            self.assertEqual(body["error"]["type"], "invalid_request_error", bad)
        self.assertEqual(self.engine.calls, [])

    def test_a_sequence_longer_than_the_context(self):
        self.engine.max_context = 4
        self.assertEqual(self.request("/v1/score", {"tokens": [1, 2, 3, 4, 5]})[0], 400)

    def test_an_engine_without_the_command_says_so(self):
        self.engine.score = None                             # a build of the engine that predates the command
        code, body = self.request("/v1/score", {"tokens": [1, 2]})
        self.assertEqual(code, 400)
        self.assertIn("teacher-forced scoring", body["error"]["message"])

    def test_busy_and_a_dead_engine(self):
        self.svc.vram_wait_s = 0.2
        with self.svc.fifo:                                 # a request is running
            self.assertEqual(self.request("/v1/score", {"tokens": [1, 2]})[0], 409)
        self.assertEqual(self.engine.calls, [])
        self.engine.fail = EngineDied("the engine ended")
        self.assertEqual(self.request("/v1/score", {"tokens": [1, 2]})[0], 503)

    def test_the_last_token_alone_has_nothing_to_score(self):
        code, body = self.request("/v1/score", {"tokens": [7, 8]})
        self.assertEqual(code, 200, body)
        self.assertEqual(body["logprobs"], [-2.0, None])
        self.assertEqual(body["scored"], 1)


if __name__ == "__main__":
    unittest.main()
