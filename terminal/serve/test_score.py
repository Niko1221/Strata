"""serve/test_score.py - unit tests for StrataEngine.score and POST /v1/score.

    python -m unittest terminal/serve/test_score.py -v
"""
from __future__ import annotations

import io
import json
import queue
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (  # noqa: E402
    ByteTokenizer,
    EngineDied,
    MockEngine,
    Service,
    StrataEngine,
    serve,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeScoreEngine(MockEngine):
    def __init__(self, tokenizer, *, score_supported: bool = True):
        super().__init__(tokenizer, "</think>\n\nok", max_context=4096)
        self.info = {"score": 1 if score_supported else 0, "lora": 48, "lora_reload": 1}
        self.score_calls: list[dict] = []
        self.score_error: Exception | None = None

    def score(self, token_ids: list[int], from_pos: int = 0, lora: bool | None = None,
              cvec: bool | None = None, timeout: float = 300.0) -> dict:
        self.score_calls.append({
            "tokens": list(token_ids),
            "from_pos": from_pos,
            "lora": lora,
            "cvec": cvec,
        })
        if self.score_error is not None:
            raise self.score_error
        count = len(token_ids) - 1 - from_pos
        base = -0.5 if lora is False else -1.0
        logprobs = [round(base - 0.1 * i, 6) for i in range(count)]
        sum_lp = sum(logprobs)
        return {
            "count": count,
            "from": from_pos,
            "mean_nll": -sum_lp / count,
            "sum_logprob": sum_lp,
            "ms": 7.5,
            "logprobs": logprobs,
        }


class _HTTPMixin:
    base: str

    def request(self, method: str, path: str, body=None, headers: dict | None = None):
        hdrs = dict(headers or {})
        data = None
        if body is not None:
            if isinstance(body, (bytes, bytearray)):
                data = bytes(body)
            else:
                data = json.dumps(body).encode("utf-8")
                hdrs.setdefault("Content-Type", "application/json")
        req = urllib.request.Request(self.base + path, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())


class StrataEngineScoreWireTest(unittest.TestCase):
    def _make_engine(self, responses: list[str | None]):
        eng = object.__new__(StrataEngine)
        eng.batch = 0
        eng.ended = False
        eng.lines = queue.Queue()
        for line in responses:
            eng.lines.put(line)
        stdin_buf = io.StringIO()

        class FakeProc:
            def __init__(self):
                self.stdin = stdin_buf

            def poll(self):
                return None

            def wait(self, timeout=None):
                return 1

        eng.proc = FakeProc()
        return eng, stdin_buf

    def test_ok_basic_and_options(self):
        eng, stdin_buf = self._make_engine([
            "INFO ignore=1\n",
            "SC 2 0 1.250000 4.5 -1.000000 -1.500000\n",
        ])
        res = eng.score([10, 20, 30])
        self.assertEqual(stdin_buf.getvalue(), "SCORE 10,20,30\n")
        self.assertEqual(res["count"], 2)
        self.assertEqual(res["from"], 0)
        self.assertAlmostEqual(res["mean_nll"], 1.25)
        self.assertAlmostEqual(res["sum_logprob"], -2.5)
        self.assertAlmostEqual(res["ms"], 4.5)
        self.assertEqual(res["logprobs"], [-1.0, -1.5])

        eng2, stdin_buf2 = self._make_engine([
            "SC 2 2 0.750000 3.2 -0.500000 -1.000000\n",
        ])
        res2 = eng2.score([1, 2, 3, 4, 5], from_pos=2, lora=False, cvec=True)
        self.assertEqual(stdin_buf2.getvalue(), "SCORE lora=0 cvec=1 from=2 1,2,3,4,5\n")
        self.assertEqual(res2["count"], 2)
        self.assertEqual(res2["from"], 2)
        self.assertAlmostEqual(res2["sum_logprob"], -1.5)

    def test_err_reply_raises_value_error(self):
        eng, _ = self._make_engine(["ERR score: token id out of vocabulary range\n"])
        with self.assertRaisesRegex(ValueError, "out of vocabulary range"):
            eng.score([1, 9999999])

    def test_disconnect_and_malformed_raise_engine_died(self):
        eng_eof, _ = self._make_engine([None])
        with self.assertRaises(EngineDied):
            eng_eof.score([1, 2, 3])

        eng_bad, _ = self._make_engine(["SC 1 0 1.0 2.0 -1.0\n"])  # expected count 2, got 1
        with self.assertRaises(EngineDied):
            eng_bad.score([1, 2, 3])

        eng_junk, _ = self._make_engine(["DONE 0\n"])
        with self.assertRaises(EngineDied):
            eng_junk.score([1, 2])

    def test_client_validation(self):
        eng, stdin_buf = self._make_engine([])
        with self.assertRaises(ValueError):
            eng.score([1])
        with self.assertRaises(ValueError):
            eng.score([1, True])
        with self.assertRaises(ValueError):
            eng.score([1, -5])
        with self.assertRaises(ValueError):
            eng.score([1, 2, 3], from_pos=2)
        with self.assertRaises(ValueError):
            eng.score([1, 2, 3], lora=1)  # type: ignore[arg-type]
        self.assertEqual(stdin_buf.getvalue(), "")


class ScoreEndpointTest(_HTTPMixin, unittest.TestCase):
    def _start(self, engine, *, api_key: str = ""):
        tok = ByteTokenizer()
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.api_key = api_key
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.base = f"http://127.0.0.1:{httpd.server_address[1]}"
        return svc, tok

    def test_raw_tokens_and_from(self):
        tok = ByteTokenizer()
        eng = FakeScoreEngine(tok)
        self._start(eng)

        status, body = self.request("POST", "/v1/score", {"tokens": [10, 20, 30, 40]})
        self.assertEqual(status, 200)
        self.assertEqual(body["count"], 3)
        self.assertEqual(body["from"], 0)
        self.assertEqual(body["tokens"], [10, 20, 30, 40])
        self.assertEqual(len(body["logprobs"]), 3)
        self.assertAlmostEqual(body["sum_logprob"], sum(body["logprobs"]))
        self.assertAlmostEqual(body["mean_nll"], -body["sum_logprob"] / 3)
        self.assertEqual(eng.score_calls[-1], {"tokens": [10, 20, 30, 40], "from_pos": 0, "lora": None, "cvec": None})

        status, body = self.request("POST", "/v1/score", {"tokens": [10, 20, 30, 40], "from": 2})
        self.assertEqual(status, 200)
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["from"], 2)
        self.assertEqual(len(body["logprobs"]), 1)
        self.assertEqual(eng.score_calls[-1]["from_pos"], 2)

    def test_prompt_and_completion_encoding(self):
        tok = ByteTokenizer()
        eng = FakeScoreEngine(tok)
        self._start(eng)

        prompt = "Hi!"
        completion = "Yes."
        prompt_ids = tok.encode(prompt)
        comp_ids = tok.encode(completion)

        status, body = self.request("POST", "/v1/score", {"prompt": prompt, "completion": completion})
        self.assertEqual(status, 200)
        self.assertEqual(body["tokens"], prompt_ids + comp_ids)
        self.assertEqual(body["from"], len(prompt_ids) - 1)
        self.assertEqual(body["count"], len(comp_ids))
        self.assertEqual(len(body["logprobs"]), len(comp_ids))
        self.assertEqual(eng.score_calls[-1]["tokens"], prompt_ids + comp_ids)
        self.assertEqual(eng.score_calls[-1]["from_pos"], len(prompt_ids) - 1)

    def test_lora_and_cvec_forwarding(self):
        tok = ByteTokenizer()
        eng = FakeScoreEngine(tok)
        self._start(eng)

        status_on, body_on = self.request("POST", "/v1/score", {"tokens": [1, 2, 3], "lora": True, "cvec": False})
        self.assertEqual(status_on, 200)
        self.assertEqual(eng.score_calls[-1]["lora"], True)
        self.assertEqual(eng.score_calls[-1]["cvec"], False)

        status_off, body_off = self.request("POST", "/v1/score", {"tokens": [1, 2, 3], "lora": False, "cvec": True})
        self.assertEqual(status_off, 200)
        self.assertEqual(eng.score_calls[-1]["lora"], False)
        self.assertEqual(eng.score_calls[-1]["cvec"], True)
        self.assertNotEqual(body_on["logprobs"], body_off["logprobs"])

    def test_409_unloaded_unsupported_and_busy(self):
        tok = ByteTokenizer()
        # Unsupported engine (score=0)
        eng_no = FakeScoreEngine(tok, score_supported=False)
        self._start(eng_no)
        status, body = self.request("POST", "/v1/score", {"tokens": [1, 2, 3]})
        self.assertEqual(status, 409)
        self.assertEqual(body["error"]["type"], "model_busy")

        # Unloaded engine
        eng_ok = FakeScoreEngine(tok, score_supported=True)
        svc, _ = self._start(eng_ok)
        svc.engine = None
        status, _ = self.request("POST", "/v1/score", {"tokens": [1, 2, 3]})
        self.assertEqual(status, 409)
        svc.engine = eng_ok

        # Busy (FIFO held)
        svc.fifo.acquire()
        try:
            status, body = self.request("POST", "/v1/score", {"tokens": [1, 2, 3]})
            self.assertEqual(status, 409)
            self.assertEqual(body["error"]["type"], "model_busy")
            self.assertEqual(eng_ok.score_calls, [])
        finally:
            svc.fifo.release()

        # Busy reported by engine batch slots
        eng_ok.score_error = ValueError("cannot score while batch slots are actively decoding")
        status, _ = self.request("POST", "/v1/score", {"tokens": [1, 2, 3]})
        self.assertEqual(status, 409)

    def test_400_invalid_inputs_and_engine_errors(self):
        tok = ByteTokenizer()
        eng = FakeScoreEngine(tok)
        self._start(eng)

        bad_payloads = [
            {},
            {"tokens": [1]},
            {"tokens": [1, 2], "prompt": "a", "completion": "b"},
            {"tokens": [1, -1]},
            {"tokens": [1, True]},
            {"tokens": [1, 2, 3], "from": 2},
            {"tokens": [1, 2, 3], "from": -1},
            {"tokens": [1, 2, 3], "lora": "yes"},
            {"tokens": [1, 2, 3], "cvec": 1},
            {"prompt": "hi"},
            {"completion": "hi"},
            {"prompt": "", "completion": "hi"},
            {"prompt": "hi", "completion": ""},
            {"prompt": "hi", "completion": "ok", "from": 0},
        ]
        for payload in bad_payloads:
            status, _ = self.request("POST", "/v1/score", payload)
            self.assertEqual(status, 400, f"expected 400 for {payload!r}")

        # Engine refusal -> 400
        eng.score_error = ValueError("sequence length exceeds context")
        status, err_body = self.request("POST", "/v1/score", {"tokens": [1, 2, 3]})
        self.assertEqual(status, 400)
        self.assertIn("exceeds context", err_body["error"]["message"])

        # Engine crash -> 503
        eng.score_error = EngineDied("the engine stopped unexpectedly")
        status, _ = self.request("POST", "/v1/score", {"tokens": [1, 2, 3]})
        self.assertEqual(status, 503)

    def test_auth_guard(self):
        tok = ByteTokenizer()
        eng = FakeScoreEngine(tok)
        self._start(eng, api_key="secret-token")

        status, _ = self.request("POST", "/v1/score", {"tokens": [1, 2, 3]})
        self.assertEqual(status, 401)

        status, _ = self.request(
            "POST", "/v1/score", {"tokens": [1, 2, 3]},
            headers={"Authorization": "Bearer wrong"},
        )
        self.assertEqual(status, 401)

        status, body = self.request(
            "POST", "/v1/score", {"tokens": [1, 2, 3]},
            headers={"Authorization": "Bearer secret-token"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["count"], 2)


if __name__ == "__main__":
    unittest.main()
