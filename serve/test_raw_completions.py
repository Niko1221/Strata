"""The raw-prompt routes for code completion: POST /completion and POST /infill (llama.cpp's native shapes) and the
`suffix` of POST /v1/completions (fill-in-the-middle).  No model and no GPU: MockEngine replays a scripted answer and
ByteTokenizer (one id per byte, specials from 256) stands in for the pack's tokenizer.

What stays true:

  * the prompt reaches the engine as the client wrote it: a raw prompt's special tokens as single ids; FIM markers as
    their ids, and the same marker text inside the client's code as text; /infill's layout as llama.cpp builds it;
  * the answer comes back byte for byte (no tool-call, thinking or newline handling);
  * /completion and /infill sample with the request's own fields only; up to 16 stop strings on these routes, 4 on chat;
  * t_max_predict_ms ends the answer at a newline; strata_prefix pins a prefix; what is not supported is a 400.

    python -m unittest serve.test_raw_completions -v
"""
from __future__ import annotations

import json
import socket
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service, StrataEngine, serve  # noqa: E402

TEMPLATE = ROOT / "serve" / "chat_template.jinja"
CTX = 4096
FIM = ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>"]


class FimTokenizer(ByteTokenizer):
    """Qwen's FIM and repo-level markers, as CONTROL tokens: matched only with parse_special."""
    SPECIALS = ByteTokenizer.SPECIALS + FIM


class FimOnlyTokenizer(ByteTokenizer):
    """The three FIM markers without <|repo_name|> and <|file_sep|> (/infill's layout without them)."""
    SPECIALS = ByteTokenizer.SPECIALS + ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"]


class RecordingEngine(MockEngine):
    """What the engine was asked: the prompt ids, the sampling dict and max_new of every request."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompts, self.samplings, self.max_news = [], [], []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        self.samplings.append(dict(sampling or {}))
        self.max_news.append(max_new)
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class SlowPromptEngine(MockEngine):
    """Reads a "long prompt" for up to 20 s without a token, stopping on cancel (as serve/test_server.py's)."""
    cancelled_after = None

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.cancelled_after = None
        t0 = time.monotonic()
        while time.monotonic() - t0 < 20:
            if cancel.is_set():
                self.cancelled_after = time.monotonic() - t0
                return
            time.sleep(0.05)
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class DyingEngine(MockEngine):
    """Ends with EngineDied after two tokens: an error after the stream's headers were sent."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        for i, t in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
            if i == 2:
                raise EngineDied("the engine stopped unexpectedly (exit code -9)")
            yield t


class Server:
    """One serve() on a free port for a test class."""

    def __init__(self, tok, script, engine_cls=RecordingEngine, ctx=CTX, delay_s=0.0):
        self.tok = tok
        self.engine = engine_cls(tok, script, max_context=ctx, delay_s=delay_s)
        self.svc = Service(self.engine, tok, ChatTemplate(TEMPLATE))
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body, headers=None):
        """-> (status, JSON body)."""
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def stream(self, path, body):
        """-> the "data: " payloads of a streamed answer, in order (strings: "[DONE]" stays a string)."""
        req = urllib.request.Request(self.base + path, data=json.dumps({**body, "stream": True}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            status = r.status
            raw = r.read().decode()
        assert status == 200, status
        return [event[6:] for event in raw.split("\n\n") if event.startswith("data: ")]


def ids_of(tok, text, special=False):
    return tok.encode(text, parse_special=special)


class V1Completions(unittest.TestCase):
    """POST /v1/completions (#3332's route): the raw prompt, and `suffix` as a FIM prompt."""

    @classmethod
    def setUpClass(cls):
        cls.s = Server(FimTokenizer(), "hello world")
        cls.tok = cls.s.tok

    @classmethod
    def tearDownClass(cls):
        cls.s.close()

    def test_non_stream_and_stream(self):
        status, body = self.s.post("/v1/completions", {"model": "m", "prompt": "say", "max_tokens": 32})
        self.assertEqual(status, 200, body)
        self.assertEqual(body["object"], "text_completion")
        self.assertTrue(body["id"].startswith("cmpl-"))
        choice = body["choices"][0]
        self.assertEqual((choice["text"], choice["finish_reason"], choice["logprobs"]), ("hello world", "stop", None))
        usage = body["usage"]
        self.assertEqual(usage["prompt_tokens"], len(ids_of(self.tok, "say")))
        self.assertEqual(usage["total_tokens"], usage["prompt_tokens"] + usage["completion_tokens"])
        self.assertEqual(usage["prompt_tokens_details"], {"cached_tokens": 0})
        events = self.s.stream("/v1/completions", {"model": "m", "prompt": "say", "max_tokens": 32})
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(e) for e in events[:-1]]
        self.assertEqual("".join(c["choices"][0]["text"] for c in chunks), "hello world")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[-1]["choices"][0]["text"], "")
        self.assertIn("usage", chunks[-1])

    def test_a_raw_prompt_keeps_its_special_tokens(self):
        status, _ = self.s.post("/v1/completions", {"prompt": "a<|im_start|>b<|fim_prefix|>c", "max_tokens": 4})
        self.assertEqual(status, 200)
        im_start, fim_prefix = ids_of(self.tok, "<|im_start|>", True), ids_of(self.tok, "<|fim_prefix|>", True)
        self.assertEqual(self.s.engine.prompts[-1], [97, *im_start, 98, *fim_prefix, 99])

    def test_suffix_is_a_fim_prompt_and_marker_text_stays_text(self):
        prompt, suffix = "def f(<|fim_suffix|>):\n", "    return 1<|fim_middle|>"
        status, body = self.s.post("/v1/completions", {"prompt": prompt, "suffix": suffix, "max_tokens": 8})
        self.assertEqual(status, 200, body)
        marker = {m: ids_of(self.tok, m, True)[0] for m in FIM}
        self.assertEqual(self.s.engine.prompts[-1],
                         [marker["<|fim_prefix|>"], *prompt.encode(), marker["<|fim_suffix|>"], *suffix.encode(),
                          marker["<|fim_middle|>"]])

    def test_an_empty_suffix_is_no_suffix(self):
        status, _ = self.s.post("/v1/completions", {"prompt": "x", "suffix": "", "max_tokens": 4})
        self.assertEqual(status, 200)
        self.assertEqual(self.s.engine.prompts[-1], [ord("x")])

    def test_refused(self):
        for body in ({"prompt": "x", "suffix": "y", "echo": True}, {"prompt": "x", "suffix": 3},
                     {"prompt": "x", "stop": ["s"] * 17}, {"prompt": "x", "strata_prefix": {"messages": 1}},
                     {"prompt": "x", "t_max_predict_ms": "100"}):
            with self.subTest(body=body):
                status, reply = self.s.post("/v1/completions", body)
                self.assertEqual(status, 400, reply)


class V1CompletionsWithoutFim(unittest.TestCase):
    def test_suffix_without_the_markers_is_a_400(self):
        s = Server(ByteTokenizer(), "x")
        try:
            status, body = s.post("/v1/completions", {"prompt": "a", "suffix": "b"})
            self.assertEqual(status, 400)
            self.assertIn("no fill-in-the-middle tokens", json.dumps(body))
            self.assertEqual(s.engine.prompts, [])
        finally:
            s.close()


class Completion(unittest.TestCase):
    """POST /completion: llama.cpp's native shape."""

    @classmethod
    def setUpClass(cls):
        cls.s = Server(FimTokenizer(), "first line\nsecond line")
        cls.tok = cls.s.tok

    @classmethod
    def tearDownClass(cls):
        cls.s.close()

    def test_non_stream(self):
        status, body = self.s.post("/completion", {"prompt": "go<|im_start|>", "n_predict": 64})
        self.assertEqual(status, 200, body)
        self.assertEqual(body["content"], "first line\nsecond line")
        self.assertEqual(body["stop"], True)
        self.assertEqual((body["stop_type"], body["stopping_word"], body["truncated"]), ("eos", "", False))
        self.assertEqual(body["tokens_evaluated"], len(ids_of(self.tok, "go<|im_start|>", True)))
        self.assertEqual(body["tokens_evaluated"], 3)               # "<|im_start|>" is one id
        self.assertEqual(body["tokens_cached"], 0)
        self.assertEqual(body["id_slot"], 0)
        self.assertEqual(body["tokens"], [])
        self.assertGreater(body["tokens_predicted"], 0)
        self.assertEqual(body["generation_settings"]["n_predict"], 64)

    def test_stream_has_pieces_then_the_end_and_no_done(self):
        events = self.s.stream("/completion", {"prompt": "go", "n_predict": 64})
        self.assertNotIn("[DONE]", events)
        items = [json.loads(e) for e in events]                     # every event is JSON (clients parse them all)
        pieces, last = items[:-1], items[-1]
        self.assertTrue(pieces)
        for p in pieces:
            self.assertEqual((p["stop"], p["id_slot"], p["tokens"]), (False, 0, []))
        self.assertEqual("".join(p["content"] for p in pieces), "first line\nsecond line")
        self.assertEqual((last["stop"], last["content"], last["stop_type"]), (True, "", "eos"))

    def test_a_token_id_prompt_and_a_list_of_one_string(self):
        status, _ = self.s.post("/completion", {"prompt": [104, 105, 256], "n_predict": 2})
        self.assertEqual(status, 200)
        self.assertEqual(self.s.engine.prompts[-1], [104, 105, 256])
        status, _ = self.s.post("/completion", {"prompt": ["hi<|im_end|>"], "n_predict": 2})
        self.assertEqual(status, 200)
        self.assertEqual(self.s.engine.prompts[-1], [104, 105, *ids_of(self.tok, "<|im_end|>", True)])

    def test_stop_string_and_limit(self):
        status, body = self.s.post("/completion", {"prompt": "go", "stop": ["\n"]})
        self.assertEqual(status, 200)
        self.assertEqual((body["content"], body["stop_type"], body["stopping_word"]), ("first line", "word", "\n"))
        status, body = self.s.post("/completion", {"prompt": "go", "n_predict": 3})
        self.assertEqual((body["content"], body["stop_type"]), ("fir", "limit"))
        status, body = self.s.post("/completion", {"prompt": "go", "max_tokens": 4})   # max_tokens is read too
        self.assertEqual((body["content"], body["stop_type"]), ("firs", "limit"))

    def test_accepted_with_no_effect(self):
        status, body = self.s.post("/completion", {
            "prompt": "go", "n_predict": 3, "cache_prompt": True, "id_slot": -1, "n_keep": 0, "n_indent": 4,
            "samplers": ["top_k"], "typical_p": 1.0, "dry_multiplier": 0, "mirostat": 0, "xtc_probability": 0,
            "t_max_prompt_ms": 500, "response_fields": ["content"], "timings_per_token": False, "user": "u",
            "stream_options": {"include_usage": True}, "logprobs": False, "n_probs": 0, "unknown_field": 1})
        self.assertEqual(status, 200, body)
        self.assertEqual(body["content"], "fir")

    def test_refused(self):
        for body, says in (({"prompt": "x", "logprobs": 1}, "logprobs"), ({"prompt": "x", "logprobs": True}, "logprobs"),
                           ({"prompt": "x", "n_probs": 3}, "n_probs"), ({"prompt": ["a", "b"]}, "one prompt"),
                           ({"prompt": [[1], [2]]}, "one prompt"), ({"prompt": [104, 999]}, "outside the vocabulary"),
                           ({"prompt": [-1]}, "outside the vocabulary"), ({"prompt": ""}, "empty"),
                           ({"prompt": 5}, "prompt must be"), ({"prompt": "x", "n_predict": 1.5}, "n_predict"),
                           ({"prompt": "x", "stop": ["s"] * 17}, "at most 16"),
                           ({"prompt": "x", "temperature": "warm"}, "temperature")):
            with self.subTest(body=body):
                status, reply = self.s.post("/completion", body)
                self.assertEqual(status, 400, reply)
                self.assertIn(says, reply["error"]["message"])

    def test_the_answer_comes_back_byte_for_byte(self):
        script = "\n\n<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call></think>\n\n  done\n"
        s = Server(FimTokenizer(), script)
        try:
            for path, body, text in (("/completion", {"prompt": "x"}, lambda b: b["content"]),
                                     ("/v1/completions", {"prompt": "x", "max_tokens": 200},
                                      lambda b: b["choices"][0]["text"])):
                with self.subTest(path=path, stream=False):
                    status, reply = s.post(path, body)
                    self.assertEqual(status, 200, reply)
                    self.assertEqual(text(reply), script)
                with self.subTest(path=path, stream=True):
                    items = [json.loads(e) for e in s.stream(path, body) if e != "[DONE]"]
                    got = "".join(i["content"] if path == "/completion" else i["choices"][0]["text"] for i in items)
                    self.assertEqual(got, script)
        finally:
            s.close()


class Infill(unittest.TestCase):
    """POST /infill: llama.cpp's format_prompt_infill layout, exactly."""

    @classmethod
    def setUpClass(cls):
        cls.s = Server(FimTokenizer(), "return 1")
        cls.tok = cls.s.tok
        cls.m = {m: ids_of(cls.tok, m, True)[0] for m in FIM}

    @classmethod
    def tearDownClass(cls):
        cls.s.close()

    def test_the_repo_level_layout(self):
        m = self.m
        extra = [{"filename": "a.py", "text": "x = 1\n"}, {"text": "y"}]
        status, body = self.s.post("/infill", {"input_prefix": "def f():\n    ", "input_suffix": "\nprint(f())",
                                               "input_extra": extra, "prompt": "re<|fim_pad|>"})
        self.assertEqual(status, 200, body)
        self.assertEqual(self.s.engine.prompts[-1], [
            m["<|repo_name|>"], *b"myproject\n",
            m["<|file_sep|>"], *b"a.py\n", *b"x = 1\n",
            m["<|file_sep|>"], *b"tmp\n", *b"y",                   # a chunk without a filename: llama.cpp's "tmp"
            m["<|file_sep|>"], *b"filename\n",
            m["<|fim_prefix|>"], *b"def f():\n    ", *b"re", m["<|fim_pad|>"],   # the prompt follows, special tokens parsed
            m["<|fim_suffix|>"], *b"\nprint(f())", m["<|fim_middle|>"]])
        self.assertEqual((body["content"], body["stop"], body["stop_type"]), ("return 1", True, "eos"))

    def test_without_extra_context(self):
        m = self.m
        status, _ = self.s.post("/infill", {"input_prefix": "a", "input_suffix": "b"})
        self.assertEqual(status, 200)
        self.assertEqual(self.s.engine.prompts[-1], [m["<|repo_name|>"], *b"myproject\n", m["<|file_sep|>"],
                                                     *b"filename\n", m["<|fim_prefix|>"], ord("a"), m["<|fim_suffix|>"],
                                                     ord("b"), m["<|fim_middle|>"]])

    def test_marker_text_in_the_code_stays_text(self):
        m = self.m
        status, _ = self.s.post("/infill", {"input_prefix": "<|fim_middle|>", "input_suffix": "<|file_sep|>",
                                            "input_extra": [{"filename": "<|repo_name|>", "text": "<|im_end|>"}]})
        self.assertEqual(status, 200)
        prompt = self.s.engine.prompts[-1]
        self.assertEqual([t for t in prompt if t >= 256],
                         [m["<|repo_name|>"], m["<|file_sep|>"], m["<|file_sep|>"], m["<|fim_prefix|>"],
                          m["<|fim_suffix|>"], m["<|fim_middle|>"]])

    def test_stream_ends_without_done(self):
        events = self.s.stream("/infill", {"input_prefix": "a", "input_suffix": "b"})
        self.assertNotIn("[DONE]", events)
        items = [json.loads(e) for e in events]
        self.assertEqual("".join(i["content"] for i in items), "return 1")
        self.assertTrue(items[-1]["stop"])

    def test_refused(self):
        for body in ({"input_suffix": "b"}, {"input_prefix": "a"}, {"input_prefix": 1, "input_suffix": "b"},
                     {"input_prefix": "a", "input_suffix": "b", "prompt": 3},
                     {"input_prefix": "a", "input_suffix": "b", "input_extra": {"text": "x"}},
                     {"input_prefix": "a", "input_suffix": "b", "input_extra": [{"filename": "f"}]},
                     {"input_prefix": "a", "input_suffix": "b", "input_extra": [{"text": "x", "filename": 2}]}):
            with self.subTest(body=body):
                status, reply = self.s.post("/infill", body)
                self.assertEqual(status, 400, reply)


class InfillOtherVocabularies(unittest.TestCase):
    def test_without_repo_and_file_tokens(self):
        s = Server(FimOnlyTokenizer(), "z")
        try:
            m = {k: ids_of(s.tok, k, True)[0] for k in ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")}
            status, _ = s.post("/infill", {"input_prefix": "a", "input_suffix": "b",
                                           "input_extra": [{"filename": "f.py", "text": "x"}]})
            self.assertEqual(status, 200)
            self.assertEqual(s.engine.prompts[-1], [*b"\n\n--- snippet ---\n\n", *b"x", m["<|fim_prefix|>"], ord("a"),
                                                    m["<|fim_suffix|>"], ord("b"), m["<|fim_middle|>"]])
        finally:
            s.close()

    def test_without_fim_tokens(self):
        s = Server(ByteTokenizer(), "z")
        try:
            status, body = s.post("/infill", {"input_prefix": "a", "input_suffix": "b"})
            self.assertEqual(status, 400)
            self.assertEqual(body["error"]["message"],
                             "this model has no fill-in-the-middle tokens (<|fim_prefix|>, <|fim_suffix|>, <|fim_middle|>)")
            self.assertEqual(s.engine.prompts, [])
        finally:
            s.close()


class Sampling(unittest.TestCase):
    """/completion and /infill send the engine the request's own sampling; chat's defaults are not added."""

    @classmethod
    def setUpClass(cls):
        cls.s = Server(FimTokenizer(), "ok")
        cls.s.svc.model_arch = "qwen4exp"                             # the card's instruct set: temperature 0.7, ...
        cls.s.svc.sampling_defaults = {"temperature": 0.6, "presence_penalty": 1.5}   # a config's "sampling" block

    @classmethod
    def tearDownClass(cls):
        cls.s.close()

    def test_nothing_is_added(self):
        for path, body in (("/completion", {"prompt": "x"}), ("/infill", {"input_prefix": "a", "input_suffix": "b"})):
            with self.subTest(path=path):
                status, _ = self.s.post(path, body)
                self.assertEqual(status, 200)
                sampling = self.s.engine.samplings[-1]
                for key in ("temperature", "top_p", "top_k", "presence_penalty"):
                    self.assertNotIn(key, sampling)
                self.assertEqual(StrataEngine.sampling_keys(sampling), "")    # greedy, no penalties

    def test_sent_values_pass_through(self):
        status, _ = self.s.post("/completion", {"prompt": "x", "temperature": 0.2, "top_k": 10, "repeat_penalty": 1.1,
                                                "repeat_last_n": 32})
        self.assertEqual(status, 200)
        keys = StrataEngine.sampling_keys(self.s.engine.samplings[-1])
        for want in (" temperature=0.2", " top_k=10", " penalty_repeat=1.1", " penalty_last_n=32"):
            self.assertIn(want, keys)
        self.assertNotIn("penalty_present", keys)

    def test_v1_completions_keeps_the_defaults(self):
        # #3332's route merges them, as chat does; a FIM client there sends its own values
        status, _ = self.s.post("/v1/completions", {"prompt": "x", "suffix": "y", "max_tokens": 4})
        self.assertEqual(status, 200)
        self.assertEqual(self.s.engine.samplings[-1]["presence_penalty"], 1.5)


class Stops(unittest.TestCase):
    NINE = ["<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>",
            "<|file_sep|>", "<|im_start|>", "<|im_end|>"]

    @classmethod
    def setUpClass(cls):
        cls.s = Server(FimTokenizer(), "abc def")

    @classmethod
    def tearDownClass(cls):
        cls.s.close()

    def test_nine_stop_strings_on_the_raw_routes(self):
        for path, body, text in (("/completion", {"prompt": "x"}, lambda b: b["content"]),
                                 ("/infill", {"input_prefix": "a", "input_suffix": "b"}, lambda b: b["content"]),
                                 ("/v1/completions", {"prompt": "x", "max_tokens": 32},
                                  lambda b: b["choices"][0]["text"])):
            with self.subTest(path=path, stream=False):
                status, reply = self.s.post(path, {**body, "stop": self.NINE})
                self.assertEqual(status, 200, reply)
                self.assertEqual(text(reply), "abc def")
            with self.subTest(path=path, stream=True):
                events = self.s.stream(path, {**body, "stop": self.NINE})
                items = [json.loads(e) for e in events if e != "[DONE]"]
                self.assertFalse([i for i in items if "error" in i], items)   # ran to its end, no error event
                self.assertEqual(events[-1] == "[DONE]", path == "/v1/completions")

    def test_a_stop_string_cuts_the_answer(self):
        status, body = self.s.post("/v1/completions", {"prompt": "x", "stop": [*self.NINE, " "], "max_tokens": 32})
        self.assertEqual((body["choices"][0]["text"], body["choices"][0]["finish_reason"]), ("abc", "stop"))
        status, body = self.s.post("/infill", {"input_prefix": "a", "input_suffix": "b", "stop": ["c d"]})
        self.assertEqual((body["content"], body["stop_type"], body["stopping_word"]), ("ab", "word", "c d"))

    def test_chat_keeps_four(self):
        status, body = self.s.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                                                           "stop": ["a", "b", "c", "d", "e"]})
        self.assertEqual(status, 400)
        self.assertIn("at most 4", body["error"]["message"])


class TimeLimit(unittest.TestCase):
    """llama.cpp's t_max_predict_ms: past the limit, the answer ends at the next newline ("limit" / "length")."""

    @classmethod
    def setUpClass(cls):
        cls.s = Server(FimTokenizer(), "aaaa\nbbbb\ncccc\n", delay_s=0.02)   # the first newline 80 ms after the first token

    @classmethod
    def tearDownClass(cls):
        cls.s.close()

    def test_ends_at_a_newline(self):
        status, body = self.s.post("/infill", {"input_prefix": "a", "input_suffix": "b", "t_max_predict_ms": 1})
        self.assertEqual(status, 200)
        self.assertEqual((body["content"], body["stop_type"]), ("aaaa\n", "limit"))
        status, body = self.s.post("/v1/completions", {"prompt": "x", "max_tokens": 64, "t_max_predict_ms": 1})
        self.assertEqual((body["choices"][0]["text"], body["choices"][0]["finish_reason"]), ("aaaa\n", "length"))

    def test_a_limit_not_reached(self):
        status, body = self.s.post("/completion", {"prompt": "x", "t_max_predict_ms": 60000})
        self.assertEqual((body["content"], body["stop_type"]), ("aaaa\nbbbb\ncccc\n", "eos"))


class Prefix(unittest.TestCase):
    """strata_prefix on the raw routes turns into the engine's pin=N."""

    @classmethod
    def setUpClass(cls):
        cls.s = Server(FimTokenizer(), "ok")

    @classmethod
    def tearDownClass(cls):
        cls.s.close()

    def pin(self):
        sampling = self.s.engine.samplings[-1]
        keys = StrataEngine.sampling_keys(sampling)
        return sampling.get("strata_prefix"), keys

    def test_tokens_and_chars(self):
        self.s.post("/completion", {"prompt": "hello world", "strata_prefix": {"tokens": 3}})
        self.assertEqual(self.pin(), ({"tokens": 3}, " pin=3"))
        self.s.post("/completion", {"prompt": "hello world", "strata_prefix": {"chars": 5}})
        self.assertEqual(self.pin(), ({"tokens": 5}, " pin=5"))
        # /infill: the first N characters of input_prefix, after the extra context and <|fim_prefix|>
        extra = [{"filename": "a.py", "text": "import os\n"}]
        self.s.post("/infill", {"input_prefix": "def f():\n", "input_suffix": "x", "input_extra": extra,
                                "strata_prefix": {"chars": 4}})
        start = self.s.engine.prompts[-1].index(ids_of(self.s.tok, "<|fim_prefix|>", True)[0]) + 1
        self.assertEqual(self.pin(), ({"tokens": start + 4}, f" pin={start + 4}"))
        # /v1/completions with suffix: of prompt, after <|fim_prefix|>
        self.s.post("/v1/completions", {"prompt": "abcdef", "suffix": "z", "max_tokens": 2, "strata_prefix": {"chars": 3}})
        self.assertEqual(self.pin(), ({"tokens": 4}, " pin=4"))

    def test_one_that_leaves_no_prefix_is_ignored(self):
        for prefix in ({"tokens": 1000}, {"chars": 1000}):
            with self.subTest(prefix=prefix):
                status, _ = self.s.post("/completion", {"prompt": "hello", "strata_prefix": prefix})
                self.assertEqual(status, 200)
                self.assertEqual(self.pin(), ({}, ""))

    def test_malformed_is_a_400(self):
        for prefix in ({"tokens": "3"}, {"messages": 1}, {"tokens": 0}, {"tokens": 2, "chars": 2}, [3], {"chars": True}):
            with self.subTest(prefix=prefix):
                status, _ = self.s.post("/completion", {"prompt": "hello", "strata_prefix": prefix})
                self.assertEqual(status, 400)


class NeverTruncated(unittest.TestCase):
    def test_a_prompt_that_does_not_fit(self):
        s = Server(FimTokenizer(), "ok", ctx=64)
        try:
            for path, body in (("/completion", {"prompt": "x" * 100}),
                               ("/infill", {"input_prefix": "x" * 100, "input_suffix": "y"}),
                               ("/v1/completions", {"prompt": "x" * 100, "suffix": "y"})):
                with self.subTest(path=path):
                    status, reply = s.post(path, body)
                    self.assertEqual(status, 400, reply)
                    self.assertIn("never truncated", reply["error"]["message"])
            self.assertEqual(s.engine.prompts, [])
            status, body = s.post("/completion", {"prompt": "x" * 40, "n_predict": 1000})   # fits: max tokens shortened
            self.assertEqual(status, 200)
            self.assertEqual(s.engine.max_news[-1], 64 - 8 - 40)
        finally:
            s.close()


class Safety(unittest.TestCase):
    """The same admission, origin and key checks as every inference route."""

    def test_a_foreign_page_and_the_api_key(self):
        s = Server(FimTokenizer(), "ok")
        try:
            for path in ("/completion", "/infill"):
                with self.subTest(path=path):
                    status, _ = s.post(path, {"prompt": "x", "input_prefix": "a", "input_suffix": "b"},
                                       headers={"Origin": "http://evil.example"})
                    self.assertEqual(status, 403)
            s.svc.api_key = "secret"
            for path in ("/completion", "/infill"):
                with self.subTest(path=path, key=False):
                    status, _ = s.post(path, {"prompt": "x", "input_prefix": "a", "input_suffix": "b"})
                    self.assertEqual(status, 401)
                with self.subTest(path=path, key=True):
                    status, _ = s.post(path, {"prompt": "x", "input_prefix": "a", "input_suffix": "b"},
                                       headers={"Authorization": "Bearer secret"})
                    self.assertEqual(status, 200)
            self.assertEqual(len(s.engine.prompts), 2)
        finally:
            s.close()

    def test_the_monitor_records_the_output(self):
        s = Server(FimTokenizer(), "out")
        s.svc.api_monitor = True
        try:
            for path, body in (("/completion", {"prompt": "x"}), ("/infill", {"input_prefix": "a", "input_suffix": "b"})):
                for stream in (False, True):
                    with self.subTest(path=path, stream=stream):
                        if stream:
                            s.stream(path, body)
                        else:
                            s.post(path, body)
                        record = s.svc.request_records(s.svc.request_records()[0]["id"])
                        self.assertEqual((record["path"], record["output"]), (path, "out"))
                        self.assertEqual(record["usage"]["completion_tokens"], s.svc.history[-1]["output_tokens"])
        finally:
            s.close()

    def test_an_error_mid_stream_has_no_done(self):
        s = Server(FimTokenizer(), "abcdef", engine_cls=DyingEngine)
        try:
            events = s.stream("/completion", {"prompt": "x"})
            self.assertNotIn("[DONE]", events)
            items = [json.loads(e) for e in events]
            self.assertIn("error", items[-1])
            self.assertEqual("".join(i.get("content", "") for i in items[:-1]), "ab")
        finally:
            s.close()


class ClientHangUp(unittest.TestCase):
    """#430 #431: a client that hangs up during the prompt read cancels the request, as on chat."""

    @classmethod
    def setUpClass(cls):
        cls.s = Server(FimTokenizer(), "OK", engine_cls=SlowPromptEngine)

    @classmethod
    def tearDownClass(cls):
        cls.s.close()

    def hang_up(self, path, stream):
        body = json.dumps({"prompt": "a long prompt", "input_prefix": "a", "input_suffix": "b", "stream": stream}).encode()
        self.s.engine.cancelled_after = None
        c = socket.create_connection(("127.0.0.1", self.s.port))
        c.sendall(f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n".encode()
                  + b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        time.sleep(1.0)
        c.close()
        t0 = time.monotonic()
        while self.s.engine.cancelled_after is None and time.monotonic() - t0 < 10:
            time.sleep(0.05)
        self.assertIsNotNone(self.s.engine.cancelled_after, "the request was not cancelled")
        self.assertLess(self.s.engine.cancelled_after, 3.0)

    def test_completion(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.hang_up("/completion", stream)

    def test_infill(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.hang_up("/infill", stream)


if __name__ == "__main__":
    unittest.main()
