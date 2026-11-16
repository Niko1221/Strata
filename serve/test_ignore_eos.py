"""ignore_eos (vLLM's request field): the reply runs to its token budget, over the model's stop tokens.  No GPU, no pack.

    python -m unittest serve.test_ignore_eos -v
"""
import http.client
import json
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import IM_END, ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]
APIS = (("/v1/chat/completions", "max_tokens"),
        ("/v1/messages", "max_tokens"),
        ("/v1/responses", "max_output_tokens"))


class LegEngine(MockEngine):
    """The script and a stop token on every generate, as a model that ends its turn.  Records each leg's prompt and
    budget, and sets `last` as a real engine's DONE does (leg by leg: only the first read the prompt)."""

    def __init__(self, tok, script="ok", **kw):
        super().__init__(tok, script, **kw)
        self.legs = []
        self.clock = False
        self.pause = 0.0            # seconds before each token, as a pass that takes time

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.legs.append((list(ids), max_new))
        first = len(self.legs) == 1
        made = 0
        try:
            for t in self.script[:max_new]:
                if self.pause:
                    time.sleep(self.pause)
                made += 1
                yield t
        finally:
            if self.clock:
                self.last = {"prompt_ms": 100.0 if first else 2.0, "decode_ms": 50.0, "generated": made,
                             "reused": 1 if first else len(ids) - 1, "prompt_read": len(ids) - 1 if first else 1}


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.end = cls.tok.encode(IM_END, parse_special=True)
        cls.engine = LegEngine(cls.tok, max_context=4096)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.engine.legs = []
        self.engine.clock = False
        self.engine.pause = 0.0
        self.engine.script = self.tok.encode("ok", parse_special=True) + self.end
        self.engine.last = None
        self.svc.fit_max_tokens = True
        self.svc.repeat_stop_tokens = 256

    def post(self, path, body):
        body = dict(body)
        if path == "/v1/responses":
            body.setdefault("input", "hi")
        else:
            body.setdefault("messages", [{"role": "user", "content": "hi"}])
            body.setdefault("chat_template_kwargs", {"enable_thinking": False})     # the script is the answer
            if path == "/v1/messages":
                body.setdefault("thinking", {"type": "disabled"})
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        try:
            c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            raw = r.read().decode()
            return r.status, raw
        finally:
            c.close()

    def chat(self, **body):
        status, raw = self.post("/v1/chat/completions", body)
        self.assertEqual(status, 200, raw)
        return json.loads(raw)

    def chat_stream(self, **body):
        status, raw = self.post("/v1/chat/completions", {**body, "stream": True})
        self.assertEqual(status, 200, raw)
        chunks = [json.loads(line[6:]) for line in raw.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
        finish = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
        usage = [c["usage"] for c in chunks if c.get("usage")]
        return text, finish[-1], usage[-1]


class Default(Base):
    def test_absent_false_and_null_stop_at_the_stop_token(self):
        for extra in ({}, {"ignore_eos": False}, {"ignore_eos": None}):
            with self.subTest(extra=extra):
                self.engine.legs = []
                got = self.chat(max_tokens=8, **extra)
                self.assertEqual(len(self.engine.legs), 1)
                self.assertEqual(got["choices"][0]["finish_reason"], "stop")
                self.assertEqual(got["choices"][0]["message"]["content"], "ok")
                self.assertEqual(got["usage"]["completion_tokens"], 3)      # o, k and the stop token, as always

    def test_false_needs_no_budget(self):
        self.assertEqual(self.chat(ignore_eos=False)["choices"][0]["finish_reason"], "stop")


class Continuation(Base):
    def test_each_leg_is_the_prompt_and_all_that_was_produced_with_the_budget_left(self):
        got = self.chat(max_tokens=8, ignore_eos=True)
        o, k, e = ord("o"), ord("k"), self.end[0]
        prompt = self.engine.legs[0][0]
        self.assertEqual([b for _, b in self.engine.legs], [8, 5, 2])
        self.assertEqual(self.engine.legs[1][0], prompt + [o, k, e])
        self.assertEqual(self.engine.legs[2][0], prompt + [o, k, e, o, k, e])
        self.assertEqual(got["choices"][0]["finish_reason"], "length")
        self.assertEqual(got["choices"][0]["message"]["content"], "okokok")     # no stop token in the text
        self.assertEqual(got["usage"]["completion_tokens"], 8)                  # the two stop tokens count
        self.assertEqual(got["usage"]["prompt_tokens"], len(prompt))            # the real prompt, not with the output

    def test_a_stop_token_as_the_last_of_the_budget_is_length_and_no_more_legs(self):
        got = self.chat(max_tokens=3, ignore_eos=True)
        self.assertEqual(len(self.engine.legs), 1)
        self.assertEqual(got["choices"][0]["finish_reason"], "length")
        self.assertEqual(got["usage"]["completion_tokens"], 3)

    def test_stream_and_whole_reply_agree(self):
        whole = self.chat(max_tokens=8, ignore_eos=True)
        self.engine.legs = []
        text, finish, usage = self.chat_stream(max_tokens=8, ignore_eos=True)
        self.assertEqual(text, whole["choices"][0]["message"]["content"])
        self.assertEqual(finish, whole["choices"][0]["finish_reason"])
        self.assertEqual(usage["completion_tokens"], whole["usage"]["completion_tokens"])
        self.assertEqual(usage["prompt_tokens"], whole["usage"]["prompt_tokens"])

    def test_max_completion_tokens_is_a_budget(self):
        got = self.chat(max_completion_tokens=5, ignore_eos=True)
        self.assertEqual(got["usage"]["completion_tokens"], 5)

    def test_figures_over_the_legs(self):
        self.engine.clock = True
        got = self.chat(max_tokens=8, ignore_eos=True)
        prompt = len(self.engine.legs[0][0])
        self.assertEqual(got["usage"]["prompt_tokens"], prompt)
        self.assertEqual(got["usage"]["prompt_tokens_details"]["cached_tokens"], 1)     # the first leg's, not its own output
        t = got["timings"]
        self.assertEqual(t["cache_n"], 1)
        self.assertEqual(t["prompt_n"], prompt - 1)
        self.assertEqual(t["predicted_n"], 8)
        self.assertEqual(t["predicted_ms"], 154.0)                                      # three legs, two reads again
        self.assertEqual(t["predicted_per_second"], round(8 / 0.154, 1))
        self.assertEqual(t["prompt_ms"], 100.0)                                         # the prompt's own read
        self.assertEqual(t["prompt_per_second"], round((prompt - 1) / 0.1, 1))


    def test_a_control_token_past_a_stop_token_is_counted_and_not_text(self):
        self.engine.script = self.tok.encode("<|im_start|>x", parse_special=True) + self.end
        got = self.chat(max_tokens=6, ignore_eos=True)
        self.assertEqual(got["usage"]["completion_tokens"], 6)
        self.assertEqual(got["choices"][0]["message"]["content"], "<|im_start|>xx")     # the first pass is as always
        self.assertEqual(self.engine.legs[1][0][-3:], self.engine.script)


class Ends(Base):
    def test_a_client_stop_string_still_ends_the_reply_with_stop(self):
        got = self.chat(max_tokens=64, ignore_eos=True, stop=["okok"])
        self.assertEqual(got["choices"][0]["finish_reason"], "stop")
        self.assertFalse(got["choices"][0]["message"]["content"])               # found across the stop token
        self.assertEqual(len(self.engine.legs), 2)
        self.engine.legs = []
        text, finish, _ = self.chat_stream(max_tokens=64, ignore_eos=True, stop=["okok"])
        self.assertEqual((text, finish), ("", "stop"))

    def test_a_stop_string_in_newlines_the_parser_held_back_is_stop(self):
        # Seen on the real engine: "\n", stop token, "\n", stop token ... and "stop": ["\n"]
        self.engine.script = self.tok.encode("\n", parse_special=True) + self.end
        got = self.chat(max_tokens=8, ignore_eos=True, stop=["\n"])
        self.assertEqual(got["choices"][0]["finish_reason"], "stop")
        self.assertFalse(got["choices"][0]["message"]["content"])
        self.assertEqual(self.chat_stream(max_tokens=8, ignore_eos=True, stop=["\n"])[1], "stop")

    def test_the_repeat_guard_counts_stop_tokens_and_ends_as_length(self):
        self.svc.repeat_stop_tokens = 4
        self.engine.script = self.end
        got = self.chat(max_tokens=100, ignore_eos=True)
        self.assertEqual(got["choices"][0]["finish_reason"], "length")
        self.assertEqual(got["usage"]["completion_tokens"], 4)
        self.assertEqual(len(self.engine.legs), 4)

    def test_the_repeat_guard_off_runs_to_the_budget(self):
        self.svc.repeat_stop_tokens = 0
        self.engine.script = self.end
        got = self.chat(max_tokens=6, ignore_eos=True)
        self.assertEqual(got["usage"]["completion_tokens"], 6)

    def test_the_context_limit_shortens_the_budget(self):
        got = self.chat(max_tokens=100000, ignore_eos=True)
        prompt = len(self.engine.legs[0][0])
        room = self.engine.legs[0][1]
        self.assertLess(prompt + room, self.engine.max_context)
        self.assertEqual(got["usage"]["completion_tokens"], room)
        self.assertEqual(got["choices"][0]["finish_reason"], "length")
        self.assertLessEqual(max(len(p) + b for p, b in self.engine.legs), prompt + room)

    def test_over_the_context_without_fit_is_refused_as_before(self):
        self.svc.fit_max_tokens = False
        status, raw = self.post("/v1/chat/completions", {"max_tokens": 100000, "ignore_eos": True})
        self.assertEqual(status, 400, raw)
        self.assertIn("exceeds the context window", raw)
        self.assertEqual(self.engine.legs, [])

    def test_a_cancelled_request_sends_no_more_legs(self):
        cancel = threading.Event()
        ids = self.tok.encode("hi", parse_special=True)
        run = self.svc.run(ids, False, None, 100, {"ignore_eos": True}, cancel)
        got = []
        for kind, x in run:
            if kind == "event":
                got.append(x)
                cancel.set()
            elif kind == "done":
                self.assertEqual(x["finish"], "cancel")
        self.assertEqual(len(self.engine.legs), 1)


class Refused(Base):
    def test_a_value_that_is_not_a_boolean(self):
        for path, key in APIS:
            for value in ("true", 1, 0, [], {}):
                for stream in (False, True):
                    with self.subTest(path=path, value=value, stream=stream):
                        status, raw = self.post(path, {key: 8, "ignore_eos": value, "stream": stream})
                        self.assertEqual(status, 400, raw)
                        self.assertIn("ignore_eos", raw)
        self.assertEqual(self.engine.legs, [])

    def test_no_token_budget(self):
        for path, key in APIS:
            for budget in ({}, {key: 0}, {key: -1}):
                for stream in (False, True):
                    with self.subTest(path=path, budget=budget, stream=stream):
                        status, raw = self.post(path, {**budget, "ignore_eos": True, "stream": stream})
                        self.assertEqual(status, 400, raw)
                        self.assertIn("ignore_eos", raw)
                        self.assertIn(key, raw)
        self.assertEqual(self.engine.legs, [])

    def test_logprobs(self):
        for extra in ({"logprobs": True}, {"logprobs": True, "top_logprobs": 2}, {"logprobs": 2}):
            for stream in (False, True):
                with self.subTest(extra=extra, stream=stream):
                    status, raw = self.post("/v1/chat/completions", {"max_tokens": 8, "ignore_eos": True, "stream": stream, **extra})
                    self.assertEqual(status, 400, raw)
                    self.assertIn("ignore_eos", raw)
                    self.assertIn("logprobs", raw)
        self.assertEqual(self.engine.legs, [])

    def test_responses_says_which_field(self):
        status, raw = self.post("/v1/responses", {"ignore_eos": True})
        self.assertEqual(status, 400, raw)
        self.assertEqual(json.loads(raw)["error"]["param"], "ignore_eos")

    def test_the_responses_budget_is_its_own_field(self):
        status, raw = self.post("/v1/responses", {"max_tokens": 8, "ignore_eos": True})
        self.assertEqual(status, 400, raw)


class RefusedBeside(Base):
    """A reply that runs over its stop tokens cannot be a structured answer or a tool round: 400s before generation."""
    CHAT_TOOL = {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}
    RESPONSES_TOOL = {"type": "function", "name": "f", "parameters": {"type": "object", "properties": {}}}
    ANTHROPIC_TOOL = {"name": "f", "input_schema": {"type": "object", "properties": {}}}
    SCHEMA = {"name": "s", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}

    def refused(self, path, budget, field, **extra):
        for stream in (False, True):
            with self.subTest(path=path, extra=extra, stream=stream):
                status, raw = self.post(path, {budget: 8, "ignore_eos": True, "stream": stream, **extra})
                self.assertEqual(status, 400, raw)
                self.assertIn("ignore_eos", raw)
                self.assertIn(field, raw)
        self.assertEqual(self.engine.legs, [])      # nothing was generated

    def allowed(self, path, budget, **extra):
        self.engine.legs = []
        status, raw = self.post(path, {budget: 8, "ignore_eos": True, **extra})
        self.assertEqual(status, 200, raw)
        self.assertGreater(len(self.engine.legs), 1, "a stop token did not start another pass")

    def test_response_format_on_chat(self):
        self.refused("/v1/chat/completions", "max_tokens", "response_format", response_format={"type": "json_object"})
        self.refused("/v1/chat/completions", "max_tokens", "response_format",
                     response_format={"type": "json_schema", "json_schema": self.SCHEMA})

    def test_text_format_on_responses(self):
        self.refused("/v1/responses", "max_output_tokens", "text.format", text={"format": {"type": "json_object"}})
        self.refused("/v1/responses", "max_output_tokens", "text.format",
                     text={"format": {"type": "json_schema", **self.SCHEMA}})

    def test_plain_text_format_is_no_format(self):
        self.allowed("/v1/chat/completions", "max_tokens", response_format={"type": "text"})
        self.allowed("/v1/responses", "max_output_tokens", text={"format": {"type": "text"}})

    def test_tools_on_each_route(self):
        self.refused("/v1/chat/completions", "max_tokens", "tools", tools=[self.CHAT_TOOL])
        self.refused("/v1/messages", "max_tokens", "tools", tools=[self.ANTHROPIC_TOOL])
        self.refused("/v1/responses", "max_output_tokens", "tools", tools=[self.RESPONSES_TOOL])

    def test_an_empty_tools_list_is_no_tools(self):
        for path, budget in APIS:
            with self.subTest(path=path):
                self.allowed(path, budget, tools=[])

    def test_tool_choice_none_offers_no_tools(self):
        self.allowed("/v1/chat/completions", "max_tokens", tools=[self.CHAT_TOOL], tool_choice="none")
        self.allowed("/v1/responses", "max_output_tokens", tools=[self.RESPONSES_TOOL], tool_choice="none")
        self.allowed("/v1/messages", "max_tokens", tools=[self.ANTHROPIC_TOOL], tool_choice={"type": "none"})

    def test_strata_mcp(self):
        self.refused("/v1/chat/completions", "max_tokens", "strata_mcp", strata_mcp=True)

    def test_strata_mcp_false_is_no_mcp(self):
        self.allowed("/v1/chat/completions", "max_tokens", strata_mcp=False)

    def test_responses_says_ignore_eos_is_the_param(self):
        status, raw = self.post("/v1/responses", {"max_output_tokens": 8, "ignore_eos": True, "tools": [self.RESPONSES_TOOL]})
        self.assertEqual(status, 400, raw)
        self.assertEqual(json.loads(raw)["error"]["param"], "ignore_eos")

    def test_without_the_flag_none_of_these_is_touched(self):
        for flag in ({}, {"ignore_eos": False}, {"ignore_eos": None}):
            for extra in ({"tools": [self.CHAT_TOOL]}, {"strata_mcp": True}, {"response_format": {"type": "json_object"}}):
                with self.subTest(flag=flag, extra=extra):
                    self.engine.legs = []
                    self.engine.script = self.tok.encode('{"a":"b"}', parse_special=True) + self.end
                    status, raw = self.post("/v1/chat/completions", {"max_tokens": 32, **flag, **extra})
                    self.assertEqual(status, 200, raw)
                    self.assertEqual(len(self.engine.legs), 1)


class KeepAlive(Base):
    """The idle clock runs over the passes.  A pass is short, so a clock restarted on each would never reach KEEPALIVE_S."""
    NEWLINE = "\n"

    def keep_alives(self, **body):
        with mock.patch("serve.server.KEEPALIVE_S", 0.1):
            status, raw = self.post("/v1/chat/completions", {"stream": True, **body})
        self.assertEqual(status, 200, raw)
        self.assertIn('"finish_reason":"length"', raw.replace(" ", ""))
        return raw.count(": keep-alive")

    def test_passes_of_held_back_new_lines(self):
        self.engine.script = self.tok.encode(self.NEWLINE, parse_special=True) + self.end
        self.engine.pause = 0.02            # each pass takes 0.04 s: never 0.1 s by itself
        self.assertGreaterEqual(self.keep_alives(max_tokens=40, ignore_eos=True), 3)

    def test_passes_of_only_a_stop_token(self):
        self.engine.script = list(self.end)
        self.engine.pause = 0.03
        self.svc.repeat_stop_tokens = 0     # the guard would end it early; and these leave before the keep-alive line
        self.assertGreaterEqual(self.keep_alives(max_tokens=20, ignore_eos=True), 3)

    def test_passes_of_control_tokens_the_text_leaves_out(self):
        im_start = self.tok.encode(self.tok.control_tokens[0], parse_special=True)
        self.engine.script = im_start + im_start + self.end     # the two tokens make no text and no event
        self.engine.pause = 0.02
        self.svc.repeat_stop_tokens = 0
        self.assertGreaterEqual(self.keep_alives(max_tokens=30, ignore_eos=True), 3)

    def test_text_going_out_still_restarts_the_clock(self):
        self.engine.pause = 0.02            # "ok" and a stop token: text at least every 0.06 s, 0.6 s in all
        self.assertLessEqual(self.keep_alives(max_tokens=30, ignore_eos=True), 2)   # (a busy machine can stall once or twice)

    def test_without_the_flag_one_pass_of_new_lines_is_as_before(self):
        self.engine.script = self.tok.encode(self.NEWLINE * 12, parse_special=True) + self.end
        self.engine.pause = 0.02
        with mock.patch("serve.server.KEEPALIVE_S", 0.1):
            status, raw = self.post("/v1/chat/completions", {"stream": True, "max_tokens": 12})
        self.assertEqual(status, 200, raw)
        self.assertGreaterEqual(raw.count(": keep-alive"), 1)
        self.assertEqual(len(self.engine.legs), 1)


class After(Base):
    def test_a_normal_request_after_one_is_as_before(self):
        before = self.chat(max_tokens=8)
        plain_prompt = self.engine.legs[0][0]
        self.engine.legs = []
        self.chat(max_tokens=8, ignore_eos=True)
        self.engine.legs = []
        after = self.chat(max_tokens=8)
        self.assertEqual(len(self.engine.legs), 1)
        self.assertEqual(self.engine.legs[0][0], plain_prompt)
        self.assertEqual(after["choices"][0]["finish_reason"], "stop")
        self.assertEqual(after["choices"][0]["message"], before["choices"][0]["message"])
        self.assertEqual(after["usage"], before["usage"])


class Routes(Base):
    def test_responses(self):
        status, raw = self.post("/v1/responses", {"max_output_tokens": 8, "ignore_eos": True})
        self.assertEqual(status, 200, raw)
        got = json.loads(raw)
        self.assertEqual(got["status"], "incomplete")
        self.assertEqual(got["incomplete_details"]["reason"], "max_output_tokens")
        self.assertEqual(got["usage"]["output_tokens"], 8)
        self.assertEqual(got["usage"]["input_tokens"], len(self.engine.legs[0][0]))
        self.assertEqual([b for _, b in self.engine.legs], [8, 5, 2])
        text = "".join(p.get("text", "") for item in got["output"] for p in item.get("content") or [])
        self.assertEqual(text, "okokok")

    def test_responses_stream(self):
        status, raw = self.post("/v1/responses", {"max_output_tokens": 8, "ignore_eos": True, "stream": True})
        self.assertEqual(status, 200, raw)
        self.assertEqual(len(self.engine.legs), 3)
        done = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")][-1]
        self.assertEqual(done["response"]["status"], "incomplete")
        self.assertEqual(done["response"]["usage"]["output_tokens"], 8)

    def test_anthropic(self):
        status, raw = self.post("/v1/messages", {"max_tokens": 8, "ignore_eos": True})
        self.assertEqual(status, 200, raw)
        got = json.loads(raw)
        self.assertEqual(got["stop_reason"], "max_tokens")
        self.assertEqual(got["usage"]["output_tokens"], 8)
        u = got["usage"]                # #1013: the prompt as Anthropic's three fields
        self.assertEqual(u["input_tokens"] + u["cache_creation_input_tokens"] + u["cache_read_input_tokens"],
                         len(self.engine.legs[0][0]))
        self.assertEqual(got["content"][-1]["text"], "okokok")


if __name__ == "__main__":
    unittest.main()
