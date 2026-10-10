"""serve/test_literal_think_guard.py - #1814: the reply that ends inside its thinking with no answer at all.

Reproduced on an RTX 3090 with Qwen3.8-Flash-Next IQ3_XXS on 0.1.41: the model is about to write its closing thinking
tag and the turn-end token comes instead, so the turn ends with finish_reason "stop", the reasoning stops
mid-sentence, and the client gets an empty answer.  6 of 9 trigger runs, streaming and whole, with and without
STRATA_PREFILL_CPU_SHARE.

serve/fixtures/literal_think_specimens.json holds the raw text of five real turns (dumped with STRATA_DEBUG=1), each
tagged with its `kind`: `no-answer` is this bug, `answered` reached its answer, `length-cut` ended on max_tokens.  This
test replays them through a mock engine.

The parser cannot tell this end from a normal one, and nothing in the token stream says "I meant it".  Two layers, so
that the fix is not optional:

  * the end is always REPORTED - one log line, `totals.ended_inside_thinking` (and /metrics), and
    `"ended_inside_thinking": true` on the response.  This changes no output, so it has no switch.
  * the reply is continued with the thinking closed the way #123's wrap-up closes it, which is opt-in
    ("literal_think_guard"), like #1053: it appends the close to the prompt and spends another pass.

    python -m unittest serve.test_literal_think_guard -v
"""
from __future__ import annotations

import json
import sys
import threading
import time
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, OutputParser, THINK_END  # noqa: E402
from serve.server import ByteTokenizer, LITERAL_THINK_RETRIES, REASONING_CLOSE, Service, serve  # noqa: E402
from serve.test_server import CTX, ThinkingEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPECIMENS = json.loads((ROOT / "serve/fixtures/literal_think_specimens.json").read_text(encoding="utf-8"))["specimens"]
NO_ANSWER = [s for s in SPECIMENS if s["kind"] == "no-answer"]
ANSWERED = [s for s in SPECIMENS if s["kind"] == "answered"]
LENGTH_CUT = [s for s in SPECIMENS if s["kind"] == "length-cut"]


class SpecimenEngine(ThinkingEngine):
    """Replays one dumped turn.  A prompt that already ends with the close the guard appends gets the answer, like the
    model continuing after the close."""

    def __init__(self, tok, raw: str):
        super().__init__(tok)
        self.raw = raw
        self.max_context = max(CTX, len(self.tok.encode(raw)) + 4096)    # the byte tokenizer: one token per byte

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        done = self.tok.decode(ids).endswith(REASONING_CLOSE)
        text = self.ANSWER if done else self.raw
        for t in (self.tok.encode(text) + self.tok.encode("<|im_end|>", parse_special=True))[:max_new]:
            if cancel.is_set():
                return
            yield t


class ContentFlipEngine(SpecimenEngine):
    """The other side of the same end: the model writes the tag as ordinary text and stops, so the parser takes it for
    the marker.  The guard has to put the tag back into the reasoning before the close is appended - otherwise the tag
    lands in the answer."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        done = self.tok.decode(ids).endswith(REASONING_CLOSE)
        text = self.ANSWER if done else (self.raw + THINK_END)
        for t in (self.tok.encode(text) + self.tok.encode("<|im_end|>", parse_special=True))[:max_new]:
            if cancel.is_set():
                return
            yield t


class AlwaysStopsEngine(SpecimenEngine):
    """Every pass ends inside the thinking with no answer, and nothing is ever handed to the client (only the newlines
    the answer parser holds back): the cap's test, with `answered` false on every pass."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        for t in (self.tok.encode("\n\n") + self.tok.encode("<|im_end|>", parse_special=True))[:max_new]:
            if cancel.is_set():
                return
            yield t


def _loop_threads() -> set:
    """The hardware sampler's threads -- ``serve()`` names them ``<...>(_loop)``, and serve/test_responses'
    NoLeakedSampler counts exactly these, so this file must not leave one behind."""
    return {t for t in threading.enumerate() if t.name.endswith("(_loop)")}


class Guard(unittest.TestCase):
    """#1814: on by default, capped, and it leaves a reply that produced its answer alone."""

    def setUp(self):
        self.tok = ByteTokenizer()
        self.boot(SpecimenEngine, NO_ANSWER[0]["raw"])

    def tearDown(self):
        self.close()

    def boot(self, engine_cls, raw="", guard_on=True):
        self.engine = engine_cls(self.tok, raw)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.literal_think_guard = guard_on
        self._loops_before = _loop_threads()      # the sampler this server starts must not outlive it
        self._closed = False
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        return self

    def close(self):
        if getattr(self, "_closed", True):
            return                                # tearDown after the test already closed it
        self._closed = True
        httpd = getattr(self, "httpd", None)
        if httpd is None:
            return
        httpd.shutdown()
        httpd.server_close()
        # The hardware sampler stops with the server, but a moment later.  Wait for it: a lingering `(_loop)`
        # thread throws off serve/test_responses' NoLeakedSampler, which counts exactly these threads.
        deadline = time.time() + 5.0
        while time.time() < deadline and (_loop_threads() - self._loops_before):
            time.sleep(0.02)

    def chat(self, max_tokens: int = 4000):
        body = {"model": "m", "messages": [{"role": "user", "content": "2+2?"}], "max_tokens": max_tokens}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode())["choices"][0]

    def chat_full(self, max_tokens: int = 4000) -> dict:
        """The whole response: the reporting layer's flag sits beside `choices`, not inside a choice."""
        body = {"model": "m", "messages": [{"role": "user", "content": "2+2?"}], "max_tokens": max_tokens}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode())

    def test_off_by_default(self):
        self.close()
        self.engine = SpecimenEngine(self.tok, NO_ANSWER[0]["raw"])
        svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.assertFalse(svc.literal_think_guard)       # opt-in, like #1053 (the maintainer's choice for it)

    def test_the_corpus_holds_the_real_turns(self):
        self.assertGreaterEqual(len(NO_ANSWER), 3)
        for sp in NO_ANSWER:
            self.assertEqual(sp["stopped_on"], "<|im_end|>", sp["raw"][-40:])
            self.assertNotIn(THINK_END, sp["raw"], sp["raw"][-40:])      # the answer never came out

    def test_every_no_answer_specimen_is_answered(self):
        for sp in NO_ANSWER:
            self.close()
            self.boot(SpecimenEngine, sp["raw"])
            with self.subTest(tail=sp["raw"][-40:]):
                c = self.chat()
                self.assertEqual(c["message"]["content"], SpecimenEngine.ANSWER)
                self.assertEqual(len(self.engine.prompts), 2)            # the reply, then the continuation

    def test_the_reasoning_of_the_specimen_is_kept(self):
        c = self.chat()
        self.assertIn(self.engine.raw[:80], c["message"]["reasoning_content"])

    def test_the_continuation_gets_the_thinking_closed(self):
        self.chat()
        self.assertTrue(self.tok.decode(self.engine.prompts[1]).endswith(REASONING_CLOSE))

    def test_a_reply_that_answered_is_not_touched(self):
        for sp in ANSWERED:
            self.close()
            self.boot(SpecimenEngine, sp["raw"])
            with self.subTest(tail=sp["raw"][-40:]):
                c = self.chat()
                self.assertTrue(c["message"]["content"].strip())
                self.assertEqual(len(self.engine.prompts), 1)

    def test_a_length_cut_is_not_touched(self):
        for sp in LENGTH_CUT:
            self.close()
            self.boot(SpecimenEngine, sp["raw"])
            self.chat()
            self.assertEqual(len(self.engine.prompts), 1)                # the guard only acts on finish_reason "stop"

    def test_off_the_reply_stays_empty(self):
        self.close()
        self.boot(SpecimenEngine, NO_ANSWER[0]["raw"], guard_on=False)
        c = self.chat()
        self.assertEqual(len(self.engine.prompts), 1)
        self.assertFalse(c["message"].get("content"))

    def test_the_empty_reply_says_why_it_is_empty(self):
        """The reporting layer needs no switch: with the guard off there is still a log line, a total, and the flag."""
        self.close()
        self.boot(SpecimenEngine, NO_ANSWER[0]["raw"], guard_on=False)
        full = self.chat_full()
        self.assertTrue(full["ended_inside_thinking"])
        self.assertFalse(full["choices"][0]["message"].get("content"))
        self.assertEqual(self.svc.totals["ended_inside_thinking"], 1)

    def test_the_reply_the_guard_answered_is_reported_too(self):
        full = self.chat_full()
        self.assertTrue(full["ended_inside_thinking"])
        self.assertEqual(full["choices"][0]["message"]["content"], SpecimenEngine.ANSWER)
        self.assertEqual(self.svc.totals["ended_inside_thinking"], 1)

    def test_a_normal_reply_carries_no_flag(self):
        for sp in ANSWERED:
            self.close()
            self.boot(SpecimenEngine, sp["raw"])
            with self.subTest(tail=sp["raw"][-40:]):
                full = self.chat_full()
                self.assertNotIn("ended_inside_thinking", full)
                self.assertNotIn("ended_inside_thinking", self.svc.totals)

    def test_the_tag_written_as_text_does_not_leak_into_the_answer(self):
        self.close()
        self.boot(ContentFlipEngine, NO_ANSWER[0]["raw"])
        c = self.chat()
        self.assertEqual(c["message"]["content"], ContentFlipEngine.ANSWER)   # the answer, nothing else
        self.assertNotIn(THINK_END, c["message"]["content"])
        self.assertIn(THINK_END, c["message"]["reasoning_content"])           # the tag stayed reasoning

    def test_the_retries_are_capped(self):
        self.close()
        self.boot(AlwaysStopsEngine, NO_ANSWER[0]["raw"])
        c = self.chat()
        self.assertEqual(len(self.engine.prompts), 1 + LITERAL_THINK_RETRIES)   # the cap, not a loop
        self.assertIn(c["finish_reason"], ("stop", "length"))


class TheOtherSideOfTheSameEnd(unittest.TestCase):
    """When the tag really is written as text the parser switches to the answer and stops there: the same empty reply,
    seen as `parser.state == "content"` instead of `"reasoning"`."""

    def test_the_tag_as_text_switches_the_parser_to_the_answer(self):
        p = OutputParser(thinking=True, tools=None, stream_tools=True)
        evs = p.feed("thinking about the tag: " + THINK_END)
        self.assertEqual(p.state, "content")
        self.assertEqual([e.kind for e in evs], ["reasoning"])

    def test_reopening_puts_it_back(self):
        p = OutputParser(thinking=True, tools=None, stream_tools=True)
        p.feed("thinking about the tag: " + THINK_END)
        p.reopen_reasoning()
        self.assertEqual(p.state, "reasoning")
        self.assertEqual([e.kind for e in p.feed(" more thinking")], ["reasoning"])


if __name__ == "__main__":
    unittest.main()
