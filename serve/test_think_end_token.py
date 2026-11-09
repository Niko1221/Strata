"""#4077/#5605: DeepSeek-V4.1's reasoning ends at its `</think>` control token, not at the same text spelled out with
ordinary tokens.  The parser in token mode (think_end=THINK_END_TOKEN) and the server that feeds it (Service.think_end_id),
over a fake engine that replays exact token ids (no GPU, no pack, no model).

    python -m unittest serve.test_think_end_token -v
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve.dsml import DsmlOutputParser  # noqa: E402
from serve.frontend import THINK_END, THINK_END_TOKEN, OutputParser  # noqa: E402
from serve.server import REASONING_WRAP_UP  # noqa: E402
from serve.test_dsv41 import EOS, SPECIALS, ApiBase, DsEngine  # noqa: E402

M = THINK_END_TOKEN                     # what the server feeds the parser for the control token


def split(parser_cls, text, step, **kw):
    p = parser_cls(thinking=kw.pop("thinking", True), think_end=M, **kw)
    ev = []
    for k in range(0, len(text), step):
        ev += p.feed(text[k:k + step])
    ev += p.finish()
    return {kind: "".join(e.text or "" for e in ev if e.kind == kind) for kind in ("reasoning", "content")}


class TokenModeParser(unittest.TestCase):
    """The parser alone: the mark is the end of the thinking, a spelled-out tag is text."""

    def check(self, text, reasoning, content, **kw):
        for cls in (OutputParser, DsmlOutputParser):
            for step in (1, 3, 7, len(text) or 1):
                with self.subTest(cls=cls.__name__, step=step):
                    got = split(cls, text, step, **dict(kw))
                    self.assertEqual(got["reasoning"], reasoning)
                    self.assertEqual(got["content"].strip(), content)
                    self.assertNotIn(M, got["reasoning"] + got["content"])

    def test_a_spelled_tag_inside_the_thinking_is_reasoning_4077(self):
        thought = "The format ends the thinking with\n</think>\nso I must not write it too early. Now answer."
        self.check(thought + M + "\n\nThe answer.", thought, "The answer.")

    def test_a_spelled_tag_mid_sentence_is_reasoning(self):
        thought = "DS may also produce quotes before </think>. Our 0.1.59 includes the fix. Answer briefly."
        self.check(thought + M + "Patched and live.", thought, "Patched and live.")

    def test_a_closing_quote_right_before_the_token_ends_the_thinking_5605(self):
        thought = 'They said "Reply with the number only."'
        self.check(thought + M + "2", thought, "2")

    def test_a_quoted_spelled_tag_is_reasoning(self):
        thought = 'The tag is written `</think>` or "</think>" in the docs.'
        self.check(thought + M + "Done.", thought, "Done.")

    def test_a_normal_turn(self):
        self.check("Simple.\n" + M + "\n\nThe answer.", "Simple.\n", "The answer.")

    def test_a_spelled_tag_in_the_answer_is_kept(self):
        self.check("Explain the tag." + M + "DeepSeek ends its thinking with </think>.", "Explain the tag.",
                   "DeepSeek ends its thinking with </think>.")

    def test_a_second_token_is_not_text(self):
        self.check("Think." + M + "Answer" + M + " goes on.", "Think.", "Answer goes on.")

    def test_thinking_off_drops_a_stray_token(self):
        self.check("Plain" + M + " answer.", "", "Plain answer.", thinking=False)

    def test_the_text_mode_is_unchanged(self):
        """Without the mark the parser reads the tag's text as before (models whose `</think>` is not one token)."""
        for step in (1, 3, 7):
            p, ev = OutputParser(thinking=True), []
            text = "Reasoning.</think>The answer."
            for k in range(0, len(text), step):
                ev += p.feed(text[k:k + step])
            ev += p.finish()
            self.assertFalse(p.by_token)
            self.assertEqual("".join(e.text for e in ev if e.kind == "content"), "The answer.")

    def test_the_dsml_fallback_never_sends_the_thinking_as_the_answer(self):
        """No token at all, the end spelled out instead (a degraded quant): the answer is what follows the last
        spelled tag, not the whole thinking (the 2-bit fallback re-sent all of it)."""
        for step in (1, 3, 7):
            p, ev = DsmlOutputParser(thinking=True, think_end=M), []
            text = "Short thought.</think>The answer."
            for k in range(0, len(text), step):
                ev += p.feed(text[k:k + step])
            ev += p.finish("stop")
            self.assertEqual("".join(e.text for e in ev if e.kind == "content"), "The answer.")

    def test_the_dsml_fallback_without_any_tag_is_unchanged(self):
        p = DsmlOutputParser(thinking=True, think_end=M)
        ev = p.feed("17 x 23 = 391.") + p.finish("stop")
        self.assertEqual("".join(e.text for e in ev if e.kind == "content"), "17 x 23 = 391.")


def spelled(text: str) -> list[int]:
    """`text` as the mini tokenizer's byte tokens: a `</think>` written with ordinary tokens, never its special id."""
    return [len(SPECIALS) + b for b in text.encode("utf-8")]


class IdEngine(DsEngine):
    """DsEngine replaying exact id lists, so a script can hold a spelled-out `</think>` next to the real token."""

    def __init__(self, tok, scripts):
        super().__init__(tok, "x")
        self.scripts = [list(s) + [EOS] for s in scripts]
        self.script = self.scripts[0]


class TokenModeApi(ApiBase):
    def start_ids(self, *scripts):
        self.start("x")                                   # the server, the family and the cleanup
        self.svc.engine = engine = IdEngine(self.tok, scripts)
        self.engine = engine
        return engine

    def end(self):
        return self.tok.encode(THINK_END, parse_special=True)

    def test_the_family_reads_the_token(self):
        self.start("x")
        self.assertEqual(self.svc.think_end_id, self.end()[0])
        self.assertEqual(len(self.end()), 1)

    def stream(self, path, body):
        status, raw = self.post(path, {**body, "stream": True}, raw=True)
        self.assertEqual(status, 200)
        self.assertNotIn("\\udb83", raw.lower())     # the mark (U+F0E0F) never reaches a client, escaped or not
        self.assertNotIn(M, raw)
        return raw

    def test_4077_over_both_apis(self):
        thought = "The format ends with\n</think>\nbut not yet. Now answer."
        ids = spelled(thought) + self.end() + spelled("The answer.")
        self.start_ids(ids)
        status, r = self.chat()
        self.assertEqual(status, 200, r)
        msg = r["choices"][0]["message"]
        self.assertEqual(msg["reasoning_content"], thought)
        self.assertEqual(msg["content"].strip(), "The answer.")
        raw = self.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 400})
        chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
        deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
        self.assertEqual("".join(d.get("content") or "" for d in deltas).strip(), "The answer.")
        self.assertEqual("".join(d.get("reasoning_content") or "" for d in deltas), thought)
        status, r = self.post("/v1/messages", {"model": "x", "max_tokens": 400, "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200, r)
        blocks = {b["type"]: b for b in r["content"]}
        self.assertEqual(blocks["thinking"]["thinking"], thought)
        self.assertEqual(blocks["text"]["text"].strip(), "The answer.")
        self.stream("/v1/messages", {"model": "x", "max_tokens": 400, "messages": [{"role": "user", "content": "hi"}]})

    def test_5605_over_the_api(self):
        thought = 'They said "Reply with the number only."'
        self.start_ids(spelled(thought) + self.end() + spelled("2"))
        status, r = self.chat()
        msg = r["choices"][0]["message"]
        self.assertEqual((msg["reasoning_content"], msg["content"].strip()), (thought, "2"))

    def test_a_spelled_end_without_the_token_is_not_sent_as_the_answer(self):
        """A degraded quant that spells the end instead of writing the token: the DSML fallback answers with the text
        after the last spelled tag, never the thinking itself."""
        self.start_ids(spelled("Brief thought.</think>The answer."))
        status, r = self.chat()
        self.assertEqual(r["choices"][0]["message"]["content"].strip(), "The answer.")

    def test_the_close_retry_still_closes_the_thinking_1053(self):
        self.start_ids(spelled("Thinking about </think> and more."), spelled("The answer."))
        self.svc.reasoning_close_retry = True
        status, r = self.chat()
        self.assertEqual(status, 200, r)
        msg = r["choices"][0]["message"]
        self.assertEqual(msg["content"].strip(), "The answer.")
        self.assertIn("</think> and more.", msg["reasoning_content"])
        self.assertEqual(self.engine.turns, 2)                     # closed once, then answered

    def test_the_budget_wrap_up_still_closes_the_thinking_123(self):
        forever = spelled("thinking about </think> ") * 40
        self.start_ids(forever, spelled("The answer."))
        status, r = self.chat(max_tokens=600, reasoning_budget_tokens=20)
        self.assertEqual(status, 200, r)
        msg = r["choices"][0]["message"]
        self.assertEqual(msg["content"].strip(), "The answer.")
        self.assertEqual(self.engine.turns, 2)                     # wrapped up, then answered
        self.assertTrue(self.tok.decode(self.engine.last_prompt).endswith(
            self.tok.decode(self.tok.encode(REASONING_WRAP_UP, parse_special=True))))


if __name__ == "__main__":
    unittest.main()
