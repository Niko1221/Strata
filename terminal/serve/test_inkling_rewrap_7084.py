"""#7084: after the thinking budget's wrap-up Inkling-2-Mini sometimes opens a SECOND thinking turn; the server had cleared the budget at the wrap-up, so nothing
capped that turn and it ran to max_tokens (32,000 hidden tokens, finish "length", an empty reply).

The budget stays armed on Inkling after a wrap-up, for a re-opened thinking turn: at most REWRAP_BUDGET tokens of it (or the budget itself when smaller); the
second wrap-up closes it AND opens the model's text turn (`<|message_model|><|content_text|>`, as the empty-reply retry does), so the model has to answer.  Other
families keep what they had: the budget is spent once.

    python -m pytest serve/test_inkling_rewrap_7084.py
"""
from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve import inkling_dialect  # noqa: E402
from serve.test_inkling_reasoning_budget import BudgetCase, ThinkingEngine  # noqa: E402

OPENER = "<|message_model|><|content_text|>"


class ReopeningEngine(ThinkingEngine):
    """Thinks (THOUGHT) and is wrapped up; then it opens a second thinking turn of REOPEN bytes, no end of its own, until it is wrapped up again.  After a wrap-up that
    also opened the text turn it answers.  `reopens` is how many times it re-opens (default: every time it is allowed to think after a bare wrap-up)."""

    REOPEN = "Hmm, wait, let me reconsider everything once more from the start. " * 400     # about 26,000 bytes: never ends by itself in a test

    def __init__(self, tok, thought, reopens=99, **kw):
        super().__init__(tok, thought, **kw)
        self.opener = tok.encode(OPENER, parse_special=True)
        self.reopens_left = reopens

    def _script(self, ids):
        tail = list(ids[-(len(self.wrap) + len(self.opener)):])
        if tail == self.wrap + self.opener:                     # wrapped up with the answer's opener: the text turn is open
            return self.tok.encode(self.answer) + self.tok.encode("<|end_message|>", parse_special=True) + self.end
        if list(ids[-len(self.wrap):]) == self.wrap and self.reopens_left > 0:
            self.reopens_left -= 1
            return self.tok.encode("<|message_model|><|content_thinking|>", parse_special=True) + self.tok.encode(self.REOPEN)
        return super()._script(ids)


class Rewrap(BudgetCase):
    def start_reopening(self, reopens=99):
        from serve.frontend import ChatTemplate
        from serve.server import Service
        from serve.test_inkling_reasoning_budget import InkTokenizer, TEMPLATE
        self.tok = InkTokenizer()
        self.engine = ReopeningEngine(self.tok, self.THOUGHT, reopens=reopens)
        self.svc = Service(self.engine, self.tok, ChatTemplate(TEMPLATE))
        self.assertTrue(self.svc.inkling)

    def test_a_reopened_thinking_turn_is_capped_and_the_model_answers(self):
        self.start_reopening()
        content, reasoning, done = self.run_request(max_new=4000, reasoning_budget_tokens=60)
        self.assertEqual(content, "The answer is 4.")
        self.assertEqual(done["finish"], "stop")
        passes = self.engine.passes
        self.assertEqual(len(passes), 3)                         # thinking, the re-opened thinking, the answer
        self.assertLess(passes[1]["sent"], 100)                  # the second thinking turn was cut near its budget, not at max_tokens
        self.assertLess(done["completion_tokens"], 400)
        self.assertGreaterEqual(done["reasoning_tokens"], 100)   # both turns count as thinking

    def test_the_second_wrap_up_opens_the_answer_the_first_does_not(self):
        # the third pass's prompt: ... wrap-up, the re-opened turn, the wrap-up AND the text turn's opener
        seen = []
        self.start_reopening()
        orig = self.engine.generate

        def spy(ids, *a, **k):
            seen.append(list(ids))
            return orig(ids, *a, **k)
        self.engine.generate = spy
        self.run_request(max_new=4000, reasoning_budget_tokens=60)
        self.assertEqual(len(seen), 3)
        wrap, opener = self.engine.wrap, self.engine.opener
        self.assertEqual(seen[1][-len(wrap):], wrap)
        self.assertNotEqual(seen[1][-len(opener):], opener)      # the first wrap-up leaves the choice of the next message to the model
        self.assertEqual(seen[2][-(len(wrap) + len(opener)):], wrap + opener)

    def test_a_model_that_answers_after_the_first_wrap_up_is_unchanged(self):
        self.start()
        content, _, done = self.run_request(reasoning_budget_tokens=60)
        self.assertEqual(content, "The answer is 4.")
        self.assertEqual(len(self.engine.passes), 2)
        self.assertEqual(done["finish"], "stop")

    def test_the_second_budget_is_the_smaller_of_the_budget_and_the_cap(self):
        from serve import server
        self.assertEqual(server.INKLING_REWRAP_BUDGET, 512)
        self.start_reopening()
        _, _, done = self.run_request(max_new=8000, reasoning_budget_tokens=2048)
        # the first turn (the 1,800-byte thought) ends by itself inside 2,048 tokens only if the engine says so; here it is cut at 2,048 or ends: either way
        # the re-opened turn is cut at the cap, so the whole reply stays far below max_tokens
        self.assertEqual(done["finish"], "stop")
        self.assertLess(done["completion_tokens"], 2048 + 512 + 400)

    def test_without_the_fix_the_second_turn_runs_to_max_tokens(self):
        # the old code: the budget is cleared at the wrap-up, so the re-opened turn is not counted
        self.start_reopening()
        from serve import server
        saved = server.INKLING_REWRAP_BUDGET
        try:
            server.INKLING_REWRAP_BUDGET = 0                      # 0: the old behaviour (the budget is spent once)
            _, _, done = self.run_request(max_new=3000, reasoning_budget_tokens=60)
        finally:
            server.INKLING_REWRAP_BUDGET = saved
        self.assertEqual(done["finish"], "length")                # the old result: all of max_tokens, no answer
        self.assertGreaterEqual(done["completion_tokens"], 2900)

    def test_a_model_that_reopens_twice_is_wrapped_up_each_time(self):
        self.start_reopening(reopens=2)
        # the second re-opening happens after a wrap-up that already carried the opener, which the engine script treats as the text turn (it answers);
        # so one re-opening is all this script can show; the bound is the point: never more passes than wrap-ups + 1
        content, _, done = self.run_request(max_new=4000, reasoning_budget_tokens=60)
        self.assertEqual(content, "The answer is 4.")
        self.assertLessEqual(len(self.engine.passes), 3)


class OtherFamiliesKeepTheirBudgetSpentOnce(unittest.TestCase):
    def test_the_rule_is_inkling_only(self):
        text = (ROOT / "serve" / "server.py").read_text(encoding="utf-8")
        i = text.index("                        if wrap:\n                            wraps += 1")
        self.assertIn("if self.inkling", text[i:i + 900])
        self.assertIn("budget = st[\"thinking_budget\"] = None", text[i:i + 1400])      # the other families: cleared as before


if __name__ == "__main__":
    unittest.main()
