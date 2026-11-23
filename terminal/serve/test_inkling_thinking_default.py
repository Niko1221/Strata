"""A default thinking budget for Inkling-2-Mini, and never an empty reply (#6612, #6620, #6734, #6794).

The model writes hidden thinking whatever the request says.  Without a budget a runaway thought used the whole max_tokens and the client got
nothing.  Now: a default budget (STRATA_INKLING_THINKING_BUDGET, 2048 tokens; 0 = none) when neither the request nor the config names one; the
answer keeps its share of max_tokens even when max_tokens is small; a reply that ends with no text and no call is retried once with the answer opened;
and when nothing visible can be produced the reply says so instead of being empty.  Runs through Service.run with the real chat template.

    python -m pytest serve/test_inkling_thinking_default.py
"""
from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve import inkling_dialect  # noqa: E402
from serve.server import NO_ANSWER_LENGTH, NO_ANSWER_STOP  # noqa: E402
from serve.test_inkling_reasoning_budget import BudgetCase, InkTokenizer, ThinkingEngine  # noqa: E402


class ModalEngine(ThinkingEngine):
    """normal: thought, then the answer (answers at once after the wrap-up or after an opened answer);
    stops_empty: thought, then the block's end and the stop token, no answer, unless the prompt ends with the opened answer;
    never_answers: thinks again whatever it is given (the wrap-up and an opened answer included)."""
    mode = "normal"

    def _script(self, ids):
        t = self.tok
        end_msg = t.encode("<|end_message|>", parse_special=True)
        opener = t.encode("<|message_model|><|content_text|>", parse_special=True)
        think = t.encode("<|message_model|><|content_thinking|>", parse_special=True) + t.encode(self.thought)
        ans_tail = t.encode(self.answer) + end_msg + self.end
        if self.mode == "never_answers":
            return think + end_msg + think
        if ids[-len(opener):] == opener:                 # the server opened the answer
            return ans_tail
        if ids[-len(self.wrap):] == self.wrap or (self.passes and ids[-1:] == end_msg):   # the wrap-up (long, or the bare block end after a first pass)
            return opener + ans_tail
        if self.mode == "stops_empty":
            return think + end_msg + self.end
        return think + end_msg + opener + ans_tail


class DefaultBudget(BudgetCase):
    LONG = "Let me think step by step about two plus two. " * 120          # 5,400 bytes of hidden thinking

    def start(self, thought=None, mode="normal", budget=None):
        self.tok = InkTokenizer()
        self.engine = ModalEngine(self.tok, self.LONG if thought is None else thought)
        self.engine.mode = mode
        from serve.frontend import ChatTemplate
        from serve.server import Service
        from serve.test_inkling_reasoning_budget import TEMPLATE
        self.svc = Service(self.engine, self.tok, ChatTemplate(TEMPLATE))
        if budget is not None:
            self.svc.inkling_thinking_budget = budget

    def test_the_default_budget_is_by_level(self):
        self.start()
        self.assertIsNone(self.svc.inkling_thinking_budget)                          # no single number set: each level has its own
        for level, want in (("none", 1024), ("minimal", 1024), ("low", 2048), ("medium", 4096), ("high", 8192), ("max", 16384)):
            self.assertEqual(self.svc.reasoning_budget({"strata_effort": level}), want, level)
        self.assertEqual(self.svc.reasoning_budget({"strata_effort": 0.5}), 4096)         # a number between the words: the next word up (medium)
        self.assertEqual(self.svc.reasoning_budget({}), 2048)                        # no level known: the default level (low)
        self.svc.inkling_thinking_budget = 777                                       # STRATA_INKLING_THINKING_BUDGET: one number for all
        self.assertEqual(self.svc.reasoning_budget({"strata_effort": "high"}), 777)
        self.assertEqual(self.svc.reasoning_budget({"reasoning_budget_tokens": 300}), 300)     # the request wins
        self.assertIsNone(self.svc.reasoning_budget({"reasoning_budget_tokens": 0}))           # 0 in the request: no budget

    def test_a_runaway_thought_is_cut_at_the_default(self):
        self.start()
        content, _, done = self.run_request(max_new=8000)
        self.assertEqual(content, "The answer is 4.")
        self.assertEqual(len(self.engine.passes), 2)
        self.assertLess(self.engine.passes[0]["sent"], 2048 + 80)                  # cut at the budget, not at the end of 5,400 bytes + the answer

    def test_the_setting_zero_means_no_default(self):
        self.start(budget=0)
        content, _, done = self.run_request(max_new=8000)
        self.assertEqual(content, "The answer is 4.")
        self.assertEqual(len(self.engine.passes), 1)                                # the answer room (6,000) is far above the 5,400-byte thought

    def test_a_request_budget_of_zero_turns_the_default_off(self):
        self.start()
        content, _, done = self.run_request(max_new=8000, reasoning_budget_tokens=0)
        self.assertEqual(content, "The answer is 4.")
        self.assertEqual(len(self.engine.passes), 1)

    def test_a_typical_thought_is_left_alone(self):
        self.start(thought="short thought " * 20)                                   # 280 tokens, well under the default
        content, _, done = self.run_request(max_new=8000)
        self.assertEqual(content, "The answer is 4.")
        self.assertEqual(len(self.engine.passes), 1)

    def test_a_small_max_tokens_keeps_half_for_the_answer(self):
        for mt in (100, 300, 600):
            with self.subTest(max_tokens=mt):
                self.start()
                content, _, done = self.run_request(max_new=mt)
                self.assertEqual(content, "The answer is 4.")
                self.assertEqual(done["finish"], "stop")

    def test_a_reply_that_stops_with_nothing_is_retried_with_the_answer_opened(self):
        self.start(thought="short thought " * 20, mode="stops_empty")
        content, _, done = self.run_request(max_new=4000)
        self.assertEqual(content, "The answer is 4.")
        self.assertEqual(len(self.engine.passes), 2)
        self.assertEqual(done["finish"], "stop")

    def test_when_nothing_can_be_produced_the_reply_says_so(self):
        self.start(thought="x " * 50, mode="never_answers")
        content, _, done = self.run_request(max_new=400)
        self.assertEqual(content, NO_ANSWER_LENGTH)
        self.assertEqual(done["finish"], "length")                                   # finish_reason stays what happened

    def test_an_empty_stop_after_the_retry_says_so_too(self):
        class Silent(ModalEngine):
            def _script(self, ids):                                                  # stops at once, whatever it is given
                return self.end
        self.start()
        self.engine = Silent(self.tok, "")
        from serve.frontend import ChatTemplate
        from serve.server import Service
        from serve.test_inkling_reasoning_budget import TEMPLATE
        self.svc = Service(self.engine, self.tok, ChatTemplate(TEMPLATE))
        content, _, done = self.run_request(max_new=400)
        self.assertEqual(content, NO_ANSWER_STOP)
        self.assertEqual(done["finish"], "stop")

    def test_a_reply_with_a_tool_call_is_never_an_empty_reply(self):
        class Calls(ModalEngine):
            def _script(self, ids):
                t = self.tok
                call = ('<|message_model|>read_file<|content_invoke_tool_json|>{"name":"read_file","args":{"path":"a.py"}}<|end_message|>'
                        "<|content_model_end_sampling|>")
                return t.encode(call, parse_special=True)
        self.start()
        self.engine = Calls(self.tok, "")
        from serve.frontend import ChatTemplate
        from serve.server import Service
        from serve.test_inkling_reasoning_budget import TEMPLATE
        self.svc = Service(self.engine, self.tok, ChatTemplate(TEMPLATE))
        tools = [{"name": "read_file", "description": "d", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}]
        ids = self.tok.encode("<|message_user|><|content_text|>open a.py<|end_message|><|message_model|><|content_thinking|><|end_message|>", parse_special=True)
        kinds = [p.kind for k, p in self.svc.run(ids, True, tools, 400, {}, threading.Event()) if k == "event"]
        self.assertIn("tool_call", kinds)
        self.assertNotIn("content", kinds)                                           # no note next to a real call
        self.assertEqual(len(self.engine.passes), 1)


class ToolErrorBlock(unittest.TestCase):
    """The model can open a `<|content_tool_error|>` text block (token 200022).  From robertkarl/UnbiasedStrata (MIT) by @robertkarl: its parser reads
    it as an answer text kind.  Here it was not a known tag, so its text showed up in the reply with the tag spelled out."""

    def test_the_block_reads_as_text_and_the_tag_is_not_shown(self):
        d = inkling_dialect.InklingDialect()
        out = d.feed_channels("<|message_model|><|content_tool_error|>the file was not found<|end_message|><|content_model_end_sampling|>", final=True)
        text = "".join(t for k, t in out if k == "content")
        self.assertEqual(text, "the file was not found")
        self.assertNotIn("content_tool_error", text)

    def test_as_a_token_it_drives_the_state_machine_too(self):
        d = inkling_dialect.InklingDialect()
        out = []
        for piece, special in (("<|message_model|>", True), ("<|content_tool_error|>", True), ("no such file", False), ("<|end_message|>", True)):
            out += d.feed_token(piece, is_special=special)
        out += d.feed_channels("", final=True)
        self.assertEqual("".join(t for k, t in out if k == "content"), "no such file")


if __name__ == "__main__":
    unittest.main()
