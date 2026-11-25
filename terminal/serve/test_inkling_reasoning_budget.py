"""#6734: `reasoning_budget_tokens` is enforced on Inkling-2-Mini's hidden thinking ("thinking: 7953 of budget 4000").

The base parser of Inkling's output starts in content mode and the dialect hides the `<|content_thinking|>` block, so the server's
budget check (`parser.state == "reasoning"`) never counted a token.  The parser now says where the thinking is (`reasoning_now`),
and the budget closes it with Inkling's own wrap-up (`<|end_message|>`), after which the model opens its answer.

    python -m pytest serve/test_inkling_reasoning_budget.py
"""
from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve import inkling_dialect  # noqa: E402
from serve.frontend import ChatTemplate, InklingOutputParser  # noqa: E402
from serve.server import ByteTokenizer, Service  # noqa: E402

TEMPLATE = ROOT / "tests" / "data" / "inkling" / "chat_template.jinja"


class InkTokenizer(ByteTokenizer):
    SPECIALS = ["<|endoftext|>", "<|content_model_end_sampling|>", "<|message_model|>", "<|message_user|>", "<|message_system|>",
                "<|message_tool|>", "<|content_thinking|>", "<|content_text|>", "<|content_xml|>", "<|content_invoke_tool_json|>",
                "<|end_message|>"]


class ThinkingEngine:
    """Writes `<|message_model|><|content_thinking|>THOUGHT<|end_message|><|message_model|><|content_text|>ANSWER` unless it is
    asked after the server's wrap-up, where it answers at once."""
    inkling_engine = True
    batch = 0
    can_hold = False
    can_hold_batch = False
    max_context = 65536

    def __init__(self, tok, thought, answer="The answer is 4."):
        self.tok, self.thought, self.answer = tok, thought, answer
        self.last = {}
        self.passes: list[dict] = []
        self.wrap = tok.encode(inkling_dialect.REASONING_WRAP_UP, parse_special=True)
        self.end = tok.encode("<|content_model_end_sampling|>", parse_special=True)

    def _script(self, ids):
        opener = self.tok.encode("<|message_model|><|content_text|>", parse_special=True)
        ans = opener + self.tok.encode(self.answer) + self.tok.encode("<|end_message|>", parse_special=True) + self.end
        if ids[-len(self.wrap):] == self.wrap:
            return [self.tok.encode("<|message_model|>", parse_special=True)[0]] + ans[1:]
        think = self.tok.encode("<|message_model|><|content_thinking|>", parse_special=True) + self.tok.encode(self.thought)
        return think + self.tok.encode("<|end_message|>", parse_special=True) + ans

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        out = self._script(list(ids))
        row = {"prompt": len(ids), "sent": 0, "finish": "length"}
        self.passes.append(row)
        for t in out:
            if row["sent"] >= max_new:
                break
            if cancel.is_set():
                row["finish"] = "cancel"
                return
            row["sent"] += 1
            if t in self.end:
                row["finish"] = "stop"
                self.last = dict(row)
                yield t
                return
            yield t
        self.last = dict(row)


class BudgetCase(unittest.TestCase):
    THOUGHT = "Let me think step by step about two plus two. " * 40          # 1800 bytes of hidden thinking

    def start(self, thought=None):
        self.tok = InkTokenizer()
        self.engine = ThinkingEngine(self.tok, self.THOUGHT if thought is None else thought)
        self.svc = Service(self.engine, self.tok, ChatTemplate(TEMPLATE))
        self.assertTrue(self.svc.inkling)
        self.assertIs(self.svc.parser_cls, InklingOutputParser)

    def run_request(self, thinking=True, max_new=4000, **sampling):
        ids = self.tok.encode("<|message_user|><|content_text|>2+2?<|end_message|><|message_model|><|content_thinking|><|end_message|>",
                              parse_special=True)
        content, reasoning, done = "", "", None
        for kind, payload in self.svc.run(ids, thinking, [], max_new, sampling, threading.Event()):
            if kind == "event" and payload.kind == "content":
                content += payload.text or ""
            elif kind == "event" and payload.kind == "reasoning":
                reasoning += payload.text or ""
            elif kind == "done":
                done = payload
        return content, reasoning, done


class InklingBudget(BudgetCase):
    def test_the_parser_says_where_the_thinking_is(self):
        p = InklingOutputParser(thinking=False, tools=None, stream_tools=True)
        self.assertFalse(p.reasoning_now)
        p.feed("<|message_model|><|content_thinking|>abc")
        self.assertTrue(p.reasoning_now)
        self.assertEqual(p.state, "content")          # (the base parser never leaves content mode: why the budget did not count)
        self.assertTrue(p.clean)
        p.feed("<|end_message|>")
        self.assertFalse(p.clean)                      # the block's end waits to be confirmed by the next token
        p.feed("<|message_model|><|content_text|>hi")
        self.assertFalse(p.reasoning_now)

    def test_a_budget_cuts_the_hidden_thinking_and_the_model_answers(self):
        self.start()
        content, reasoning, done = self.run_request(reasoning_budget_tokens=60)
        self.assertEqual(content, "The answer is 4.")
        self.assertTrue(reasoning.startswith(self.THOUGHT[:50]))     # the thinking is the reasoning channel; it ends at the budget (and the wrap-up)
        self.assertEqual(done["finish"], "stop")
        passes = self.engine.passes
        self.assertEqual(len(passes), 2)
        # the first pass was cut at the budget (the parser's 60 thinking tokens), not run to the end of the thought
        self.assertLess(passes[0]["sent"], 90)
        # (the wrap-up's own tokens are written into the thinking too)
        self.assertLessEqual(done["reasoning_tokens"], 60 + len(self.engine.wrap) + 8)
        self.assertGreaterEqual(done["reasoning_tokens"], 60)

    def test_without_the_fix_the_budget_never_fires(self):
        # the old check: the base parser's state is "content" for a hidden block, so `thought` stayed 0
        self.start()
        p = InklingOutputParser(thinking=False, tools=None, stream_tools=True)
        p.feed("<|message_model|><|content_thinking|>abc")
        self.assertNotEqual(p.state, "reasoning")

    def test_a_budget_larger_than_the_thought_changes_nothing(self):
        self.start(thought="short thought")
        content, _, done = self.run_request(reasoning_budget_tokens=500)
        self.assertEqual(content, "The answer is 4.")
        self.assertEqual(len(self.engine.passes), 1)

    def test_a_request_with_thinking_off_is_cut_as_well(self):
        # Inkling thinks whatever the request says (effort "none" included), so an explicit budget holds with enable_thinking false
        self.start()
        content, _, done = self.run_request(thinking=False, reasoning_budget_tokens=60)
        self.assertEqual(content, "The answer is 4.")
        self.assertEqual(len(self.engine.passes), 2)

    def test_the_answer_room_cuts_a_runaway_thought_that_would_eat_max_tokens(self):
        # no budget in the request: max_tokens 1400 with a 1800-token thought used to end empty at "length" (#6620 / #6734)
        self.start()
        content, _, done = self.run_request(max_new=1400)
        self.assertEqual(content, "The answer is 4.")
        self.assertEqual(done["finish"], "stop")

    def test_a_plain_answer_is_never_counted(self):
        # a reply with no thinking at all spends no budget, whatever its length
        self.start()
        self.engine._script = lambda ids: (self.tok.encode("<|message_model|><|content_text|>", parse_special=True)
                                           + self.tok.encode("word " * 300) + self.tok.encode("<|end_message|>", parse_special=True)
                                           + self.engine.end)
        content, _, done = self.run_request(reasoning_budget_tokens=20, max_new=3000)
        self.assertEqual(content, "word " * 300)
        self.assertEqual(len(self.engine.passes), 1)
        self.assertEqual(done["reasoning_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
