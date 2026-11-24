"""serve/test_think_eos_ban.py - #5839: a pass that starts inside the thinking cannot end on an end token.

The opt-in "think_eos_ban" gives every engine pass that starts inside the thinking the end tokens banned (`logit_bias`),
so the thinking can only end at `</think>` or at its budget; the answer is written by passes that may end on an end token
as usual.  With a budget the engine's own hold (GEN hold=N:ID) ends a pass at `</think>`; without one the server cuts the
pass where the parser leaves the thinking.  With the switch off no pass is changed.

    python -m unittest serve.test_think_eos_ban -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import ByteTokenizer, MockEngine  # noqa: E402
from serve.server_parts.base import HOLD_KEY  # noqa: E402
from serve.test_quoted_eos_recovery import (  # noqa: E402
    CALL, IM_END, SCHEMA, SessionEngine, service_for, run)

THINK_END = "</think>"


class ThinkTokenizer(ByteTokenizer):
    """ByteTokenizer with </think> as one token, as the real tokenizer has it: the engine's hold needs one token."""
    SPECIALS = ByteTokenizer.SPECIALS + [THINK_END]
    ALWAYS = (THINK_END,)

    THINK_END = 256 + len(ByteTokenizer.SPECIALS)


class HoldSessionEngine(SessionEngine):
    """SessionEngine on an engine that takes GEN hold=N:ID: the pass ends at `hold_at` when the server asks for a hold."""
    info = {"logit_bias": 1}
    can_hold = True

    def __init__(self, tokenizer, text, hold_at):
        super().__init__(tokenizer, text)
        self.hold_at = hold_at

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        for t in super().generate(ids, max_new, sampling, cancel, embeddings):
            yield t
            if t == self.hold_at and (sampling or {}).get(HOLD_KEY):
                self.last["finish"] = "hold"     # the engine's hold: the pass ends at </think>, the caller goes on
                return


class ForcedCallEngine(MockEngine):
    """Thinks, then answers in text - unless its prompt already opens a call (a forced tool_choice): then it finishes it."""
    THOUGHT, ANSWER = "I could look this up. ", "The answer is 4."
    ARGS = "<parameter=q>\n2+2\n</parameter>\n</function>\n</tool_call>"

    def __init__(self, tokenizer):
        super().__init__(tokenizer, "x")
        self.info, self.prompts, self.samplings = {"logit_bias": 1}, [], []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        self.samplings.append(dict(sampling or {}))
        prompt = self.tok.decode(ids)
        if prompt.endswith("<tool_call>\n"):
            text = "<function=search>\n" + self.ARGS
        else:
            text = self.THOUGHT + THINK_END + "\n\n" + self.ANSWER
        for t in (self.tok.encode(text) + self.tok.encode(IM_END, parse_special=True))[:max_new]:
            if cancel.is_set():
                return
            yield t


class ThinkBan(unittest.TestCase):
    BAN = {257: -100.0, 258: -100.0}                 # ByteTokenizer's ids: <|im_end|> is 257, <|endoftext|> 258

    def biases(self, engine):
        """The logit_bias of every pass the engine was given (None: none)."""
        return [s.get("logit_bias") for s in engine.samplings]

    def test_the_thinking_pass_is_banned_and_the_answer_pass_is_not(self):
        tokenizer = ByteTokenizer()
        engine = SessionEngine(tokenizer, "Let me think.\n" + THINK_END + "\n\nThe answer.")
        engine.info = {"logit_bias": 1}
        service = service_for(engine, tokenizer, quoted_eos_recovery=True, think_eos_ban=True)
        r = run(service, tokenizer)
        self.assertEqual(self.biases(engine), [self.BAN, None])
        self.assertEqual((r["reasoning"], r["content"]), ("Let me think.\n", "The answer."))
        self.assertEqual((r["done"]["finish"], r["done"]["think_eos_bans"]), ("stop", 1))
        self.assertEqual(service.history[-1]["think_eos_bans"], 1)
        first, second = engine.prompts
        self.assertEqual(second[:len(first)], first)       # the answer pass continues the thinking pass
        self.assertGreater(len(second), len(first))

    def test_the_switch_is_needed(self):
        tokenizer = ByteTokenizer()
        engine = SessionEngine(tokenizer, "Let me think.\n" + THINK_END + "\n\nThe answer.")
        engine.info = {"logit_bias": 1}
        service = service_for(engine, tokenizer, quoted_eos_recovery=True)
        r = run(service, tokenizer)
        self.assertEqual(self.biases(engine), [None])
        self.assertEqual(r["content"], "The answer.")
        self.assertNotIn("think_eos_bans", r["done"])
        self.assertNotIn("think_eos_bans", service.history[-1])

    def test_an_engine_without_token_biases_gets_none(self):
        tokenizer = ByteTokenizer()
        engine = SessionEngine(tokenizer, "Let me think.\n" + THINK_END + "\n\nThe answer.")
        r = run(service_for(engine, tokenizer, quoted_eos_recovery=True, think_eos_ban=True), tokenizer)
        self.assertEqual(self.biases(engine), [None])
        self.assertNotIn("think_eos_bans", r["done"])

    def test_a_request_without_thinking_is_never_banned(self):
        tokenizer = ByteTokenizer()
        engine = SessionEngine(tokenizer, "The answer.")
        engine.info = {"logit_bias": 1}
        run(service_for(engine, tokenizer, quoted_eos_recovery=True, think_eos_ban=True), tokenizer, thinking=False)
        self.assertEqual(self.biases(engine), [None])

    def test_the_clients_own_biases_stay(self):
        tokenizer = ByteTokenizer()
        engine = SessionEngine(tokenizer, "Let me think.\n" + THINK_END + "\n\nThe answer.")
        engine.info = {"logit_bias": 1}
        own = {"logit_bias": {5: 2.0}}
        run(service_for(engine, tokenizer, think_eos_ban=True), tokenizer, sampling=own)
        self.assertEqual(self.biases(engine), [{5: 2.0, **self.BAN}, {5: 2.0}])
        self.assertEqual(own, {"logit_bias": {5: 2.0}})     # the request's own object is not changed

    def test_the_engines_hold_at_the_think_end_resumes_unbanned(self):
        tokenizer = ThinkTokenizer()
        engine = HoldSessionEngine(tokenizer, "Short thought.\n" + THINK_END + "\n\nThe answer.", ThinkTokenizer.THINK_END)
        service = service_for(engine, tokenizer, quoted_eos_recovery=True, think_eos_ban=True)
        r = run(service, tokenizer)
        self.assertEqual(self.biases(engine), [self.BAN, None])
        self.assertEqual((r["reasoning"], r["content"]), ("Short thought.\n", "The answer."))
        self.assertEqual(r["done"]["think_eos_bans"], 1)

    def test_a_quoted_think_tag_met_on_a_resumed_pass_bans_the_thinking_again(self):
        """A bare </think> the thought mentions waits for the character after it, and the engine's hold ends the pass
        there.  The resumed pass is not banned (the tag may be the end of the thinking); the space after it says it was
        a quoted one: the pass is cut, and the thinking goes on under the ban."""
        tokenizer = ThinkTokenizer()
        engine = HoldSessionEngine(tokenizer, "It ends with " + THINK_END + " and then the answer.\n" + THINK_END + "\n\nOK",
                                   ThinkTokenizer.THINK_END)
        service = service_for(engine, tokenizer, quoted_eos_recovery=True, think_eos_ban=True)
        r = run(service, tokenizer)
        self.assertEqual(self.biases(engine), [self.BAN, None, self.BAN, None])
        self.assertEqual(r["reasoning"], "It ends with " + THINK_END + " and then the answer.\n")
        self.assertEqual(r["content"], "OK")
        self.assertEqual(r["done"]["finish"], "stop")

    def test_a_list_of_markers_in_the_thinking_does_not_end_it(self):
        """The model writes a template out in its reasoning, a marker a line, and the list ends with `</think>` itself:
        the tag is the next item, and the thinking goes on under the ban until the real one after it."""
        tokenizer = ThinkTokenizer()
        listed = "模板里是这样写的：\n<|im_start|>user\n<|im_start|>assistant\n" + THINK_END + "\n然后是正文。\n"
        engine = HoldSessionEngine(tokenizer, listed + THINK_END + "\n\nOK", ThinkTokenizer.THINK_END)
        r = run(service_for(engine, tokenizer, quoted_eos_recovery=True, think_eos_ban=True), tokenizer)
        self.assertEqual((r["reasoning"], r["content"]), (listed, "OK"))
        self.assertEqual(self.biases(engine), [self.BAN, self.BAN, None])

    def test_a_tag_inside_inline_code_is_known_at_once_and_the_thinking_goes_on_banned(self):
        tokenizer = ThinkTokenizer()
        engine = HoldSessionEngine(tokenizer, "The tag `" + THINK_END + "` is quoted.\n" + THINK_END + "\n\nOK",
                                   ThinkTokenizer.THINK_END)
        service = service_for(engine, tokenizer, quoted_eos_recovery=True, think_eos_ban=True)
        r = run(service, tokenizer)
        self.assertEqual(self.biases(engine), [self.BAN, self.BAN, None])
        self.assertEqual(r["reasoning"], "The tag `" + THINK_END + "` is quoted.\n")
        self.assertEqual(r["content"], "OK")

    def test_without_the_ban_no_pass_is_banned_and_none_is_cut(self):
        """The engine's hold ends a pass at every </think>, quoted or not, and the server goes on each time, as before."""
        tokenizer = ThinkTokenizer()
        engine = HoldSessionEngine(tokenizer, "It ends with " + THINK_END + " and then the answer.\n" + THINK_END + "\n\nOK",
                                   ThinkTokenizer.THINK_END)
        service = service_for(engine, tokenizer, quoted_eos_recovery=True)
        r = run(service, tokenizer)
        self.assertEqual(self.biases(engine), [None, None, None])
        self.assertEqual((r["content"], r["done"]["finish"]), ("OK", "stop"))
        self.assertNotIn("think_eos_bans", r["done"])

    def test_a_forced_call_still_opens_where_the_thinking_ends(self):
        tokenizer = ByteTokenizer()
        engine = ForcedCallEngine(tokenizer)
        service = service_for(engine, tokenizer, quoted_eos_recovery=True, think_eos_ban=True)
        r = run(service, tokenizer, tools=SCHEMA + [{"name": "search", "parameters": {"type": "object", "properties": {"q": {}}}}],
                force="<tool_call>\n")
        self.assertEqual([(c.name, c.arguments) for c in r["calls"]], [("search", {"q": "2+2"})])
        self.assertEqual(self.biases(engine), [self.BAN, None])
        self.assertEqual(r["done"]["think_eos_bans"], 1)


if __name__ == '__main__':
    unittest.main()
