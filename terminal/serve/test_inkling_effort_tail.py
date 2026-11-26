"""#7241 (opt-in): the thinking level of Inkling-2-Mini at the END of the prompt, so a request that changes only the level keeps the cached conversation.

With the real chat template (terminal/tests/data/inkling/chat_template.jinja) the level line `Thinking effort level: X` is written at the START of the
prompt, so a changed level means a different start and the engine reads the whole conversation again (measured on an RTX 3090: +10 s at 20K tokens,
+29 s at 50K, +71 s at 100K).  With the config's "effort_position": "end" the history is rendered at the constant default level and a request at another
level adds the template's own system turn at the end, right before the answer.  Control's half only: the ENGINE must checkpoint in front of that turn
(--tail-role-token); an engine without it leaves the option off.  These tests pin the prompt text, the config switch and that nothing changes without it.

    python -m pytest serve/test_inkling_effort_tail.py
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve import frontend, inkling_dialect, server  # noqa: E402

TEMPLATE = ROOT / "tests" / "data" / "inkling" / "chat_template.jinja"
TURN = "<|message_system|><|content_text|>Thinking effort level: {}<|end_message|>"
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file.",
                                           "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]


def chat(n):
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer", "reasoning_content": "thought one"}, {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer", "reasoning_content": "thought two"}, {"role": "user", "content": "third question"}]
    return msgs[:n]


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tpl = frontend.ChatTemplate(TEMPLATE)

    def svc(self, end):
        return types.SimpleNamespace(hy4=False, inkling=True, template=self.tpl, effort_end=end)

    def render(self, messages, tools=None, end=True, **kwargs):
        return server.Service.render_prompt(self.svc(end), messages, tools, kwargs)


class Tail(Base):
    def test_the_trailing_turn_is_the_templates_own_line_at_the_requests_level(self):
        for ask, shown in (("high", "0.9"), ("medium", "0.7"), ("max", "0.99"), ("xhigh", "0.99"), ("none", "0"), ("minimal", "0.1"), (0.5, "0.5")):
            head, tail = inkling_dialect.effort_tail({"reasoning_effort": ask})
            self.assertEqual((head, tail), ("low", TURN.format(shown)), ask)
        self.assertEqual(inkling_dialect.effort_tail({"enable_thinking": False}), ("low", TURN.format("0")))

    def test_the_trailing_turn_reads_like_the_template_writes_it(self):
        # the same text the template puts in its header for each level
        for ask in ("none", "minimal", "low", "medium", "high", "max", 0.33):
            header = self.tpl.render([{"role": "user", "content": "x"}], reasoning_effort=ask)
            level = header.split("Thinking effort level: ")[1].split("<|end_message|>")[0]
            self.assertEqual(inkling_dialect.EFFORT_TAIL_TURN.format(level), TURN.format(level))
            if ask != "low":
                self.assertEqual(inkling_dialect.effort_tail({"reasoning_effort": ask})[1], TURN.format(level), ask)

    def test_the_default_level_has_no_trailing_turn_and_renders_as_without_the_option(self):
        for kw in ({}, {"reasoning_effort": "low"}, {"reasoning_effort": 0.2}):
            self.assertEqual(inkling_dialect.effort_tail(kw)[1], "")
            self.assertEqual(self.render(chat(4), **kw), self.render(chat(4), end=False, **kw))

    def test_a_changed_level_keeps_everything_before_the_answer(self):
        base = self.render(chat(4))                                     # default level
        for kw in ({"reasoning_effort": "high"}, {"reasoning_effort": "none"}, {"reasoning_effort": "medium"}):
            with self.subTest(kw=kw):
                prompt = self.render(chat(4), **kw)
                history = self.tpl.render(inkling_dialect.prepare_messages(chat(4)), **inkling_dialect.template_kwargs({}))
                self.assertTrue(prompt.startswith(history + TURN.format(inkling_dialect.level_text(inkling_dialect.effort_of(kw)[1]))), prompt[-200:])
                self.assertTrue(prompt.startswith(history))
                self.assertTrue(base.startswith(history))
                # without the option the level sits in the header: another start
                self.assertFalse(self.render(chat(4), end=False, **kw).startswith(history))

    def test_the_next_turn_starts_with_the_conversation_of_the_turn_before_even_when_the_level_changed(self):
        # turn 1 at high, turn 2 at low (a different level): with the option the new prompt still starts with the whole first conversation;
        # that is what the engine's checkpoint in front of the trailing turn reuses
        first = self.render(chat(4), reasoning_effort="high")
        second = self.render(chat(6), reasoning_effort="low")
        conversation_one = first[:first.rindex(TURN.format("0.9"))]
        self.assertTrue(second.startswith(conversation_one), "the first turn's conversation is the start of the second prompt")
        self.assertGreater(len(conversation_one), 300)
        # and without the option nothing of it matches past the level line in the header
        a = self.render(chat(4), end=False, reasoning_effort="high")
        b = self.render(chat(6), end=False, reasoning_effort="low")
        common = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        self.assertLess(common, len(conversation_one) // 2)

    def test_the_answer_opens_after_the_trailing_turn(self):
        text = self.render(chat(4), reasoning_effort="none")            # level 0: the text opener
        self.assertTrue(text.endswith(TURN.format("0") + "<|message_model|><|content_text|>"), text[-160:])
        think = self.render(chat(4), reasoning_effort="high")           # thinking: the model opens its own thinking turn
        self.assertTrue(think.endswith(TURN.format("0.9")), think[-160:])
        tools = self.render(chat(4), tools=TOOLS, reasoning_effort="none")
        self.assertTrue(tools.endswith(TURN.format("0") + "<|message_model|><|content_thinking|><|end_message|>"), tools[-200:])

    def test_a_prefilled_reply_keeps_the_level_at_the_top(self):
        msgs = chat(4) + [{"role": "assistant", "content": "Sure, "}]
        self.assertEqual(self.render(msgs, reasoning_effort="high"), self.render(msgs, end=False, reasoning_effort="high"))

    def test_without_the_option_nothing_changes(self):
        off = types.SimpleNamespace(hy4=False, inkling=True, template=self.tpl)            # an old Service has no effort_end at all
        for kw in ({}, {"reasoning_effort": "high"}, {"reasoning_effort": "none"}, {"enable_thinking": False}):
            self.assertEqual(server.Service.render_prompt(off, chat(4), None, kw), self.render(chat(4), end=False, **kw))
            self.assertNotIn(TURN.format("0.9"), self.render(chat(4), end=False, **kw)[-80:])


class Config(unittest.TestCase):
    class Tok:
        def encode(self, text, parse_special=False):
            return [200012] if text == "<|message_system|>" else [8678] if text == "system" else [1, 2]

    def run_cfg(self, exe_text, cfg, name="strata-inkling.exe", tok=None):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / name
            exe.write_bytes(exe_text)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                got = server.effort_end_args(cfg, str(exe), tok or self.Tok())
            return got, out.getvalue()

    def test_an_inkling_engine_with_the_checkpoint_gets_the_system_role_token(self):
        got, said = self.run_cfg(b"... --tail-role-token ID ...", {"effort_position": "end"})
        self.assertEqual(got, ["--tail-role-token", "200012"])
        self.assertIn("blends", said)

    def test_an_inkling_engine_without_it_leaves_the_level_at_the_top_and_says_so_in_its_own_words(self):
        got, said = self.run_cfg(b"... --serve ...", {"effort_position": "end"})
        self.assertIsNone(got)
        self.assertIn("Inkling-2-Mini needs an Inkling engine that keeps a checkpoint", said)
        self.assertNotIn("0.1.39", said)                     # the main engine's sentence is not about this one

    def test_the_default_start_is_untouched(self):
        for cfg in ({}, {"effort_position": "start"}):
            self.assertEqual(self.run_cfg(b"--tail-role-token", cfg), (None, ""))

    def test_the_main_engine_keeps_its_own_rule(self):
        got, _ = self.run_cfg(b"... --tail-role-token ID ...", {"effort_position": "end"}, name="strata.exe")
        self.assertEqual(got, ["--tail-role-token", "8678"])

    def test_a_template_without_a_single_system_token_is_refused_quietly(self):
        class Two:
            def encode(self, text, parse_special=False):
                return [1, 2]
        got, said = self.run_cfg(b"--tail-role-token", {"effort_position": "end"}, tok=Two())
        self.assertIsNone(got)
        self.assertIn("level stays at the top", said)


class PrefixHeads(Base):
    def test_the_strata_prefix_head_of_an_inkling_conversation_is_rendered_at_the_constant_level(self):
        # the strata_prefix cut renders the history without the request's level when the option is on (the same text render_prompt starts with)
        src = Path(server.__file__).read_text(encoding="utf-8")
        self.assertIn("self.effort_end and (self.inkling or off or effort not in", src)


if __name__ == "__main__":
    unittest.main()
