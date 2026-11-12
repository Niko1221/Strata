"""serve/test_tool_call_recovery_bare.py - a declared tool's call written without its `<tool_call>` wrapper is a call,
also while the model is still thinking, and a lone `<parameter=arguments>` is the argument object itself.

Seen live (Qwen3.8-Flash-Next, UD-IQ3_XXS, opencode through /v1/chat/completions): the model wrote

    <function=exec>
    <parameter=arguments>{"path": "/dev/shm/bcbench-work/search3.py", ...}</parameter>
    </function>

inside its thinking, with no `<tool_call>` around it.  The block came back as reasoning text (and a written
`</tool_call>` even as answer text), so the agent waited for a tool result that never came.  Where `<function=NAME>` of
a declared tool starts a line outside code, it is that call - in the reasoning as in the answer - and the whole
argument object reaches the client as the call's arguments, not as one parameter named "arguments".

    python -m unittest serve.test_tool_call_recovery_bare -v
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import OutputParser  # noqa: E402

TOOLS = [{"name": "exec", "parameters": {"properties": {"path": {"type": "string"},
                                                       "content": {"type": "string"}}}}]
CALL = ('<function=exec>\n<parameter=arguments>{"path": "/dev/shm/bcbench-work/search3.py", '
        '"content": "print(1)"}</parameter>\n</function>')
ARGS = {"path": "/dev/shm/bcbench-work/search3.py", "content": "print(1)"}
WRAPPED = '<tool_call>\n<function=exec>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>'


def run(text, thinking=True, stream_tools=True, step=3, tools=TOOLS, finish="stop", recover=True):
    p = OutputParser(thinking=thinking, tools=list(tools), stream_tools=stream_tools, recover=recover)
    evs = []
    for i in range(0, len(text), step):
        evs += p.feed(text[i:i + step])
    evs += p.finish(finish)
    return evs


def calls(evs):
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


def text(evs):
    return "".join(e.text or "" for e in evs if e.kind in ("content", "reasoning"))


class InTheThinking(unittest.TestCase):
    def test_the_call_is_delivered(self):
        for stream in (True, False):
            for step in (1, 3, 100_000):
                evs = run(CALL + "\n</think>\n\nDone.", stream_tools=stream, step=step)
                self.assertEqual(calls(evs), [("exec", ARGS)], (stream, step))
                self.assertEqual(text(evs), "Done.")

    def test_a_written_closer_next_to_it_is_dropped(self):
        for step in (1, 3, 100_000):
            evs = run(CALL + "\n</tool_call>\n</think>\n\nDone.", step=step)
            self.assertEqual(calls(evs), [("exec", ARGS)], step)
            self.assertEqual(text(evs), "Done.")

    def test_a_turn_that_ends_on_the_call(self):
        self.assertEqual(calls(run(CALL)), [("exec", ARGS)])
        self.assertEqual(calls(run(CALL + "\n</tool_call>")), [("exec", ARGS)])

    def test_prose_after_it_keeps_it_reasoning(self):
        evs = run(CALL + "\nLet me run it.\n</think>\n\nDone.")
        self.assertEqual(calls(evs), [])
        self.assertIn("<function=exec>", text(evs))

    def test_mid_sentence_and_code_stay_text(self):
        for t in ("Use <function=exec> for it.\n</think>\n\nDone.",
                  "```\n" + CALL + "\n```\n</think>\n\nDone.",
                  "Writing `<function=exec>` in prose.\n</think>\n\nDone."):
            evs = run(t)
            self.assertEqual(calls(evs), [], t)

    def test_an_undeclared_name_stays_text(self):
        evs = run(CALL.replace("exec", "not_a_tool") + "\n</think>\n\nDone.")
        self.assertEqual(calls(evs), [])

    def test_the_wrapped_form_is_unchanged(self):
        for stream in (True, False):
            evs = run("<tool_call>\n" + CALL + "\n</tool_call>\n</think>\n\nDone.", stream_tools=stream)
            self.assertEqual(calls(evs), [("exec", ARGS)])
            self.assertEqual(text(evs), "Done.")


class InTheAnswer(unittest.TestCase):
    def test_the_call_is_delivered(self):
        for step in (1, 3, 100_000):
            evs = run(CALL, thinking=False, step=step)
            self.assertEqual(calls(evs), [("exec", ARGS)], step)
            self.assertEqual(text(evs), "")

    def test_a_written_closer_after_it_is_dropped(self):
        evs = run(CALL + "\n</tool_call>", thinking=False)
        self.assertEqual(calls(evs), [("exec", ARGS)])
        self.assertEqual(text(evs), "")


class ArgumentsWrapper(unittest.TestCase):
    def test_unwrapped_when_the_tool_does_not_declare_it(self):
        for stream in (True, False):
            evs = run(WRAPPED.replace(
                "<parameter=path>\na.py\n</parameter>",
                '<parameter=arguments>{"path": "a.py"}</parameter>'), thinking=False, stream_tools=stream)
            self.assertEqual(calls(evs), [("exec", {"path": "a.py"})], stream)

    def test_kept_when_the_tool_declares_it(self):
        tools = [{"name": "exec", "parameters": {"properties": {"arguments": {"type": "object"}}}}]
        evs = run(WRAPPED.replace(
            "<parameter=path>\na.py\n</parameter>",
            '<parameter=arguments>{"path": "a.py"}</parameter>'), thinking=False, tools=tools)
        self.assertEqual(calls(evs), [("exec", {"arguments": {"path": "a.py"}})])

    def test_streamed_arguments_are_the_call_s_arguments(self):
        text = ('<tool_call>\n<function=exec>\n<parameter=arguments>{"path": "a.py"}</parameter>\n'
                '</function>\n</tool_call>')
        for stream in (True, False):
            p = OutputParser(thinking=False, tools=TOOLS, stream_tools=stream, recover=True)
            evs = []
            for i in range(0, len(text), 3):
                evs += p.feed(text[i:i + 3])
            evs += p.finish("stop")
            streamed = "".join(e.text or "" for e in evs if e.kind == "tool_args")
            if stream:
                self.assertEqual(json.loads(streamed), {"path": "a.py"})
            self.assertEqual(calls(evs), [("exec", {"path": "a.py"})])


if __name__ == "__main__":
    unittest.main()
