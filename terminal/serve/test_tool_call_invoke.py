"""serve/test_tool_call_invoke.py - #5969's parser part (recovery on): a call written in Claude's
`<invoke name="NAME"><parameter name="K">V</parameter></invoke>` dialect inside the `<tool_call>` wrapper is the call
(cases from PR #5969's test_tool_call_envelope.py InvokeDialect, Sergey Zinchenko; the incident reply ended at
`</invoke>` with no `</tool_call>`, and leaked into the answer as raw XML with finish "stop").  A cut call, an
undeclared name, prose and code stay text; with recovery off nothing changes.

    python -m pytest serve/test_tool_call_invoke.py -q
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import frontend  # noqa: E402

BASH = [{"name": "Bash", "parameters": {"properties": {"command": {"type": "string"},
                                                       "description": {"type": "string"},
                                                       "timeout": {"type": "integer"}},
                                         "required": ["command"]}},
        {"name": "Read", "parameters": {"properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}]
INVOKE_REPLY = ('Waiting for the bu<tool_call>\n<invoke name="Bash">\n'
                '<parameter name="command">sleep 1; tail -3 "build.log"</parameter>\n'
                '<parameter name="description">peek build progress</parameter>\n</invoke>')


def run(text, width=0, stream=True, recover=True, finish=None, thinking=False):
    p = frontend.OutputParser(thinking=thinking, stream_tools=stream, recover=recover, tools=BASH)
    evs = []
    for k in range(0, len(text), width or len(text) or 1):
        evs += p.feed(text[k:k + (width or len(text))])
    return evs + p.finish(finish)


def calls(evs):
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


def text(evs):
    return "".join(e.text or "" for e in evs if e.kind == "content")


class InvokeDialect(unittest.TestCase):
    def test_the_incident_reply_is_a_call(self):
        for stream in (False, True):
            for width in (0, 1, 2, 3, 7, 16):
                with self.subTest(stream=stream, width=width):
                    evs = run(INVOKE_REPLY, width, stream)
                    self.assertEqual(calls(evs), [("Bash", {"command": 'sleep 1; tail -3 "build.log"',
                                                            "description": "peek build progress"})])
                    self.assertEqual(text(evs), "Waiting for the bu")

    def test_an_explicit_closer_and_a_following_call(self):
        raw = ('<tool_call>\n<invoke name="Read">\n<parameter name="file_path">/a</parameter>\n</invoke>\n</tool_call>\n'
               '<tool_call>\n<invoke name="Bash">\n<parameter name="command">ls</parameter>\n<parameter name="timeout">5</parameter>\n'
               '</invoke>\n</tool_call>')
        for width in (0, 1, 5, 13):
            with self.subTest(width=width):
                evs = run(raw, width)
                self.assertEqual(calls(evs), [("Read", {"file_path": "/a"}), ("Bash", {"command": "ls", "timeout": 5})])
                self.assertEqual(text(evs).strip(), "")

    def test_a_value_may_hold_the_dialect_tags(self):
        value = 'echo "</parameter> and </invoke> and <parameter name=x>"'
        raw = f'<tool_call>\n<invoke name="Bash">\n<parameter name="command">{value}</parameter>\n</invoke>\n</tool_call>'
        for width in (0, 1, 6):
            with self.subTest(width=width):
                self.assertEqual(calls(run(raw, width)), [("Bash", {"command": value})])

    def test_an_undeclared_name_stays_text(self):
        raw = '<tool_call>\n<invoke name="TaskCreate">\n<parameter name="subject">x</parameter>\n</invoke>\n</tool_call>'
        for width in (0, 1, 4):
            with self.subTest(width=width):
                evs = run(raw, width)
                self.assertEqual(calls(evs), [])
                self.assertEqual(text(evs), raw)

    def test_prose_and_code_are_not_a_call(self):
        for raw in ('Write <tool_call><invoke name="Bash"> to call it.\n```\n<tool_call>\n<invoke name="Bash">\n'
                    '<parameter name="command">ls</parameter>\n</invoke>\n```',
                    "The <invoke tag is Claude's."):
            with self.subTest(raw=raw[:30]):
                evs = run(raw, 3)
                self.assertEqual(calls(evs), [])
                self.assertEqual(text(evs), raw)

    def test_a_call_cut_by_the_token_limit_stays_text(self):
        raw = '<tool_call>\n<invoke name="Bash">\n<parameter name="command">sleep 1; tail'
        for width in (0, 1, 7):
            evs = run(raw, width, finish="length")
            self.assertEqual((calls(evs), text(evs)), ([], raw))

    def test_the_output_ending_inside_the_opener_loses_nothing(self):
        for raw in ('ok<tool_call>\n<invoke name="Bash">', 'ok<tool_call>\n<invoke name="Bash">\n\n',
                    'ok<tool_call>\n<invoke name="Ba', 'ok<tool_call>\n<inv', 'ok<tool_call>'):
            for width in (0, 1, 4):
                with self.subTest(raw=raw[-14:], width=width):
                    evs = run(raw, width)
                    self.assertEqual((calls(evs), text(evs)), ([], raw))

    def test_a_dialect_call_after_the_reasoning(self):
        evs = run("plan</think>\n\nGo.\n" + INVOKE_REPLY, 5, thinking=True)
        self.assertEqual([c[0] for c in calls(evs)], ["Bash"])
        self.assertEqual([e.text for e in evs if e.kind == "reasoning"], ["plan"])
        self.assertNotIn("invoke", text(evs))

    def test_off_without_recovery_as_before(self):
        evs = run(INVOKE_REPLY, 4, recover=False)
        self.assertEqual(calls(evs), [])


# #6022 (Pi on /v1/chat/completions, Qwen3.8-Flash-Next IQ3_S and UD-IQ4_XS): the tool names Pi declares
PI = [{"name": "bash", "parameters": {"properties": {"command": {"type": "string"}}, "required": ["command"]}},
      {"name": "write", "parameters": {"properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                                       "required": ["path", "content"]}},
      {"name": "read", "parameters": {"properties": {"path": {"type": "string"}}, "required": ["path"]}}]


def run_pi(text, width=0, stream=True, recover=True, finish=None):
    p = frontend.OutputParser(thinking=False, stream_tools=stream, recover=recover, tools=PI)
    evs = []
    for k in range(0, len(text), width or len(text) or 1):
        evs += p.feed(text[k:k + (width or len(text))])
    return evs + p.finish(finish)


class InvokePrelude(unittest.TestCase):
    """#6022: a line between `<tool_call>` and `<invoke>` - the `<function=function_calls>` wrapper or a bare `=` - and
    the body closed by `</function>` instead of `</invoke>`; both turns ended with finish "stop" and no call."""
    WRAPPED = ('<tool_call>\n<function=function_calls>\n<invoke name="bash">\n'
               '<parameter name="command">winget search --name Paseo 2>&1 | head -15</parameter>\n'
               '</function>\n</tool_call>')
    BARE_EQ = ('<tool_call>\n=\n<invoke name="write">\n<parameter name="path">C:\\Users\\me\\codex-reply-54655.md\n'
               '</parameter>\n<parameter=content>\n# reply\n</parameter>\n</function>\n</tool_call>')

    def test_the_function_calls_wrapper_is_read(self):
        for stream in (False, True):
            for width in (0, 1, 2, 5, 13):
                with self.subTest(stream=stream, width=width):
                    evs = run_pi("Searching.\n" + self.WRAPPED, width, stream)
                    self.assertEqual(calls(evs), [("bash", {"command": "winget search --name Paseo 2>&1 | head -15"})])
                    self.assertEqual(text(evs), "Searching.")

    def test_a_bare_equals_line_and_mixed_parameter_styles(self):
        for width in (0, 1, 3, 11):
            with self.subTest(width=width):
                evs = run_pi(self.BARE_EQ, width)
                self.assertEqual(calls(evs), [("write", {"path": "C:\\Users\\me\\codex-reply-54655.md",
                                                         "content": "# reply"})])
                self.assertEqual(text(evs).strip(), "")

    def test_the_wrapper_around_a_closed_invoke(self):
        raw = ('<tool_call>\n<function_calls>\n<invoke name="read">\n<parameter name="path">/a</parameter>\n</invoke>\n'
               '</function_calls>\n</tool_call>\nDone.')
        for width in (0, 1, 4):
            with self.subTest(width=width):
                evs = run_pi(raw, width)
                self.assertEqual(calls(evs), [("read", {"path": "/a"})])
                self.assertEqual(text(evs).strip(), "Done.")

    def test_an_undeclared_name_after_the_wrapper_stays_text(self):
        raw = self.WRAPPED.replace('"bash"', '"execute_bash"')
        for width in (0, 1, 6):
            with self.subTest(width=width):
                evs = run_pi(raw, width)
                self.assertEqual(calls(evs), [])
                self.assertEqual(text(evs), raw)

    def test_a_prelude_without_an_invoke_is_text(self):
        for raw in ('<tool_call>\n=\nnothing here', '<tool_call>\n<function=function_calls>\n'):
            for width in (0, 1, 3):
                with self.subTest(raw=raw[-12:], width=width):
                    evs = run_pi(raw, width)
                    self.assertEqual((calls(evs), text(evs)), ([], raw))

    def test_off_without_recovery_as_before(self):
        for raw in (self.WRAPPED, self.BARE_EQ):
            self.assertEqual(calls(run_pi(raw, 4, recover=False)), [])


if __name__ == "__main__":
    unittest.main()
