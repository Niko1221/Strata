"""#3441 (a call whose envelope drifted by a token) and #3075 (a call for a tool the request never declared)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontend  # noqa: E402

TOOLS = [{"name": "functions.exec", "parameters": {"properties": {"input": {"type": "string"}}}},
         {"name": "shell", "parameters": {"properties": {"cmd": {"type": "string"}}}},
         {"name": "ns.web_search", "parameters": {"properties": {"q": {"type": "string"}}}}]


def body(opener, param="input", value="1+1", name=None):
    return f"<tool_call>\n{opener}\n<parameter={param}>\n{value}\n</parameter>\n</function>\n</tool_call>"


def run(text, width=0, **kw):
    """-> (events, parser); `width` 0 feeds the text whole, else that many characters at a time."""
    p = frontend.OutputParser(thinking=False, stream_tools=True, **kw)
    evs = []
    if width:
        for k in range(0, len(text), width):
            evs += p.feed(text[k:k + width])
    else:
        evs += p.feed(text)
    evs += p.finish()
    return evs, p


def calls(evs):
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


def text(evs):
    return "".join(e.text or "" for e in evs if e.kind == "content")


class DriftedEnvelope(unittest.TestCase):
    def test_a_dropped_function_keyword_is_still_the_call(self):
        for opener in ("<=functions.exec>", "<function:functions.exec>", "<function =functions.exec>",
                       "<functions.exec>", "<= functions.exec >"):
            for width in (0, 1, 2, 3, 7):
                with self.subTest(opener=opener, width=width):
                    evs, p = run(body(opener), width, tools=TOOLS)
                    self.assertEqual(calls(evs), [("functions.exec", {"input": "1+1"})])
                    self.assertEqual(text(evs), "")
                    self.assertEqual(p.unread, 0)

    def test_the_template_form_is_unchanged(self):
        evs, p = run(body("<function=functions.exec>"), 0, tools=TOOLS)
        self.assertEqual(calls(evs), [("functions.exec", {"input": "1+1"})])

    def test_an_undeclared_name_in_a_drifted_opener_stays_text(self):
        for width in (0, 1, 4):
            with self.subTest(width=width):
                raw = body("<=missing>")
                evs, p = run(raw, width, tools=TOOLS)
                self.assertEqual(calls(evs), [])
                self.assertEqual(text(evs), raw)
                self.assertEqual(p.unread, 1)
                self.assertFalse([e for e in evs if e.kind == "tool_start"])

    def test_prose_about_the_tag_is_not_a_drift(self):
        for raw in ("Use a <tool_call> block for that.", "<tool_call> then the name"):
            evs, p = run(raw, 3, tools=TOOLS)
            self.assertEqual(calls(evs), [])
            self.assertEqual(text(evs), raw)
            self.assertEqual(p.unread, 0)

    def test_streamed_and_whole_agree(self):
        raw = "Let me run it.\n" + body("<=shell>", "cmd", "ls") + "\nDone."
        whole, _ = run(raw, 0, tools=TOOLS)
        for width in (1, 2, 5, 11):
            streamed, _ = run(raw, width, tools=TOOLS)
            self.assertEqual(calls(streamed), calls(whole))
            self.assertEqual(text(streamed), text(whole))
        self.assertEqual(calls(whole), [("shell", {"cmd": "ls"})])
        self.assertEqual(text(whole), "Let me run it.Done.")


BASH = [{"name": "Bash", "parameters": {"properties": {"command": {"type": "string"},
                                                       "description": {"type": "string"},
                                                       "timeout": {"type": "integer"}},
                                         "required": ["command"]}},
        {"name": "Read", "parameters": {"properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}]

# A reply seen in the wild: prose cut mid-word, then Claude's `<invoke name=...>` dialect inside a `<tool_call>`, ended by
# the stop token right after `</invoke>` (no `</tool_call>`).
INVOKE_REPLY = ('Waiting for the bu<tool_call>\n<invoke name="Bash">\n'
                '<parameter name="command">sleep 1; tail -3 "build.log"</parameter>\n'
                '<parameter name="description">peek build progress</parameter>\n</invoke>')


class InvokeDialect(unittest.TestCase):
    """A call written in Claude's `<invoke name="NAME"><parameter name="K">V</parameter></invoke>` form inside the `<tool_call>`
    wrapper is the call.  The engine's tool-name constraint cannot help here (it arms on `<function=`), and without this the whole
    call leaked into the answer with finish_reason stop, so the agent stopped and the user saw raw XML."""

    def test_the_incident_reply_is_a_call(self):
        for width in (0, 1, 2, 3, 7, 16):
            with self.subTest(width=width):
                evs, p = run(INVOKE_REPLY, width, recover=True, tools=BASH)
                self.assertEqual(calls(evs), [("Bash", {"command": 'sleep 1; tail -3 "build.log"',
                                                        "description": "peek build progress"})])
                self.assertEqual(text(evs), "Waiting for the bu")

    def test_an_explicit_closer_and_a_following_call(self):
        raw = ('<tool_call>\n<invoke name="Read">\n<parameter name="file_path">/a</parameter>\n</invoke>\n</tool_call>\n'
               '<tool_call>\n<invoke name="Bash">\n<parameter name="command">ls</parameter>\n<parameter name="timeout">5</parameter>\n'
               '</invoke>\n</tool_call>')
        for width in (0, 1, 5, 13):
            with self.subTest(width=width):
                evs, p = run(raw, width, recover=True, tools=BASH)
                self.assertEqual(calls(evs), [("Read", {"file_path": "/a"}), ("Bash", {"command": "ls", "timeout": 5})])
                self.assertEqual(text(evs), "")

    def test_a_value_may_hold_the_dialect_tags(self):
        value = 'echo "</parameter> and </invoke> and <parameter name=x>"'
        raw = f'<tool_call>\n<invoke name="Bash">\n<parameter name="command">{value}</parameter>\n</invoke>\n</tool_call>'
        for width in (0, 1, 6):
            with self.subTest(width=width):
                evs, p = run(raw, width, recover=True, tools=BASH)
                self.assertEqual(calls(evs), [("Bash", {"command": value})])

    def test_an_undeclared_name_stays_text(self):
        raw = '<tool_call>\n<invoke name="TaskCreate">\n<parameter name="subject">x</parameter>\n</invoke>\n</tool_call>'
        for width in (0, 1, 4):
            with self.subTest(width=width):
                evs, p = run(raw, width, recover=True, tools=BASH)
                self.assertEqual(calls(evs), [])
                self.assertEqual(text(evs), raw)

    def test_prose_and_code_are_not_a_call(self):
        for raw in ('Write <tool_call><invoke name="Bash"> to call it.\n```\n<tool_call>\n<invoke name="Bash">\n'
                    '<parameter name="command">ls</parameter>\n</invoke>\n```',
                    "The <invoke tag is Claude's."):
            with self.subTest(raw=raw[:30]):
                evs, p = run(raw, 3, recover=True, tools=BASH)
                self.assertEqual(calls(evs), [])
                self.assertEqual(text(evs), raw)

    def test_a_call_cut_by_the_token_limit_stays_text(self):
        raw = '<tool_call>\n<invoke name="Bash">\n<parameter name="command">sleep 1; tail'
        p = frontend.OutputParser(thinking=False, stream_tools=True, recover=True, tools=BASH)
        evs = p.feed(raw) + p.finish("length")
        self.assertEqual(calls(evs), [])
        self.assertEqual(text(evs), raw)

    def test_the_output_ending_inside_the_opener_loses_nothing(self):
        for raw in ('ok<tool_call>\n<invoke name="Bash">', 'ok<tool_call>\n<invoke name="Bash">\n\n',
                    'ok<tool_call>\n<invoke name="Ba', 'ok<tool_call>\n<inv', 'ok<tool_call>'):
            for width in (0, 1, 4):
                with self.subTest(raw=raw[-14:], width=width):
                    evs, p = run(raw, width, recover=True, tools=BASH)
                    self.assertEqual((calls(evs), text(evs)), ([], raw))

    def test_a_dialect_call_after_the_reasoning(self):
        p = frontend.OutputParser(thinking=True, tools=BASH, stream_tools=True, recover=True)
        evs = []
        raw = "plan</think>\n\nGo.\n" + INVOKE_REPLY
        for k in range(0, len(raw), 5):
            evs += p.feed(raw[k:k + 5])
        evs += p.finish()
        self.assertEqual([c[0] for c in calls(evs)], ["Bash"])
        self.assertEqual([e.text for e in evs if e.kind == "reasoning"], ["plan"])
        self.assertNotIn("invoke", text(evs))

    def test_off_without_recovery_as_before(self):
        evs, p = run(INVOKE_REPLY, 4, tools=BASH)
        self.assertEqual(calls(evs), [])
        self.assertEqual(text(evs), INVOKE_REPLY)


def streamed_args(evs):
    """The arguments a streaming client assembles from tool_args events, one string per call."""
    import json
    out, cur = [], None
    for e in evs:
        if e.kind == "tool_start":
            cur = ""
            out.append(None)
        elif e.kind == "tool_args" and cur is not None:
            cur += e.text
            out[-1] = cur
    return [json.loads(s) if s else None for s in out]


class MoreDialects(unittest.TestCase):
    """The other spellings Qwen3.x models are seen to write for a call (Llama/Groq JSON body, Anthropic's parameter attribute inside
    `<function=>`, `<function_calls>` with a batch of `<invoke>`s).  Each is the call, and a streaming client sees the same arguments."""

    WIDTHS = (0, 1, 2, 3, 7, 16)

    def check(self, raw, expect, streams=True):
        """`streams`: a `<function=>` call is announced and its arguments streamed (the client assembles the same arguments);
        an `<invoke>` call is held whole and comes as one event."""
        for width in self.WIDTHS:
            with self.subTest(width=width):
                evs, p = run(raw, width, recover=True, tools=BASH)
                self.assertEqual(calls(evs), expect)
                self.assertEqual(streamed_args(evs), [a for _, a in expect] if streams else [])
                self.assertEqual(text(evs), "")
                self.assertEqual(p.unread, 0)

    def test_a_json_body(self):
        self.check('<tool_call>\n<function=Bash>\n{"command": "ls -la", "timeout": 5}\n</function>\n</tool_call>',
                   [("Bash", {"command": "ls -la", "timeout": 5})])

    def test_a_json_body_in_a_call_without_the_wrapper(self):
        evs, p = run('<function=Bash>{"command": "ls"}</function>', 3, recover=True, tools=BASH)
        self.assertEqual(calls(evs), [("Bash", {"command": "ls"})])

    def test_a_parameter_attribute_inside_function(self):
        self.check('<tool_call>\n<function=Bash>\n<parameter name="command">echo "</tool_call>"</parameter>\n'
                   '<parameter name="timeout">5</parameter>\n</function>\n</tool_call>',
                   [("Bash", {"command": 'echo "</tool_call>"', "timeout": 5})])

    def test_a_function_calls_wrapper(self):
        self.check('<function_calls>\n<invoke name="Bash">\n<parameter name="command">ls</parameter>\n</invoke>\n</function_calls>',
                   [("Bash", {"command": "ls"})], streams=False)

    def test_a_batch_in_one_wrapper(self):
        raw = ('<function_calls>\n<invoke name="Read">\n<parameter name="file_path">/a</parameter>\n</invoke>\n'
               '<invoke name="Bash">\n<parameter name="command">ls</parameter>\n</invoke>\n</function_calls>')
        for width in self.WIDTHS:
            with self.subTest(width=width):
                evs, p = run(raw, width, recover=True, tools=BASH)
                self.assertEqual(calls(evs), [("Read", {"file_path": "/a"}), ("Bash", {"command": "ls"})])
                self.assertEqual(text(evs), "")

    def test_prose_about_function_calls_is_text(self):
        for raw in ("Wrap them in <function_calls> and close it.", '<function_calls> then <invoke name="Nope">',
                    'Use <function_calls>\n<invoke name="Nope">\n</invoke>\n</function_calls> for that.'):
            for width in (0, 3):
                with self.subTest(raw=raw[:24], width=width):
                    evs, p = run(raw, width, recover=True, tools=BASH)
                    self.assertEqual((calls(evs), text(evs)), ([], raw))

    def test_a_wrapper_still_arriving_loses_nothing(self):
        for raw in ('ok<function_calls>', 'ok<function_calls>\n<invoke name="Bash">', "ok<function_c"):
            with self.subTest(raw=raw):
                evs, p = run(raw, 2, recover=True, tools=BASH)
                self.assertEqual((calls(evs), text(evs)), ([], raw))

    def test_a_body_that_cannot_be_read_is_named_in_the_notes(self):
        evs, p = run('<tool_call>\n<function=Bash>\nls -la\n</function>\n</tool_call>', 0, recover=True, tools=BASH)
        self.assertEqual(calls(evs), [("Bash", {})])
        self.assertTrue([n for n in p.notes if "unreadable" in n and "ls -la" in n], p.notes)
        evs, p = run('<tool_call>\n<function=Bash>\n</function>\n</tool_call>', 0, recover=True, tools=BASH)      # truly empty: no note
        self.assertEqual((calls(evs), p.notes), ([("Bash", {})], []))


class OnlyDeclaredTools(unittest.TestCase):
    def test_a_call_for_another_tool_stays_text(self):
        raw = body("<function=get_indicators>", "symbol", "600076")
        for width in (0, 1, 3, 9):
            with self.subTest(width=width):
                evs, p = run("I will check.\n" + raw, width, tools=TOOLS)
                self.assertEqual(calls(evs), [])
                self.assertFalse([e for e in evs if e.kind in ("tool_start", "tool_args")])
                self.assertEqual(text(evs), "I will check.\n" + raw)

    def test_an_unparsed_call_is_counted_and_named_for_the_log(self):
        """T1 rail: a call that went back as text is no longer silent; the server prints its count and the names the model wrote."""
        raw = body("<function=get_indicators>", "symbol", "600076")
        evs, p = run("I will check."+chr(10) + raw, 4, tools=TOOLS)
        self.assertEqual((p.unread, p.unread_names), (1, ["get_indicators"]))
        evs, p = run(body("<function=shell>", "cmd", "ls"), 4, tools=TOOLS)         # a declared call is no unread call
        self.assertEqual((p.unread, p.unread_names), (0, []))

    def test_a_declared_call_after_an_undeclared_one_is_still_a_call(self):
        raw = body("<function=get_indicators>") + "\n" + body("<function=shell>", "cmd", "ls")
        evs, p = run(raw, 5, tools=TOOLS)
        self.assertEqual(calls(evs), [("shell", {"cmd": "ls"})])
        self.assertIn("get_indicators", text(evs))

    def test_the_responses_spelling_of_a_namespace_tool_counts(self):
        evs, p = run(body("<function=ns__web_search>", "q", "x"), 4, tools=TOOLS)
        # PR #3556: the call carries the declared spelling (ns.web_search), the one the client offered, not the model's flat one
        self.assertEqual(calls(evs), [("ns.web_search", {"q": "x"})])

    def test_no_tools_declared(self):
        raw = body("<function=search>", "q", "2+2")
        evs, p = run(raw, 3, tools=None, declared_only=True)
        self.assertEqual((calls(evs), text(evs)), ([], raw))
        evs, p = run(raw, 3, tools=[], declared_only=True)
        self.assertEqual((calls(evs), text(evs)), ([], raw))
        # without the flag (the Responses route: a Codex compaction request keeps its tools in the prompt only)
        evs, p = run(raw, 3, tools=None)
        self.assertEqual(calls(evs), [("search", {"q": "2+2"})])

    def test_the_reasoning_state_is_as_before(self):
        p = frontend.OutputParser(thinking=True, tools=TOOLS, stream_tools=True)
        evs = p.feed("think " + body("<function=shell>", "cmd", "ls") + "</think>\n\nok") + p.finish()
        self.assertEqual(calls(evs), [])


if __name__ == "__main__":
    unittest.main()
