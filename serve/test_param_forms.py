"""serve/test_param_forms.py - #2451: parameter tags written next to the template's form keep their arguments.

Seen live (Qwen3.8-Flash-Next Coder IQ1_M, a coding agent with 21 tools, "tool_call_recovery": true): the model wrote
`<parameter name="command">`, `<parameter name=command>` or a nameless `<parameter>` holding all arguments as JSON.
The calls carried the whole command but reached the client with empty arguments.  The samples below are the model's
raw output from the server log (STRATA_DEBUG), shortened where marked.

    python -m unittest serve.test_param_forms -v
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import OutputParser  # noqa: E402

TOOLS = [
    {"name": "Bash", "parameters": {"properties": {"command": {"type": "string"},
                                                   "timeout": {"type": "number"}}}},
    {"name": "retry_code", "parameters": {"properties": {"attempt_id": {"type": "string"},
                                                         "feedback": {"type": "string"}}}},
]
GREP = ("grep -rn \"correction_fit\\|job_type\" lotto_graph/services/lstm/correction/runner.py 2>/dev/null "
        "| grep -v __pycache__ | sed -n '1,40p'")
NAMED_DQ = f'<tool_call>\n<function=Bash>\n<parameter name="command">{GREP}\n</parameter>\n</function>\n</tool_call>'
NAMED_BARE = ("<tool_call>\n<function=Bash>\n<parameter name=command>\nls lotto_graph/worker.py 2>/dev/null; "
              "grep -rln \"job_type\" lotto_graph/worker.py\n</parameter>\n</function>\n</tool_call>")
NAMED_SQ = "<tool_call>\n<function=Bash>\n<parameter name='command'>\nls -la\n</parameter>\n</function>\n</tool_call>"
BARE_JSON = ('<tool_call>\n<function=retry_code>\n<parameter>\n{"attempt_id": "bf3b0e1cb16b", "feedback": '
             '"save_backtest has NO `error_message` parameter (shortened); put it in metrics={\\"error_type\\": 1}"}'
             '\n</parameter>\n</function>\n</tool_call>')
CANONICAL = "<tool_call>\n<function=Bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>"


def run(text, *, recover=True, stream_tools=True, step=3, thinking=True):
    p = OutputParser(thinking=thinking, tools=TOOLS, stream_tools=stream_tools, recover=recover)
    evs = []
    for i in range(0, len(text), step):                # fed in small pieces, as a stream arrives
        evs += p.feed(text[i:i + step])
    evs += p.finish("stop")
    return evs


def calls(text, **kw):
    return [(e.call.name, e.call.arguments) for e in run(text, **kw) if e.kind == "tool_call"]


def visible(text, **kw):
    return "".join(e.text for e in run(text, **kw) if e.kind in ("content", "reasoning") and e.text)


class ParameterForms(unittest.TestCase):
    def check(self, call, expected):
        for stream_tools in (False, True):
            for step in (1, 3, 7, 1000):
                for lead in ("Running it.</think>\n\n", "Let me check.\n"):   # after / inside the thinking
                    with self.subTest(stream_tools=stream_tools, step=step, lead=lead[:8]):
                        self.assertEqual(calls(lead + call, stream_tools=stream_tools, step=step), [expected])

    def test_name_in_double_quotes(self):
        self.check(NAMED_DQ, ("Bash", {"command": GREP}))

    def test_name_without_quotes(self):
        self.check(NAMED_BARE, ("Bash", {"command": "ls lotto_graph/worker.py 2>/dev/null; "
                                                    "grep -rln \"job_type\" lotto_graph/worker.py"}))

    def test_name_in_single_quotes(self):
        self.check(NAMED_SQ, ("Bash", {"command": "ls -la"}))

    def test_nameless_parameter_holding_json(self):
        self.check(BARE_JSON, ("retry_code", {
            "attempt_id": "bf3b0e1cb16b",
            "feedback": 'save_backtest has NO `error_message` parameter (shortened); put it in '
                        'metrics={"error_type": 1}'}))

    def test_the_template_form_is_unchanged(self):
        self.check(CANONICAL, ("Bash", {"command": "ls -la"}))

    def test_the_streamed_arguments_are_the_call_s_arguments(self):
        # the server streams a call's JSON as "tool_args" pieces while it is written: they must add up to the same
        # arguments as the final "tool_call", not to {}
        for text in (NAMED_DQ, NAMED_BARE, BARE_JSON):
            for step in (1, 3, 1000):
                with self.subTest(text=text[:40], step=step):
                    evs = run("Go.</think>\n\n" + text, step=step)
                    streamed = "".join(e.text for e in evs if e.kind == "tool_args")
                    final = [e.call.arguments for e in evs if e.kind == "tool_call"]
                    self.assertEqual([json.loads(streamed)], final)

    def test_the_rewrites_are_counted_for_the_log(self):
        p = OutputParser(thinking=False, tools=TOOLS, stream_tools=True, recover=True)
        p.feed(NAMED_DQ)
        p.finish("stop")
        self.assertEqual(p.params_rewritten, 1)
        p = OutputParser(thinking=False, tools=TOOLS, stream_tools=True, recover=True)
        p.feed(CANONICAL)
        p.finish("stop")
        self.assertEqual(p.params_rewritten, 0)

    def test_a_non_string_value_from_json_keeps_its_type(self):
        call = ('<tool_call>\n<function=Bash>\n<parameter>{"command": "sleep 1", "timeout": 30}</parameter>\n'
                '</function>\n</tool_call>')
        self.check(call, ("Bash", {"command": "sleep 1", "timeout": 30}))


class OnlyWhereItBelongs(unittest.TestCase):
    def test_off_without_tool_call_recovery(self):
        # without the switch a client gets what it got before: the arguments are lost
        self.assertEqual(calls("Go.</think>\n\n" + NAMED_DQ, recover=False), [("Bash", {})])

    def test_text_outside_a_call_is_untouched(self):
        text = 'The tag <parameter name="x"> and <parameter>{"a": 1}</parameter> are examples.</think>\n\nDone.'
        self.assertEqual(visible(text), text.replace("</think>\n\n", ""))

    def test_a_value_that_shows_the_forms_is_kept_210(self):
        # a file that documents the call format (#210): the forms inside a value are its text, not tags
        doc = 'See <parameter name="x"> and <parameter>{"a": 1}</parameter> in the docs.'
        call = ("<tool_call>\n<function=Write>\n<parameter=content>\n" + doc + "\n</parameter>\n"
                "</function>\n</tool_call>")
        tools = TOOLS + [{"name": "Write", "parameters": {"properties": {"content": {"type": "string"}}}}]
        for step in (1, 3, 1000):
            with self.subTest(step=step):
                p = OutputParser(thinking=True, tools=tools, stream_tools=True, recover=True)
                evs = []
                text = "Go.</think>\n\n" + call
                for i in range(0, len(text), step):
                    evs += p.feed(text[i:i + step])
                evs += p.finish("stop")
                self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"content": doc}])
                self.assertEqual(json.loads("".join(e.text for e in evs if e.kind == "tool_args")), {"content": doc})
                self.assertEqual(p.params_rewritten, 0)

    def test_a_value_holding_function_end_keeps_the_next_parameter(self):
        # #210: a value may contain </function>; only a </parameter> followed by the next tag ends it
        call = ('<tool_call>\n<function=Write>\n<parameter=content>\nuse </function> to close\n</parameter>\n'
                '<parameter name="path">\na.md\n</parameter>\n</function>\n</tool_call>')
        tools = TOOLS + [{"name": "Write", "parameters": {"properties": {"content": {"type": "string"},
                                                                         "path": {"type": "string"}}}}]
        for step in range(1, 8):
            with self.subTest(step=step):
                p = OutputParser(thinking=True, tools=tools, stream_tools=True, recover=True)
                evs, text = [], "Go.</think>\n\n" + call
                for i in range(0, len(text), step):
                    evs += p.feed(text[i:i + step])
                evs += p.finish("stop")
                self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"],
                                 [{"content": "use </function> to close", "path": "a.md"}])

    def test_prose_that_names_a_function_is_untouched(self):
        text = 'Use <function=Bash> for it, e.g. <parameter name="x"> here.</think>\n\nDone.'
        self.assertEqual(visible(text), visible(text, recover=False))

    def test_a_nameless_parameter_that_is_not_json_stays_as_written(self):
        call = "<tool_call>\n<function=Bash>\n<parameter>\nls\n</parameter>\n</function>\n</tool_call>"
        self.assertEqual(calls("Go.</think>\n\n" + call, recover=True),
                         calls("Go.</think>\n\n" + call, recover=False))


if __name__ == "__main__":
    unittest.main()
