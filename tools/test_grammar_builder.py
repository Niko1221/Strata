"""The tool-call grammar builder (issue #3862), without a GPU: the prose guard's boundary behaviour,
the template's literals spelled exactly, the escaping, and the MCP routing fallback.

The named-grammar check is structural: Python has no grammar core, so the assertions read the GBNF text
- the same literals the chat template writes (rendered below for one call) must appear in the rules.
"""
import unittest

from serve.frontend import ChatTemplate
from serve.grammar_tools import build_tool_grammar, gbnf_escape, tool_grammar_choice, with_think_prologue
from pathlib import Path

TOOLS = [{"name": "get_weather",
          "parameters": {"type": "object", "required": ["city"],
                         "properties": {"city": {"type": "string"},
                                        "unit": {"type": "string", "enum": ["c", "f"]}}}},
         {"name": "send_sms",
          "parameters": {"type": "object", "required": ["to", "body"],
                         "properties": {"to": {"type": "string"},
                                        "body": {"type": "string"},
                                        "urgent": {"type": "boolean"}}}}]


def chain_accept(seq: str, text: str) -> bool:
    """Mirror of exclude_seq's chain: state k consumes seq[k] to advance, restarts at 0 on a mismatch,
    and the last state refuses the completing character."""
    k = 0
    for c in text:
        if c == seq[k]:
            k += 1
            if k == len(seq):
                return False
        else:
            k = 0
    return True


def convert(sub):
    if sub.get("type") == "boolean":
        return 'root ::= boolean\nboolean ::= "true" | "false"\n'
    raise AssertionError(f"unexpected subschema {sub}")


def build(tool_choice="auto", name=None, thinking=False):
    return build_tool_grammar(TOOLS, tool_choice, name, thinking, convert)


class Prose(unittest.TestCase):
    def test_the_guard_rejects_the_call_marker_and_nothing_it_must_not(self):
        for ok in ("", "<", "<too", "<tool_cal", "a ```python fence``` is prose", "</system>\n<|im_start|>"):
            self.assertTrue(chain_accept("<tool_call>", ok), ok)
        for bad in ("<tool_call>", "x<tool_call>", "<tool_call>\n<function=a>"):
            self.assertFalse(chain_accept("<tool_call>", bad), bad)

    def test_the_auto_root_offers_prose_and_calls(self):
        g = build("auto")
        root = [line for line in g.splitlines() if line.startswith("answer ::=")][0]
        self.assertIn("prose", root)
        self.assertIn("calls", root)
        self.assertIn('"\\n\\n"', root)                       # the template's prose prefix rides along


class Named(unittest.TestCase):
    def test_the_named_grammar_spells_the_template(self):
        g = build("named", name="get_weather")
        lines = g.splitlines()
        self.assertEqual([line for line in lines if line.startswith("root ::=")], ["root ::= answer"])
        answer = [line for line in lines if line.startswith("answer ::=")][0]
        self.assertIn("call-get-weather-0", answer)
        call = [line for line in lines if line.startswith("call-get-weather-0 ::=")][0]
        # the template's literals, in its order: open, required parameter, optional parameter, close
        self.assertIn('"<tool_call>\\n<function=get_weather>\\n"', call)
        self.assertIn("call-get-weather-0-city ", call)        # the required parameter, in schema order
        self.assertIn('"<parameter=city>\\n"', g)
        self.assertIn('"\\n</parameter>\\n"', g)
        self.assertIn('"</function>\\n</tool_call>"', call)
        self.assertIn("( call-get-weather-0-unit )?", call)   # "unit" is not required
        # the other tool's rules are built (they ride the shared calls list) but nothing reaches them:
        # the answer is the named call, and only the "required" roots open calls-one
        self.assertEqual(answer, "answer ::= call-get-weather-0")
        self.assertIn("call-get-weather-0 | call-send-sms-1", [l for l in lines if l.startswith("calls-one ::=")][0])

    def test_the_string_value_rule_refuses_the_parameter_end(self):
        g = build("named", name="get_weather")
        self.assertIn('"\\n</parameter>\\n"', g)                # the parameter tail the value must not write
        self.assertIn("[^\\n]", g)                             # chain state 0 refuses the newline...
        self.assertIn("[^>]", g)                               # ... and the last state the closing bracket

    def test_a_non_string_parameter_inlines_the_converter_under_a_namespace(self):
        g = build("named", name="send_sms")
        self.assertIn("call-send-sms-1-urgent-boolean ::=", g)   # renamed, literals kept
        self.assertIn('"true"', g)

    def test_the_template_renders_exactly_what_the_grammar_demands(self):
        # the one alignment check that matters: the literals below are the text the template writes
        tpl = ChatTemplate(Path(__file__).resolve().parents[1] / "serve" / "chat_template.jinja")
        text = tpl.render([{"role": "user", "content": "weather?"},
                           {"role": "assistant", "content": "",
                            "tool_calls": [{"function": {"name": "get_weather",
                                                         "arguments": {"city": "Oslo", "unit": "c"}}}]}],
                          TOOLS)
        for lit in ('<tool_call>\n<function=get_weather>\n', '<parameter=city>\n', '\n</parameter>\n',
                    '</function>\n</tool_call>'):
            self.assertIn(lit, text)
        g = build("named", name="get_weather")
        for lit in ('<tool_call>\n<function=get_weather>\n', '\n</parameter>\n', '</function>\n</tool_call>'):
            self.assertIn('"' + gbnf_escape(lit) + '"', g)


class Shape(unittest.TestCase):
    def test_parallel_calls_join_with_the_template_newline(self):
        g = build("required")
        self.assertIn('calls-rest ::= "\\n" calls-one calls-rest', g)
        self.assertIn("call-get-weather-0", g)
        self.assertIn("call-send-sms-1", g)

    def test_thinking_prepends_the_prologue(self):
        g = build("required", thinking=True)
        self.assertIn('root ::= reason "</think>\\n" ws answer', g)                # the template's closer and blank line
        self.assertTrue([line for line in g.splitlines() if line.startswith("root ::=")][0].startswith(
            "root ::= reason"))

    def test_literal_escaping(self):
        self.assertEqual(gbnf_escape('say "hi"\n\\x'), 'say \\"hi\\"\\n\\\\x')


class Routing(unittest.TestCase):
    def test_mcp_present_means_the_prompt_path_and_the_response_says_so(self):
        self.assertEqual(tool_grammar_choice(True, TOOLS, "auto", True), ("prompt", "prompt_and_validate", None))

    def test_the_engine_decides(self):
        self.assertEqual(tool_grammar_choice(True, TOOLS, "auto", False), ("grammar", "auto", None))
        self.assertEqual(tool_grammar_choice(True, TOOLS, {"type": "function", "function": {"name": "send_sms"}}, False),
                         ("grammar", "named", "send_sms"))
        self.assertEqual(tool_grammar_choice(False, TOOLS, "required", False), ("prompt", None, None))
        self.assertEqual(tool_grammar_choice(True, TOOLS, "none", False), ("none", None, None))
        self.assertEqual(tool_grammar_choice(True, None, "auto", False), ("none", None, None))
        # a named choice the old code degrades to "auto" keeps the prompt path (grammar would be a policy change)
        self.assertEqual(tool_grammar_choice(True, TOOLS, {"type": "function", "function": {"name": "ghost"}}, False),
                         ("prompt", None, None))


class ThinkPrologue(unittest.TestCase):
    def test_the_answer_grammar_becomes_the_region_after_the_closer(self):
        g = with_think_prologue('root ::= "a" | "b"\n')
        self.assertIn('root ::= reason "</think>\\n" answer', g)
        self.assertIn("answer ::= ws ( fenced | answer-root )", g)
        self.assertIn('fenced ::= "```json" ws answer-root ws "```"', g)   # the habit is legal, canary strips it
        self.assertIn('ws ::= ( " " | ', g)                               # free whitespace, zero or more
        self.assertIn('answer-root ::= "a" | "b"', g)
        self.assertIn('reason ::= ', g)                            # free text that never spells the closer
        self.assertIn('"<" reason-1-0', g)                     # the closer chain
        self.assertIn('"`" reason-0-1', g)                     # the fenced-answer habit is refused too


if __name__ == "__main__":
    unittest.main()
