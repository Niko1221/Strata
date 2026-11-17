"""The tool-call grammar builder (issue #3862), without a GPU: the prose guard's boundary behaviour,
the template's literals spelled exactly, the escaping, and the MCP routing fallback.

The named-grammar check is structural: Python has no grammar core, so the assertions read the GBNF text
- the same literals the chat template writes (rendered below for one call) must appear in the rules.
The exclusion automaton itself is not spelled here: it comes from the same grammar_schema CLI the server
uses (--ac-grammar), so these tests exercise the shipped emitter rather than a second copy of it.
"""
import unittest

from serve.frontend import ChatTemplate
from serve.grammar_schema import exclusion_grammar
from serve.grammar_tools import build_tool_grammar, exclude_seq, gbnf_escape, tool_grammar_choice, with_think_prologue
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


def _converter_dir():
    root = Path(__file__).resolve().parents[1]
    for name in ("build-release", "build"):
        if (root / name / "grammar_schema.exe").exists():
            return root / name
    return None


EXE_DIR = _converter_dir()


def build_exclusion(prefix, seqs):
    """The shipped Aho-Corasick emitter, or a skip when the CLI has not been built."""
    if EXE_DIR is None:
        raise unittest.SkipTest("grammar_schema.exe not built: no exclusion constructor to test")
    return exclusion_grammar(EXE_DIR, prefix, seqs)


def convert(sub, raw_string=False, forbidden_sequence=None, document=None):
    # `document` is what a $ref would resolve against; these fakes spell one node and ignore it
    if sub.get("type") == "boolean":
        return 'root ::= boolean\nboolean ::= "true" | "false"\n'
    if sub.get("type") == "string":
        rules = []
        text = exclude_seq(forbidden_sequence or "</parameter>", "text", rules, build_exclusion)
        return "\n".join([f"root ::= {text}", *rules]) + "\n"
    raise AssertionError(f"unexpected subschema {sub}")


def _advance(seq: str, state: int, ch: str):
    candidate = seq[:state] + ch
    if candidate.endswith(seq):
        return None
    for size in range(min(len(seq) - 1, len(candidate)), 0, -1):
        if candidate.endswith(seq[:size]):
            return size
    return 0


def chain_accept(seq: str, text: str) -> bool:
    """The ORACLE: what the exclusion language must be, stated independently of the shipped emitter.

    Nothing in serve/ spells an automaton any more (the exclusion grammar comes from the converter's
    Aho-Corasick port), so this exists to say which strings are legal.  Conformance of the emitted
    grammar is checked where a real parser is available: tools/grammar_strict_test.cpp runs the engine's
    own GBNF parser over these same cases."""
    if not seq:
        raise ValueError("an excluded sequence cannot be empty")
    state = 0
    for ch in text:
        state = _advance(seq, state, ch)
        if state is None:
            return False
    return True


def build(tool_choice="auto", name=None, thinking=False):
    return build_tool_grammar(TOOLS, tool_choice, name, thinking, convert, build_exclusion)


class Prose(unittest.TestCase):
    def test_the_guard_language_the_emitter_must_match(self):
        for ok in ("", "<", "<too", "<tool_cal", "a ```python fence``` is prose", "</system>\n<|im_start|>"):
            self.assertTrue(chain_accept("<tool_call>", ok), ok)
        for bad in ("<tool_call>", "x<tool_call>", "<tool_call>\n<function=a>"):
            self.assertFalse(chain_accept("<tool_call>", bad), bad)
        self.assertFalse(chain_accept("</parameter>", "safe <</parameter>"))

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
        self.assertIn("call-get-weather-0-param-0 ", call)   # the required parameter, in schema order
        self.assertIn('"<parameter=city>\\n"', g)
        self.assertIn('"\\n</parameter>\\n"', g)
        self.assertIn('"</function>\\n</tool_call>"', call)
        self.assertIn("( call-get-weather-0-param-1 )?", call)   # "unit" is not required
        # the other tool's rules are built (they ride the shared calls list) but nothing reaches them:
        # the answer is the named call, and only the "required" roots open calls-one
        self.assertEqual(answer, "answer ::= call-get-weather-0")
        self.assertIn("call-get-weather-0 | call-send-sms-1", [l for l in lines if l.startswith("calls-one ::=")][0])

    def test_the_string_value_rule_refuses_the_parameter_end(self):
        g = build("named", name="get_weather")
        self.assertIn('"\\n</parameter>\\n"', g)                # the parameter tail the value must not write
        # The value's start state is accepting (it opens with an empty alternative) and takes '<' only as a
        # transition; every other character loops back.  Conformance of the rest - that no derivation
        # completes "</parameter>" - is checked against the engine's own parser in grammar_strict_test.
        start = [l for l in g.splitlines() if l.startswith("call-get-weather-0-param-0-text ::=")][0]
        self.assertTrue(start.endswith("::= | [<] call-get-weather-0-param-0-text-01 | [^<] call-get-weather-0-param-0-text"),
                        start)
        self.assertIn("[^<>] call-get-weather-0-param-0-text", g)   # the last state refuses '>'

    def test_a_non_string_parameter_inlines_the_converter_under_a_namespace(self):
        g = build("named", name="send_sms")
        self.assertIn("call-send-sms-1-param-2-boolean ::=", g)   # renamed, literals kept
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


class Framing(unittest.TestCase):
    """The grammar writes the template's markers around a tool's name and its parameter keys, and
    serve/frontend reads them back by cutting at the first '>'.  A name or key holding '>' makes the two
    disagree, so the builder runs the exact text it demands through the real parser and refuses a tool
    whose call would arrive as a different one (measured: `ev>il` read back as `ev`)."""

    def tool(self, name, keys):
        return {"name": name, "parameters": {"type": "object", "additionalProperties": False,
                                             "required": list(keys),
                                             "properties": {k: {"type": "string"} for k in keys}}}

    def test_a_name_with_a_closing_angle_is_refused(self):
        with self.assertRaisesRegex(ValueError, "would be read back as 'ev'"):
            build_tool_grammar([self.tool("ev>il", ["a"])], "required", None, False, convert, build_exclusion)

    def test_a_parameter_key_with_a_closing_angle_is_refused(self):
        with self.assertRaisesRegex(ValueError, "parameter 'a>b'"):
            build_tool_grammar([self.tool("ok", ["a>b"])], "required", None, False, convert, build_exclusion)

    def test_ordinary_and_angle_free_names_still_build(self):
        for name, keys in (("get_weather", ["city"]), ("ok", ["a<b"]), ("send.sms", ["to", "body"])):
            text = build_tool_grammar([self.tool(name, keys)], "required", None, False, convert, build_exclusion)
            self.assertIn("root ::= answer", text)


class References(unittest.TestCase):
    """A JSON Pointer is rooted at the DOCUMENT, but the builder spells one parameter at a time.  So a `$ref`
    naming a sibling parameter (`#/properties/func/properties/vpc`) has no target inside the subschema, and
    the request failed with "does not resolve inside this schema" for a schema written correctly (measured
    in a harness turn).  The parameters document rides along with every per-parameter conversion for that."""

    PARAMS = {
        "type": "object",
        "additionalProperties": False,
        "required": ["func"],
        "properties": {
            "func": {"type": "object", "additionalProperties": False, "required": ["vpc"],
                     "properties": {"vpc": {"type": "object", "additionalProperties": False,
                                            "required": ["id"],
                                            "properties": {"id": {"type": "string"}}}}},
            "instance": {"$ref": "#/properties/func/properties/vpc"},
        },
    }

    def test_a_ref_to_a_sibling_parameter_resolves(self):
        from serve import grammar_schema as conv

        seen = []

        def convert_doc(sub, raw_string=False, forbidden_sequence=None, document=None):
            seen.append(document)
            return conv.schema_to_grammar(_converter_dir(), sub, tool_value=raw_string,
                                          forbidden_sequence=forbidden_sequence, document=document)

        text = build_tool_grammar([{"name": "plan", "parameters": self.PARAMS}], "required", None, False,
                                  convert_doc, build_exclusion)
        # the target's own property rule: it can only appear by resolving the sibling reference
        self.assertIn("id-kv", text)
        self.assertTrue(all(doc is self.PARAMS for doc in seen), "every parameter got the document")

    def test_the_same_reference_without_the_document_is_an_error(self):
        # the control: what the server did before the document was passed.  It must keep failing loudly
        # instead of quietly spelling an unconstrained value.
        from serve import grammar_schema as conv
        with self.assertRaisesRegex(ValueError, "does not resolve"):
            conv.schema_to_grammar(_converter_dir(), self.PARAMS["properties"]["instance"],
                                   tool_value=True, forbidden_sequence="</parameter>")


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
        g = with_think_prologue('root ::= "a" | "b"\n', build_exclusion)
        self.assertIn('root ::= reason "</think>\\n" answer', g)
        self.assertIn("answer ::= ws answer-root", g)
        self.assertNotIn("fenced", g)
        self.assertIn('ws ::= ( " " | ', g)                               # bounded whitespace
        self.assertIn('answer-root ::= "a" | "b"', g)
        self.assertIn('reason ::= ', g)                            # free text that never spells the closer
        # the Aho-Corasick state chain: '<' advances toward "</think>", a backtick toward the fence, and
        # every other character loops back to the start state
        self.assertIn('[<] reason-01', g)
        self.assertIn('[`] reason-09', g)
        # the state that a completed closer would reach has no production at all, and the state just before
        # it refuses the character that would complete it ([^<>] for "</think>", [^<`n] for "```json")
        self.assertNotIn("reason-08 ::= ", g)
        self.assertIn("[^<>`] reason", g)
        self.assertIn("[^<`n] reason", g)


if __name__ == "__main__":
    unittest.main()
