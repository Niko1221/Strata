import io

SRC = "serve/test_grammar.py"
ANCHOR = "class GrammarConverter"
BLOCK = '''TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
          "parameters": {"type": "object", "additionalProperties": False, "required": ["city"],
                         "properties": {"city": {"type": "string"}, "unit": {"type": "string"}}}}}]
CALL_TEXT = "<tool_call>\\n<function=get_weather>\\n<parameter=city>\\nParis\\n</parameter>\\n</function>\\n</tool_call>"


class ToolGrammar(GrammarBase):
    """#3862: on a grammar-capable engine a tool call rides the sampler - the grammar spells the
    template serialization - and the prompt path stays for MCP, tool_choice shapes the grammar
    would change, and engines that never said grammar=1."""

    @unittest.skipUnless(HAVE_CONVERTER, "grammar_schema CLI is not built")
    def test_required_runs_the_call_grammar_and_the_parser_still_reads_the_reply(self):
        engine = self.capable(script=CALL_TEXT)
        code, reply = self.chat(tools=TOOLS, tool_choice="required")
        self.assertEqual(code, 200, reply)
        self.assertIn('"\\u003ctool_call\\u003e\\\\n<function=get_weather>\\\\n"', engine.grammars[-1])
        self.assertIn("answer ::= call-get-weather-0", engine.grammars[-1])      # required: prose is not allowed
        self.assertEqual(engine.last_sampling.get("grammar"), grammar_api.grammar_sha16(engine.grammars[-1]))
        prompt = self.tok.decode(engine.last_prompt)
        self.assertTrue(prompt.endswith("</think>\\n\\n"))                        # the template tail; no forced prefix
        self.assertEqual(prompt.count("<tool_call>\\n<function=get_weather>\\n"), 1)   # the tool list's example, once
        message = reply["choices"][0]["message"]
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"]), {"city": "Paris"})
        self.assertEqual(reply["choices"][0]["finish_reason"], "tool_calls")

    @unittest.skipUnless(HAVE_CONVERTER, "grammar_schema CLI is not built")
    def test_auto_lets_the_model_answer_in_prose(self):
        engine = self.capable(script="It is 21 degrees in Paris.")
        code, reply = self.chat(tools=TOOLS, tool_choice="auto")
        self.assertEqual(code, 200, reply)
        self.assertIn('answer ::= prose | calls | prose "\\\\n\\\\n" calls', engine.grammars[-1])
        self.assertEqual(engine.last_sampling.get("grammar"), grammar_api.grammar_sha16(engine.grammars[-1]))

    @unittest.skipUnless(HAVE_CONVERTER, "grammar_schema CLI is not built")
    def test_named_pins_one_function(self):
        engine = self.capable(script=CALL_TEXT)
        code, reply = self.chat(tools=TOOLS, tool_choice={"type": "function", "function": {"name": "get_weather"}})
        self.assertEqual(code, 200, reply)
        self.assertIn("answer ::= call-get-weather-0", engine.grammars[-1])

    def test_named_tool_choice_shape_the_grammar_refuses_stays_on_the_prompt_path(self):
        engine = self.capable(script="It is 21 degrees in Paris.")
        code, reply = self.chat(tools=TOOLS, tool_choice={"type": "function", "function": {"name": "no_such"}})
        self.assertEqual(code, 200, reply)
        self.assertEqual(engine.grammars, [])                                   # the old degrade-to-auto path

    def test_no_tools_no_grammar(self):
        engine = self.capable()
        self.assertEqual(self.chat(tool_choice="required")[0], 200)             # required without tools
        self.assertEqual(engine.grammars, [])
        self.assertEqual(self.chat(tools=TOOLS, tool_choice="none")[0], 200)    # none: tools are not offered
        self.assertEqual(engine.grammars, [])

    def test_mcp_stays_on_the_prompt_path_and_says_so(self):
        # a grammar-capable engine, the web app's MCP opt-in: the hosted tools' schemas are not validated
        # here, so the prompt path takes the turn and the reply declares the method
        self.capable()
        code, reply = self.chat(tools=TOOLS, strata_mcp=True)
        self.assertEqual(code, 200, reply)
        self.assertEqual(self.engine.grammars, [])
        self.assertEqual(reply["strata"]["structured_output"]["method"], "prompt_and_validate")

    def test_a_capabilityless_engine_keeps_the_tool_prompt_path(self):
        self.capable(script=CALL_TEXT)
        self.engine.info = {}                                                   # never said grammar=1
        code, reply = self.chat(tools=TOOLS, tool_choice="required")
        self.assertEqual(code, 200, reply)
        self.assertEqual(self.engine.grammars, [])
        self.assertNotIn("grammar", self.engine.last_sampling)
        self.assertEqual(reply["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "get_weather")

    @unittest.skipUnless(HAVE_CONVERTER, "grammar_schema CLI is not built")
    def test_the_responses_route_spells_the_call_too(self):
        engine = self.capable(script=CALL_TEXT)
        code, reply = self.request("/v1/responses", {"input": "Plan the shot.", "tools": TOOLS,
                                                     "tool_choice": "required", "reasoning_effort": "none"})
        self.assertEqual(code, 200, reply)
        self.assertIn("answer ::= call-get-weather-0", engine.grammars[-1])
        self.assertEqual(engine.last_sampling.get("grammar"), grammar_api.grammar_sha16(engine.grammars[-1]))
        self.assertIn("get_weather", json.dumps(reply))

    @unittest.skipUnless(HAVE_CONVERTER, "grammar_schema CLI is not built")
    def test_the_anthropic_route_spells_the_call_too(self):
        engine = self.capable(script=CALL_TEXT)
        code, reply = self.request("/v1/messages", {"model": "m", "max_tokens": 256,
                                                    "messages": [{"role": "user", "content": "Weather in Paris?"}],
                                                    "tools": [{"name": "get_weather", "description": "Weather",
                                                               "input_schema": TOOLS[0]["function"]["parameters"]}],
                                                    "tool_choice": {"type": "any"}})
        self.assertEqual(code, 200, reply)
        self.assertIn("answer ::= call-get-weather-0", engine.grammars[-1])


'''

s = io.open(SRC, encoding="utf-8").read()
assert s.count(ANCHOR) == 1, s.count(ANCHOR)
s = s.replace(ANCHOR, BLOCK + ANCHOR, 1)
io.open(SRC, "w", encoding="utf-8").write(s)
print("inserted", s.count("class ToolGrammar"), "class, syntax:")
import ast
ast.parse(s)
print("ok")
