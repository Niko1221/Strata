"""serve/test_glm.py - the server's GLM-5.2 / GLM-5.3 side: the tool-call parser, the forced call, the chat template.

No model and no engine: the parser is fed text, the template is rendered.  The template's byte-level agreement with
colibri's GLM-5.2 renderer (itself checked against zai-org's chat_template.jinja) was measured once with colibri's
openai_server.render_chat on ten conversations (docs/GLM53.md); here its shape is pinned down.
"""
import json
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate, GlmOutputParser, glm_call_end, glm_forced_call, parse_glm_tool_call

ROOT = Path(__file__).resolve().parents[1]
TOOLS = [{"name": "get_weather", "description": "Weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                                                          "note": {"type": "string"}}}}]


def run(text, thinking=False, tools=TOOLS, step=None):
    """Feed `text` whole (step None) or in pieces of `step` characters; -> (events, joined tool_args per call id)."""
    p = GlmOutputParser(thinking=thinking, tools=tools, stream_tools=True)
    events = []
    pieces = [text] if step is None else [text[i:i + step] for i in range(0, len(text), step)]
    for piece in pieces:
        events += p.feed(piece)
    events += p.finish()
    streamed = {}
    for e in events:
        if e.kind == "tool_args":
            streamed[e.call.id] = streamed.get(e.call.id, "") + e.text
    return events, streamed


def summary(events):
    out = []
    for e in events:
        if e.kind in ("content", "reasoning"):
            if out and out[-1][0] == e.kind:
                out[-1] = (e.kind, out[-1][1] + e.text)
            else:
                out.append((e.kind, e.text))
        elif e.kind == "tool_call":
            out.append(("tool_call", e.call.name, json.dumps(e.call.arguments, sort_keys=True)))
    return out


class GlmToolCalls(unittest.TestCase):
    CALL = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>" \
           "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>"

    def test_a_call_in_the_answer(self):
        events, _ = run("Let me check." + self.CALL)
        self.assertEqual(summary(events), [("content", "Let me check."),
                                           ("tool_call", "get_weather", json.dumps({"city": "Paris", "days": 3}, sort_keys=True))])

    def test_streamed_in_any_pieces_equals_whole(self):
        whole = summary(run("Sure." + self.CALL + "\n\nDone.")[0])
        for step in (1, 2, 3, 7, 13):
            self.assertEqual(summary(run("Sure." + self.CALL + "\n\nDone.", step=step)[0]), whole, step)

    def test_streamed_arguments_are_the_call_as_json(self):
        events, streamed = run(self.CALL, step=1)
        call = [e.call for e in events if e.kind == "tool_call"][0]
        self.assertEqual(json.loads(streamed[call.id]), call.arguments)
        self.assertTrue(any(e.kind == "tool_start" and e.call.name == "get_weather" for e in events))

    def test_schema_types(self):
        call = parse_glm_tool_call("get_weather<arg_key>city</arg_key><arg_value>123</arg_value>"
                                   "<arg_key>days</arg_key><arg_value>4</arg_value>", TOOLS[0])
        self.assertEqual(call.arguments, {"city": "123", "days": 4})   # a string parameter stays a string
        call = parse_glm_tool_call("x<arg_key>a</arg_key><arg_value>{\"k\": [1, 2]}</arg_value>")
        self.assertEqual(call.arguments, {"a": {"k": [1, 2]}})         # no schema: JSON when it parses

    def test_a_value_may_contain_the_closing_tags(self):
        body = ("get_weather<arg_key>note</arg_key><arg_value>write </arg_value> and </tool_call> literally"
                "</arg_value><arg_key>city</arg_key><arg_value>Rome</arg_value>")
        end = glm_call_end(body + "</tool_call>tail")
        self.assertEqual(end, len(body))
        self.assertEqual(parse_glm_tool_call(body, TOOLS[0]).arguments,
                         {"note": "write </arg_value> and </tool_call> literally", "city": "Rome"})

    def test_whitespace_between_parts_is_allowed(self):
        events, _ = run("<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Oslo</arg_value>\n</tool_call>")
        self.assertEqual(summary(events)[-1][1:], ("get_weather", json.dumps({"city": "Oslo"})))

    def test_prose_naming_the_tag_is_content(self):
        events, _ = run("Use a <tool_call> block when you need one.")
        self.assertEqual(summary(events), [("content", "Use a <tool_call> block when you need one.")])

    def test_a_call_at_the_end_of_the_reasoning_is_a_call(self):
        events, _ = run("I should look it up.\n" + self.CALL, thinking=True)
        kinds = summary(events)
        self.assertEqual(kinds[0], ("reasoning", "I should look it up.\n"))
        self.assertEqual(kinds[-1][0], "tool_call")

    def test_the_answer_after_think(self):
        events, _ = run("thinking...</think>\n\nParis.", thinking=True)
        self.assertEqual(summary(events), [("reasoning", "thinking..."), ("content", "Paris.")])

    def test_forced_call(self):
        self.assertEqual(glm_forced_call("required", TOOLS), "<tool_call>get_weather")
        two = TOOLS + [{"name": "other"}]
        self.assertEqual(glm_forced_call("required", two), "<tool_call>")
        self.assertEqual(glm_forced_call({"type": "function", "function": {"name": "other"}}, two), "<tool_call>other")
        self.assertIsNone(glm_forced_call("auto", two))


class GlmTemplate(unittest.TestCase):
    tpl = ChatTemplate(ROOT / "serve/glm/chat_template.jinja")

    def test_prefix_turns_and_generation_prompt(self):
        off = self.tpl.render([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
                              enable_thinking=False)
        self.assertEqual(off, "[gMASK]<sop><|system|>Be brief.<|user|>Hi<|assistant|><think></think>")
        on = self.tpl.render([{"role": "user", "content": [{"type": "text", "text": "Hi"}]}])
        self.assertEqual(on, "[gMASK]<sop><|system|>Reasoning Effort: High<|user|>Hi<|assistant|><think>")
        low = self.tpl.render([{"role": "user", "content": "Hi"}], reasoning_effort="low")
        self.assertIn("Reasoning Effort: Low", low)

    def test_tools_calls_and_results(self):
        msgs = [{"role": "user", "content": "weather?"},
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_weather",
                                                                                 "arguments": {"city": "Paris", "days": 2}}}]},
                {"role": "tool", "content": "{\"t\": 21}"}, {"role": "tool", "content": "b"}]
        out = self.tpl.render(msgs, tools=TOOLS, enable_thinking=False)
        self.assertIn("<tools>\n" + json.dumps(TOOLS[0], ensure_ascii=False) + "\n</tools>", out)
        self.assertIn("<|assistant|><think></think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris"
                      "</arg_value><arg_key>days</arg_key><arg_value>2</arg_value></tool_call>", out)
        self.assertIn("<|observation|><tool_response>{\"t\": 21}</tool_response><tool_response>b</tool_response>", out)
        self.assertEqual(out.count("<|observation|>"), 1)          # one per run of tool messages


class GlmCapabilities(unittest.TestCase):
    def test_config_rejects_unsupported_engine_commands(self):
        from serve.server import engine_args
        base = {"family": "glm", "args": ["--model", "model", "--gpu", "0"]}
        self.assertEqual(engine_args(base), base["args"])
        for config in ({"parallel": 2}, {"vision": {"exe": "encoder"}}, {"vram_elastic": True},
                       {"gpu": [0, 1]}, {"effort_position": "end"}, {"args": ["--batch", "2"]}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                engine_args({**base, **config})

    def test_unsupported_operations_do_not_contact_engine(self):
        from unittest.mock import Mock
        from serve.server import Service
        svc = object.__new__(Service)
        svc.family = "glm"
        svc.engine = Mock()
        status, body = svc.slot_action("0", "save", "test.bin")
        self.assertEqual(status, 501)
        with self.assertRaisesRegex(ValueError, "cannot resize"):
            svc.vram(1000)
        self.assertEqual(svc.engine.mock_calls, [])

    def test_settings_do_not_offer_qwen_effort_placement(self):
        from serve import runconfig
        cfg = {"family": "glm", "args": []}
        self.assertNotIn("effort_position", [k["key"] for k in runconfig.view(cfg, "glm.json")["keys"]])
        with self.assertRaises(ValueError): runconfig.apply(cfg, {"effort_position": "end"})


if __name__ == "__main__":
    unittest.main()
