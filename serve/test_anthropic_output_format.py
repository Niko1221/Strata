"""output_config.format on /v1/messages: the schema reaches the prompt and the answer is checked, without a GPU or model pack.

Anthropic's page: "The request carries the schema in `output_config.format` with `type: "json_schema"`."
The invalid replies here are not JSON objects at all, so they fail with or without the jsonschema package.

    python -m unittest serve.test_anthropic_output_format -v
"""
import contextlib
import importlib.util
import io
import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve import dsml
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False}
FORMAT = {"format": {"type": "json_schema", "schema": SCHEMA}}
TOOLS = [{"name": "get", "description": "w",
          "input_schema": {"type": "object", "properties": {"c": {"type": "string"}}, "required": ["c"]}}]
TOOL_CALL = "</think>\n\n<tool_call>\n<function=get>\n<parameter=c>x</parameter>\n</function>\n</tool_call>"


def events_of(raw):
    pairs = []
    for block in raw.split("\n\n"):
        lines = block.split("\n")
        if lines[0].startswith("event: "):
            pairs.append((lines[0][7:], json.loads(lines[1][6:])))
    return pairs


def stream_text(events):
    return "".join(e["delta"]["text"] for name, e in events
                   if name == "content_block_delta" and e["delta"]["type"] == "text_delta")


class OutputFormat(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, "ok", max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, script=None, path="/v1/messages", **fields):
        if script is not None:
            self.engine = self.svc.engine = MockEngine(self.tok, script, max_context=16384)
        body = {"model": "x", "max_tokens": 200, "messages": [{"role": "user", "content": "hi"}], **fields}
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = self.raw = response.read().decode()
            return response.status, events_of(data) if fields.get("stream") else json.loads(data)

    def prompt(self):
        return self.tok.decode(self.engine.last_prompt)

    def test_schema_reaches_the_prompt_and_the_answer_is_canonical(self):
        for script in ('</think>\n\n{"a": 1}', '</think>\n\n```json\n{"a": 1}\n```', '</think>\n\nHere: {"a": 1} - done'):
            with self.subTest(script=script):
                code, reply = self.post(script, output_config=FORMAT)
                self.assertEqual(code, 200, reply)
                self.assertEqual(reply["content"], [{"type": "text", "text": '{"a":1}'}])
                self.assertEqual(reply["stop_reason"], "end_turn")
                self.assertNotIn("strata_structured_output_error", reply)
                self.assertIn("OUTPUT FORMAT REQUIREMENT", self.prompt())
                self.assertIn('"additionalProperties":false', self.prompt())
                code, events = self.post(script, output_config=FORMAT, stream=True)
                self.assertEqual(code, 200)
                self.assertEqual(stream_text(events), '{"a":1}')
                self.assertEqual([name for name, _ in events],
                                 ["message_start", "content_block_start", "content_block_delta", "content_block_stop",
                                  "message_delta", "message_stop"])

    def test_system_prompt_is_kept_beside_the_schema(self):
        self.post('</think>\n\n{"a": 1}', output_config=FORMAT, system="Be terse today.")
        self.assertIn("Be terse today.", self.prompt())
        self.assertIn("OUTPUT FORMAT REQUIREMENT", self.prompt())

    def test_thinking_stays_in_its_own_block(self):
        code, reply = self.post('Consider.</think>\n\n{"a": 1}', output_config=FORMAT, thinking={"type": "enabled", "budget_tokens": 100})
        self.assertEqual(code, 200, reply)
        self.assertEqual([b["type"] for b in reply["content"]], ["thinking", "text"])
        self.assertEqual(reply["content"][1]["text"], '{"a":1}')

    def test_a_failed_check_is_a_200_with_the_flag_not_a_5xx(self):
        # a 502 here made Claude Code's prompt hook ask again, a whole generation each time, until the hook timed out
        for script, text in (("</think>\n\nnot json at all", "not json at all"), ("</think>\n\n[1]", "[1]")):
            with self.subTest(script=script):
                code, reply = self.post(script, output_config=FORMAT)
                self.assertEqual(code, 200, reply)
                self.assertEqual(reply["content"], [{"type": "text", "text": text}])
                self.assertEqual(reply["stop_reason"], "end_turn")
                self.assertTrue(reply["strata_structured_output_error"])
                code, events = self.post(script, output_config=FORMAT, stream=True)
                self.assertEqual(code, 200)
                self.assertEqual(stream_text(events), text)
                self.assertNotIn("error", [name for name, _ in events])
                delta = next(e for name, e in events if name == "message_delta")
                self.assertTrue(delta["strata_structured_output_error"])
                self.assertEqual(events[-1][0], "message_stop")

    def test_an_object_against_the_schema_is_flagged_and_its_values_stay_out_of_the_log(self):
        class Rejects:              # jsonschema's message quotes the value; the package is optional, so a stand-in here
            def iter_errors(self, value):
                yield mock.Mock(absolute_path=("a",), message="'secret-value' is not of type 'integer'")

        window = io.StringIO()
        with mock.patch("serve.server.prepare_format", lambda spec, messages, with_tools: (messages, Rejects())), \
                contextlib.redirect_stdout(window):
            code, reply = self.post('</think>\n\n{"a": "secret-value"}', output_config=FORMAT)
        self.assertEqual(code, 200, reply)
        self.assertEqual(reply["content"], [{"type": "text", "text": '{"a": "secret-value"}'}])
        self.assertIn("/a", reply["strata_structured_output_error"])
        self.assertIn("output_config.format: model output failed the JSON Schema at /a\n", window.getvalue())
        self.assertNotIn("secret-value", window.getvalue())

    @unittest.skipUnless(importlib.util.find_spec("jsonschema"), "needs the optional jsonschema package")
    def test_real_schema_check(self):
        for script in ('{"a": "x"}', '{"a": 1, "b": 2}', "{}"):
            with self.subTest(script=script):
                code, reply = self.post("</think>\n\n" + script, output_config=FORMAT)
                self.assertEqual(code, 200, reply)
                self.assertEqual(reply["content"], [{"type": "text", "text": script}])
                self.assertIn("failed the JSON Schema", reply["strata_structured_output_error"])

    def test_keep_alives_go_out_while_the_answer_is_held(self):
        # one piece each 0.1 s with no ping from the engine: the hold itself must keep the stream alive
        self.engine = self.svc.engine = MockEngine(self.tok, '</think>\n\n{"a":   1}' + " " * 8, max_context=16384, delay_s=0.1)
        code, events = self.post(output_config=FORMAT, stream=True)
        self.assertEqual(code, 200)
        self.assertEqual(stream_text(events), '{"a":1}')
        self.assertIn(": keep-alive", self.raw.split("event: message_start")[0])

    def test_a_turn_with_no_think_close_has_no_second_copy_as_thinking(self):
        # DeepSeek's parser hands the reasoning over as the answer (serve/dsml.py); the canonical text no longer matches it
        self.svc.parser_cls = dsml.DsmlOutputParser
        for script, compact in (('{"a": 1}', False), ('{"a":1}', True)):
            with self.subTest(script=script):
                code, reply = self.post(script, output_config=FORMAT)
                self.assertEqual(code, 200, reply)
                self.assertEqual(reply["content"], [{"type": "text", "text": '{"a":1}'}])
                code, events = self.post(script, output_config=FORMAT, stream=True)
                self.assertEqual(code, 200)
                self.assertEqual(stream_text(events), '{"a":1}')
                self.assertEqual({e["index"] for name, e in events if "index" in e}, {0})
                self.assertEqual([e["content_block"]["type"] for name, e in events if name == "content_block_start"], ["text"])

    def test_max_tokens_goes_back_as_it_is(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                code, reply = self.post('</think>\n\n{"a": 111111111111}', output_config=FORMAT, max_tokens=18, stream=stream)
                self.assertEqual(code, 200, reply)
                if stream:
                    delta = next(e for name, e in reply if name == "message_delta")
                    stop, text = delta["delta"]["stop_reason"], stream_text(reply)
                else:
                    delta, stop, text = reply, reply["stop_reason"], reply["content"][0]["text"]
                self.assertEqual(stop, "max_tokens")
                self.assertTrue(text.startswith('{"a": 1'), text)      # not canonical: the model's own, cut short
                self.assertNotIn("strata_structured_output_error", delta)

    def test_a_tool_call_goes_back_as_it_is(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                code, reply = self.post(TOOL_CALL, output_config=FORMAT, tools=TOOLS, max_tokens=500, stream=stream)
                self.assertEqual(code, 200, reply)
                self.assertIn("applies only to your final answer", self.prompt())
                if stream:
                    delta = next(e for name, e in reply if name == "message_delta")
                    self.assertEqual(delta["delta"]["stop_reason"], "tool_use")
                    self.assertTrue(any(name == "content_block_start" and e["content_block"]["type"] == "tool_use"
                                        for name, e in reply))
                else:
                    delta = reply
                    self.assertEqual(reply["stop_reason"], "tool_use")
                    self.assertEqual([b["input"] for b in reply["content"] if b["type"] == "tool_use"], [{"c": "x"}])
                self.assertNotIn("strata_structured_output_error", delta)

    def test_a_bad_format_is_a_400_naming_the_field(self):
        for bad in ({"type": "json_object"}, {"type": "json_schema"}, {"type": "json_schema", "schema": {"type": "array"}},
                    "json_schema"):
            for path in ("/v1/messages", "/v1/messages/count_tokens"):
                with self.subTest(bad=bad, path=path):
                    code, reply = self.post(path=path, output_config={"format": bad})
                    self.assertEqual(code, 400, reply)
                    self.assertEqual(reply["error"]["type"], "invalid_request_error")
                    self.assertIn("output_config.format", reply["error"]["message"])
                    self.assertNotIn("response_format", reply["error"]["message"])

    def test_no_format_changes_nothing(self):
        for config in ({"format": None}, {"effort": "medium"}, None):
            with self.subTest(config=config):
                fields = {} if config is None else {"output_config": config}
                code, reply = self.post("</think>\n\nnot json at all", **fields)
                self.assertEqual(code, 200, reply)
                self.assertEqual(reply["content"], [{"type": "text", "text": "not json at all"}])
                self.assertNotIn("strata_structured_output_error", reply)
                self.assertNotIn("OUTPUT FORMAT REQUIREMENT", self.prompt())

    def test_count_tokens_counts_the_prompt_the_request_reads(self):
        code, count = self.post(path="/v1/messages/count_tokens", output_config=FORMAT)
        self.assertEqual(code, 200, count)
        self.post('</think>\n\n{"a": 1}', output_config=FORMAT)
        self.assertEqual(count["input_tokens"], len(self.engine.last_prompt))


if __name__ == "__main__":
    unittest.main()
