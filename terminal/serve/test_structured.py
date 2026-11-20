"""Exercise the JSON contract through HTTP, including H3's strict Story shape."""
import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve import structured
from serve.server import ByteTokenizer, MockEngine, Service, serve

try:
    import jsonschema  # noqa: F401
    HAVE_JSONSCHEMA = True
except ImportError:                  # optional: json_schema answers are then only checked to be one object
    HAVE_JSONSCHEMA = False


STORY = {"title": "Photobooth", "sheet": "A single continuous shot.", "cast": [],
         "boxes": [{"box": 1, "duration_ms": 15000, "new_shot": True, "beat": "Hailee changes poses.",
                    "on_screen": ["Hailee"], "references": [], "ends": "She smiles."}], "suggestions": []}


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


TEXT = {"type": "string"}
STORY_SCHEMA = object_schema({
    "title": TEXT, "sheet": TEXT, "cast": {"type": "array", "items": object_schema({"name": TEXT,
        "references": {"type": "array", "items": TEXT, "maxItems": 0}})},
    "boxes": {"type": "array", "minItems": 1, "maxItems": 1, "items": object_schema({
        "box": {"type": "integer", "minimum": 1, "maximum": 1},
        "duration_ms": {"type": "integer", "minimum": 1, "maximum": 15000}, "new_shot": {"type": "boolean"},
        "beat": TEXT, "on_screen": {"type": "array", "items": TEXT},
        "references": {"type": "array", "items": TEXT, "maxItems": 0}, "ends": TEXT})},
    "suggestions": {"type": "array", "items": object_schema({"what": TEXT, "why": TEXT,
        "boxes": {"type": "array", "items": {"type": "integer"}}, "description": TEXT})}})
FORMAT = {"type": "json_schema", "json_schema": {"name": "h3_story", "strict": True, "schema": STORY_SCHEMA}}
EXEC = [{"type": "function", "function": {"name": "exec_command", "description": "Runs a command.",
         "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}]

# a tool the model may call, and its call in the chat template's form: with tools offered, a turn may end in a call
TOOLS = [{"type": "function", "function": {"name": "exec_command",
          "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}]
TOOL_CALL = ("Let me look.\n</think>\n\n<tool_call>\n<function=exec_command>\n<parameter=cmd>\ncat a.txt\n"
             "</parameter>\n</function>\n</tool_call>")


class Structured(unittest.TestCase):
    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read().decode()
            return response.status, data if "event-stream" in response.headers.get("Content-Type", "") else json.loads(data)

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, json.dumps(STORY), max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def chat(self, script=None, **kwargs):
        if script is not None:
            self.engine = self.svc.engine = MockEngine(self.tok, script, max_context=16384)
        req = {"messages": [{"role": "system", "content": "Write a film plan."},
                            {"role": "user", "content": "Photobooth. 15 seconds."}],
               "response_format": FORMAT, "reasoning_effort": "none", **kwargs}
        return self.request("/v1/chat/completions", req)

    def messages(self, script=None, **kwargs):
        """POST /v1/messages with the Anthropic native carrier output_config.format (what tool-eval-bench --format
        anthropic sends), the same schema as `chat`; `kwargs` adds or replaces request fields."""
        if script is not None:
            self.engine = self.svc.engine = MockEngine(self.tok, script, max_context=16384)
        req = {"model": "x", "max_tokens": 900, "thinking": {"type": "disabled"},
               "messages": [{"role": "user", "content": "Write a film plan. Photobooth. 15 seconds."}],
               "output_config": {"format": {"type": "json_schema", "schema": STORY_SCHEMA}}, **kwargs}
        return self.request("/v1/messages", req)

    ANTHROPIC_TOOLS = [{"name": "exec_command",
                        "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}},
                                         "required": ["cmd"]}}]

    def test_anthropic_tool_use_turn_passes_through_under_a_format(self):
        # tools and output_config.format together: the schema is for the final answer, a turn that ends in a
        # call is not an answer - the call goes out as written, with no validation error
        code, reply = self.messages(TOOL_CALL, tools=self.ANTHROPIC_TOOLS)
        self.assertEqual(code, 200, reply)
        self.assertEqual(reply["stop_reason"], "tool_use")
        calls = [b for b in reply["content"] if b["type"] == "tool_use"]
        self.assertEqual(calls[0]["name"], "exec_command")
        self.assertEqual(calls[0]["input"], {"cmd": "cat a.txt"})
        self.assertIn("Let me look.", "".join(b.get("text", "") for b in reply["content"] if b["type"] == "text"))

    def test_anthropic_end_turn_answer_is_canonical_json(self):
        # the answer's text is held while the schema is in force and leaves as the canonical JSON, one delta;
        # the directive reaches the prompt as on the chat route (the schema itself is what the model is told)
        code, reply = self.messages(json.dumps({**STORY, "suggestions": []}))
        self.assertEqual(code, 200, reply)
        self.assertEqual(reply["stop_reason"], "end_turn")
        text = "".join(b["text"] for b in reply["content"] if b["type"] == "text")
        self.assertEqual(text, json.dumps(STORY, ensure_ascii=False, separators=(",", ":")))
        self.assertIn("OUTPUT FORMAT REQUIREMENT", self.tok.decode(self.engine.last_prompt))
        self.assertIn('"duration_ms"', self.tok.decode(self.engine.last_prompt))
        _, raw = self.messages(json.dumps(STORY), stream=True)
        deltas = [json.loads(line[6:]) for line in raw.splitlines()
                  if line.startswith("data: {") and '"text_delta"' in line]
        self.assertEqual(len(deltas), 1)                       # the canonical JSON replaces the held deltas
        self.assertEqual(deltas[0]["delta"]["text"], json.dumps(STORY, ensure_ascii=False, separators=(",", ":")))

    def test_anthropic_non_json_answer_is_502_and_the_stream_sends_the_error_event(self):
        code, reply = self.messages("## Film Plan - Photobooth")
        self.assertEqual(code, 502, reply)
        self.assertEqual(reply["error"]["type"], "structured_output_failed")
        code, raw = self.messages("## Film Plan - Photobooth", stream=True)
        self.assertEqual(code, 200)                            # the stream started; the failure is an event
        self.assertIn("event: error", raw)
        self.assertIn("structured_output_failed", raw)
        self.assertNotIn("text_delta", raw)                    # no text block went out before the failure

    def test_anthropic_response_format_is_the_secondary_carrier(self):
        # a gateway client sends OpenAI's response_format; it is read when the request carries no output_config,
        # and an output_config answers for the format when both are there
        code, reply = self.messages(json.dumps(STORY), output_config=None, response_format=FORMAT)
        self.assertEqual(code, 200, reply)
        text = "".join(b["text"] for b in reply["content"] if b["type"] == "text")
        self.assertEqual(json.loads(text), STORY)
        # the same non-JSON answer: 502 while the response_format is in force, plain when an output_config
        # (carrying no format) stands in front of it
        self.assertEqual(self.messages("## Film Plan", output_config=None, response_format=FORMAT)[0], 502)
        code, reply = self.messages("## Film Plan", output_config={}, response_format=FORMAT)
        self.assertEqual(code, 200, reply)
        # an output_config that is there but carries no format (its text form, or only the thinking effort) asks
        # for nothing: a plain answer stays a plain answer
        code, reply = self.messages("Prose.", output_config={"format": {"type": "text"}})
        self.assertEqual(code, 200, reply)
        self.assertEqual(reply["content"][0]["text"], "Prose.")
        code, reply = self.messages("Prose.", output_config={"effort": "high"})
        self.assertEqual(code, 200, reply)

    def test_anthropic_malformed_format_is_400_before_the_model_is_loaded(self):
        with mock.patch.object(self.svc, "load") as load:
            for fmt in ([], {"type": "json_schema"}, {"type": "json_schema", "schema": "not an object"},
                        {"type": "unsupported"}):
                with self.subTest(fmt=fmt):
                    code, reply = self.messages(output_config={"format": fmt})
                    self.assertEqual(code, 400, reply)
                    self.assertEqual(reply["error"]["type"], "invalid_request_error")
            # a secondary response_format is malformed the same way
            self.assertEqual(self.messages(None, output_config=None, response_format={"type": "unsupported"})[0], 400)
            load.assert_not_called()

    def test_anthropic_without_a_format_is_unchanged(self):
        # no format, no validator: the text streams as it always did, deltas and all
        code, raw = self.messages("Prose answer.", output_config=None, stream=True)
        self.assertEqual(code, 200)
        self.assertIn("event: content_block_start", raw)
        self.assertIn("text_delta", raw)
        self.assertIn("event: message_stop", raw)

    def test_h3_schema_success_and_stream_with_reasoning(self):
        code, reply = self.chat()
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(reply["choices"][0]["message"]["content"]), STORY)
        prompt = self.tok.decode(self.engine.last_prompt)
        self.assertIn("OUTPUT FORMAT REQUIREMENT", prompt)
        self.assertIn('"duration_ms"', prompt)
        code, raw = self.chat("Consider the shot.</think>\n" + json.dumps(STORY), stream=True, reasoning_effort="low")
        self.assertEqual(code, 200)
        chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        self.assertEqual(json.loads(content), STORY)
        self.assertIn("reasoning_content", raw)
        self.assertIn("data: [DONE]", raw)

    @unittest.skipUnless(HAVE_JSONSCHEMA, "jsonschema is not installed")
    def test_invalid_generation_is_an_error_and_never_streams_content(self):
        for script in ("## Film Plan - Photobooth", '{"title":"Only a title"}',
                       json.dumps({**STORY, "extra": True}),
                       json.dumps({**STORY, "boxes": [{**STORY["boxes"][0], "duration_ms": "15000"}]})):
            with self.subTest(script=script):
                code, reply = self.chat(script)
                self.assertEqual(code, 502)
                self.assertEqual(reply["error"]["code"], "structured_output_failed")
                _, raw = self.chat(script, stream=True)
                self.assertIn('"structured_output_failed"', raw)
                self.assertNotIn('"delta"', raw)
                self.assertIn("data: [DONE]", raw)

    def test_object_mode_strict_json_and_truncation(self):
        fmt = {"type": "json_object"}
        self.assertEqual(self.chat('{"answer":4}', response_format=fmt)[0], 200)
        for script in ('[]', '{"x":NaN}', '{"x":1,"x":2}', '{"x":1e999}', 'no json at all', '{"x":'):
            self.assertEqual(self.chat(script, response_format=fmt)[0], 502)
        # #762: the JSON object is taken out of a code fence or the prose around it (the object itself stays strict)
        for script in ('```json\n{}\n```', 'Here it is: {"answer":4} - done'):
            self.assertEqual(self.chat(script, response_format=fmt)[0], 200, script)
        self.assertEqual(self.chat('{"answer":4}', response_format=fmt, max_tokens=5)[0], 502)
        # Plain text callers retain normal streaming, without a JSON validation gate.
        self.assertEqual(self.chat("Prose.", response_format={"type": "text"})[0], 200)

    def test_latex_escape_repair_is_opt_in_and_only_after_a_strict_failure(self):
        # #2893: \alpha is not a JSON escape. Off by default: still a 502. On: one retry after the strict parse failed.
        bad = r'{"answer":"the angle \alpha and \gamma_0"}'
        self.assertEqual(self.chat(bad, response_format={"type": "json_object"})[0], 502)
        self.assertEqual(self.chat(bad, response_format={"type": "json_object", "repair_latex_escapes": False})[0], 502)
        code, reply = self.chat(bad, response_format={"type": "json_object", "repair_latex_escapes": True})
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(reply["choices"][0]["message"]["content"]),
                         {"answer": "the angle \\alpha and \\gamma_0"})
        # streaming asks for the same check before any content leaves
        _, raw = self.chat(bad, stream=True, response_format={"type": "json_object", "repair_latex_escapes": True})
        self.assertIn(r"\\alpha", raw)
        # the repair does not hide what is wrong with the answer itself
        fmt = {"type": "json_object", "repair_latex_escapes": True}
        for script in ('[]', r'{"x":NaN}', r'{"a":"\alpha","a":2}', 'no json at all', r'{"x":"\alpha'):
            self.assertEqual(self.chat(script, response_format=fmt)[0], 502, script)
        self.assertEqual(self.chat(r'{"answer":"\alpha"}', response_format=fmt, max_tokens=5)[0], 502)

    def test_latex_escape_repair_helper(self):
        fix = structured.repair_latex_escapes
        self.assertEqual(fix(r'{"a":"\alpha \gamma_0"}'), r'{"a":"\\alpha \\gamma_0"}')
        for same in (r'{"a":"x\ny\t\"q\" \\ \/ \u00e9"}', '{"a": 1}', ''):
            self.assertEqual(fix(same), same)
        self.assertEqual(fix(r'{"a":"\u12"}'), r'{"a":"\\u12"}')        # \u without four hex digits
        self.assertEqual(fix(r'{"a":"\\alpha"}'), r'{"a":"\\alpha"}')    # an escaped backslash is skipped as a pair
        self.assertEqual(fix("\\"), "\\\\")                              # a lone trailing backslash
        # \b is a valid escape: untouched, so it parses (as a backspace), as it does without the option
        self.assertEqual(json.loads(fix(r'{"a":"\beta \alpha"}'))["a"], "\beta \\alpha")

    def test_the_repair_option_reaches_the_responses_format(self):
        from serve.responses import text_format
        self.assertEqual(text_format({"text": {"format": {"type": "json_object", "repair_latex_escapes": True}}}),
                         {"type": "json_object", "repair_latex_escapes": True})
        self.assertEqual(text_format({"text": {"format": {"type": "json_object"}}}), {"type": "json_object"})

    def test_rejected_schema_does_not_load_the_engine_or_retry(self):
        with mock.patch.object(self.svc, "load") as load:
            code, _ = self.chat(response_format={"type": "json_schema"})
            self.assertEqual(code, 400)
            load.assert_not_called()
        self.engine = self.svc.engine = MockEngine(self.tok, "Markdown is not JSON", max_context=16384)
        with mock.patch.object(self.engine, "generate", wraps=self.engine.generate) as generate:
            self.assertEqual(self.chat()[0], 502)
            self.assertEqual(generate.call_count, 1)

    @unittest.skipUnless(HAVE_JSONSCHEMA, "jsonschema is not installed")
    def test_invalid_schema_is_400_and_local_refs_work(self):
        for schema in ({"type": "object", "properties": {"x": {"type": "invalid"}}},
                       {"type": "object", "$ref": "https://other.example/schema"}):
            fmt = {"type": "json_schema", "json_schema": {"name": "test", "strict": True, "schema": schema}}
            self.assertEqual(self.chat(response_format=fmt)[0], 400)
        schema = object_schema({"answer": {"$ref": "#/$defs/n"}})
        schema["$defs"] = {"n": {"type": "integer"}}
        fmt = {"type": "json_schema", "json_schema": {"name": "local", "schema": schema}}
        self.assertEqual(self.chat('{"answer":4}', response_format=fmt)[0], 200)

    def test_malformed_formats_are_rejected_before_loading(self):
        formats = ([], {"type": "unsupported"},
                   {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
                   {"type": "json_schema", "json_schema": {"name": "x", "strict": "true", "schema": {"type": "object"}}})
        with mock.patch.object(self.svc, "load") as load:
            for fmt in formats:
                with self.subTest(fmt=fmt):
                    self.assertEqual(self.chat(response_format=fmt)[0], 400)
            self.assertEqual(self.chat(strata_mcp=True)[0], 400)
            self.assertEqual(self.request("/v1/chat/completions", [])[0], 400)
            load.assert_not_called()

    def test_a_tool_call_passes_through_with_a_format(self):
        call = ("Let me look.\n</think>\n\nChecking.\n\n<tool_call>\n<function=exec_command>\n<parameter=cmd>\n"
                "cat a.txt\n</parameter>\n</function>\n</tool_call>")
        for stream in (False, True):
            with self.subTest(stream=stream):
                code, reply = self.chat(call, tools=EXEC, stream=stream)
                self.assertEqual(code, 200, reply)
                if stream:
                    chunks = [json.loads(line[6:]) for line in reply.splitlines() if line.startswith("data: {")]
                    calls = [tc for c in chunks for tc in c["choices"][0]["delta"].get("tool_calls") or []]
                    self.assertEqual(calls[0]["function"]["name"], "exec_command")
                    self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")
                else:
                    message = reply["choices"][0]["message"]
                    self.assertEqual(reply["choices"][0]["finish_reason"], "tool_calls")
                    self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"]), {"cmd": "cat a.txt"})
        self.assertIn("applies only to your final answer", self.tok.decode(self.engine.last_prompt))

    def test_the_final_answer_with_tools_is_the_schema_object(self):
        code, reply = self.chat(tools=EXEC)
        self.assertEqual(code, 200, reply)
        self.assertEqual(json.loads(reply["choices"][0]["message"]["content"]), STORY)
        code, raw = self.chat(tools=EXEC, stream=True)
        chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
        self.assertEqual(json.loads("".join(c["choices"][0]["delta"].get("content", "") for c in chunks)), STORY)

    @unittest.skipUnless(HAVE_JSONSCHEMA, "jsonschema is not installed")
    def test_a_final_answer_with_tools_that_breaks_the_schema_fails(self):
        code, reply = self.chat('{"title":"Only a title"}', tools=EXEC)
        self.assertEqual((code, reply["error"]["code"]), (502, "structured_output_failed"))

    def test_without_jsonschema_an_object_is_still_required(self):
        # jsonschema is optional: without it json_schema answers are checked to be one JSON object, json_object
        # never needs it, and nothing is imported at server start
        with mock.patch.object(structured, "_jsonschema", False):
            self.assertEqual(self.chat()[0], 200)
            for script in ("## Film Plan - Photobooth", "[1, 2]"):
                code, reply = self.chat(script)
                self.assertEqual(code, 502, script)
                self.assertEqual(reply["error"]["code"], "structured_output_failed")
            self.assertEqual(self.chat('{"answer":4}', response_format={"type": "json_object"})[0], 200)

    def test_root_unions_of_object_shapes_are_accepted(self):
        # A root anyOf of object shapes (sent by apps written against llama.cpp's grammar path) still means
        # "one JSON object", so it is accepted; a non-object answer is still refused.
        claim = {"type": "object", "additionalProperties": False, "required": ["claim_text"],
                 "properties": {"claim_text": {"type": "string", "minLength": 1}}}
        empty = {"type": "object", "additionalProperties": False, "required": ["outcome"],
                 "properties": {"outcome": {"type": "string", "enum": ["no_substantive_content"]}}}
        schemas = [
            {"anyOf": [claim, empty]},
            {"oneOf": [claim, empty]},
            {"allOf": [claim, {"required": ["claim_text"]}]},
            {"$ref": "#/$defs/claim", "$defs": {"claim": claim}},
            {"anyOf": [{"$ref": "#/$defs/claim"}, {"anyOf": [empty]}], "$defs": {"claim": claim}},
        ]
        for schema in schemas:
            fmt = {"type": "json_schema", "json_schema": {"name": "union", "strict": True, "schema": schema}}
            with self.subTest(schema=schema):
                self.assertEqual(self.chat('{"claim_text":"The term is 12 months."}', response_format=fmt)[0], 200)
                self.assertEqual(self.chat("[1, 2]", response_format=fmt)[0], 502)

    def test_schemas_that_allow_non_objects_are_still_rejected(self):
        obj = {"type": "object"}
        schemas = [
            {"type": "array", "items": obj},
            {"type": "string"},
            {"type": ["object", "null"]},
            {"anyOf": [obj, {"type": "string"}]},
            {"anyOf": []},
            {"oneOf": [obj, {"type": "array"}]},
            {"allOf": [{"type": "string"}]},
            {"properties": {"x": {"type": "string"}}},           # no type: any JSON value validates
            {"$ref": "#/$defs/a", "$defs": {"a": {"$ref": "#/$defs/a"}}},   # reference cycle
            {"$ref": "#/$defs/missing", "$defs": {}},
            {"$ref": "#/$defs/s", "$defs": {"s": {"type": "string"}}},
        ]
        with mock.patch.object(self.svc, "load") as load:
            for schema in schemas:
                fmt = {"type": "json_schema", "json_schema": {"name": "bad", "schema": schema}}
                with self.subTest(schema=schema):
                    code, reply = self.chat(response_format=fmt)
                    self.assertEqual(code, 400)
                    self.assertIn("only JSON objects", reply["error"]["message"])
            load.assert_not_called()

    def test_the_server_does_not_import_jsonschema_at_start(self):
        import subprocess
        import sys
        code = ("import sys; import serve.server; "
                "sys.exit(1 if any(m == 'jsonschema' or m.startswith('jsonschema.') for m in sys.modules) else 0)")
        r = subprocess.run([sys.executable, "-c", code], cwd=str(Path(__file__).resolve().parents[1]))
        self.assertEqual(r.returncode, 0)

    # ------------------------------------------------------------------ T2b: lark response_format through HTTP
    def masked_chat(self, script, response_format):
        """chat() on an engine that takes token masks (a real engine's INFO token_mask key): the Constraint
        path of serve/constrain.py is really taken when the format survives prepare_format."""
        self.engine = self.svc.engine = MockEngine(self.tok, script, max_context=16384)
        self.engine.supports_mask = True
        req = {"messages": [{"role": "system", "content": "Write a film plan."},
                            {"role": "user", "content": "Photobooth. 15 seconds."}],
               "response_format": response_format, "reasoning_effort": "none"}
        return self.request("/v1/chat/completions", req)

    def test_lark_format_reaches_the_constraint_and_is_not_json_validated(self):
        # a well-formed lark rf passes prepare_format (no 400), reaches the Constraint (T1's passthrough)
        # and the turn is not JSON-validated afterwards: prose stays a 200 where json_object would be a 502.
        # Without llguidance installed the same request free-decodes to 200 (mask_available() is False),
        # so the 200 holds in both environments; the constraint assertion only runs with llguidance.
        from serve import constrain
        rf = {"type": "lark", "grammar": 'start: "yes" | "no"'}
        code, reply = self.masked_chat("yes", rf)
        self.assertEqual(code, 200)
        self.assertEqual(reply["choices"][0]["message"]["content"], "yes")
        if constrain.available():
            self.assertIsNotNone(self.engine.last_sampling_constraint)
        # a plain-text answer under a JSON format would be a 502: under lark it is a 200 (no after-the-turn gate)
        code, _ = self.chat("Prose, not JSON.", response_format=rf)
        self.assertEqual(code, 200)

    def test_lark_that_cannot_build_degrades_to_free_decoding(self):
        # a lark grammar llguidance cannot compile drops the mask (the existing "constrained decoding off"
        # fallback in _with_constraint): the turn free-decodes to a 200, never a 502 or 400.
        code, reply = self.masked_chat("Sure thing.", {"type": "lark", "grammar": "start: [[["})
        self.assertEqual(code, 200)
        self.assertEqual(reply["choices"][0]["message"]["content"], "Sure thing.")
        self.assertIsNone(self.engine.last_sampling_constraint)

    def test_malformed_lark_is_400_before_loading(self):
        with mock.patch.object(self.svc, "load") as load:
            for fmt in ({"type": "lark"}, {"type": "lark", "grammar": ""}, {"type": "lark", "grammar": 7}):
                with self.subTest(fmt=fmt):
                    code, reply = self.chat(response_format=fmt)
                    self.assertEqual(code, 400)
                    self.assertIn("grammar", reply["error"]["message"])
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
