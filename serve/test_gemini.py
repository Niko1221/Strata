"""serve/test_gemini.py - the Gemini API (POST /v1beta/models/{model}:generateContent) against the mock engine (no GPU, no pack).

    python -m unittest serve.test_gemini -v

The MockEngine answers with a fixed token sequence, so the test checks the request's shape, the events' order, the
answer's text, the token counts, and the tool call's arguments - in the shapes @google/genai 1.30.0 (Gemini CLI's
client) sends and reads: contents -> template messages, generationConfig -> the sampling keys, thinkingConfig -> the
thinking level, and candidates[].content.parts / finishReason / usageMetadata back out.
"""
from __future__ import annotations

import contextlib
import http.client
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import gemini  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.gemini import (GeminiError, contents_to_messages, error_body as gemini_error_body,  # noqa: E402
                          gemini_chunks, gemini_collect, sampling_of, tool_choice_of_request)
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service, serve  # noqa: E402
from serve.server import IM_END  # noqa: E402

try:
    import jsonschema  # noqa: F401
    HAVE_JSONSCHEMA = True
except ImportError:                  # optional: json_schema answers are then only checked to be one object
    HAVE_JSONSCHEMA = False

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ChatTemplate(ROOT / "serve/chat_template.jinja")

CALL = ("Let me look.\n</think>\n\nChecking.\n\n<tool_call>\n<function=exec_command>\n<parameter=cmd>\ncat a.txt\n"
        "</parameter>\n</function>\n</tool_call>")
ANSWER = "Read it.\n</think>\n\nThe file says before."
DECL = {"name": "exec_command", "description": "Runs a command.",
        "parametersJsonSchema": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}
GEMINI_TOOLS = [{"functionDeclarations": [DECL]}]


class GeminiParse(unittest.TestCase):
    def test_contents_keep_their_roles(self):
        req = {"contents": [{"role": "user", "parts": [{"text": "hi"}]},
                            {"role": "model", "parts": [{"text": "hello"}]},
                            {"role": "user", "parts": [{"text": "again"}]}],
               "generationConfig": {"temperature": 0.5}}
        messages, tools, kw = contents_to_messages(req)
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])
        self.assertEqual(messages[0]["content"], "hi")
        self.assertEqual(sampling_of(req), {"temperature": 0.5})

    def test_system_instruction_becomes_the_system_message(self):
        req = {"systemInstruction": {"parts": [{"text": "Answer in one word."}]},
               "contents": [{"parts": [{"text": "hi"}]}]}
        messages, tools, kw = contents_to_messages(req)
        self.assertEqual(messages[0], {"role": "system", "content": "Answer in one word."})
        self.assertEqual(len(messages), 2)

    def test_a_tool_round_reaches_the_encoder_as_the_other_routes_put_it(self):
        """An earlier answer's call and its tool's result become the template's tool_calls and tool messages."""
        req = {"contents": [{"role": "user", "parts": [{"text": "what is in a.txt?"}]},
                            {"role": "model", "parts": [{"functionCall": {"name": "exec_command",
                                                                          "args": {"cmd": "cat a.txt"}}}]},
                            {"role": "user", "parts": [{"functionResponse": {"id": "c1", "name": "exec_command",
                                                                             "response": {"output": "before"}}}]}],
               "tools": GEMINI_TOOLS}
        messages, tools, kw = contents_to_messages(req)
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool"])
        self.assertEqual(messages[1]["tool_calls"],
                         [{"function": {"name": "exec_command", "arguments": {"cmd": "cat a.txt"}}}])
        self.assertEqual(messages[2]["content"], "before")
        self.assertEqual(tools, [{"name": "exec_command", "description": "Runs a command.",
                                  "parameters": DECL["parametersJsonSchema"]}])

    def test_inline_data_images_reach_the_encoder_and_uploaded_files_are_refused(self):
        req = {"contents": [{"role": "user", "parts": [{"inlineData": {"mimeType": "image/png", "data": "QUx4"}},
                                                        {"text": "what is in it?"}]}]}
        messages, tools, kw = contents_to_messages(req)
        self.assertEqual(messages[0]["content"],
                         [{"type": "image", "source": "data:image/png;base64,QUx4"},
                          {"type": "text", "text": "what is in it?"}])
        with self.assertRaises(GeminiError) as bad:
            contents_to_messages({"contents": [{"role": "user",
                                                "parts": [{"fileData": {"fileUri": "x", "mimeType": "image/png"}}]}]})
        self.assertIn("fileData", str(bad.exception))

    def test_generation_config_becomes_the_sampling_keys(self):
        req = {"contents": [{"parts": [{"text": "hi"}]}],
               "generationConfig": {"temperature": 0.3, "topP": 0.9, "topK": 5, "maxOutputTokens": 40,
                                     "stopSequences": ["END"], "seed": 7,
                                     "thinkingConfig": {"thinkingBudget": 4000}}}
        self.assertEqual(sampling_of(req), {"temperature": 0.3, "top_p": 0.9, "top_k": 5, "max_tokens": 40,
                                            "stop": ["END"], "seed": 7})
        # the budget picks a level, as Anthropic's budget_tokens does (under 8K: medium)
        self.assertEqual(contents_to_messages(req)[2], {"reasoning_effort": "medium"})

    def test_thinking_budget_zero_turns_thinking_off(self):
        req = {"contents": [{"parts": [{"text": "hi"}]}],
               "generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}}
        self.assertEqual(contents_to_messages(req)[2], {"enable_thinking": False})

    def test_minus_one_lets_the_model_choose(self):
        """-1 is Gemini's dynamic thinking: the model decides, so the level stays as the shared settings left it -
        it is not the 0 above, which turns thinking off."""
        req = {"contents": [{"parts": [{"text": "hi"}]}],
               "generationConfig": {"thinkingConfig": {"thinkingBudget": -1}}}
        self.assertEqual(contents_to_messages(req)[2], {})

    def test_only_image_inline_data_is_read(self):
        """Gemini CLI sends a PDF or an audio file inline too; this server's encoder reads pictures, so the
        part is refused rather than sent along as a picture that is not one."""
        req = {"contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": "application/pdf", "data": "JVBER"}}]}]}
        with self.assertRaises(GeminiError) as bad:
            contents_to_messages(req)
        self.assertIn("only image/", str(bad.exception))
        self.assertEqual(gemini_error_body(str(bad.exception))["error"]["code"], 400)

    def test_a_tool_result_stays_next_to_the_call_it_answers(self):
        """A user turn can carry the reader's text and a functionResponse in one parts list; the tool message
        goes first, so the text does not land between the call and its result."""
        req = {"contents": [{"role": "user", "parts": [{"text": "what is in a.txt?"}]},
                            {"role": "model", "parts": [{"functionCall":
                                                         {"name": "exec_command", "args": {"cmd": "cat a.txt"}}}]},
                            {"role": "user", "parts": [{"functionResponse":
                                                        {"name": "exec_command", "response": {"output": "before"}}},
                                                       {"text": "and now?"}]}]}
        messages = contents_to_messages(req)[0]
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool", "user"])
        self.assertEqual(messages[2]["content"], "before")

    def test_gemini_3_spells_the_level_as_a_word(self):
        """thinkingConfig.thinkingLevel (Gemini 3) takes the same words the other routes' reasoning_effort
        takes; the 2.5 budget still wins when a client sends both."""
        for level, want in (("minimal", {"enable_thinking": False}), ("low", {"reasoning_effort": "low"}),
                            ("medium", {"reasoning_effort": "medium"}), ("high", {"reasoning_effort": "xhigh"})):
            req = {"contents": [{"parts": [{"text": "hi"}]}],
                   "generationConfig": {"thinkingConfig": {"thinkingLevel": level}}}
            self.assertEqual(contents_to_messages(req)[2], want)
        both = {"contents": [{"parts": [{"text": "hi"}]}],
                "generationConfig": {"thinkingConfig": {"thinkingLevel": "high", "thinkingBudget": 1000}}}
        self.assertEqual(contents_to_messages(both)[2], {"reasoning_effort": "low"})
        with self.assertRaises(GeminiError) as e:
            contents_to_messages({"contents": [], "generationConfig":
                                  {"thinkingConfig": {"thinkingLevel": "everything"}}})
        self.assertEqual(gemini_error_body(str(e.exception))["error"]["code"], 400)

    def test_gemini_schema_type_names_are_normalized(self):
        """Gemini's own schemas spell the JSON Schema type names in caps (STRING, OBJECT); the template's
        tool parser reads the lowercase names.  A property actually named "type" keeps its schema."""
        decl = {"name": "exec_command", "description": "Runs a command.",
                "parametersJsonSchema": {"type": "OBJECT", "properties": {"cmd": {"type": "STRING"},
                                                                          "type": {"type": "STRING"}},
                                          "required": ["cmd"]}}
        tools = contents_to_messages({"contents": [], "tools": [{"functionDeclarations": [decl]}]})[1]
        self.assertEqual(tools[0]["parameters"],
                         {"type": "object", "properties": {"cmd": {"type": "string"},
                                                           "type": {"type": "string"}}, "required": ["cmd"]})

    def test_tool_config_mode_forces_the_call(self):
        """Gemini's own names: `allowedFunctionNames` (not "allowedFunctionCalls") and the modes AUTO, ANY,
        NONE, VALIDATED.  One allowed name is the forced call the other routes take; ANY with no names (or
        several) is only "a tool has to be called"."""
        for cfg, want in (({"mode": "AUTO"}, None), ({"mode": "NONE"}, {"type": "none"}),
                          ({"mode": "ANY"}, "any"),
                          ({"mode": "ANY", "allowedFunctionNames": ["exec_command"]},
                           {"type": "tool", "name": "exec_command"}),
                          ({"mode": "VALIDATED", "allowedFunctionNames": ["exec_command"]},
                           {"type": "tool", "name": "exec_command"})):
            req = {"contents": [], "toolConfig": {"functionCallingConfig": cfg}}
            self.assertEqual(tool_choice_of_request(req), want)

    def test_a_body_that_is_not_a_generate_content_request_says_so(self):
        for bad in ({"contents": "hi"}, {"contents": [{"parts": "hi"}]},
                    {"contents": [{"parts": [{"text": 5}]}]},
                    {"contents": [{"role": "model", "parts": [{"functionCall": {"args": {}}}]}]},
                    {"contents": [{"role": "user", "parts": [{"inlineData": {"mimeType": "image/png"}}]}]},
                    {"contents": [], "tools": [{"functionDeclarations": [{"description": "no name"}]}]},
                    {"contents": [], "generationConfig": {"thinkingConfig": {"thinkingBudget": "all"}}}):
            with self.assertRaises((GeminiError, ValueError)) as e:
                contents_to_messages(bad)
            self.assertEqual(gemini_error_body(str(e.exception))["error"]["code"], 400)


class Server(unittest.TestCase):
    script = ANSWER
    engine_class = MockEngine

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = self.engine_class(self.tok, self.script, max_context=16384)
        self.svc = Service(self.engine, self.tok, TEMPLATE)
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def ask(self, path="/v1beta/models/mock:generateContent", body=None, headers=None):
        """(status, parsed body: a dict, or the list of SSE payloads of a stream)."""
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                c.request("POST", path, body=json.dumps(body or {}).encode(),
                          headers={"Content-Type": "application/json", **(headers or {})})
                r = c.getresponse()
                raw = r.read().decode()
            if r.getheader("Content-Type") == "text/event-stream":
                # the way @google/genai's reader does: every `data: ` block is one whole response, JSON-parsed, and
                # there is no [DONE] sentinel to stop at
                self.assertNotIn("[DONE]", raw)
                return r.status, [json.loads(b[len("data: "):]) for b in raw.split("\n\n")
                                  if b.startswith("data: ")]
            return r.status, json.loads(raw)
        finally:
            c.close()

    def get(self, path):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                c.request("GET", path)
                r = c.getresponse()
                return r.status, json.loads(r.read().decode())
        finally:
            c.close()

    def replay(self, text):
        """The mock engine's next answer, in its own construction: token ids plus the end-of-turn token."""
        self.engine.script = self.tok.encode(text, parse_special=True) + self.tok.encode(IM_END, parse_special=True)

    def parts(self, payload):
        return payload["candidates"][0]["content"]["parts"]

    def text(self, payload):
        return "".join(p.get("text", "") for p in self.parts(payload) if not p.get("thought"))

    def calls(self, payload):
        return [p["functionCall"] for p in self.parts(payload) if "functionCall" in p]


class OverHttp(Server):
    def test_non_streaming_answer(self):
        code, r = self.ask(body={"contents": [{"parts": [{"role": "user", "text": "hi"}]}]})
        self.assertEqual((code, r["candidates"][0]["finishReason"]), (200, "STOP"))
        self.assertEqual(self.text(r), "The file says before.")
        self.assertEqual(r["modelVersion"], self.svc.model)
        u = r["usageMetadata"]
        self.assertEqual({k: v for k, v in u.items() if k != "thoughtsTokenCount"},
                         {"promptTokenCount": len(self.engine.last_prompt),
                          "candidatesTokenCount": len(self.tok.encode(ANSWER)) + 1,
                          "totalTokenCount": len(self.engine.last_prompt) + len(self.tok.encode(ANSWER)) + 1})
        self.assertGreater(u["thoughtsTokenCount"], 0)          # the model thought before it answered

    def test_stream_event_shapes(self):
        code, events = self.ask("/v1beta/models/mock:streamGenerateContent?alt=sse",
                                {"contents": [{"parts": [{"role": "user", "text": "hi"}]}],
                                 "generationConfig": {"temperature": 0.5}})
        self.assertEqual(code, 200)
        self.assertTrue(all("responseId" in e and "modelVersion" in e for e in events))
        self.assertEqual("".join(self.text(e) for e in events), "The file says before.")
        self.assertEqual(events[-1]["candidates"][0]["finishReason"], "STOP")
        self.assertEqual(events[-1]["usageMetadata"]["promptTokenCount"], len(self.engine.last_prompt))

    def test_the_prompt_is_the_template_rendered_for_the_conversation(self):
        """The prompt this server reads is the template's, the same one the OpenAI route renders for the same
        conversation."""
        self.ask(body={"contents": [{"parts": [{"role": "user", "text": "what is in a.txt?"}]}]})
        prompt = self.tok.decode(self.engine.last_prompt)
        self.assertIn("what is in a.txt?", prompt)
        self.assertIn("<|im_start|>assistant", prompt)          # the turn the model will answer

    def test_a_tool_call_comes_back_whole(self):
        """Gemini's format has no partial form for a call: the streamed call arrives as one functionCall part."""
        self.replay(CALL)
        code, r = self.ask(body={"contents": [{"parts": [{"role": "user", "text": "what is in a.txt?"}]}],
                                 "tools": GEMINI_TOOLS})
        self.assertEqual(code, 200)
        self.assertEqual(self.calls(r), [{"name": "exec_command", "args": {"cmd": "cat a.txt"}}])

    def test_streamed_tool_call(self):
        self.replay(CALL)
        code, events = self.ask("/v1beta/models/mock:streamGenerateContent?alt=sse",
                                {"contents": [{"parts": [{"role": "user", "text": "what is in a.txt?"}]}],
                                 "tools": GEMINI_TOOLS})
        self.assertEqual(code, 200)
        calls = [c for e in events for c in
                 (p["functionCall"] for p in self.parts(e) if "functionCall" in p)]
        self.assertEqual(calls, [{"name": "exec_command", "args": {"cmd": "cat a.txt"}}])
        self.assertEqual(events[-1]["candidates"][0]["finishReason"], "STOP")

    def test_tool_config_any_forces_the_call(self):
        self.replay(CALL)
        code, r = self.ask(body={"contents": [{"parts": [{"role": "user", "text": "what is in a.txt?"}]}],
                                 "tools": GEMINI_TOOLS,
                                 "toolConfig": {"functionCallingConfig": {"mode": "ANY"}}})
        self.assertEqual(code, 200)
        self.assertEqual(self.calls(r), [{"name": "exec_command", "args": {"cmd": "cat a.txt"}}])

    def test_the_reasoning_arrives_as_its_own_parts(self):
        code, events = self.ask("/v1beta/models/mock:streamGenerateContent?alt=sse",
                                {"contents": [{"parts": [{"role": "user", "text": "hi"}]}],
                                 "generationConfig": {"thinkingConfig": {"thinkingBudget": 4000}}})
        self.assertEqual(code, 200)
        parts = [p for e in events for p in self.parts(e)]
        thoughts = [p for p in parts if p.get("thought")]
        answer = [p for p in parts if not p.get("thought")]
        self.assertEqual("".join(p["text"] for p in thoughts), "Read it.\n")
        self.assertEqual("".join(p["text"] for p in answer), "The file says before.")
        self.assertLess(parts.index(thoughts[-1]), parts.index(answer[0]))
        self.assertGreater(events[-1]["usageMetadata"]["thoughtsTokenCount"], 0)

    def test_include_thoughts_false_hides_the_thought_parts(self):
        """Gemini 3's word: the model still thinks (thoughtsTokenCount counts it), the client just does not
        see the parts - in the stream and in the one collected answer."""
        body = {"contents": [{"parts": [{"role": "user", "text": "hi"}]}],
                "generationConfig": {"thinkingConfig": {"thinkingBudget": 4000, "includeThoughts": False}}}
        code, r = self.ask(body=body)
        self.assertEqual(code, 200)
        self.assertFalse([p for p in self.parts(r) if p.get("thought")])
        self.assertEqual(self.text(r), "The file says before.")
        self.assertGreater(r["usageMetadata"]["thoughtsTokenCount"], 0)
        code, events = self.ask("/v1beta/models/mock:streamGenerateContent?alt=sse", body)
        self.assertEqual(code, 200)
        self.assertFalse([p for e in events for p in self.parts(e) if p.get("thought")])

    def test_count_tokens_does_not_run_the_model(self):
        body = {"contents": [{"parts": [{"role": "user", "text": "hi"}]}]}
        code, r = self.ask("/v1beta/models/mock:countTokens", body)
        self.assertEqual(code, 200)
        self.assertNotIn("candidates", r)
        messages, tools, kw = contents_to_messages(body)
        # CountTokensResponse's own field is totalTokens - a client reading anything else gets nothing
        self.assertEqual(r["totalTokens"], len(self.svc.encode_prompt(messages, tools, kw)))

    def test_the_models_list(self):
        code, r = self.get("/v1beta/models")
        self.assertEqual((code, r["totalResults"], r["models"][0]["name"]),
                         (200, 1, f"models/{self.svc.model}"))
        self.assertIn("generateContent", r["models"][0]["supportedGenerationMethods"])

    def test_errors_use_the_gemini_format(self):
        """The SDK reads error.code and throws an ApiError when it is 400..599.  A bad field has to reach the
        client as that 400: the field's name is not an HTTP status, and a status the server cannot write leaves
        the client with a dropped connection."""
        for bad in ({"contents": "hi"},
                    {"contents": [{"parts": [{"text": 5}]}]},
                    {"contents": [{"role": "model", "parts": [{"functionCall": {"args": {}}}]}]},
                    {"contents": [], "tools": [{"functionDeclarations": [{"description": "no name"}]}]},
                    {"contents": [], "generationConfig":
                         {"thinkingConfig": {"thinkingLevel": "everything"}}}):
            code, r = self.ask(body=bad)
            self.assertEqual((code, r["error"]["code"]), (400, 400))
        code, r = self.ask("/v1beta/models/mock:embedContents", {})
        self.assertEqual((code, r["error"]["code"]), (404, 404))
        self.assertIn("embedContents", r["error"]["message"])

    def test_gemini_cli_asks_its_background_calls_for_json(self):
        """Gemini CLI's next-speaker check, loop detection and chat compression send responseMimeType/responseSchema
        (its schemas spell the type names in caps) and parse the answer as JSON - prose would fail quietly.
        Strata's structured output answers them, and the answer is only sent once it validates."""
        self.replay('Consider it.\n</think>\n\n{"next": "user"}')
        fmt = {"responseMimeType": "application/json",
               "responseSchema": {"type": "OBJECT", "properties": {"next": {"type": "STRING"}},
                                  "required": ["next"]}}
        code, r = self.ask(body={"contents": [{"parts": [{"text": "who speaks next?"}]}],
                                 "generationConfig": fmt})
        self.assertEqual(code, 200)
        self.assertEqual([p for p in self.parts(r) if not p.get("thought")],
                         [{"text": '{"next":"user"}'}])
        code, events = self.ask("/v1beta/models/mock:streamGenerateContent?alt=sse",
                                {"contents": [{"parts": [{"text": "who speaks next?"}]}],
                                 "generationConfig": fmt})
        self.assertEqual(code, 200)
        self.assertEqual([p for e in events for p in self.parts(e) if not p.get("thought")],
                         [{"text": '{"next":"user"}'}])
        # the newer spelling, responseJsonSchema, answers the same way
        code, r = self.ask(body={"contents": [{"parts": [{"text": "who speaks next?"}]}],
                                 "generationConfig": {"responseMimeType": "application/json",
                                                      "responseJsonSchema": {"type": "object",
                                                                             "properties": {"next": {"type": "string"}},
                                                                             "required": ["next"]}}})
        self.assertEqual(code, 200)
        self.assertEqual([p for p in self.parts(r) if not p.get("thought")],
                         [{"text": '{"next":"user"}'}])

    def test_prose_is_not_passed_off_as_json(self):
        """A structured answer that does not validate ends the turn with an error, as the OpenAI route's does -
        in the stream it arrives as an error chunk, since the headers are already sent."""
        fmt = {"responseMimeType": "application/json"}
        code, r = self.ask(body={"contents": [{"parts": [{"text": "hi"}]}], "generationConfig": fmt})
        self.assertEqual((code, r["error"]["code"]), (502, 502))
        self.assertIn("JSON", r["error"]["message"])
        code, events = self.ask("/v1beta/models/mock:streamGenerateContent?alt=sse",
                                {"contents": [{"parts": [{"text": "hi"}]}], "generationConfig": fmt})
        self.assertEqual(code, 200)
        self.assertEqual(events[-1]["error"]["code"], 502)

    def test_every_schema_spelling_reaches_response_format(self):
        """responseJsonSchema is what @google/genai sends; it and the other spellings carry the schema, not only
        "some JSON" - plain JSON mode would pass an answer that lacks a required key."""
        schema = {"type": "object", "properties": {"next": {"type": "string"}}, "required": ["next"]}
        for key in ("responseSchema", "responseJsonSchema", "response_schema", "response_json_schema"):
            fmt = gemini.response_format_of({"generationConfig": {"responseMimeType": "application/json",
                                                                  key: schema}})
            self.assertEqual(fmt, {"type": "json_schema", "json_schema": {"name": "gemini", "schema": schema}}, key)

    @unittest.skipUnless(HAVE_JSONSCHEMA, "jsonschema is not installed")
    def test_response_json_schema_is_enforced_not_just_json(self):
        """An answer that is JSON but breaks the responseJsonSchema is refused (only jsonschema can tell)."""
        self.replay('Consider it.\n</think>\n\n{"other": 1}')
        schema = {"type": "object", "properties": {"next": {"type": "string"}}, "required": ["next"]}
        code, r = self.ask(body={"contents": [{"parts": [{"text": "who speaks next?"}]}],
                                 "generationConfig": {"responseMimeType": "application/json",
                                                      "responseJsonSchema": schema}})
        self.assertEqual((code, r["error"]["code"]), (502, 502))


class DyingEngine(MockEngine):
    """The engine ends a few tokens into its answer (issue #27's out-of-memory killer)."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        for i, t in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
            if i == 5:
                raise EngineDied("the engine stopped unexpectedly (exit code -9)")
            yield t


class EngineFailure(Server):
    script = ANSWER
    engine_class = DyingEngine

    def test_the_engine_ending_mid_stream_says_so(self):
        """The stream has started, so no 400 can replace it: the turn ends with an error chunk the SDK reads
        as an ApiError, as the OpenAI and Anthropic routes end their streams."""
        code, events = self.ask("/v1beta/models/mock:streamGenerateContent?alt=sse",
                                {"contents": [{"parts": [{"text": "hi"}]}]})
        self.assertEqual(code, 200)
        self.assertEqual(events[-1]["error"]["code"], 503)
        self.assertIn("restarts it", events[-1]["error"]["message"])

    def test_the_engine_ending_before_the_answer_says_so(self):
        """The one collected answer: a 503 in Gemini's shape, not the OpenAI one the outer handler writes."""
        code, r = self.ask(body={"contents": [{"parts": [{"text": "hi"}]}]})
        self.assertEqual((code, r["error"]["code"]), (503, 503))
        self.assertIn("restarts it", r["error"]["message"])


class ToolRoundTrip(Server):
    script = [CALL, ANSWER]

    def test_gemini_cli_sends_everything_back_and_the_prompt_continues(self):
        """A second turn that carries the first answer's call and the tool's result continues the same prompt -
        the conversation cache is reused, as the OpenAI path's rounds are."""
        user = {"role": "user", "parts": [{"text": "what is in a.txt?"}]}
        code, r = self.ask(body={"contents": [user], "tools": GEMINI_TOOLS})
        self.assertEqual(code, 200)
        first_prompt = self.tok.decode(self.engine.last_prompt)
        call = self.calls(r)[0]
        self.assertEqual((call["name"], call["args"]), ("exec_command", {"cmd": "cat a.txt"}))
        code, r = self.ask(body={"contents": [user,
                                             {"role": "model", "parts": [{"functionCall": call}]},
                                             {"role": "user", "parts": [{"functionResponse":
                                                                        {"name": "exec_command",
                                                                         "response": {"output": "before"}}}]}],
                                 "tools": GEMINI_TOOLS})
        self.assertEqual(code, 200, r)
        second_prompt = self.tok.decode(self.engine.last_prompt)
        self.assertTrue(second_prompt.startswith(first_prompt), second_prompt[-600:])
        self.assertIn("before", second_prompt)
        self.assertEqual(self.text(r), "The file says before.")


class WithKey(Server):
    def test_the_google_key_header_is_accepted(self):
        """@google/genai sends x-goog-api-key (and some Google endpoints use ?key=); the other routes keep
        accepting only what they always did."""
        self.svc.api_key = "s3cret"
        body = {"contents": [{"parts": [{"role": "user", "text": "hi"}]}]}
        code, r = self.ask(body=body)
        self.assertEqual((code, "API key" in r["error"]["message"]), (401, True))
        for headers in ({"x-goog-api-key": "s3cret"}, {"x-api-key": "s3cret"}, {"Authorization": "Bearer s3cret"}):
            code, r = self.ask(body=body, headers=headers)
            self.assertEqual((code, r["candidates"][0]["finishReason"]), (200, "STOP"))
        code, r = self.ask(body=body, headers=None)
        self.assertEqual(code, 401)
        code, r = self.get("/v1beta/models?key=s3cret")
        self.assertEqual((code, r["totalResults"]), (200, 1))


if __name__ == "__main__":
    unittest.main()
