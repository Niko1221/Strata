"""The grammar path (issue #3862): response_format rides the sampler when the engine can constrain
decoding, the prompt path when it cannot, and the bare `grammar` parameter is the client's own GBNF.

The converter CLI (grammar_schema) is exercised for real when the build tree has it; the engine-side
protocol (the GRAMMAR line, the pump's routing) belongs to the GPU acceptance suite (Task 12)."""
import json
import os
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve import grammar as grammar_api
from serve import grammar_schema as grammar_conv
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve

BUILD = Path(__file__).resolve().parents[1] / "build"
CONVERTER = BUILD / ("grammar_schema.exe" if os.name == "nt" else "grammar_schema")
HAVE_CONVERTER = CONVERTER.exists()

STORY = {"title": "Photobooth", "boxes": [{"box": 1, "beat": "Hailee changes poses."}]}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["title", "boxes"],
          "properties": {"title": {"type": "string"},
                         "boxes": {"type": "array", "minItems": 1,
                                   "items": {"type": "object", "additionalProperties": False,
                                             "required": ["box", "beat"],
                                             "properties": {"box": {"type": "integer"},
                                                            "beat": {"type": "string"}}}}}}
FORMAT = {"type": "json_schema", "json_schema": {"name": "story", "strict": True, "schema": SCHEMA}}
RAW_GRAMMAR = 'root ::= "yes" | "no"\n'


class GrammarBase(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, json.dumps(STORY), max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def capable(self, script=None, converter=True):
        """Swap in a MockEngine whose INFO line says it constrains decoding (0.2.3+)."""
        self.engine = self.svc.engine = MockEngine(self.tok, script if script is not None else json.dumps(STORY),
                                                   max_context=16384)
        self.engine.info = {"grammar": 1}
        if converter:
            self.engine.spawn = [str(BUILD / "strata.exe")]   # where the schema converter CLI is looked up
        return self.engine

    def request(self, path, body, method="POST"):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"}, method=method)
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read().decode()
            return response.status, data if "event-stream" in response.headers.get("Content-Type", "") else json.loads(data)

    def chat(self, **kwargs):
        body = {"messages": [{"role": "user", "content": "Plan the shot."}], "reasoning_effort": "none", **kwargs}
        return self.request("/v1/chat/completions", body)


class GrammarPath(GrammarBase):
    @unittest.skipUnless(HAVE_CONVERTER, "grammar_schema CLI is not built")
    def test_json_schema_streams_under_the_grammar_not_the_prompt(self):
        engine = self.capable()
        code, reply = self.chat(response_format=FORMAT)
        self.assertEqual(code, 200, reply)
        self.assertEqual(json.loads(reply["choices"][0]["message"]["content"]), STORY)
        prompt = self.tok.decode(engine.last_prompt)
        self.assertNotIn("OUTPUT FORMAT REQUIREMENT", prompt)      # the constraint is in the sampler now
        self.assertIn("root ::=", engine.grammars[-1])             # the converted GBNF reached the engine
        self.assertEqual(engine.last_sampling.get("grammar"), grammar_api.grammar_sha16(engine.grammars[-1]))

    @unittest.skipUnless(HAVE_CONVERTER, "grammar_schema CLI is not built")
    def test_the_same_schema_registers_once_per_engine(self):
        engine = self.capable()
        self.assertEqual(self.chat(response_format=FORMAT)[0], 200)
        self.assertEqual(self.chat(response_format=FORMAT)[0], 200)
        self.assertEqual(len(engine.grammars), 1)                  # the content hash is the memo

    @unittest.skipUnless(HAVE_CONVERTER, "grammar_schema CLI is not built")
    def test_the_json_object_grammar_and_an_unbuffered_stream(self):
        engine = self.capable()
        code, raw = self.chat(response_format={"type": "json_object"}, stream=True)
        self.assertEqual(code, 200, raw)
        self.assertEqual(engine.grammars[-1], grammar_api.JSON_OBJECT_GBNF)
        chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
        deltas = [c for c in chunks if c["choices"][0].get("delta", {}).get("content")]
        self.assertGreater(len(deltas), 1)                         # streamed piece by piece, not one buffered delta
        self.assertEqual(json.loads("".join(c["choices"][0]["delta"]["content"] for c in deltas)), STORY)
        self.assertIn("data: [DONE]", raw)

    @unittest.skipUnless(HAVE_CONVERTER, "grammar_schema CLI is not built")
    def test_the_canary_calls_a_broken_constraint_a_grammar_engine_bug(self):
        # the mock ignores the grammar and answers in prose: the schema was token-by-token's promise, so a
        # stream that fails its own schema is reported as the engine's bug, and the stream still completes
        self.capable(script="## Film Plan\nNot JSON at all.")
        code, reply = self.chat(response_format=FORMAT)
        self.assertEqual(code, 502)
        self.assertEqual(reply["error"]["code"], "structured_output_failed")
        self.assertIn("grammar engine bug", reply["error"]["message"])
        _, raw = self.chat(response_format=FORMAT, stream=True)
        self.assertIn("grammar engine bug", raw)
        self.assertIn("data: [DONE]", raw)

    def test_a_capabilityless_engine_keeps_the_prompt_path_byte_for_byte(self):
        # the pre-#3862 contract: directive in the prompt, sampling keys carry no grammar
        code, reply = self.chat(response_format=FORMAT)
        self.assertEqual(code, 200, reply)
        prompt = self.tok.decode(self.engine.last_prompt)
        self.assertIn("OUTPUT FORMAT REQUIREMENT", prompt)
        self.assertNotIn("grammar", self.engine.last_sampling)
        self.assertEqual(self.engine.grammars, [])
        keys = StrataEngine.sampling_keys({"temperature": 0.6, "top_k": 40, "seed": 7, "max_tokens": 100})
        self.assertNotIn("grammar", keys)
        self.assertEqual(keys, " temperature=0.6 top_k=40 seed=7")        # byte-identical to before the key

    def test_a_bare_grammar_is_the_clients_text_and_a_capability_is_required(self):
        self.assertEqual(self.chat(grammar=RAW_GRAMMAR)[0], 400)            # engine never said grammar=1
        engine = self.capable(converter=False)
        code, reply = self.chat(grammar=RAW_GRAMMAR)
        self.assertEqual(code, 200, reply)
        self.assertEqual(engine.grammars[-1], RAW_GRAMMAR)                  # no conversion: the text as sent
        self.assertEqual(engine.last_sampling.get("grammar"), grammar_api.grammar_sha16(RAW_GRAMMAR))
        for bad in ("", "   ", 12, {"file": "x"}, "x" * ((1 << 20) + 1)):
            with self.subTest(bad=repr(bad)[:40]):
                code, reply = self.chat(grammar=bad)
                self.assertEqual(code, 400)
                self.assertIn("grammar", reply["error"]["message"])

    def test_the_status_line_says_how_structured_output_works(self):
        code, status = self.request("/v1/status", None, method="GET")
        self.assertEqual(code, 200)
        self.assertEqual(status["structured_output"]["method"], "prompt_and_validate")
        self.assertFalse(status["structured_output"]["constrained_decoding"])
        self.capable(converter=False)
        code, status = self.request("/v1/status", None, method="GET")
        self.assertEqual(status["structured_output"]["method"], "grammar")
        self.assertTrue(status["structured_output"]["constrained_decoding"])
        self.assertFalse(status["structured_output"]["stream_buffered"])


class GrammarConverter(unittest.TestCase):
    @unittest.skipUnless(HAVE_CONVERTER, "grammar_schema CLI is not built")
    def test_the_converter_turns_a_schema_into_a_root_grammar(self):
        text = grammar_conv.schema_to_grammar(BUILD, SCHEMA)
        self.assertIn("root ::=", text)
        self.assertEqual(text, grammar_conv.schema_to_grammar(BUILD, SCHEMA))   # the LRU returns the same text

    def test_a_missing_converter_falls_back_to_the_prompt_path(self):
        fmt = {"type": "json_schema", "json_schema": {"name": "s", "schema": SCHEMA}}
        messages, validator = None, None
        engine = MockEngine(ByteTokenizer(), "{}")
        engine.info = {"grammar": 1}                    # capable, but no spawn: no converter to be found
        grammar_text, messages, validator = grammar_conv.plan_response_format(engine, fmt, [{"role": "user", "content": "x"}])
        self.assertIsNone(grammar_text)                 # the prompt path takes the turn
        self.assertIsNotNone(validator)


class GrammarWire(unittest.TestCase):
    def test_sha16_is_the_content_hash_of_the_text(self):
        self.assertEqual(len(grammar_api.grammar_sha16(RAW_GRAMMAR)), 16)
        self.assertNotEqual(grammar_api.grammar_sha16(RAW_GRAMMAR), grammar_api.grammar_sha16(RAW_GRAMMAR + " "))

    def unescape(self, wire):
        """The engine's one-pass unescape (generate.cpp, GRAMMAR): \\n to newline, \\\\ to a backslash."""
        out, i = [], 0
        while i < len(wire):
            if wire[i] == "\\" and i + 1 < len(wire) and wire[i + 1] in ("n", "\\"):
                out.append("\n" if wire[i + 1] == "n" else "\\")
                i += 2
            else:
                out.append(wire[i])
                i += 1
        return "".join(out)

    def test_the_wire_line_survives_a_multi_line_grammar(self):
        text = grammar_api.JSON_OBJECT_GBNF
        wire = grammar_api.wire(text)
        self.assertNotIn("\n", wire)                   # one line: the engine's getline reads it whole
        self.assertEqual(self.unescape(wire), text)    # ... and the unescape gives the grammar back

    def test_check_param(self):
        self.assertEqual(grammar_api.check_param(RAW_GRAMMAR), RAW_GRAMMAR)
        for bad in (None, 12, "", "  ", "x" * ((1 << 20) + 1)):
            with self.assertRaises(ValueError):
                grammar_api.check_param(bad)


if __name__ == "__main__":
    unittest.main()
