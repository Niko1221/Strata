"""serve/test_constrain.py - constrained decoding (serve/constrain.py) without a GPU:
  * the grammar and the masks themselves (llguidance over the byte tokenizer),
  * MockEngine under a mask over HTTP (chat and Responses, with tools, with thinking),
  * the engine protocol (GEN mask=1 / MQ / MK / MF) against serve/fake_mask_engine.py, a stand-in `strata --serve`.
Skipped without llguidance (python -m pip install llguidance).

    python -m unittest serve.test_constrain -v
"""
from __future__ import annotations
import base64
import json
import os
import stat
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve import constrain as C
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ChatTemplate(ROOT / "serve/chat_template.jinja")
need = unittest.skipUnless(C.available(), "llguidance is not installed")
SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"], "additionalProperties": False}
FMT = {"type": "json_schema", "json_schema": {"name": "num", "strict": True, "schema": SCHEMA}}
TEXT_FMT = {"type": "json_schema", "name": "num", "strict": True, "schema": SCHEMA}
TOOLS = [{"type": "function", "name": "exec_command", "description": "Runs a command.", "strict": False,
          "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}]
CALL = ("Let me look.\n</think>\n\nChecking.\n\n<tool_call>\n<function=exec_command>\n<parameter=cmd>\ncat a.txt\n"
        "</parameter>\n</function>\n</tool_call>")


# ------------------------------------------------------------------------------------------------ the masks
@need
class Masks(unittest.TestCase):
    tok = ByteTokenizer()

    def setUp(self):
        stop = set(self.tok.encode("<|im_end|>", parse_special=True) + self.tok.encode("<|endoftext|>", parse_special=True))
        self.vocab = C.Vocab(self.tok, stop)

    def replay(self, script, tools=False, armed=True, rf=FMT):
        c = C.Constraint(self.vocab, C.grammar(rf, tools, self.vocab), armed)
        ids = self.tok.encode(script, parse_special=True) + self.tok.encode("<|im_end|>", parse_special=True)
        out = list(c.mock_replay(ids, 400, threading.Event()))
        return c, self.tok.decode([t for t in out if t not in self.vocab.eos])

    def test_prose_and_wrong_types_cannot_get_out(self):
        for s, want in (('{"n": 3}', {"n": 3}), ('Sure! {"n": 3} hope that helps', {"n": 3}),
                        ('{"n": "3"}', {"n": 3}), ('{"n": 3, "extra": true}', {"n": 3})):
            with self.subTest(s=s):
                c, text = self.replay(s)
                self.assertEqual(json.loads(text), want)
                self.assertIsNone(c.failed)

    def test_tool_call_only_with_tools(self):
        call = CALL[CALL.index("<tool_call>"):]
        self.assertEqual(self.replay(call, tools=True)[1], call)
        self.assertNotIn("<tool_call>", self.replay(call, tools=False)[1])

    def test_thinking_is_free_and_the_window_is_cut_at_its_end(self):
        c, text = self.replay('plan {x}</think>\n\n{"n": 5}', armed=False)
        self.assertTrue(text.startswith("plan {x}</think>"))
        self.assertEqual(json.loads(text.split("</think>")[1]), {"n": 5})
        c = C.Constraint(self.vocab, C.grammar(FMT, False, self.vocab), armed=False)
        self.assertEqual(c.reply(), "MF cut=" + ",".join(map(str, self.vocab.cut)))
        for t in self.tok.encode("x</think>"):
            c.feed(t)
        reply = c.reply()
        self.assertTrue(reply.startswith("MK "))
        bits = base64.b64decode(reply[3:])
        self.assertEqual({v for v in range(self.vocab.n) if C.Constraint.allowed(bits, v)},
                         {ord(ch) for ch in " \t\r\n{"})

    def test_a_token_outside_the_mask_gives_the_mask_up(self):
        c = C.Constraint(self.vocab, C.grammar(FMT, False, self.vocab), armed=True)
        c.feed(ord("x"))                                     # an engine that ignored the mask
        self.assertIsNotNone(c.failed)
        self.assertEqual(c.reply(), "MF")

    def test_formats(self):
        self.assertIsNone(C.format_of({}))
        self.assertIsNone(C.format_of({"text": {"format": {"type": "text"}}}))
        self.assertEqual(C.format_of({"text": {"format": TEXT_FMT}})["json_schema"]["schema"], SCHEMA)
        self.assertEqual(json.loads(self.replay('ok {"a": [1]}', rf={"type": "json_object"})[1]), {"a": [1]})


# ------------------------------------------------------------------------------------------------ over HTTP
class Http(unittest.TestCase):
    tok = ByteTokenizer()

    def start(self, engine):
        self.engine = engine
        self.svc = Service(engine, self.tok, TEMPLATE)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            r = urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as e:
            r = e
        with r:
            return r.status, json.loads(r.read().decode())

    def chat(self, **kw):
        body = {"messages": [{"role": "user", "content": "a number"}], "response_format": FMT,
                "reasoning_effort": "none", **kw}
        return self.post("/v1/chat/completions", body)


@need
class MockUnderMask(Http):
    def setUp(self):
        engine = MockEngine(self.tok, 'Sure! {"n": "7"} done', max_context=16384)
        engine.supports_mask = True
        self.start(engine)

    def test_prose_around_the_json_becomes_the_schema(self):
        code, r = self.chat()
        self.assertEqual(code, 200, r)
        self.assertEqual(json.loads(r["choices"][0]["message"]["content"]), {"n": 7})
        with urllib.request.urlopen(self.base + "/v1/status", timeout=10) as s:
            self.assertTrue(json.loads(s.read())["structured_output"]["constrained_decoding"])

    def test_without_the_mask_the_same_script_is_a_502(self):
        self.engine.supports_mask = False
        code, r = self.chat()
        self.assertEqual((code, r["error"]["code"]), (502, "structured_output_failed"))

    def test_thinking_stays_free(self):
        self.engine.scripts = [self.tok.encode('Plan {x}.</think>\n\nSure {"n": 1}<|im_end|>', parse_special=True)]
        self.engine.script = self.engine.scripts[0]
        code, r = self.chat(reasoning_effort="high")
        self.assertEqual(code, 200, r)
        msg = r["choices"][0]["message"]
        self.assertIn("Plan {x}.", msg.get("reasoning_content") or "")
        self.assertEqual(json.loads(msg["content"]), {"n": 1})

    def test_responses_tool_turn_then_masked_answer(self):
        end = self.tok.encode("<|im_end|>", parse_special=True)
        self.engine.scripts = [self.tok.encode(CALL, parse_special=True) + end,
                               self.tok.encode('Read it.\n</think>\n\nHere: {"n": "3"}', parse_special=True) + end]
        self.engine.script, self.engine.turns = self.engine.scripts[0], 0
        user = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "count"}]}
        body = {"model": "m", "input": [user], "tools": TOOLS, "text": {"format": TEXT_FMT}, "store": False}
        code, r = self.post("/v1/responses", body)
        self.assertEqual(code, 200, r)
        fc = [o for o in r["output"] if o["type"] == "function_call"]
        self.assertEqual(len(fc), 1)
        code, r = self.post("/v1/responses", {**body, "input": [user, *r["output"], {
            "type": "function_call_output", "call_id": fc[0]["call_id"], "output": "3 lines"}]})
        self.assertEqual(code, 200, r)
        self.assertEqual(r["output"][-1]["content"][0]["text"], '{"n":3}')


# ------------------------------------------------------------------------------------------------ the protocol
@need
@unittest.skipIf(os.name == "nt", "the fake engine is started through a POSIX shell script")
class FakeEngine(Http):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        d = Path(self.dir.name)
        self.script, self.log = d / "script.txt", d / "mq.log"
        self.script.write_text('Sure! {"n": "7"} done', encoding="utf-8")
        os.environ["FAKE_SCRIPT_FILE"], os.environ["FAKE_LOG"] = str(self.script), str(self.log)
        self.exe = d / "strata"
        self.exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{ROOT / "serve/fake_mask_engine.py"}" "$@"\n')
        self.exe.chmod(self.exe.stat().st_mode | stat.S_IEXEC)

    def tearDown(self):
        super().tearDown()
        self.engine.close()
        for k in ("FAKE_SCRIPT_FILE", "FAKE_LOG"):
            os.environ.pop(k, None)
        self.dir.cleanup()

    def test_the_engine_takes_the_masks(self):
        self.start(StrataEngine(str(self.exe), ["--max-context", "8192"]))
        self.assertTrue(self.engine.supports_mask)
        code, r = self.chat()
        self.assertEqual(code, 200, r)
        self.assertEqual(json.loads(r["choices"][0]["message"]["content"]), {"n": 7})
        answers = self.log.read_text().split()
        self.assertGreater(answers.count("MK"), 0)

    def test_an_older_engine_is_unchanged(self):
        self.start(StrataEngine(str(self.exe), ["--max-context", "8192", "--no-mask"]))
        self.assertFalse(self.engine.supports_mask)
        code, r = self.chat()
        self.assertEqual((code, r["error"]["code"]), (502, "structured_output_failed"))
        self.assertFalse(self.log.exists())

    def test_a_stop_while_the_engine_waits_for_a_mask(self):
        self.start(StrataEngine(str(self.exe), ["--max-context", "8192"]))
        self.script.write_text('{"n": 123456789}', encoding="utf-8")
        sampling = self.svc._with_constraint({"response_format": FMT}, None, False)
        gen = self.engine.generate(self.tok.encode("hi"), 100, sampling, threading.Event())
        self.assertEqual([next(gen), next(gen)], [ord("{"), ord('"')])
        gen.close()                                          # STOP; the engine's next MQ is answered MF, then DONE
        self.script.write_text('{"n": 4}', encoding="utf-8")
        code, r = self.chat()
        self.assertEqual(code, 200, r)
        self.assertEqual(json.loads(r["choices"][0]["message"]["content"]), {"n": 4})


if __name__ == "__main__":
    unittest.main()
