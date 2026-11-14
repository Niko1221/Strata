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
from serve import frontend as FE
from serve import responses as R
from serve.frontend import ChatTemplate
from serve.server import IM_END, ByteTokenizer, MockEngine, Service, StrataEngine, serve

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


# --- T3d: the mock tokenizer with <tool_call> as ONE special token, as the real Qwen vocabulary has it
# (the server's ByteTokenizer stand-in has no tag token, so vocab.single(TOOL_CALL) -> None there and a
# name-enumerated call can never feed the tag as one token; appended last, existing ids are unchanged).
class TagTokenizer(ByteTokenizer):
    SPECIALS = ByteTokenizer.SPECIALS + ["<tool_call>"]


# ------------------------------------------------------------------------------------------------ the masks
@need
class Masks(unittest.TestCase):
    tok = ByteTokenizer()

    def setUp(self):
        stop = set(self.tok.encode("<|im_end|>", parse_special=True) + self.tok.encode("<|endoftext|>", parse_special=True))
        self.stop = stop                                  # saved for enum_setup (additive, no behavior change)
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

    # --- the MTP window lookahead (1-1 / 1-2): MQ d... -> MKN rows, state restored ---
    def draft_reply(self, script, drafts):
        """One Constraint fed `script` (the tokens accepted so far), then asked with `drafts` -> (constraint, reply)."""
        c = C.Constraint(self.vocab, C.grammar(FMT, False, self.vocab), armed=True)
        for t in self.tok.encode(script, parse_special=True):
            c.feed(t)
        return c, c.reply(drafts)

    def db(self, s):
        return self.tok.encode(s, parse_special=True)

    def test_1_state_matches_after_the_lookahead_rolls_back(self):
        for script, drafts in (("", self.db('{"n": 3}')), ('{"n"', self.db(": 3}")),
                               ('{"n": 12', self.db("34}"))):
            with self.subTest(script=script):
                c, _ = self.draft_reply(script, drafts)
                before = c.m.compute_bitmask()
                c.reply(drafts)                              # the lookahead consumes, then rolls back
                self.assertEqual(c.m.compute_bitmask(), before)
                self.assertIsNone(c.failed)

    def test_2_all_drafts_match_gives_one_row_per_token(self):
        c, reply = self.draft_reply('{"n"', self.db(": 3}"))
        self.assertTrue(reply.startswith("MKN "), reply[:24])
        rows = int(reply.split()[1])
        self.assertEqual(rows, len(self.db(": 3}")) + 1)
        self.assertEqual(len(reply.split()) - 2, rows)       # one b64 word list per row
        self.assertIsNone(c.failed)

    def test_3_draft_that_leaves_the_grammar_stops_at_k_plus_1_rows(self):
        drafts = self.db(' "zz"')                            # the string value breaks the schema (n is an integer)
        c, reply = self.draft_reply('{"n":', drafts)
        self.assertTrue(reply.startswith("MKN "), reply[:24])
        rows = int(reply.split()[1])
        self.assertEqual(rows, 2)                            # k = 1 validated draft (' ' consumes, '"' is refused: bit 0 in the mask, validate_tokens also returns 1), so k+1 = 2 rows
        self.assertEqual(len(reply.split()) - 2, rows)       # no row past the token that leaves the grammar
        self.assertIsNone(c.failed)                          # a bad draft is not a failed turn
        self.assertEqual(c.m.compute_bitmask(), c.mask())    # the state is still the head's

    def test_4_draft_wrong_at_the_head_gives_a_single_MK_row(self):
        c, reply = self.draft_reply('{"n"', self.db("3}"))   # after '"' the grammar wants ':', not '3'
        self.assertTrue(reply.startswith("MK "), reply[:24]) # one row keeps the current MK form (1-3)
        self.assertFalse(reply.startswith("MKN"))
        self.assertIsNone(c.failed)

    def test_6_rows_stop_when_the_grammar_is_finished(self):
        drafts = self.db('3}"')                              # closes the object; anything after is outside the grammar
        c, reply = self.draft_reply('{"n": ', drafts)
        self.assertTrue(reply.startswith("MKN "), reply[:24])
        rows = int(reply.split()[1])
        self.assertEqual(rows, 2)                            # head + '3'; the row after '}' is cut (is_stopped)
        self.assertEqual(len(reply.split()) - 2, rows)
        for t in drafts[:2]:
            self.assertTrue(c.m.consume_tokens([t]))
        self.assertTrue(c.m.is_stopped())                    # the closing brace ends the grammar
        self.assertIsNone(c.failed)

    def test_7_an_old_engine_gets_the_unchanged_single_MK_row(self):
        c = C.Constraint(self.vocab, C.grammar(FMT, False, self.vocab), armed=True)
        for t in self.tok.encode('{"n"', parse_special=True):
            c.feed(t)
        reply = c.reply()                                    # an engine that knows only MK sends no drafts
        self.assertTrue(reply.startswith("MK "), reply[:24])
        self.assertFalse(reply.startswith("MKN"))
        self.assertEqual(len(reply.split()), 2)              # "MK" + one b64 word - the answer form is unchanged
        self.assertEqual(len(base64.b64decode(reply[3:])), len(c.m.compute_bitmask()))
        self.assertIsNone(c.failed)

    def test_8_after_giving_the_mask_up_a_draft_window_is_MF(self):
        c = C.Constraint(self.vocab, C.grammar(FMT, False, self.vocab), armed=True)
        c.feed(ord("x"))                                     # outside the mask -> failed ("give up")
        self.assertIsNotNone(c.failed)
        self.assertEqual(c.reply(self.db('{"n": 3}')), "MF") # even with drafts: no MK, no MKN
        c2 = C.Constraint(self.vocab, C.grammar(FMT, True, self.vocab), armed=True)
        if c2.tool_call is not None:                         # only when the tokenizer has one token for the tag
            c2.feed(c2.tool_call)                            # a tool call also gives the mask up
            self.assertIsNotNone(c2.failed)
            self.assertEqual(c2.reply(self.db("cat")), "MF")

    def test_a_token_outside_the_mask_gives_the_mask_up(self):
        c = C.Constraint(self.vocab, C.grammar(FMT, False, self.vocab), armed=True)
        c.feed(ord("x"))                                     # an engine that ignored the mask
        self.assertIsNotNone(c.failed)
        self.assertEqual(c.reply(), "MF")

    # --- T3b: the name enum (T3a opt-in, STRATA_TOOL_NAME_ENUM=1) ---
    def enum_setup(self, names):
        """A name-enumerated constraint over TOOLS' declared names + `names`, as the server builds it
        when STRATA_TOOL_NAME_ENUM=1 (serve/server.py: grammar(..., tool_names=names), name_enum=True).
        Built on a TagTokenizer vocab (the real Qwen vocabulary has <tool_call> as ONE special token;
        the plain ByteTokenizer stand-in does not, so c.tool_call would be None there).  The class
        attribute self.vocab stays the plain ByteTokenizer one for every other test."""
        tool_names = C.safe_tool_names(TOOLS) + [n for n in names if n not in C.safe_tool_names(TOOLS)]
        vt = C.Vocab(TagTokenizer(), self.stop)
        c = C.Constraint(vt, C.grammar(FMT, True, vt, tool_names=tool_names),
                         armed=True, name_enum=True)
        return tool_names, c, vt

    def feed_text(self, c, text, vocab=None):
        for t in (vocab or self.vocab).tok.encode(text, parse_special=True):
            c.feed(t)

    def allowed_set(self, c, vocab=None):
        vocab = vocab or self.vocab
        bits = c.mask()
        self.assertIsNotNone(bits)                           # the mask is still live here
        return {v for v in range(vocab.n) if C.Constraint.allowed(bits, v)}

    def test_the_name_enum_lists_the_declared_names(self):
        tool_names, c, vt = self.enum_setup(["read_file"])
        self.assertEqual(tool_names, ["exec_command", "read_file"])
        self.assertIsNotNone(c.tool_call)                    # the TagTokenizer writes the tag as one special token
        c.feed(c.tool_call)                                  # the tag sits in the grammar under name_enum
        self.assertIsNone(c.failed)                          # the tag alone no longer frees the turn (T3a)
        self.assertTrue(c.await_name)
        after_tag = self.allowed_set(c, vt)
        # the mask has shrunk to the declared names' first bytes (e / r) plus the optional WS? run
        self.assertEqual({vt.bytes_of(t) for t in after_tag},
                         {b"e", b"r", b" ", b"\t", b"\r", b"\n"})
        self.feed_text(c, "exec", vt)                        # mid-name: the mask forces the continuation byte of the longest name
        self.assertIsNone(c.failed)
        self.assertTrue(c.await_name)
        self.assertEqual({vt.bytes_of(t) for t in self.allowed_set(c, vt)}, {b"_"})

    def test_the_rest_after_the_name_stays_free(self):
        """Once the enumerated name is behind the matcher, the separator frees the turn: the mask opens,
        failed='tool call', and a call with no closing tag still finishes - the parser reads the body as
        it always did.  (On a byte vocabulary REST can never be fully free - the invalid single bytes stay
        masked forever - so the release fires at the separator, one token later than _all_free did.)"""
        tool_names, c, vt = self.enum_setup(["read_file"])
        self.assertIsNotNone(c.tool_call)
        c.feed(c.tool_call)                                  # the tag in the grammar (name_enum keeps it armed)
        self.assertIsNone(c.failed)                          # the tag alone does not free the turn (T3a)
        self.assertTrue(c.await_name)                        # the name is masked next
        self.feed_text(c, "exec_command\n", vt)              # the declared name + the separator the NAME rule requires
        self.assertEqual(c.failed, "tool call")              # the separator freed the turn (byte vocab: REST never fully free)
        self.assertFalse(c.await_name)
        self.assertIsNone(c.mask())                          # free for the rest of the turn
        self.feed_text(c, "\n<parameter=cmd>ls -la")         # the body, closing tags missing - all free
        self.assertEqual(c.reply(), "MF cut=" + ",".join(map(str, vt.cut)))  # free for the rest of the turn (test_8 form): no 502 path

    def test_formats(self):
        self.assertIsNone(C.format_of({}))
        self.assertIsNone(C.format_of({"text": {"format": {"type": "text"}}}))
        self.assertEqual(C.format_of({"text": {"format": TEXT_FMT}})["json_schema"]["schema"], SCHEMA)
        self.assertEqual(json.loads(self.replay('ok {"a": [1]}', rf={"type": "json_object"})[1]), {"a": [1]})

    # --- T1 (spec item 1): a lark response_format goes through grammar() to the matcher unchanged ---
    def test_lark_grammar_builds_and_masks(self):
        rf = {"type": "lark", "grammar": 'start: "yes" | "no"'}
        self.assertEqual(C.format_of({"response_format": rf}), rf)      # accepted unchanged (the same dict)
        vt = C.Vocab(TagTokenizer(), self.stop)                         # the T3d fixture vocabulary
        g = C.grammar(rf, False, vt)                                    # the lark text straight through
        c = C.Constraint(vt, g, armed=True)
        self.assertIsNone(c.failed)
        reply = c.reply()
        self.assertTrue(reply.startswith("MK "), reply[:24])            # masked from the first token
        bits = base64.b64decode(reply[3:])
        self.assertEqual({vt.bytes_of(t) for t in range(vt.n) if C.Constraint.allowed(bits, t)},
                         {b"y", b"n"})                                  # only the grammar's first letters
        self.feed_text(c, "yes", vt)                                    # the word consumes to the end
        self.assertIsNone(c.failed)
        c2, text = self.replay("sure: yes", rf=rf)                      # prose outside the grammar is dropped
        self.assertEqual(text, "yes")
        self.assertIsNone(c2.failed)
        with self.assertRaises(ValueError):                             # grammar()'s own guard
            C.grammar({"type": "lark"}, False, vt)                      # lark without a grammar string


# ------------------------------------------------------------------------ the lark response_format (spec item 1)
class LarkFormat(unittest.TestCase):
    """format_of's lark acceptance: a well-formed {type: "lark", grammar: "..."} passes through unchanged;
    a malformed lark is treated exactly like an unknown type (None).  Pure dict logic - runs without
    llguidance."""

    def test_lark_is_accepted_as_is(self):
        rf = {"type": "lark", "grammar": 'start: "a"'}
        self.assertEqual(C.format_of({"response_format": rf}), rf)

    def test_bad_lark_matches_the_unknown_type(self):
        for rf in ({"type": "lark"},                                  # no grammar at all
                   {"type": "lark", "grammar": ""},                   # empty string
                   {"type": "lark", "grammar": 5},                    # not a string
                   {"type": "lark", "grammar": None},
                   {"type": "lark", "grammar": ["start: \"a\""]},     # list, not string
                   {"type": "no-such-format"}):                       # the existing unknown type
            with self.subTest(rf=rf):
                self.assertIsNone(C.format_of({"response_format": rf}))


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

    # --- the same tool turn, but the answer comes out through the MTP window (MockUnderMask 拡張, 6b-1) ---
    def test_9_multi_row_tool_turn_then_masked_answer(self):
        engine = MTPWindowMock(self.tok, [CALL, 'Read it.\n</think>\n\nHere: {"n": "3"}'], max_context=16384)
        engine.supports_mask = True
        self.start(engine)
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
        self.assertTrue(any(rep.startswith("MKN") for rep in engine.mask_log), engine.mask_log[:4])
        obj, _ = engine.answer_json(self.tok)
        self.assertEqual(obj, {"n": 3})

    # --- 6b-2: 採用→feed の順序 — the server's state always matches the tokens that actually came out ---
    def test_10_the_servers_state_matches_the_emitted_tokens(self):
        engine = MTPWindowMock(self.tok, 'Sure! {"n": 7} done', max_context=16384)
        engine.supports_mask = True
        self.start(engine)
        sampling = self.svc._with_constraint({"response_format": FMT}, None, False)
        c = sampling[C.KEY]
        emitted = list(engine.generate(self.tok.encode("hi"), 200, sampling, threading.Event()))
        self.assertIsNone(c.failed)                              # every fed token was inside the grammar
        self.assertTrue(c.m.is_stopped())                        # the state has consumed exactly the emitted JSON
        obj, json_toks = engine.answer_json(self.tok)            # the emitted tokens up to the closing brace
        self.assertEqual(obj, {"n": 7})
        c2 = C.Constraint(c.vocab, C.grammar(FMT, False, c.vocab), armed=True)
        for t in json_toks:                                      # the same tokens replayed into a fresh matcher
            c2.feed(t)                                           # land on the same bitmask: state == emitted sequence
            self.assertIsNone(c2.failed)
        self.assertEqual(c2.m.compute_bitmask(), c.m.compute_bitmask())

    # --- 6b-3: ドラフトが全部外れ続けても (rows=1 の連続) the output is still right ---
    def test_11_drafts_that_always_miss_still_give_the_right_answer(self):
        engine = MTPWindowMock(self.tok, 'Sure! {"n": "7"} done', max_context=16384)
        engine.supports_mask = True
        engine.reject = ord("}")                                 # '}' never leads a valid draft here -> every window is 1 row
        self.start(engine)
        code, r = self.chat()
        self.assertEqual(code, 200, r)
        self.assertEqual(json.loads(r["choices"][0]["message"]["content"]), {"n": 7})
        self.assertGreater(len(engine.mask_log), 1)              # many windows, not one lucky lookahead
        self.assertFalse(any(rep.startswith("MKN") for rep in engine.mask_log), engine.mask_log[:4])
        self.assertTrue(all(rep.startswith("MK ") for rep in engine.mask_log if rep.startswith(("MK ", "MKN"))))

    # --- T1 (spec item 1): a lark grammar llguidance cannot compile degrades to free decoding ---
    def test_12_a_broken_lark_grammar_degrades_to_free_decoding(self):
        """The _with_constraint fallback rides a broken lark the way T3b's ChatToolsWithJsonSchema run 3
        rides a broken name-enum: the sampling carries no constraint and the script comes out verbatim
        (free decoding - the 200 path, never a 502).  Exercised at the Service level (the test_10 way):
        the HTTP route end-to-end needs serve/structured.py prepare_format to accept the lark type,
        whose allowlist 400s every non-JSON type today - outside this card's edit scope (see the task
        comment)."""
        sampling = self.svc._with_constraint(
            {"response_format": {"type": "lark", "grammar": "start: [[["}}, None, False)
        self.assertNotIn(C.KEY, sampling)                              # the constraint was given up
        emitted = list(self.engine.generate(self.tok.encode("hi"), 200, sampling, threading.Event()))
        self.assertEqual(self.tok.decode(emitted), 'Sure! {"n": "7"} done' + IM_END)  # free, verbatim


class MTPWindowMock(MockEngine):
    """MockEngine extended with the MTP verify window (token_mask=2): every window it asks the server with
    `MQ d1 d2 ...` (the next draft_n script tokens as drafts), reads the answer (MK / MKN <rows> ...) and feeds
    the window the way the real engine does - the validated drafts (rows-1, checked against the row masks with a
    matcher of its own), then its own pick under the last row.  A bare `MK` is a window where the head rejected
    the drafts: only the head's token is fed.  mask_log keeps every answer the server sent, emitted the token
    sequence the engine produced, in order."""

    def __init__(self, tokenizer, script, max_context: int = 32768, draft_n: int = 4):
        super().__init__(tokenizer, script, max_context)
        self.draft_n = draft_n
        self.reject = None                                  # when set, this token leads every draft (always rejected)
        self.mask_log: list[str] = []
        self.emitted: list[int] = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_prompt = list(ids)
        self.last_embeddings = embeddings
        if len(self.scripts) > 1:
            self.script = self.scripts[min(self.turns, len(self.scripts) - 1)]
            self.turns += 1
        c = (sampling or {}).get(C.KEY) if getattr(self, "supports_mask", False) else None
        if c is None:
            yield from super().generate(ids, max_new, sampling, cancel, embeddings)
            return
        i, out = 0, 0

        def emit(t):
            self.emitted.append(t)
            return t

        while out < max_new and not cancel.is_set():
            drafts = list(self.script[i:i + self.draft_n])
            if self.reject is not None:
                drafts = [self.reject] + drafts             # a draft head the grammar never accepts
            rep = c.reply(drafts=drafts or None)
            self.mask_log.append(rep)
            # rows-1 = the drafts the server's rows validated (1-2); a bare "MK" or "MF" is a one-token window
            f = rep.split()
            rows = int(f[1]) if rep.startswith("MKN") else 1
            for _ in range(rows - 1):                       # adopt the validated drafts, in feed order
                if i >= len(self.script):
                    break
                t = self.script[i]
                i += 1
                c.feed(t)
                out += 1
                yield emit(t)
                if t in c.vocab.eos:
                    return
            # the head's token is picked under the window's last row (None: the window is free, "MF")
            bits = base64.b64decode(f[-1]) if rep.startswith(("MK ", "MKN ")) else None
            t = None
            while i < len(self.script):
                cand = self.script[i]
                i += 1
                if bits is None or C.Constraint.allowed(bits, cand):
                    t = cand
                    break
            if t is None:                                   # the script ran out: end the turn if the grammar may
                eos = next((e for e in c.vocab.eos if bits is None or C.Constraint.allowed(bits, e)), None)
                ff = c.m.compute_ff_tokens() if bits is not None else []
                t = eos if eos is not None else (ff[0] if ff else None)
                if t is None:
                    return                                  # free and nothing forced: the turn ends short
            c.feed(t)
            out += 1
            yield emit(t)
            if t in c.vocab.eos:
                return

    def answer_tokens(self):
        """The answer part of the emitted sequence: from the first token that opens the JSON ('{' = 123)."""
        return self.emitted[self.emitted.index(123):]

    def answer_json(self, tok):
        """The JSON object the engine emitted under the mask, parsed straight out of the token sequence
        (raw_decode stops at the closing brace, so prose after it and the end-of-turn token are ignored)."""
        obj, end = json.JSONDecoder().raw_decode(tok.decode(self.answer_tokens()))
        return obj, self.answer_tokens()[:end]


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


# ----------------------------------------------------------------- the chat route with tools + a JSON format
@need
class ChatToolsWithJsonSchema(Http):
    """T5 (plan §2 論点6): the chat route takes tools and a JSON format together while the masks are live;
    the old 400 stays where they are not; a tool-call turn passes the validator untouched."""

    def setUp(self):
        engine = MockEngine(self.tok, 'Sure: {"n": 7}', max_context=16384)
        engine.supports_mask = True
        self.start(engine)
        self._saved_enum = os.environ.get("STRATA_TOOL_NAME_ENUM")

    def tearDown(self):
        super().tearDown()
        if self._saved_enum is None:
            os.environ.pop("STRATA_TOOL_NAME_ENUM", None)
        else:
            os.environ["STRATA_TOOL_NAME_ENUM"] = self._saved_enum

    def test_an_undeclared_name_degrades_to_free_decoding(self):
        """T3b: with STRATA_TOOL_NAME_ENUM=1 a grammar problem (e.g. a name the regex cannot carry) must
        degrade the request to free decoding - the answer still comes back, never a 502."""
        os.environ["STRATA_TOOL_NAME_ENUM"] = "1"
        code, r = self.chat_tools()
        self.assertEqual(code, 200, r)                       # the fallback in _with_constraint: no 502
        self.assertEqual(json.loads(r["choices"][0]["message"]["content"]), {"n": 7})

    def chat_tools(self, **kw):
        return self.chat(tools=TOOLS, **kw)

    def test_tools_with_json_schema_is_not_400_when_the_mask_is_live(self):
        code, r = self.chat_tools()
        self.assertEqual(code, 200, r)                          # the old blanket 400 is gone under the mask
        self.assertEqual(json.loads(r["choices"][0]["message"]["content"]), {"n": 7})
        with urllib.request.urlopen(self.base + "/v1/status", timeout=10) as s:
            self.assertTrue(json.loads(s.read())["structured_output"]["tools_with_json"])

    def test_an_engine_without_the_mask_still_returns_400(self):
        self.engine.supports_mask = False
        code, r = self.chat_tools()
        self.assertEqual(code, 400, r)
        self.assertIn("not supported", r["error"]["message"])

    def test_a_tool_call_turn_passes_the_validator_untouched(self):
        self.engine.scripts = [self.tok.encode(CALL, parse_special=True) + self.tok.encode(IM_END, parse_special=True),
                               self.tok.encode('Done: {"n": 3}', parse_special=True)
                               + self.tok.encode(IM_END, parse_special=True)]
        self.engine.script, self.engine.turns = self.engine.scripts[0], 0
        code, r = self.chat_tools()
        self.assertEqual(code, 200, r)                          # the call is not validated against the schema
        msg = r["choices"][0]["message"]
        self.assertEqual(r["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual([c["function"]["name"] for c in msg["tool_calls"]], ["exec_command"])


# ----------------------------------------------------------------- the bare tool call under the mask
@need
class ChatBareToolMask(Http):
    """T6 (plan §2 論点4, STRATA_TOOL_MASK=1 opt-in): bare tool calls - no response_format - are constrained:
    the tag can only arrive through the calls branch (the name sits behind the enum, the separator frees the
    turn), prose stays possible and a prose turn ends at the EOS like an unconstrained one."""


    def setUp(self):
        engine = MockEngine(self.tok, 'I checked it directly: 7', max_context=16384)
        engine.supports_mask = True
        self.start(engine)
        self._saved = {k: os.environ.get(k) for k in ("STRATA_TOOL_MASK", "STRATA_TOOL_NAME_ENUM")}

    def tearDown(self):
        super().tearDown()
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def chat_tools(self, **kw):
        body = {"messages": [{"role": "user", "content": "run ls"}], "tools": TOOLS,
                "reasoning_effort": "none", **kw}
        return self.post("/v1/chat/completions", body)

    def test_off_by_default(self):
        code, r = self.chat_tools()
        self.assertEqual(code, 200, r)
        self.assertIsNone(self.engine.last_sampling_constraint)   # the sampling carried no constraint key

    def test_prose_answer_turn_under_the_mask(self):
        """A prose turn (no call) ends at the EOS under the bare-call mask: masked=0 for prose text."""
        os.environ["STRATA_TOOL_MASK"] = "1"
        code, r = self.chat_tools()
        self.assertEqual(code, 200, r)
        self.assertEqual(r["choices"][0]["message"]["content"], "I checked it directly: 7")

    def test_a_bare_tool_call_is_masked_and_the_enum_ignored_without_the_flag(self):
        """With STRATA_TOOL_MASK=1 alone (no STRATA_TOOL_NAME_ENUM), the tag feeds the free-REST branch."""
        os.environ["STRATA_TOOL_MASK"] = "1"
        self.engine.scripts = [self.tok.encode(CALL, parse_special=True) + self.tok.encode(IM_END, parse_special=True)]
        self.engine.script, self.engine.turns = self.engine.scripts[0], 0
        code, r = self.chat_tools()
        self.assertEqual(code, 200, r)
        msg = r["choices"][0]["message"]
        self.assertEqual(r["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual([c["function"]["name"] for c in msg["tool_calls"]], ["exec_command"])

    def test_the_enum_with_the_bare_mask_ends_at_the_separator(self):
        """STRATA_TOOL_MASK=1 + STRATA_TOOL_NAME_ENUM=1: the tag feeds the enum, the separator frees the
        turn (案1), and the call passes untouched - measured: finish_reason tool_calls, 200."""
        os.environ["STRATA_TOOL_MASK"] = "1"
        os.environ["STRATA_TOOL_NAME_ENUM"] = "1"
        self.engine.scripts = [self.tok.encode(CALL, parse_special=True) + self.tok.encode(IM_END, parse_special=True)]
        self.engine.script, self.engine.turns = self.engine.scripts[0], 0
        code, r = self.chat_tools()
        self.assertEqual(code, 200, r)
        msg = r["choices"][0]["message"]
        self.assertEqual(r["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual([c["function"]["name"] for c in msg["tool_calls"]], ["exec_command"])

    # --- the Responses route: bare tools, no text.format - the same shared generation loop
    def test_responses_bare_tool_call_under_the_mask(self):
        os.environ["STRATA_TOOL_MASK"] = "1"
        self.engine.scripts = [self.tok.encode(CALL, parse_special=True) + self.tok.encode(IM_END, parse_special=True)]
        self.engine.script, self.engine.turns = self.engine.scripts[0], 0
        body = {"model": "x", "input": [{"role": "user", "content": [{"type": "text", "text": "run ls"}]}],
                "tools": [{"type": "function", "name": "exec_command", "description": "Runs a command.",
                           "parameters": TOOLS[0]["parameters"]}],
                "stream": False}
        code, r = self.post("/v1/responses", body)
        self.assertEqual(code, 200, r)
        out = r.get("output") or []
        calls = [c for c in out if c.get("type") == "function_call"]
        self.assertEqual([c["name"] for c in calls], ["exec_command"])

    # --- the Anthropic route: bare tools, no format - the sampling carries no response_format either
    def test_anthropic_bare_tool_call_under_the_mask(self):
        os.environ["STRATA_TOOL_MASK"] = "1"
        self.engine.scripts = [self.tok.encode(CALL, parse_special=True) + self.tok.encode(IM_END, parse_special=True)]
        self.engine.script, self.engine.turns = self.engine.scripts[0], 0
        body = {"model": "x", "max_tokens": 512, "stream": False,
                "tools": [{"name": "exec_command", "description": "Runs a command.",
                           "input_schema": TOOLS[0]["parameters"]}],
                "messages": [{"role": "user", "content": "run ls"}]}
        code, r = self.post("/v1/messages", body)
        self.assertEqual(code, 200, r)
        blocks = [b for b in (r.get("content") or []) if isinstance(b, dict) and b.get("type") == "tool_use"]
        self.assertEqual([b["name"] for b in blocks], ["exec_command"])


# --------------------------------------------------------- custom tool grammars (spec item 2, cards T2a / T2c)
class _CallEndTagTokenizer(TagTokenizer):
    """The production pack's shape this design assumes: the call closer is ONE type-4 special token,
    like the call opener (T3d).  The plain fixture omits it - probe7 measured the single lookup as
    None - so the release path needs its own fixture (notes/T2d-impl.md)."""
    SPECIALS = TagTokenizer.SPECIALS + [FE.CALL_END]


@need
class CustomToolGrammars(unittest.TestCase):
    """T2d: the custom_calls route the parents wired - responses.request_tools records the grammar
    entries under CUSTOM_CALLS_KEY (T2c) and constrain.grammar()'s custom_calls branch builds the
    probe7 call grammar (T2a).  The server.py route that consumes the key is NOT wired yet (it sits
    outside every card's edit scope), so `_route` below composes the two sides the way the T4 route
    will, and the degradation test pins the never-a-502 contract the route must keep.

    Payload safety (the gateway cuts text at the tag-closer character): no tag string is written as
    a literal here - every tag is assembled from serve.frontend's own constants plus chr(62)."""

    GT = chr(62)                                       # the tag closer, built, never typed

    def setUp(self):
        self.tok = TagTokenizer()                      # <tool_call> is ONE special token (T3d)
        self.stop = set(self.tok.encode(IM_END, parse_special=True))   # the real server's stop set
        self.vocab = C.Vocab(self.tok, self.stop)

    def _entries(self, name, definition):
        return [{"name": name, "definition": definition}]

    def _constraint(self, entries):
        g = C.grammar({"type": "custom_calls", "entries": entries}, False, self.vocab)
        return C.Constraint(self.vocab, g, armed=True, custom_calls=True)

    def _call(self, name, value):
        """The full call text, assembled: CALL_START nl FUNC_START name GT nl PARAM_START input GT nl
        value nl PARAM_END nl FUNC_END nl CALL_END (the probe7 shape)."""
        gt, nl = self.GT, "\n"
        return (FE.CALL_START + nl + FE.FUNC_START + name + gt + nl
                + FE.PARAM_START + "input" + gt + nl + value + nl + FE.PARAM_END + nl
                + FE.FUNC_END + nl + FE.CALL_END)

    def _feed(self, c, text):
        for t in self.tok.encode(text, parse_special=True):
            c.feed(t)

    def _consumes(self, gtext, text):
        """The probe7 raw() check: feed the whole text into a fresh matcher; return (ok, report)."""
        g = C._llg.LLMatcher.grammar_from_lark(gtext)
        m = C._llg.LLMatcher(self.vocab.lltok, g, log_level=0)
        self.assertFalse(m.is_error(), m.get_error())
        for i, t in enumerate(self.tok.encode(text, parse_special=True)):
            if not m.consume_token(t):
                return False, f"stopped at token #{i} {self.tok.decode([t])!r}"
        return True, "full consume"

    def _route(self, req):
        """How the T4 route consumes the T2c key (the probe7 / T2a contract): the entries go to
        grammar() directly - NOT through format_of, which returns None for this type - and the
        Constraint is built with custom_calls=True.  A build failure returns None: the request keeps
        its no-constraint shape and decodes free (the server._with_constraint except-shape), never a
        502."""
        entries = req.get(R.CUSTOM_CALLS_KEY)
        if not entries:
            return None
        try:
            g = C.grammar({"type": "custom_calls", "entries": entries}, False, self.vocab)
            return C.Constraint(self.vocab, g, armed=True, custom_calls=True)
        except Exception:
            return None

    # --- 1: the grammar builds from a minimal one-tool entry list and masks from the first token ---
    def test_custom_calls_grammar_builds_and_masks(self):
        c = self._constraint(self._entries("patch", "[a-z]+=[0-9]+"))
        self.assertIsNone(c.failed)                                # armed and live from the start
        reply = c.reply()
        self.assertTrue(reply.startswith("MK "), reply[:24])       # masked from the first token
        bits = base64.b64decode(reply[3:])
        allowed = {v for v in range(self.vocab.n) if C.Constraint.allowed(bits, v)}
        self.assertIn(c.tool_call, allowed)                        # the call opens through the tag token
        lt = [t for t in range(self.vocab.n) if self.vocab.bytes_of(t) == b"<"]
        self.assertTrue(lt and lt[0] not in allowed)               # PROSE cannot start with the tag opener
        y = [t for t in range(self.vocab.n) if self.vocab.bytes_of(t) == b"y"]
        self.assertTrue(y and y[0] in allowed)                     # a prose first byte is free

    # --- 2: the closed call frees the turn on this fixture (measured 2026-10-28, notes/T2d-impl.md):
    # the fixture tokenizer holds no single token for the call closer, so the turn frees through the
    # calls rule's trailing REST - the mask stays live but every valid-UTF-8 token is allowed (the
    # invalid single bytes stay masked forever, the T3a REST finding), never refusing the call ---
    def test_closed_call_ends_free_and_reply_is_all_permission(self):
        c = self._constraint(self._entries("patch", "[a-z]+=[0-9]+"))
        self.assertIsNone(c.call_end)                  # the fixture has no single call-closer token
        self._feed(c, self._call("patch", "x=1"))
        self.assertIsNone(c.failed)                    # the call consumed cleanly to the end
        bits = c.mask()
        self.assertIsNotNone(bits)                     # the mask stays live through REST
        allowed = {v for v in range(self.vocab.n) if C.Constraint.allowed(bits, v)}
        self.assertTrue(all(v not in allowed for v in range(0x80, 0xC0)))   # invalid single bytes stay masked
        y = [t for t in range(self.vocab.n) if self.vocab.bytes_of(t) == b"y"]
        self.assertTrue(y and y[0] in allowed)         # every valid token stays free
        self.assertTrue(c.reply().startswith("MK "), c.reply()[:8])

    # --- 2b: a pack whose tokenizer holds the call closer as ONE type-4 token - T3 moved the
    # envelope tokens to vocab.literal, so the grammar consumes the closing special token through
    # the `<[id]>` reference and the release branch fires (the T2a-era marker-byte refusal of the
    # plain-text closer literal is fixed; notes/T3-impl.md) ---
    def test_a_single_call_end_pack_releases_at_the_close(self):
        tok = _CallEndTagTokenizer()
        vocab = C.Vocab(tok, set(tok.encode(IM_END, parse_special=True)))
        g = C.grammar({"type": "custom_calls", "entries": self._entries("patch", "[a-z]+=[0-9]+")}, False, vocab)
        c = C.Constraint(vocab, g, armed=True, custom_calls=True)
        self.assertIsNotNone(c.call_end)               # the release token exists on this vocabulary
        for t in tok.encode(self._call("patch", "x=1"), parse_special=True):
            c.feed(t)                                   # the VARIANT tokenizer: the closer is ONE token
        self.assertEqual(c.failed, "tool call")        # released exactly at the closing token (T2a)
        self.assertIsNone(c.mask())                    # free from there on
        self.assertTrue(c.reply().startswith("MF"), c.reply()[:16])

    def test_prose_turn_never_releases_early(self):
        c = self._constraint(self._entries("patch", "[a-z]+=[0-9]+"))
        self._feed(c, "just prose, nothing more.")
        self.assertIsNone(c.failed)                    # a prose turn keeps the mask live to the end
        self.assertIsNotNone(c.mask())

    # --- 3: prefix pair - the shorter name is excluded, the longer one passes ---
    def test_prefix_pair_keeps_the_longer_name(self):
        entries = [{"name": "exec_command", "definition": "ls"}, {"name": "exec", "definition": "ps"}]
        gtext, kept = C._custom_calls_grammar(entries, self.vocab)
        self.assertEqual(kept, ["exec_command"])       # the shorter prefix name is dropped
        ok_long, rep = self._consumes(gtext, self._call("exec_command", "ls"))
        self.assertTrue(ok_long, rep)                  # the longer name consumes to the end
        ok_short, _ = self._consumes(gtext, self._call("exec", "ps"))
        self.assertFalse(ok_short)                     # the shorter name is not in the grammar



    # --- 4: the value lexeme consumes exactly what the custom grammar declares: one lexeme ending at
    # the parameter closer; a definition that allows newlines takes a multi-line value in one lexeme,
    # and one that forbids them stops (probe7's measured case) ---
    def test_the_value_lexeme_consumes_what_the_definition_declares(self):
        gtext, _ = C._custom_calls_grammar(self._entries("patch", "[a-z]+=[0-9]+"), self.vocab)
        ok, rep = self._consumes(gtext, self._call("patch", "x=1"))
        self.assertTrue(ok, rep)                       # a one-line value consumes to the end
        ok2, _ = self._consumes(gtext, self._call("patch", "x=1\ny=2"))
        self.assertFalse(ok2)                          # the definition forbids the newline: probe7's stop
        nl = chr(10)
        wide, _ = C._custom_calls_grammar(self._entries("patch", "[a-z0-9" + nl + "=]+"), self.vocab)
        ok3, rep3 = self._consumes(wide, self._call("patch", "a" + nl + "b" + nl + "c"))
        self.assertTrue(ok3, rep3)                     # a newline-allowing definition: multi-line in one lexeme

    # --- 5: an entry without a usable (string) definition is dropped; with no usable entry left the
    # grammar raises and the route composition returns None - the request keeps its old shape ---
    def test_an_entry_without_a_string_definition_is_dropped(self):
        with self.assertRaises(ValueError):
            C.grammar({"type": "custom_calls", "entries": [{"name": "patch"}]}, False, self.vocab)
        with self.assertRaises(ValueError):
            C.grammar({"type": "custom_calls", "entries": [{"name": "patch", "definition": 7}]}, False, self.vocab)
        self.assertIsNone(self._route({R.CUSTOM_CALLS_KEY: [{"name": "patch"}]}))

    # --- 6: a definition llguidance cannot compile degrades to free decoding - the route composition
    # catches the build error and hands back None (never a 502; the request decodes free) ---
    def test_a_broken_definition_degrades_to_free_decoding(self):
        self.assertIsNone(self._route({R.CUSTOM_CALLS_KEY: [{"name": "patch", "definition": "start: [["}]}))



# ---------------------------------------------------------------- T3: the conditional body binding
@need
class BodyBinding(unittest.TestCase):
    """T3 (spec item 3): the call body binds only when the tokenizer holds the call opener as ONE
    type-4 token (the T3d lesson - a tokenizer that cannot produce the opener as written cannot be
    bound past it); otherwise the name enumeration stays and the body stays the free REST.  The
    envelope tokens ride vocab.literal, so the bound grammar matches both pack shapes (the T2a-era
    marker-byte refusal of the plain-text closer is gone).  Measured decisions: notes/T3-impl.md."""

    GT = chr(62)                                       # the tag closer, built, never typed

    def setUp(self):
        self.tok = TagTokenizer()                      # the opener IS one special token (T3d)
        self.stop = set(self.tok.encode(IM_END, parse_special=True))
        self.vocab = C.Vocab(self.tok, self.stop)

    def _g(self, defn, vocab=None, name="patch"):
        v = vocab or self.vocab
        return C.grammar({"type": "custom_calls", "entries": [{"name": name, "definition": defn}]}, False, v)

    def _call(self, value):
        gt, nl = self.GT, "\n"
        return (FE.CALL_START + nl + FE.FUNC_START + "patch" + gt + nl
                + FE.PARAM_START + "input" + gt + nl + value + nl + FE.PARAM_END + nl
                + FE.FUNC_END + nl + FE.CALL_END)

    def _consumes(self, vocab, gtext, text, tok=None):
        tok = tok or self.tok
        g = C._llg.LLMatcher.grammar_from_lark(gtext)
        m = C._llg.LLMatcher(vocab.lltok, g, log_level=0)
        self.assertFalse(m.is_error(), m.get_error())
        for i, t in enumerate(tok.encode(text, parse_special=True)):
            if not m.consume_token(t):
                return False, "stopped at #" + str(i)
        return True, "full consume"

    # --- 1: an opener-single vocabulary binds the body: one-line AND multi-line values FULL CONSUME ---
    def test_bound_body_consumes_the_values_the_definition_declares(self):
        g = self._g("[a-z]+=[0-9]+")
        self.assertIn("/[a-z]+=[0-9]+", g)             # the definition rides the value lexeme (bound)
        ok, rep = self._consumes(self.vocab, g, self._call("x=1"))
        self.assertTrue(ok, rep)                       # a one-line value consumes to the end
        nl = chr(10)
        wide = self._g("[a-z0-9" + nl + "=]+")
        self.assertIn("/[a-z0-9\\n=]+", wide)          # the real newline became the regex escape
        ok2, rep2 = self._consumes(self.vocab, wide, self._call("a" + nl + "b" + nl + "c"))
        self.assertTrue(ok2, rep2)                     # a multi-line value consumes in one lexeme

    # --- 2: a vocabulary without the opener as ONE token falls back: names only, the body is the
    # free REST - even a definition-violating value consumes (the bound shape would refuse it) ---
    def test_an_openerless_vocabulary_falls_back_to_the_free_body(self):
        tok = ByteTokenizer()                          # no specials for the opener: TOOL_CALL not single
        vocab = C.Vocab(tok, set(tok.encode(IM_END, parse_special=True)))
        self.assertIsNone(vocab.single(FE.CALL_START))
        g = self._g("[a-z]+=[0-9]+", vocab=vocab)
        self.assertNotIn("/[a-z]+=[0-9]+", g)          # no value lexeme: the body is unbound
        gt = self.GT
        call = (FE.CALL_START + "\n" + FE.FUNC_START + "patch" + gt + "\n"
                + FE.PARAM_START + "input" + gt + "\n" + "1=x" + "\n" + FE.PARAM_END + "\n"
                + FE.FUNC_END + "\n" + FE.CALL_END)
        ok, rep = self._consumes(vocab, g, call, tok=tok)
        self.assertTrue(ok, rep)                       # the violating value consumes: the REST is free

    # --- 3: a structured (non-string) definition drops the tool - R9 measured: only usable entries
    # reach the grammar, the rest is left to the free path after the turn's validation ---
    def test_a_structured_definition_drops_the_tool(self):
        entries = [{"name": "patch", "definition": "[a-z]+=[0-9]+"},
                   {"name": "obj_tool", "definition": {"type": "object", "properties": {}}}]
        g = C.grammar({"type": "custom_calls", "entries": entries}, False, self.vocab)
        self.assertIn('"patch"', g)                    # the usable entry is bound
        self.assertNotIn("obj_tool", g)                # the structured-definition tool is dropped (R9)

    # --- 4: the answer branch stays the probe P5 shape: prose may start with anything but the
    # tag-opener byte, and the call opens through its own special token ---
    def test_prose_starts_free_except_the_tag_opener_byte(self):
        c = C.Constraint(self.vocab, self._g("[a-z]+=[0-9]+"), armed=True, custom_calls=True)
        reply = c.reply()
        self.assertTrue(reply.startswith("MK "), reply[:24])
        bits = base64.b64decode(reply[3:])
        allowed = {v for v in range(self.vocab.n) if C.Constraint.allowed(bits, v)}
        lt = [t for t in range(self.vocab.n) if self.vocab.bytes_of(t) == b"<"]
        self.assertTrue(lt and lt[0] not in allowed)   # the tag opener cannot start the turn
        y = [t for t in range(self.vocab.n) if self.vocab.bytes_of(t) == b"y"]
        self.assertTrue(y and y[0] in allowed)         # prose starts free


if __name__ == "__main__":
    unittest.main()
