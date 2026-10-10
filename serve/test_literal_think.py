"""Token identity and bounded literal-tag continuations, without a model or GPU."""
import threading
import unittest
import json
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve


class ThinkTokenizer(ByteTokenizer):
    SPECIALS = ByteTokenizer.SPECIALS + ["<think>", "</think>"]
    ALWAYS = ("<think>", "</think>")

    def token_bytes(self, t):
        return self.SPECIALS[t - 256].encode() if t >= 256 else bytes([t])


class ScriptEngine(MockEngine):
    def __init__(self, tok, scripts):
        super().__init__(tok, "", max_context=4096)
        self.scripts, self.prompts = scripts, []
        self.cancel_on_close = False

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        script = self.scripts[min(len(self.prompts), len(self.scripts) - 1)]
        self.prompts.append(list(ids))
        try:
            for t in script[:max_new]:
                if cancel.is_set():
                    return
                yield t
        finally:
            if self.cancel_on_close:
                cancel.set()


class LiteralThink(unittest.TestCase):
    def setUp(self):
        self.tok = ThinkTokenizer()
        self.end = self.tok.encode("</think>", parse_special=True)
        self.stop = self.tok.encode("<|im_end|>", parse_special=True)

    def plain(self, text):
        return self.tok.encode(text, plain=[(0, len(text))])

    def run_script(self, scripts, thinking=True, guard=False, max_new=512, sampling=None, cancel_on_close=False):
        engine = ScriptEngine(self.tok, scripts)
        engine.cancel_on_close = cancel_on_close
        svc = Service(engine, self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        svc.literal_think_guard = guard
        rows = list(svc.run([65], thinking, [], max_new, sampling or {}, threading.Event()))
        text = {k: "".join(ev.text or "" for kind, ev in rows if kind == "event" and ev.kind == k)
                for k in ("reasoning", "content")}
        return text, rows[-1][1], engine

    def test_ordinary_tag_does_not_close_reasoning(self):
        text, done, _ = self.run_script([self.plain("Read `</think>` as text. ") + self.end + self.plain("42") + self.stop])
        self.assertEqual(text, {"reasoning": "Read `</think>` as text. ", "content": "42"})
        self.assertEqual(done["finish"], "stop")

    def test_literal_tag_in_prose_does_not_route_following_reasoning_to_answer(self):
        reasoning = "The literal </think> tag can appear in prose before the explanation continues. "
        text, _, _ = self.run_script([self.plain(reasoning) + self.end + self.plain("42") + self.stop])
        self.assertEqual(text, {"reasoning": reasoning, "content": "42"})

    def test_real_midline_marker_closes_reasoning(self):
        text, _, engine = self.run_script([self.plain("Done.") + self.end + self.plain("42") + self.stop], guard=True)
        self.assertEqual(text, {"reasoning": "Done.", "content": "42"})
        self.assertEqual(len(engine.prompts), 1)

    def test_quoted_special_is_replaced_in_continuation(self):
        prefix = self.plain("Read `")
        text, done, engine = self.run_script([prefix + self.end + self.stop,
                                            self.plain("` as text. ") + self.end + self.plain("42") + self.stop], guard=True)
        self.assertEqual(text, {"reasoning": "Read `</think>` as text. ", "content": "42"})
        self.assertEqual(engine.prompts[1], [65] + prefix + self.plain("</think>"))
        self.assertEqual(done["finish"], "stop")

    def test_content_special_is_literal(self):
        text, _, engine = self.run_script([self.plain("Tag: ") + self.end + self.stop,
                                          self.plain(" done") + self.stop], thinking=False, guard=True)
        self.assertEqual(text["content"], "Tag: </think> done")
        self.assertEqual(len(engine.prompts), 2)

    def test_guard_off_does_not_change_special_marker(self):
        text, _, engine = self.run_script([self.plain("Read `") + self.end + self.stop])
        self.assertEqual(text["reasoning"], "Read `")
        self.assertEqual(len(engine.prompts), 1)

    def test_guard_off_does_not_continue_after_stop_in_code(self):
        text, done, engine = self.run_script([self.plain("Read `") + self.stop], guard=False)
        self.assertEqual(text, {"reasoning": "Read `", "content": ""})
        self.assertEqual(done["finish"], "stop")
        self.assertEqual(len(engine.prompts), 1)

    def test_guard_continues_after_eos_in_unclosed_reasoning_code(self):
        text, done, engine = self.run_script(
            [self.plain("Read `") + self.stop,
             self.plain("` as literal text. Done.") + self.end + self.plain("42") + self.stop], guard=True)
        self.assertEqual(text, {"reasoning": "Read `</think>` as literal text. Done.", "content": "42"})
        self.assertEqual(done["finish"], "stop")
        self.assertEqual(len(engine.prompts), 2)

    def test_fenced_special_is_replaced(self):
        text, _, engine = self.run_script([self.plain("```xml\n") + self.end,
                                          self.plain("\n```\nDone.") + self.end + self.plain("42") + self.stop], guard=True)
        self.assertIn("```xml\n</think>\n```", text["reasoning"])
        self.assertEqual(text["content"], "42")
        self.assertEqual(len(engine.prompts), 2)

    def test_client_stop_during_replacement_does_not_resume(self):
        text, done, engine = self.run_script([self.end], thinking=False, guard=True, sampling={"stop": "</th"})
        self.assertEqual(text["content"], "")
        self.assertEqual(done["finish"], "stop")
        self.assertEqual(len(engine.prompts), 1)

    def test_no_room_for_replacement_does_not_resume(self):
        _, done, engine = self.run_script([self.plain("`") + self.end], guard=True, max_new=2)
        self.assertEqual(done["completion_tokens"], 2)
        self.assertEqual(len(engine.prompts), 1)

    def test_substitution_limit(self):
        _, done, engine = self.run_script([self.end + self.stop], thinking=False, guard=True, max_new=2048)
        self.assertEqual(len(engine.prompts), 65)
        self.assertLessEqual(done["completion_tokens"], 2048)

    def test_partial_utf8_before_real_marker_is_preserved(self):
        text, _, _ = self.run_script([[0xE2] + self.end + self.plain("42") + self.stop])
        self.assertEqual(text, {"reasoning": "\ufffd", "content": "42"})

    def test_terminal_real_marker_closes_without_newline(self):
        text, done, _ = self.run_script([self.plain("Done.") + self.end + self.stop], guard=True)
        self.assertEqual(text, {"reasoning": "Done.", "content": ""})
        self.assertEqual(done["finish"], "stop")

    def test_cancel_while_draining_does_not_resume(self):
        _, done, engine = self.run_script([self.plain("`") + self.end], guard=True, cancel_on_close=True)
        self.assertEqual(len(engine.prompts), 1)
        self.assertEqual(done["finish"], "cancel")

    def test_replacement_then_budget_keeps_rewritten_prefix(self):
        text, _, engine = self.run_script([self.plain("`") + self.end, self.plain("` more"),
                                          self.plain("42") + self.stop], guard=True,
                                         sampling={"reasoning_budget_tokens": 5})
        self.assertEqual(len(engine.prompts), 3)
        self.assertEqual(engine.prompts[2][:10], [65] + self.plain("`</think>"))
        self.assertEqual(text["content"], "42")


class LiteralThinkHTTP(unittest.TestCase):
    """The client sees the same complete reasoning and answer in both API formats."""
    REASONING = "Read `</think>` as text. Done."

    def setUp(self):
        self.tok = ThinkTokenizer()
        self.plain = lambda text: self.tok.encode(text, plain=[(0, len(text))])
        self.end = self.tok.encode("</think>", parse_special=True)
        self.stop = self.tok.encode("<|im_end|>", parse_special=True)
        self.engine = ScriptEngine(self.tok, [self.plain(self.REASONING) + self.end + self.plain("42") + self.stop])
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, stream):
        body = {"model": "m", "messages": [{"role": "user", "content": "Explain the closing tag."}],
                "max_tokens": 512, "stream": stream}
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=30) as response:
            self.assertEqual(response.status, 200)
            raw = response.read().decode()
        return raw if stream else json.loads(raw)

    def assert_openai(self, stream):
        result = self.request("/v1/chat/completions", stream)
        if stream:
            self.assertTrue(result.rstrip().endswith("data: [DONE]"))
            chunks = [json.loads(line[6:]) for line in result.splitlines() if line.startswith("data: {")]
            choices = [chunk["choices"][0] for chunk in chunks if chunk.get("choices")]
            reasoning = "".join(c["delta"].get("reasoning_content", "") for c in choices)
            content = "".join(c["delta"].get("content", "") for c in choices)
            self.assertEqual([c["finish_reason"] for c in choices if c.get("finish_reason")], ["stop"])
        else:
            choice = result["choices"][0]
            reasoning, content = choice["message"]["reasoning_content"], choice["message"]["content"]
            self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual((reasoning, content), (self.REASONING, "42"))

    def test_openai_ordinary_tag_streaming(self):
        self.assert_openai(True)
        self.assertEqual(len(self.engine.prompts), 1)

    def test_openai_ordinary_tag_nonstreaming(self):
        self.assert_openai(False)
        self.assertEqual(len(self.engine.prompts), 1)

    def test_openai_quoted_special_repair_both_formats(self):
        self.svc.literal_think_guard = True
        scripts = [self.plain("Read `") + self.end + self.stop,
                   self.plain("` as text. Done.") + self.end + self.plain("42") + self.stop]
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.engine.scripts = scripts
                self.engine.prompts.clear()
                self.assert_openai(stream)
                self.assertEqual(len(self.engine.prompts), 2)

    def test_anthropic_ordinary_tag_streaming(self):
        raw = self.request("/v1/messages", True)
        events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
        deltas = [ev["delta"] for ev in events if ev["type"] == "content_block_delta"]
        self.assertEqual("".join(d.get("thinking", "") for d in deltas), self.REASONING)
        self.assertEqual("".join(d.get("text", "") for d in deltas), "42")
        self.assertEqual([ev["delta"]["stop_reason"] for ev in events if ev["type"] == "message_delta"], ["end_turn"])
        self.assertEqual(events[-1]["type"], "message_stop")
        self.assertEqual(len(self.engine.prompts), 1)


if __name__ == "__main__":
    unittest.main()
