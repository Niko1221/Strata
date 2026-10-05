"""No-progress tool loop guard tests.

    python tools/test_tool_loop_guard.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate
from serve.server import (
    ByteTokenizer,
    MockEngine,
    Service,
    serve,
    tool_loop_max_repeats_from_config,
    tool_loop_notice,
)

ROOT = Path(__file__).resolve().parents[1]
TOOL = {"type": "function", "function": {"name": "exec_command", "parameters": {"type": "object"}}}
NOTICE = "Stopped by Strata no-progress guard"


class CountingEngine(MockEngine):
    def __init__(self, tok, script="ok", max_context=4096):
        super().__init__(tok, "</think>\n\n" + script, max_context=max_context)
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        yield from super().generate(*args, **kwargs)




class FakeVision:
    def __init__(self):
        self.dir = Path(tempfile.mkdtemp(prefix="strata-vision-test-"))
        self.restarts = 0
        self.encoded = 0
        self._alive = False

    def alive(self):
        return self._alive

    def restart(self):
        self.restarts += 1
        self._alive = True

    def encode(self, source):
        if not self._alive:
            raise ValueError("vision encoder down before load")
        self.encoded += 1
        path = self.dir / f"image-{self.encoded}.sve"
        path.write_bytes(b"fake")
        return path, 1

class NoRenderTemplate:
    def render(self, *args, **kwargs):
        raise AssertionError("guarded prepare must not render prompts")


class LoadingService(Service):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_calls = 0

    def load(self):
        self.load_calls += 1
        return super().load()


def pair(name="exec_command", args=None, output="same"):
    return [{"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": name, "arguments": {"cmd": "pwd"} if args is None else args}}]},
        {"role": "tool", "content": output}]


def history(n, **kw):
    msgs = [{"role": "user", "content": "go"}]
    for _ in range(n):
        msgs += pair(**kw)
    return msgs


class ToolLoopGuard(unittest.TestCase):
    def test_trips_at_eight_and_engine_is_not_called(self):
        tok = ByteTokenizer()
        engine = CountingEngine(tok)
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.tool_loop_max_repeats = 8
        ids, thinking, max_new = svc.prepare(history(8), [TOOL["function"]], {}, 20)
        out = list(svc.run(ids, thinking, [TOOL["function"]], max_new, {}, threading.Event()))
        self.assertEqual(engine.calls, 0)
        self.assertIn(NOTICE, out[0][1].text)
        self.assertEqual(out[-1], ("done", {"finish": "stop", "completion_tokens": 0, "reused": 0,
                                            "timings": None, "reasoning_tokens": 0}))

    def test_below_threshold_and_default_zero_do_not_stop(self):
        self.assertIsNone(tool_loop_notice(history(7), 8))
        self.assertIsNone(tool_loop_notice(history(8), 0))

    def test_changed_args_results_or_exit_codes_do_not_stop(self):
        self.assertIsNone(tool_loop_notice(history(7) + pair(args={"cmd": "ls"}), 8))
        self.assertIsNone(tool_loop_notice(history(7) + pair(output="different"), 8))
        a = "Chunk ID: one\nWall time: 1.0000 seconds\nProcess exited with code 1\nOutput:\nsame"
        b = "Chunk ID: two\nWall time: 1.0000 seconds\nProcess exited with code 2\nOutput:\nsame"
        self.assertIsNone(tool_loop_notice(history(1, output=a) + pair(output=b), 2))

    def test_user_reset_and_ambiguous_histories_do_not_stop(self):
        self.assertIsNone(tool_loop_notice(history(7) + [{"role": "user", "content": "try another way"}] + pair(), 8))
        multi = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "exec_command", "arguments": {"cmd": "pwd"}}},
            {"function": {"name": "exec_command", "arguments": {"cmd": "pwd"}}}]}, {"role": "tool", "content": "same"}]
        self.assertIsNone(tool_loop_notice(multi, 2))
        self.assertIsNone(tool_loop_notice(history(1) + [{"role": "tool", "content": "extra"}], 2))

    def test_older_async_or_multitool_before_suffix_do_not_veto(self):
        older_async = "Chunk ID: abc\nWall time: 1.0000 seconds\nSession ID: 7\nOutput:\nstill running"
        msgs = history(1, output=older_async) + history(8)[1:]
        self.assertIsNotNone(tool_loop_notice(msgs, 8))
        old_multi = [{"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "exec_command", "arguments": {"cmd": "pwd"}}},
            {"function": {"name": "exec_command", "arguments": {"cmd": "pwd"}}}]}, {"role": "tool", "content": "same"}]
        self.assertIsNotNone(tool_loop_notice([{"role": "user", "content": "go"}] + old_multi + history(8)[1:], 8))

    def test_malformed_write_stdin_and_running_outputs_do_not_stop(self):
        self.assertIsNone(tool_loop_notice(history(2, args="{bad json"), 2))
        self.assertIsNone(tool_loop_notice(history(2, name="write_stdin"), 2))
        self.assertIsNone(tool_loop_notice(history(2, name="mcp.write_stdin"), 2))
        running = "Chunk ID: abc\nWall time: 1.0000 seconds\nSession ID: 7\nOutput:\nstill running"
        self.assertIsNone(tool_loop_notice(history(2, output=running), 2))
        self.assertIsNone(tool_loop_notice(history(2, output="Script running with cell ID abc"), 2))
        self.assertIsNone(tool_loop_notice(history(2, output="Process running with session ID 7"), 2))

    def test_wrapper_metadata_is_ignored_but_payload_timestamps_and_nonzero_exit_are_preserved(self):
        a = "Chunk ID: one\nWall time: 1.0000 seconds\nProcess exited with code 1\nOutput:\n2026-10-06T01:00:00Z ok"
        b = "Chunk ID: two\nWall time: 9.0000 seconds\nProcess exited with code 1\nOutput:\n2026-10-06T01:00:00Z ok"
        self.assertIsNotNone(tool_loop_notice(history(1, output=a) + pair(output=b), 2))
        c = "Chunk ID: three\nWall time: 9.0000 seconds\nProcess exited with code 1\nOutput:\n2026-10-06T01:00:01Z ok"
        self.assertIsNone(tool_loop_notice(history(1, output=a) + pair(output=c), 2))
        self.assertIsNone(tool_loop_notice(history(1, output="Chunk ID: a\nProcess exited with code 0\nOutput:\nsame") +
                                            pair(output="Chunk ID: b\nProcess exited with code 0\nOutput:\nsame\n"), 2))
        inner = "Chunk ID: a\nWall time: 1\nProcess exited with code 0\nOutput:\nChunk ID: payload\nProcess running with session ID payload\n"
        same_inner = "Chunk ID: b\nWall time: 9\nProcess exited with code 0\nOutput:\nChunk ID: payload\nProcess running with session ID payload\n"
        changed_inner = "Chunk ID: c\nWall time: 9\nProcess exited with code 0\nOutput:\nChunk ID: payload\nProcess running with session ID changed\n"
        self.assertIsNotNone(tool_loop_notice(history(1, output=inner) + pair(output=same_inner), 2))
        self.assertIsNone(tool_loop_notice(history(1, output=inner) + pair(output=changed_inner), 2))

    def test_thread_local_notice_is_request_isolated(self):
        tok = ByteTokenizer()
        engine = CountingEngine(tok)
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.tool_loop_max_repeats = 2
        barrier = threading.Barrier(2)
        results = []

        def stopped():
            ids, thinking, max_new = svc.prepare(history(2), [TOOL["function"]], {}, 20)
            barrier.wait()
            results.append(next(svc.run(ids, thinking, [TOOL["function"]], max_new, {}, threading.Event()))[1].text)

        def normal():
            ids, thinking, max_new = svc.prepare([{"role": "user", "content": "hi"}], None, {}, 20)
            barrier.wait()
            results.append(list(svc.run(ids, thinking, None, max_new, {}, threading.Event()))[-1][1]["completion_tokens"])

        ts = [threading.Thread(target=stopped), threading.Thread(target=normal)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(engine.calls, 1)
        self.assertTrue(any(isinstance(x, str) and NOTICE in x for x in results))
        self.assertTrue(any(isinstance(x, int) and x > 0 for x in results))

    def test_config_validation(self):
        self.assertEqual(tool_loop_max_repeats_from_config({}), 0)
        self.assertEqual(tool_loop_max_repeats_from_config({"tool_loop_max_repeats": 2}), 2)
        for bad in (True, 1, -1, 1.5, "8"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                tool_loop_max_repeats_from_config({"tool_loop_max_repeats": bad})


class ToolLoopGuardHttp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = CountingEngine(cls.tok, "should not run", max_context=0)
        cls.svc = LoadingService(cls.engine, cls.tok, NoRenderTemplate())
        cls.svc.tool_loop_max_repeats = 2
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=30) as r:
            text = r.read().decode()
        if not body.get("stream"):
            return json.loads(text)
        events = []
        for part in text.strip().split("\n\n"):
            lines = part.splitlines()
            data = [line[5:].strip() for line in lines if line.startswith("data:")]
            if data and data[0] != "[DONE]":
                events.append(json.loads(data[0]))
        return events

    def test_chat_responses_and_anthropic_all_stop_without_loading_or_generating(self):
        for stream in (False, True):
            with self.subTest(api="chat", stream=stream):
                chat = self.post("/v1/chat/completions", {"model": "m", "messages": history(2),
                                                           "tools": [TOOL], "stream": stream})
                if stream:
                    self.assertTrue(any(NOTICE in c["choices"][0]["delta"].get("content", "") for c in chat))
                    self.assertEqual(chat[-1]["choices"][0]["finish_reason"], "stop")
                    self.assertEqual(chat[-1]["usage"]["completion_tokens"], 0)
                else:
                    self.assertIn(NOTICE, chat["choices"][0]["message"]["content"])
                    self.assertEqual(chat["usage"]["completion_tokens"], 0)

            resp_input = [{"type": "message", "role": "user", "content": "go"}]
            for i in range(2):
                resp_input += [{"type": "function_call", "call_id": f"c{i}", "name": "exec_command",
                                "arguments": json.dumps({"cmd": "pwd"})},
                               {"type": "function_call_output", "call_id": f"c{i}", "output": "same"}]
            with self.subTest(api="responses", stream=stream):
                resp = self.post("/v1/responses", {"model": "m", "input": resp_input, "stream": stream,
                                                   "tools": [{"type": "function", "name": "exec_command"}]})
                if stream:
                    self.assertTrue(any(e.get("delta") and NOTICE in e["delta"] for e in resp))
                    self.assertEqual(resp[-1]["type"], "response.completed")
                    self.assertEqual(resp[-1]["response"]["usage"]["output_tokens"], 0)
                else:
                    self.assertIn(NOTICE, resp["output"][0]["content"][0]["text"])
                    self.assertEqual(resp["usage"]["output_tokens"], 0)

            anth = [{"role": "user", "content": "go"}]
            for i in range(2):
                anth += [{"role": "assistant", "content": [{"type": "tool_use", "id": f"c{i}",
                                                              "name": "exec_command", "input": {"cmd": "pwd"}}]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"c{i}",
                                                         "content": "same"}]}]
            with self.subTest(api="anthropic", stream=stream):
                msg = self.post("/v1/messages", {"model": "m", "messages": anth, "max_tokens": 20,
                                                 "stream": stream,
                                                 "tools": [{"name": "exec_command", "input_schema": {}}]})
                if stream:
                    self.assertTrue(any((e.get("delta") or {}).get("text") and NOTICE in e["delta"]["text"]
                                        for e in msg))
                    deltas = [e for e in msg if e.get("type") == "message_delta"]
                    self.assertEqual(deltas[-1]["delta"]["stop_reason"], "end_turn")
                    self.assertEqual(deltas[-1]["usage"]["output_tokens"], 0)
                else:
                    self.assertIn(NOTICE, msg["content"][0]["text"])
                    self.assertEqual(msg["usage"]["output_tokens"], 0)
        self.assertEqual(self.svc.load_calls, 0)
        self.assertEqual(self.engine.calls, 0)


class ToolLoopGuardColdVisionHttp(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = CountingEngine(self.tok, "vision ok")
        self.vision = FakeVision()
        self.svc = LoadingService(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"),
                                  vision=self.vision)
        self.svc.tool_loop_max_repeats = 2
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def test_unguarded_chat_loads_stopped_vision_before_prepare(self):
        body = {"model": "m", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "look"}, {"type": "image_url", "image_url": "data:image/png;base64,AA=="}]}]}
        self.post("/v1/chat/completions", body)
        self.assertEqual((self.svc.load_calls, self.vision.restarts, self.vision.encoded), (1, 1, 1))

    def test_unguarded_responses_loads_stopped_vision_before_prepare(self):
        body = {"model": "m", "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "look"}, {"type": "input_image", "image_url": "data:image/png;base64,AA=="}]}]}
        self.post("/v1/responses", body)
        self.assertEqual((self.svc.load_calls, self.vision.restarts, self.vision.encoded), (1, 1, 1))

    def test_unguarded_anthropic_loads_stopped_vision_before_prepare(self):
        body = {"model": "m", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image", "source": {"type": "url", "url": "data:image/png;base64,AA=="}}]}], "max_tokens": 20}
        self.post("/v1/messages", body)
        self.assertEqual((self.svc.load_calls, self.vision.restarts, self.vision.encoded), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
