"""Responses protocol tests, using the model-independent parser and mock engine.

    python -m unittest serve.test_responses -v
"""
import json
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate, Event, OutputParser
from serve.responses import Accumulator, ResponsesError, events, normalize
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]
FUNCTION = {"type": "function", "name": "echo", "parameters": {
    "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}
CUSTOM = {"type": "custom", "name": "apply_patch", "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"}}


def call(name, text, parameter="text"):
    return f"<tool_call>\n<function={name}>\n<parameter={parameter}>\n{text}\n</parameter>\n</function>\n</tool_call>"


def generate(body, script, max_context=32768):
    request = normalize(body)
    tok = ByteTokenizer()
    engine = MockEngine(tok, script, max_context=max_context)
    svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    ids, thinking, max_new = svc.prepare(request.messages, request.tools, request.kwargs, request.max_new)
    acc = Accumulator(request, svc.model, len(ids))
    output = list(events(acc, svc.run(ids, thinking, request.tools, max_new, request.body, threading.Event())))
    return acc.response, output, engine


class Inputs(unittest.TestCase):
    def test_web_search_error_explains_codex_configuration(self):
        for kind in ("web_search", "web_search_preview"):
            with self.subTest(kind=kind), self.assertRaises(ResponsesError) as caught:
                normalize({"input": "hi", "tools": [{"type": kind}]})
            self.assertIn('web_search = "disabled"', str(caught.exception))
            self.assertIn("top level", str(caught.exception))
            self.assertEqual(caught.exception.code, "unsupported_feature")

    def test_string_instructions_and_shared_defaults(self):
        req = normalize({"input": "hi", "instructions": "Be brief", "max_output_tokens": 7,
                         "reasoning": {"effort": "low"}, "temperature": 0},
                        {"max_tokens": 200, "reasoning_effort": "high"})
        self.assertEqual(req.max_new, 7)
        self.assertEqual(req.kwargs, {"reasoning_effort": "low"})
        self.assertEqual(req.messages, [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "hi"}])
        self.assertEqual(req.body["temperature"], 0)
        self.assertFalse(req.body["store"])
        self.assertEqual(normalize({"input": "hi"}, {"max_tokens": 9}).max_new, 9)

    def test_namespace_names_are_stable_and_distinct(self):
        tools = [FUNCTION, {"type": "namespace", "name": "editor", "tools": [FUNCTION, CUSTOM]},
                 {"type": "function", "name": "editor__echo"}]
        a = normalize({"input": "hi", "tools": tools})
        b = normalize({"input": "hi", "tools": list(reversed(tools))})
        self.assertEqual(set(a.registry), set(b.registry))
        self.assertEqual(len(a.registry), 4)
        req = normalize({"input": "hi", "tools": tools,
                         "tool_choice": {"type": "custom", "name": "apply_patch", "namespace": "editor"}})
        self.assertTrue(req.required)
        self.assertEqual(len(req.tools), 1)
        self.assertEqual(next(iter(req.registry.values())).namespace, "editor")

    def test_input_tool_declarations(self):
        item = {"type": "additional_tools", "role": "developer", "tools": [CUSTOM]}
        req = normalize({"input": [item, {"role": "user", "content": "edit"}, item],
                         "tool_choice": {"type": "custom", "name": "apply_patch"}})
        self.assertEqual(len(req.registry), 1)
        self.assertTrue(req.required)
        self.assertNotIn("additional_tools", json.dumps(req.messages))

    def test_call_result_history_and_reasoning(self):
        req = normalize({"tools": [FUNCTION, CUSTOM], "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "edit"}]},
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Consider it"}], "summary": []},
            {"role": "assistant", "content": [{"type": "output_text", "text": "Checking"}]},
            {"type": "function_call", "name": "echo", "arguments": '{"text":"hello"}', "call_id": "c1"},
            {"type": "custom_tool_call", "name": "apply_patch", "input": "patch\n", "call_id": "c2"},
            {"type": "custom_tool_call_output", "call_id": "c2", "output": "patched"},
            {"type": "function_call_output", "call_id": "c1", "output": [{"type": "input_text", "text": "hello"}]},
        ]})
        self.assertEqual(req.messages[1]["reasoning_content"], "Consider it")
        self.assertEqual(req.messages[1]["content"], "Checking")
        self.assertEqual(req.messages[1]["tool_calls"][1]["function"]["arguments"], {"input": "patch\n"})
        self.assertIn("call_id=c2", req.messages[2]["content"])
        self.assertIn("call_id=c1", req.messages[3]["content"])

    def test_images_and_late_instructions(self):
        req = normalize({"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,eA=="}]},
                                   {"role": "developer", "content": "Keep authority"}]})
        self.assertEqual(req.messages[0]["role"], "system")
        self.assertEqual(req.messages[1]["content"][0]["type"], "image")

    def test_assistant_messages_do_not_move_across_calls(self):
        req = normalize({"tools": [FUNCTION], "input": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Before"},
            {"type": "function_call", "name": "echo", "arguments": '{}', "call_id": "c"},
            {"role": "assistant", "content": "After"},
            {"type": "function_call_output", "call_id": "c", "output": "ok"}]})
        self.assertEqual(req.messages[1]["content"], "Before")
        self.assertTrue(req.messages[1]["tool_calls"])
        self.assertEqual(req.messages[2], {"role": "assistant", "content": "After"})

    def test_rejected_features_and_bad_shapes(self):
        bad = [{"store": True}, {"previous_response_id": "resp_x"}, {"background": True},
               {"conversation": "conv_x"}, {"strata_mcp": True}, {"truncation": "auto"},
               {"text": {"format": {"type": "json_schema"}}}, {"tools": [{"type": "web_search", "name": "search"}]},
               {"input": [{"type": "reasoning", "encrypted_content": "opaque"}]},
               {"input": [{"role": "user", "content": [{"type": "input_file"}]}]},
               {"input": [{"type": "function_call_output", "call_id": "missing", "output": "x"}]},
               {"input": {}}, {"input": []}, {"tools": "bad"}, {"stream": 1},
               {"reasoning": []}, {"reasoning": {"summary": "auto"}}, {"max_output_tokens": -1},
               {"tools": [FUNCTION], "tool_choice": {"type": "function", "name": "echo", "namespace": []}},
               {"tools": [{**FUNCTION, "parameters": {"properties": []}}]},
               {"tools": [{**FUNCTION, "parameters": {"properties": {"text": []}}}]},
               {"tools": [{**CUSTOM, "format": []}]},
               {"max_output_tokens": True}, {"max_output_tokens": 1.5}, {"temperature": float("nan")},
               {"tool_choice": "required"}, {"tools": [FUNCTION, FUNCTION]}]
        for extra in bad:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                normalize({"input": "hi", **extra})
        with self.assertRaises(ResponsesError):
            normalize([])
        # Codex's include request is a hint, not an encrypted-state guarantee.
        self.assertFalse(normalize({"input": "hi", "include": ["reasoning.encrypted_content"]}).body["store"])


class Protocol(unittest.TestCase):
    def test_text_reasoning_and_event_snapshots(self):
        response, evs, _ = generate({"input": "hi"}, "Thinking</think>\n\nHello 🌍")
        self.assertEqual(response["status"], "completed")
        self.assertEqual([i["type"] for i in response["output"]], ["reasoning", "message"])
        self.assertEqual(response["output"][0]["summary"], [])
        self.assertEqual(response["output"][0]["content"][0]["text"], "Thinking")
        self.assertEqual(response["output"][1]["content"][0]["text"], "Hello 🌍")
        self.assertEqual(evs[0]["response"]["output"], [])
        self.assertEqual(evs[-1]["response"], response)
        self.assertEqual([e["sequence_number"] for e in evs], list(range(len(evs))))
        text = "".join(e["delta"] for e in evs if e["type"] == "response.output_text.delta")
        self.assertEqual(text, "Hello 🌍")
        self.assertEqual(response["usage"]["total_tokens"], response["usage"]["input_tokens"] + response["usage"]["output_tokens"])
        self.assertFalse(any("choices" in e for e in evs))

    def test_multiple_calls_and_replay(self):
        body = {"input": "echo", "tools": [FUNCTION]}
        internal = normalize(body).tools[0]["name"]
        response, evs, _ = generate(body, "</think>\n\n" + call(internal, "one") + "\n" + call(internal, "two"))
        calls = response["output"]
        self.assertEqual(len(calls), 2)
        self.assertEqual([json.loads(i["arguments"]) for i in calls], [{"text": "one"}, {"text": "two"}])
        self.assertEqual(len({i["id"] for i in calls}), 2)
        for index, item in enumerate(calls):
            deltas = "".join(e["delta"] for e in evs if e["type"] == "response.function_call_arguments.delta" and e["output_index"] == index)
            self.assertEqual(deltas, item["arguments"])
        history = [{"role": "user", "content": "echo"}] + calls + [
            {"type": "function_call_output", "call_id": i["call_id"], "output": "result"} for i in reversed(calls)]
        final, _, engine = generate({"input": history, "tools": [FUNCTION]}, "</think>\n\nDone")
        self.assertEqual(final["output"][0]["content"][0]["text"], "Done")
        self.assertTrue(engine.last_prompt)

    def test_boolean_property_schema_is_prompt_guidance(self):
        body = {"input": "echo", "tools": [{**FUNCTION, "parameters": {"properties": {"text": True}}}]}
        internal = normalize(body).tools[0]["name"]
        response, _, _ = generate(body, "</think>\n\n" + call(internal, "hello"))
        self.assertEqual(json.loads(response["output"][0]["arguments"]), {"text": "hello"})

    def test_custom_input_every_chunk_boundary(self):
        patch = '\n*** Begin Patch\n*** Add File: café.txt\n+"quotes" \\ slash\n+</parameter> literal\n+</tool_call>\n*** End Patch\n\n'
        body = {"input": "edit", "tools": [{"type": "namespace", "name": "editor", "tools": [CUSTOM]}]}
        req = normalize(body)
        script = "</think>\n\n" + call(req.tools[0]["name"], patch, "input")
        for size in (1, 2, 3, 7, 31, len(script)):
            parser = OutputParser(tools=req.tools, stream_tools=True)
            acc = Accumulator(req, "m", 10)
            evs = []
            for start in range(0, len(script), size):
                for ev in parser.feed(script[start:start + size]):
                    evs += acc.feed(ev)
            for ev in parser.finish():
                evs += acc.feed(ev)
            evs += acc.finish({"finish": "stop", "completion_tokens": 100})
            item = acc.response["output"][0]
            self.assertEqual(item["input"], patch, size)
            self.assertEqual(item["namespace"], "editor")
            self.assertEqual(item["type"], "custom_tool_call")
            self.assertEqual("".join(e["delta"] for e in evs if e["type"] == "response.custom_tool_call_input.delta"), patch)
            replay = normalize({"tools": body["tools"], "input": [{"role": "user", "content": "edit"}, item,
                {"type": "custom_tool_call_output", "call_id": item["call_id"], "output": "ok"}]})
            self.assertEqual(replay.messages[1]["tool_calls"][0]["function"]["arguments"]["input"], patch)

    def test_length_does_not_complete_tool(self):
        body = {"input": "edit", "tools": [CUSTOM], "max_output_tokens": 120}
        internal = normalize(body).tools[0]["name"]
        response, evs, _ = generate(body, "</think>\n\n" + call(internal, "x" * 400, "input"))
        self.assertEqual(response["status"], "incomplete")
        self.assertEqual(response["incomplete_details"], {"reason": "max_output_tokens"})
        self.assertEqual(response["output"][0]["status"], "incomplete")
        self.assertFalse(any(e["type"] == "response.custom_tool_call_input.done" for e in evs))

    def test_tool_selection_and_cardinality(self):
        for body, script in [({"input": "hi", "tools": [FUNCTION], "tool_choice": "required"}, "</think>\n\nno call"),
                             ({"input": "hi", "tool_choice": "none"}, "</think>\n\n" + call("echo", "x"))]:
            with self.assertRaises(ResponsesError):
                generate(body, script)
        body = {"input": "hi", "tools": [FUNCTION], "parallel_tool_calls": False}
        internal = normalize(body).tools[0]["name"]
        with self.assertRaises(ResponsesError):
            generate(body, "</think>\n\n" + call(internal, "one") + call(internal, "two"))

    def test_iterator_closed_on_disconnect_and_exception(self):
        closed = []
        def run():
            try:
                yield "ping", None
                raise EngineDied("dead")
            finally:
                closed.append(True)
        acc = Accumulator(normalize({"input": "hi"}), "m", 10)
        stream = events(acc, run())
        next(stream)
        next(stream)
        next(stream)
        stream.close()
        self.assertTrue(closed)
        closed.clear()
        with self.assertRaises(EngineDied):
            list(events(acc, run()))
        self.assertTrue(closed)
        failed = acc.fail(EngineDied("dead"))
        self.assertEqual(failed["type"], "response.failed")
        self.assertEqual(failed["response"]["status"], "failed")


class HTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = MockEngine(tok, "Thinking</think>\n\nHello")
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, body, path="/v1/responses", key=None):
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers=headers)
        try:
            with self.opener.open(req, timeout=5) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    def test_stream_and_collect_parity(self):
        status, text = self.post({"input": "hi"})
        self.assertEqual(status, 200)
        collected = json.loads(text)
        status, text = self.post({"input": "hi", "stream": True}, "/v1/responses/?beta=true")
        self.assertEqual(status, 200)
        frames = text.strip().split("\n\n")
        evs = [json.loads(f.split("\ndata: ", 1)[1]) for f in frames]
        self.assertEqual(evs[-1]["type"], "response.completed")
        self.assertNotIn("[DONE]", text)
        self.assertTrue(all(f.startswith("event: " + e["type"]) for f, e in zip(frames, evs)))
        self.assertEqual([i["content"] for i in collected["output"]], [i["content"] for i in evs[-1]["response"]["output"]])
        self.assertEqual(collected["usage"], evs[-1]["response"]["usage"])

    def test_auth_errors_status_and_budget(self):
        self.svc.api_key = "test-key"
        try:
            self.assertEqual(self.post({"input": "hi"})[0], 401)
            self.assertEqual(self.post({"input": "hi"}, key="test-key")[0], 200)
        finally:
            self.svc.api_key = ""
        self.assertEqual(self.post({"input": "hi", "store": True})[0], 400)
        for body in ({"input": "hi", "max_output_tokens": 40000}, {"input": "x" * 40000}):
            status, text = self.post(body)
            self.assertEqual(status, 400)
            self.assertEqual(json.loads(text)["error"]["code"], "context_length_exceeded")
        self.assertEqual(self.post({"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "x"}]}]})[0], 400)
        response = json.loads(self.post({"input": "hi", "max_output_tokens": 2})[1])
        self.assertEqual(response["status"], "incomplete")
        self.assertIn("/v1/responses", self.svc.v1_status()["dialects"])

    def test_engine_failure_and_next_request(self):
        original = self.engine.generate
        def dying(*args, **kwargs):
            yield None
            raise EngineDied("scripted failure")
        self.engine.generate = dying
        try:
            status, text = self.post({"input": "hi", "stream": True})
            self.assertEqual(status, 200)
            self.assertIn("event: response.failed", text)
            self.assertNotIn("event: response.completed", text)
            self.assertEqual(self.post({"input": "hi"})[0], 503)
        finally:
            self.engine.generate = original
        self.assertEqual(self.post({"input": "hi"})[0], 200)
        self.assertFalse(self.svc.status["busy"])

    def test_generation_error_has_server_status(self):
        status, text = self.post({"input": "hi", "tools": [FUNCTION], "tool_choice": "required"})
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(text)["error"]["type"], "server_error")
        status, text = self.post({"input": "hi", "tools": [FUNCTION], "tool_choice": "required", "stream": True})
        self.assertEqual(status, 200)
        self.assertIn("event: response.failed", text)
        self.assertNotIn("event: response.completed", text)

    def test_disconnect_cancels_and_closes_engine(self):
        closed = threading.Event()
        class TrackingEngine(MockEngine):
            def generate(engine, ids, max_new, sampling, cancel, **kwargs):
                engine.cancel = cancel
                try:
                    yield from super().generate(ids, max_new, sampling, cancel, **kwargs)
                finally:
                    closed.set()
        tok = self.svc.tok
        slow = TrackingEngine(tok, "</think>\n\n" + "x" * 3000, delay_s=0.002)
        original = self.svc.engine
        self.svc.engine = slow
        try:
            body = json.dumps({"input": "hi", "stream": True}).encode()
            with socket.create_connection(self.httpd.server_address, timeout=5) as conn:
                conn.sendall(b"POST /v1/responses HTTP/1.0\r\nContent-Type: application/json\r\nContent-Length: " +
                             str(len(body)).encode() + b"\r\n\r\n" + body)
                received = b""
                while b"response.output_text.delta" not in received:
                    chunk = conn.recv(4096)
                    self.assertTrue(chunk)
                    received += chunk
                conn.shutdown(socket.SHUT_RDWR)
            self.assertTrue(closed.wait(3))
            self.assertTrue(slow.cancel.is_set())
            deadline = time.monotonic() + 3
            while self.svc.status["busy"] and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(self.svc.status["busy"])
        finally:
            self.svc.engine = original
        self.assertEqual(self.post({"input": "hi"})[0], 200)


if __name__ == "__main__":
    unittest.main()
