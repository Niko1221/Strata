"""Responses contract and lifecycle checks using the real service and scripted engine.

    python -m unittest serve.test_responses -v
"""
import copy
import http.client
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate, Event
from serve.response_store import ResponseStore
from serve.responses import RequestError, ResponseController, Status, TERMINAL, strict_json, transition
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service, Server, main, make_handler


HAVE_JSONSCHEMA = importlib.util.find_spec("jsonschema") is not None
SCHEMA = {"type": "object", "properties": {"value": {"type": "integer"}},
          "required": ["value"], "additionalProperties": False}
FORMAT = {"format": {"type": "json_schema", "name": "result", "schema": SCHEMA, "strict": True}}
TOOLS = [{"type": "function", "name": "lookup", "strict": True, "parameters": {
    "type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"], "additionalProperties": False}}]
CALL = '<tool_call><function=lookup><parameter=q>snow ☃\nline "two"</parameter></function></tool_call>'


def normalized(value):
    """Normalize only server-generated identity/time, not content, indices or settings."""
    if isinstance(value, list):
        return [normalized(v) for v in value]
    if isinstance(value, dict):
        return {k: ("ID" if k in ("id", "call_id", "item_id") else 0 if k in ("created_at", "completed_at")
                    and v is not None else normalized(v)) for k, v in value.items()}
    return value


def assemble(test, events):
    """An independent consumer: apply deltas, then compare every done item with accumulated bytes."""
    output, terminal = [], None
    test.assertEqual([e["sequence_number"] for e in events], list(range(len(events))))
    test.assertEqual(events[0]["type"], "response.created")
    for event in events:
        kind = event["type"]
        if kind == "response.output_item.added":
            test.assertEqual(event["output_index"], len(output))
            output.append(copy.deepcopy(event["item"]))
        elif kind == "response.content_part.added":
            item = output[event["output_index"]]
            test.assertEqual(item["id"], event["item_id"])
            test.assertEqual(event["content_index"], len(item["content"]))
            item["content"].append(copy.deepcopy(event["part"]))
        elif kind in ("response.output_text.delta", "response.function_call_arguments.delta"):
            item = output[event["output_index"]]
            test.assertEqual(item["id"], event["item_id"])
            if kind == "response.output_text.delta":
                item["content"][event["content_index"]]["text"] += event["delta"]
            else:
                item["arguments"] += event["delta"]
        elif kind == "response.output_text.done":
            test.assertEqual(output[event["output_index"]]["content"][event["content_index"]]["text"], event["text"])
        elif kind == "response.function_call_arguments.done":
            test.assertEqual(output[event["output_index"]]["arguments"], event["arguments"])
        elif kind == "response.content_part.done":
            test.assertEqual(output[event["output_index"]]["content"][event["content_index"]], event["part"])
        elif kind == "response.output_item.done":
            item = output[event["output_index"]]
            item["status"] = event["item"]["status"]
            test.assertEqual(item, event["item"])
        elif kind in ("response.completed", "response.failed", "response.incomplete"):
            test.assertIsNone(terminal)
            terminal = event["response"]
            test.assertEqual(output, terminal["output"])
    test.assertIsNotNone(terminal)
    return terminal


class RecordingEngine(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.sampling = dict(sampling)
        self.drained = False
        try:
            yield from super().generate(ids, max_new, sampling, cancel, embeddings)
        finally:
            self.drained = True


class ResponseFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tok = ByteTokenizer()
        self.svc = Service(RecordingEngine(self.tok, "Hello, 世界!\n"), self.tok,
                           ChatTemplate(Path(__file__).parent / "chat_template.jinja"), model_name="test-model")
        self.store = ResponseStore(self.tmp.name)
        self.controller = ResponseController(self.svc, self.store)

    def script(self, script, engine=RecordingEngine, **kwargs):
        self.svc.engine = engine(self.tok, script, **kwargs)

    def create(self, **kwargs):
        return self.controller.create_response({"model": self.svc.model, "input": "Hello", **kwargs})

    def run_response(self, **kwargs):
        handle = self.create(**kwargs)
        events = [e for e in self.controller.events(handle) if e is not None]
        snapshot = self.controller.snapshot(handle)
        self.assertEqual(assemble(self, events), snapshot)
        self.assertEqual(self.controller.active, {})
        return snapshot, events


class ControllerTests(ResponseFixture):
    def test_transition_table_and_terminal_immutability(self):
        for current in Status:
            for event in ("start", "output", "complete", "exhaust", "fail", "stopped"):
                allowed = (current == Status.QUEUED and event in ("start", "fail", "stopped")) or (
                    current == Status.IN_PROGRESS and event != "start")
                if allowed:
                    self.assertIn(transition(current, event), Status)
                else:
                    with self.assertRaises(ValueError):
                        transition(current, event)
        handle = self.create()
        list(self.controller.events(handle))
        before = self.controller.snapshot(handle)
        self.assertEqual(self.controller.finalize_response(handle, "fail"), [])
        with self.assertRaises(ValueError):
            self.controller.append_output(handle, Event("content", "late"))
        self.assertEqual(self.controller.snapshot(handle), before)

    def test_text_stream_and_json_are_identical(self):
        first, _ = self.run_response(stream=True)
        second, _ = self.run_response(stream=False)
        self.assertEqual(normalized(first), normalized(second))
        self.assertEqual(first["output"][0]["content"][0]["text"], "Hello, 世界!\n")
        self.assertEqual(self.controller.get_response(first["id"]), first)
        self.assertNotIn("input_items", first)

    @unittest.skipUnless(HAVE_JSONSCHEMA, "optional jsonschema dependency")
    def test_function_stream_bytes_ids_and_client_loop(self):
        self.script("Before\n" + CALL + "\nAfter")
        response, _ = self.run_response(tools=TOOLS)
        self.assertEqual([i["type"] for i in response["output"]], ["message", "function_call", "message"])
        call = response["output"][1]
        self.assertNotEqual(call["id"], call["call_id"])
        self.assertEqual(json.loads(call["arguments"]), {"q": 'snow ☃\nline "two"'})
        self.script("Result understood")
        child, _ = self.run_response(previous_response_id=response["id"], input=[{
            "type": "function_call_output", "call_id": call["call_id"], "output": "42"}])
        self.assertEqual(child["status"], "completed")
        prompt = self.tok.decode(self.svc.engine.last_prompt)
        self.assertIn("Before", prompt)
        self.assertIn("After", prompt)
        self.script(CALL)
        parent, _ = self.run_response(tools=TOOLS)
        call = parent["output"][0]
        self.script("Result understood")
        child, _ = self.run_response(previous_response_id=parent["id"], input=[{
            "type": "function_call_output", "call_id": call["call_id"], "output": "42"}])
        self.assertEqual(child["status"], "completed")
        prompt = self.tok.decode(self.svc.engine.last_prompt)
        self.assertIn("lookup", prompt)
        self.assertIn("42", prompt)
        with self.assertRaises(RequestError):
            self.create(previous_response_id=parent["id"], input=[{
                "type": "function_call_output", "call_id": call["id"], "output": "wrong ID"}])

    @unittest.skipUnless(HAVE_JSONSCHEMA, "optional jsonschema dependency")
    def test_parallel_results_are_bound_by_call_id(self):
        self.script(CALL + CALL.replace("snow ☃", "rain"))
        parent, _ = self.run_response(tools=TOOLS)
        self.script("Both received")
        calls = parent["output"]
        self.run_response(previous_response_id=parent["id"], input=[
            {"type": "function_call_output", "call_id": calls[1]["call_id"], "output": "SECOND"},
            {"type": "function_call_output", "call_id": calls[0]["call_id"], "output": "FIRST"}])
        prompt = self.tok.decode(self.svc.engine.last_prompt)
        self.assertLess(prompt.index("FIRST"), prompt.index("SECOND"))

    def test_continuation_forks_and_instructions_do_not_leak(self):
        parent, _ = self.run_response(instructions="PARENT_ONLY", input="original")
        parent_saved = copy.deepcopy(parent)
        self.run_response(previous_response_id=parent["id"], input="CHILD_A", instructions="NEW_ONLY")
        prompt_a = self.tok.decode(self.svc.engine.last_prompt)
        self.run_response(previous_response_id=parent["id"], input="CHILD_B")
        prompt_b = self.tok.decode(self.svc.engine.last_prompt)
        self.assertIn("original", prompt_a)
        self.assertIn("Hello, 世界!", prompt_a)
        self.assertIn("NEW_ONLY", prompt_a)
        for text in ("PARENT_ONLY", "NEW_ONLY", "CHILD_A"):
            self.assertNotIn(text, prompt_b)
        self.assertNotIn("PARENT_ONLY", prompt_a)
        self.assertEqual(self.controller.get_response(parent["id"]), parent_saved)

    def test_store_false_missing_deleted_and_expired_parents(self):
        response, _ = self.run_response(store=False)
        self.assertEqual(list(Path(self.tmp.name).glob("*.json")), [])
        with self.assertRaises(RequestError):
            self.create(previous_response_id=response["id"])
        parent, _ = self.run_response()
        self.controller.delete_response(parent["id"])
        with self.assertRaises(RequestError):
            self.create(previous_response_id=parent["id"])
        parent, _ = self.run_response()
        self.store.retention_s = 0
        with self.assertRaises(RequestError):
            self.create(previous_response_id=parent["id"])

    def test_delete_during_generation_cannot_resurrect(self):
        handle = self.create()
        events = self.controller.events(handle)
        next(events)
        next(events)
        response_id = handle.record.response["id"]
        self.assertEqual(self.controller.get_response(response_id)["status"], "in_progress")
        self.assertEqual(self.controller.delete_response(response_id),
                         {"id": response_id, "object": "response.deleted", "deleted": True})
        self.assertFalse(handle.cancel.is_set())
        list(events)
        self.assertEqual(self.controller.snapshot(handle)["status"], "completed")
        with self.assertRaises(RequestError):
            self.controller.get_response(response_id)
        self.assertEqual(list(Path(self.tmp.name).glob("*.json")), [])

    def test_recovery_marks_interrupted_work_failed(self):
        completed, _ = self.run_response()
        queued = self.create().record.response["id"]
        controller = ResponseController(self.svc, ResponseStore(self.tmp.name))
        recovered = controller.get_response(queued)
        self.assertEqual(recovered["status"], "failed")
        self.assertIn("restarted", recovered["error"]["message"])
        self.assertEqual(controller.get_response(completed["id"]), completed)

    @unittest.skipUnless(HAVE_JSONSCHEMA, "optional jsonschema dependency")
    def test_json_keeps_whitespace_and_validates(self):
        original = ' { "value" : 7 } \n'
        self.script(original)
        response, _ = self.run_response(text=FORMAT)
        self.assertEqual(response["output"][0]["content"][0]["text"], original)
        for bad in ('{"value":"wrong"}', '{"value":1,"extra":2}', '{"value":1,"value":2}',
                    '{"value":NaN}', '{"value":1e999}', '```json\n{"value":7}\n```'):
            with self.subTest(bad=bad):
                self.script(bad)
                response, events = self.run_response(text=FORMAT)
                self.assertEqual(response["status"], "failed")
                self.assertNotIn("response.completed", [e["type"] for e in events])

    def test_missing_schema_validator_never_falls_back(self):
        with mock.patch("serve.responses.jsonschema_modules", return_value=None):
            with self.assertRaisesRegex(RequestError, "requires jsonschema"):
                self.create(text=FORMAT)
            with self.assertRaisesRegex(RequestError, "requires jsonschema"):
                self.create(tools=TOOLS)

    def test_json_object_has_no_schema_dependency(self):
        self.script('{"any": true}')
        with mock.patch("serve.responses.jsonschema_modules", return_value=None):
            response, _ = self.run_response(text={"format": {"type": "json_object"}})
        self.assertEqual(response["status"], "completed")

    @unittest.skipUnless(HAVE_JSONSCHEMA, "optional jsonschema dependency")
    def test_strict_nested_nullable_objects_and_schema_defaults(self):
        schema = {**SCHEMA, "properties": {"value": {"type": ["object", "null"], "properties": {"a": {"type": "string"}}}}}
        with self.assertRaises(RequestError):
            self.create(text={"format": {**FORMAT["format"], "schema": schema}})
        tool = {"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}
        handle = self.create(tools=[tool])
        normalized_tool = handle.record.response["tools"][0]
        self.assertTrue(normalized_tool["strict"])
        self.assertEqual(normalized_tool["parameters"]["required"], ["q"])
        self.assertFalse(normalized_tool["parameters"]["additionalProperties"])
        self.controller.close_response(handle)

    @unittest.skipUnless(HAVE_JSONSCHEMA, "optional jsonschema dependency")
    def test_tool_choice_parallel_limit_and_argument_validation(self):
        for script, options in ((CALL, {"tools": TOOLS, "tool_choice": "none"}),
                                 (CALL + CALL, {"tools": TOOLS, "parallel_tool_calls": False}),
                                 (CALL.replace("parameter=q", "parameter=unknown"), {"tools": TOOLS})):
            with self.subTest(options=options):
                self.script(script)
                response, _ = self.run_response(**options)
                self.assertEqual(response["status"], "failed")

    def test_failed_storage_never_acknowledges_completed(self):
        changes = []
        handle = self.controller.create_response({"model": self.svc.model, "input": "hi"}, observer=changes.append)
        with mock.patch.object(self.store, "update", side_effect=OSError("disk full")):
            events = list(self.controller.events(handle))
            self.assertEqual(assemble(self, events)["status"], "failed")
            self.assertEqual(self.controller.get_response(handle.record.response["id"])["status"], "failed")
        self.assertEqual([c["status"] for c in changes], ["queued", "in_progress", "failed"])
        self.assertFalse(self.controller.active)
        self.assertEqual(self.controller.get_response(handle.record.response["id"])["status"], "failed")

    def test_drain_failure_is_failed_not_confirmed_cancelled(self):
        class BadDrain(RecordingEngine):
            def generate(self, *args, **kwargs):
                try:
                    yield from super().generate(*args, **kwargs)
                finally:
                    raise RuntimeError("drain failed")
        self.script("abc", engine=BadDrain)
        handle = self.create()
        events = self.controller.events(handle)
        while next(events)["type"] != "response.output_text.delta":
            pass
        events.close()
        self.assertEqual(self.controller.snapshot(handle)["status"], "failed")
        self.assertFalse(self.controller.active)

    def test_a_queued_response_cancels_without_entering_engine(self):
        self.svc.fifo.acquire()
        handle = self.create()
        stream = self.controller.events(handle)
        next(stream)
        result = []
        thread = threading.Thread(target=lambda: result.extend(stream))
        try:
            thread.start()
            deadline = time.monotonic() + 2
            while not self.svc.status["queued"] and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertEqual(self.controller.snapshot(handle)["status"], "queued")
            handle.cancel.set()
            self.assertEqual(self.controller.snapshot(handle)["status"], "queued")
        finally:
            self.svc.fifo.release()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.controller.snapshot(handle)["status"], "cancelled")
        self.assertEqual(self.svc.engine.last_prompt, [])

    @unittest.skipUnless(HAVE_JSONSCHEMA, "optional jsonschema dependency")
    def test_stateless_replay_and_invalid_tool_result_batches(self):
        self.script(CALL)
        parent, _ = self.run_response(tools=TOOLS, store=False)
        call = parent["output"][0]
        output = {"type": "function_call_output", "call_id": call["call_id"], "output": "ok"}
        self.script("replayed")
        self.run_response(input=[{"role": "user", "content": "hi"}, *parent["output"], output], store=False)
        for items in ([output], [call], [call, output, output], [call, call, output]):
            with self.subTest(items=items), self.assertRaises(RequestError):
                self.create(input=items)

    @unittest.skipUnless(HAVE_JSONSCHEMA, "optional jsonschema dependency")
    def test_incomplete_text_json_and_function_arguments(self):
        for script, options, budget in (("long text", {}, 3), ('{"value":7}', {"text": FORMAT}, 4),
                                         (CALL, {"tools": TOOLS}, 59)):
            with self.subTest(script=script):
                self.script(script)
                response, _ = self.run_response(max_output_tokens=budget, **options)
                self.assertEqual(response["status"], "incomplete")
                self.assertEqual(response["incomplete_details"], {"reason": "max_output_tokens"})
                self.assertTrue(all(i["status"] == "incomplete" for i in response["output"]))

    def test_capabilities_rejected_before_preparing_or_loading(self):
        bad = [{"background": True}, {"reasoning": {"effort": "high"}}, {"response_format": {}},
               {"tools": [{"type": "web_search"}]}, {"tools": [{"type": "mcp"}]},
               {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "file:///x"}]}]},
               {"include": ["reasoning.encrypted_content"]}, {"conversation": "conv_x"},
               {"context_management": []}, {"tool_choice": "required"}, {"tool_choice": {"type": "function", "name": "x"}},
               {"text": {"format": {"type": "text"}, "verbosity": "high"}},
               {"stream": "true"}, {"store": 1}, {"max_output_tokens": True}, {"max_output_tokens": 0},
               {"temperature": float("nan")}, {"top_p": 1.5}, {"truncation": "auto"},
               {"tools": TOOLS, "text": FORMAT}, {"input": [{"role": "user", "content": "a", "ignored": 1}]}]
        with mock.patch.object(self.svc, "load") as load:
            for request in bad:
                with self.subTest(request=request), self.assertRaises(RequestError):
                    self.create(**request)
            load.assert_not_called()
        self.assertEqual(list(Path(self.tmp.name).glob("*.json")), [])

    @unittest.skipUnless(HAVE_JSONSCHEMA, "optional jsonschema dependency")
    def test_schema_remote_unknown_and_unresolved_refs_rejected(self):
        for schema in ({**SCHEMA, "$ref": "https://example.com/schema"},
                       {**SCHEMA, "format": "email"}, {**SCHEMA, "$ref": "#/$defs/missing"}):
            with self.subTest(schema=schema), self.assertRaises(RequestError):
                self.create(text={"format": {**FORMAT["format"], "schema": schema}})

    def test_sampling_settings_and_alias(self):
        self.svc.set_aliases(["alias"])
        response, _ = self.run_response(model="alias", temperature=0.4, top_p=0.8)
        self.assertEqual(response["model"], "alias")
        self.assertEqual(self.svc.engine.sampling["temperature"], 0.4)
        self.assertEqual(self.svc.engine.sampling["top_p"], 0.8)

    def test_budget_does_not_inherit_chat_clamping(self):
        self.svc.fit_max_tokens = True
        with self.assertRaises(RequestError):
            self.create(max_output_tokens=1000000)

    def test_engine_failure_and_next_request(self):
        class DyingEngine(RecordingEngine):
            def generate(self, *args, **kwargs):
                yield ord("x")
                raise EngineDied("test engine stopped")
        self.script("", engine=DyingEngine)
        response, _ = self.run_response()
        self.assertEqual(response["status"], "failed")
        self.assertIn("test engine stopped", response["error"]["message"])
        self.assertFalse(self.svc.status["busy"])
        self.script("next request")
        response, _ = self.run_response()
        self.assertEqual(response["status"], "completed")

    def test_disconnect_cancellation_is_confirmed_after_drain(self):
        handle = self.create()
        events = self.controller.events(handle)
        while next(events)["type"] != "response.output_text.delta":
            pass
        handle.cancel.set()
        self.assertEqual(self.controller.snapshot(handle)["status"], "in_progress")
        events.close()
        self.assertTrue(self.svc.engine.drained)
        self.assertFalse(self.svc.status["busy"])
        self.assertEqual(self.controller.get_response(handle.record.response["id"])["status"], "cancelled")
        self.assertFalse(self.controller.active)

    def test_drain_finishes_before_cancellation_and_next_generation(self):
        class DrainingEngine(RecordingEngine):
            starts = 0
            entered = threading.Event()
            release = threading.Event()

            def generate(self, *args, **kwargs):
                self.starts += 1
                try:
                    yield from super().generate(*args, **kwargs)
                finally:
                    if self.starts == 1:
                        self.entered.set()
                        if not self.release.wait(3):
                            raise RuntimeError("test did not release drain")

        self.script("abc", engine=DrainingEngine)
        first, second = self.create(), self.create()
        stream = self.controller.events(first)
        while next(stream)["type"] != "response.output_text.delta":
            pass
        closing = threading.Thread(target=stream.close)
        results = []
        next_request = threading.Thread(target=lambda: results.extend(self.controller.events(second)))
        try:
            closing.start()
            self.assertTrue(self.svc.engine.entered.wait(2))
            self.assertEqual(self.controller.snapshot(first)["status"], "in_progress")
            next_request.start()
            deadline = time.monotonic() + 2
            while not self.svc.status["queued"] and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertEqual(self.svc.engine.starts, 1)
            self.assertEqual(self.controller.snapshot(second)["status"], "queued")
        finally:
            self.svc.engine.release.set()
            closing.join(3)
            if next_request.ident is not None:
                next_request.join(3)
        self.assertFalse(closing.is_alive())
        self.assertFalse(next_request.is_alive())
        self.assertEqual(self.controller.snapshot(first)["status"], "cancelled")
        self.assertEqual(assemble(self, results)["status"], "completed")
        self.assertEqual(self.svc.engine.starts, 2)

    def test_close_before_first_event_and_cancel_while_queued(self):
        handle = self.create()
        self.controller.close_response(handle)
        self.assertEqual(self.controller.snapshot(handle)["status"], "cancelled")
        handle = self.create()
        handle.cancel.set()
        with mock.patch.object(self.svc.engine, "generate") as generate:
            list(self.controller.events(handle))
            generate.assert_not_called()
        self.assertEqual(self.svc.status["queued"], 0)

    def test_storage_only_writes_create_and_terminal(self):
        self.script("a" * 4000)
        with mock.patch.object(self.store, "_write", wraps=self.store._write) as writes:
            self.run_response()
            self.assertEqual(writes.call_count, 2)

    def test_existing_service_serializes_concurrent_responses(self):
        self.script("abc", delay_s=0.015)
        handles = [self.create(input=str(i)) for i in range(3)]
        transitions = []
        guard = threading.Lock()
        def observe(change):
            if change["status"] == "in_progress":
                with guard:
                    self.assertTrue(self.svc.status["busy"])
                    transitions.append(change)
        for handle in handles:
            handle.observer = observe
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda h: list(self.controller.events(h)), handles))
        self.assertEqual(len(transitions), 3)
        self.assertTrue(all(r[-1]["type"] == "response.completed" for r in results))
        self.assertEqual(self.svc.status["queued"], 0)
        self.assertFalse(self.svc.status["busy"])


class HTTPTests(ResponseFixture):
    def setUp(self):
        super().setUp()
        self.svc.responses = self.controller
        self.svc.api_key = "test-key"
        self.svc.cors_origins = ["https://client.example"]
        self.svc.api_monitor = True
        self.httpd = Server(("127.0.0.1", 0), make_handler(self.svc))
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(2)

    def request(self, method="POST", path="/v1/responses", body=None, headers=None):
        conn = http.client.HTTPConnection(*self.httpd.server_address, timeout=5)
        try:
            payload = json.dumps(body).encode() if body is not None and not isinstance(body, bytes) else body
            conn.request(method, path, body=payload, headers={"Authorization": "Bearer test-key",
                "Content-Type": "application/json", "Origin": "https://client.example", **(headers or {})})
            response = conn.getresponse()
            raw = response.read().decode()
            data = raw if response.getheader("Content-Type") == "text/event-stream" else json.loads(raw or "null")
            return response.status, dict(response.getheaders()), data
        finally:
            conn.close()

    def post(self, **kwargs):
        return self.request(body={"model": self.svc.model, "input": "hello", **kwargs})

    def test_http_stream_matches_json_and_monitor(self):
        code, headers, raw = self.post(stream=True)
        self.assertEqual(code, 200)
        self.assertEqual(headers["Access-Control-Allow-Origin"], "https://client.example")
        self.assertNotIn("[DONE]", raw)
        events = []
        for block in raw.strip().split("\n\n"):
            lines = block.splitlines()
            if lines[0].startswith(":"):
                continue
            event = json.loads(lines[1][6:])
            self.assertEqual(lines[0], "event: " + event["type"])
            events.append(event)
        streamed = assemble(self, events)
        code, _, collected = self.post()
        self.assertEqual(code, 200)
        self.assertEqual(normalized(streamed), normalized(collected))
        code, _, monitor = self.request("GET", "/api/requests")
        self.assertEqual(code, 200)
        trace = monitor["requests"][0]
        self.assertEqual(trace["state"], "completed")
        self.assertEqual([t["status"] for t in trace["transitions"]], ["queued", "in_progress", "completed"])
        self.assertNotIn("transitions", collected)

    @unittest.skipUnless(importlib.util.find_spec("openai"), "optional official SDK smoke test")
    def test_official_openai_sdk(self):
        from openai import NotFoundError, OpenAI
        with OpenAI(api_key="test-key", base_url=f"http://127.0.0.1:{self.httpd.server_address[1]}/v1",
                    _strict_response_validation=True, timeout=5, max_retries=0) as client:
            response = client.responses.create(model=self.svc.model, input="hi")
            self.assertEqual(response.output_text, "Hello, 世界!\n")
            self.assertEqual(client.responses.retrieve(response.id).model_dump(), response.model_dump())
            self.assertEqual(len(client.responses.input_items.list(response.id).data), 1)
            client.responses.delete(response.id)  # the SDK's delete method intentionally returns None
            with self.assertRaises(NotFoundError):
                client.responses.retrieve(response.id)
            with client.responses.stream(model=self.svc.model, input="hi") as stream:
                final = stream.get_final_response()
            self.assertEqual(final.output_text, response.output_text)
            for history in ({"role": "assistant", "content": "Earlier answer"},
                            {"role": "assistant", "content": [{"type": "input_text", "text": "Earlier answer"}]},
                            final.output[0].model_dump(exclude_none=True)):
                replay = client.responses.create(model=self.svc.model, input=[
                    {"role": "user", "content": "Earlier question"}, history,
                    {"role": "user", "content": "Next question"}])
                items = client.responses.input_items.list(replay.id, order="asc").data
                self.assertEqual(items[1].role, "assistant")
                self.assertEqual(items[1].status, "completed")
                self.assertEqual(items[1].content[0].type, "output_text")
                self.assertEqual(items[1].content[0].annotations, [])
            if not HAVE_JSONSCHEMA:
                return
            self.script(CALL)
            with client.responses.stream(model=self.svc.model, input="hi", tools=TOOLS) as stream:
                final = stream.get_final_response()
            self.assertEqual(final.output[0].type, "function_call")
            self.assertEqual(json.loads(final.output[0].arguments), {"q": 'snow ☃\nline "two"'})
            self.script("Tool result received")
            replay = client.responses.create(model=self.svc.model, input=[
                {"role": "user", "content": "hi"},
                final.output[0].model_dump(exclude_none=True, exclude={"parsed_arguments"}),
                {"type": "function_call_output", "call_id": final.output[0].call_id, "output": "Found"}])
            items = client.responses.input_items.list(replay.id, order="asc").data
            self.assertEqual(items[1].call_id, items[2].call_id)
            self.assertEqual(items[2].output, "Found")

    def test_auth_cors_all_routes_and_delete(self):
        _, _, response = self.post()
        path = "/v1/responses/" + response["id"]
        for method, route, body in (("POST", "/v1/responses", {}), ("GET", path, None),
                                    ("GET", path + "/input_items", None), ("DELETE", path, None)):
            code, headers, _ = self.request(method, route, body, headers={"Authorization": ""})
            self.assertEqual(code, 401)
            self.assertEqual(headers["Access-Control-Allow-Origin"], "https://client.example")
        code, headers, _ = self.request("OPTIONS", path)
        self.assertEqual(code, 204)
        self.assertIn("DELETE", headers["Access-Control-Allow-Methods"])
        self.assertEqual(self.request("GET", path)[2], response)
        code, _, deleted = self.request("DELETE", path)
        self.assertEqual(code, 200)
        self.assertEqual(deleted, {"id": response["id"], "object": "response.deleted", "deleted": True})
        self.assertEqual(self.request("GET", path)[0], 404)

    def test_keyless_host_guard_covers_responses_routes(self):
        _, _, response = self.post()
        path = "/v1/responses/" + response["id"]
        self.svc.api_key = ""
        for method, route, body in (("POST", "/v1/responses", {"model": self.svc.model, "input": "hi"}),
                                    ("GET", path, None), ("GET", path + "/input_items", None),
                                    ("DELETE", path, None), ("OPTIONS", path, None)):
            with self.subTest(method=method, route=route):
                code, _, _ = self.request(method, route, body, headers={"Host": "rebind.example"})
                self.assertEqual(code, 403)
        self.assertEqual(self.controller.get_response(response["id"]), response)
        self.assertEqual(len(self.store.records()), 1)

    def test_keyless_create_preserves_upstream_browser_guard(self):
        self.svc.api_key = ""
        body = {"model": self.svc.model, "input": "hi", "stream": True}
        for origin in ("https://foreign.example", "null"):
            for content_type in ("text/plain", "application/json"):
                with self.subTest(origin=origin, content_type=content_type):
                    code, headers, _ = self.request(body=body, headers={
                        "Origin": origin, "Content-Type": content_type})
                    self.assertEqual(code, 403)
                    self.assertNotEqual(headers.get("Content-Type"), "text/event-stream")
        self.assertEqual(self.request(body=body, headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.store.records(), [])
        self.assertFalse(self.controller.active)
        body["stream"] = False
        for headers in ({}, {"Origin": "", "Content-Type": "text/plain"}):
            # An explicitly allowed browser page, and an SDK/curl request with no Origin.
            code, _, response = self.request(body=body, headers=headers)
            self.assertEqual(code, 200)
            self.assertEqual(response["status"], "completed")

    def test_input_pagination_defaults_and_cursors(self):
        inputs = [{"role": "user", "content": str(i)} for i in range(23)]
        _, _, response = self.post(input=inputs)
        path = "/v1/responses/" + response["id"] + "/input_items"
        code, _, page = self.request("GET", path)
        self.assertEqual(code, 200)
        self.assertEqual(len(page["data"]), 20)
        self.assertTrue(page["has_more"])
        self.assertEqual(page["data"][0]["content"][0]["text"], "22")
        _, _, tail = self.request("GET", path + "?after=" + page["last_id"])
        self.assertEqual(len(tail["data"]), 3)
        self.assertFalse(tail["has_more"])
        _, _, asc = self.request("GET", path + "?order=asc&limit=1")
        self.assertEqual(asc["data"][0]["content"][0]["text"], "0")
        for query in ("limit=0", "limit=101", "order=bad", "after=missing", "limit=2&limit=3", "include=x"):
            self.assertEqual(self.request("GET", path + "?" + query)[0], 400)

    def test_disabled_routes_and_old_endpoints(self):
        self.svc.responses = None
        for method, path, body in (("POST", "/v1/responses", {}), ("GET", "/v1/responses/resp_x", None),
                                   ("GET", "/v1/responses/resp_x/input_items", None), ("DELETE", "/v1/responses/resp_x", None)):
            self.assertEqual(self.request(method, path, body)[0], 404)
        self.assertEqual(list(Path(self.tmp.name).glob("*.json")), [])
        for path, req in (("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
                          ("/v1/messages", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100})):
            self.assertEqual(self.request("POST", path, req)[0], 200)

    def test_http_validation_before_sse(self):
        for body in (b'{"model":"test-model","input":"hi","stream":true,"background":true}',
                     b'{"model":"test-model","input":"hi","input":"duplicate"}',
                     b'{"model":"test-model","input":"hi","stream":true,"store":false,"metadata":{"x":"\\ud800"}}',
                     b'{"model":"test-model","input":"hi","temperature":NaN}', b'[]', b'null', b'{bad'):
            code, headers, data = self.request(body=body)
            self.assertEqual(code, 400)
            self.assertEqual(headers["Content-Type"], "application/json")
            self.assertIn("error", data)
        self.assertEqual(self.post(previous_response_id="resp_missing", stream=True)[0], 404)
        self.assertEqual(self.post(max_output_tokens=10000000, stream=True)[0], 400)
        for path in ("/v1/responses/resp_x/cancel", "/v1/responses/compact"):
            self.assertEqual(self.request("POST", path, {})[0], 404)

    def test_corrupt_storage_is_an_http_error(self):
        _, _, response = self.post()
        path = Path(self.tmp.name) / (response["id"] + ".json")
        for contents in ("not json", "{}"):
            path.write_text(contents, encoding="utf-8")
            for suffix in ("", "/input_items"):
                code, _, error = self.request("GET", "/v1/responses/" + response["id"] + suffix)
                self.assertEqual(code, 500)
                self.assertEqual(error["error"]["type"], "server_error")

    @unittest.skipUnless(HAVE_JSONSCHEMA, "optional jsonschema dependency")
    def test_http_generation_failure_is_failed_response(self):
        self.script("invalid JSON")
        code, _, response = self.post(text=FORMAT)
        self.assertEqual(code, 200)
        self.assertEqual(response["status"], "failed")
        code, _, stream = self.post(stream=True, text=FORMAT)
        self.assertEqual(code, 200)
        self.assertIn("event: response.failed", stream)
        self.assertNotIn("event: response.completed", stream)

    def test_http_disconnect_stops_and_drains(self):
        self.script("x" * 10000, delay_s=0.002)
        conn = http.client.HTTPConnection(*self.httpd.server_address, timeout=5)
        conn.request("POST", "/v1/responses", json.dumps({"model": self.svc.model, "input": "hi", "stream": True}),
                     {"Authorization": "Bearer test-key", "Content-Type": "application/json"})
        response = conn.getresponse()
        response_id = None
        while True:
            line = response.readline().decode()
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if event["type"] == "response.created":
                    response_id = event["response"]["id"]
                if event["type"] == "response.output_text.delta":
                    break
        response.close()
        conn.close()
        deadline = time.monotonic() + 4
        while self.controller.active and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(self.controller.active)
        self.assertTrue(self.svc.engine.drained)
        self.assertEqual(self.controller.get_response(response_id)["status"], "cancelled")
        self.script("healthy")
        self.assertEqual(self.post()[2]["status"], "completed")


class StartupTests(unittest.TestCase):
    def test_feature_switch_defaults_config_and_cli(self):
        test_thread = threading.current_thread()

        def stop_main_loop(seconds):
            if threading.current_thread() is test_thread:
                raise KeyboardInterrupt
            time.sleep(seconds)

        for config, flag, enabled in (({}, False, False), ({"experimental_responses": False}, False, False),
                                      ({"experimental_responses": True}, False, True),
                                      ({"experimental_responses": False}, True, True)):
            with self.subTest(config=config, flag=flag), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "server.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                argv = ["server", "--engine", "mock", "--config", str(path), "--port", "0"]
                if flag:
                    argv.append("--experimental-responses")
                with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, {"STRATA_API_KEY": "test-key"}), \
                     mock.patch("serve.server.serve") as start, mock.patch("serve.server.signal.signal"), \
                     mock.patch("serve.server.time", wraps=time) as server_time:
                    server_time.sleep.side_effect = stop_main_loop
                    self.assertEqual(main(), 0)
                svc = start.call_args.args[0]
                self.assertEqual(svc.responses is not None, enabled)
                self.assertEqual((Path(directory) / ".responses").exists(), enabled)

    def test_invalid_config_boolean_exits_before_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.json"
            path.write_text('{"experimental_responses":"false"}', encoding="utf-8")
            with mock.patch.object(sys, "argv", ["server", "--config", str(path)]), \
                 mock.patch("serve.server.serve") as start, self.assertRaises(SystemExit):
                main()
            start.assert_not_called()
            self.assertFalse((Path(directory) / ".responses").exists())


if __name__ == "__main__":
    unittest.main()
