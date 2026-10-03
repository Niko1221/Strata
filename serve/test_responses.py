"""Responses contract tests over real HTTP + the existing semantic Service/MockEngine.

All scripted engine output is synthetic. Client capture/SDK receipts name their
provenance separately; these tests do not establish Codex or native GPU support.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import socket
from pathlib import Path
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock
from cryptography.fernet import Fernet

from serve import server
from serve.frontend import ChatTemplate, Event, ToolCall
from serve.responses import (RequestError, create_response, execute_response, resolve_input,
                             strict_json, transition, validate_request)
from serve.response_replay import KEY_ENV, ReplayCodec

ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen3.8-flash-next"


def service(script="Hello, 猫.", engine_class=server.MockEngine, **kwargs):
    tok = server.ByteTokenizer()
    engine = engine_class(tok, script, **kwargs)
    svc = server.Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    svc.experimental_responses = True
    svc.responses_replay = ReplayCodec(Fernet.generate_key())
    return svc


def request(**kwargs):
    return {"model": MODEL, "input": "Hello", "store": False, **kwargs}


@contextlib.contextmanager
def listening(svc):
    with mock.patch.object(svc, "start_telemetry"):
        httpd = server.serve(svc, port=0)
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def http(base, body=None, path="/v1/responses", headers=None, method=None):
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    req = urllib.request.Request(base + path, data=data,
        headers={"Content-Type": "application/json", **(headers or {})}, method=method)
    try:
        result = urllib.request.urlopen(req, timeout=15)
    except urllib.error.HTTPError as exc:
        result = exc
    with result:
        return result.status, dict(result.headers), result.read()


class Normalization(unittest.TestCase):
    def setUp(self):
        self.svc = service()

    def test_explicit_stateless_contract(self):
        for body in (request(store=True), {"model": MODEL, "input": "hi"}, request(store=0)):
            with self.subTest(body=body), self.assertRaises(RequestError) as err:
                create_response(self.svc, body)
            self.assertEqual(err.exception.param, "store")
        self.assertEqual(self.svc.engine.last_prompt, [])

    def test_priority_order_and_content_bearing_ids(self):
        req = request(instructions="top", input=[
            {"role": "developer", "content": "developer"}, {"role": "system", "content": "system"},
            {"role": "user", "content": [{"type": "input_text", "text": "first"}]},
            {"id": "msg_not_stored", "type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "answer", "annotations": [], "logprobs": []}]},
            {"role": "user", "content": "second"}])
        original = copy.deepcopy(req)
        normalized = resolve_input(validate_request(req, self.svc))
        self.assertEqual([m["role"] for m in normalized], ["system", "developer", "system", "user", "assistant", "user"])
        self.assertEqual([m["content"] for m in normalized], ["top", "developer", "system", "first", "answer", "second"])
        self.assertEqual(req, original)

    def test_unsupported_capabilities_rejected_before_load(self):
        cases = [dict(previous_response_id="resp_missing"), dict(background=True),
                 dict(text={"format": {"type": "json_schema", "schema": {}}}),
                 dict(text={"format": {"type": "json_object"}}), dict(text={"verbosity": "high"}),
                 dict(reasoning={"summary": "unknown"}), dict(include=["file_search_call.results"]),
                 dict(input=[{"type": "item_reference", "id": "msg_missing"}]),
                 dict(input=[{"role": "user", "content": [{"type": "input_image", "image_url": "file://private"}]}]),
                 dict(grammar="root ::= \"x\""), dict(response_format={"type": "json_object"}),
                 dict(conversation="conv_any"), dict(prompt_cache_key=42), dict(tool_choice="required"),
                 dict(tools=[{"type": "web_search"}]), dict(client_metadata={"bad_type": 42})]
        with mock.patch.object(self.svc, "load") as load:
            for extra in cases:
                with self.subTest(extra=extra), self.assertRaises(RequestError):
                    create_response(self.svc, request(**extra))
            load.assert_not_called()

    def test_bad_types_and_limits(self):
        for extra in (dict(model="missing"), dict(stream=1), dict(max_output_tokens=True),
                      dict(max_output_tokens=0), dict(input={}), dict(input=[]), dict(temperature=float("nan")),
                      dict(top_p=0), dict(top_p=2), dict(metadata={"a": 4}), dict(instructions=9)):
            with self.subTest(extra=extra), self.assertRaises(RequestError):
                create_response(self.svc, request(**extra))

    def test_json_transport_is_strict_without_schema_support(self):
        for raw in ('{"store":false,"store":true}', '{"x":NaN}', '{"x":1e999}', '{"x":"\\ud800"}', '{'):
            with self.subTest(raw=raw), self.assertRaises(RequestError):
                strict_json(raw)

    def test_known_harmless_metadata_and_effective_sampling(self):
        self.svc.sampling_defaults = {"temperature": 0.7, "top_p": 0.8, "top_k": 9}
        p = create_response(self.svc, request(metadata={"purpose": "fixture"}, reasoning={"effort": "none"}))
        self.assertEqual(p.sampling, {"temperature": 0.7, "top_p": 0.8})
        self.assertEqual(p.assembler.snapshot()["metadata"], {"purpose": "fixture"})


class Lifecycle(unittest.TestCase):
    def test_normal_execution_and_terminal_immutability(self):
        svc = service("hello")
        p = create_response(svc, request())
        events = list(execute_response(svc, p, threading.Event()))
        final = p.assembler.snapshot()
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["output"][0]["content"][0]["text"], "hello")
        self.assertEqual(final["usage"]["output_tokens"], 6)
        self.assertEqual(events[-1]["response"], final)
        for action in (lambda: p.assembler.append_output(Event("content", "late")),
                       lambda: p.assembler.finalize_response("fail")):
            with self.assertRaises(ValueError):
                action()
            self.assertEqual(p.assembler.snapshot(), final)
        final["output"].clear()
        self.assertEqual(len(p.assembler.snapshot()["output"]), 1)

    def test_budget_means_incomplete(self):
        svc = service("hello")
        p = create_response(svc, request(max_output_tokens=2))
        list(execute_response(svc, p, threading.Event()))
        result = p.assembler.snapshot()
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["incomplete_details"], {"reason": "max_output_tokens"})
        self.assertEqual(result["output"][0]["content"][0]["text"], "he")

    def test_bad_transition_does_not_invent_lifecycle_states(self):
        self.assertEqual(transition("queued", "start"), "in_progress")
        self.assertEqual(transition("in_progress", "output"), "in_progress")
        for state, event in (("queued", "output"), ("completed", "fail"), ("in_progress", "delete"),
                             ("in_progress", "request_cancel")):
            with self.assertRaises(ValueError):
                transition(state, event)


class HttpBoundary(unittest.TestCase):
    def test_flag_off_and_old_endpoints(self):
        svc = service("</think>old")
        svc.experimental_responses = False
        with listening(svc) as base:
            self.assertEqual(http(base, request())[0], 404)
            self.assertEqual(http(base, {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
                                  path="/v1/chat/completions")[0], 200)
        self.assertFalse(hasattr(svc, "response_store"))

    def test_auth_cors_and_host_origin_guards(self):
        svc = service()
        svc.api_key = "fixture-only"
        svc.cors_origins = ["https://client.example"]
        with listening(svc) as base:
            self.assertEqual(http(base, request())[0], 401)
            code, headers, body = http(base, request(), headers={"Authorization": "Bearer fixture-only",
                                                               "Origin": "https://client.example"})
            self.assertEqual(code, 200)
            self.assertEqual(headers["Access-Control-Allow-Origin"], "https://client.example")
            self.assertEqual(json.loads(body)["status"], "completed")
            self.assertEqual(http(base, method="OPTIONS", headers={"Origin": "https://client.example"})[0], 204)
            svc.api_key = ""
            self.assertEqual(http(base, request(), headers={"Host": "evil.example"})[0], 403)
            self.assertEqual(http(base, request(), headers={"Origin": "https://evil.example"})[0], 403)

    def test_no_storage_or_previous_instruction_inheritance(self):
        svc = service()
        with listening(svc) as base:
            _, _, body = http(base, request(instructions="FIRST_ONLY"))
            first = json.loads(body)
            self.assertEqual(http(base, path="/v1/responses/" + first["id"])[0], 404)
            self.assertEqual(http(base, request(input=[{"role": "user", "content": "old"}, *first["output"],
                                                      {"role": "user", "content": "new"}]))[0], 200)
            self.assertNotIn("FIRST_ONLY", svc.tok.decode(svc.engine.last_prompt))
            self.assertEqual(http(base, request(previous_response_id=first["id"]))[0], 400)

    def test_invalid_json_and_context_before_success(self):
        svc = service(max_context=1024)
        with listening(svc) as base:
            for raw in (b'{"store":false,"store":true}', b'[]', b'{"input":NaN}'):
                self.assertEqual(http(base, raw)[0], 400)
            self.assertEqual(http(base, request(max_output_tokens=100000))[0], 400)
            self.assertEqual(svc.engine.last_prompt, [])

    def test_monitor_observes_final_snapshot(self):
        svc = service()
        svc.api_monitor = True
        with listening(svc) as base:
            code, _, body = http(base, request())
            self.assertEqual(code, 200)
            self.assertEqual(json.loads(svc.api_requests[-1]["response"]), json.loads(body))


class Startup(unittest.TestCase):
    def test_cli_config_and_default_activation(self):
        for cfg, flag, expected in (({}, False, False), ({"experimental_responses": True}, False, True),
                                    ({}, True, True), ({"experimental_responses": False}, True, True)):
            with self.subTest(cfg=cfg, flag=flag), tempfile.TemporaryDirectory() as directory:
                config = Path(directory) / "config.json"
                config.write_text(json.dumps(cfg), encoding="utf-8")
                argv = ["server", "--port", "0", "--config", str(config), "--tokenizer", directory]
                if flag:
                    argv.append("--experimental-responses")
                with mock.patch("sys.argv", argv), mock.patch.object(server, "serve") as start, \
                     mock.patch.object(ReplayCodec, "load", return_value=ReplayCodec(Fernet.generate_key())) as key_load, \
                     mock.patch.object(server, "time", mock.Mock(wraps=time, sleep=mock.Mock(side_effect=KeyboardInterrupt))), \
                     mock.patch.object(server.signal, "signal"), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(server.main(), 0)
                self.assertIs(start.call_args.args[0].experimental_responses, expected)
                self.assertEqual(key_load.call_count, int(expected))


def sse_events(raw):
    events, comments = [], []
    for block in raw.decode("utf-8").split("\n\n"):
        if not block:
            continue
        if block.startswith(":"):
            comments.append(block)
            continue
        lines = block.splitlines()
        name = next(line[7:] for line in lines if line.startswith("event: "))
        event = json.loads("\n".join(line[6:] for line in lines if line.startswith("data: ")))
        if name != event["type"]:
            raise AssertionError("SSE name differs from JSON event type")
        events.append(event)
    return events, comments


def normalized(obj):
    if isinstance(obj, list):
        return [normalized(x) for x in obj]
    if isinstance(obj, dict):
        return {k: ("<volatile>" if k in ("id", "call_id", "created_at", "completed_at", "encrypted_content") and v is not None
                    else normalized(v)) for k, v in obj.items()}
    return obj


class Streaming(unittest.TestCase):
    def test_unicode_reassembly_matches_final_json(self):
        text = 'A 猫 🐈 e\u0301\n"quoted" \\ end'
        svc = service(text)
        with listening(svc) as base:
            code, _, body = http(base, request())
            self.assertEqual(code, 200)
            final = json.loads(body)
            code, headers, raw = http(base, request(stream=True))
        self.assertEqual(code, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        self.assertNotIn(b"[DONE]", raw)
        events, _ = sse_events(raw)
        self.assertEqual([e["sequence_number"] for e in events], list(range(len(events))))
        self.assertEqual([e["type"] for e in events[:4]], ["response.created", "response.in_progress",
            "response.output_item.added", "response.content_part.added"])
        self.assertEqual(events[0]["response"]["output"], [])
        item_id = events[2]["item"]["id"]
        text_so_far = ""
        for event in events[3:]:
            if "item_id" in event:
                self.assertEqual(event["item_id"], item_id)
                self.assertEqual(event["output_index"], 0)
                self.assertEqual(event["content_index"], 0)
            if event["type"] == "response.output_text.delta":
                text_so_far += event["delta"]
            elif event["type"] == "response.output_text.done":
                self.assertEqual(event["text"], text_so_far)
            elif event["type"] == "response.content_part.done":
                self.assertEqual(event["part"]["text"], text_so_far)
            elif event["type"] == "response.output_item.done":
                self.assertEqual(event["item"]["content"][0]["text"], text_so_far)
        self.assertEqual(text_so_far, text)
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(normalized(events[-1]["response"]), normalized(final))

    def test_empty_response_and_output_limit(self):
        for text, cap, expected in (("", 10, "completed"), ("abcdef", 2, "incomplete")):
            with self.subTest(expected=expected), listening(service(text)) as base:
                code, _, raw = http(base, request(stream=True, max_output_tokens=cap))
                self.assertEqual(code, 200)
                events, _ = sse_events(raw)
                self.assertEqual(events[-1]["type"], "response." + expected)
                self.assertEqual(sum(e["type"] in ("response.completed", "response.incomplete", "response.failed")
                                     for e in events), 1)
                if text:
                    self.assertEqual(events[-1]["response"]["output"][0]["content"][0]["text"], "ab")

    def test_preflight_error_precedes_stream_headers(self):
        svc = service()
        with listening(svc) as base:
            for body in (request(stream=True, store=True), request(stream=True, max_output_tokens=999999)):
                code, headers, _ = http(base, body)
                self.assertEqual(code, 400)
                self.assertEqual(headers["Content-Type"], "application/json")
            with mock.patch.object(svc, "load", side_effect=server.EngineStarting("not ready")):
                code, headers, _ = http(base, request(stream=True))
                self.assertEqual(code, 503)
                self.assertEqual(headers["Content-Type"], "application/json")

    def test_error_after_headers_is_failed_and_next_request_works(self):
        for fail_after in (0, 3):
            class FaultEngine(server.MockEngine):
                fail = True
                closed = False

                def generate(self, *args, **kwargs):
                    try:
                        if self.fail:
                            self.fail = False
                            yield None
                            for n, token in enumerate(super().generate(*args, **kwargs)):
                                if n == fail_after:
                                    raise server.EngineDied("scripted generation failure")
                                yield token
                        else:
                            yield from super().generate(*args, **kwargs)
                    finally:
                        self.closed = True

            svc = service("abcdef", engine_class=FaultEngine)
            with self.subTest(fail_after=fail_after), listening(svc) as base:
                code, _, raw = http(base, request(stream=True))
                self.assertEqual(code, 200)
                events, comments = sse_events(raw)
                self.assertIn(": keep-alive", comments)
                self.assertEqual(events[-1]["type"], "response.failed")
                self.assertNotIn("response.completed", [e["type"] for e in events])
                self.assertTrue(svc.engine.closed)
                self.assertFalse(svc.status["busy"])
                code, _, body = http(base, request())
                self.assertEqual(code, 200)
                self.assertEqual(json.loads(body)["output"][0]["content"][0]["text"], "abcdef")

    def test_missing_done_is_failure_not_completion(self):
        svc = service()
        p = create_response(svc, request())
        def broken(*args, **kwargs):
            yield "start", None
            yield "event", Event("content", "partial")
        with mock.patch.object(svc, "run", broken):
            events = list(execute_response(svc, p, threading.Event()))
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertIn("without a generation outcome", events[-1]["response"]["error"]["message"])

    def test_nonstream_failure_is_not_http_success(self):
        svc = service()
        def broken(*args, **kwargs):
            yield "start", None
            raise ValueError("scripted failure")
        with listening(svc) as base, mock.patch.object(svc, "run", broken):
            code, _, body = http(base, request())
            self.assertEqual(code, 500)
            self.assertEqual(json.loads(body)["status"], "failed")

    def test_iterator_close_confirms_stop_after_drain(self):
        svc = service()
        p = create_response(svc, request())
        cancel = threading.Event()
        observed = []
        def running(*args, **kwargs):
            try:
                yield "start", None
                yield "event", Event("content", "x")
                yield "ping", None
            finally:
                observed.append((cancel.is_set(), p.assembler.snapshot()["status"]))
        with mock.patch.object(svc, "run", running):
            events = execute_response(svc, p, cancel)
            while next(events)["type"] != "response.output_text.delta":
                pass
            cancel.set()
            self.assertEqual(p.assembler.snapshot()["status"], "in_progress")
            events.close()
        self.assertEqual(observed, [(True, "in_progress")])
        self.assertEqual(p.assembler.snapshot()["status"], "cancelled")

    def test_disconnect_queued_prefill_decode_and_next_request(self):
        for phase in ("queued", "prefill", "decode"):
            class BlockingEngine(server.MockEngine):
                calls = 0
                stopped = threading.Event()

                def generate(self, ids, max_new, sampling, cancel):
                    self.calls += 1
                    if phase != "queued" and self.calls == 1:
                        try:
                            if phase == "decode":
                                yield from self.tok.encode("partial")
                            while not cancel.wait(0.02):
                                yield None
                        finally:
                            self.stopped.set()
                        return
                    yield from super().generate(ids, max_new, sampling, cancel)

            svc = service("next request", engine_class=BlockingEngine)
            with self.subTest(phase=phase), listening(svc) as base:
                if phase == "queued":
                    svc.fifo.acquire()
                port = int(base.rsplit(":", 1)[1])
                sock = socket.create_connection(("127.0.0.1", port), timeout=5)
                body = json.dumps(request(stream=True)).encode()
                sock.sendall(b"POST /v1/responses HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n" +
                             f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
                target = b"response.created" if phase == "queued" else (
                    b"response.output_text.delta" if phase == "decode" else b": keep-alive")
                received = b""
                while target not in received:
                    received += sock.recv(65536)
                sock.shutdown(socket.SHUT_RDWR)
                sock.close()
                if phase == "queued":
                    time.sleep(0.65)  # existing disconnect watcher runs every 0.5 seconds
                    svc.fifo.release()
                else:
                    self.assertTrue(svc.engine.stopped.wait(4), "engine was not stopped/drained")
                deadline = time.monotonic() + 4
                while (svc.status["busy"] or svc.status["queued"]) and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertFalse(svc.status["busy"])
                self.assertEqual(svc.status["queued"], 0)
                if phase == "queued":
                    self.assertEqual(svc.engine.calls, 0)
                code, _, body = http(base, request())
                self.assertEqual(code, 200)
                self.assertEqual(json.loads(body)["output"][0]["content"][0]["text"], "next request")


def function_tool(name="echo", **kwargs):
    return {"type": "function", "name": name, "strict": False,
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}}, **kwargs}


def function_script(text, name="echo"):
    return f"<tool_call><function={name}><parameter=text>{text}</parameter></function></tool_call>"


class Functions(unittest.TestCase):
    def test_argument_deltas_and_items_are_exact(self):
        value = 'quote " slash \\ new\n猫'
        script = "First\n" + function_script(value) + function_script("second") + "\nLast"
        svc = service(script)
        with listening(svc) as base:
            _, _, body = http(base, request(tools=[function_tool()]))
            final = json.loads(body)
            _, _, raw = http(base, request(tools=[function_tool()], stream=True))
        self.assertEqual(final["status"], "completed")
        self.assertEqual([x["type"] for x in final["output"]], ["message", "function_call", "function_call", "message"])
        events, _ = sse_events(raw)
        self.assertEqual(normalized(events[-1]["response"]), normalized(final))
        items, accumulated = {}, {}
        for event in events:
            kind = event["type"]
            if kind == "response.output_item.added":
                index = event["output_index"]
                self.assertEqual(index, len(items))
                items[index] = event["item"]
                accumulated[index] = ""
                if event["item"]["type"] == "function_call":
                    self.assertNotEqual(event["item"]["id"], event["item"]["call_id"])
            elif kind in ("response.function_call_arguments.delta", "response.output_text.delta"):
                self.assertEqual(event["item_id"], items[event["output_index"]]["id"])
                accumulated[event["output_index"]] += event["delta"]
            elif kind in ("response.function_call_arguments.done", "response.output_text.done"):
                self.assertEqual(event.get("arguments", event.get("text")), accumulated[event["output_index"]])
        self.assertEqual(json.loads(accumulated[1]), {"text": value})
        self.assertEqual(json.loads(accumulated[2]), {"text": "second"})
        self.assertEqual([e["sequence_number"] for e in events], list(range(len(events))))

    def test_client_owned_function_result_replay(self):
        svc = service([function_script("argument"), "result acknowledged"])
        svc.mcp = mock.Mock()
        history = [{"role": "user", "content": "Use echo"}]
        with listening(svc) as base:
            _, _, body = http(base, request(input=history, tools=[function_tool()]))
            first = json.loads(body)
            call = first["output"][0]
            history += first["output"] + [{"type": "function_call_output", "call_id": call["call_id"],
                                           "output": "client-owned-result"}]
            before = copy.deepcopy(history)
            code, _, body = http(base, request(input=history, tools=[function_tool()]))
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["output"][0]["content"][0]["text"], "result acknowledged")
        self.assertEqual(history, before)
        self.assertIn("client-owned-result", svc.tok.decode(svc.engine.last_prompt))
        self.assertEqual(svc.mcp.mock_calls, [])

    def test_call_id_controls_result_binding(self):
        history = [{"role": "user", "content": "go"},
                   {"type": "function_call", "id": "fc_A", "call_id": "call_A", "name": "echo", "arguments": '{ "text" : "A" }'},
                   {"type": "function_call", "id": "fc_B", "call_id": "call_B", "name": "echo", "arguments": '{"text":"B"}'},
                   {"type": "function_call_output", "call_id": "call_B", "output": "result B"},
                   {"type": "function_call_output", "call_id": "call_A", "output": "result A"}]
        original = copy.deepcopy(history)
        messages = resolve_input(request(input=history))
        self.assertEqual([x["content"] for x in messages[-2:]], ["result A", "result B"])
        self.assertEqual(history, original)
        for bad in (history[:3], history[:2] + history[3:4], history + [history[-1]], history[:2] + [history[1]]):
            with self.subTest(bad=bad), self.assertRaises(RequestError):
                resolve_input(request(input=bad))

    def test_strict_omission_custom_invalid_namespace_and_choice_rejected(self):
        svc = service()
        missing_strict = function_tool()
        del missing_strict["strict"]
        bodies = [request(tools=[function_tool(strict=True)]), request(tools=[missing_strict]),
                  request(tools=[function_tool(strict=None)]),
                  request(tools=[{"type": "custom", "name": "patch", "format": {"type": "grammar", "syntax": "lark", "definition": 'start: "x"'}}]),
                  request(tools=[{"type": "namespace", "name": "group", "tools": [function_tool()]}]),
                  request(tools=[function_tool()], tool_choice={"type": "function", "name": "echo"}),
                  request(tools=[function_tool()], parallel_tool_calls=False),
                  request(tools=[function_tool(), function_tool()])]
        with mock.patch.object(svc, "load") as load:
            for body in bodies:
                with self.subTest(body=body), self.assertRaises(RequestError):
                    create_response(svc, body)
            load.assert_not_called()

    def test_non_strict_parameters_are_descriptions_not_enforced_schemas(self):
        tool = function_tool(parameters={"type": "object", "properties": {"text": {"type": "string", "enum": ["other"]}},
                                         "required": ["unprovided"], "additionalProperties": False})
        svc = service(function_script("not in enum"))
        p = create_response(svc, request(tools=[tool]))
        list(execute_response(svc, p, threading.Event()))
        self.assertEqual(p.assembler.snapshot()["status"], "completed")
        self.assertEqual(json.loads(p.assembler.snapshot()["output"][0]["arguments"]), {"text": "not in enum"})

    def test_disabled_or_undeclared_model_call_fails(self):
        for tools, choice in (([], "auto"), ([function_tool()], "none"), ([function_tool("other")], "auto")):
            with self.subTest(choice=choice, tools=tools):
                svc = service(function_script("x"))
                p = create_response(svc, request(tools=tools, tool_choice=choice))
                list(execute_response(svc, p, threading.Event()))
                self.assertEqual(p.assembler.snapshot()["status"], "failed")
                self.assertEqual(p.assembler.snapshot()["output"], [])
                self.assertIn("undeclared or disabled function: 'echo'", p.assembler.snapshot()["error"]["message"])

    def test_partial_function_budget_is_not_repaired(self):
        svc = service(function_script("a long argument"))
        cap = len(svc.tok.encode("<tool_call><function=echo><parameter=text>a l"))
        p = create_response(svc, request(tools=[function_tool()], max_output_tokens=cap))
        events = list(execute_response(svc, p, threading.Event()))
        final = p.assembler.snapshot()
        self.assertEqual(final["status"], "incomplete")
        args = final["output"][0]["arguments"]
        self.assertEqual(args, "".join(e["delta"] for e in events if e and e["type"] == "response.function_call_arguments.delta"))
        with self.assertRaises(ValueError):
            json.loads(args)

    def test_normal_stop_inside_function_is_failure(self):
        svc = service("<tool_call><function=echo><parameter=text>unfinished")
        p = create_response(svc, request(tools=[function_tool()]))
        list(execute_response(svc, p, threading.Event()))
        self.assertEqual(p.assembler.snapshot()["status"], "failed")

    def test_unsupported_reasoning_and_opaque_replay_rejected(self):
        svc = service()
        for body in (request(reasoning={"effort": "minimal"}), request(reasoning={"effort": "none", "summary": "auto"}),
                     request(reasoning={"context": "all_turns"}), request(reasoning={"mode": "pro"}),
                     request(include=["file_search_call.results"]),
                     request(input=[{"type": "reasoning", "id": "rs_x", "summary": [], "encrypted_content": "opaque"}])):
            with self.subTest(body=body), self.assertRaises(RequestError):
                create_response(svc, body)
        p = create_response(svc, request())
        def unexpected(*args, **kwargs):
            yield "start", None
            yield "event", Event("reasoning", "raw thinking")
        with mock.patch.object(svc, "run", unexpected):
            list(execute_response(svc, p, threading.Event()))
        self.assertEqual(p.assembler.snapshot()["status"], "failed")
        self.assertEqual(p.assembler.snapshot()["output"], [])

    def test_codex_initial_request_passes_supported_capability_gate(self):
        fixture = ROOT / "docs/responses-evidence/R0/request-fixtures/codex-initial-1.json"
        body = json.loads(fixture.read_text(encoding="utf-8"))["body"]
        svc = service()
        svc.engine.max_context = 131072  # byte tokenizer, not a claim about native model token counts
        prepared = create_response(svc, body)
        self.assertTrue(prepared.thinking)
        self.assertGreater(prepared.summary_reserve, 0)
        self.assertIn("multi_agent_v1.spawn_agent", prepared.assembler.allowed_tools)
        self.assertEqual(prepared.assembler.snapshot()["prompt_cache_key"], body["prompt_cache_key"])

    def test_namespaces_keep_same_named_members_distinct_and_replay(self):
        tools = [function_tool()] + [{"type": "namespace", "name": name, "description": name + " functions",
                                     "tools": [function_tool()]} for name in ("one", "two")]
        svc = service("".join(function_script(name, name) for name in ("echo", "one.echo", "two.echo")))
        history = [{"role": "user", "content": "Use each function"}]
        with listening(svc) as base:
            _, _, body = http(base, request(input=history, tools=tools))
            final = json.loads(body)
            _, _, raw = http(base, request(input=history, tools=tools, stream=True))
        events, _ = sse_events(raw)
        self.assertEqual(normalized(events[-1]["response"]), normalized(final))
        self.assertEqual([(x.get("namespace"), x["name"]) for x in final["output"]],
                         [(None, "echo"), ("one", "echo"), ("two", "echo")])
        self.assertEqual(final["tools"], tools)
        history += final["output"]
        history += [{"type": "function_call_output", "call_id": item["call_id"], "output": item.get("namespace", "flat")}
                    for item in reversed(final["output"])]
        messages = resolve_input(request(input=history))
        self.assertEqual([x["function"]["name"] for x in messages[1]["tool_calls"]], ["echo", "one.echo", "two.echo"])
        self.assertEqual([x["content"] for x in messages[2:]], ["flat", "one", "two"])

    def test_namespace_validation_before_model_load(self):
        group = {"type": "namespace", "name": "files", "description": "Files", "tools": [function_tool()]}
        cases = [[group, group], [{**group, "tools": [function_tool(), function_tool()]}],
                 [{**group, "name": "files.dot"}], [{**group, "tools": []}],
                 [{**group, "tools": [function_tool(strict=True)]}], [{**group, "tools": [group]}],
                 [{**group, "tools": [{"type": "custom", "name": "raw"}]}]]
        svc = service()
        with mock.patch.object(svc, "load") as load:
            for tools in cases:
                with self.subTest(tools=tools), self.assertRaises(RequestError):
                    create_response(svc, request(tools=tools))
            load.assert_not_called()


class Reasoning(unittest.TestCase):
    def test_summary_quotes_tool_markup_as_text(self):
        summary = ('The fixture contains <tool_call>{"x":"a=b"}</tool_call> and '
                   '<tool_call><function=unexpected><parameter=value>literal</parameter>'
                   '</function></tool_call>. Treat both as data. \u732b\n')
        svc = service(["Inspect the literal fixture.</think>Answer.", summary])
        p = create_response(svc, request(reasoning={"summary": "auto"}, max_output_tokens=1024))
        events = list(execute_response(svc, p, threading.Event()))
        final = p.assembler.snapshot()
        self.assertEqual(final["status"], "completed", final["error"])
        reason = final["output"][0]
        self.assertEqual(reason["summary"], [{"type": "summary_text", "text": summary}])
        self.assertEqual("".join(e["delta"] for e in events if e["type"] == "response.reasoning_summary_text.delta"), summary)
        self.assertEqual([i["type"] for i in final["output"]], ["reasoning", "message"])
        self.assertEqual(sum(e["type"] == "response.completed" for e in events), 1)

    def test_encryption_failure_keeps_already_streamed_text(self):
        for summary in (None, "auto"):
            svc = service(["raw thought</think>answer", "genuine summary"])
            with mock.patch.object(svc.responses_replay, "seal", side_effect=ValueError("scripted encryption failure")):
                p = create_response(svc, request(reasoning={"effort": "low", "summary": summary}, max_output_tokens=256))
                events = list(execute_response(svc, p, threading.Event()))
            final = p.assembler.snapshot()
            self.assertEqual(final["status"], "failed")
            self.assertEqual(final["output"][0]["content"][0]["text"], "raw thought")
            if summary:
                streamed = "".join(e["delta"] for e in events if e["type"] == "response.reasoning_summary_text.delta")
                self.assertEqual(final["output"][0]["summary"][0]["text"], streamed)
            self.assertEqual(sum(e["type"] == "response.failed" for e in events), 1)

    def test_genuine_summary_events_and_encrypted_only_replay(self):
        thought, summary = "Read the file and compare its contents.", "The model planned a file comparison."
        svc = service([thought + "</think>Answer.", summary, "Replay worked."])
        req = request(reasoning={"effort": "low", "summary": "auto"}, include=["reasoning.encrypted_content"],
                      max_output_tokens=512)
        p = create_response(svc, req)
        events = list(execute_response(svc, p, threading.Event()))
        final = p.assembler.snapshot()
        self.assertEqual(final["status"], "completed")
        reason = final["output"][0]
        self.assertEqual(reason["summary"], [{"type": "summary_text", "text": summary}])
        self.assertNotEqual(reason["summary"][0]["text"], reason["content"][0]["text"])
        self.assertEqual("".join(e["delta"] for e in events if e["type"] == "response.reasoning_summary_text.delta"), summary)
        self.assertEqual(final["usage"]["output_tokens"], len(svc.tok.encode(thought + "</think>Answer." + summary)) + 2)
        self.assertLessEqual(final["usage"]["output_tokens"], req["max_output_tokens"])
        self.assertEqual(sum(e["type"] == "response.in_progress" for e in events), 1)
        self.assertEqual([e["item"] for e in events if e["type"] == "response.output_item.done" and e["output_index"] == 0], [reason])
        closed = next(i for i, e in enumerate(events) if e["type"] == "response.output_item.done" and e["output_index"] == 0)
        opened = next(i for i, e in enumerate(events) if e["type"] == "response.output_item.added" and e["output_index"] == 1)
        self.assertLess(closed, opened)
        encrypted_only = {k: v for k, v in reason.items() if k != "content"}
        history = [{"role": "user", "content": "First question"}, encrypted_only, final["output"][1],
                   {"role": "user", "content": "Continue"}]
        again = create_response(svc, request(input=history))
        list(execute_response(svc, again, threading.Event()))
        self.assertIn(thought, svc.tok.decode(svc.engine.last_prompt))
        self.assertEqual(again.assembler.snapshot()["output"][0]["content"][0]["text"], "Replay worked.")

    def test_replay_key_restart_and_tampering(self):
        key = Fernet.generate_key().decode("ascii")
        with mock.patch.dict(os.environ, {KEY_ENV: key}):
            svc = service("visible thought</think>answer")
            svc.responses_replay = ReplayCodec.load()
            p = create_response(svc, request(reasoning={"effort": "low"}))
            list(execute_response(svc, p, threading.Event()))
            item = p.assembler.snapshot()["output"][0]
            original = copy.deepcopy(item)
            restored = ReplayCodec.load().restore(MODEL, item)
            self.assertEqual(restored["content"], item["content"])
            self.assertEqual(item, original)
            broken = item["encrypted_content"][:-6] + "X" + item["encrypted_content"][-5:]
            cases = [(MODEL, {**item, "encrypted_content": broken}), ("different-model", item),
                     (MODEL, {**item, "id": "rs_wrong"}), (MODEL, {**item, "summary": [{"type": "summary_text", "text": "forged"}]}),
                     (MODEL, {**item, "content": [{"type": "reasoning_text", "text": "forged"}]}),
                     (MODEL, {**item, "encrypted_content": "another-provider-token"})]
            for model, wrong in cases:
                with self.subTest(item=wrong), self.assertRaises(ValueError):
                    svc.responses_replay.restore(model, wrong)
            with self.assertRaises(ValueError):
                ReplayCodec(Fernet.generate_key()).restore(MODEL, item)
        for invalid in ("", "invalid key"):
            with mock.patch.dict(os.environ, {KEY_ENV: invalid}), self.assertRaises(ValueError):
                ReplayCodec.load()

    def test_summary_limit_and_failure_are_terminal_once(self):
        for script, expected in (("long summary " * 20, "incomplete"),
                                 (None, "failed")):
            # Function-shaped text is literal in a summary. Exercise a genuine
            # service failure separately from exhausting the summary budget.
            svc = service(["short thought</think>answer", script or ""])
            original_run = svc.run

            def run(*args, **kwargs):
                if script is None and kwargs.get("parse_tools") is False:
                    raise RuntimeError("scripted summary failure")
                yield from original_run(*args, **kwargs)

            svc.run = run
            p = create_response(svc, request(reasoning={"summary": "auto"}, max_output_tokens=80))
            events = list(execute_response(svc, p, threading.Event()))
            final = p.assembler.snapshot()
            self.assertEqual(final["status"], expected)
            self.assertEqual(sum(e["type"] in ("response.completed", "response.failed", "response.incomplete") for e in events), 1)
            self.assertEqual(final["output"][0]["status"], "incomplete")
            self.assertNotIn("encrypted_content", final["output"][0])
            if expected == "incomplete":
                self.assertEqual(final["usage"]["output_tokens"], 80)
            self.assertFalse(svc.status["busy"])

    def test_disconnect_during_summary_and_clean_next_request(self):
        svc = service(["think</think>answer", "summary text", "clean next answer"])
        p = create_response(svc, request(reasoning={"summary": "auto"}, max_output_tokens=128))
        cancel = threading.Event()
        events = execute_response(svc, p, cancel)
        while next(events)["type"] != "response.reasoning_summary_text.delta":
            pass
        self.assertEqual(p.assembler.snapshot()["status"], "in_progress")
        events.close()
        self.assertTrue(cancel.is_set())
        self.assertFalse(svc.status["busy"])
        self.assertEqual(p.assembler.snapshot()["status"], "cancelled")
        next_response = create_response(svc, request())
        list(execute_response(svc, next_response, threading.Event()))
        self.assertEqual(next_response.assembler.snapshot()["output"][0]["content"][0]["text"], "clean next answer")

    def test_visible_reasoning_stream_json_and_tools_match(self):
        thought = 'Check the file, then answer. \u732b\n'
        script = thought + "</think>Answer. " + function_script("x", "files.echo")
        tools = [{"type": "namespace", "name": "files", "description": "File helpers", "tools": [function_tool()]}]
        svc = service(script)
        with listening(svc) as base:
            _, _, body = http(base, request(tools=tools, reasoning={"effort": "low"}))
            final = json.loads(body)
            _, _, raw = http(base, request(tools=tools, reasoning={"effort": "low"}, stream=True))
        events, _ = sse_events(raw)
        self.assertEqual(normalized(events[-1]["response"]), normalized(final))
        self.assertEqual(final["status"], "completed")
        self.assertEqual([x["type"] for x in final["output"]], ["reasoning", "message", "function_call"])
        item = final["output"][0]
        self.assertEqual(item["content"], [{"type": "reasoning_text", "text": thought}])
        self.assertEqual(item["summary"], [])
        restored = svc.responses_replay.restore(MODEL, item)
        self.assertEqual(restored["content"], item["content"])
        text = ""
        for event in events:
            if event["type"].startswith("response.reasoning_text."):
                self.assertEqual((event["output_index"], event["content_index"]), (0, 0))
                self.assertEqual(event["item_id"], events[2]["item"]["id"])
                if event["type"].endswith("delta"):
                    text += event["delta"]
                else:
                    self.assertEqual(event["text"], text)
        self.assertEqual(text, thought)
        self.assertEqual(final["usage"]["output_tokens_details"]["reasoning_tokens"], len(svc.tok.encode(thought + "</think>")))
        self.assertEqual([e["sequence_number"] for e in events], list(range(len(events))))

    def test_visible_replay_has_no_cache_or_instruction_dependency(self):
        item = {"type": "reasoning", "id": "rs_visible", "summary": [], "status": "completed",
                "content": [{"type": "reasoning_text", "text": "Earlier visible reasoning."}]}
        call = {"type": "function_call", "call_id": "call_old", "name": "echo", "namespace": "files", "arguments": '{}'}
        for following in ({"role": "assistant", "content": "Earlier answer."}, call):
            history = [{"role": "user", "content": "Earlier question"}, item, following]
            if following is call:
                history.append({"type": "function_call_output", "call_id": "call_old", "output": "client result"})
            history.append({"role": "user", "content": "Continue"})
            original = copy.deepcopy(history)
            messages = resolve_input(request(input=history))
            self.assertEqual(messages[1]["reasoning_content"], "Earlier visible reasoning.")
            fresh = service("New answer")
            p = create_response(fresh, request(input=history))
            list(execute_response(fresh, p, threading.Event()))
            self.assertIn("<think>\nEarlier visible reasoning.\n</think>", fresh.tok.decode(fresh.engine.last_prompt))
            self.assertEqual(history, original)
        for bad in ([item], [item, {"role": "user", "content": "Continue"}],
                    [{**item, "encrypted_content": "opaque reasoning replay state"}, {"role": "assistant", "content": "answer"}],
                    [{**item, "summary": [{"type": "summary_text", "text": "A summary"}]}, {"role": "assistant", "content": "answer"}]):
            with self.subTest(history=bad), self.assertRaises(RequestError):
                resolve_input(request(input=bad))

    def test_effort_mapping_and_budget(self):
        for effort, native in (("none", None), ("low", "low"), ("medium", "medium"), ("high", "xhigh"),
                               ("xhigh", "xhigh"), ("max", "xhigh")):
            svc = service("abcdef</think>answer")
            with mock.patch.object(svc, "prepare", wraps=svc.prepare) as prepare:
                p = create_response(svc, request(reasoning={"effort": effort}, max_output_tokens=3))
            self.assertEqual(prepare.call_args.args[2].get("reasoning_effort"), native)
            self.assertEqual(p.thinking, effort != "none")
            events = list(execute_response(svc, p, threading.Event()))
            final = p.assembler.snapshot()
            self.assertEqual(final["status"], "incomplete")
            self.assertEqual(final["output"][0]["content"][0]["text"], "abc")
            self.assertEqual(final["output"][0]["status"], "incomplete")
            self.assertEqual(final["usage"]["output_tokens_details"]["reasoning_tokens"], 0 if effort == "none" else 3)
            self.assertEqual(events[-1]["response"], final)

    def test_reasoning_disconnect_drains_before_terminal(self):
        svc = service("thinking for a long time</think>answer")
        p = create_response(svc, request(reasoning={"effort": "low"}))
        cancel = threading.Event()
        stream = execute_response(svc, p, cancel)
        while next(stream)["type"] != "response.reasoning_text.delta":
            pass
        stream.close()
        self.assertTrue(cancel.is_set())
        self.assertFalse(svc.status["busy"])
        self.assertEqual(p.assembler.snapshot()["status"], "cancelled")
        later = create_response(svc, request(reasoning={"effort": "low"}))
        list(execute_response(svc, later, threading.Event()))
        self.assertEqual(later.assembler.snapshot()["status"], "completed")


if __name__ == "__main__":
    unittest.main()
