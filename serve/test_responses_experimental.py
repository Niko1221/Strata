"""Experimental Responses contracts, with a mock engine and real HTTP / durable storage."""
import copy
import contextlib
import http.client
import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from serve import response_runner, responses
from serve.response_store import ResponseStore
from serve.test_responses import Server, ANSWER, TOOLS, CALL, codex_request


class Store(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ResponseStore(self.tmp.name)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def record(self, text, parent=None, status="completed"):
        req, context = self.store.resolve({"input": text, "previous_response_id": parent}, "m")
        asm = responses.Assembler(req, "m", 3, {}, False)
        self.store.begin(asm.response, context)
        if status != "in_progress":
            asm.response.update(status=status, output=[{"id": responses.new_id("msg"), "type": "message",
                "role": "assistant", "content": [{"type": "output_text", "text": "answer " + text}]}])
            self.store.finish(asm.response)
        return asm.response

    def test_branch_restart_and_parent_deletion(self):
        root = self.record("root")
        left = self.record("left", root["id"])
        right = self.record("right", root["id"])
        self.store.close()
        self.store = ResponseStore(self.tmp.name)
        self.assertEqual(self.store.get(left["id"]), left)
        self.store.delete(root["id"])
        with self.assertRaises(responses.ResponsesError):
            self.store.get(root["id"])
        req, _ = self.store.resolve({"input": "next", "previous_response_id": left["id"]}, "m")
        self.assertIn("root", json.dumps(req))
        self.assertIn("left", json.dumps(req))
        self.assertNotIn("right", json.dumps(req))
        self.store.delete(left["id"])
        self.store.delete(right["id"])
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM responses").fetchone()[0], 0)

    def test_active_delete_does_not_resurrect(self):
        item = self.record("active", status="in_progress")
        self.store.delete(item["id"])
        item["status"] = "completed"
        self.store.finish(item)
        with self.assertRaises(responses.ResponsesError):
            self.store.get(item["id"])
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM responses").fetchone()[0], 0)

    def test_restart_marks_unfinished_failed(self):
        item = self.record("interrupted", status="in_progress")
        self.store.close()
        self.store = ResponseStore(self.tmp.name)
        stored = self.store.get(item["id"])
        self.assertEqual(stored["status"], "failed")
        self.assertEqual(stored["error"]["code"], "server_restarted")

    def test_terminal_immutable_incomplete_resumable_and_model_checked(self):
        item = self.record("short", status="incomplete")
        changed = copy.deepcopy(item)
        changed["status"] = "completed"
        self.store.finish(changed)
        self.assertEqual(self.store.get(item["id"])["status"], "incomplete")
        self.store.resolve({"input": "more", "previous_response_id": item["id"]}, "m")
        with self.assertRaises(responses.ResponsesError):
            self.store.resolve({"input": "more", "previous_response_id": item["id"]}, "another-model")

    def test_capacity_expiry_and_single_owner(self):
        with self.assertRaises(ValueError):
            ResponseStore(self.tmp.name)
        self.store.max_bytes = 1024
        with self.assertRaises(responses.ResponsesError) as caught:
            self.record("x" * 4096)
        self.assertEqual(caught.exception.status, 507)
        self.store.max_bytes = 1024 * 1024
        root = self.record("expires")
        self.store.retention_s = 1
        with patch("serve.response_store.time.time", return_value=time.time() + 2):
            with self.assertRaises(responses.ResponsesError):
                self.store.get(root["id"])

    def test_pagination(self):
        root = self.record("root")
        child = self.record("child", root["id"])
        page = self.store.input_items(child["id"], {"order": ["asc"], "limit": ["1"]})
        self.assertTrue(page["has_more"])
        next_page = self.store.input_items(child["id"], {"order": ["asc"], "after": [page["last_id"]]})
        self.assertEqual(len(next_page["data"]), 2)
        self.assertNotEqual(page["last_id"], next_page["first_id"])


class PersistenceHttp(Server):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.svc.response_store = ResponseStore(self.tmp.name)

    def tearDown(self):
        super().tearDown()
        self.svc.response_store.close()
        self.tmp.cleanup()

    def request(self, method, rid, suffix="", headers=None):
        client = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            client.request(method, "/v1/responses/" + rid + suffix, headers=headers or {})
            result = client.getresponse()
            return result.status, json.loads(result.read())
        finally:
            client.close()

    def test_create_chain_retrieve_and_instructions_do_not_carry(self):
        code, first = self.post({"input": "first", "instructions": "OLD INSTRUCTIONS"})
        self.assertEqual(code, 200)
        self.assertTrue(first["store"])
        code, child = self.post({"input": "second", "previous_response_id": first["id"],
                                 "instructions": "NEW INSTRUCTIONS"})
        self.assertEqual(code, 200)
        self.assertEqual(child["previous_response_id"], first["id"])
        prompt = self.tok.decode(self.engine.last_prompt)
        self.assertIn("first", prompt)
        self.assertIn("second", prompt)
        self.assertIn("NEW INSTRUCTIONS", prompt)
        self.assertNotIn("OLD INSTRUCTIONS", prompt)
        self.assertEqual(self.request("GET", child["id"]), (200, child))
        self.assertEqual(self.request("DELETE", first["id"])[0], 200)
        self.assertEqual(self.request("GET", first["id"])[0], 404)
        self.assertEqual(self.post({"input": "third", "previous_response_id": child["id"]})[0], 200)

    def test_store_false_reads_parent_but_retains_no_child(self):
        _, first = self.post({"input": "first"})
        code, child = self.post({"input": "second", "previous_response_id": first["id"], "store": False})
        self.assertEqual(code, 200)
        self.assertFalse(child["store"])
        self.assertEqual(child["previous_response_id"], first["id"])
        self.assertEqual(self.request("GET", child["id"])[0], 404)

    def test_persistence_keeps_title_fast_path(self):
        self.svc.codex_thread_titles = True
        with patch.object(self.svc, "run", side_effect=AssertionError("title must not generate")):
            code, result = self.post(codex_request("thread_title",
                meta={"x-codex-turn-metadata": json.dumps({"thread_source": "thread_title"})}))
        self.assertEqual(code, 200)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.request("GET", result["id"])[1], result)

    def test_stream_terminal_matches_durable_get(self):
        code, events = self.post({"input": "first", "stream": True})
        self.assertEqual(code, 200)
        final = events[-1]["response"]
        self.assertEqual(self.request("GET", final["id"]), (200, final))
        self.assertEqual([e["sequence_number"] for e in events], list(range(len(events))))

    def test_interrupted_reasoning_reaches_the_prompt(self):
        code, _ = self.post({"input": [{"role": "user", "content": "first"},
            {"type": "reasoning", "status": "completed", "summary": [],
             "content": [{"type": "reasoning_text", "text": "standalone completed thought"}]},
            {"role": "user", "content": "continue"}]})
        self.assertEqual(code, 200)
        self.assertIn("standalone completed thought", self.tok.decode(self.engine.last_prompt))

    def test_auth_origin_and_pagination(self):
        _, first = self.post({"input": "first"})
        self.svc.api_key = "test-key"
        self.assertEqual(self.request("GET", first["id"])[0], 401)
        self.assertEqual(self.request("DELETE", first["id"])[0], 401)
        self.assertEqual(self.request("GET", first["id"], "/input_items?order=asc&limit=1",
                                     {"Authorization": "Bearer test-key"})[0], 200)
        self.svc.api_key = None
        self.assertEqual(self.request("DELETE", first["id"], headers={"Origin": "https://evil.invalid"})[0], 403)
        self.assertEqual(self.request("DELETE", first["id"], headers={"Origin": f"http://127.0.0.1:{self.port}"})[0], 200)

    def test_disabled_does_not_load_old_records(self):
        store = self.svc.response_store
        try:
            _, first = self.post({"input": "first"})
            self.svc.response_store = None
            self.assertEqual(self.request("GET", first["id"])[0], 404)
            self.assertEqual(self.post({"input": "second", "previous_response_id": first["id"]})[0], 400)
            self.assertFalse(self.post({"input": "standalone"})[1]["store"])
        finally:
            self.svc.response_store = store

    def test_failed_generation_is_persisted(self):
        def broken(*args, **kwargs):
            raise ValueError("injected failure")
            yield
        with patch.object(self.svc, "run", broken):
            code, events = self.post({"input": "fail", "stream": True})
        self.assertEqual(code, 200)
        self.assertEqual(events[-1]["type"], "response.failed")
        stored = self.request("GET", events[0]["response"]["id"])[1]
        self.assertEqual(stored["status"], "failed")
        self.assertEqual(stored["error"]["message"], "injected failure")
        self.assertEqual(stored, events[-1]["response"])
        self.assertEqual([e["sequence_number"] for e in events], list(range(len(events))))
        self.assertEqual(self.post({"input": "good"})[0], 200)

    def test_store_full_at_completion_never_acknowledges_success(self):
        original = self.svc.response_store._capacity
        calls = 0
        def capacity(extra):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise responses.ResponsesError("full", code="response_store_full", status=507)
            return original(extra)
        with patch.object(self.svc.response_store, "_capacity", capacity):
            code, events = self.post({"input": "hi", "stream": True})
        self.assertEqual(code, 200)
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertNotIn("response.completed", [e["type"] for e in events])
        self.assertEqual([e["sequence_number"] for e in events], list(range(len(events))))
        self.assertEqual(self.request("GET", events[-1]["response"]["id"])[1], events[-1]["response"])

    def test_disk_error_at_admission_is_a_json_error(self):
        with patch.object(self.svc.response_store, "_capacity", side_effect=sqlite3.OperationalError("disk full")):
            code, body = self.post({"input": "hi"})
        self.assertEqual(code, 500)
        self.assertEqual(body["error"]["code"], "response_store_error")

    def test_official_sdk_create_retrieve_continue_list_delete(self):
        try:
            from openai import OpenAI
        except ImportError:
            self.skipTest("optional OpenAI SDK is not installed")
        with OpenAI(base_url=f"http://127.0.0.1:{self.port}/v1", api_key="test", max_retries=0) as client:
            first = client.responses.create(model="strata", input="first", store=True)
            self.assertEqual(first.status, "completed")
            self.assertEqual(client.responses.retrieve(first.id).output_text, first.output_text)
            child = client.responses.create(model="strata", input="second", previous_response_id=first.id)
            self.assertEqual(child.previous_response_id, first.id)
            page = client.responses.input_items.list(child.id, order="asc", limit=1)
            self.assertEqual(len(page.data), 1)
            self.assertTrue(page.has_more)
            client.responses.delete(child.id)
            self.assertEqual(self.request("GET", child.id)[0], 404)

    def test_tool_output_chaining_and_summary_persistence(self):
        self.engine.scripts = [self.tok.encode(s, parse_special=True) + self.tok.encode("<|im_end|>", parse_special=True)
                               for s in [CALL, "Looked at the file.", ANSWER]]
        self.svc.response_summaries = True
        code, first = self.post({"input": "read file", "tools": TOOLS, "reasoning": {"summary": "concise"}})
        self.assertEqual(code, 200)
        self.assertEqual(first["output"][0]["summary"][0]["text"], "Looked at the file.")
        self.assertEqual(self.request("GET", first["id"])[1], first)
        call = next(item for item in first["output"] if item["type"] == "function_call")
        code, second = self.post({"input": [{"type": "function_call_output", "call_id": call["call_id"],
                                              "output": "before"}], "previous_response_id": first["id"], "tools": TOOLS})
        self.assertEqual(code, 200)
        self.assertEqual(second["status"], "completed")
        self.assertIn("cat a.txt", self.tok.decode(self.engine.last_prompt))


class SummaryHttp(Server):
    script = [ANSWER, "It read the file before answering."]

    def test_disabled_skips_second_pass_even_when_requested(self):
        code, result = self.post({"input": "hi", "reasoning": {"summary": "concise"}, "max_output_tokens": 256})
        self.assertEqual(code, 200)
        self.assertEqual(self.engine.turns, 1)
        self.assertEqual(result["output"][0]["summary"], [])
        self.assertEqual(result["output"][1]["content"][0]["text"], "The file says before.")

    def test_enabled_genuine_summary_event_order_and_combined_budget(self):
        self.svc.response_summaries = True
        code, events = self.post({"input": "hi", "stream": True, "reasoning": {"summary": "concise"},
                                  "max_output_tokens": 256})
        self.assertEqual(code, 200)
        self.assertEqual(self.engine.turns, 2)
        self.assertEqual([e["sequence_number"] for e in events], list(range(len(events))))
        types = [e["type"] for e in events]
        self.assertLess(types.index("response.reasoning_summary_text.done"), types.index("response.output_item.done"))
        final = events[-1]["response"]
        self.assertEqual(final["status"], "completed")
        reasoning = final["output"][0]
        self.assertEqual(reasoning["summary"][0]["text"], self.script[1])
        self.assertEqual(reasoning["content"][0]["text"], "Read it.\n")
        self.assertLessEqual(final["usage"]["output_tokens"], 256)
        self.assertEqual(final["usage"]["output_tokens"], sum(len(self.tok.encode(s)) + 1 for s in self.script))
        self.assertEqual(final["usage"]["total_tokens"], final["usage"]["input_tokens"] + final["usage"]["output_tokens"])

    def test_enabled_without_request_still_one_pass(self):
        self.svc.response_summaries = True
        self.assertEqual(self.post({"input": "hi"})[0], 200)
        self.assertEqual(self.engine.turns, 1)

    def test_no_reasoning_does_not_generate_a_summary(self):
        self.svc.response_summaries = True
        code, result = self.post({"input": "hi", "reasoning": {"effort": "none", "summary": "concise"}})
        self.assertEqual(code, 200)
        self.assertEqual(self.engine.turns, 1)
        self.assertTrue(all(item["type"] != "reasoning" for item in result["output"]))

    def test_summary_failure_is_not_success_and_fifo_is_released(self):
        self.svc.response_summaries = True
        original = self.svc.run
        count = 0
        def fail_second(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise ValueError("summary failed")
            yield from original(*args, **kwargs)
        with patch.object(self.svc, "run", fail_second):
            code, events = self.post({"input": "hi", "stream": True, "reasoning": {"summary": "concise"}})
        self.assertEqual(code, 200)
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertEqual(self.post({"input": "next"})[0], 200)

    def test_invalid_or_tiny_summary_budget(self):
        self.svc.response_summaries = True
        for body in ({"reasoning": {"summary": "wrong"}},
                     {"reasoning": {"summary": "concise"}, "max_output_tokens": 3}):
            self.assertEqual(self.post({"input": "hi", **body})[0], 400)
        self.assertEqual(self.engine.turns, 0)

    def test_summary_length_exhaustion_keeps_primary_answer(self):
        self.svc.response_summaries = True
        self.engine.scripts[1] = self.tok.encode("summary " * 100)
        code, result = self.post({"input": "hi", "reasoning": {"summary": "concise"}, "max_output_tokens": 128})
        self.assertEqual(code, 200)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["usage"]["output_tokens"], 128)
        self.assertEqual(result["output"][1]["content"][0]["text"], "The file says before.")

    def test_summary_can_quote_tool_syntax_as_literal_text(self):
        self.svc.response_summaries = True
        quoted = "<tool_call><function=example></function></tool_call>"
        self.engine.scripts[1] = self.tok.encode(quoted, parse_special=True) + self.tok.encode("<|im_end|>", parse_special=True)
        code, result = self.post({"input": "hi", "reasoning": {"summary": "concise"}, "max_output_tokens": 256})
        self.assertEqual(code, 200)
        self.assertEqual(result["output"][0]["summary"][0]["text"], quoted)
        self.assertEqual(result["status"], "completed")


class Cancellation(unittest.TestCase):
    def test_close_drains_primary_before_marking_cancelled(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ResponseStore(folder)
            try:
                req, context = store.resolve({"input": "cancel"}, "m")
                asm = responses.Assembler(req, "m", 1, {}, False)
                store.begin(asm.response, context)
                closed = []
                class Service:
                    response_store = store
                    def run(self, *args):
                        try:
                            yield "ping", None
                        finally:
                            closed.append(True)
                cancel = threading.Event()
                stream = response_runner.generate(Service(), [], False, None, 10, req, cancel, asm, None)
                next(stream)
                next(stream)
                next(stream)
                stream.close()
                self.assertEqual(closed, [True])
                self.assertTrue(cancel.is_set())
                self.assertEqual(store.get(asm.response["id"])["status"], "cancelled")
            finally:
                store.close()


class ProcessRestart(unittest.TestCase):
    @contextlib.contextmanager
    def server(self, folder, flags):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        config = folder / "config.json"
        config.write_text(json.dumps({"experimental_responses_persistence": True,
            "responses_store_path": str(folder / "store"), "experimental_responses_summaries": True}))
        with (folder / "server.log").open("wb") as log:
            process = subprocess.Popen([sys.executable, "-m", "serve.server", "--engine", "mock",
                "--host", "127.0.0.1", "--port", str(port), "--config", str(config),
                "--script", ANSWER, "--script", "Summary of the reasoning.", *flags],
                cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            try:
                deadline = time.monotonic() + 15
                while True:
                    self.assertIsNone(process.poll(), "mock server failed to start; see " + str(folder / "server.log"))
                    try:
                        if self.http(port, "GET", "/health")[0] == 200:
                            break
                    except OSError:
                        pass
                    if time.monotonic() > deadline:
                        self.fail("mock server did not become ready")
                    time.sleep(0.05)
                yield port
            finally:
                process.kill()  # exercise recovery and release of the OS ownership lock after an abrupt exit
                process.wait(timeout=10)

    def http(self, port, method, path, body=None):
        client = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            client.request(method, path, body=json.dumps(body) if body is not None else None,
                           headers={"Content-Type": "application/json"})
            result = client.getresponse()
            return result.status, json.loads(result.read())
        finally:
            client.close()

    def test_cli_overrides_config_and_history_survives_real_process_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with self.server(folder, ["--no-experimental-responses-summaries"]) as port:
                code, first = self.http(port, "POST", "/v1/responses", {"input": "first",
                    "reasoning": {"summary": "concise"}, "max_output_tokens": 256})
                self.assertEqual(code, 200)
                self.assertTrue(first["store"])
                self.assertEqual(first["output"][0]["summary"], [])
            with self.server(folder, ["--experimental-responses-summaries"]) as port:
                self.assertEqual(self.http(port, "GET", "/v1/responses/" + first["id"]), (200, first))
                code, second = self.http(port, "POST", "/v1/responses", {"input": "next",
                    "previous_response_id": first["id"], "reasoning": {"summary": "concise"}, "max_output_tokens": 256})
                self.assertEqual(code, 200)
                self.assertEqual(second["output"][0]["summary"][0]["text"], "Summary of the reasoning.")
            with self.server(folder, ["--no-experimental-responses-persistence"]) as port:
                self.assertEqual(self.http(port, "GET", "/v1/responses/" + first["id"])[0], 404)


if __name__ == "__main__":
    unittest.main()
