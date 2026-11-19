"""Local HTTP contract tests; synthetic conversations and isolated databases only."""
import copy
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import tempfile
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from contextlib import closing
from unittest.mock import patch

from serve import llama_memory as lm


class SyntheticStore:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS memories(memory_id INTEGER PRIMARY KEY, "
                        "content TEXT, state TEXT DEFAULT 'active', access_state TEXT DEFAULT 'accessible', "
                        "belief_status TEXT DEFAULT 'current')")

    def remember(self, content, **_):
        self.db.execute("INSERT INTO memories(content) VALUES (?)", (content,))
        self.db.commit()

    def close(self):
        self.db.close()


class LlamaMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-llama-memory-")
        self.root = Path(self.temp.name) / "local"
        self.key = "synthetic-memory-principal"
        self.requests = []
        self.release = threading.Event()
        self.mode = "stop"
        self.summary_text = "Synthetic blue preference."
        self.cancel = threading.Event()
        self.emit_cancel = False
        self.frames = []
        test = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                self.assert_route = self.path
                if self.path != "/v1/chat/completions":
                    self.send_error(404)
                    return
                size = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(size))
                test.requests.append((body, self.headers.get("Authorization")))
                if test.mode == "blocked":
                    test.release.wait(10)
                summary = body["messages"][0]["content"] == lm.SUMMARY_SYSTEM
                text = test.summary_text if summary else "The synthetic answer."
                finish = "stop" if summary else test.mode
                if body["stream"]:
                    events = [
                        {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]},
                        {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
                        {"choices": [], "usage": {"completion_tokens": 4}},
                    ]
                    if test.mode == "tools":
                        events[0]["choices"][0]["delta"]["tool_calls"] = [{"id": "synthetic"}]
                        events[1]["choices"][0]["finish_reason"] = "stop"
                    data = "".join("data: " + json.dumps(e) + "\n\n" for e in events)
                    if test.mode != "truncated":
                        data += "data: [DONE]\n\n"
                    data = data.encode()
                else:
                    message = {"role": "assistant", "content": text}
                    if not summary and test.mode == "tools":
                        message["tool_calls"] = [{"id": "synthetic"}]
                        finish = "stop"
                    data = json.dumps({"choices": [{"index": 0, "message": message,
                                                    "finish_reason": finish}]}).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream" if body["stream"] else "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.config = {
            "port": self.server.server_port,
            "server_key": "synthetic-operator-key",
            "model": "synthetic-local",
            "args": [arg for flag, value in lm.PRIVATE_FLAGS.items()
                     for arg in ([flag] if value is None else [flag, value])],
            "hakua_harness": {"enabled": True, "users": [{
                "principal": "local", "key_env": "TEST_LLAMA_MEMORY_KEY", "root": str(self.root)}]},
        }
        self.env = patch.dict(os.environ, {"TEST_LLAMA_MEMORY_KEY": self.key})
        self.env.start()
        self.modules = patch.dict("sys.modules", {
            "hakua_memory": types.ModuleType("hakua_memory"),
            "hakua_memory.ebbinghaus": types.ModuleType("hakua_memory.ebbinghaus"),
            "hakua_memory.ebbinghaus.store": types.SimpleNamespace(EbbinghausMemoryStore=SyntheticStore)})
        self.modules.start()

    def tearDown(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.modules.stop()
        self.env.stop()
        self.temp.cleanup()

    def request(self, read=False, store=False, stream=True, turn="one"):
        return {"messages": [{"role": "user", "content": "Synthetic blue preference?"}],
                "stream": stream, "strata_memory": {
                    "read": read, "store_summary": store, "turn_id": turn}}

    def emit(self, value):
        self.frames.append(value)
        if self.emit_cancel and "event" in value:
            self.cancel.set()

    def run_turn(self, request=None, config=None, key=None):
        return lm.run(config or self.config, request or self.request(),
                      key or self.key, self.emit, self.cancel)

    def test_read_only_and_no_consent_flags_create_no_profile(self):
        for read in (False, True):
            self.run_turn(self.request(read=read))
            self.assertFalse(self.root.exists())
            self.assertEqual(len(self.requests), 1 + int(read))
        self.assertTrue(all(r[0]["cache_prompt"] is False for r in self.requests))
        self.assertTrue(all(r[1] == "Bearer synthetic-operator-key" for r in self.requests))

    def test_completed_stream_summarizes_once_and_recalls_after_restart(self):
        self.run_turn(self.request(store=True))
        self.assertEqual(self.frames[-1], {"done": True, "memory": "stored"})
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[1][0]["messages"][0]["content"], lm.SUMMARY_SYSTEM)
        self.frames.clear()
        self.run_turn(self.request(read=True, turn="recall"))
        self.assertIn("Synthetic blue preference.", self.requests[-1][0]["messages"][0]["content"])
        self.assertEqual(self.frames[-1]["memory"], "not-stored")
        # A fresh Harness in each worker shares the durable at-most-once ledger.
        self.run_turn(self.request(store=True))
        self.assertEqual(len(self.requests), 4)
        with closing(sqlite3.connect(self.root / "ebbinghaus.db")) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM memories").fetchone()[0], 1)

    def test_nonstream_stores_and_returns_original_answer(self):
        self.run_turn(self.request(store=True, stream=False))
        self.assertEqual(self.frames[0]["response"]["choices"][0]["message"]["content"],
                         "The synthetic answer.")
        self.assertEqual(self.frames[-1]["memory"], "stored")

    def test_acknowledgment_and_no_fact_summaries_create_no_store(self):
        for index, text in enumerate(("Saved.", "OK!", "Done", "NO_MEMORY", "Acknowledged.")):
            with self.subTest(text=text):
                self.summary_text = text
                self.run_turn(self.request(store=True, turn="ack-" + str(index)))
                self.assertFalse(self.root.exists())
                self.assertEqual(self.frames[-1]["memory"], "no-summary")

    def test_summary_contains_only_current_turn_data(self):
        request = self.request(store=True)
        request["messages"].insert(0, {"role": "assistant", "content": "Synthetic old conversation excluded from summary"})
        self.run_turn(request)
        summary_messages = self.requests[-1][0]["messages"]
        self.assertEqual(len(summary_messages), 2)
        self.assertIn(request["messages"][-1]["content"], summary_messages[-1]["content"])
        self.assertNotIn("Synthetic old conversation", json.dumps(summary_messages))

    def test_length_tool_and_error_finishes_never_store(self):
        for stream in (False, True):
            for finish in ("length", "tools", "error"):
                with self.subTest(stream=stream, finish=finish):
                    self.mode = finish
                    self.frames.clear()
                    self.run_turn(self.request(store=True, stream=stream))
                    self.assertFalse(self.root.exists())
                    self.assertEqual(self.frames[-1]["memory"], "not-stored")

    def test_truncated_stream_never_stores(self):
        self.mode = "truncated"
        with self.assertRaises(RuntimeError):
            self.run_turn(self.request(store=True))
        self.assertFalse(self.root.exists())

    def test_cancelled_stream_never_stores(self):
        self.emit_cancel = True
        with self.assertRaises(InterruptedError):
            self.run_turn(self.request(store=True))
        self.assertFalse(self.root.exists())

    def test_wrong_principal_and_disabled_harness_fail_before_http(self):
        with self.assertRaises(ValueError):
            self.run_turn(key="another-principal")
        cfg = {**self.config, "hakua_harness": {"enabled": False}}
        with self.assertRaises(ValueError):
            self.run_turn(config=cfg)
        self.assertEqual(self.requests, [])
        self.assertFalse(self.root.exists())

    def test_cache_policy_conflicts_fail_before_http(self):
        for flag, value in lm.PRIVATE_FLAGS.items():
            with self.subTest(flag=flag):
                cfg = copy.deepcopy(self.config)
                cfg["args"].remove(flag)
                with self.assertRaises(ValueError):
                    self.run_turn(config=cfg)
        for extra in (["--slots"], ["--cache-ram=42"], ["--cache-prompt"],
                      ["--slot-save-path", "ignored"], ["--verbose"]):
            with self.subTest(extra=extra):
                cfg = {**self.config, "args": self.config["args"] + extra}
                with self.assertRaises(ValueError):
                    self.run_turn(config=cfg)
        self.assertEqual(self.requests, [])

    def test_no_http_control_over_identity_transport_or_tools(self):
        for field, value in (("root", "ignored"), ("base_url", "http://example.com"),
                             ("tools", []), ("cache_prompt", True), ("n", 2)):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    self.run_turn({**self.request(), field: value})
        for messages in ([{"role": "tool", "content": "x"}],
                         [{"role": "user", "content": [{"text": "x"}]}],
                         [{"role": "user", "content": "x", "tool_calls": []}]):
            with self.assertRaises(ValueError):
                self.run_turn({**self.request(), "messages": messages})
        self.assertEqual(self.requests, [])

    def test_explicit_boolean_consent_and_bounded_request_required(self):
        for consent in (None, False, {"read": True}, {
                "read": 1, "store_summary": True, "turn_id": "one"}):
            with self.assertRaises(ValueError):
                self.run_turn({**self.request(), "strata_memory": consent})
        for maximum in (0, 4097, True, "512"):
            with self.assertRaises(ValueError):
                self.run_turn({**self.request(), "max_tokens": maximum})
        with self.assertRaises(ValueError):
            self.run_turn({**self.request(), "messages": [{"role": "user", "content": "x" * 65537}]})
        self.assertEqual(self.requests, [])

    def test_owner_pipe_closure_stops_worker_without_creating_a_store(self):
        self.mode = "blocked"
        child_env = {k: v for k, v in os.environ.items()
                     if not any(word in k.upper() for word in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))}
        child_env["TEST_LLAMA_MEMORY_KEY"] = self.key
        child = subprocess.Popen([sys.executable, "-B", "-m", "serve.llama_memory"],
                                 cwd=Path(__file__).resolve().parents[1], env=child_env,
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            child.stdin.write((json.dumps({"config": self.config, "request": self.request(store=True),
                                         "key": self.key}) + "\n").encode())
            child.stdin.flush()
            deadline = time.monotonic() + 5
            while not self.requests and child.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(self.requests, "worker reached the synthetic local model")
            child.stdin.close()
            self.assertEqual(child.wait(timeout=3), 1)
            self.assertFalse(self.root.exists())
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
            for stream in (child.stdin, child.stdout, child.stderr):
                stream.close()

    def test_loopback_port_and_operator_key_are_required(self):
        for port in ("http://example.com", 1, 65536, True):
            with self.assertRaises(ValueError):
                lm.LocalLlama(port, "synthetic-operator-key", self.cancel)
        for key in ("", "short", "synthetic-key\nheader"):
            with self.assertRaises(ValueError):
                lm.LocalLlama(12345, key, self.cancel)


if __name__ == "__main__":
    unittest.main()
