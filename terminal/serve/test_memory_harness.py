"""Synthetic local-only harness checks; no user profiles or actual model."""
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import types
import unittest
from contextlib import closing, redirect_stdout
from unittest.mock import patch

from serve import memory_harness as mh
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, _debug_req, make_handler, serve

ROOT = Path(__file__).resolve().parents[1]


class FakeStore:
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS memories(memory_id INTEGER PRIMARY KEY, content TEXT UNIQUE, "
                          "state TEXT DEFAULT 'active', access_state TEXT DEFAULT 'accessible', "
                          "belief_status TEXT DEFAULT 'current')")

    def remember(self, content, **kwargs):
        self.conn.execute("INSERT OR IGNORE INTO memories(content) VALUES (?)", (content,))
        self.conn.commit()

    def close(self):
        self.conn.close()


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"TEST_MEMORY_A": "synthetic-a", "TEST_MEMORY_B": "synthetic-b"})
        self.env.start()
        self.h = mh.Harness([{"principal": p, "key_env": "TEST_MEMORY_" + p.upper(), "root": str(self.root / p)}
                             for p in ("a", "b")], ["synthetic-a", "synthetic-b"])
        self.modules = patch.dict("sys.modules", {"hakua_memory": types.ModuleType("hakua_memory"),
                           "hakua_memory.ebbinghaus": types.ModuleType("hakua_memory.ebbinghaus"),
                           "hakua_memory.ebbinghaus.store": types.SimpleNamespace(EbbinghausMemoryStore=FakeStore)})
        self.modules.start()

    def tearDown(self):
        self.modules.stop()
        self.env.stop()
        self.temp.cleanup()

    def begin(self, key="synthetic-a", turn="one"):
        return self.h.begin(key, {"read": True, "store_summary": True, "turn_id": turn},
                            [{"role": "user", "content": "synthetic blue preference"}], threading.Event())

    def test_no_consent_no_files_and_same_message_object(self):
        messages = [{"role": "user", "content": "hi"}]
        turn, got = self.h.begin("", None, messages, threading.Event())
        self.assertIsNone(turn)
        self.assertIs(got, messages)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_identity_and_paths_cannot_come_from_request(self):
        for key in ("", "wrong"):
            with self.assertRaises(ValueError):
                self.begin(key)
        with self.assertRaises(ValueError):
            self.h.begin("synthetic-a", {"read": True, "store_summary": True, "turn_id": "x", "principal": "b"}, [], threading.Event())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_principal_maps_are_immutable_and_separate(self):
        with self.assertRaises(TypeError):
            self.h.users[b"x"] = self.h.principal("synthetic-a")
        self.assertNotEqual(self.h.principal("synthetic-a"), self.h.principal("synthetic-b"))

    def test_mapping_requires_existing_keys(self):
        with self.assertRaises(ValueError):
            mh.Harness([{"principal": "a", "key_env": "TEST_MEMORY_A", "root": str(self.root / "a")}], ["other"])
        with patch.dict(os.environ, {"TEST_MEMORY_A": "é"}):
            with self.assertRaisesRegex(ValueError, "ASCII"):
                mh.Harness([{"principal": "a", "key_env": "TEST_MEMORY_A", "root": str(self.root / "a")}], ["é"])
        self.assertIsNone(mh.from_config(None))

    def test_native_session_cache_must_be_disabled_before_memory_start(self):
        config = {"enabled": True, "users": []}
        safe = ["--prompt-cache", "0", "--conversation-cache-mib", "0"]
        mh.require_private_engine(config, safe)
        for args in (["--prompt-cache", "0"], ["--conversation-cache-mib", "0"],
                     ["--prompt-cache", "1", "--conversation-cache-mib", "0"],
                     [*safe, "--prompt-cache", "6"]):
            with self.subTest(args=args), self.assertRaises(ValueError):
                mh.require_private_engine(config, args)
        with self.assertRaises(ValueError):
            mh.require_private_engine(config, safe, slot_save_path="H:/shared-sessions")
        with self.assertRaisesRegex(ValueError, "STRATA_BATCH_TRACE"):
            mh.require_private_engine(config, safe, batch_trace=True)
        mh.require_private_engine(None, [])

    def test_configure_harness_rejects_bad_auth_before_any_engine_load(self):
        """CPU gate: false keys, shared keys, nested roots and bad persona fail closed."""
        safe = ["--prompt-cache", "0", "--conversation-cache-mib", "0"]
        self.assertIsNone(mh.configure_harness(None, ["synthetic-a"]))
        self.assertIsNone(mh.configure_harness({"enabled": False, "users": []}, ["synthetic-a"]))
        with self.assertRaisesRegex(ValueError, "existing server API key"):
            mh.configure_harness({"enabled": True, "users": [
                {"principal": "a", "key_env": "TEST_MEMORY_A", "root": str(self.root / "a")}]}, [])
        with self.assertRaisesRegex(ValueError, "distinct ASCII existing key"):
            mh.configure_harness({"enabled": True, "users": [
                {"principal": "a", "key_env": "TEST_MEMORY_A", "root": str(self.root / "a")}]},
                ["wrong-key"], engine_argv=safe)
        nested = self.root / "parent"
        nested.mkdir()
        (nested / "child").mkdir()
        with self.assertRaisesRegex(ValueError, "roots must be separate"):
            mh.configure_harness({"enabled": True, "users": [
                {"principal": "a", "key_env": "TEST_MEMORY_A", "root": str(nested)},
                {"principal": "b", "key_env": "TEST_MEMORY_B", "root": str(nested / "child")}]},
                ["synthetic-a", "synthetic-b"], engine_argv=safe)
        with self.assertRaisesRegex(ValueError, "persona_sha256"):
            mh.configure_harness({"enabled": True, "users": [
                {"principal": "a", "key_env": "TEST_MEMORY_A", "root": str(self.root / "ok-a"),
                 "persona_sha256": {"SOUL.md": "not-a-hash"}}]},
                ["synthetic-a"], engine_argv=safe)
        with patch.dict(os.environ, {"TEST_MEMORY_A": "synthetic-a", "TEST_MEMORY_SHARED": "synthetic-a",
                                     "TEST_MEMORY_B": "synthetic-a"}):
            with self.assertRaisesRegex(ValueError, "distinct ASCII existing key"):
                mh.configure_harness({"enabled": True, "users": [
                    {"principal": "a", "key_env": "TEST_MEMORY_A", "root": str(self.root / "share-a")},
                    {"principal": "b", "key_env": "TEST_MEMORY_B", "root": str(self.root / "share-b")}]},
                    ["synthetic-a"], engine_argv=safe)
        with self.assertRaisesRegex(ValueError, "prompt-cache"):
            mh.configure_harness({"enabled": True, "users": [
                {"principal": "a", "key_env": "TEST_MEMORY_A", "root": str(self.root / "cache-a")}]},
                ["synthetic-a"], engine_argv=["--prompt-cache", "1", "--conversation-cache-mib", "0"])
        ok = mh.configure_harness({"enabled": True, "users": [
            {"principal": "a", "key_env": "TEST_MEMORY_A", "root": str(self.root / "ok-a")},
            {"principal": "b", "key_env": "TEST_MEMORY_B", "root": str(self.root / "ok-b")}]},
            ["synthetic-a", "synthetic-b", "host-only-key"], engine_argv=safe)
        self.assertIsNotNone(ok)
        self.assertEqual(ok.principal("synthetic-a").name, "a")
        self.assertIsNone(ok.principal("host-only-key"), "Host key must not map to a memory principal")

    def test_mapped_request_without_memory_consent_never_enters_debug_log(self):
        output = io.StringIO()
        with patch.dict(os.environ, {"STRATA_DEBUG": "1"}), redirect_stdout(output):
            _debug_req("openai", {}, [{"role": "user", "content": "private text"}], None,
                       32, False, 10, private=True)
        self.assertEqual(output.getvalue(), "")

    def test_responses_global_compaction_tools_are_disabled_with_memory(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "answer", max_context=131072), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.api_key = ["synthetic-a", "synthetic-b"]
        svc.memory_harness = self.h
        svc.codex_compaction_cache = True
        httpd = serve(svc, port=0)
        body = {"input": "private request", "max_output_tokens": 32,
                "reasoning": {"effort": "none"},
                "client_metadata": {"x-codex-turn-metadata": json.dumps({
                    "request_kind": "compaction", "session_id": "shared-id", "thread_id": "shared-id"})}}
        try:
            with patch("serve.server.responses_api.prompt_tools", side_effect=AssertionError("shared cache read")), \
                    patch("serve.server.responses_api.prompt_made", side_effect=AssertionError("shared cache write")):
                c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=20)
                c.request("POST", "/v1/responses", json.dumps(body),
                          {"Authorization": "Bearer synthetic-b", "Content-Type": "application/json"})
                r = c.getresponse()
                data = r.read()
                c.close()
                self.assertEqual(r.status, 200, data)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_mapped_key_cannot_use_operator_mcp_or_local_image_paths(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "answer"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.api_key = ["synthetic-admin", "synthetic-a", "synthetic-b"]
        svc.memory_harness = self.h
        handler = object.__new__(make_handler(svc))
        handler.headers = {"Authorization": "Bearer synthetic-a"}
        with self.assertRaisesRegex(ValueError, "server-local image"):
            handler._no_local_images([{"role": "user", "content": [
                {"type": "image", "source": str(self.root / "b" / "SOUL.md")}]}])
        httpd = serve(svc, port=0)
        try:
            body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 32,
                    "strata_mcp": True}
            c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=20)
            c.request("POST", "/v1/chat/completions", json.dumps(body),
                      {"Authorization": "Bearer synthetic-a", "Content-Type": "application/json"})
            r = c.getresponse()
            data = r.read()
            c.close()
            self.assertEqual(r.status, 400, data)
            self.assertIn(b"memory principals cannot use server MCP tools", data)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_audited_memory_request_is_rejected_before_prepare_or_recall(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "answer"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.memory_harness = self.h
        handler = object.__new__(make_handler(svc))
        handler.headers = {"Authorization": "Bearer synthetic-a"}
        handler.memory_turn = None
        with patch("serve.server.perf_e2e.current", return_value=object()), \
                patch.object(svc, "prepare", side_effect=AssertionError("audit must not change")), \
                patch.object(self.h, "begin", side_effect=AssertionError("must not read")):
            with self.assertRaisesRegex(ValueError, "audited request"):
                handler._prepare([{"role": "user", "content": "hello"}], None, {}, 128,
                                 req={"strata_memory": {"read": True, "store_summary": True, "turn_id": "audit"}},
                                 cancel=threading.Event())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_read_and_summary_consent_are_independent(self):
        messages = [{"role": "user", "content": "blue preference"}]
        with patch.object(self.h, "recall", side_effect=AssertionError("no read consent")):
            turn, got = self.h.begin("synthetic-a", {"read": False, "store_summary": True, "turn_id": "write"},
                                     messages, threading.Event())
            self.assertIs(got, messages)
            self.assertTrue(turn.store_summary)
        turn, _ = self.h.begin("synthetic-a", {"read": True, "store_summary": False, "turn_id": "read"},
                               messages, threading.Event())
        self.h.complete(turn, "answer", lambda *args: self.fail("no store consent"))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_cancellation_during_local_summary_leaves_no_files(self):
        turn, _ = self.begin()
        def cancel_summary(user, answer, cancel):
            cancel.set()
            return "summary"
        self.h.complete(turn, "answer", cancel_summary)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_persona_files_are_isolated_and_only_read_with_consent(self):
        for name in ("a", "b"):
            folder = self.root / name
            folder.mkdir()
            (folder / "SOUL.md").write_text("synthetic persona " + name, encoding="utf-8")
            (folder / "USER.md").write_text("synthetic user " + name, encoding="utf-8")
        self.h = mh.Harness([
            {"principal": name, "key_env": "TEST_MEMORY_" + name.upper(), "root": str(self.root / name),
             "persona_sha256": {file: hashlib.sha256((self.root / name / file).read_bytes()).hexdigest()
                                for file in ("SOUL.md", "USER.md")}}
            for name in ("a", "b")], ["synthetic-a", "synthetic-b"])
        _, messages = self.begin()
        context = messages[0]["content"]
        self.assertIn("synthetic persona a", context)
        self.assertNotIn("synthetic persona b", context)
        _, messages = self.h.begin("synthetic-a", {"read": False, "store_summary": False, "turn_id": "none"},
                                    [{"role": "user", "content": "hello"}], threading.Event())
        self.assertEqual(len(messages), 1)

    def test_persona_requires_immutable_operator_approval_and_bounded_exact_bytes(self):
        folder = self.root / "a"
        folder.mkdir()
        path = folder / "SOUL.md"
        path.write_bytes(b"approved persona")
        _, messages = self.begin()
        self.assertEqual(len(messages), 1)  # no approval means no persona
        approved = {"SOUL.md": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.h = mh.Harness([{"principal": "a", "key_env": "TEST_MEMORY_A", "root": str(folder),
                              "persona_sha256": approved}], ["synthetic-a"])
        approved["SOUL.md"] = "0" * 64  # caller mutation cannot alter the binding
        self.assertIn("approved persona", self.begin()[1][0]["content"])
        path.write_bytes(b"changed persona")
        self.assertEqual(len(self.begin()[1]), 1)
        path.write_bytes(b"x" * 8193)
        self.h = mh.Harness([{"principal": "a", "key_env": "TEST_MEMORY_A", "root": str(folder),
                              "persona_sha256": {"SOUL.md": hashlib.sha256(path.read_bytes()).hexdigest()}}],
                            ["synthetic-a"])
        self.assertEqual(len(self.begin()[1]), 1)  # even approved oversized content is omitted
        for approval in ({"elsewhere.md": "a" * 64}, {"SOUL.md": "bad"}, []):
            with self.subTest(approval=approval), self.assertRaises(ValueError):
                mh.Harness([{"principal": "a", "key_env": "TEST_MEMORY_A", "root": str(folder),
                             "persona_sha256": approval}], ["synthetic-a"])

    def test_memory_md_requires_exact_approval_and_explicit_larger_file_cap(self):
        folder = self.root / "a"
        folder.mkdir()
        memory = folder / "MEMORY.md"
        memory.write_bytes(b"synthetic approved context " + b"x" * 10000)
        approved = hashlib.sha256(memory.read_bytes()).hexdigest()
        user = {"principal": "a", "key_env": "TEST_MEMORY_A", "root": str(folder)}
        # A file is never approved just because it has the expected name.
        self.assertEqual(len(self.begin()[1]), 1)
        self.h = mh.Harness([{**user, "persona_sha256": {"MEMORY.md": approved}}], ["synthetic-a"])
        self.assertEqual(len(self.begin()[1]), 1)  # 8192-byte default remains
        self.h = mh.Harness([{**user, "persona_sha256": {"MEMORY.md": approved},
                              "persona_max_bytes": 12000}], ["synthetic-a"])
        before = hashlib.sha256(memory.read_bytes()).digest()
        self.assertIn("synthetic approved context", self.begin()[1][0]["content"])
        self.assertEqual(hashlib.sha256(memory.read_bytes()).digest(), before)
        self.assertEqual([p.name for p in folder.iterdir()], ["MEMORY.md"])
        memory.write_bytes(b"changed context")
        self.assertEqual(len(self.begin()[1]), 1)  # changed bytes need new approval
        memory.write_bytes(b"x" * 12001)
        self.h = mh.Harness([{**user, "persona_sha256": {"MEMORY.md": hashlib.sha256(memory.read_bytes()).hexdigest()},
                              "persona_max_bytes": 12000}], ["synthetic-a"])
        self.assertEqual(len(self.begin()[1]), 1)  # approved but above configured cap
        for cap in (True, False, 0, -1, 32769, 8192.0, "12000", None):
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, "persona_max_bytes"):
                mh.Harness([{**user, "persona_max_bytes": cap}], ["synthetic-a"])

    def test_hardlinked_persona_and_memory_files_are_not_read(self):
        turn, _ = self.begin()
        self.h.complete(turn, "answer", lambda *args: "blue private memory")
        other = self.root / "b"
        other.mkdir()
        (self.root / "a" / "SOUL.md").write_text("private persona", encoding="utf-8")
        (other / "SOUL.md").hardlink_to(self.root / "a" / "SOUL.md")
        (other / "ebbinghaus.db").hardlink_to(self.root / "a" / "ebbinghaus.db")
        self.h = mh.Harness([{"principal": "b", "key_env": "TEST_MEMORY_B", "root": str(other),
                              "persona_sha256": {"SOUL.md": hashlib.sha256((other / "SOUL.md").read_bytes()).hexdigest()}}],
                            ["synthetic-b"])
        _, messages = self.begin("synthetic-b")
        self.assertEqual(len(messages), 1)

    def test_dangling_profile_database_links_cannot_redirect_writes(self):
        for database in ("strata-turns.db", "ebbinghaus.db"):
            with self.subTest(database=database):
                turn, _ = self.begin(turn=database.replace(".", "_"))
                turn.principal.root.mkdir(exist_ok=True)
                outside = self.root / ("outside-" + database)
                link = turn.principal.root / database
                try:
                    link.symlink_to(outside)
                except OSError as error:
                    self.skipTest("symlinks unavailable: " + str(error))
                with self.assertRaisesRegex(ValueError, "inside its profile"):
                    self.h.complete(turn, "answer", lambda *args: "private summary")
                self.assertFalse(outside.exists())
                link.unlink()

    def test_incomplete_or_tool_outputs_never_trigger_summary(self):
        turn, _ = self.begin()
        cases = [
            ("openai", [{"choices": [{"delta": {"content": "partial"}, "finish_reason": "length"}]}]),
            ("responses", [{"type": "response.output_text.delta", "delta": "partial"},
                           {"type": "response.completed", "response": {"status": "completed", "output": [{"type": "function_call"}]}}]),
            ("anthropic", [("content_block_delta", {"delta": {"text": "partial"}}),
                           ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}})]),
        ]
        for api, values in cases:
            def events():
                yield from values
            list(mh.capture(events(), api, self.h, turn, lambda *args: self.fail("incomplete turn")))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_failed_writer_is_uncertain_and_never_replayed(self):
        turn, _ = self.begin()
        with patch.object(sys.modules["hakua_memory.ebbinghaus.store"], "EbbinghausMemoryStore",
                          side_effect=RuntimeError("crash")) as writer:
            with self.assertRaises(RuntimeError):
                self.h.complete(turn, "answer", lambda *args: "summary")
            self.assertEqual(self.h.complete(turn, "answer", lambda *args: self.fail("no replay")), "uncertain")
            self.assertEqual(writer.call_count, 1)

    def test_wal_store_is_skipped_without_sidecar_changes(self):
        principal = self.h.principal("synthetic-a")
        principal.root.mkdir()
        db = principal.root / "ebbinghaus.db"
        with closing(sqlite3.connect(db)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE sample(value TEXT)")
            conn.commit()
            before = {p.name: p.read_bytes() for p in principal.root.iterdir()}
            self.assertEqual(self.h.recall(principal, "blue"), [])
            self.assertEqual(before, {p.name: p.read_bytes() for p in principal.root.iterdir()})

    def test_local_summary_inputs_and_idempotency(self):
        turn, _ = self.begin()
        calls = []
        def summary(user, answer, cancel):
            calls.append((user, answer))
            return "Blue preference summary"
        self.h.complete(turn, "completed answer", summary)
        self.h.complete(turn, "completed answer", summary)
        self.assertEqual(calls, [(turn.user, "completed answer")])
        with closing(sqlite3.connect(turn.principal.root / "ebbinghaus.db")) as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM memories").fetchone()[0], 1)
        with self.assertRaises(ValueError):
            self.h.complete(turn, "different answer", summary)
        self.assertFalse((self.root / "b").exists())

    def test_readonly_recall_filters_and_never_changes_files(self):
        turn, _ = self.begin()
        self.h.complete(turn, "answer", lambda *args: "blue synthetic memory")
        db = turn.principal.root / "ebbinghaus.db"
        with closing(sqlite3.connect(db)) as conn, conn:
            conn.execute("INSERT INTO memories(content,state) VALUES ('blue private archived','archived')")
            conn.execute("INSERT INTO memories(content,access_state) VALUES ('blue private hidden','blocked')")
        before = {p.name: hashlib.sha256(p.read_bytes()).digest() for p in turn.principal.root.iterdir()}
        self.assertEqual(len(self.h.recall(turn.principal, "blue")), 1)
        after = {p.name: hashlib.sha256(p.read_bytes()).digest() for p in turn.principal.root.iterdir()}
        self.assertEqual(before, after)
        self.assertEqual(self.h.recall(self.h.principal("synthetic-b"), "blue"), [])
        with closing(mh._read_db(db)) as conn:
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("DELETE FROM memories")

    def test_japanese_partial_match_and_english_word_recall(self):
        """F09: space-less Japanese n-grams hit; English words still work; empty query is inert."""
        turn, _ = self.begin()
        db = turn.principal.root / "ebbinghaus.db"
        turn.principal.root.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db)) as conn, conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memories(memory_id INTEGER PRIMARY KEY, content TEXT UNIQUE, "
                "state TEXT DEFAULT 'active', access_state TEXT DEFAULT 'accessible', "
                "belief_status TEXT DEFAULT 'current')")
            conn.execute("INSERT INTO memories(content) VALUES (?)", ("ボブにゃんはコーヒーが好きです。",))
            conn.execute("INSERT INTO memories(content) VALUES (?)", ("Bob likes coffee.",))
        before = {p.name: hashlib.sha256(p.read_bytes()).digest() for p in turn.principal.root.iterdir()}
        jp = self.h.recall(turn.principal, "コーヒー")
        self.assertEqual(len(jp), 1, jp)
        self.assertIn("コーヒー", jp[0])
        en = self.h.recall(turn.principal, "coffee")
        self.assertEqual(len(en), 1, en)
        self.assertIn("coffee", en[0].casefold())
        self.assertEqual(self.h.recall(turn.principal, ""), [])
        self.assertEqual(self.h.recall(turn.principal, "x" * 600), [])
        after = {p.name: hashlib.sha256(p.read_bytes()).digest() for p in turn.principal.root.iterdir()}
        self.assertEqual(before, after)

    def test_cancel_before_completion_never_creates_files(self):
        turn, _ = self.begin()
        turn.cancel.set()
        self.h.complete(turn, "answer", lambda *args: self.fail("must not summarize"))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_generator_close_and_error_never_write(self):
        turn, _ = self.begin()
        chunks = iter([{"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]}])
        # Real generators expose close; close after one chunk mimics disconnect.
        def events():
            yield from chunks
            raise RuntimeError("synthetic engine failure")
        stream = mh.capture(events(), "openai", self.h, turn, lambda *args: self.fail("must not summarize"))
        next(stream)
        stream.close()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_api_dialects_use_same_harness_and_local_model(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "synthetic visible answer", max_context=131072), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.api_key = ["synthetic-admin", "synthetic-a", "synthetic-b"]
        svc.memory_harness = self.h
        httpd = serve(svc, port=0)
        try:
            dialects = [
                ("/v1/chat/completions", {"messages": [{"role": "user", "content": "blue synthetic preference"}]}),
                ("/v1/messages", {"messages": [{"role": "user", "content": "blue synthetic preference"}]}),
                ("/v1/responses", {"input": "blue synthetic preference"}),
            ]
            for index, (path, body) in enumerate([(p, {**b, "stream": stream})
                                                  for p, b in dialects for stream in (False, True)]):
                body.update({"max_tokens": 128, "max_output_tokens": 128,
                             "chat_template_kwargs": {"enable_thinking": False}, "thinking": {"type": "disabled"},
                             "reasoning": {"effort": "none"},
                             "strata_memory": {"read": True, "store_summary": True, "turn_id": "api_" + str(index)}})
                c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=20)
                c.request("POST", path, json.dumps(body), {"Authorization": "Bearer synthetic-a", "Content-Type": "application/json"})
                r = c.getresponse()
                data = r.read()
                c.close()
                self.assertEqual(r.status, 200, (path, data))
            with closing(sqlite3.connect(self.root / "a" / "ebbinghaus.db")) as conn:
                self.assertEqual(conn.execute("SELECT count(*) FROM memories").fetchone()[0], 6)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_http_auth_and_token_overflow_fail_before_generation(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "answer", max_context=262144), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.api_key = ["synthetic-admin", "synthetic-a", "synthetic-b"]
        svc.memory_harness = self.h
        httpd = serve(svc, port=0)
        body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 200000,
                "chat_template_kwargs": {"enable_thinking": False},
                "strata_memory": {"read": False, "store_summary": False, "turn_id": "budget"}}
        try:
            with patch.object(svc.engine, "generate", side_effect=AssertionError("must not run")):
                for key, expected in [("", 401), ("wrong", 401), ("synthetic-admin", 400), ("synthetic-a", 400)]:
                    c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=20)
                    c.request("POST", "/v1/chat/completions", json.dumps(body),
                              {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
                    r = c.getresponse()
                    out = r.read()
                    c.close()
                    self.assertEqual(r.status, expected, out)
            self.assertEqual(list(self.root.iterdir()), [])
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_engine_context_clipping_is_rejected_only_when_opted_in(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "answer", max_context=4096), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.api_key = ["synthetic-a", "synthetic-b"]
        svc.memory_harness = self.h
        httpd = serve(svc, port=0)
        try:
            for opted_in, expected in [(True, 400), (False, 200)]:
                body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 4096,
                        "chat_template_kwargs": {"enable_thinking": False}}
                if opted_in:
                    body["strata_memory"] = {"read": False, "store_summary": False, "turn_id": "small_ctx"}
                c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=20)
                c.request("POST", "/v1/chat/completions", json.dumps(body),
                          {"Authorization": "Bearer synthetic-a", "Content-Type": "application/json"})
                r = c.getresponse()
                out = r.read()
                c.close()
                self.assertEqual(r.status, expected, out)
            self.assertEqual(list(self.root.iterdir()), [])
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_memory_keys_cannot_read_shared_state_and_private_runs_keep_no_text(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "synthetic secret answer", max_context=4096), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.api_key = ["synthetic-admin", "synthetic-a", "synthetic-b"]
        svc.memory_harness = self.h
        svc.api_monitor = True
        httpd = serve(svc, port=0)
        try:
            for key in ("synthetic-a", "synthetic-b"):
                for path in ("/status", "/api/requests", "/metrics", "/v1/telemetry", "/config", "/slots", "/settings"):
                    c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=20)
                    c.request("GET", path, headers={"Authorization": "Bearer " + key})
                    r = c.getresponse()
                    out = r.read()
                    c.close()
                    self.assertEqual(r.status, 403, (key, path, out))
            seen = []
            original = svc._note
            def note(*args, **kwargs):
                original(*args, **kwargs)
                seen.append(dict(svc.status))
            with patch.object(svc, "_note", side_effect=note):
                c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=20)
                body = {"messages": [{"role": "user", "content": "synthetic secret request"}], "max_tokens": 128,
                        "chat_template_kwargs": {"enable_thinking": False},
                        "strata_memory": {"read": False, "store_summary": True, "turn_id": "private"}}
                c.request("POST", "/v1/chat/completions", json.dumps(body),
                          {"Authorization": "Bearer synthetic-a", "Content-Type": "application/json"})
                r = c.getresponse()
                r.read()
                c.close()
                self.assertEqual(r.status, 200)
            self.assertTrue(seen)
            self.assertFalse(any(row.get("tail") or row.get("tool") for row in seen))
            self.assertEqual(svc.request_records(), [])
        finally:
            httpd.shutdown()
            httpd.server_close()


class RealWriterSyntheticTests(unittest.TestCase):
    def test_http_mock_server_uses_real_writer_and_isolates_two_principals(self):
        try:
            import hakua_memory.ebbinghaus.store  # noqa: F401 - optional local package
        except ImportError as error:
            self.skipTest("hakua_memory is not installed: " + str(error))
        with tempfile.TemporaryDirectory(prefix="strata-hakua-http-") as directory, \
                patch.dict(os.environ, {"STRATA_HTTP_A": "synthetic-http-a", "STRATA_HTTP_B": "synthetic-http-b"}):
            root = Path(directory)
            harness = mh.Harness([
                {"principal": name, "key_env": "STRATA_HTTP_" + name.upper(), "root": str(root / name)}
                for name in ("a", "b")], ["synthetic-http-a", "synthetic-http-b"])
            tok = ByteTokenizer()
            svc = Service(MockEngine(tok, "synthetic azure preference", max_context=131072), tok,
                          ChatTemplate(ROOT / "serve/chat_template.jinja"))
            svc.api_key = ["synthetic-http-a", "synthetic-http-b"]
            svc.memory_harness = harness
            httpd = serve(svc, port=0)
            def request(key, turn_id, *, read=True, store=True, max_tokens=128):
                body = {"messages": [{"role": "user", "content": "synthetic azure preference"}],
                        "max_tokens": max_tokens, "reasoning_effort": "none", "stream": False,
                        "strata_memory": {"read": read, "store_summary": store, "turn_id": turn_id}}
                conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=20)
                try:
                    conn.request("POST", "/v1/chat/completions", json.dumps(body),
                                 {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
                    response = conn.getresponse()
                    return response.status, json.loads(response.read())
                finally:
                    conn.close()
            try:
                status, first = request("synthetic-http-a", "same_turn")
                self.assertEqual(status, 200, first)
                self.assertEqual(first["choices"][0]["finish_reason"], "stop")
                db_a = root / "a" / "ebbinghaus.db"
                self.assertTrue(db_a.is_file())
                with closing(sqlite3.connect(db_a)) as conn:
                    self.assertEqual(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0], 1)
                before = hashlib.sha256(db_a.read_bytes()).digest()
                self.assertEqual(request("synthetic-http-a", "same_turn")[0], 200)
                self.assertEqual(hashlib.sha256(db_a.read_bytes()).digest(), before)
                self.assertTrue(harness.recall(harness.principal("synthetic-http-a"), "azure"))
                self.assertEqual(request("synthetic-http-b", "b_read", store=False)[0], 200)
                self.assertEqual(harness.recall(harness.principal("synthetic-http-b"), "azure"), [])
                status, incomplete = request("synthetic-http-b", "b_short", read=False, max_tokens=1)
                self.assertEqual(status, 200, incomplete)
                self.assertEqual(incomplete["choices"][0]["finish_reason"], "length")
                self.assertFalse((root / "b" / "ebbinghaus.db").exists(), "incomplete text must not fall back to raw storage")
            finally:
                httpd.shutdown()
                httpd.server_close()

    def test_optional_real_hakua_writer_stores_once_and_recall_does_not_write(self):
        try:
            import hakua_memory.ebbinghaus.store  # noqa: F401 - optional local package
        except ImportError as error:
            self.skipTest("hakua_memory is not installed: " + str(error))
        with tempfile.TemporaryDirectory(prefix="strata-hakua-integration-") as directory, \
                patch.dict(os.environ, {"STRATA_SYNTHETIC_MEMORY_KEY": "synthetic-local-test-key"}):
            root = Path(directory) / "profile"
            harness = mh.Harness([{"principal": "synthetic", "key_env": "STRATA_SYNTHETIC_MEMORY_KEY",
                                   "root": str(root)}], ["synthetic-local-test-key"])
            turn, _ = harness.begin("synthetic-local-test-key",
                                    {"read": True, "store_summary": True, "turn_id": "synthetic_turn_1"},
                                    [{"role": "user", "content": "synthetic blue preference"}], threading.Event())
            self.assertEqual(harness.complete(turn, "synthetic completed response",
                                              lambda *args: "synthetic blue preference summary"), "stored")
            self.assertEqual(harness.complete(turn, "synthetic completed response",
                                              lambda *args: self.fail("summary replayed")), "stored")
            with closing(sqlite3.connect(root / "ebbinghaus.db")) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0], 1)
            before = {p.name: hashlib.sha256(p.read_bytes()).digest() for p in root.iterdir() if p.is_file()}
            self.assertEqual(len(harness.recall(turn.principal, "blue")), 1)
            after = {p.name: hashlib.sha256(p.read_bytes()).digest() for p in root.iterdir() if p.is_file()}
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
