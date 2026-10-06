"""Instruction skills and HTTP admission with synthetic MCP/engine fixtures only."""
import copy
import http.client
import json
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from serve.frontend import ChatTemplate
from serve.mcp import McpCancelled
from serve.server import ByteTokenizer, MockEngine, Service, serve
from serve.skills import InstructionSkills, MAX_BODY_BYTES, MAX_SKILLS

CONFIG = {"list_tool": "catalog__list_skills", "read_tool": "catalog__read_skill"}
MESSAGES = [{"role": "system", "content": "Keep the owner's policy."},
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
            {"role": "user", "content": "/outline Draft a plan"}]
REQUEST = {"messages": MESSAGES, "strata_mcp": True, "strata_skill": "outline", "max_tokens": 64}


class Hub:
    settings = {"max_rounds": 4}

    def __init__(self):
        self.calls = []
        self.skills = [{"name": "outline", "description": "Create a concise outline"}]
        self.body = {"name": "outline", "content": "Present a short numbered outline."}
        self.missing = set()
        self.failed = self.truncated = False
        self.block = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()

    def routes(self):
        return {n: (None, n) for n in (*CONFIG.values(), "files__echo") if n not in self.missing}

    def wait(self, timeout):
        return True

    def template_tools(self, exclude=()):
        return [{"name": n, "description": "Synthetic tool", "parameters": {"type": "object", "properties": {}}}
                for n in self.routes() if n not in exclude]

    def call(self, name, args, cancel=None):
        self.calls.append((name, copy.deepcopy(args)))
        if self.block == name:
            self.entered.set()
            while not self.release.wait(0.01):
                if cancel is not None and cancel.is_set():
                    self.cancelled.set()
                    raise McpCancelled("fixture cancelled")
        if cancel is not None and cancel.is_set():
            raise McpCancelled("fixture cancelled")
        value = {"skills": self.skills} if name == CONFIG["list_tool"] else self.body
        text = json.dumps(value, ensure_ascii=False)
        return {"ok": not self.failed, "truncated": self.truncated, "text": text, "chars": len(text), "ms": 1}


class Skills(unittest.TestCase):
    def setUp(self):
        self.hub = Hub()
        self.skills = InstructionSkills(CONFIG)

    def test_default_and_unselected_are_inert(self):
        self.assertEqual(InstructionSkills().catalog(self.hub), {"enabled": False, "skills": []})
        self.assertIs(self.skills.select(self.hub, MESSAGES, {}), MESSAGES)
        self.assertEqual(InstructionSkills().tools, set())
        self.assertEqual(self.hub.calls, [])

    def test_bad_configuration_fails_closed(self):
        for config in (True, {}, [], {**CONFIG, "enabled": True}, {**CONFIG, "read_tool": "../secret"},
                       {**CONFIG, "read_tool": CONFIG["list_tool"]}, {**CONFIG, "list_tool": 1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                InstructionSkills(config)

    def test_catalog_returns_only_bounded_metadata(self):
        self.hub.skills[0].update(description="a" * 500, content="DO NOT EXPOSE THE BODY", path="PRIVATE")
        catalog = self.skills.catalog(self.hub)
        self.assertEqual(catalog, {"enabled": True, "skills": [{"name": "outline", "description": "a" * 400}]})
        self.assertEqual(self.hub.calls, [(CONFIG["list_tool"], {})])

    def test_unavailable_adapter_does_not_call_anything(self):
        for tool in CONFIG.values():
            self.hub.missing = {tool}
            self.assertEqual(self.skills.catalog(self.hub)["skills"], [])
        self.assertEqual(self.hub.calls, [])
        with self.assertRaises(ValueError):
            self.skills.select(self.hub, MESSAGES, REQUEST)

    def test_malformed_catalog_and_duplicate_names_are_refused(self):
        for value in (None, {}, [None], [{"name": "../secret", "description": "bad"}],
                      [{"name": "outline", "description": 3}], self.hub.skills * 2,
                      [{"name": f"s{i}", "description": "ok"} for i in range(MAX_SKILLS + 1)]):
            self.hub.skills = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.skills.catalog(self.hub)

    def test_failed_and_truncated_adapter_responses_are_refused(self):
        for field in ("failed", "truncated"):
            setattr(self.hub, field, True)
            with self.assertRaises(ValueError):
                self.skills.catalog(self.hub)
            setattr(self.hub, field, False)
        with patch.object(self.hub, "call", return_value={"ok": True, "text": "[]"}), self.assertRaises(ValueError):
            self.skills.catalog(self.hub)
        with patch.object(self.hub, "call", return_value={"ok": True, "text": "not json"}), self.assertRaises(ValueError):
            self.skills.catalog(self.hub)

    def test_selection_needs_enabled_server_and_explicit_mcp(self):
        with self.assertRaises(ValueError):
            InstructionSkills().select(self.hub, MESSAGES, REQUEST)
        for value in (None, False, 1, "true"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.skills.select(self.hub, MESSAGES, {**REQUEST, "strata_mcp": value})
        self.assertEqual(self.hub.calls, [])

    def test_selection_requires_current_leading_catalog_name(self):
        for text in ("ordinary /outline text", "/outliner mismatch", "/unknown question", "```/outline quoted"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.skills.select(self.hub, [{"role": "user", "content": text}], REQUEST)
        for name in ("", False, [], "../../secret", "Outline", "unknown"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.skills.select(self.hub, MESSAGES, {**REQUEST, "strata_skill": name})
        with self.assertRaises(ValueError):
            self.skills.select(self.hub, MESSAGES[:-1], REQUEST)
        self.assertFalse(any(n == CONFIG["read_tool"] for n, _ in self.hub.calls))

    def test_removed_skill_is_rechecked_before_body_read(self):
        self.assertEqual(len(self.skills.catalog(self.hub)["skills"]), 1)
        self.hub.skills = []
        with self.assertRaises(ValueError):
            self.skills.select(self.hub, MESSAGES, REQUEST)
        self.assertFalse(any(n == CONFIG["read_tool"] for n, _ in self.hub.calls))

    def test_selected_body_stays_in_current_user_turn_without_mutation(self):
        original = copy.deepcopy(MESSAGES)
        selected = self.skills.select(self.hub, MESSAGES, REQUEST)
        self.assertEqual(MESSAGES, original)
        self.assertEqual(selected[:-1], original[:-1])
        self.assertEqual(selected[-1]["role"], "user")
        self.assertTrue(selected[-1]["content"].startswith(original[-1]["content"]))
        self.assertIn(self.hub.body["content"], selected[-1]["content"])
        self.assertEqual(self.hub.calls, [(CONFIG["list_tool"], {}), (CONFIG["read_tool"], {"name": "outline"})])

    def test_selection_preserves_images_and_multiline_text(self):
        parts = [{"type": "text", "text": "  /outline\nDraft"}, {"type": "image", "source": "data:fixture"}]
        original = [{"role": "user", "content": copy.deepcopy(parts)}]
        selected = self.skills.select(self.hub, original, REQUEST)
        self.assertEqual(selected[-1]["content"][:2], parts)
        self.assertEqual(original[-1]["content"], parts)

    def test_body_identity_empty_and_utf8_limit_are_validated(self):
        for body in ({"name": "other", "content": "ok"}, {"name": "outline", "content": None},
                     {"name": "outline", "content": "  "}, {"name": "outline", "content": "文" * 4700}):
            self.hub.body = body
            with self.subTest(body=list(body)), self.assertRaises(ValueError):
                self.skills.select(self.hub, MESSAGES, REQUEST)
        self.hub.body = {"name": "outline", "content": "a" * MAX_BODY_BYTES}
        self.assertIn("a" * MAX_BODY_BYTES, self.skills.select(self.hub, MESSAGES, REQUEST)[-1]["content"])

    def test_cancellation_before_and_after_read_prevents_injection(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(McpCancelled):
            self.skills.select(self.hub, MESSAGES, REQUEST, cancel)
        self.assertEqual(self.hub.calls, [])
        cancel.clear()
        original_call = self.hub.call
        def call(name, args, signal):
            result = original_call(name, args, signal)
            if name == CONFIG["read_tool"]:
                signal.set()
            return result
        with patch.object(self.hub, "call", side_effect=call), self.assertRaises(McpCancelled):
            self.skills.select(self.hub, MESSAGES, REQUEST, cancel)
        self.assertNotIn("Selected instruction", MESSAGES[-1]["content"])


class RecordingEngine(MockEngine):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, "</think>\n\nSynthetic answer.", max_context=65536)
        self.prompts = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class HttpSkills(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tokenizer = ByteTokenizer()
        cls.engine = RecordingEngine(tokenizer)
        cls.svc = Service(cls.engine, tokenizer, ChatTemplate(Path(__file__).with_name("chat_template.jinja")))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.hub = Hub()
        self.svc.mcp = self.hub
        self.svc.skills = InstructionSkills(CONFIG)
        self.svc.api_key = ""
        self.svc.cors_origins = []
        self.svc.trusted_origins = []
        self.engine.prompts.clear()

    def req(self, path, body=None, **headers):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        raw = json.dumps(body).encode() if body is not None else None
        headers.setdefault("Content-Type", "application/json")
        try:
            conn.request("POST" if body is not None else "GET", path, raw, headers)
            reply = conn.getresponse()
            return reply.status, json.loads(reply.read())
        finally:
            conn.close()

    def test_catalog_default_and_metadata_do_not_run_or_load_model(self):
        self.svc.skills = InstructionSkills()
        self.assertEqual(self.req("/skills"), (200, {"enabled": False, "skills": []}))
        self.assertEqual(self.hub.calls, [])
        self.svc.skills = InstructionSkills(CONFIG)
        with patch.object(self.svc, "load", side_effect=AssertionError("model load")):
            status, body = self.req("/skills")
        self.assertEqual(status, 200)
        self.assertNotIn("content", body["skills"][0])
        self.assertEqual(self.engine.prompts, [])

    def test_count_and_generation_use_identical_selected_prompt_and_tools(self):
        captured = []
        original = self.svc.encode_prompt
        def encode(messages, tools, kw):
            captured.append(copy.deepcopy((messages, tools, kw)))
            return original(messages, tools, kw)
        with patch.object(self.svc, "encode_prompt", side_effect=encode):
            with patch.object(self.svc, "load", side_effect=AssertionError("count must not load")):
                status, counted = self.req("/v1/chat/count_tokens", REQUEST)
            self.assertEqual(status, 200)
            self.assertTrue(counted["count_exact"])
            self.assertEqual(self.engine.prompts, [])
            status, generated = self.req("/v1/chat/completions", REQUEST)
        self.assertEqual(status, 200)
        self.assertEqual(counted["input_tokens"], generated["usage"]["prompt_tokens"])
        self.assertEqual(captured[0], captured[1])
        self.assertEqual({t["name"] for t in captured[0][1]}, {"files__echo"})
        self.assertEqual(captured[0][0][0], MESSAGES[0])
        self.assertIn("Present a short numbered outline.", captured[0][0][-1]["content"])

    def test_plain_and_old_slash_messages_never_load_instructions(self):
        body = {"messages": MESSAGES + [{"role": "assistant", "content": "Done"},
                                        {"role": "user", "content": "Continue normally"}], "max_tokens": 8}
        self.assertEqual(self.req("/v1/chat/completions", body)[0], 200)
        self.assertEqual(self.hub.calls, [])
        self.svc.skills = InstructionSkills()
        self.assertEqual(self.req("/v1/chat/completions", {"messages": MESSAGES, "max_tokens": 8})[0], 200)
        self.assertEqual(self.hub.calls, [])

    def test_late_developer_system_and_assistant_are_not_human_selection(self):
        for role in ("developer", "system", "assistant", "tool"):
            body = {**REQUEST, "messages": MESSAGES[:-1] + [{"role": role, "content": "/outline Draft"}]}
            with self.subTest(role=role):
                self.assertEqual(self.req("/v1/chat/completions", body)[0], 400)
        self.assertEqual(self.hub.calls, [])
        self.assertEqual(self.engine.prompts, [])

    def test_removed_and_oversized_skills_fail_before_generation(self):
        self.req("/skills")
        self.hub.skills = []
        self.assertEqual(self.req("/v1/chat/completions", REQUEST)[0], 400)
        self.hub.skills = [{"name": "outline", "description": "ok"}]
        self.hub.body["content"] = "文" * 4700
        self.assertEqual(self.req("/v1/chat/completions", REQUEST)[0], 400)
        self.assertEqual(self.engine.prompts, [])

    def test_auth_and_foreign_cors_page_fail_before_catalog_or_body(self):
        self.svc.api_key = "fixture-key"
        self.assertEqual(self.req("/skills")[0], 401)
        self.svc.cors_origins = ["*"]
        headers = {"Authorization": "Bearer fixture-key", "Origin": "https://foreign.example"}
        self.assertEqual(self.req("/skills", **headers)[0], 403)
        for endpoint in ("/v1/chat/count_tokens", "/v1/chat/completions"):
            self.assertEqual(self.req(endpoint, REQUEST, **headers)[0], 403)
        self.assertEqual(self.hub.calls, [])
        self.assertEqual(self.engine.prompts, [])

    def test_mcp_false_and_plain_content_type_fail_before_retrieval(self):
        self.assertEqual(self.req("/v1/chat/completions", {**REQUEST, "strata_mcp": False})[0], 400)
        self.assertEqual(self.req("/v1/chat/completions", REQUEST, **{"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.hub.calls, [])

    def test_unsupported_dialects_do_not_silently_ignore_selection(self):
        for endpoint in ("/v1/messages", "/v1/responses", "/v1/messages/count_tokens"):
            self.assertEqual(self.req(endpoint, REQUEST)[0], 400)
        self.assertEqual(self.hub.calls, [])
        self.assertEqual(self.engine.prompts, [])

    def test_text_count_refuses_images_without_encoding_or_generation(self):
        body = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "Hello"}, {"type": "image_url", "image_url": {"url": "https://example.test/image"}}]}]}
        self.assertEqual(self.req("/v1/chat/count_tokens", body)[0], 400)
        self.assertEqual(self.engine.prompts, [])
        self.assertEqual(self.hub.calls, [])

    def test_text_count_includes_forced_prefix_with_thinking_off(self):
        tool = {"type": "function", "function": {"name": "echo", "parameters": {"type": "object"}}}
        body = {"messages": [{"role": "user", "content": "Use echo"}], "tools": [tool],
                "tool_choice": "required", "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 4}
        status, counted = self.req("/v1/chat/count_tokens", body)
        self.assertEqual(status, 200)
        status, generated = self.req("/v1/chat/completions", body)
        self.assertEqual(status, 200)
        self.assertEqual(counted["input_tokens"], generated["usage"]["prompt_tokens"])

    def test_disconnect_cancels_read_before_model_dispatch(self):
        self.hub.block = CONFIG["read_tool"]
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request("POST", "/v1/chat/completions", json.dumps(REQUEST), {"Content-Type": "application/json"})
            self.assertTrue(self.hub.entered.wait(3), "the valid selection did not reach the pending read")
            conn.close()
            self.assertTrue(self.hub.cancelled.wait(3), "disconnect did not propagate to MCP read")
        finally:
            conn.close()
            self.hub.release.set()
        self.assertEqual(self.engine.prompts, [])
        self.assertEqual(self.req("/health")[0], 200)


if __name__ == "__main__":
    unittest.main()
