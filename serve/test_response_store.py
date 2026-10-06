"""Stored Responses against the existing mock engine; no model files or GPU needed.

    python -m unittest serve.test_response_store -v
"""
import concurrent.futures
import contextlib
import http.client
import io
import importlib.util
import json
import unittest

from serve.response_store import ResponseStore
from serve.responses import ResponsesError
from serve.test_responses import ANSWER, CALL, TOOLS, Server


class Store(unittest.TestCase):
    def save(self, store, response_id, text="answer", **req):
        response = {"id": response_id, "store": True, "output": [{"role": "assistant", "content": text}]}
        store.save({"input": "question", **req}, response)
        return response

    def test_expiry_and_delete_release_the_budget(self):
        now = [0]
        store = ResponseStore(10000, ttl_seconds=10, clock=lambda: now[0])
        self.save(store, "resp_a")
        now[0] = 9
        self.assertEqual(store.get("resp_a")["id"], "resp_a")
        now[0] = 10
        with self.assertRaises(ResponsesError) as e:
            store.get("resp_a")
        self.assertEqual(e.exception.status, 404)
        self.assertEqual(store.bytes_used, 0)
        self.save(store, "resp_b")
        self.assertEqual(store.delete("resp_b"), {"id": "resp_b", "object": "response.deleted", "deleted": True})
        self.assertEqual(store.bytes_used, 0)

    def test_count_and_byte_limits_evict_oldest_and_reject_oversized_records(self):
        store = ResponseStore(10000, max_entries=2)
        for response_id in ("resp_a", "resp_b", "resp_c"):
            self.save(store, response_id)
        with self.assertRaises(ResponsesError):
            store.get("resp_a")
        one = store.bytes_used // 2
        store = ResponseStore(one + 1)
        self.save(store, "resp_a")
        self.save(store, "resp_b")
        with self.assertRaises(ResponsesError):
            store.get("resp_a")
        with self.assertRaises(ResponsesError) as e:
            self.save(store, "resp_big", "字" * 10000)
        self.assertEqual((e.exception.status, e.exception.code), (413, "response_store_limit_exceeded"))
        self.assertLessEqual(store.bytes_used, store.max_bytes)
        self.assertEqual(store.get("resp_b")["id"], "resp_b")

    def test_snapshots_do_not_share_mutable_input_or_output(self):
        store = ResponseStore(10000)
        items = [{"role": "user", "content": "original"}]
        response = self.save(store, "resp_a", input=items)
        items[0]["content"] = "mutated"
        response["output"].clear()
        store.get("resp_a")["output"].clear()
        replay = store.prepare({"previous_response_id": "resp_a", "input": "next"})
        self.assertEqual(replay["input"][0]["content"], "original")
        self.assertEqual(replay["input"][1]["content"], "answer")
        self.assertTrue(store.get("resp_a")["output"])

    def test_concurrent_saves_gets_and_deletes_keep_accounting_consistent(self):
        store = ResponseStore(1000000)
        def round_trip(n):
            response_id = f"resp_{n}"
            self.save(store, response_id)
            self.assertEqual(store.get(response_id)["id"], response_id)
            store.delete(response_id)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(round_trip, range(64)))
        self.assertEqual(store.bytes_used, 0)
        self.assertFalse(store.records)

    def test_invalid_budgets_and_lifetimes(self):
        for values in ({"max_bytes": 0}, {"max_bytes": True}, {"max_bytes": 100, "max_entries": 0},
                       {"max_bytes": 100, "ttl_seconds": 0}, {"max_bytes": 100, "ttl_seconds": float("nan")},
                       {"max_bytes": 100, "ttl_seconds": True}):
            with self.assertRaises(ValueError):
                ResponseStore(**values)


class StoredOverHttp(Server):
    def setUp(self):
        super().setUp()
        self.svc.responses_store = ResponseStore(1024 * 1024)

    def resource(self, method, response_id, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                c.request(method, f"/v1/responses/{response_id}", headers=headers or {})
                r = c.getresponse()
                return r.status, json.loads(r.read())
        finally:
            c.close()

    def test_save_retrieve_delete_and_not_found(self):
        code, response = self.post({"input": "你好", "store": True})
        self.assertEqual(code, 200, response)
        self.assertTrue(response["store"])
        response_id = response["id"]
        self.assertEqual(self.resource("GET", response_id), (200, response))
        self.assertEqual(self.resource("DELETE", response_id),
                         (200, {"id": response_id, "object": "response.deleted", "deleted": True}))
        for method in ("GET", "DELETE"):
            code, error = self.resource(method, response_id)
            self.assertEqual((code, error["error"]["code"]), (404, "not_found"))

    def test_enabled_defaults_to_store_and_false_does_not_store(self):
        for field, expected in (({}, True), ({"store": None}, True), ({"store": False}, False)):
            code, response = self.post({"input": "hi", **field})
            self.assertEqual(code, 200, response)
            self.assertEqual(response["store"], expected)
            self.assertEqual(self.resource("GET", response["id"])[0], 200 if expected else 404)

    def test_disabled_keeps_the_stateless_behavior(self):
        self.svc.responses_store = None
        # as before storage existed: `store` is not looked at (the answer says false) and nothing can be retrieved
        for field in ({}, {"store": True}, {"store": "false"}, {"previous_response_id": ""},
                      {"previous_response_id": None}):
            code, response = self.post({"input": "hi", **field})
            self.assertEqual(code, 200, response)
            self.assertEqual((response["store"], response["previous_response_id"]), (False, None))
            self.assertEqual(self.resource("GET", response["id"])[0], 404)
        code, error = self.post({"input": "hi", "previous_response_id": "resp_x"})
        self.assertEqual((code, error["error"]["param"], error["error"]["code"]),
                         (400, "previous_response_id", "unsupported_parameter"))

    def test_continuation_includes_previous_input_reasoning_and_answer(self):
        _, first = self.post({"input": "first question"})
        code, second = self.post({"previous_response_id": first["id"], "input": "follow-up", "store": False})
        self.assertEqual(code, 200, second)
        self.assertEqual(second["previous_response_id"], first["id"])
        prompt = self.tok.decode(self.engine.last_prompt)
        for text in ("first question", "Read it.", "The file says before.", "follow-up"):
            self.assertIn(text, prompt)
        self.assertEqual(self.resource("GET", second["id"])[0], 404)

    def test_a_continuation_reads_the_same_prompt_as_the_whole_history(self):
        # what keeps the conversation cache: the replay is the prompt a client sending everything gets, and the
        # previous prompt is its start
        _, first = self.post({"input": "first question"})
        before = self.engine.last_prompt[:]
        self.post({"previous_response_id": first["id"], "input": "follow-up"})
        replayed = self.engine.last_prompt[:]
        self.post({"input": [{"role": "user", "content": "first question"}, *first["output"],
                             {"role": "user", "content": "follow-up"}], "store": False})
        self.assertEqual(replayed, self.engine.last_prompt)
        self.assertEqual(replayed[:len(before)], before)

    def test_children_have_independent_history_and_survive_parent_deletion(self):
        _, root = self.post({"input": "root question"})
        _, left = self.post({"previous_response_id": root["id"], "input": "left branch"})
        _, right = self.post({"previous_response_id": root["id"], "input": "right branch"})
        self.assertNotIn("left branch", self.tok.decode(self.engine.last_prompt))
        self.resource("DELETE", root["id"])
        code, response = self.post({"previous_response_id": left["id"], "input": "third turn"})
        self.assertEqual(code, 200, response)
        prompt = self.tok.decode(self.engine.last_prompt)
        for text in ("root question", "left branch", "third turn"):
            self.assertIn(text, prompt)
        self.assertNotIn("right branch", prompt)
        self.assertEqual(self.resource("GET", right["id"])[0], 200)

    def test_top_level_instructions_are_not_inherited(self):
        _, first = self.post({"input": "first", "instructions": "ONLY_THE_FIRST_TURN"})
        code, response = self.post({"input": "next", "previous_response_id": first["id"],
                                   "instructions": "ONLY_THE_NEXT_TURN"})
        self.assertEqual(code, 200, response)
        prompt = self.tok.decode(self.engine.last_prompt)
        self.assertNotIn("ONLY_THE_FIRST_TURN", prompt)
        self.assertIn("ONLY_THE_NEXT_TURN", prompt)

    def test_function_output_can_continue_a_stored_tool_call(self):
        self.engine.scripts = [self.tok.encode(s + "<|im_end|>", parse_special=True) for s in (CALL, ANSWER)]
        _, first = self.post({"input": "read a.txt", "tools": TOOLS})
        call = next(x for x in first["output"] if x["type"] == "function_call")
        code, response = self.post({"previous_response_id": first["id"], "tools": TOOLS,
                                   "input": [{"type": "function_call_output", "call_id": call["call_id"],
                                              "output": "UNIQUE_TOOL_RESULT"}]})
        self.assertEqual(code, 200, response)
        prompt = self.tok.decode(self.engine.last_prompt)
        self.assertIn("cat a.txt", prompt)
        self.assertIn("UNIQUE_TOOL_RESULT", prompt)
        self.assertEqual(self.resource("GET", first["id"])[1]["output"], first["output"])

    def test_streamed_and_incomplete_responses_are_retrievable(self):
        for fields in ({"stream": True}, {"stream": True, "max_output_tokens": 5}):
            code, events = self.post({"input": "hi", **fields})
            self.assertEqual(code, 200)
            response = events[-1]["response"]
            self.assertEqual(events[-1]["type"], "response.incomplete" if "max_output_tokens" in fields
                             else "response.completed")
            self.assertEqual(self.resource("GET", response["id"]), (200, response))

    def test_invalid_and_expired_previous_ids_do_not_run_the_engine(self):
        now = [0]
        self.svc.responses_store = ResponseStore(100000, ttl_seconds=10, clock=lambda: now[0])
        _, first = self.post({"input": "first"})
        previous_prompt = self.engine.last_prompt[:]
        now[0] = 10
        for response_id in (first["id"], "resp_missing"):
            code, error = self.post({"input": "next", "previous_response_id": response_id})
            self.assertEqual((code, error["error"]["param"]), (404, "previous_response_id"))
        self.assertEqual(self.engine.last_prompt, previous_prompt)
        for field in ({"previous_response_id": []}, {"previous_response_id": ""}, {"store": "false"}):
            self.assertEqual(self.post({"input": "hi", **field})[0], 400)

    def test_oversized_storage_is_an_error_in_both_output_modes(self):
        self.svc.responses_store = ResponseStore(50)
        code, error = self.post({"input": "hi"})
        self.assertEqual((code, error["error"]["code"]), (413, "response_store_limit_exceeded"))
        _, events = self.post({"input": "hi", "stream": True})
        self.assertEqual(events[-1]["type"], "response.failed")
        response = events[-1]["response"]
        self.assertEqual(response["error"]["code"], "response_store_limit_exceeded")
        self.assertNotIn("completed_at", response)
        self.assertEqual(self.resource("GET", response["id"])[0], 404)

    def test_a_replay_too_large_to_keep_is_refused_before_the_model_runs(self):
        store = self.svc.responses_store
        _, first = self.post({"input": "first"})
        replay = store.prepare({"input": "next", "previous_response_id": first["id"]})["input"]
        store.max_bytes = len(json.dumps(replay, ensure_ascii=False, separators=(",", ":")).encode()) - 1
        previous_prompt = self.engine.last_prompt[:]
        code, error = self.post({"input": "next", "previous_response_id": first["id"]})
        self.assertEqual((code, error["error"]["code"]), (413, "response_store_limit_exceeded"))
        self.assertEqual(self.engine.last_prompt, previous_prompt)
        code, response = self.post({"input": "next", "previous_response_id": first["id"], "store": False})
        self.assertEqual(code, 200, response)                # nothing to keep: the same replay runs

    def test_errors_in_a_continuation_name_the_clients_own_items(self):
        _, first = self.post({"input": "first"})
        code, error = self.post({"previous_response_id": first["id"], "input": [{"role": "user", "content": "ok"}, 42]})
        self.assertEqual((code, error["error"]["param"]), (400, "input[1]"))
        code, error = self.post({"previous_response_id": first["id"], "input": [{"type": "item_reference", "id": "x"}]})
        self.assertEqual((code, error["error"]["param"], error["error"]["code"]),
                         (400, "input[0]", "unsupported_parameter"))
        self.assertIn("not supported", error["error"]["message"])

    def test_other_response_routes_are_named_unsupported(self):
        _, response = self.post({"input": "hi"})
        for method in ("GET", "DELETE"):
            code, error = self.resource(method, response["id"] + "/input_items")
            self.assertEqual(code, 404)
            self.assertIn("not supported", error["error"]["message"])
        self.assertEqual(self.resource("GET", response["id"])[0], 200)    # the response itself is untouched

    def test_failed_structured_output_is_not_saved(self):
        code, error = self.post({"input": "hi", "text": {"format": {"type": "json_object"}}})
        self.assertEqual((code, error["error"]["code"]), (502, "structured_output_failed"))
        _, events = self.post({"input": "hi", "text": {"format": {"type": "json_object"}}, "stream": True})
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertEqual(self.resource("GET", events[-1]["response"]["id"])[0], 404)
        self.assertFalse(self.svc.responses_store.records)

    def test_retrieve_stream_is_explicitly_refused(self):
        _, response = self.post({"input": "hi"})
        code, error = self.resource("GET", response["id"] + "?stream=true")
        self.assertEqual((code, error["error"]["param"]), (400, "stream"))

    def test_crud_authentication_host_and_foreign_page_checks(self):
        _, response = self.post({"input": "hi"})
        response_id = response["id"]
        self.svc.api_key = "secret"
        for method in ("GET", "DELETE"):
            self.assertEqual(self.resource(method, response_id)[0], 401)
        self.assertEqual(self.resource("GET", response_id, {"Authorization": "Bearer secret"})[0], 200)
        self.svc.api_key = ""
        self.assertEqual(self.resource("GET", response_id, {"Host": "evil.example"})[0], 403)
        self.assertEqual(self.resource("DELETE", response_id, {"Host": "evil.example"})[0], 403)
        self.assertEqual(self.resource("DELETE", response_id, {"Origin": "https://evil.example"})[0], 403)
        self.assertEqual(self.resource("GET", response_id)[0], 200)
        self.svc.cors_origins = ["https://allowed.example"]
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            c.request("OPTIONS", f"/v1/responses/{response_id}", headers={"Origin": "https://allowed.example"})
            r = c.getresponse()
            self.assertIn("DELETE", r.getheader("Access-Control-Allow-Methods"))
            r.read()
        finally:
            c.close()
        # the DELETE itself, as a browser sends it after that preflight: no body, so no JSON content type
        code, deleted = self.resource("DELETE", response_id, {"Origin": "https://allowed.example"})
        self.assertEqual((code, deleted["deleted"]), (200, True))


@unittest.skipUnless(importlib.util.find_spec("openai"), "optional OpenAI SDK not installed")
class OpenAIClient(Server):
    def setUp(self):
        super().setUp()
        self.svc.responses_store = ResponseStore(1024 * 1024)

    def test_sdk_create_continue_retrieve_stream_and_delete(self):
        from openai import OpenAI, NotFoundError
        with OpenAI(base_url=f"http://127.0.0.1:{self.port}/v1", api_key="none", max_retries=0) as client:
            first = client.responses.create(model="strata", input="first question", store=True)
            self.assertTrue(first.store)
            second = client.responses.create(model="strata", input="next question", previous_response_id=first.id)
            self.assertEqual(second.previous_response_id, first.id)
            self.assertEqual(client.responses.retrieve(second.id).model_dump(), second.model_dump())
            self.assertIn("first question", self.tok.decode(self.engine.last_prompt))
            with client.responses.stream(model="strata", input="third question",
                                         previous_response_id=second.id) as stream:
                events = list(stream)
                final = stream.get_final_response()
            self.assertEqual(events[-1].type, "response.completed")
            self.assertEqual(final.output_text, "The file says before.")
            saved = client.responses.retrieve(final.id)
            # The SDK's stream helper adds client-side `parsed` fields which are absent on the HTTP response.
            self.assertEqual((saved.id, saved.status, saved.output_text, saved.usage, saved.previous_response_id),
                             (final.id, final.status, final.output_text, final.usage, final.previous_response_id))
            client.responses.delete(first.id)
            with self.assertRaises(NotFoundError):
                client.responses.retrieve(first.id)


if __name__ == "__main__":
    unittest.main()
