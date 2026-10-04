"""serve/test_responses.py - #451: the Responses API (POST /v1/responses) against the mock engine (no GPU, no pack).

    python -m unittest serve.test_responses -v
"""
from __future__ import annotations

import base64
import contextlib
import http.client
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.responses import (ENCRYPTED_PREFIX, ResponsesError, input_messages, request_tools,  # noqa: E402
                             template_kwargs, text_format)
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ChatTemplate(ROOT / "serve/chat_template.jinja")

CALL = ("Let me look.\n</think>\n\nChecking.\n\n<tool_call>\n<function=exec_command>\n<parameter=cmd>\ncat a.txt\n"
        "</parameter>\n</function>\n</tool_call>")
ANSWER = "Read it.\n</think>\n\nThe file says before."
TOOLS = [{"type": "function", "name": "exec_command", "description": "Runs a command.", "strict": False,
          "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}]


# ------------------------------------------------------------------------------------------------ parsing
class Parsing(unittest.TestCase):
    def test_a_string_input_is_one_user_message(self):
        self.assertEqual(input_messages({"input": "hi", "instructions": "Be brief."}),
                         [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}])

    def test_leading_developer_messages_join_the_system_message_later_ones_become_user(self):
        msgs = input_messages({"instructions": "I", "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "D1"},
                                                                 {"type": "input_text", "text": "D2"}]},
            {"role": "user", "content": "u"},
            {"type": "message", "role": "developer", "content": "late"}]})
        self.assertEqual(msgs, [{"role": "system", "content": "I\n\nD1D2"}, {"role": "user", "content": "u"},
                                {"role": "user", "content": "late"}])

    def test_codex_history_becomes_one_assistant_turn_per_answer(self):
        items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "edit it"}]},
                 {"type": "reasoning", "id": "rs_1", "summary": [],
                  "content": [{"type": "reasoning_text", "text": "look first"}]},
                 {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Looking."}]},
                 {"type": "function_call", "id": "fc_1", "call_id": "c1", "name": "exec_command",
                  "arguments": "{\"cmd\":\"cat a\"}"},
                 {"type": "function_call", "id": "fc_2", "call_id": "c2", "name": "spawn_agent",
                  "namespace": "multi_agent_v1", "arguments": "{}"},
                 # results in the order they finished, not the order of the calls
                 {"type": "function_call_output", "call_id": "c2", "output": "agent"},
                 {"type": "function_call_output", "call_id": "c1",
                  "output": [{"type": "input_text", "text": "before"}]},
                 {"type": "reasoning", "id": "rs_2", "summary": [{"type": "summary_text", "text": "s"}],
                  "content": [{"type": "reasoning_text", "text": "now write"}]},
                 {"type": "function_call", "call_id": "c3", "name": "exec_command", "arguments": "{\"cmd\":\"w\"}"},
                 {"type": "function_call_output", "call_id": "c3", "output": "ok"}]
        self.assertEqual(input_messages({"input": items}), [
            {"role": "user", "content": "edit it"},
            {"role": "assistant", "content": "Looking.", "reasoning_content": "look first", "tool_calls": [
                {"function": {"name": "exec_command", "arguments": {"cmd": "cat a"}}},
                {"function": {"name": "multi_agent_v1.spawn_agent", "arguments": {}}}]},
            {"role": "tool", "content": "before"}, {"role": "tool", "content": "agent"},
            {"role": "assistant", "content": "", "reasoning_content": "now write", "tool_calls": [
                {"function": {"name": "exec_command", "arguments": {"cmd": "w"}}}]},
            {"role": "tool", "content": "ok"}])

    def test_reasoning_comes_back_from_our_encrypted_content_and_foreign_tokens_are_ignored(self):
        enc = ENCRYPTED_PREFIX + base64.b64encode(json.dumps({"text": "hidden"}).encode()).decode()
        for item, want in (({"encrypted_content": enc, "summary": []}, "hidden"),
                           ({"encrypted_content": "gAAAAAB-someone-elses", "summary": []}, None)):
            msgs = input_messages({"input": [{"role": "user", "content": "q"}, {"type": "reasoning", **item},
                                             {"role": "assistant", "content": "a"}]})
            self.assertEqual(msgs[1].get("reasoning_content"), want)
        with self.assertRaises(ResponsesError) as e:
            input_messages({"input": [{"type": "reasoning", "encrypted_content": ENCRYPTED_PREFIX + "!!"}]})
        self.assertEqual(e.exception.param, "input[0].encrypted_content")

    def test_bad_arguments_from_a_cut_off_call_do_not_refuse_the_conversation(self):
        msgs = input_messages({"input": [{"role": "user", "content": "q"},
                                         {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{\"a"},
                                         {"type": "function_call_output", "call_id": "c", "output": "x"}]})
        self.assertEqual(msgs[1]["tool_calls"][0]["function"]["arguments"], {"arguments": "{\"a"})

    def test_unsupported_items_are_named(self):
        for items, param in (([{"type": "item_reference", "id": "msg_1"}], "input[0]"),
                             ([{"type": "web_search_call"}], "input[0].type"),
                             ([{"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}],
                              "input[0].content[0].type"),
                             ([{"role": "robot", "content": "x"}], "input[0].role")):
            with self.assertRaises(ResponsesError) as e:
                input_messages({"input": items})
            self.assertEqual(e.exception.param, param)
        with self.assertRaises(ResponsesError):
            input_messages({})

    def test_tools_namespaces_custom_and_hosted(self):
        tools, names, skipped = request_tools({"tools": [
            *TOOLS, {"type": "namespace", "name": "mcp_fs", "description": "Files.",
                     "tools": [{"type": "function", "name": "read", "parameters": {"type": "object"}}]},
            {"type": "custom", "name": "apply_patch", "description": "Patch.",
             "format": {"type": "grammar", "syntax": "lark", "definition": "start: \"x\""}},
            {"type": "web_search"}]})
        self.assertEqual([t["name"] for t in tools], ["exec_command", "mcp_fs.read", "apply_patch"])
        self.assertEqual(tools[1]["description"], "Files.")
        self.assertIn("start: \"x\"", tools[2]["description"])
        self.assertEqual(names["mcp_fs.read"], ("mcp_fs", "read", "function"))
        self.assertEqual(names["apply_patch"], (None, "apply_patch", "custom"))
        self.assertEqual(skipped, ["web_search"])
        self.assertIsNone(request_tools({"tools": TOOLS, "tool_choice": "none"})[0])

    def test_effort_and_text_format(self):
        self.assertEqual(template_kwargs({"reasoning": {"effort": "minimal"}}, {}), {"enable_thinking": False})
        self.assertEqual(template_kwargs({"reasoning": {"effort": "high"}}, {}), {"reasoning_effort": "xhigh"})
        self.assertEqual(template_kwargs({"reasoning": {"summary": "auto"}}, {}), {})      # Codex: the default
        self.assertEqual(template_kwargs({}, {"reasoning_effort": "low"}), {"reasoning_effort": "low"})
        with self.assertRaises(ResponsesError):
            template_kwargs({"reasoning": {"effort": "huge"}}, {})
        self.assertEqual(text_format({"text": {"format": {"type": "json_schema", "name": "n", "schema": {}}}}),
                         {"type": "json_schema", "json_schema": {"name": "n", "schema": {}}})
        self.assertIsNone(text_format({"text": {"format": {"type": "text"}}}))


# ------------------------------------------------------------------------------------------------ over HTTP
class Server(unittest.TestCase):
    script = ANSWER

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, self.script, max_context=16384)
        self.svc = Service(self.engine, self.tok, TEMPLATE)
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, body, path="/v1/responses", headers=None):
        """(status, parsed body: a dict, or the list of SSE events of a stream)."""
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                c.request("POST", path, body=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json", **(headers or {})})
                r = c.getresponse()
                raw = r.read().decode()
            if r.getheader("Content-Type") == "text/event-stream":
                events = []
                for block in raw.split("\n\n"):
                    lines = block.strip().splitlines()
                    if not lines or lines[0].startswith(":"):
                        continue
                    name = lines[0][len("event: "):]
                    data = json.loads(lines[1][len("data: "):])
                    self.assertEqual(name, data["type"])
                    events.append(data)
                return r.status, events
            return r.status, json.loads(raw)
        finally:
            c.close()



class OverHttp(Server):
    def test_non_streaming_text_with_reasoning(self):
        code, r = self.post({"model": "m", "input": "hi", "store": False})
        self.assertEqual(code, 200)
        self.assertEqual((r["object"], r["status"], r["error"]), ("response", "completed", None))
        rs, msg = r["output"]
        self.assertEqual(rs["type"], "reasoning")
        self.assertEqual(rs["content"], [{"type": "reasoning_text", "text": "Read it.\n"}])
        self.assertEqual(rs["summary"], [])
        self.assertNotIn("encrypted_content", rs)
        self.assertEqual(msg["type"], "message")
        self.assertEqual(msg["content"][0]["text"], "The file says before.")
        u = r["usage"]
        self.assertEqual(u["output_tokens"], len(self.tok.encode(ANSWER)) + 1)        # + the end-of-turn token
        self.assertEqual(u["output_tokens_details"]["reasoning_tokens"], len("Read it.\n</think>"))
        self.assertEqual(u["total_tokens"], u["input_tokens"] + u["output_tokens"])
        self.assertEqual(r["store"], False)

    def test_stream_event_order(self):
        code, events = self.post({"model": "m", "input": "hi", "stream": True,
                                  "include": ["reasoning.encrypted_content"]})
        self.assertEqual(code, 200)
        types = [e["type"] for e in events]
        deltas_r = types.count("response.reasoning_text.delta")
        deltas_t = types.count("response.output_text.delta")
        self.assertGreater(deltas_r, 0)
        self.assertGreater(deltas_t, 0)
        self.assertEqual(types, ["response.created", "response.in_progress",
                                 "response.output_item.added", *["response.reasoning_text.delta"] * deltas_r,
                                 "response.reasoning_text.done", "response.output_item.done",
                                 "response.output_item.added", "response.content_part.added",
                                 *["response.output_text.delta"] * deltas_t,
                                 "response.output_text.done", "response.content_part.done",
                                 "response.output_item.done", "response.completed"])
        self.assertEqual([e["sequence_number"] for e in events], list(range(len(events))))
        final = events[-1]["response"]
        self.assertEqual(final["status"], "completed")
        rs, msg = final["output"]
        self.assertEqual([e["item"]["id"] for e in events if e["type"] == "response.output_item.done"],
                         [rs["id"], msg["id"]])
        self.assertEqual("".join(e["delta"] for e in events if e["type"] == "response.output_text.delta"),
                         msg["content"][0]["text"])
        self.assertTrue(rs["encrypted_content"].startswith(ENCRYPTED_PREFIX))
        # the encrypted_content alone brings the thinking back into the prompt
        replay = input_messages({"input": [{"role": "user", "content": "hi"},
                                           {"type": "reasoning", "summary": [], "encrypted_content":
                                            rs["encrypted_content"]}, msg]})
        self.assertEqual(replay[1]["reasoning_content"], "Read it.\n")

    def test_max_output_tokens_ends_incomplete(self):
        code, r = self.post({"model": "m", "input": "hi", "max_output_tokens": 5})
        self.assertEqual(code, 200)
        self.assertEqual(r["status"], "incomplete")
        self.assertEqual(r["incomplete_details"], {"reason": "max_output_tokens"})
        self.assertEqual(r["output"][0]["status"], "incomplete")
        code, events = self.post({"model": "m", "input": "hi", "max_output_tokens": 5, "stream": True})
        self.assertEqual(events[-1]["type"], "response.incomplete")

    def test_effort_none_does_not_think(self):
        self.engine.scripts = [self.tok.encode("Hello.<|im_end|>", parse_special=True)]
        self.engine.script = self.engine.scripts[0]
        code, r = self.post({"model": "m", "input": "hi", "reasoning": {"effort": "none"}})
        self.assertEqual([o["type"] for o in r["output"]], ["message"])
        self.assertIn("<think>\n\n</think>", self.tok.decode(self.engine.last_prompt))

    def test_errors_use_the_responses_format(self):
        for body, param in (({"model": "m"}, "input"), ({"model": "m", "input": "x", "previous_response_id": "r"},
                                                         "previous_response_id"),
                            ({"model": "m", "input": "x", "reasoning": {"effort": "huge"}}, "reasoning.effort")):
            code, r = self.post(body)
            self.assertEqual(code, 400)
            self.assertEqual(set(r["error"]), {"message", "type", "param", "code"})
            self.assertEqual(r["error"]["param"], param)
        code, r = self.post({}, path="/v1/responses/resp_123/cancel")
        self.assertEqual(code, 404)
        self.assertEqual(r["error"]["code"], "not_found")
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        c.request("POST", "/v1/responses", body=b"{not json", headers={"Content-Type": "application/json"})
        r = c.getresponse()
        err = json.loads(r.read())["error"]
        c.close()
        self.assertEqual((r.status, err["type"], err["param"]), (400, "invalid_request_error", None))

    def test_too_long_for_the_context(self):
        code, r = self.post({"model": "m", "input": "x" * 20000})
        self.assertEqual(code, 400)
        self.assertEqual(r["error"]["code"], "context_length_exceeded")

    def test_api_key_and_foreign_pages(self):
        self.svc.api_key = "s3cret"
        self.assertEqual(self.post({"model": "m", "input": "hi"})[0], 401)
        self.assertEqual(self.post({"model": "m", "input": "hi"}, headers={"Authorization": "Bearer s3cret"})[0], 200)
        self.svc.api_key = None
        code, r = self.post({"model": "m", "input": "hi"}, headers={"Origin": "https://evil.example.com"})
        self.assertEqual(code, 403)

    def test_json_schema_text_format(self):
        self.engine.scripts = [self.tok.encode("</think>\n\n{\"n\": 3}<|im_end|>", parse_special=True)]
        self.engine.script = self.engine.scripts[0]
        fmt = {"type": "json_schema", "name": "num", "strict": True,
               "schema": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}}
        code, r = self.post({"model": "m", "input": "a number", "text": {"format": fmt}})
        self.assertEqual(code, 200, r)
        self.assertEqual(r["output"][-1]["content"][0]["text"], "{\"n\":3}")
        code, events = self.post({"model": "m", "input": "a number", "text": {"format": fmt}, "stream": True})
        self.assertEqual([e["delta"] for e in events if e["type"] == "response.output_text.delta"], ["{\"n\":3}"])
        self.assertEqual(events[-1]["type"], "response.completed")
        # an answer that fails the schema
        bad = {**fmt, "schema": {**fmt["schema"], "properties": {"n": {"type": "string"}}}}
        code, r = self.post({"model": "m", "input": "a number", "text": {"format": bad}})
        self.assertEqual((code, r["error"]["code"]), (502, "structured_output_failed"))
        code, events = self.post({"model": "m", "input": "a number", "text": {"format": bad}, "stream": True})
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertEqual(events[-1]["response"]["error"]["code"], "structured_output_failed")

    def test_monitor_and_status(self):
        self.svc.api_monitor = True
        self.post({"model": "m", "input": "hi", "stream": True})
        rec = self.svc.request_records()[0]
        self.assertEqual(rec["path"], "/v1/responses")
        full = self.svc.request_records(rec["id"])
        self.assertEqual(full["output"], "The file says before.")
        self.assertEqual(full["reasoning"], "Read it.\n")
        self.assertIn("/v1/responses", self.svc.v1_status()["dialects"])


class ToolRoundTrip(Server):
    script = [CALL, ANSWER]

    def test_function_call_round_trip(self):
        user = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "what is in a.txt?"}]}
        code, events = self.post({"model": "m", "instructions": "You are a coding agent.", "input": [user],
                                  "tools": TOOLS, "store": False, "stream": True,
                                  "include": ["reasoning.encrypted_content"], "reasoning": {"summary": "auto"},
                                  "parallel_tool_calls": True, "tool_choice": "auto"})
        self.assertEqual(code, 200)
        first_prompt = self.tok.decode(self.engine.last_prompt)
        types = [e["type"] for e in events]
        n_args = types.count("response.function_call_arguments.delta")
        self.assertGreater(n_args, 0)
        call_part = types[types.index("response.output_text.done") + 2:]
        self.assertEqual(call_part, ["response.output_item.done", "response.output_item.added",
                                     *["response.function_call_arguments.delta"] * n_args,
                                     "response.function_call_arguments.done", "response.output_item.done",
                                     "response.completed"])
        output = events[-1]["response"]["output"]
        self.assertEqual([o["type"] for o in output], ["reasoning", "message", "function_call"])
        fc = output[2]
        self.assertEqual((fc["name"], json.loads(fc["arguments"]), fc["status"]),
                         ("exec_command", {"cmd": "cat a.txt"}, "completed"))
        self.assertTrue(fc["call_id"])
        added = next(e for e in events if e["type"] == "response.output_item.added"
                     and e["item"]["type"] == "function_call")
        self.assertEqual((added["item"]["call_id"], added["item"]["arguments"]), (fc["call_id"], ""))
        # Codex sends everything back, plus the result: the prompt continues exactly where the model stopped
        code, r = self.post({"model": "m", "instructions": "You are a coding agent.", "tools": TOOLS, "store": False,
                             "input": [user, *output, {"type": "function_call_output", "call_id": fc["call_id"],
                                                       "output": "before"}]})
        self.assertEqual(code, 200, r)
        second_prompt = self.tok.decode(self.engine.last_prompt)
        self.assertTrue(second_prompt.startswith(first_prompt + CALL + "<|im_end|>"), second_prompt[-600:])
        self.assertIn("<tool_response>\nbefore\n</tool_response>", second_prompt)
        self.assertEqual(r["output"][-1]["content"][0]["text"], "The file says before.")

    def test_namespace_and_custom_calls_come_back_under_their_own_names(self):
        self.engine.scripts = [self.tok.encode(
            "</think>\n\n<tool_call>\n<function=multi_agent_v1.spawn_agent>\n<parameter=message>\nhi\n</parameter>\n"
            "</function>\n</tool_call>\n<tool_call>\n<function=apply_patch>\n<parameter=input>\n*** Begin Patch\n"
            "</parameter>\n</function>\n</tool_call><|im_end|>", parse_special=True)]
        self.engine.script = self.engine.scripts[0]
        tools = [{"type": "namespace", "name": "multi_agent_v1", "description": "Agents.", "tools": [
            {"type": "function", "name": "spawn_agent", "strict": False,
             "parameters": {"type": "object", "properties": {"message": {"type": "string"}}}}]},
            {"type": "custom", "name": "apply_patch", "description": "Edit files."}]
        code, r = self.post({"model": "m", "input": "go", "tools": tools})
        self.assertEqual(code, 200, r)
        ns, custom = r["output"]
        self.assertEqual((ns["type"], ns["namespace"], ns["name"], json.loads(ns["arguments"])),
                         ("function_call", "multi_agent_v1", "spawn_agent", {"message": "hi"}))
        self.assertEqual((custom["type"], custom["name"], custom["input"]),
                         ("custom_tool_call", "apply_patch", "*** Begin Patch"))
        msgs = input_messages({"input": [{"role": "user", "content": "go"}, ns, custom,
                                         {"type": "function_call_output", "call_id": ns["call_id"], "output": "1"},
                                         {"type": "custom_tool_call_output", "call_id": custom["call_id"],
                                          "output": "2"}]})
        self.assertEqual([c["function"]["name"] for c in msgs[1]["tool_calls"]],
                         ["multi_agent_v1.spawn_agent", "apply_patch"])


if __name__ == "__main__":
    unittest.main()
