"""Archive tools use the real MCP router without starting a process or model."""
import json
import pathlib
import sqlite3
import sys
import threading
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from serve.chat_memory import MAX_RESULT_CHARS, MemoryProvider
from serve.chat_archive import ChatArchive
from serve.mcp import McpCancelled, McpHub


class Archive:
    def __init__(self):
        self.calls = []
        self.error = None
        self.result = {"hits": [{"chat_id": "chat", "branch_id": "branch", "index": 3,
                                 "text": "The exact path is C:/AI_Models/Strata"}]}
        self.after = None

    def lookup(self, *arguments):
        self.calls.append(arguments)
        if self.error:
            raise self.error
        if self.after:
            self.after()
        return self.result

    def search(self, query, chat_id=None, limit=5):
        return self.lookup("search", query, chat_id, limit)

    def recall(self, chat_id, branch_id, start=0, count=5, offset=0):
        return self.lookup("recall", chat_id, branch_id, start, count, offset)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.archive = Archive()
        self.hub = McpHub({})
        self.provider = MemoryProvider(self.archive)
        self.hub.register_builtin(self.provider)
        self.hub.routes()

    def test_tools_count_status_and_no_external_process(self):
        self.hub.start(wait=True)
        tools = self.hub.template_tools()
        self.assertEqual({t["name"] for t in tools}, {"memory__search", "memory__recall"})
        self.assertEqual(len(self.hub.openai_tools()), 2)
        self.assertEqual([t["name"] for t in self.hub.template_tools(exclude={"memory__search"})],
                         ["memory__recall"])
        status = self.hub.status()
        self.assertEqual(status["tools"], 2)
        self.assertEqual(status["servers"][0]["transport"], "builtin")
        self.assertEqual(status["servers"][0]["status"], "ready")
        self.hub.close()
        self.assertEqual(self.provider.status, "stopped")

    def test_search_and_recall_route_and_defaults(self):
        response = self.hub.call("memory__search", {"query": "exact path"})
        self.assertTrue(response["ok"])
        self.assertEqual(json.loads(response["text"]), self.archive.result)
        self.assertEqual(self.archive.calls[-1], ("search", "exact path", None, 5))
        self.hub.call("memory__recall", {"chat_id": "chat", "branch_id": "branch", "start": 3})
        self.assertEqual(self.archive.calls[-1], ("recall", "chat", "branch", 3, 5, 0))
        self.hub.call("memory__recall", {"chat_id": "chat", "branch_id": "branch", "start": 3, "offset": 6000})
        self.assertEqual(self.archive.calls[-1], ("recall", "chat", "branch", 3, 5, 6000))

    def test_same_name_configured_server_is_preserved(self):
        external = MemoryProvider(Archive())
        external.kind = "stdio"
        self.hub.servers = {"memory": external}
        name = self.hub.register_builtin(MemoryProvider(self.archive))
        self.assertEqual(name, "memory_2")
        self.assertIs(self.hub.servers["memory"], external)
        self.assertEqual(len(self.hub.template_tools()), 4)
        result = self.hub.call("memory_2__search", {"query": "path"})
        self.assertTrue(result["ok"])
        self.assertEqual(external.archive.calls, [])

    def test_cleaned_name_collision_and_duplicate_registration(self):
        another = MemoryProvider(Archive(), name="memory!")
        self.hub.register_builtin(another)
        new = MemoryProvider(Archive(), name="memory?")
        self.assertEqual(self.hub.register_builtin(new), "memory__2")
        with self.assertRaises(ValueError):
            self.hub.register_builtin(new)

    def test_invalid_arguments_never_reach_archive(self):
        invalid = [("search", {"query": ""}), ("search", {"query": "x" * 201}),
                   ("search", {"query": "x", "limit": True}), ("search", {"query": "x", "limit": 11}),
                   ("search", {"query": "x", "path": "private.db"}),
                   ("recall", {"chat_id": "chat", "branch_id": "branch", "start": -1}),
                   ("recall", {"chat_id": "chat", "branch_id": "branch", "count": 1.0}),
                   ("recall", {"chat_id": "chat", "branch_id": "branch", "offset": -1}),
                   ("recall", {"chat_id": "chat", "branch_id": "branch", "offset": True}),
                   ("recall", {"chat_id": "chat", "branch_id": "branch", "offset": 1.0}),
                   ("recall", {"chat_id": "chat", "branch_id": "branch", "offset": 100000001}),
                   ("recall", {"chat_id": "chat"})]
        for tool, args in invalid:
            with self.subTest(tool=tool, arguments=args):
                response = self.hub.call("memory__" + tool, args)
                self.assertFalse(response["ok"])
        self.assertEqual(self.archive.calls, [])

    def test_imported_tool_calls_and_image_data_are_inert(self):
        self.archive.result = {"records": [{"index": 2, "message": {
            "role": "assistant", "tool_calls": [{"function": {"name": "shell", "arguments": "delete files"}}],
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,SECRET"}}]}}]}
        result = self.hub.call("memory__recall", {"chat_id": "chat", "branch_id": "branch"})
        self.assertTrue(result["ok"])
        data = json.loads(result["text"])
        self.assertEqual(data["records"][0]["message"]["tool_calls"][0]["function"]["name"], "shell")
        self.assertNotIn("SECRET", result["text"])
        self.assertEqual(len(self.archive.calls), 1)

    def test_cancel_before_and_after_lookup(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(McpCancelled):
            self.hub.call("memory__search", {"query": "path"}, cancel)
        self.assertEqual(self.archive.calls, [])
        cancel.clear()
        self.archive.after = cancel.set
        with self.assertRaises(McpCancelled):
            self.hub.call("memory__search", {"query": "path"}, cancel)

    def test_safe_database_and_argument_errors(self):
        for error in (sqlite3.OperationalError("C:/secret/private.db sensitive rows"),
                      ValueError("sensitive archive content")):
            self.archive.error = error
            result = self.hub.call("memory__search", {"query": "path"})
            self.assertFalse(result["ok"])
            self.assertNotIn("secret", result["text"])
            self.assertNotIn("sensitive", result["text"])

    def test_provider_character_limit_and_hub_character_limit(self):
        self.archive.result = {"hits": ["中" * MAX_RESULT_CHARS]}
        result = self.hub.call("memory__search", {"query": "path"})
        self.assertFalse(result["ok"])
        self.assertLess(len(result["text"]), 200)
        self.archive.result = {"hits": ["a" * 500]}
        self.hub.settings["max_result_chars"] = 100
        result = self.hub.call("memory__search", {"query": "path"})
        self.assertTrue(result["ok"])
        self.assertTrue(result["truncated"])
        self.assertGreater(result["chars"], 100)

    def test_real_archive_chinese_record_paging_through_hub(self):
        original = "中文分页必须保留全部内容和精确路径 C:/AI_Models/Strata。" * 1500
        with tempfile.TemporaryDirectory() as directory:
            archive = ChatArchive(pathlib.Path(directory) / "archive.sqlite3")
            try:
                archive.save({"id": "chat", "title": "Chinese archive", "activeBranchId": "branch",
                              "branches": [{"id": "branch", "messages": [
                                  {"role": "assistant", "text": original}]}]})
                hub = McpHub({})
                hub.register_builtin(MemoryProvider(archive))
                start, offset, pages = 0, 0, []
                for _ in range(50):
                    response = hub.call("memory__recall", {"chat_id": "chat", "branch_id": "branch",
                                                           "start": start, "offset": offset, "count": 1})
                    self.assertTrue(response["ok"], response["text"])
                    self.assertFalse(response["truncated"])
                    self.assertLessEqual(response["chars"], MAX_RESULT_CHARS)
                    if not pages:
                        self.assertGreater(len(response["text"].encode("utf-8")), MAX_RESULT_CHARS)
                    result = json.loads(response["text"])
                    pages.append(result["records"][0]["text"])
                    if result["next_start"] is None:
                        break
                    start, offset = result["next_start"], result["next_offset"] or 0
                else:
                    self.fail("recall paging did not terminate")
                self.assertGreater(len(pages), 1)
                restored = json.loads("".join(pages))
                self.assertEqual(restored["text"], original)
            finally:
                archive.close()


if __name__ == "__main__":
    unittest.main()
