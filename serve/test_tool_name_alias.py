"""Codex MCP names that are not the declared namespace.member form.

Observed with Qwen3.8-Flash-Next through Codex CLI against Strata v0.2.2. The model
wrote <function=mcp__filesystem=write_file>, <function=mcp__filesystem_edit_file>,
and later a bare <function=write_file>. The declared tool was
mcp__filesystem.write_file. Stock main already accepts namespace__member. The
equals sign, the dropped underscore, and a unique bare member have to as well.
A name that matches nothing stays text (#3075). Two tools that share a member
are not guessed.

    python -m unittest serve.test_tool_name_alias -v
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import OutputParser  # noqa: E402
from serve.responses import Assembler, request_tools  # noqa: E402

REQ = {
    "tools": [{
        "type": "namespace",
        "name": "mcp__filesystem",
        "tools": [
            {"type": "function", "name": "write_file", "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"]}},
            {"type": "function", "name": "edit_file", "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "edits": {"type": "string"}},
                "required": ["path", "edits"]}},
            {"type": "function", "name": "list_allowed_directories", "parameters": {
                "type": "object", "properties": {}}},
        ],
    }],
}


def events(text, tools, thinking=False):
    parser = OutputParser(thinking=thinking, tools=tools, stream_tools=True)
    evs = parser.feed(text)
    evs += parser.finish("stop")
    return evs


def text_of(evs):
    return "".join(e.text or "" for e in evs if e.kind in ("content", "reasoning"))


def response_items(text, tools, names, req=REQ):
    asm = Assembler(req, "qwen", 10, names, encrypted=False)
    out = []
    parser_events = events(text, tools)
    for ev in parser_events:
        out += asm.feed(ev)
    out += asm.finish({"finish": "stop", "completion_tokens": 1, "prompt_tokens": 10})
    items = [ev["item"] for ev in out if ev.get("type") == "response.output_item.done"]
    return items, parser_events


class DriftedMcpNames(unittest.TestCase):
    def setUp(self):
        self.tools, self.names, _skipped = request_tools(REQ)

    def _call(self, text, tools=None, names=None):
        items, evs = response_items(text, tools or self.tools, names or self.names)
        calls = [i for i in items if i["type"] == "function_call"]
        return calls, evs

    def test_equals_write_is_the_namespace_tool(self):
        raw = (
            "<tool_call>\n<function=mcp__filesystem=write_file>\n<parameter=path>\n/tmp/x.py\n</parameter>\n"
            "<parameter=content>\nprint(\"not used\")\n</parameter>\n</function>\n</tool_call>"
        )
        calls, evs = self._call(raw)
        self.assertEqual(text_of(evs).strip(), "")
        self.assertEqual(len(calls), 1)
        self.assertEqual((calls[0]["namespace"], calls[0]["name"]), ("mcp__filesystem", "write_file"))
        args = json.loads(calls[0]["arguments"])
        self.assertEqual(args["path"], "/tmp/x.py")
        self.assertIn('print("not used")', args["content"])

    def test_single_underscore_edit_keeps_a_string_argument(self):
        raw = (
            "<tool_call>\n<function=mcp__filesystem_edit_file>\n<parameter=path>\n/tmp/a.txt\n</parameter>\n"
            "<parameter=edits>\nsay \"hi\"\n</parameter>\n</function>\n</tool_call>"
        )
        calls, evs = self._call(raw)
        self.assertEqual(text_of(evs).strip(), "")
        self.assertEqual((calls[0]["namespace"], calls[0]["name"]), ("mcp__filesystem", "edit_file"))
        args = json.loads(calls[0]["arguments"])
        self.assertEqual(args["path"], "/tmp/a.txt")
        self.assertIsInstance(args["edits"], str)
        self.assertIn('"hi"', args["edits"])

    def test_bare_write_file_is_the_one_namespace_tool(self):
        raw = (
            "<tool_call>\n<function=write_file>\n<parameter=path>\n/tmp/patch-tencut.py\n</parameter>\n"
            "<parameter=content>\nprint(\"ten\")\n</parameter>\n</function>\n</tool_call>"
        )
        calls, evs = self._call(raw)
        self.assertEqual(text_of(evs).strip(), "")
        self.assertEqual((calls[0]["namespace"], calls[0]["name"]), ("mcp__filesystem", "write_file"))
        args = json.loads(calls[0]["arguments"])
        self.assertEqual(args["path"], "/tmp/patch-tencut.py")
        self.assertEqual(args["content"], 'print("ten")')

    def test_bare_list_allowed_directories_is_a_call(self):
        raw = "<tool_call>\n<function=list_allowed_directories>\n</function>\n</tool_call>"
        calls, evs = self._call(raw)
        self.assertEqual(text_of(evs).strip(), "")
        self.assertEqual((calls[0]["namespace"], calls[0]["name"]),
                         ("mcp__filesystem", "list_allowed_directories"))
        self.assertEqual(json.loads(calls[0]["arguments"]), {})

    def test_double_underscore_still_maps(self):
        raw = (
            "<tool_call>\n<function=mcp__filesystem__write_file>\n<parameter=path>\n/tmp/z\n</parameter>\n"
            "<parameter=content>\nok\n</parameter>\n</function>\n</tool_call>"
        )
        calls, _evs = self._call(raw)
        self.assertEqual((calls[0]["namespace"], calls[0]["name"]), ("mcp__filesystem", "write_file"))
        self.assertEqual(json.loads(calls[0]["arguments"])["content"], "ok")

    def test_equals_inside_thinking_is_still_a_call(self):
        raw = (
            "<tool_call>\n<function=mcp__filesystem=write_file>\n<parameter=path>\n/tmp/x.py\n</parameter>\n"
            "<parameter=content>\nok\n</parameter>\n</function>\n</tool_call>"
        )
        evs = events(raw + "</think>\n", self.tools, thinking=True)
        found = [e for e in evs if e.kind == "tool_call"]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].call.arguments["path"], "/tmp/x.py")

    def test_an_unknown_name_stays_text(self):
        raw = "<tool_call>\n<function=mcp__filesystem=not_a_tool>\n</function>\n</tool_call>"
        calls, evs = self._call(raw)
        self.assertEqual(calls, [])
        self.assertIn("not_a_tool", text_of(evs))

    def test_two_namespaces_sharing_a_member_stay_text(self):
        req = {"tools": [
            {"type": "namespace", "name": "mcp__filesystem", "tools": [
                {"type": "function", "name": "write_file", "parameters": {
                    "type": "object", "properties": {"path": {"type": "string"}}}}]},
            {"type": "namespace", "name": "mcp__other", "tools": [
                {"type": "function", "name": "write_file", "parameters": {
                    "type": "object", "properties": {"path": {"type": "string"}}}}]},
        ]}
        tools, names, _ = request_tools(req)
        raw = "<tool_call>\n<function=write_file>\n<parameter=path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>"
        calls, evs = self._call(raw, tools, names)
        self.assertEqual(calls, [])
        self.assertIn("write_file", text_of(evs))

    def test_a_flat_tool_with_the_member_name_wins(self):
        req = {"tools": [
            {"type": "function", "name": "write_file", "parameters": {
                "type": "object", "properties": {"path": {"type": "string"}}}},
            {"type": "namespace", "name": "mcp__filesystem", "tools": [
                {"type": "function", "name": "write_file", "parameters": {
                    "type": "object", "properties": {"path": {"type": "string"},
                                                     "content": {"type": "string"}}}}]},
        ]}
        tools, names, _ = request_tools(req)
        self.assertEqual(names["write_file"], (None, "write_file", "function"))
        raw = "<tool_call>\n<function=write_file>\n<parameter=path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>"
        calls, evs = self._call(raw, tools, names)
        self.assertEqual(text_of(evs).strip(), "")
        self.assertEqual(calls[0]["name"], "write_file")
        self.assertNotIn("namespace", calls[0])


if __name__ == "__main__":
    unittest.main()
