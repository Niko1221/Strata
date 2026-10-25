"""The live /v1/messages path (stream=true) with Claude Code-shaped tools: the real HTTP handler, a scripted engine, the
config flags of the Coder-L run (tool_call_recovery).  The tool_use inputs a client reassembles from the SSE
input_json_delta pieces are exact and typed by the request's declared schema."""
import json
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {"tool_call_recovery": True, "reasoning_effort": "none"}   # r2/cl-mtp.json (MTP is the engine's)

TOOLS = [
    {"name": "Grep", "description": "search", "input_schema": {
        "type": "object", "$schema": "http://json-schema.org/draft-07/schema#", "additionalProperties": False,
        "required": ["pattern"], "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"},
            "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"]},
            "-n": {"type": "boolean"}, "-A": {"type": "number"}, "head_limit": {"type": "number"}}}},
    {"name": "Read", "description": "read", "input_schema": {
        "type": "object", "required": ["file_path"], "properties": {
            "file_path": {"type": "string"}, "limit": {"type": "number"}, "offset": {"type": "number"}}}},
    {"name": "Edit", "description": "edit", "input_schema": {
        "type": "object", "required": ["file_path", "old_string", "new_string"], "properties": {
            "file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"},
            "replace_all": {"type": "boolean"}}}},
    {"name": "Write", "description": "write", "input_schema": {
        "type": "object", "required": ["file_path", "content"], "properties": {
            "file_path": {"type": "string"}, "content": {"type": "string"}}}},
    {"name": "Bash", "description": "run", "input_schema": {
        "type": "object", "required": ["command"], "properties": {
            "command": {"type": "string"}, "timeout": {"type": "number"}, "description": {"type": "string"}}}}]

PE = "</parameter>"


def call(name, *params, close=True):
    body = "".join("<parameter=%s>\n%s\n%s\n" % (k, v, PE) for k, v in params)
    return "<tool_call>\n<function=%s>\n%s%s" % (name, body, "</function>\n</tool_call>" if close else "")


class LiveTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = MockEngine(cls.tok, "")
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.svc.tool_call_recovery = CONFIG["tool_call_recovery"]
        cls.svc.config_effort = CONFIG["reasoning_effort"]
        cls.httpd = serve(cls.svc, port=0)
        cls.base = "http://127.0.0.1:%d" % cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def inputs(self, script):
        """-> the (name, input) of every tool_use block, reassembled from the SSE events as a client does."""
        ids = self.tok.encode(script, parse_special=True) + self.tok.encode("<|im_end|>", parse_special=True)
        self.engine.script = ids
        self.engine.scripts = [ids]
        body = {"model": "m", "max_tokens": 4000, "stream": True, "tools": TOOLS,
                "messages": [{"role": "user", "content": "go"}]}
        req = urllib.request.Request(self.base + "/v1/messages?beta=true", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        blocks, parts = [], {}
        with urllib.request.urlopen(req, timeout=60) as r:
            name = None
            for raw in r.read().decode().splitlines():
                if raw.startswith("event: "):
                    name = raw[7:]
                elif raw.startswith("data: "):
                    d = json.loads(raw[6:])
                    if name == "content_block_start" and d["content_block"]["type"] == "tool_use":
                        blocks.append((d["index"], d["content_block"]["name"]))
                    elif name == "content_block_delta" and d["delta"]["type"] == "input_json_delta":
                        parts.setdefault(d["index"], []).append(d["delta"]["partial_json"])
        return [(n, json.loads("".join(parts.get(i, [])))) for i, n in blocks]

    def test_declared_closer_opens_the_parameter(self):
        text = ("<tool_call>\n<function=Grep>\n<parameter=path>\n/x/src\n" + PE + "\n</output_mode>\ncontent\n" + PE +
                "\n<parameter=pattern>\neffectiveMainConfig|x\n" + PE + "\n</function>\n</tool_call>")
        self.assertEqual(self.inputs(text), [("Grep", {"path": "/x/src", "output_mode": "content",
                                                       "pattern": "effectiveMainConfig|x"})])

    def test_types(self):
        got = self.inputs(call("Read", ("file_path", "/x/a.py"), ("limit", "2000")) +
                          call("Grep", ("pattern", "p"), ("-n", "true"), ("head_limit", "20")) +
                          call("Bash", ("command", "ls"), ("timeout", "120000")))
        self.assertEqual(got, [("Read", {"file_path": "/x/a.py", "limit": 2000}),
                               ("Grep", {"pattern": "p", "-n": True, "head_limit": 20}),
                               ("Bash", {"command": "ls", "timeout": 120000})])

    def test_anthropic_form_closers_end_the_value(self):
        """Coder-L (2026-10-25) closed a Read with `</invoke>` instead of `</function>`: `limit` came as a string
        holding the closers, and `allow_large` / a second `<invoke>` were merged into it."""
        head = call("Read", ("file_path", "/a/b.yaml"), ("offset", "410"), close=False)
        want = [("Read", {"file_path": "/a/b.yaml", "offset": 410, "limit": 25})]
        for tail in ("<parameter=limit>\n25\n" + PE + "\n</invoke>\n</tool_call>",
                     "<parameter=limit>\n25\n" + PE + "\n</invoke>",
                     "<parameter=limit>\n25\n" + PE + "\n</invoke>\n<parameter=allow_large>\ntrue\n" + PE +
                     "\n</function>\n</tool_call>",
                     "<parameter=limit>\n25\n" + PE + "\n<parameter name=\"allow_large\">false\n</function>"):
            self.assertEqual(self.inputs(head + tail), want, tail)

    def test_cut_call_is_not_a_bare_brace(self):
        for text in ("<tool_call>\n<function=Grep>\n", call("Grep", ("pattern", "p"), close=False)):
            for n, args in self.inputs(text):
                self.assertIsInstance(args, dict)

    def test_streamed_block_carries_the_parsed_arguments(self):
        """2026-10-25 04:00:57 UTC (raw log, path shortened): an Edit opened with a bare `<replace_all>` line reached
        the client as a tool_use with the input `{}`: the stream's reader stopped at the line, the parser read past it."""
        old = ("  const sameBuild = manifest && manifest.stamp === stamp;\n  if (process.env.PRUNE_DIST_DEBUG === \"1\") "
               "console.error(\"DBG sameBuild:\", sameBuild, \"stamp:\", stamp, \"manStamp:\", manifest && manifest.stamp);")
        new = "  const sameBuild = manifest && manifest.stamp === stamp;"
        path = "/x/prune-dist-rerun/scripts/prune-dist.mjs"
        text = ("The instrumented trace shows the fix works correctly: p3 (same build) is idempotent, p4 (new build) "
                "rotates. Let me remove the debug line and run the tests.\n\n<tool_call>\n<function=Edit>\n"
                "<replace_all>\nfalse\n" + PE + "\n<parameter=file_path>\n" + path + "\n" + PE +
                "\n<parameter=old_string>\n" + old + "\n" + PE + "\n<parameter=new_string>\n" + new + "\n" + PE +
                "\n</function>\n</tool_call>")
        got = self.inputs(text)
        self.assertEqual(got, [("Edit", {"file_path": path, "old_string": old, "new_string": new})])
        for _, args in got:
            self.assertTrue(args)

    def test_stub_alone_is_not_a_call(self):
        """2026-10-25 (raw log): `<function=Bash>` then a lone `<` and `</` was sent as a Bash call without arguments."""
        self.assertEqual(self.inputs("<tool_call>\n<function=Bash>\n<\n</"), [])

    def test_cut_before_a_parameter_opens_no_block(self):
        self.assertEqual(self.inputs("<tool_call>\n<function=Edit>\n<replace_all>\nfalse\n"), [])


if __name__ == "__main__":
    unittest.main()
