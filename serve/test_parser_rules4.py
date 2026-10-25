"""Four Coder-L forms (live raw samples, 2026-10-25): a closer without its `>`, a bare opener as the first parameter,
a degenerate tail of closing tags after the last value, and whitespace after `<` / `</` in a declared tag.  Streamed
(token splits), non-streamed, and through the Anthropic SSE emitter."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import OutputParser, drop_tail, fix_tags, parse_tool_call  # noqa: E402
from serve.test_live_tools import LiveTools  # noqa: E402
from serve.test_parser_stub import SCHEMAS, stream  # noqa: E402

D = "/Users/x/trees/"
CASES = {
    # raw af04af55f0e2: `</parameter` without its `>`, an opener line follows
    "closer": ("<tool_call>\n<function=Grep>\n<parameter=-n>\ntrue\n</parameter\n<parameter=-C>\n15\n</parameter>\n"
               "<parameter=output_mode>\ncontent\n</parameter>\n<parameter=pattern>\nfunction workersYamlDataOf\n"
               "</parameter>\n</function>\n</tool_call>",
               [("Grep", {"-n": True, "-C": 15, "output_mode": "content", "pattern": "function workersYamlDataOf"})]),
    # raw 22618f756571: `<pattern>` right after the function opener
    "first": ("<tool_call>\n<function=Grep>\n<pattern>\ninterface OptDef|type OptDef\n</parameter>\n<parameter=path>\n"
              "/x/src/cli/define.ts\n</parameter>\n<parameter=output_mode>\ncontent\n</parameter>\n<parameter=-n>\ntrue\n"
              "</parameter>\n<parameter=-A>\n25\n</parameter>\n</function>\n</tool_call>",
              [("Grep", {"pattern": "interface OptDef|type OptDef", "path": "/x/src/cli/define.ts",
                         "output_mode": "content", "-n": True, "-A": 25})]),
    # raw ef55331d9303: the tail of closing tags, cut at the end
    "tail": ("<tool_call>\n<function=Grep>\n<parameter=pattern>\npruneDist|prune-dist\n</parameter>\n<=path>\n"
             "/x/prune-dist-rerun\n</parameter>\n<=output_mode>\nfiles_with_matches\n</parameter>\n</end_parameter>\n"
             "</invoke>\n</invoke>\n</invoke>\n</invoke>\n</invoke",
             [("Grep", {"pattern": "pruneDist|prune-dist", "path": "/x/prune-dist-rerun",
                        "output_mode": "files_with_matches"})]),
    # raw 51edfe8627d8: `<` and `</` and the name on the next line, the closer cut, elements for the rest
    "spaced": ("<tool_call>\n<function=Bash>\n<parameter=command>\nnpx vitest run src/cli/commands/worker-priority.test.ts"
               " 2>&1 | tail -15\n</parameter>\n<\ndescription>\nRun worker-priority tests again\n</\ndescription\n"
               "</parameter>\n<timeout>\n115000\n</timeout>\n</function>\n</tool_call>",
               [("Bash", {"command": "npx vitest run src/cli/commands/worker-priority.test.ts 2>&1 | tail -15",
                          "description": "Run worker-priority tests again", "timeout": 115000})]),
    "spaced_inline": ("<tool_call>\n<function=Bash>\n<parameter=command>\nls | tail -15\n</parameter>\n< description>\n"
                      "Run it\n</ description>\n</function>\n</tool_call>",
                      [("Bash", {"command": "ls | tail -15", "description": "Run it"})]),
    "tail_before_end": ("<tool_call>\n<function=Grep>\n<parameter=pattern>\nx\n</parameter>\n</end_parameter>\n</function>\n"
                        "</tool_call>", [("Grep", {"pattern": "x"})]),
}
TOKENS = (1, 3, 7, 1000)


def pieces(text, n):
    return [text[i:i + n] for i in range(0, len(text), n)]


class Rules4(unittest.TestCase):
    setUpClass = LiveTools.__dict__["setUpClass"]
    tearDownClass = LiveTools.__dict__["tearDownClass"]
    inputs = LiveTools.inputs

    def test_streamed(self):
        for key, (text, want) in CASES.items():
            for n in TOKENS:
                for tail in ("", "<|im_end|>"):
                    p = OutputParser(thinking=False, tools=SCHEMAS, stream_tools=True, recover=True)
                    events = []
                    for piece in pieces(text + (tail if key == "tail" else ""), n):
                        events += p.feed(piece)
                    events += p.finish()
                    calls = [(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"]
                    self.assertEqual(calls, want, (key, n, tail))
                    self.assertEqual([e for e in events if e.kind == "content" and e.text.strip()
                                      and "<|im_end|>" not in e.text], [], (key, n))
                    if key in ("closer", "first"):       # pieces are read before the rewrite is possible; the client gets the final call
                        args = "".join(e.text for e in events if e.kind == "tool_args")
                        self.assertEqual(json.loads(args), want[0][1], (key, n))

    def test_stub_stream_helper(self):
        for key, (text, want) in CASES.items():
            self.assertEqual(stream(text, "tokens")[0], want, key)

    def test_non_streamed(self):
        for key, (text, want) in CASES.items():
            body = text.split("<tool_call>", 1)[1].rsplit("</tool_call>", 1)[0]
            schema = next(s for s in SCHEMAS if s["name"] == want[0][0])
            call = parse_tool_call(body, schema)
            self.assertEqual((call.name, call.arguments), want[0], key)

    def test_anthropic_sse(self):
        for key, (text, want) in CASES.items():
            self.assertEqual(self.inputs(text), want, key)

    def test_helpers(self):
        self.assertEqual(drop_tail("<parameter=a>\nx\n</parameter>\n</end_parameter>\n</invoke"), "<parameter=a>\nx\n</parameter>")
        self.assertEqual(drop_tail("<parameter=a>\nx\n</parameter>\n</function>"), "<parameter=a>\nx\n</parameter>\n</function>")
        text = "<function=Bash>\n<parameter=command>\nx\n</parameter>\n< description>\nd\n</ description>\n</function>"
        self.assertEqual(fix_tags(text, {"Bash": SCHEMAS[0]}), text) if SCHEMAS[0]["name"] != "Bash" else None

    def test_value_text_stays(self):
        """A `<` / `</` line, a cut closer or closing tags inside a value, text following, are the value's own."""
        for body in ("a\n< b\nc", "a\n</parameter\ntext\nb", "a\n</div>\nb", "a\n< description is\nb"):
            text = ("<tool_call>\n<function=Write>\n<parameter=file_path>\n/t/a\n</parameter>\n<parameter=content>\n" + body +
                    "\n</parameter>\n</function>\n</tool_call>")
            for n in (1, 1000):
                p = OutputParser(thinking=False, tools=SCHEMAS, stream_tools=True, recover=True)
                events = []
                for piece in pieces(text, n):
                    events += p.feed(piece)
                events += p.finish()
                self.assertEqual([e.call.arguments for e in events if e.kind == "tool_call"],
                                 [{"file_path": "/t/a", "content": body}], (body, n))


if __name__ == "__main__":
    unittest.main()
