"""Coder-L, 2026-10-25, raw samples: (H) declared parameters written as XML elements `<NAME>value</NAME>`, mixed with
`<parameter=NAME>` ones; (I) a call closed by a stray `</user>` instead of `</function>`; (J) an opener cut before its `>`
and nothing else - a call without arguments, dropped.  Streamed (token splits), non-streamed, and through the Anthropic
SSE emitter."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import OutputParser, parse_tool_call  # noqa: E402
from serve.test_live_tools import TOOLS, LiveTools  # noqa: E402

SCHEMAS = [{"name": t["name"], "parameters": t["input_schema"]} for t in TOOLS]
PE = "</parameter>"
OLD = ("/** Shipped defaults merged under the user's main config, resolved at call time. */\n"
       "async function defaultConfigReader(): Promise<ConfigReader> {\n"
       "  const { getConfigValueOp } = await import(\"../config/main-config-ops.js\");\n"
       "  return () => ({ release: { forgeTokenEnv: v } });\n}")
NEW = OLD.replace("getConfigValueOp", "effectiveMainConfig").replace("main-config-ops", "shipped-defaults")
PATH = "/x/src/builds/mirror-sync.ts"
H = ("<tool_call>\n<function=Edit>\n<parameter=replace_all>\nfalse\n" + PE + "\n<parameter=file_path>\n" + PATH + "\n" + PE +
     "\n<old_string>\n" + OLD + "\n</old_string>\n<new_string>\n" + NEW + "\n</new_string>\n</function>\n</tool_call>")
H_ARGS = {"replace_all": False, "file_path": PATH, "old_string": OLD, "new_string": NEW}
I = ("<tool_call>\n<function=Bash>\n<parameter=command>\ngrep -n x src/daemon/config.ts\n" + PE +
     "\n<parameter=description>\nInspect maskKey and loadConfig\n" + PE + "\n</user>\n</tool_call>")
I_ARGS = {"command": "grep -n x src/daemon/config.ts", "description": "Inspect maskKey and loadConfig"}
J = "<tool_call>\n<function=Bash>\n<parameter=command\n" + PE + "\n</function>\n</tool_call>"
CASES = {
    "H": (H, [("Edit", H_ARGS)]),
    # elements only, the first right after the function opener; closed by a stray tag; typed values
    "H2": ("<tool_call>\n<function=Bash>\n<command>\nls -la\n</command>\n<timeout>\n5000\n</timeout>\n</user>\n</tool_call>",
           [("Bash", {"command": "ls -la", "timeout": 5000})]),
    "H3": ("<tool_call>\n<function=Edit>\n<file_path>\n" + PATH + "\n</file_path>\n<old_string>\na <b>c</b>\n</old_string>\n"
           "<new_string>\nx\n</new_string>\n</function>\n</tool_call>",
           [("Edit", {"file_path": PATH, "old_string": "a <b>c</b>", "new_string": "x"})]),
    "I": (I, [("Bash", I_ARGS)]),
    "J": (J, []),
    "J2": (J + "\n" + I, [("Bash", I_ARGS)]),
}


def splits(text, how):
    return [text] if how == "whole" else list(text) if how == "char" else [text[i:i + 7] for i in range(0, len(text), 7)]


def stream(text, how, stream_tools=True):
    p = OutputParser(thinking=False, tools=SCHEMAS, stream_tools=stream_tools, recover=True)
    events = []
    for piece in splits(text, how):
        events += p.feed(piece)
    events += p.finish()
    pieces = {}
    for e in events:
        if e.kind == "tool_args":
            pieces.setdefault(id(e.call), []).append(e.text)
    return events, ["".join(v) for v in pieces.values()]


class XmlElementTests(unittest.TestCase):
    setUpClass = LiveTools.__dict__["setUpClass"]
    tearDownClass = LiveTools.__dict__["tearDownClass"]
    inputs = LiveTools.inputs

    def test_streamed(self):
        for key, (text, want) in CASES.items():
            for how in ("whole", "char", "seven"):
                events, args = stream(text, how)
                self.assertEqual([(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"], want, (key, how))
                self.assertEqual([json.loads(a) for a in args], [w[1] for w in want], (key, how))

    def test_not_streamed_tools(self):
        for key, (text, want) in CASES.items():
            for how in ("whole", "char", "seven"):
                events, _ = stream(text, how, False)
                self.assertEqual([(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"], want, (key, how))

    def test_parse_tool_call(self):
        for key in ("H", "H2", "H3", "I"):
            text, want = CASES[key]
            body = text.split("<tool_call>", 1)[1].rsplit("</tool_call>", 1)[0]
            schema = next(s for s in SCHEMAS if s["name"] == want[0][0])
            call = parse_tool_call(body, schema)
            self.assertEqual((call.name, call.arguments), want[0], key)

    def test_anthropic_sse(self):
        for key, (text, want) in CASES.items():
            self.assertEqual(self.inputs(text), want, key)

    def test_undeclared_tag_stays_in_the_value(self):
        text = ("<tool_call>\n<function=Bash>\n<parameter=command>\necho '<b>x</b>' </user>\n" + PE + "\n</function>\n</tool_call>")
        for how in ("whole", "char"):
            events, _ = stream(text, how)
            self.assertEqual([e.call.arguments for e in events if e.kind == "tool_call"],
                             [{"command": "echo '<b>x</b>' </user>"}], how)


if __name__ == "__main__":
    unittest.main()
