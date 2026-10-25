"""A call whose body is `<arguments>` + a JSON object (Coder-L, 2026-10-25, raw sample) instead of `<parameter=...>` tags:
the object is the call's arguments (closed by `</arguments>`, the stray `</parameter>` the model wrote, or nothing); one
cut before it closes is dropped, never a call with partial strings.  Streamed (token splits), non-streamed, and through
the Anthropic SSE emitter."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import OutputParser, parse_tool_call  # noqa: E402
from serve.test_live_tools import TOOLS, LiveTools  # noqa: E402

SCHEMAS = [{"name": t["name"], "parameters": t["input_schema"]} for t in TOOLS]
OLD = ("/** Shipped defaults merged under the user's main config. */\nasync function defaultConfigReader() {\n"
       "  const { getConfigValueOp } = await import(\"../config/main-config-ops.js\");\n  return () => ({});\n}")
NEW = OLD.replace("getConfigValueOp", "effectiveMainConfig").replace("main-config-ops", "shipped-defaults")
ARGS = {"file_path": "/x/src/builds/mirror-sync.ts", "old_string": OLD, "new_string": NEW}
JSON = json.dumps(ARGS)
HEAD = "<tool_call>\n<function=Edit>\n<arguments>\n"
PE = "</parameter>"
CASES = {      # the sample's closers: the stray `</parameter>`; the proper `</arguments>`; none
    "stray": HEAD + JSON + "\n" + PE + "\n</function>\n</tool_call>",
    "closed": HEAD + JSON + "\n</arguments>\n</function>\n</tool_call>",
    "bare": HEAD + JSON + "\n</function>\n</tool_call>",
}
CUT = HEAD + JSON[:JSON.index("new_string") + 40]        # the sample verbatim: cut inside new_string
TYPED = ('<tool_call>\n<function=Grep>\n<arguments>{"pattern": "p", "-n": "true", "head_limit": 20, "-A": "3"}'
         '</arguments>\n</function>\n</tool_call>')


def splits(text, how):
    return [text] if how == "whole" else list(text) if how == "char" else [text[i:i + 7] for i in range(0, len(text), 7)]


def stream(text, how, finish="stop"):
    p = OutputParser(thinking=False, tools=SCHEMAS, stream_tools=True, recover=True)
    events = []
    for piece in splits(text, how):
        events += p.feed(piece)
    events += p.finish(finish)
    pieces = {}
    for e in events:
        if e.kind == "tool_args":
            pieces.setdefault(id(e.call), []).append(e.text)
    return events, ["".join(v) for v in pieces.values()]


class JsonArgsTests(unittest.TestCase):
    setUpClass = LiveTools.__dict__["setUpClass"]
    tearDownClass = LiveTools.__dict__["tearDownClass"]
    inputs = LiveTools.inputs

    def test_streamed(self):
        for key, text in CASES.items():
            for how in ("whole", "char", "seven"):
                events, args = stream(text, how)
                self.assertEqual([(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"],
                                 [("Edit", ARGS)], (key, how))
                self.assertEqual([json.loads(a) for a in args], [ARGS], (key, how))
                self.assertFalse([e for e in events if e.kind in ("content", "reasoning") and e.text.strip()], (key, how))

    def test_non_streamed(self):
        for key, text in CASES.items():
            body = text.split("<tool_call>", 1)[1].rsplit("</tool_call>", 1)[0]
            call = parse_tool_call(body, SCHEMAS[2])
            self.assertEqual((call.name, call.arguments), ("Edit", ARGS), key)
            p = OutputParser(thinking=False, tools=SCHEMAS, recover=True)
            events = p.feed(text) + p.finish()
            self.assertEqual([e.call.arguments for e in events if e.kind == "tool_call"], [ARGS], key)

    def test_types_are_coerced(self):
        want = {"pattern": "p", "-n": True, "head_limit": 20, "-A": 3}
        for how in ("whole", "char"):
            events, args = stream(TYPED, how)
            self.assertEqual([e.call.arguments for e in events if e.kind == "tool_call"], [want], how)
            self.assertEqual([json.loads(a) for a in args], [want], how)
        self.assertEqual(self.inputs(TYPED), [("Grep", want)])

    def test_cut_is_dropped(self):
        for finish in ("stop", "length"):
            for how in ("whole", "char", "seven"):
                events, args = stream(CUT, how, finish)
                self.assertEqual([e for e in events if e.kind in ("tool_call", "tool_start", "tool_args")], [], (finish, how))
                self.assertEqual(args, [])
        self.assertEqual(self.inputs(CUT), [])
        p = OutputParser(thinking=False, tools=SCHEMAS, recover=True)
        events = p.feed(CUT) + p.finish("length")
        self.assertEqual([e for e in events if e.kind == "tool_call"], [])

    def test_anthropic_sse(self):
        for key, text in CASES.items():
            self.assertEqual(self.inputs(text), [("Edit", ARGS)], key)

    def test_call_after_a_json_call(self):
        text = CASES["stray"] + "\n<tool_call>\n<function=Read>\n<parameter=file_path>\n/a\n" + PE + "\n</function>\n</tool_call>"
        for how in ("whole", "char"):
            events, _ = stream(text, how)
            self.assertEqual([(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"],
                             [("Edit", ARGS), ("Read", {"file_path": "/a"})], how)
        self.assertEqual(self.inputs(text), [("Edit", ARGS), ("Read", {"file_path": "/a"})])


if __name__ == "__main__":
    unittest.main()
