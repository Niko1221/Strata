"""The model's empty stub - a lone `<` line, closed by a parameter closer or left open before the next opener - after a
value or right after the function opener (Coder-L, 2026-10-25, raw samples): the stub is skipped and the value ends at
its own closer.  Streamed (several token splits, tool_args pieces reassembled as a client does), non-streamed, and
through the Anthropic SSE emitter."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import OutputParser, parse_tool_call  # noqa: E402
from serve.test_live_tools import TOOLS, LiveTools  # noqa: E402

PE = "</parameter>"
LT = "<"
SCHEMAS = [{"name": t["name"], "parameters": t["input_schema"]} for t in TOOLS]
GREP_PATH_ARGS = {"pattern": "effectiveMainConfig|loadConfigDefaults|computedDefaults", "path": "/x/src",
                  "output_mode": "content"}
CMD = 'ls src/builds/ && grep -rn "defaultConfigReader\\|readConfig" src/builds/*.ts src/cli/commands/build.ts | head -20'
CMD_B = 'npx vitest run scripts/dist-reachable.test.ts 2>&1 | grep -E "Assertion|expected|dist-reachable.test.ts:" | head -6'
WRITE_BODY = 'import { x } from "y";\n\nit("z", () => {});\n'

CASES = {
    "A": ("<tool_call>\n<function=Bash>\n<parameter=command>\n" + CMD + "\n" + PE + "\n" + LT + "\n" + PE +
          "\n</function>\n</tool_call>", [("Bash", {"command": CMD})]),
    "B": ("<tool_call>\n<function=Bash>\n<parameter=command>\n" + CMD_B + "\n" + PE + "\n" + LT + "\n" + PE +
          "\n<parameter=description>\nCapture failing assertion line with fixed script\n" + PE +
          "\n<parameter=timeout>\n120000\n" + PE + "\n</function>\n</tool_call>",
          [("Bash", {"command": CMD_B, "description": "Capture failing assertion line with fixed script",
                     "timeout": 120000})]),
    "C": ("<tool_call>\n<function=Grep>\n" + LT + "\n" + PE + "\n<parameter=pattern>\n" + GREP_PATH_ARGS["pattern"] +
          "\n" + PE + "\n<parameter=path>\n/x/src\n" + PE + "\n<parameter=output_mode>\ncontent\n" + PE + "\n" + LT +
          "\n" + PE + "\n</function>\n</tool_call>", [("Grep", GREP_PATH_ARGS)]),
    "D": ("<tool_call>\n<Grep>\n<parameter=pattern>\nexport function paiConfigFilePath\n" + PE +
          "\n<parameter=path>\n/x/src/daemon/config.ts\n" + PE + "\n" + LT + "\n" + PE +
          "\n<parameter=output_mode>\ncontent\n" + PE + "\n<parameter=-n>\ntrue\n" + PE + "\n<parameter=-A>\n10\n" + PE +
          "\n</function>\n</tool_call>",
          [("Grep", {"pattern": "export function paiConfigFilePath", "path": "/x/src/daemon/config.ts",
                     "output_mode": "content", "-n": True, "-A": 10})]),
    # a lone `<` line left open: the real opener follows it
    "E": ("<tool_call>\n<function=Write>\n<parameter=file_path>\n/t/probe.test.ts\n" + PE + "\n" + LT +
          "\n<parameter=content>\n" + WRITE_BODY + "\n" + PE + "\n</function>\n</tool_call>",
          [("Write", {"file_path": "/t/probe.test.ts", "content": WRITE_BODY})]),
    # a stray further closer after a value's own: the value ends at the first
    "F": ("<tool_call>\n<function=Bash>\n<parameter=command>\nps -eo lstart=\n" + PE + "\n<parameter=description>\nd\n" + PE +
          "\n<parameter=timeout>\n60000\n" + PE + "\n" + PE + "\n" + PE + "\n</function>\n</tool_call>",
          [("Bash", {"command": "ps -eo lstart=", "description": "d", "timeout": 60000})]),
    # an opener cut before its `>`, then a stub closer, then the real opener: the broken empty one is dropped
    "G": ("<tool_call>\n<function=Bash>\n<parameter=command\n" + PE + "\n<parameter=command>\ngrep -rln x scripts/\n" + PE +
          "\n" + PE + "\n<parameter=description>\nList\n" + PE + "\n</function>\n</tool_call>",
          [("Bash", {"command": "grep -rln x scripts/", "description": "List"})]),
    # a lone `</` line (live raw 96493fcfb19f): the same empty stub, not part of the value
    "H": ("<tool_call>\n<function=Bash>\n<parameter=command>\nnpx vitest run src/builds/mirror-sync.test.ts 2>&1 | tail -8\n"
          "</parameter>\n<parameter=description>\nRun mirror-sync tests after fix\n</parameter>\n<parameter=timeout>\n"
          "120000\n</parameter>\n</\n</parameter>\n</function>\n</tool_call>",
          [("Bash", {"command": "npx vitest run src/builds/mirror-sync.test.ts 2>&1 | tail -8",
                     "description": "Run mirror-sync tests after fix", "timeout": 120000})]),
    # a lone tag line, then a closer (live raw 446ef0463298): the same empty stub; the damaged `<=NAME>` opener is the next parameter
    "I": ("<tool_call>\n<function=Grep>\n<parameter=pattern>\npruneDist|prune-dist\n</parameter>\n<=output_mode>\n"
          "files_with_matches\n</parameter>\n</head_after>\n\n</parameter>\n</function>\n</tool_call>",
          [("Grep", {"pattern": "pruneDist|prune-dist", "output_mode": "files_with_matches"})]),
    # a doubled bare closer after a value
    "J": ("<tool_call>\n<function=Grep>\n<parameter=pattern>\nx\n</parameter>\n<parameter=output_mode>\n"
          "files_with_matches\n</parameter>\n</parameter>\n</function>\n</tool_call>",
          [("Grep", {"pattern": "x", "output_mode": "files_with_matches"})]),
    "K": ("<tool_call>\n<function=Grep>\n<parameter=pattern>\nx\n</parameter>\n<=output_mode>\n"
          "files_with_matches\n</parameter>\n</parameter>\n</function>\n</tool_call>",
          [("Grep", {"pattern": "x", "output_mode": "files_with_matches"})]),
    # a lone `>` line inside a number's value (live raw 97959, -A for -C: the test schema declares no -C): dropped before coercion
    "L": ("<tool_call>\n<function=Grep>\n<parameter=pattern>\nx\n</parameter>\n<=output_mode>\ncontent\n</parameter>\n"
          "<=-A>\n>\n2\n</parameter>\n</function>\n</tool_call>",
          [("Grep", {"pattern": "x", "output_mode": "content", "-A": 2})]),
}


def splits(text, how):
    if how == "whole":
        return [text]
    if how == "char":
        return list(text)
    pieces, at = [], 0
    while at < len(text):          # cut so that "</parameter>\n<" and "<\n</parameter>" each arrive as one token
        for tok in (PE + "\n</head_after>\n\n" + PE, PE + "\n" + LT + "/", PE + "\n" + LT, LT + "\n" + PE, PE + "\n" + PE):
            if text.startswith(tok, at):
                pieces.append(tok)
                at += len(tok)
                break
        else:
            pieces.append(text[at])
            at += 1
    return pieces if how == "tokens" else [text[i:i + 5] for i in range(0, len(text), 5)]


def stream(text, how):
    p = OutputParser(thinking=False, tools=SCHEMAS, stream_tools=True, recover=True)
    events = []
    for piece in splits(text, how):
        events += p.feed(piece)
    events += p.finish()
    calls = [e.call for e in events if e.kind == "tool_call"]
    pieces = {}
    for e in events:
        if e.kind == "tool_args":
            pieces.setdefault(id(e.call), []).append(e.text)
    return [(c.name, c.arguments) for c in calls], ["".join(v) for v in pieces.values()]


class StubTests(unittest.TestCase):
    setUpClass = LiveTools.__dict__["setUpClass"]
    tearDownClass = LiveTools.__dict__["tearDownClass"]
    inputs = LiveTools.inputs

    def test_streamed_splits(self):
        for key, (text, want) in CASES.items():
            for how in ("whole", "char", "tokens", "five"):
                calls, args = stream(text, how)
                self.assertEqual(calls, want, (key, how))
                self.assertEqual([json.loads(a) for a in args], [w[1] for w in want], (key, how))

    def test_non_streamed(self):
        for key, (text, want) in CASES.items():
            body = text.split("<tool_call>", 1)[1].rsplit("</tool_call>", 1)[0].replace("<Grep>", "<function=Grep>")
            schema = next(s for s in SCHEMAS if s["name"] == want[0][0])
            call = parse_tool_call(body, schema)
            self.assertEqual((call.name, call.arguments), want[0], key)

    def test_anthropic_sse(self):
        for key, (text, want) in CASES.items():
            self.assertEqual(self.inputs(text), want, key)

    def test_value_that_holds_a_lone_lt_line_stays(self):
        """A `<` line inside a value, followed by text, is the value's own."""
        text = ("<tool_call>\n<function=Write>\n<parameter=file_path>\n/t/a\n" + PE + "\n<parameter=content>\na\n<\nb\n" +
                PE + "\n</function>\n</tool_call>")
        for how in ("whole", "char"):
            self.assertEqual(stream(text, how)[0], [("Write", {"file_path": "/t/a", "content": "a\n<\nb"})], how)

    def test_value_that_holds_a_tag_line_then_text_stays(self):
        """A tag line followed by text before the next closer is the value's own (not a stub)."""
        text = ("<tool_call>\n<function=Write>\n<parameter=file_path>\n/t/a\n" + PE + "\n<parameter=content>\na\n" + PE +
                "\n</div>\nb\n" + PE + "\n</function>\n</tool_call>")
        for how in ("whole", "char"):
            self.assertEqual(stream(text, how)[0], [("Write", {"file_path": "/t/a", "content": "a\n" + PE + "\n</div>\nb"})], how)


if __name__ == "__main__":
    unittest.main()
