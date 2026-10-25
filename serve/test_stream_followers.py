"""Streamed tool arguments never carry text a parameter closer still holds (Coder-L, 2026-10-25: a Bash command
arrived as `...\\n</parameter>\\n<`).  The tokens go through the parser and the Anthropic SSE emitter; the
input_json_delta parts are reassembled and parsed."""
import json
import unittest

from serve.frontend import OutputParser
from serve.server import anthropic_events

LT, GT = "<", ">"
F, FE = LT + "function=", LT + "/function" + GT
P, PE = LT + "parameter=", LT + "/parameter" + GT
CS, CE = LT + "tool_call" + GT, LT + "/tool_call" + GT

TOOLS = [{"name": "Bash", "description": "", "input_schema": {"type": "object", "properties": {
              "command": {"type": "string"}, "description": {"type": "string"}, "timeout": {"type": "number"}}}},
         {"name": "Grep", "description": "", "input_schema": {"type": "object", "properties": {
              "pattern": {"type": "string"}, "path": {"type": "string"}, "head_limit": {"type": "integer"}}}}]
SCHEMAS = [{"name": t["name"], "parameters": t["input_schema"]} for t in TOOLS]


class Svc:
    """Just enough service for anthropic_events: run() feeds the tokens to a real parser."""

    def __init__(self, tokens, recover, ended=True):
        self.tokens, self.recover, self.ended = tokens, recover, ended

    def model_for(self, req):
        return "m"

    def run(self, ids, thinking, tools, max_new, req, cancel, force=None):
        parser = OutputParser(thinking=False, tools=SCHEMAS, stream_tools=True, recover=self.recover)
        for t in self.tokens:
            for ev in parser.feed(t):
                yield "event", ev
        for ev in parser.finish("stop" if self.ended else "length"):
            yield "event", ev
        yield "final", {"finish": "stop", "completion_tokens": len(self.tokens), "reused": 0}


def streamed_inputs(tokens, recover):
    """The tool_use blocks' inputs as a client reassembles them from the SSE events."""
    parts, blocks = {}, []
    for name, data in anthropic_events(Svc(tokens, recover), {}, [1], False, TOOLS, 100, None):
        if name == "content_block_start" and data["content_block"]["type"] == "tool_use":
            blocks.append(data["index"])
        elif name == "content_block_delta" and data["delta"]["type"] == "input_json_delta":
            parts.setdefault(data["index"], []).append(data["delta"]["partial_json"])
    return [json.loads("".join(parts.get(i, []))) for i in blocks]


def call(name, inner, close=True):
    return CS + "\n" + F + name + GT + "\n" + inner + ("\n" + FE + "\n" + CE if close else "")


def chars(text):
    return list(text)


class StreamFollowers(unittest.TestCase):
    def both(self, tokens, want):
        for recover in (False, True):
            self.assertEqual(streamed_inputs(tokens, recover), want, "recover=%s" % recover)

    def test_closer_and_lone_lt_in_one_token(self):
        tokens = [CS + "\n" + F + "Bash" + GT + "\n", P + "command" + GT + "\n", "npx tsc --noEmit 2>&1 | ", "head -20",
                  "\n", PE + "\n" + LT, P[1:] + "description" + GT, "\nshow errors\n", PE + "\n", FE + "\n" + CE]
        self.both(tokens, [{"command": "npx tsc --noEmit 2>&1 | head -20", "description": "show errors"}])

    def test_every_split(self):
        text = call("Bash", P + "command" + GT + "\nls -l\n" + PE + "\n" + P + "description" + GT + "\nlist\n" + PE)
        for step in range(1, 12):
            tokens = [text[k:k + step] for k in range(0, len(text), step)]
            self.both(tokens, [{"command": "ls -l", "description": "list"}])

    def test_damaged_samples_per_token(self):
        head = P + "pattern" + GT + "\nfoo\n" + PE + "\n" + P + "path" + GT + "\n/d/src\n" + PE + "\n"
        for tail in (LT + "=limit" + GT + "\n60\n" + PE, LT + "/" + LT + "novel" + GT, P + "pattern"):
            tokens = chars(call("Grep", head + tail, close=tail != P + "pattern"))
            if tail.endswith("pattern"):    # cut inside an opener: only recovery completes the call
                self.assertEqual(streamed_inputs(tokens, True), [{"pattern": "foo", "path": "/d/src"}])
            else:
                self.both(tokens, [{"pattern": "foo", "path": "/d/src"}])

    def test_output_ends_on_held_text(self):
        head = CS + "\n" + F + "Bash" + GT + "\n" + P + "command" + GT + "\nls\n"
        for held in (PE + "\n" + LT, PE, PE[:-3], LT + "/", LT):
            got = streamed_inputs([head, held], True)
            self.assertEqual(got, [{"command": "ls"}], repr(held))

    def test_call_closers_where_the_value_should_be(self):
        for text in (CS + "\n" + F + "Bash" + GT + "\n" + P + "command" + GT + "\n" + FE + "\n" + CE,
                     CS + "\n" + F + "Bash" + GT + "\n" + P + "command" + GT + FE + "\n" + CE):
            for step in (1, 5, len(text)):
                self.both([text[k:k + step] for k in range(0, len(text), step)], [{"command": ""}])

    def test_value_cut_at_a_markup_shaped_closer(self):
        text = (CS + "\n" + F + "Grep" + GT + "\n" + P + "path" + GT + "\n" + LT + "file" + GT + "\n" + PE + "\n" + LT)
        for step in (1, 5, len(text)):
            self.assertEqual(streamed_inputs([text[k:k + step] for k in range(0, len(text), step)], True),
                             [{"path": "<file>"}])

    def test_second_call_begun_inside_a_value(self):
        first = CS + "\n" + F + "Bash" + GT + "\n" + P + "command" + GT + "\n" + LT + "cmd" + GT + "\n"
        second = CS + "\n" + F + "Grep" + GT + "\n" + P + "path" + GT + "\n/x/src/b.ts"
        for tail in ("", "\n" + PE + "\n" + FE + "\n" + CE):
            text = first + second + tail
            for step in (1, 7, len(text)):
                self.assertEqual(streamed_inputs([text[k:k + step] for k in range(0, len(text), step)], True),
                                 [{"command": "<cmd>"}, {"path": "/x/src/b.ts"}], repr(tail))

    def test_closed_call_in_a_value_stays_in_it(self):
        inner = CS + "\n" + F + "Bash" + GT + "\n" + P + "command" + GT + "\nls\n" + PE + "\n" + FE + "\n" + CE
        text = call("Bash", P + "command" + GT + "\ndoc:\n" + inner + "\n" + PE)
        for step in (1, 7, len(text)):
            self.assertEqual(streamed_inputs([text[k:k + step] for k in range(0, len(text), step)], True),
                             [{"command": "doc:\n" + inner}])


if __name__ == "__main__":
    unittest.main()
