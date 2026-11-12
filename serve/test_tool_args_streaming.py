"""A tool parameter that is an array or an object streams while the model writes it.

Every parameter the schema does not declare as a string used to be held in full until its
`</parameter>` and the tag after it had arrived, because the parser wanted the value whole to decode
it and re-send it as one piece.  A big `items` array or `taskProgress` list therefore left the
stream silent for as long as the model took to write it, and a client's idle timeout ended a healthy
request (the keep-alive for #1666 stopped the timeout; it did not give the progress back).

They now leave the stream at their JSON token boundaries.  What must still hold, and what these
tests check at every chunk boundary:

  * the pieces concatenated parse to the call's arguments, exactly as the whole answer does;
  * a call the output ended inside is still not a call, and its pieces still do not parse (#211);
  * no piece ends inside a string's escape sequence;
  * a value whose text does not open like JSON goes out whole once complete, as before.

    python -m unittest serve.test_tool_args_streaming -v
"""
import json
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, OutputParser  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402
from serve.test_server import CTX  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOOLS = [{"name": "readFiles", "parameters": {"properties": {
    "items": {"type": "array"}, "note": {"type": "string"}, "path": {}}}}]
ITEMS = [{"path": "a" * 40, "lines": [1, 2, 3]}, {"path": "b" * 40, "lines": []}]
NOTE = "n" * 60
# Written the way a model writes one: newlines and indentation between the tokens.
ARGS = {"items": ITEMS, "note": NOTE}
BODY = ("<tool_call>\n<function=readFiles>\n<parameter=items>\n" + json.dumps(ITEMS, indent=2) +
        "\n</parameter>\n<parameter=note>\n" + NOTE + "\n</parameter>\n</function>\n</tool_call>")
ITEMS_END = BODY.index("\n</parameter>")          # the array's closer; everything before it is the value


def run(text, step, tools=TOOLS):
    """The text fed in pieces of `step`, as a stream arrives."""
    p = OutputParser(thinking=False, tools=tools, stream_tools=True)
    evs = []
    for i in range(0, len(text), step):
        evs += p.feed(text[i:i + step])
    return evs + p.finish()


def streamed(evs):
    return [e.text for e in evs if e.kind == "tool_args"]


class AnArrayArgumentStreams(unittest.TestCase):
    def test_the_array_leaves_while_it_is_still_being_written(self):
        """The point of the change: most of the value is on the wire before its closer arrives, and
        no single piece carries it.  Measured in characters, not piece counts, because how many
        pieces there are depends on how the caller happens to chunk the stream."""
        half = BODY[:ITEMS_END]                   # the array is written; its closer has not arrived
        value = json.dumps(ITEMS)                 # what the value's text normalizes to
        for step in (1, 3, 7, 17):
            with self.subTest(step=step):
                p = OutputParser(thinking=False, tools=TOOLS, stream_tools=True)
                pieces = []
                for i in range(0, len(half), step):
                    pieces += streamed(p.feed(half[i:i + step]))
                got = "".join(pieces)
                self.assertTrue(got.startswith('{"items":['), got[:40])
                self.assertGreater(len(got), 10 + 3 * len(value) // 4,
                                   "the array did not leave while it was being written")
                self.assertLess(max(len(p) for p in pieces), len(value), "the value left in one piece")

    def test_the_pieces_concatenate_to_the_calls_arguments(self):
        whole = run(BODY, len(BODY))
        calls = [e.call for e in whole if e.kind == "tool_call"]
        self.assertEqual([c.arguments for c in calls], [ARGS])
        for step in (1, 2, 3, 7, 17, 4096):
            for recover in (False, True):
                with self.subTest(step=step, recover=recover):
                    p = OutputParser(thinking=False, tools=TOOLS, stream_tools=True, recover=recover)
                    evs = []
                    for i in range(0, len(BODY), step):
                        evs += p.feed(BODY[i:i + step])
                    evs += p.finish()
                    calls = [e.call for e in evs if e.kind == "tool_call"]
                    self.assertEqual([c.arguments for c in calls], [ARGS])
                    self.assertEqual(json.loads("".join(streamed(evs))), ARGS)

    def test_the_whitespace_between_tokens_does_not_leave_the_stream(self):
        """json.dumps would drop it, so the client must not see it either."""
        text = ("<tool_call>\n<function=readFiles>\n<parameter=items>\n[\n  1 ,\n\n  2\n]\n"
                "</parameter>\n</function>\n</tool_call>")
        for step in (1, 5, 4096):
            with self.subTest(step=step):
                pieces = "".join(streamed(run(text, step)))
                self.assertNotIn("\n", pieces)
                self.assertEqual(json.loads(pieces), {"items": [1, 2]})


class CutInsideAnArray(unittest.TestCase):
    """#211: a call the output ended inside must stay recognisable as unfinished."""

    def test_a_call_cut_inside_the_array_is_not_a_call(self):
        for cut in range(BODY.index("["), ITEMS_END, 7):
            for step in (1, 5, 4096):
                with self.subTest(cut=cut, step=step):
                    evs = run(BODY[:cut], step)
                    self.assertEqual([e for e in evs if e.kind == "tool_call"], [])
                    with self.assertRaises(ValueError):
                        json.loads("".join(streamed(evs)))

    def test_the_array_still_streams_before_the_cut(self):
        """A cut call is not an excuse to hold everything back until the end either."""
        cut = ITEMS_END - 20
        for step in (1, 7):
            with self.subTest(step=step):
                p = OutputParser(thinking=False, tools=TOOLS, stream_tools=True)
                pieces = []
                for i in range(0, cut, step):
                    pieces += streamed(p.feed(BODY[i:i + step]))
                self.assertGreater(len(pieces), 20, "nothing streamed until the output ended")


class AValueThatDoesNotOpenLikeJson(unittest.TestCase):
    """A bare path (a tool whose schema names no types) is not JSON: the parser could not stream
    text it would have to quote, so it goes out whole once complete, exactly as before."""

    TEXT = ("<tool_call>\n<function=readFiles>\n<parameter=path>\n" + "d" * 80 +
            "\n</parameter>\n</function>\n</tool_call>")

    def test_it_arrives_whole(self):
        for step in (1, 7, 4096):
            with self.subTest(step=step):
                evs = run(self.TEXT, step)
                calls = [e.call for e in evs if e.kind == "tool_call"]
                self.assertEqual([c.arguments for c in calls], [{"path": "d" * 80}])
                pieces = streamed(evs)
                self.assertIn('"' + "d" * 80 + '"', pieces, "the value did not arrive in one piece")


ESCAPED = [{"text": "line\nquote\" slash\\ unicodeé"}]


class EscapesAndUnicode(unittest.TestCase):
    """A split escape sequence must never leave the stream on its own."""

    TEXT = ("<tool_call>\n<function=readFiles>\n<parameter=items>\n" +
            json.dumps(ESCAPED, ensure_ascii=False) + "\n</parameter>\n</function>\n</tool_call>")

    @staticmethod
    def ends_inside_an_escape(text):
        """An odd number of trailing backslashes: the last one is still escaping something."""
        return (len(text) - len(text.rstrip("\\"))) % 2 == 1

    def test_no_piece_ends_inside_an_escape(self):
        for step in (1, 2, 3, 5, 7):
            with self.subTest(step=step):
                pieces = streamed(run(self.TEXT, step))
                for piece in pieces:
                    self.assertFalse(self.ends_inside_an_escape(piece), repr(piece))
                # the value sits inside <parameter=items>, so the pieces are the whole arguments
                self.assertEqual(json.loads("".join(pieces)), {"items": ESCAPED})


class OverHttp(unittest.TestCase):
    """The same thing a client sees: the array's argument arrives in many deltas."""

    def post(self, path, body):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, BODY, max_context=CTX), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}{path}",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode()
        finally:
            httpd.shutdown()
            httpd.server_close()
        return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]

    def test_openai_streams_the_array_in_many_deltas(self):
        # thinking off: with it on the call lands inside the reasoning and is rescued at the end of
        # the turn as one final tool_call, which deliberately does not stream (#804/#1058).
        events = self.post("/v1/chat/completions", {
            "model": "x", "max_tokens": 2000, "stream": True, "reasoning_effort": "none",
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{"type": "function", "function": {"name": "readFiles", "parameters": {
                "type": "object", "properties": {"items": {"type": "array"},
                                                 "note": {"type": "string"}}}}}]})
        pieces = [d["function"]["arguments"] for e in events for d in (e["choices"][0].get("delta") or {}).get("tool_calls") or []
                  if d.get("function", {}).get("arguments")]
        self.assertGreater(len(pieces), 20, "the array came as one delta")
        self.assertEqual(json.loads("".join(pieces)), ARGS)

    def test_anthropic_streams_the_array_in_many_deltas(self):
        events = self.post("/v1/messages", {
            "model": "x", "max_tokens": 2000, "stream": True, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{"name": "readFiles", "input_schema": {
                "type": "object", "properties": {"items": {"type": "array"},
                                                 "note": {"type": "string"}}}}],
            "anthropic_version": "vertex-2023-10-16"})
        pieces = [e["delta"]["partial_json"] for e in events
                  if e.get("type") == "content_block_delta" and e["delta"]["type"] == "input_json_delta"]
        self.assertGreater(len(pieces), 20, "the array came as one delta")
        self.assertEqual(json.loads("".join(pieces)), ARGS)


if __name__ == "__main__":
    unittest.main()
