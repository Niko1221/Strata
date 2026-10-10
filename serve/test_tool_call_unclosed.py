"""serve/test_tool_call_unclosed.py - a tool call whose `</tool_call>` the model left out ends at its own `</function>`.

Without its closer, the call used to run on to the next `</tool_call>` in the output: a second call right after it was
merged into the first (the first one's name with the second one's arguments), and a call followed by text kept
waiting for a closer that never came, so its arguments were lost.  Now what follows the call's own `</function>`
decides: `</tool_call>` closes it as before; another `<tool_call>` or text ends it there.  A `<function=` or
`<parameter=` right after it stays recover's batch (opt-in, unchanged).

    python -m unittest serve.test_tool_call_unclosed -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import OutputParser  # noqa: E402

TOOLS = [{"name": "read", "parameters": {"properties": {"path": {"type": "string"}}}},
         {"name": "write", "parameters": {"properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}]
READ = "<function=read>\n<parameter=path>\na.py\n</parameter>\n</function>"
WRITE = "<function=write>\n<parameter=path>\nb.py\n</parameter>\n<parameter=content>\nx = 1\n</parameter>\n</function>"


def run(text, stream_tools, recover, step):
    p = OutputParser(thinking=False, tools=TOOLS, stream_tools=stream_tools, recover=recover)
    evs = []
    for i in range(0, len(text), step):                # fed in small pieces, as a stream arrives
        evs += p.feed(text[i:i + step])
    evs += p.finish("stop")
    calls = [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]
    return calls, "".join(e.text for e in evs if e.kind == "content")


class UnclosedCall(unittest.TestCase):
    def check(self, text, want_calls, want_content=""):
        for stream_tools in (False, True):
            for recover in (False, True):
                for step in (1, 3, 7):
                    with self.subTest(stream_tools=stream_tools, recover=recover, step=step):
                        calls, content = run(text, stream_tools, recover, step)
                        self.assertEqual(calls, want_calls)
                        self.assertEqual(content.strip(), want_content)

    def test_closed_calls_are_unchanged(self):
        self.check("<tool_call>\n" + READ + "\n</tool_call>\n<tool_call>\n" + WRITE + "\n</tool_call>",
                   [("read", {"path": "a.py"}), ("write", {"path": "b.py", "content": "x = 1"})])

    def test_a_second_call_after_an_unclosed_one_is_its_own_call(self):
        self.check("<tool_call>\n" + READ + "\n<tool_call>\n" + WRITE + "\n</tool_call>",
                   [("read", {"path": "a.py"}), ("write", {"path": "b.py", "content": "x = 1"})])

    def test_text_after_an_unclosed_call_ends_it(self):
        self.check("<tool_call>\n" + READ + "\nDone.", [("read", {"path": "a.py"})], "Done.")

    def test_a_value_that_holds_the_tags_still_ends_at_the_calls_own_closer(self):
        value = "a </function> and a <tool_call> in a string"
        call = ("<tool_call>\n<function=write>\n<parameter=path>\nc.py\n</parameter>\n<parameter=content>\n" + value +
                "\n</parameter>\n</function>\n</tool_call>")
        self.check(call, [("write", {"path": "c.py", "content": value})])


if __name__ == "__main__":
    unittest.main()
