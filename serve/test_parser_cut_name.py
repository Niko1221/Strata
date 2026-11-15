"""A call cut inside a parameter name keeps its complete parameters instead of raising."""
import unittest

from serve.frontend import FUNC_START, PARAM_END, PARAM_START, OutputParser, parse_tool_call

TOOLS = [{"name": "Bash", "parameters": {"properties": {"command": {"type": "string"}, "description": {"type": "string"}}}}]
CUT = FUNC_START + "Bash>\n" + PARAM_START + "command>\necho ok\n" + PARAM_END + "\n" + PARAM_START + "descr"


def run(text, stream, step):
    p = OutputParser(thinking=False, tools=TOOLS, stream_tools=stream, recover=True)
    evs = []
    for k in range(0, len(text), step):
        evs += p.feed(text[k:k + step])
    return evs + p.finish()


class CutName(unittest.TestCase):
    def test_whole_call(self):
        call = parse_tool_call(CUT, TOOLS[0])
        self.assertEqual(call.arguments, {"command": "echo ok"})

    def test_cut_everywhere_never_raises(self):
        for n in range(len(CUT) + 1):
            try:
                parse_tool_call(CUT[:n], TOOLS[0])
            except ValueError as e:
                self.assertIn("malformed", str(e))

    def test_streamed(self):
        text = "<tool_call>\n" + CUT
        for stream in (False, True):
            for step in (1, 3, 7):
                calls = [e.call for e in run(text, stream, step) if e.kind == "tool_call"]
                self.assertEqual([c.arguments for c in calls], [{"command": "echo ok"}], (stream, step))

    def test_cut_in_function_name(self):
        text = "<tool_call>\n" + FUNC_START + "Ba"
        for stream in (False, True):
            for step in (1, 3, 7):
                evs = run(text, stream, step)
                self.assertFalse([e for e in evs if e.kind == "tool_call"], (stream, step))


if __name__ == "__main__":
    unittest.main()
