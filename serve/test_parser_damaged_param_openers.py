"""Coder-L raw outputs (STRATA_LOG_RAW, 2026-10-25): `<parameter=NAME: "VALUE"`, `<function=NAME>` for a parameter, an empty stub opener."""
import unittest

from serve.frontend import OutputParser

TOOLS = [
    {"name": "Grep", "parameters": {"properties": {
        "pattern": {"type": "string"}, "path": {"type": "string"}, "output_mode": {"type": "string"},
        "-n": {"type": "boolean"}}}},
    {"name": "Read", "parameters": {"properties": {
        "file_path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}}},
    {"name": "Bash", "parameters": {"properties": {"command": {"type": "string"}, "description": {"type": "string"}}}},
    {"name": "Write", "parameters": {"properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}}},
]
REST = ("<parameter=path>\nsrc/router/limits.ts\n</parameter>\n<parameter=output_mode>\ncontent\n</parameter>\n"
        "<parameter=-n>\ntrue\n</parameter>\n</function>\n</tool_call>")
SAMPLE_D = "<tool_call>\n<function=Grep>\n<parameter=pattern: \"const sh|function sh|import.*sh\"\n</parameter>\n" + REST
SAMPLE_D2 = "<tool_call>\n<function=Grep>\n<parameter=pattern: \"const sh|function sh|async function sh\n</parameter>\n" + REST
SAMPLE_C = ("<tool_call>\n<\n</think>\n\n<tool_call>\n<function=Read>\n<\n</think>\n\n<tool_call>\n"
            "<function=file_path>\nsrc/cli/impl/daemon.ts\n</parameter>\n<parameter=offset>\n120\n</parameter>\n"
            "<limit>\n140\n</parameter>\n</function>\n</tool_call>")
SAMPLE_E = ("<tool_call>\n<function=Bash>\n<parameter=wait_on>\n<parameter=command>\ngrep -n \"execFileSync\" src/workers/wait-on.ts\n"
            "</parameter>\n<parameter=description>\nCheck execFileSync uses in wait-on.ts\n</parameter>\n</function>\n</tool_call>")


def calls(text, step=None):
    p = OutputParser(thinking=False, tools=TOOLS, recover=True)
    evs = []
    for k in range(0, len(text), step or len(text)):
        evs += p.feed(text[k:k + (step or len(text))])
    evs += p.finish()
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


class DamagedParamOpenerTests(unittest.TestCase):
    def test_d_colon_value_opener(self):
        for text, pat in ((SAMPLE_D, "const sh|function sh|import.*sh"), (SAMPLE_D2, "const sh|function sh|async function sh")):
            for step in (None, 1, 3, 7):
                self.assertEqual(calls(text, step), [("Grep", {
                    "pattern": pat, "path": "src/router/limits.ts", "output_mode": "content", "-n": True})], step)

    def test_c_function_opener_for_parameter(self):
        for step in (None, 1, 3, 7):
            self.assertEqual(calls(SAMPLE_C, step), [("Read", {
                "file_path": "src/cli/impl/daemon.ts", "offset": 120, "limit": 140})], step)

    def test_e_empty_stub_opener(self):
        for step in (None, 1, 3, 7):
            self.assertEqual(calls(SAMPLE_E, step), [("Bash", {
                "command": 'grep -n "execFileSync" src/workers/wait-on.ts',
                "description": "Check execFileSync uses in wait-on.ts"})], step)

    def test_values_stay_byte_for_byte(self):
        content = "a\n<parameter=x: y\n<function=z>\n<function=content>\n<parameter=nope>\n<parameter=file_path: q\nend"
        text = ("<tool_call>\n<function=Write>\n<parameter=file_path>\na.txt\n</parameter>\n"
                "<parameter=content>\n%s\n</parameter>\n</function>\n</tool_call>" % content)
        for step in (None, 1, 5):
            self.assertEqual(calls(text, step), [("Write", {"file_path": "a.txt", "content": content})], step)


if __name__ == "__main__":
    unittest.main()
