"""A parameter closer ends the value before any tag-shaped follower (Coder-L's damaged calls, 2026-10-24/25)."""
import unittest

from serve.frontend import OutputParser

LT, GT = "<", ">"
F, FE = LT + "function=", LT + "/function" + GT
P, PE = LT + "parameter=", LT + "/parameter" + GT
CS, CE = LT + "tool_call" + GT, LT + "/tool_call" + GT


def tool(name, **props):
    return {"name": name, "input_schema": {"type": "object", "properties": {k: {"type": v} for k, v in props.items()}}}


TOOLS = [tool("Grep", pattern="string", path="string", head_limit="integer"),
         tool("Read", file_path="string", offset="integer"),
         tool("Bash", command="string", description="string", timeout="number", run_in_background="boolean")]
SCHEMAS = [{"name": t["name"], "parameters": t["input_schema"]} for t in TOOLS]


def calls(text, stream, step):
    p = OutputParser(thinking=False, tools=SCHEMAS, stream_tools=stream, recover=True)
    evs = []
    for k in range(0, len(text), step):
        evs += p.feed(text[k:k + step])
    evs += p.finish()
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


def body(name, inner, close=True):
    return CS + "\n" + F + name + GT + "\n" + inner + ("\n" + FE + "\n" + CE if close else "")


CASES = {
    "damaged_opener_undeclared": (
        body("Grep", P + "pattern" + GT + "\nfoo\n" + PE + "\n" + P + "path" + GT + "\n/d/src\n" + PE + "\n"
             + LT + "=limit" + GT + "\n60\n" + PE),
        ("Grep", {"pattern": "foo", "path": "/d/src"})),
    "garbled_closer_follower": (
        body("Grep", P + "pattern" + GT + "\nfoo\n" + PE + "\n" + P + "path" + GT + "\n/d/src\n" + PE + "\n"
             + LT + "/" + LT + "novel" + GT),
        ("Grep", {"pattern": "foo", "path": "/d/src"})),
    "opener_without_gt": (
        CS + "\n" + F + "Grep" + GT + "\n" + P + "path" + GT + "\n/d/src/cli " + PE + " " + P + "pattern",
        ("Grep", {"path": "/d/src/cli "})),
    "closed_by_name": (
        body("Read", P + "file_path" + GT + "\n/d/f\n" + LT + "/file_path" + GT + "\n" + PE + "\n"
             + P + "offset" + GT + "\n5\n" + PE),
        ("Read", {"file_path": "/d/f", "offset": 5})),
    "name_runs_past_closer": (
        body("Bash", P + "command " + PE + "\necho hi\n" + PE + "\n" + P + "timeout" + GT + "\n2.5\n" + PE),
        ("Bash", {"command": "echo hi", "timeout": 2.5})),
    "name_null_dropped": (
        body("Bash", P + "command" + GT + "\nls\n" + PE + "\n" + P + " null " + PE + "\nx\n" + PE),
        ("Bash", {"command": "ls"})),
    "name_think_dropped": (
        body("Bash", P + "command" + GT + "\nls\n" + PE + "\n" + P + "command " + LT + "/think" + GT + "\nx\n" + PE),
        ("Bash", {"command": "x"})),
    "function_is_no_parameter": (
        body("Bash", P + "command" + GT + "\nls\n" + PE + "\n" + P + "function" + GT + "\nBash\n" + PE),
        ("Bash", {"command": "ls"})),
    "coerced_after_cut": (
        body("Bash", P + "command" + GT + "\nls\n" + PE + "\n" + P + "run_in_background " + GT + "\ntrue\n" + PE),
        ("Bash", {"command": "ls", "run_in_background": True})),
    "plain_text_after_closer_stays": (
        body("Bash", P + "command" + GT + "\necho a\n" + PE + " then text\n" + PE),
        ("Bash", {"command": "echo a" + "\n" + PE + " then text"})),
}


class Followers(unittest.TestCase):
    def test_cases(self):
        for label, (text, want) in CASES.items():
            for stream in (False, True):
                for step in (1, 5, len(text)):
                    with self.subTest(label, stream=stream, step=step):
                        self.assertEqual(calls(text, stream, step), [want])


if __name__ == "__main__":
    unittest.main()
