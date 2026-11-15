"""Coder-L's captured raw outputs: `<=Read>` for `<function=Read>` (the "function" token lost)."""
import unittest

from serve.frontend import OutputParser

TOOLS = [{"name": "Read", "parameters": {"properties": {"file_path": {"type": "string"}}}}]
# eval/work/easy/coder-l/rep1/worker.json, the turn that ended the run
CAPTURED = "<tool_call>\n<=Read>\n<parameter=file_path>\nsrc/router/router.ts\n</parameter>\n</function>"
# eval/work/hard/coder-l/rep1/worker.json: no tool name at all, nothing to recover
CAPTURED_HARD = "<tool_call>\n<<\n</parameter>\n</function>\n</tool_call>"
GOOD = "<tool_call>\n<function=Read>\n<parameter=file_path>\nsrc/router/router.ts\n</parameter>\n</function>\n</tool_call>"


def run(text, recover, step=None):
    p = OutputParser(thinking=False, tools=TOOLS, recover=recover)
    evs = []
    for k in range(0, len(text), step or len(text)):
        evs += p.feed(text[k:k + (step or len(text))])
    evs += p.finish()
    return evs


def calls(evs):
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


class DamagedOpenerTests(unittest.TestCase):
    def test_captured_parses_with_recovery_at_every_chunking(self):
        for step in (None, 1, 2, 3, 7):
            self.assertEqual(calls(run(CAPTURED + "\n</tool_call>", True, step)),
                             [("Read", {"file_path": "src/router/router.ts"})], step)

    def test_unclosed_wrapper_behaves_like_the_template_form(self):
        good = GOOD[:-len("\n</tool_call>")]
        self.assertEqual(calls(run(CAPTURED, True)), calls(run(good, True)))

    def test_template_form_unchanged(self):
        for step in (None, 1, 5):
            for recover in (False, True):
                self.assertEqual(calls(run(GOOD, recover, step)),
                                 [("Read", {"file_path": "src/router/router.ts"})])

    def test_undeclared_name_stays_text(self):
        evs = run("<tool_call>\n<=Nope>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>", True)
        self.assertEqual(calls(evs), [])

    def test_hopeless_capture_is_not_a_call(self):
        self.assertEqual(calls(run(CAPTURED_HARD, True)), [])

    def test_recovery_off_keeps_the_damaged_form_as_text(self):
        # upstream's loose opener (#3441) recovers it with recovery off too
        self.assertEqual(calls(run(CAPTURED + "\n</tool_call>", False)), [("Read", {"file_path": "src/router/router.ts"})])

    def test_prose_with_a_le_is_not_a_call(self):
        # "<=" outside a wrapper is ordinary text (comparisons, arrows), never a call.
        for text in ("if a <= b then ok", "```py\nx = a <= b\n```", "I'll use <=Read> here"):
            self.assertEqual(calls(run(text, True)), [])
        # a declared name written "<=NAME>" mid-sentence inside the wrapper is prose naming the format
        self.assertEqual(calls(run("<tool_call>\nI'll use <=Read> here\n</tool_call>", True)), [])
        # "<=NAME>" inside a code fence stays text even on its own line
        self.assertEqual(calls(run("```\n<tool_call>\n<=Read>\n</tool_call>\n```", True)), [])

    def test_damaged_opener_needs_the_name_whole_on_one_line(self):
        # a newline before ">", a space after "<=", or no ">" at all: not a call, held or released as text
        for text in ("<tool_call>\n<=Read\nmore\n</tool_call>",):
            self.assertEqual(calls(run(text, True)), [])


if __name__ == "__main__":
    unittest.main()


BARE = ("<tool_call>\n<Read>\n<parameter=file_path>\n/some/path/worker-config.ts\n</parameter>\n"
        "<parameter=offset>\n900\n</parameter>\n<parameter=limit>\n240\n</parameter>\n</function>\n</tool_call>")
TOOLS2 = [{"name": "Read", "parameters": {"properties": {"file_path": {"type": "string"},
                                                        "offset": {"type": "integer"}, "limit": {"type": "integer"}}}}]
BARE_ARGS = {"file_path": "/some/path/worker-config.ts", "offset": 900, "limit": 240}


def run2(text, recover, step=None):
    p = OutputParser(thinking=False, tools=TOOLS2, recover=recover)
    evs = []
    for k in range(0, len(text), step or len(text)):
        evs += p.feed(text[k:k + (step or len(text))])
    return evs + p.finish()


class BareNameOpenerTests(unittest.TestCase):
    def test_captured_parses_at_every_chunking(self):
        for step in (None, 1, 2, 3, 7):
            self.assertEqual(calls(run2(BARE, True, step)), [("Read", BARE_ARGS)], step)

    def test_name_closer(self):
        text = BARE.replace("</function>", "</Read>")
        for step in (None, 1, 4):
            self.assertEqual(calls(run2(text, True, step)), [("Read", BARE_ARGS)], step)

    def test_recovery_off_is_text(self):
        self.assertEqual(calls(run2(BARE, False)), [("Read", BARE_ARGS)])  # upstream's loose opener is always on

    def test_undeclared_name_stays_text(self):
        self.assertEqual(calls(run2(BARE.replace("<Read>", "<Foo>"), True)), [])

    def test_prose_with_name_tag_stays_text(self):
        for text in ("use <Read> to open it", "<Read>\n<parameter=file_path>\nx\n</parameter>\n</function>",
                     "<tool_call>\nI'll use <Read> here\n</tool_call>",
                     "```\n<tool_call>\n<Read>\n</tool_call>\n```"):
            self.assertEqual(calls(run2(text, True)), [], text)

    def test_template_form_unchanged(self):
        good = "<tool_call>\n<function=Read>\n<parameter=file_path>\nx\n</parameter>\n</function>\n</tool_call>"
        for recover in (False, True):
            self.assertEqual(calls(run2(good, recover)), [("Read", {"file_path": "x"})])


# Coder-L's live final turns (worker 20261024-134227-44180): `<=NAME>` with several parameters
LIVE_TOOLS = [{"name": n, "parameters": {"properties": {}}} for n in ("Read", "Bash", "Edit")]
LIVE_BASH = ("<tool_call>\n<=Bash>\n<parameter=command>\nsed -n '1,20p' src/utils/timeouts.ts\n</parameter>\n"
             "<parameter=description>\ntimeouts.ts head\n</parameter>\n</function>\n</tool_call>")
LIVE_EDIT = ("<tool_call>\n<=Edit>\n<parameter=replace_all>\nfalse\n</parameter>\n<parameter=file_path>\nsrc/a.ts\n"
             "</parameter>\n<parameter=old_string>\nf(a: number | string): x > y {\n  return \"q\";\n}\n</parameter>\n"
             "<parameter=new_string>\nasync f(a) {\n}\n</parameter>\n</invoke>\n\n</parameter>\n</function>\n</tool_call>")


class LiveTurnTests(unittest.TestCase):
    def feed(self, text, chunks, recover=True):
        p = OutputParser(thinking=False, tools=LIVE_TOOLS, recover=recover)
        evs = []
        for c in chunks:
            evs += p.feed(c)
        return evs + p.finish()

    def test_live_turns_every_chunking(self):
        import random
        rng = random.Random(8035)
        for text, name in ((LIVE_BASH, "Bash"), (LIVE_EDIT, "Edit")):
            splits = [[text[k:k + s] for k in range(0, len(text), s)] for s in (len(text), 1, 2, 3, 7)]
            for _ in range(200):
                cuts = sorted(rng.sample(range(1, len(text)), rng.randint(1, 40)))
                splits.append([text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)])])
            for chunks in splits:
                evs = self.feed(text, chunks)
                self.assertEqual([e.call.name for e in evs if e.kind == "tool_call"], [name])
                self.assertFalse([e for e in evs if e.kind == "content" and e.text.strip()])

    def test_recovery_is_opt_in(self):
        self.assertEqual(len(calls(self.feed(LIVE_BASH, [LIVE_BASH], recover=False))), 1)  # upstream's loose opener is always on
