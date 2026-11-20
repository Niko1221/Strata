"""Tests for ab_suites: loop check, request type, frozen corpus.  No GPU, no network."""
import json
import os
import re
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_suites  # noqa: E402
from ab_suites import base_check, git_grep, git_show, looping, need_corpus  # noqa: E402


def resp(content, finish="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}


def corpus_ok():
    try:
        need_corpus()
        return True
    except SystemExit:
        return False


class Loop(unittest.TestCase):
    def test_looping(self):
        self.assertTrue(looping("a b c d e f g h " * 5))
        self.assertFalse(looping(" ".join(f"w{i}" for i in range(300))))

    def test_base_check(self):
        self.assertEqual(base_check(resp("")), (False, "leer"))
        self.assertEqual(base_check({"choices": [{"message": {"content": None}}]}), (False, "leer"))
        self.assertEqual(base_check(resp("a b c d e f g h " * 5)), (False, "Schleife"))
        self.assertEqual(base_check(resp("Hello there.", "length")), (True, ""))


class Corpus(unittest.TestCase):
    def test_need_corpus_names_the_tag(self):
        with self.assertRaises(SystemExit) as cm:
            need_corpus(rev="v0.0.0-nope")
        self.assertIn("v0.0.0-nope", str(cm.exception))

    def test_need_corpus_without_git(self):
        with mock.patch.object(ab_suites.shutil, "which", side_effect=FileNotFoundError), \
                mock.patch.object(ab_suites.subprocess, "run", side_effect=FileNotFoundError):
            with self.assertRaises(SystemExit) as cm:
                need_corpus()
        self.assertIn("git", str(cm.exception))

    @unittest.skipUnless(corpus_ok(), "tag v0.2.12 not available")
    def test_git_show_and_grep_at_tag(self):
        self.assertIn("Strata", git_show("README.md"))
        out = git_grep("adapt_every")
        self.assertTrue(out)
        self.assertFalse(any(line.startswith("v0.2.12:") for line in out.splitlines()))
        with self.assertRaises(RuntimeError):
            git_show("no/such/file")


def gguf_file(path, kv):
    """Tensor-less GGUF v3 file: kv maps key to str or int (u32)."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv))
    for k, v in kv.items():
        out += s(k) + (struct.pack("<I", 8) + s(v) if isinstance(v, str) else struct.pack("<II", 4, v))
    Path(path).write_bytes(out)


class Ladder(unittest.TestCase):
    def test_ladder_steps(self):
        self.assertEqual(ab_suites.ladder_steps(262144), [8192, 16384, 32768, 65536, 131072, 261120])
        self.assertEqual(ab_suites.ladder_steps(32768), [8192, 16384, 31744])
        self.assertEqual(ab_suites.ladder_steps(8000), [])

    @unittest.skipUnless(corpus_ok(), "tag v0.2.12 not available")
    def test_ladder_same_bytes_twice(self):
        a = json.dumps([r.body for r in ab_suites.ladder_reqs(65536)])
        self.assertEqual(a, json.dumps([r.body for r in ab_suites.ladder_reqs(65536)]))

    @unittest.skipUnless(corpus_ok(), "tag v0.2.12 not available")
    def test_ladder_needles_placed_and_checked(self):
        req = next(r for r in ab_suites.ladder_reqs(65536) if r.tag == "ladder.16k")
        self.assertEqual(req.suite, "ladder")
        text = req.body["messages"][0]["content"]
        self.assertTrue(text.startswith("Document 16384\n"))
        self.assertEqual(req.body["max_tokens"], 256)
        self.assertEqual(req.body["seed"], ab_suites.SEED)
        body = text[:text.rindex(ab_suites.LADDER_QUESTION)]
        found = list(re.finditer(r"The access code for room (\d+) is ([A-Z]+-\d+)\.", body))
        self.assertEqual(len(found), 4)
        codes = [m.group(2) for m in found]
        for m, (lo, hi) in zip(found, [(.05, .15), (.30, .40), (.55, .65), (.80, .90)]):
            self.assertTrue(lo <= m.start() / len(body) <= hi, m.start() / len(body))

        def r(content, n):
            return {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": n}}
        self.assertEqual(req.check(r(" ".join(c.lower() for c in codes), 16384)), (True, "4/4 Codes"))
        self.assertEqual(req.check(r(" ".join(codes[:3]), 16384)), (False, "3/4 Codes"))
        self.assertEqual(req.check(r(" ".join(codes), 200)), (False, "Cache-Treffer oder falsche Länge"))
        none = {"choices": [{"message": {"content": " ".join(codes)}}]}
        self.assertEqual(req.check(none), (False, "Cache-Treffer oder falsche Länge"))
        # the server's usage.prompt_tokens is the full prompt; timings.prompt_n is what was read (cache hit = small)
        hit = dict(r(" ".join(codes), 16384), timings={"prompt_n": 200})
        self.assertEqual(req.check(hit), (False, "Cache-Treffer oder falsche Länge"))
        self.assertEqual(req.check(dict(hit, timings={"prompt_n": 0})), (False, "Cache-Treffer oder falsche Länge"))
        self.assertEqual(req.check(dict(hit, timings={"prompt_n": 16000})), (True, "4/4 Codes"))
        self.assertEqual(req.check(dict(none, usage={"prompt_tokens": 16384})), (True, "4/4 Codes"))


    def test_top_step_leaves_a_margin_the_others_do_not(self):
        fake = lambda n, *a: (("x" * 79 + "\n") * (n // 80 + 1))[:n]  # noqa: E731 - no git needed
        with mock.patch.object(ab_suites, "corpus_text", fake):
            reqs = ab_suites.ladder_reqs(65536)
        steps = ab_suites.ladder_steps(65536)

        def size(req):  # the characters of the corpus: without header, needle sentences and question
            c = req.body["messages"][0]["content"]
            body = c[c.index("\n") + 1:c.rindex("\n\n" + ab_suites.LADDER_QUESTION)]
            return len(re.sub(r"\nThe access code for room \d+ is [A-Z]+-\d+\.", "", body))
        sizes = [size(r) for r in reqs]
        self.assertEqual(sizes[:-1], [int(s * ab_suites.CHARS_PER_TOKEN) for s in steps[:-1]])
        self.assertEqual(sizes[-1], int(steps[-1] * 0.97 * ab_suites.CHARS_PER_TOKEN))

    def test_ladder_check_fails_a_cache_hit_inside_the_ten_percent(self):
        check = ab_suites.ladder_check(["A-11"], 16384)

        def r(prompt_n, tokens):
            return {"choices": [{"message": {"content": "a-11"}}], "timings": {"prompt_n": prompt_n}, "usage": {"prompt_tokens": tokens}}
        self.assertEqual(check(r(15000, 16384)), (True, "1/1 Codes"))
        self.assertEqual(check(r(15000, 17000)), (False, "Cache-Treffer oder falsche Länge"))  # 15000 < 0.9 * 17000
        self.assertEqual(check(r(15300, 17000)), (True, "1/1 Codes"))


class ModelContext(unittest.TestCase):
    def test_model_context_reads_gguf(self):
        with tempfile.TemporaryDirectory() as d:
            gguf_file(Path(d) / "m-00001-of-00002.gguf",
                      {"general.architecture": "qwen35moe", "qwen35moe.context_length": 262144})
            gguf_file(Path(d) / "m-00002-of-00002.gguf", {"general.architecture": "qwen35moe"})
            self.assertEqual(ab_suites.model_context(Path(d)), 262144)
            self.assertEqual(ab_suites.model_context(Path(d) / "m-00001-of-00002.gguf"), 262144)

    def test_model_context_missing_key_names_flag(self):
        with tempfile.TemporaryDirectory() as d:
            gguf_file(Path(d) / "m.gguf", {"general.architecture": "qwen35moe"})
            with self.assertRaises(SystemExit) as cm:
                ab_suites.model_context(Path(d))
            self.assertIn("--context", str(cm.exception))

    def test_model_context_unreadable_file_names_file_and_flag(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "m.gguf").write_bytes(b"not a gguf file")
            with self.assertRaises(SystemExit) as cm:
                ab_suites.model_context(Path(d) / "m.gguf")
        self.assertIn("m.gguf", str(cm.exception))
        self.assertIn("--context N", str(cm.exception))

    def test_model_context_missing_path_names_it(self):
        with self.assertRaises(SystemExit) as cm:
            ab_suites.model_context(Path("no/such/model.gguf"))
        self.assertIn("model.gguf", str(cm.exception))


def think_resp(content, reasoning="", finish="stop"):
    return {"choices": [{"message": {"content": content, "reasoning_content": reasoning}, "finish_reason": finish}]}


def by_tag(reqs):
    return {r.tag: r for r in reqs}


class Think(unittest.TestCase):
    def test_think_reqs_without_run_code_has_no_code_tasks(self):
        reqs = ab_suites.think_reqs(8192, False)
        self.assertEqual([r.tag for r in reqs], [f"think.math{i}" for i in range(1, 5)] + ["think.read1", "think.read2"])
        for r in reqs:
            self.assertEqual(r.body["reasoning_budget_tokens"], 6144)
            self.assertEqual(r.body["max_tokens"], 8192)
            self.assertIs(r.body["chat_template_kwargs"]["enable_thinking"], True)
            self.assertEqual((r.body["temperature"], r.body["seed"], r.body["reasoning_effort"]), (0, 7, "high"))
            self.assertTrue(r.body["messages"][0]["content"].endswith("End your reply with a line 'ANTWORT: <answer>'."))
        self.assertEqual([r.tag for r in ab_suites.think_reqs(8192, True)][-2:], ["think.code1", "think.code2"])

    def test_answer_only_in_reasoning_is_wrong(self):
        check = by_tag(ab_suites.think_reqs(8192, False))["think.math1"].check
        self.assertEqual(check(think_resp("", "ANTWORT: 334")), (False, "keine Antwort"))
        self.assertIsNone(ab_suites.answer_of(think_resp("so far", "ANTWORT: 334")))

    def test_empty_content_at_length_is_token_limit(self):
        check = by_tag(ab_suites.think_reqs(8192, False))["think.math1"].check
        self.assertEqual(check(think_resp("", "still thinking", "length")), (False, "Token-Grenze erreicht"))

    def test_math_answer_checked(self):
        reqs = by_tag(ab_suites.think_reqs(8192, False))
        for tag, right in zip(("think.math1", "think.math2", "think.math3", "think.math4"), ab_suites.MATH):
            check, n = reqs[tag].check, right[1]
            self.assertTrue(check(think_resp(f"... ANTWORT: 1 ... ANTWORT: {n}"))[0])
            self.assertTrue(check(think_resp(f"ANTWORT: **{n}**."))[0])
            self.assertFalse(check(think_resp(f"ANTWORT: {n + 1}"))[0])
            self.assertFalse(check(think_resp(f"ANTWORT: {n}\nANTWORT: 1"))[0])
            self.assertEqual(check(think_resp("no answer", finish="length")), (False, "Token-Grenze erreicht"))
            self.assertEqual(check(think_resp("no answer")), (False, "keine Antwort"))

    def test_answers_in_the_forms_the_model_writes(self):
        reqs = by_tag(ab_suites.think_reqs(8192, False))
        for text in (r"ANTWORT: \boxed{334}", "ANTWORT: $334$", "ANTWORT: x = 334", "ANTWORT: 334 (found by counting)"):
            self.assertTrue(reqs["think.math1"].check(think_resp(text))[0], text)
        self.assertTrue(reqs["think.math2"].check(think_resp("ANTWORT: **2,549**"))[0])
        self.assertFalse(reqs["think.math2"].check(think_resp("ANTWORT: 2,550"))[0])
        self.assertFalse(reqs["think.math1"].check(think_resp("ANTWORT: none"))[0])
        self.assertTrue(reqs["think.read2"].check(think_resp("ANTWORT: Line 3"))[0])
        self.assertFalse(reqs["think.read2"].check(think_resp("ANTWORT: Line 4"))[0])
        self.assertFalse(reqs["think.read2"].check(think_resp("ANTWORT: Line 13"))[0])
        for text in (r"ANTWORT: \boxed{9-5-0-10-28}", "ANTWORT: $9-5-0-10-28$", r"ANTWORT: $\boxed{9-5-0-10-28}$"):
            self.assertTrue(reqs["think.read1"].check(think_resp(text))[0], text)

    def test_read_answer_compared_as_text(self):
        reqs = by_tag(ab_suites.think_reqs(8192, False))
        self.assertTrue(reqs["think.read1"].check(think_resp("ANTWORT: `9-5-0-10-28`."))[0])
        self.assertFalse(reqs["think.read1"].check(think_resp("ANTWORT: 9-5-0-10-27"))[0])
        self.assertTrue(reqs["think.read2"].check(think_resp('ANTWORT:   "3" '))[0])
        self.assertFalse(reqs["think.read2"].check(think_resp("ANTWORT: 4"))[0])

    def test_tasks_have_working_reference_solutions(self):
        # the math answers and the print line, recomputed here
        self.assertEqual(sum(1 for n in range(1, 2001) if (n % 6 == 0 or n % 15 == 0) and n % 10), 334)
        self.assertEqual(pow(7, 2026, 1000), 649)
        ns = {}
        exec(ab_suites.READ_SNIPPET.replace("print(", "result = ("), ns)
        self.assertEqual(ns["result"], ab_suites.READ[0][1])
        self.assertEqual(ab_suites.BUG_SNIPPET.splitlines()[2].split(None, 1)[0], ab_suites.READ[1][1])

    def test_code_tasks_pass_reference_and_fail_stub(self):
        reqs = ab_suites.think_reqs(8192, True)[-2:]
        for req, (name, _, _, ref) in zip(reqs, ab_suites.CODE):
            self.assertIn(f"defines {name}.", req.body["messages"][0]["content"])
            ok, why = req.check(think_resp(f"first\n```python\nx = 1\n```\nfinal\n```python\n{ref}\n```"))
            self.assertTrue(ok, why)
            self.assertEqual(req.check(think_resp(f"```python\ndef {name}(*a):\n    return None\n```"))[0], False)
            self.assertEqual(req.check(think_resp("prose only")), (False, "kein Code"))
            self.assertEqual(req.check(think_resp("prose", finish="length")), (False, "Token-Grenze erreicht"))

    def test_no_process_without_run_code(self):
        with mock.patch("subprocess.run", side_effect=AssertionError("process started")), \
                mock.patch("subprocess.Popen", side_effect=AssertionError("process started")):
            for r in ab_suites.think_reqs(8192, False):
                for text in ("ANTWORT: 1", "```python\nimport os\nos.system('x')\n```", ""):
                    r.check(think_resp(text))

    def test_run_code_pass_fail_timeout(self):
        made = []
        real = tempfile.mkdtemp
        with mock.patch("tempfile.mkdtemp", side_effect=lambda: made.append(real()) or made[-1]):
            self.assertEqual(ab_suites.run_code("def f(x): return x+1", "from solution import f\nassert f(1)==2"),
                             (True, "Tests: 1 von 1"))
            self.assertEqual(ab_suites.run_code("def f(x): return x", "from solution import f\nassert f(1)==2\nassert f(0)==0"),
                             (False, "Tests: 1 von 2"))
            self.assertEqual(ab_suites.run_code("def f(x: return", "from solution import f\nassert f(1)==2"),
                             (False, "Tests: 0 von 1"))
            self.assertEqual(ab_suites.run_code("while True: pass", "from solution import f\nassert f(1)==2", timeout=2),
                             (False, "Zeitlimit"))
        self.assertEqual(len(made), 4)
        for d in made:
            self.assertFalse(Path(d).exists())

    def test_run_code_env_is_clean(self):
        tests = ("import os\nassert 'PATH' in os.environ\n"
                 "for k in ('STRATA_X', 'HOME', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA'):\n    assert k not in os.environ, k\n"
                 "assert not any(k.startswith('STRATA_') for k in os.environ)")
        with mock.patch.dict(os.environ, {"STRATA_X": "1", "HOME": "/h", "USERPROFILE": "u"}):
            ok, why = ab_suites.run_code("", tests)
        self.assertTrue(ok, why)


def tool_resp(name, args):
    call = {"id": "c1", "type": "function", "function": {"name": name, "arguments": args}}
    return {"choices": [{"message": {"content": None, "tool_calls": [call]}, "finish_reason": "tool_calls"}]}


class Agent(unittest.TestCase):
    @unittest.skipUnless(corpus_ok(), "tag v0.2.12 not available")
    def test_agent_turn_count_and_order(self):
        reqs, left = ab_suites.agent_reqs(262144)
        self.assertEqual((len(reqs), left), (23, []))
        self.assertEqual(reqs[0].tag, "agent.server-option.0")
        for a, b in zip(reqs, reqs[1:]):
            if a.tag.rsplit(".", 1)[0] == b.tag.rsplit(".", 1)[0]:
                self.assertEqual(b.body["messages"][:len(a.body["messages"])], a.body["messages"])
        self.assertEqual(reqs[0].body["messages"][-1]["role"], "tool")
        self.assertEqual((reqs[0].body["temperature"], reqs[0].body["seed"], reqs[0].body["max_tokens"]), (0, 7, 256))

    @unittest.skipUnless(corpus_ok(), "tag v0.2.12 not available")
    def test_agent_small_context_leaves_turns_out(self):
        reqs, left = ab_suites.agent_reqs(16384)
        self.assertTrue(left)
        self.assertEqual(len(reqs) + len(left), 23)
        self.assertTrue(all(t.startswith("agent.") for t in left))

    def test_agent_check(self):
        en = ab_suites.agent_check("en")
        fr = ab_suites.agent_check("fr")
        self.assertEqual(en(tool_resp("read_file", '{"path": "x"}')), (True, "Tool-Call read_file"))
        self.assertEqual(en(tool_resp("read_file", '{"pth": "x"}')), (False, "Tool-Call ungültig"))
        self.assertEqual(en(tool_resp("read_file", "not json")), (False, "Tool-Call ungültig"))
        self.assertFalse(en(tool_resp("nope", '{"path": "x"}'))[0])
        self.assertEqual(fr(resp("Je lis le fichier pour la suite dans une minute.")), (True, "Text"))
        self.assertEqual(fr(resp("I will read the file now.")), (False, "nicht Französisch"))
        self.assertEqual(en(resp("I will read the file now.")), (True, "Text"))
        self.assertEqual(en(resp("")), (False, "leer"))
        self.assertEqual(en(resp("a b c d e f g h " * 5)), (False, "Schleife"))

    @unittest.skipUnless(corpus_ok(), "tag v0.2.12 not available")
    def test_git_grep_errors_and_dash_pattern(self):
        self.assertEqual(git_grep("zzz_no_such_pattern_zzz"), "")  # exit 1 = no match
        with self.assertRaises(RuntimeError):
            git_grep("(")  # invalid regex: git exits 128
        git_grep("-x-")  # not read as an option
        self.assertTrue(git_grep("def .*ram"))


if __name__ == "__main__":
    unittest.main()
