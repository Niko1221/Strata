"""Tests for ab_suites: loop check, request type, frozen corpus.  No GPU, no network."""
import json
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
        no_usage = {"choices": [{"message": {"content": " ".join(codes)}}]}
        self.assertEqual(req.check(no_usage), (False, "Cache-Treffer oder falsche Länge"))
        self.assertEqual(req.check(dict(no_usage, timings={"prompt_n": 16000})), (True, "4/4 Codes"))


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

    def test_model_context_missing_path_names_it(self):
        with self.assertRaises(SystemExit) as cm:
            ab_suites.model_context(Path("no/such/model.gguf"))
        self.assertIn("model.gguf", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
