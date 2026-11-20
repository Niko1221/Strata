"""Tests for ab_suites: loop check, request type, frozen corpus.  No GPU, no network."""
import sys
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


if __name__ == "__main__":
    unittest.main()
