"""Tests for setup.py's free-space check when a model download resumes (#425): the shards already finished and the
.part files download() continues are not asked for again; the room for the pack, the margin and a fresh download stay
as they were.  setup.main() as tools/test_setup_golden.py runs it (every outside effect mocked, no downloads), with the
model folder filled with small stand-in files.

    python -m unittest tools.test_setup_resume
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from test_setup_golden import PROFILES, install  # noqa: E402

M = "IQ2_XS"
FAM = setup.FAMILIES["qwen"]
SHARDS = [FAM["file"].format(q=M, i=i) for i in (1, 2)]
DOWNLOAD = setup.MODELS[M]["download_gb"]
ARENA = setup.MODELS[M]["arena_gb"] + 1          # the low-RAM mode's experts.bin, written after the download


class Free(float):
    """free_gb()'s answer; keeps the space setup compared it with."""
    asked: list

    def __lt__(self, need):
        self.asked.append(need)
        return float(self) < need


class Resume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models = Path(self.tmp.name) / "models"
        self.folder = self.models / M
        self.folder.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def put(self, name, size, finished=True):
        p = self.folder / name
        p.write_bytes(b"\0" * size)
        if finished:
            setup.mark(p)
        return size

    def need(self, prof="32GB-2x24GB", free=900.0, extra=()):
        """-> (the space setup asked for, exit code, printed text); 32 GB of RAM: the low-RAM mode, as in #425."""
        f = Free(free)
        f.asked = []
        ram, found = PROFILES[prof]
        code, out, _, _ = install(ram, found, ["--family", "qwen", "--model", M, "--no-start",
                                               "--models-dir", str(self.models)],
                                  extra=[mock.patch.object(setup, "free_gb", lambda p: f), *extra])
        self.assertEqual(len(f.asked), 1, out[-3000:])
        return f.asked[0], code, out

    def test_fresh_download_asks_for_all_of_it(self):
        need, code, out = self.need()
        self.assertEqual(code, 0, out[-3000:])
        self.assertAlmostEqual(need, DOWNLOAD + 8 + ARENA)       # #425's "~112 GB"

    def test_fresh_download_with_enough_ram(self):
        need, code, out = self.need("96GB-1x16GB")              # no low-RAM mode: no experts.bin
        self.assertEqual(code, 0, out[-3000:])
        self.assertAlmostEqual(need, DOWNLOAD + 8)

    def test_resumed_download_asks_only_for_the_rest(self):
        got = self.put(SHARDS[0], 3_000_000) + self.put(SHARDS[1] + ".part", 1_250_000, finished=False)
        need, code, out = self.need()
        self.assertEqual(code, 0, out[-3000:])
        self.assertAlmostEqual(need, DOWNLOAD - got / 1e9 + 8 + ARENA, places=9)

    def test_too_little_space_says_what_is_already_here(self):
        self.put(SHARDS[0], 3_000_000)
        need, code, out = self.need(free=1.0)
        self.assertEqual(code, 1)
        self.assertIn(f"need ~{need:.0f} GB (not counting the 0.0 GB of {M} already downloaded)", out)

    def test_fresh_download_message_unchanged(self):
        need, code, out = self.need(free=1.0)
        self.assertEqual(code, 1)
        self.assertIn(f"need ~{need:.0f} GB\n", out)

    def test_an_unfinished_shard_is_not_counted(self):
        """A shard file without its mark that is not whole (copied in by hand, cut short) is downloaded again."""
        self.put(SHARDS[0], 3_000_000, finished=False)
        need, code, out = self.need(extra=[mock.patch.object(setup, "whole_shard", lambda s: False)])
        self.assertAlmostEqual(need, DOWNLOAD + 8 + ARENA)

    def test_whole_shards_without_marks_ask_for_no_download(self):
        """Both shards whole but never marked (#173: copied in by hand): marked, nothing left to download."""
        for name in SHARDS:
            self.put(name, 3_000_000, finished=False)
        need, code, out = self.need(extra=[mock.patch.object(setup, "whole_shard", lambda s: True)])
        self.assertEqual(code, 0, out[-3000:])
        self.assertAlmostEqual(need, 8 + ARENA)

    def test_finished_part_files_ask_for_no_download(self):
        """Every byte in .part files (stopped before the rename): no download space; the floor is 0, never less."""
        for name in SHARDS:
            self.put(name + ".part", 1_000_000, finished=False)
        with mock.patch.dict(setup.MODELS[M], download_gb=0.0015):
            need, code, out = self.need()
        self.assertAlmostEqual(need, 8 + ARENA)


class DownloadedBytes(unittest.TestCase):
    def test_counts_finished_shards_and_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            a, b, c = t / "a.gguf", t / "b.gguf", t / "c.gguf"
            a.write_bytes(b"x" * 10)
            setup.mark(a)
            (t / "b.gguf.part").write_bytes(b"x" * 7)
            c.write_bytes(b"x" * 5)                           # no mark: not counted
            self.assertEqual(setup.downloaded_bytes([a, b, c]), 17)
            self.assertEqual(setup.downloaded_bytes([t / "d.gguf"]), 0)

    def test_a_hard_linked_shard_counts_once_per_folder(self):
        """Shard 2 shared with another size (setup links it): it is this model's file, no new space; counted once."""
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            (t / "A").mkdir()
            (t / "B").mkdir()
            src, dst = t / "A" / "s2.gguf", t / "B" / "s2.gguf"
            src.write_bytes(b"x" * 9)
            setup.mark(src)
            try:
                os.link(src, dst)
            except OSError:
                self.skipTest("no hard links here")
            setup.mark(dst)
            self.assertEqual(setup.downloaded_bytes([t / "B" / "s1.gguf", dst]), 9)


class FinishedPart(unittest.TestCase):
    """A .part with every byte (stopped between the last byte and the rename): download() renames it instead of asking
    for a range past its end, which the server answers with 416 - retried 30 times, 10 s apart."""

    def test_a_complete_part_is_finished_without_a_request(self):
        seen = []

        class Head:
            headers = {"Content-Length": "11"}

        def urlopen(req, timeout=None):
            seen.append((req.get_method(), req.headers.get("Range")))
            if req.get_method() == "HEAD":
                return Head()
            raise urllib.error.HTTPError(req.full_url, 416, "Range Not Satisfiable", {}, None)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(setup.urllib.request, "urlopen", urlopen), \
                mock.patch.object(setup.time, "sleep", lambda s: None), contextlib.redirect_stdout(io.StringIO()):
            dst = Path(tmp) / "m.gguf"
            dst.with_name("m.gguf.part").write_bytes(b"model bytes")
            setup.download("https://example.com/m.gguf", dst)
            self.assertEqual(dst.read_bytes(), b"model bytes")
            self.assertTrue(setup.done(dst))
        self.assertEqual(seen, [("HEAD", None)])


if __name__ == "__main__":
    unittest.main()
