"""A temp cleaner (Windows' Storage Sense, systemd-tmpfiles) can remove the idle encoder's work dir while the
model is unloaded: every later start then failed on the encoder's cwd ("[WinError 267] The directory name is
invalid" on Windows) on every retry, and only a full server restart healed it.  The start now recreates the
dir (and drops the cache, whose files went with the dir).

    python -m serve.test_vision_dir
"""
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import serve.server as server


class Dir(unittest.TestCase):
    @staticmethod
    def _vision():
        v = server.Vision.__new__(server.Vision)
        v.spawn = (["strata-vision"], None, None)
        v.dir = Path(tempfile.mkdtemp(prefix="strata-vision-test-"))
        v.cache = {"k": (v.dir / "k.sve", 12)}          # the dir's files went with it
        v.proc, v.stopped = None, True
        return v

    @staticmethod
    def _fake_popen(seen):
        def popen(what, args, **kw):
            seen["cwd"] = kw.get("cwd")
            return SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO("READY 2560\n"),
                                   kill=lambda: None, wait=lambda timeout=None: 0, poll=lambda: None)
        return popen

    def test_start_recreates_a_removed_dir(self):
        v = self._vision()
        seen = {}
        shutil.rmtree(v.dir)                            # what the temp cleaner does
        with mock.patch.object(server, "popen", self._fake_popen(seen)):
            v._start()
        self.assertTrue(v.dir.exists())                 # a fresh dir took the removed one's place
        self.assertIs(seen["cwd"], v.dir)               # the encoder was started in it
        self.assertEqual(v.cache, {})                   # the vanished files are not served as cache hits

    def test_start_keeps_a_dir_that_is_there(self):
        v = self._vision()
        seen = {}
        with mock.patch.object(server, "popen", self._fake_popen(seen)):
            v._start()
        self.assertIs(seen["cwd"], v.dir)
        self.assertIn("k", v.cache)                     # an intact dir keeps its cache


if __name__ == "__main__":
    unittest.main()
