"""A cached image whose .sve file was deleted under the server (a temp cleaner) is encoded again instead of handing the
engine a missing file (the request then raised OSError and the client saw a dropped connection, ECONNRESET).

    python -m serve.test_vision_cache_missing
"""
import io
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import serve.server as server

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class CacheMissing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        v = server.Vision.__new__(server.Vision)
        v.dir = Path(self.tmp.name)
        v.lock = threading.Lock()
        v.cache = {}
        v.proc = SimpleNamespace(stdin=io.StringIO())

        def readline(timeout, what):                 # the encoder: writes the .sve it was asked for, answers OK
            name = v.proc.stdin.getvalue().splitlines()[-1].split()[2]
            (v.dir / name).write_bytes(b"emb")
            return "OK 256\n"
        v._readline = readline
        self.v = v

    def tearDown(self):
        self.tmp.cleanup()

    def encodes_sent(self):
        return sum(1 for line in self.v.proc.stdin.getvalue().splitlines() if line.startswith("ENC "))

    def test_present_file_is_a_cache_hit(self):
        first = self.v.encode(PNG)
        self.assertEqual(self.v.encode(PNG), first)
        self.assertEqual(self.encodes_sent(), 1)

    def test_deleted_file_is_encoded_again(self):
        path, n = self.v.encode(PNG)
        path.unlink()
        again, n2 = self.v.encode(PNG)
        self.assertTrue(again.is_file())
        self.assertEqual((again, n2), (path, n))
        self.assertEqual(self.encodes_sent(), 2)


if __name__ == "__main__":
    unittest.main()
