"""The vision cache must never hand a request a file that another request can delete.

An eviction used to unlink a cached `.sve` while another request still had the path: `prepare()`
raised FileNotFoundError on `read_bytes()`, the handler died before it wrote a response, and the
client saw "Provider request failed with HTTP 502" with an empty body (measured on a 140k-token
session resending 30+ frames per turn).

Two properties are pinned here, with no GPU (a fake encoder writes the `.sve`):
  - a hit whose file is gone is not a hit: the picture is encoded again;
  - files a request has not finished copying are held out of eviction, and are deleted at release
    if the cache dropped them in the meantime.
"""
import base64
import tempfile
import threading
import unittest
from pathlib import Path

from serve.server import Vision, write_temporary

# 1x1 PNG: enough for Vision.normalize to pass the bytes through untouched.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class FakeEncoder:
    """The strata-vision process without the GPU: writes the .sve file the ENC line names (its cwd)."""

    def __init__(self, workdir, tokens=7):
        self.dir = workdir
        self.tokens = tokens
        self.calls = 0
        self.stdin = self
        self._lines = []

    def write(self, line):
        self.calls += 1
        _, _, out = line.split()
        (self.dir / out).write_bytes(b"embeddings")
        self._lines.append("OK %d\n" % self.tokens)

    def flush(self):
        pass

    def readline(self):
        return self._lines.pop(0)


def vision():
    v = object.__new__(Vision)          # no encoder process, no GPU, no model
    v.dir = Path(tempfile.mkdtemp(prefix="test-vision-"))
    v.cache = {}
    v.in_use = set()
    v.lock = threading.Lock()
    v.stopped = False
    v.proc = FakeEncoder(v.dir)
    v._readline = lambda timeout, what: v.proc.readline()
    return v


def image(n):
    """A distinct image per n: the key is the hash of the bytes, so each one is its own entry."""
    return "data:image/png;base64," + base64.b64encode(PNG + bytes([n % 256, n // 256])).decode()


class VisionCacheTest(unittest.TestCase):
    def test_a_hit_returns_the_same_file_and_does_not_encode_again(self):
        v = vision()
        first, n1 = v.encode(image(0))
        second, n2 = v.encode(image(0))
        self.assertEqual(first, second)
        self.assertEqual((n1, n2), (7, 7))
        self.assertEqual(v.proc.calls, 1)

    def test_a_hit_whose_file_is_gone_encodes_again(self):
        """A dead path is not a hit: returning it is what raised FileNotFoundError in prepare()."""
        v = vision()
        first, _ = v.encode(image(0))
        first.unlink()
        again, n = v.encode(image(0))
        self.assertEqual(v.proc.calls, 2)
        self.assertEqual(again, first)
        self.assertTrue(again.exists())
        self.assertEqual(n, 7)

    def test_a_held_file_survives_another_requests_evictions(self):
        """The regression: the copy in prepare() happens after the turn, so a later request can evict it."""
        v = vision()
        held, _ = v.encode(image(0))
        v.hold([held])
        for i in range(1, 200):
            v.encode(image(i))
        self.assertTrue(held.exists(), "a file a request still needs was deleted")
        v.release([held])
        self.assertTrue(held.exists(), "a released file was deleted while it was still cached")
        for i in range(200, 400):
            v.encode(image(i))
        self.assertFalse(held.exists(), "an evicted file was left on disk")

    def test_a_request_keeps_its_own_files_while_it_encodes_many(self):
        v = vision()
        encoded = []
        for i in range(100):
            encoded.append(v.encode(image(i), keep=[p for p, _ in encoded]))
        for path, _ in encoded:
            self.assertTrue(path.exists())

    def test_the_map_stays_bounded(self):
        v = vision()
        for i in range(200):
            v.encode(image(i))
        self.assertLessEqual(len(v.cache), 64)

    def test_the_combined_file_holds_every_picture(self):
        v = vision()
        encoded = [v.encode(image(i)) for i in range(3)]
        v.hold([p for p, _ in encoded])
        combined = v.dir / "req-test.sve"
        try:
            write_temporary(combined, [p for p, _ in encoded])
        finally:
            v.release([p for p, _ in encoded])
        self.assertEqual(combined.read_bytes(), b"embeddings" * 3)


if __name__ == "__main__":
    unittest.main()
