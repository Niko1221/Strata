"""download() when the server does not say how long the file is in its HEAD answer (#1915: ModelScope's /resolve/
gives no Content-Length).  The length must then come from the GET answers themselves, and a stream that ends early
must be resumed, not renamed to the final name.  No network: a local HTTP server plays the host.

    python -m unittest tools.test_setup_download_length
"""
from __future__ import annotations

import hashlib
import http.client
import http.server
import os
import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

DATA = os.urandom(1_000_000)


class Host(http.server.BaseHTTPRequestHandler):
    """HEAD carries no Content-Length.  GET honours Range; `cut` bytes in, the first answer closes the connection
    cleanly (no reset).  `mode` picks how a GET says its length: "range" (Content-Range + Content-Length, as ModelScope
    and Hugging Face do), "chunked" (Transfer-Encoding: chunked) or "close" (nothing at all: the end is the close)."""
    protocol_version = "HTTP/1.1"
    mode = "range"
    cut = None
    always = False                      # cut every answer, not only the first

    def log_message(self, *a):
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Connection", "close")
        self.end_headers()

    def do_GET(self):
        start = 0
        m = None if self.mode == "norange" else re.match(r"bytes=(\d+)-", self.headers.get("Range", ""))
        if m:
            start = int(m.group(1))
        self.server.gets.append(start)
        if start >= len(DATA):
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{len(DATA)}")
            self.send_header("Connection", "close")
            self.end_headers()
            return
        body = b"" if self.mode == "empty" else DATA[start:]
        first = len(self.server.gets) == 1 or self.always
        if first and type(self).cut is not None:
            body = body[:type(self).cut]
        self.send_response(206 if start else 200)
        if self.mode in ("range", "norange", "wrongrange"):
            if start:
                first_byte = 0 if self.mode == "wrongrange" else start
                self.send_header("Content-Range", f"bytes {first_byte}-{len(DATA) - 1}/{len(DATA)}")
            self.send_header("Content-Length", str(len(DATA) - start))
        elif self.mode == "empty":
            self.send_header("Content-Length", "0")
        elif self.mode == "chunked":
            if start:
                self.send_header("Content-Range", f"bytes {start}-{len(DATA) - 1}/{len(DATA)}")
            self.send_header("Transfer-Encoding", "chunked")
        if self.mode == "close" and start:
            self.send_header("Content-Range", f"bytes {start}-{len(DATA) - 1}/{len(DATA)}")
        self.send_header("Connection", "close")
        self.end_headers()
        if self.mode == "chunked":
            self.wfile.write(b"%x\r\n" % len(body) + body + b"\r\n")
            if not (first and type(self).cut is not None):
                self.wfile.write(b"0\r\n\r\n")
        else:
            self.wfile.write(body)
        self.close_connection = True


class DownloadLength(unittest.TestCase):
    def setUp(self):
        self.warned = []
        for name in ("say", "ok"):
            p = mock.patch.object(setup, name, lambda *a, **k: None)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(setup, "warn", lambda *a, **k: self.warned.append(a))
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(setup.time, "sleep", lambda s: None)
        p.start()
        self.addCleanup(p.stop)
        env = mock.patch.dict(os.environ, {"STRATA_SOURCE": "huggingface"})
        env.start()
        self.addCleanup(env.stop)
        setup._sources.clear()
        self.addCleanup(setup._sources.clear)
        self.gets = []
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dst = Path(self.tmp.name) / "shard.gguf"

    def serve(self, mode, cut=None, always=False):
        handler = type("H", (Host,), {"mode": mode, "cut": cut, "always": always})
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        srv.gets = self.gets
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return "http://127.0.0.1:%d/shard.gguf" % srv.server_address[1]

    def finished(self):
        return self.dst.with_name(self.dst.name + ".done").exists()

    def test_early_clean_close_is_resumed(self):
        """The #1915 shape: no length in HEAD, the stream ends cleanly part-way.  The GET's own length says it is
        short, so the rest is asked for with a range and the file is whole before it gets its name."""
        setup.download(self.serve("range", cut=300_000), self.dst)
        self.assertEqual(self.gets, [0, 300_000])
        self.assertEqual(self.dst.read_bytes(), DATA)
        self.assertTrue(self.finished())

    def test_whole_file_without_head_length(self):
        setup.download(self.serve("range"), self.dst)
        self.assertEqual(self.dst.read_bytes(), DATA)
        self.assertTrue(self.finished())

    def test_short_file_never_gets_the_final_name(self):
        """Every answer ends early: setup stops, the bytes stay in the .part for the next run to resume."""
        with self.assertRaises(SystemExit):
            setup.download(self.serve("range", cut=10_000, always=True), self.dst)
        self.assertFalse(self.dst.exists())
        self.assertTrue(self.dst.with_name(self.dst.name + ".part").exists())
        self.assertFalse(self.finished())

    def test_a_server_that_ignores_the_range_restarts_from_zero(self):
        """A 200 to a ranged request is the whole file: the .part is rewritten from byte 0, not appended to."""
        part = self.dst.with_name(self.dst.name + ".part")
        part.write_bytes(DATA[:300_000])
        setup.download(self.serve("norange"), self.dst)
        self.assertEqual(self.dst.read_bytes(), DATA)
        self.assertTrue(self.finished())

    def test_chunked_cut_is_resumed(self):
        """Chunked: a stream that stops before the last chunk is an interrupted download, not a finished one."""
        setup.download(self.serve("chunked", cut=300_000), self.dst)
        self.assertEqual(self.gets, [0, 300_000])
        self.assertEqual(self.dst.read_bytes(), DATA)
        self.assertTrue(self.finished())

    def test_chunked_whole_is_finished(self):
        setup.download(self.serve("chunked"), self.dst)
        self.assertEqual(self.dst.read_bytes(), DATA)
        self.assertTrue(self.finished())

    def test_no_length_at_all_stays_partial(self):
        with self.assertRaises(SystemExit):
            setup.download(self.serve("close"), self.dst)
        self.assertFalse(self.dst.exists())
        self.assertEqual(self.dst.with_name(self.dst.name + ".part").read_bytes(), DATA)
        self.assertFalse(self.finished())

    def test_complete_partial_recovers_from_416(self):
        self.dst.with_name(self.dst.name + ".part").write_bytes(DATA)
        setup.download(self.serve("range"), self.dst)
        self.assertEqual(self.gets, [len(DATA)])
        self.assertEqual(self.dst.read_bytes(), DATA)
        self.assertTrue(self.finished())

    def test_oversized_partial_is_not_accepted_by_416(self):
        part = self.dst.with_name(self.dst.name + ".part")
        part.write_bytes(DATA + b"extra")
        with self.assertRaises(SystemExit):
            setup.download(self.serve("range"), self.dst)
        self.assertEqual(self.gets, [len(DATA) + 5])
        self.assertFalse(self.dst.exists())
        self.assertFalse(self.finished())
        self.assertEqual(part.read_bytes(), DATA + b"extra")

    def test_modelscope_size_resumes_clean_close_and_reuses_metadata_for_hash(self):
        # Neither HEAD nor GET has a length; the published ModelScope size must be used.
        url = self.serve("close", cut=300_000)
        digest = hashlib.sha256(DATA).hexdigest()
        with mock.patch.object(setup, "model_source", return_value="modelscope"), \
                mock.patch.object(setup, "ms_file", return_value=("repo", "file")), \
                mock.patch.object(setup, "ms_url", return_value=url), \
                mock.patch.object(setup, "reachable", return_value=True), \
                mock.patch.object(setup, "ms_meta", return_value=(len(DATA), digest)) as meta, \
                mock.patch.object(setup, "verify_sha256", wraps=setup.verify_sha256) as verify:
            setup.download(url, self.dst)
        meta.assert_called_once_with("repo", "file")
        verify.assert_called_once_with(self.dst, len(DATA), digest)
        self.assertEqual(self.gets, [0, 300_000])
        self.assertEqual(self.dst.read_bytes(), DATA)
        self.assertIn(digest, self.dst.with_name(self.dst.name + ".done").read_text())

    def test_wrong_range_is_rejected_before_writing(self):
        part = self.dst.with_name(self.dst.name + ".part")
        part.write_bytes(DATA[:300_000])
        with self.assertRaises(SystemExit):
            setup.download(self.serve("wrongrange"), self.dst)
        self.assertEqual(part.read_bytes(), DATA[:300_000])
        self.assertFalse(self.dst.exists())
        self.assertFalse(self.finished())

    def test_inconsistent_full_response_preserves_partial(self):
        part = self.dst.with_name(self.dst.name + ".part")
        part.write_bytes(DATA[:300_000])
        url = self.serve("norange")
        with mock.patch.object(setup, "model_source", return_value="modelscope"), \
                mock.patch.object(setup, "ms_file", return_value=("repo", "file")), \
                mock.patch.object(setup, "ms_url", return_value=url), \
                mock.patch.object(setup, "reachable", return_value=True), \
                mock.patch.object(setup, "ms_meta", return_value=(len(DATA) + 1, "")), \
                self.assertRaises(SystemExit):
            setup.download(url, self.dst)
        self.assertEqual(part.read_bytes(), DATA[:300_000])
        self.assertFalse(self.dst.exists())

    def test_explicit_zero_length_file_is_complete(self):
        setup.download(self.serve("empty"), self.dst)
        self.assertEqual(self.dst.read_bytes(), b"")
        self.assertTrue(self.finished())


class MsMetaCutShort(unittest.TestCase):
    def test_file_list_cut_short_is_unknown_not_a_crash(self):
        class R:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): raise http.client.IncompleteRead(b"{")
        setup._ms_files.pop("x/y", None)
        with mock.patch.object(setup.urllib.request, "urlopen", return_value=R()):
            self.assertIsNone(setup.ms_meta("x/y", "f.gguf"))


if __name__ == "__main__":
    unittest.main()
