"""Conversation-cache log backlogs, using local files and a CPU-only HTTP server."""
import json
import tempfile
import unittest
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, ConvCacheLog, MockEngine, Service, serve


class ConversationCacheBacklog(unittest.TestCase):
    PARK = b"strata serve: conversation cache: parked 5000 tokens in 12.0 ms; parked=1 bytes=104857600 evictions=0\n"
    RESTORE = b"strata serve: conversation cache: restored 5000 tokens (exact) in 30.0 ms; parked=0 bytes=0\n"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = Path(self.tmp.name) / "strata.log"
        self.log.write_bytes(b"")
        self.cache = ConvCacheLog()

    def poll(self):
        return self.cache.poll(str(self.log), 0)

    def append(self, data):
        with self.log.open("ab") as f:
            f.write(data)

    def test_backlog_larger_than_one_mib_keeps_early_and_late_events(self):
        self.append(self.PARK)
        self.assertEqual(self.poll()["parks"], 1)
        noise = b"strata serve: ordinary log output\n" * (self.cache.READ_MAX // 32 + 1)
        self.append(self.RESTORE + noise + self.PARK + noise + self.RESTORE)
        state = self.poll()
        self.assertEqual((state["parked"], state["bytes"], state["parks"], state["restores"]), (0, 0, 2, 2))
        self.assertEqual(self.cache.pos, self.log.stat().st_size)
        self.assertEqual(self.poll(), state)

    def test_events_crossing_small_chunks_and_long_lines(self):
        long_park = self.PARK.replace(b" in 12.0 ms;", b" in 12.0 ms; " + b"x" * 4096 + b";")
        self.log.write_bytes("earlier run: 中文\n".encode() + long_park)
        start = self.log.stat().st_size
        self.append("本次运行\n".encode() + long_park + self.RESTORE)
        for limit in (7, 64, 257):
            with self.subTest(chunk_size=limit):
                cache = ConvCacheLog()
                cache.READ_MAX = limit
                state = cache.poll(str(self.log), start)
                self.assertEqual((state["parks"], state["restores"], state["parked"]), (1, 1, 0))
                self.assertEqual(cache.pos, self.log.stat().st_size)

    def test_unterminated_event_is_read_once_after_it_is_completed(self):
        self.cache.READ_MAX = 32
        noise = b"ordinary line\n" * 20
        cut = len(self.PARK) - 10
        self.append(noise + self.PARK[:cut])
        self.assertEqual(self.poll()["parks"], 0)
        self.assertEqual(self.cache.pos, len(noise))
        self.assertEqual(self.poll()["parks"], 0)
        self.append(self.PARK[cut:] + self.RESTORE)
        state = self.poll()
        self.assertEqual((state["parks"], state["restores"]), (1, 1))
        self.assertEqual(self.poll(), state)

    def test_reads_are_bounded_and_stop_at_the_initial_file_size(self):
        self.cache.READ_MAX = 32
        self.append(self.PARK)
        real_open, sizes = open, []

        @contextmanager
        def growing_open(path, mode):
            with real_open(path, mode) as f:
                def read(size):
                    sizes.append(size)
                    data = f.read(size)
                    if len(sizes) == 1:
                        self.append(self.RESTORE)
                    return data
                wrapped = mock.Mock(wraps=f)
                wrapped.read.side_effect = read
                yield wrapped

        with mock.patch("serve.server.open", side_effect=growing_open, create=True):
            state = self.poll()
        self.assertEqual((state["parks"], state["restores"]), (1, 0))
        self.assertEqual(self.cache.pos, len(self.PARK))
        self.assertGreater(len(sizes), 1)
        self.assertTrue(all(0 < size <= self.cache.READ_MAX for size in sizes))
        self.assertEqual(self.poll()["restores"], 1)

    def test_read_error_preserves_completed_events_and_retries_the_partial_line(self):
        self.cache.READ_MAX = len(self.PARK) + 5
        self.append(self.PARK + self.RESTORE)
        real_open = open

        @contextmanager
        def failing_open(path, mode):
            with real_open(path, mode) as f:
                calls = 0
                def read(size):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise OSError("temporary read failure")
                    return f.read(size)
                wrapped = mock.Mock(wraps=f)
                wrapped.read.side_effect = read
                yield wrapped

        with mock.patch("serve.server.open", side_effect=failing_open, create=True):
            state = self.poll()
        self.assertEqual((state["parks"], state["restores"]), (1, 0))
        self.assertEqual(self.cache.pos, len(self.PARK))
        state = self.poll()
        self.assertEqual((state["parks"], state["restores"]), (1, 1))
        self.assertEqual(self.poll(), state)

    def test_missing_file_truncation_and_new_run_keep_the_existing_contract(self):
        self.log.unlink()
        self.assertEqual(self.poll()["parks"], 0)
        self.log.write_bytes(self.PARK)
        self.assertEqual(self.poll()["parks"], 1)
        self.log.write_bytes(self.RESTORE)  # same run, shorter file: resume at the new file's beginning
        self.assertEqual((self.poll()["parks"], self.poll()["restores"]), (1, 1))
        start = self.log.stat().st_size
        self.assertEqual(self.cache.poll(str(self.log), start)["parks"], 0)
        self.append(self.PARK)
        self.assertEqual(self.cache.poll(str(self.log), start)["parks"], 1)

    def test_http_metrics_catches_up_after_more_than_one_mib(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, "Hello.", max_context=4096)
        engine.log_path, engine.log_start = str(self.log), 0
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            def read():
                with urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/metrics", timeout=5) as r:
                    return json.loads(r.read())["conversation_cache"]
            self.append(self.PARK)
            self.assertEqual(read()["parked"], 1)
            self.append(self.RESTORE + b"ordinary log line\n" * 70000)
            state = read()
            self.assertEqual((state["parked"], state["bytes"], state["parks"], state["restores"]), (0, 0, 1, 1))
            self.assertEqual(svc.totals["requests"], 0)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
