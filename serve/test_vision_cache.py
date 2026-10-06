"""#1072: image cache entries live until prepare() copies them, then return to the 64-entry bound.

The real Vision cache and Service run against an in-memory ENC pipe; no GPU, model or network is needed.
    python -m unittest serve.test_vision_cache -v
"""
import contextlib
import hashlib
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from serve import server
from serve.frontend import ChatTemplate

ROOT = Path(__file__).resolve().parents[1]


def picture(i):
    # PNG signature is enough for normalize()'s pass-through; no image decoder is under test.
    return b"\x89PNG\r\n\x1a\n" + i.to_bytes(4, "big")


def record(data):
    return hashlib.sha256(data).digest()


class EncoderPipe:
    def __init__(self, root):
        self.root, self.calls, self.fail_at = root, 0, None
        self.answer = ""

    def write(self, line):
        _, image, out = line.split()
        self.calls += 1
        if self.calls == self.fail_at:
            self.answer = "ERR injected encoding error\n"
        else:
            (self.root / out).write_bytes(record((self.root / image).read_bytes()))
            self.answer = "OK 1 1 1 0\n"

    def flush(self):
        pass

    def readline(self):
        return self.answer


class TrackedLock:
    """Report the second request's lock attempt without changing Lock's acquisition/release behavior."""
    def __init__(self):
        self.lock = threading.Lock()
        self.attempted = threading.Event()

    def __enter__(self):
        if threading.current_thread().name == "image-B":
            self.attempted.set()
        self.lock.acquire()
        return self

    def __exit__(self, *args):
        self.lock.release()


class VisionCacheLifetime(unittest.TestCase):
    @contextlib.contextmanager
    def service(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            v = server.Vision.__new__(server.Vision)
            v.dir, v.lock, v.cache, v.cache_batches = root, threading.RLock(), {}, 0
            v.stopped = False
            pipe = EncoderPipe(root)
            v.proc = SimpleNamespace(stdin=pipe, stdout=pipe, poll=lambda: None)
            tok = server.ByteTokenizer()
            svc = server.Service(server.MockEngine(tok, "ok", max_context=131072), tok,
                                 ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=v)
            number = 0

            def combined(*args):
                nonlocal number
                number += 1
                return root / f"req-{number}.sve"

            with mock.patch.object(server, "combined_embeddings_path", side_effect=combined), \
                    mock.patch.object(server.Vision, "load", side_effect=lambda src: picture(int(src))):
                try:
                    yield svc, v, pipe
                finally:
                    svc.drop_embeddings()

    @staticmethod
    def prepare(svc, values, **kwargs):
        messages = [{"role": "user", "content": [{"type": "image", "source": str(i)} for i in values]}]
        return svc.prepare(messages, None, {"enable_thinking": False}, **kwargs)

    def assert_idle(self, svc, v):
        self.assertEqual(v.cache_batches, 0)
        self.assertLessEqual(len(v.cache), 64)
        self.assertEqual(len(list(v.dir.glob("*.sve"))), len(v.cache))
        self.assertEqual(list(v.dir.glob("*.img")), [])
        # A different thread must be able to acquire both locks after every exit, including BaseException.
        free = threading.Event()

        def take():
            with svc.fifo, v.lock:
                free.set()

        thread = threading.Thread(target=take, daemon=True)
        thread.start()
        self.assertTrue(free.wait(2), "FIFO/vision lock leaked")
        thread.join(2)

    def test_request_larger_than_cache_keeps_all_records_in_order(self):
        for size in (64, 65, 70, 130):
            with self.subTest(size=size), self.service() as (svc, v, pipe):
                self.prepare(svc, range(size))
                self.assertEqual(svc.embeddings.path.read_bytes(), b"".join(record(picture(i)) for i in range(size)))
                self.assertEqual(len(v.cache), 64)
                self.assertEqual(pipe.calls, size)
                svc.drop_embeddings()
                self.assert_idle(svc, v)

    def test_repeated_images_encode_once_and_preserve_order(self):
        with self.service() as (svc, v, pipe):
            values = [0] * 70
            self.prepare(svc, values)
            self.assertEqual(pipe.calls, 1)
            self.assertEqual(svc.embeddings.path.read_bytes(), record(picture(0)) * 70)
            svc.drop_embeddings()
            values = list(range(70)) + [0, 4, 0, 69]
            self.prepare(svc, values)
            self.assertEqual(pipe.calls, 70)
            self.assertEqual(svc.embeddings.path.read_bytes(), b"".join(record(picture(i)) for i in values))
            svc.drop_embeddings()
            self.assert_idle(svc, v)

    def race(self, before_copy=False, tool=False):
        with self.service() as (svc, v, pipe):
            self.prepare(svc, range(64))
            svc.drop_embeddings()
            svc.fifo = TrackedLock()
            paused, resume, b_done = threading.Event(), threading.Event(), threading.Event()
            results, errors = {}, []
            original = server.write_temporary if before_copy else svc.tok.encode

            def gated(*args, **kwargs):
                at_gap = before_copy or args[0] == server.IMAGE_PAD and kwargs.get("parse_special") is False
                if threading.current_thread().name == "image-A" and at_gap:
                    paused.set()
                    if not resume.wait(3):
                        raise AssertionError("test release did not arrive")
                return original(*args, **kwargs)

            def request(name, value):
                try:
                    if tool and name == "B":
                        # Tool-image preflight is another encoder caller and must follow the same lock order.
                        messages = [{"role": "tool", "content": [{"type": "image", "source": str(value)}]}]
                        svc.prepare(messages, None, {"enable_thinking": False})
                    else:
                        self.prepare(svc, [value])
                    results[name] = svc.embeddings.path.read_bytes()
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    svc.drop_embeddings()
                    if name == "B":
                        b_done.set()

            target, attribute = (server, "write_temporary") if before_copy else (svc.tok, "encode")
            with mock.patch.object(target, attribute, side_effect=gated):
                a = threading.Thread(target=request, args=("A", 0), name="image-A", daemon=True)
                b = threading.Thread(target=request, args=("B", 64), name="image-B", daemon=True)
                a.start()
                try:
                    self.assertTrue(paused.wait(2))
                    b.start()
                    self.assertTrue(svc.fifo.attempted.wait(2))
                    self.assertFalse(b_done.wait(0.1), "second encode ran while the first still needed its paths")
                finally:
                    resume.set()
                    a.join(3)
                    if b.ident is not None:
                        b.join(3)
            self.assertFalse(a.is_alive() or b.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(results, {"A": record(picture(0)), "B": record(picture(64))})
            self.assert_idle(svc, v)

    def test_one_image_requests_cannot_evict_before_stat(self):
        self.race()

    def test_one_image_requests_cannot_evict_between_stat_and_copy(self):
        self.race(before_copy=True)

    def test_tool_preflight_waits_for_active_copy(self):
        self.race(before_copy=True, tool=True)

    def test_all_prepare_failures_trim_and_release(self):
        class Abort(BaseException):
            pass

        cases = ("encode", "markers", "starting", "no_room", "max_new", "write", "abort")
        for case in cases:
            with self.subTest(case=case), self.service() as (svc, v, pipe), contextlib.ExitStack() as stack:
                error = ValueError
                kwargs = {}
                if case == "encode":
                    pipe.fail_at = 66
                elif case == "markers":
                    stack.enter_context(mock.patch.object(svc, "encode_prompt", return_value=[42]))
                elif case == "starting":
                    svc.engine.max_context, svc.engine.starting = 0, True
                    error = server.EngineStarting
                elif case == "no_room":
                    svc.engine.max_context = 64
                elif case == "max_new":
                    kwargs["max_new"] = 131072
                else:
                    error = OSError if case == "write" else Abort
                    def fail(path, parts):
                        path.write_bytes(b"partial")
                        raise error("injected copy failure")
                    stack.enter_context(mock.patch.object(server, "write_temporary", side_effect=fail))
                try:
                    with self.assertRaises(error):
                        self.prepare(svc, range(70), **kwargs)
                finally:
                    svc.drop_embeddings()
                self.assertEqual(len(v.cache), 64)
                self.assert_idle(svc, v)

    def test_nested_batch_trims_only_on_outer_exit(self):
        with self.service() as (svc, v, pipe):
            with svc.fifo, v.batch():
                v.encode("0")
                with v.batch():
                    for i in range(1, 70):
                        v.encode(str(i))
                self.assertEqual(len(v.cache), 70)
                self.assertTrue(all(path.exists() for path, _ in v.cache.values()))
            self.assertEqual(len(v.cache), 64)
            self.assert_idle(svc, v)

    def test_combined_file_survives_raw_eviction_and_run_cleanup(self):
        with self.service() as (svc, v, pipe):
            ids, thinking, max_new = self.prepare(svc, [0])
            combined = svc.embeddings.path
            raw = next(iter(v.cache.values()))[0]
            for i in range(1, 65):
                v.encode(str(i))
            self.assertFalse(raw.exists())
            self.assertEqual(combined.read_bytes(), record(picture(0)))
            gen = svc.run(ids, thinking, None, max_new, {}, threading.Event())
            next(gen)
            self.assertTrue(combined.exists())
            gen.close()
            self.assertFalse(combined.exists())
            self.assert_idle(svc, v)

    def test_no_run_drop_and_direct_encode_remain_bounded(self):
        with self.service() as (svc, v, pipe):
            for start in range(0, 210, 70):
                self.prepare(svc, range(start, start + 70))
                svc.drop_embeddings()
                svc.drop_embeddings()
                self.assert_idle(svc, v)
            for i in range(210, 300):
                v.encode(str(i))
                self.assertLessEqual(len(v.cache), 64)
            self.assert_idle(svc, v)


if __name__ == "__main__":
    unittest.main()
