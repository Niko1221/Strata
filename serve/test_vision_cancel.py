"""A picture's encode whose client left (the web page's Stop, a closed connection) no longer holds every request.

The encoder (strata-vision) cannot stop in the middle of a picture, and the encode holds the engine's single turn, so a
text question sent after a stopped 1440p picture waited for the rest of that picture's encode (74 s on 16 CPU threads,
measured).  Now the encode whose client left is ended as soon as another request waits for the turn, unless a waiting
request sends the same picture; while nobody waits it is read to the end and kept, so a follow-up about it finds it read.
A resident encoder that was ended starts again with the next picture.

    python -m serve.test_vision_cancel
"""
import base64
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, ClientGone, MockEngine, Service, Turn, Vision

ROOT = Path(__file__).resolve().parents[1]


class FakeReader:
    """strata-vision on a pipe: each ENC line is answered after `secs` (the encode), unless kill() ends it first."""

    def __init__(self, d, secs):
        self.d, self.secs = Path(d), secs
        self.lines, self.killed = [], False
        self.ended = threading.Event()
        self.pending: queue.Queue = queue.Queue()
        self.stdin = self.stdout = self

    def write(self, text):
        self.lines.append(text)

    def flush(self):
        self.pending.put(self.lines[-1].split()[2])

    def readline(self):
        out = self.pending.get()
        if self.ended.wait(self.secs):
            return ""                                         # killed in the middle of the picture
        (self.d / out).write_bytes(b"rows")
        return "OK 3 1 3 10\n"

    def kill(self):
        self.killed = True
        self.ended.set()

    def wait(self, timeout=None):
        return -9

    def poll(self):
        return -9 if self.killed else None


def reader(d, secs):
    v = Vision.__new__(Vision)
    v.dir, v.cache, v.lock = Path(d), {}, threading.Lock()
    v.stopped, v.elastic, v.max_tokens = False, False, 0
    v.proc = FakeReader(d, secs)
    v.normalize = lambda data: data
    return v


def picture(text):
    return "data:image/png;base64," + base64.b64encode(text.encode()).decode()


def turn(url, question="what is in it?"):
    return [{"role": "user", "content": [{"type": "text", "text": question}, {"type": "image", "source": url}]}]


def service(v):
    tok = ByteTokenizer()
    return Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=v)


def until(cond, limit=5.0):
    end = time.monotonic() + limit
    while time.monotonic() < end and not cond():
        time.sleep(0.01)
    return cond()


class Request(threading.Thread):
    def __init__(self, svc, messages, cancel):
        super().__init__(daemon=True)
        self.svc, self.messages, self.cancel = svc, messages, cancel
        self.error, self.ids = None, None

    def run(self):
        try:
            self.ids, _, _ = self.svc.prepare(self.messages, None, {}, cancel=self.cancel)
        except Exception as e:  # noqa: BLE001
            self.error = e


class CancelledEncode(unittest.TestCase):
    def test_turn_counts_who_waits(self):
        t = Turn()
        self.assertTrue(t.acquire(blocking=False))
        self.assertFalse(t.acquire(blocking=False))
        self.assertEqual(t.waiting(), 0)                       # a refused try is not a wait
        th = threading.Thread(target=lambda: (t.acquire(timeout=5), t.release()))
        th.start()
        self.assertTrue(until(lambda: t.waiting() == 1))
        t.release()
        th.join(5)
        self.assertEqual(t.waiting(), 0)
        with t.waiting_for_it():
            self.assertEqual(t.waiting(), 1)
        with t:
            self.assertTrue(t.locked())
        self.assertFalse(t.locked())

    def test_stopped_as_soon_as_another_request_waits(self):
        with tempfile.TemporaryDirectory() as d:
            v = reader(d, secs=60)
            svc, url = service(v), picture("one")
            first = v.proc

            def restart():
                v.proc, v.stopped = FakeReader(d, 0.05), False
            again = mock.patch.object(v, "restart", side_effect=restart).start()
            self.addCleanup(mock.patch.stopall)
            a = Request(svc, turn(url), threading.Event())
            a.start()
            self.assertTrue(until(lambda: first.lines))         # the encode began
            a.cancel.set()                                      # its client left
            time.sleep(0.4)
            self.assertTrue(a.is_alive())                       # nobody waits: still reading
            t0 = time.monotonic()
            self.assertTrue(svc.fifo.acquire(timeout=5))        # a text request wants the turn
            took = time.monotonic() - t0
            self.assertEqual(again.call_count, 1)               # started again before the turn was given on
            self.assertTrue(v.alive())
            svc.fifo.release()
            a.join(5)
            self.assertLess(took, 2)
            self.assertIsInstance(a.error, ClientGone)
            self.assertTrue(first.killed)
            self.assertFalse(v.all_cached([url]))
            self.assertEqual(list(Path(d).glob("*.sve")), [])   # no half-written file left

            # the next request with the picture reads it on the new encoder
            b = Request(svc, turn(url), threading.Event())
            b.start()
            b.join(5)
            self.assertIsNone(b.error)
            self.assertEqual(again.call_count, 1)
            self.assertTrue(v.all_cached([url]))

    def test_an_encoder_that_did_not_start_again_starts_with_the_next_picture(self):
        with tempfile.TemporaryDirectory() as d:
            v = reader(d, secs=60)
            svc, url = service(v), picture("five")
            calls = []

            def restart():
                calls.append(1)
                if len(calls) == 1:
                    raise RuntimeError("the vision encoder did not start")
                v.proc, v.stopped = FakeReader(d, 0.05), False
            mock.patch.object(v, "restart", side_effect=restart).start()
            self.addCleanup(mock.patch.stopall)
            a = Request(svc, turn(url), threading.Event())
            a.start()
            self.assertTrue(until(lambda: v.proc.lines))
            a.cancel.set()
            self.assertTrue(svc.fifo.acquire(timeout=5))
            svc.fifo.release()
            a.join(5)
            self.assertIsInstance(a.error, ClientGone)
            self.assertFalse(v.alive())                         # the restart after the stop failed ...
            b = Request(svc, turn(url), threading.Event())
            b.start()
            b.join(5)
            self.assertIsNone(b.error)                          # ... the next picture's turn started it (_vision_up)
            self.assertEqual(len(calls), 2)

    def test_read_to_the_end_and_kept_while_nobody_waits(self):
        with tempfile.TemporaryDirectory() as d:
            v = reader(d, secs=0.6)
            svc, url = service(v), picture("two")
            a = Request(svc, turn(url), threading.Event())
            a.start()
            self.assertTrue(until(lambda: v.proc.lines))
            a.cancel.set()
            a.join(5)
            self.assertIsInstance(a.error, ClientGone)          # nothing is answered ...
            self.assertFalse(v.proc.killed)
            self.assertTrue(v.all_cached([url]))                # ... but the follow-up finds the picture read
            self.assertFalse(svc.fifo.locked())

    def test_kept_for_a_waiting_request_with_the_same_picture(self):
        with tempfile.TemporaryDirectory() as d:
            v = reader(d, secs=1.0)
            svc, url = service(v), picture("three")
            a = Request(svc, turn(url), threading.Event())
            a.start()
            self.assertTrue(until(lambda: v.proc.lines))
            a.cancel.set()
            follow = turn(url) + [{"role": "assistant", "content": ""}, {"role": "user", "content": "and now?"}]
            b = Request(svc, follow, threading.Event())         # the follow-up, sent while the picture is read
            b.start()
            a.join(5)
            b.join(5)
            self.assertIsInstance(a.error, ClientGone)
            self.assertIsNone(b.error)
            self.assertFalse(v.proc.killed)
            self.assertEqual(len(v.proc.lines), 1)              # read once, for both
            self.assertEqual(v.is_wanted(Vision.raw_key(url)), False)   # nothing left registered

    def test_a_request_that_stays_is_never_stopped(self):
        with tempfile.TemporaryDirectory() as d:
            v = reader(d, secs=0.6)
            svc, url = service(v), picture("four")
            a = Request(svc, turn(url), threading.Event())
            a.start()
            self.assertTrue(until(lambda: v.proc.lines))
            self.assertTrue(svc.fifo.acquire(timeout=5))        # waits for the whole encode
            svc.fifo.release()
            a.join(5)
            self.assertIsNone(a.error)
            self.assertFalse(v.proc.killed)

    def test_unload_with_wait_counts_as_waiting(self):
        svc = service(None)
        svc.engine.unload = lambda: None                        # an engine that can unload (the mock cannot)
        svc.engine.alive = lambda: True
        with svc.fifo:
            seen = []
            th = threading.Thread(target=lambda: seen.append(svc.unload_waiting(5)))
            th.start()
            self.assertTrue(until(lambda: svc.fifo.waiting() == 1))
        th.join(5)
        self.assertEqual(seen, ["unloaded"])                    # it got the turn once the request ended
        self.assertEqual(svc.fifo.waiting(), 0)


if __name__ == "__main__":
    unittest.main()
