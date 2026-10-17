"""serve/test_monitor_waiting.py - the api-monitor shows the engine's control-line wait (#2109 follow-up).

With --batch (parallel) requests do not queue on the server's fifo: every request is registered as running the
moment it arrives, and where it really waits is the engine's control lines - one prompt read at a time, behind
whatever read is in progress (_take_control).  Until now the request list could not show that: a row went
"queued" -> "generating" in the same second, and queue_s stayed 0.000, so a request waiting 55 s for the lines
looked identical to one that started reading at once.

The fix lives where the wait lives: _take_control marks the request's own monitor record "waiting" while it is
in the queue and "reading" when it takes the lines, and adds the seconds it spent waiting to the record's
queue_s (the server-side fifo's own wait keeps adding to the same field, as before).  MockEngine has no batch
slots, so the path is exercised with the same stubbed engine as serve/test_admit_order.py: no process, no GPU.
The wait is driven on a helper thread and the record is polled: the wait's own heartbeat only yields every 10 s,
and a test must not sleep through one to see a state that is marked before the first yield.

    python -m unittest serve.test_monitor_waiting -v
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as server  # noqa: E402
from serve.server import StrataEngine  # noqa: E402


def engine_without_process(n=2):
    """A StrataEngine with the admission state of a started one and no process: alive() is `e.up`."""
    e = StrataEngine("missing-executable", [], lazy=True)
    e.batch = n
    e.slot_busy = [False] * n
    e.slot_claims, e.slot_releasing, e._orphan_seen = set(), set(), {}
    e.slot_cv = threading.Condition()
    e.waiting, e.admitting, e.wait_lens, e.ctl_epoch = 0, 0, [], 0
    e.ctl = threading.Lock()
    e.gen = 1
    e.up = True
    e.starting = False
    e.alive = lambda: True
    return e


class MonitorWaitingTest(unittest.TestCase):
    """_take_control drives the monitor record's state and queue_s (no GPU, no process)."""

    def setUp(self):
        self.e = engine_without_process(2)
        # a record like begin_request's, hung on this thread the way the server hangs it
        self.record = {"id": "test", "state": "queued", "queue_s": 0.0}
        server._MON_TRACE.record = self.record
        server._MON_TRACE.lock = threading.Lock()

    def tearDown(self):
        server._MON_TRACE.record = None
        if self.e.ctl.locked():
            self.e.ctl.release()

    def start_wait(self, cancel=None, new=True):
        """Run the wait generator on a helper thread; its result lands in self.wait_result.
        _MON_TRACE is thread-local, exactly as on the server (begin_request runs on the request's own
        thread), so the record is hung on the helper thread the same way."""
        self.wait_result = {}
        record = server._MON_TRACE.record
        lock = server._MON_TRACE.lock

        def run():
            server._MON_TRACE.record = record
            server._MON_TRACE.lock = lock
            gen = self.e._take_control(cancel or threading.Event(), 100, new=new)
            try:
                while True:
                    next(gen)
            except StopIteration as stop:
                self.wait_result["ok"] = stop.value

        self.wait_thread = threading.Thread(target=run, daemon=True)
        self.wait_thread.start()

    def wait_for_state(self, want, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self.record["state"] != want:
            time.sleep(0.02)
        return self.record["state"] == want

    def test_free_lines_mark_reading_at_once(self):
        """Nobody holds the lines: the wait ends without a heartbeat and the record lands on 'reading',
        its queue_s counting the (sub-second) wait."""
        self.start_wait()
        self.wait_thread.join(5)
        self.assertTrue(self.wait_result.get("ok"))
        self.assertEqual(self.record["state"], "reading")
        self.assertLess(self.record["queue_s"], 1.0)

    def test_held_lines_show_waiting_then_reading(self):
        """A read in progress: the record says 'waiting' while the lines are held and nothing is counted yet;
        taking them flips it to 'reading' and adds the wait to queue_s."""
        self.e.ctl.acquire()                               # another request's read is in progress
        self.start_wait()
        self.assertTrue(self.wait_for_state("waiting"), f"never saw waiting: {self.record['state']}")
        self.assertEqual(self.record["queue_s"], 0.0)      # nothing counted until the lines are taken
        time.sleep(0.3)                                    # a measurable wait
        self.e.ctl.release()
        self.wait_thread.join(5)
        self.assertTrue(self.wait_result.get("ok"))
        self.assertEqual(self.record["state"], "reading")
        self.assertGreater(self.record["queue_s"], 0.0)

    def test_cancelled_wait_stays_waiting_and_counts_nothing(self):
        """Cancelled on the lines: the wait ends False, the record never claims 'reading', queue_s stays 0,
        and the waiting counter / wait_lens that /metrics and STRATA_ADMIT_SHORT_FIRST read go back to empty."""
        self.e.ctl.acquire()
        cancel = threading.Event()
        self.start_wait(cancel=cancel)
        self.assertTrue(self.wait_for_state("waiting"))
        time.sleep(0.6)                                    # one failed acquire cycle at least
        cancel.set()                                       # the lines stay held: the wait ends on the cancel check
        self.wait_thread.join(5)
        self.assertIs(self.wait_result.get("ok"), False)
        self.assertEqual(self.record["state"], "waiting")
        self.assertEqual(self.record["queue_s"], 0.0)
        self.assertEqual(self.e.waiting, 0)
        self.assertEqual(self.e.wait_lens, [])

    def test_no_record_is_inert(self):
        """api_monitor off (no record on the thread): the wait runs exactly as before, nothing to mark."""
        server._MON_TRACE.record = None
        gen = self.e._take_control(threading.Event(), 100, new=True)
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            self.assertTrue(stop.value)
        self.assertEqual(self.e.waiting, 0)

    def test_fifo_wait_and_line_wait_add_up(self):
        """The server's fifo wait already added to queue_s before this request reached the lines: the two waits
        sum into one field instead of overwriting each other."""
        self.record["queue_s"] = 1.5                       # what the server's fifo turn already counted
        self.e.ctl.acquire()
        self.start_wait()
        self.assertTrue(self.wait_for_state("waiting"))
        time.sleep(0.3)
        self.e.ctl.release()
        self.wait_thread.join(5)
        self.assertGreater(self.record["queue_s"], 1.5)


if __name__ == "__main__":
    unittest.main()
