"""Batch queue keep-alives and cancellation without a GPU or a child process.

The real admission and protocol readers run against in-memory engine pipes. A fake
clock replaces only blocking I/O so a quiet minute does not slow down the tests.
"""
import io
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from serve import server


def engine_with_slots(busy):
    engine = server.StrataEngine("missing-engine", [], lazy=True)
    engine.batch, engine.gen = len(busy), 1
    engine.slot_busy = list(busy)
    engine.slot_claims = {slot for slot, used in enumerate(busy) if used}
    engine.slot_releasing, engine._orphan_seen = set(), {}
    engine.slot_order = list(range(len(busy)))
    engine.slot_group = [0 for _ in busy]
    engine.slot_held = [[] for _ in busy]
    engine.slot_used = [0.0 for _ in busy]
    engine.slot_live = [None for _ in busy]
    engine.slot_q = [queue.Queue() for _ in busy]
    engine.slot_cv, engine.ctl, engine.wlock = threading.Condition(), threading.Lock(), threading.Lock()
    engine.waiting, engine.admitting, engine.wait_lens, engine.ctl_epoch = 0, 0, [], 0
    engine._yielded, engine._ctl_erred = None, False
    engine.lines = queue.Queue()
    engine.proc = SimpleNamespace(stdin=io.StringIO())
    engine.alive = lambda: True
    return engine


class BatchSlotWait(unittest.TestCase):
    def test_full_slots_keep_the_queued_stream_alive(self):
        # Removing the heartbeat in the full-slot loop must leave this stream silent.
        engine = engine_with_slots([True, True])
        cancel, now, waits = threading.Event(), [0.0], []

        def wait(timeout):
            self.assertGreater(timeout, 0, "a full queue must block rather than spin")
            waits.append(timeout)
            now[0] += timeout
            if now[0] >= 21.0:
                cancel.set()

        beats = []
        with mock.patch.object(server.time, "monotonic", lambda: now[0]), \
                mock.patch.object(engine.slot_cv, "wait", side_effect=wait):
            for token in engine.generate_batched([1, 2, 3], 1, {}, cancel):
                self.assertIsNone(token)
                beats.append(now[0])
        self.assertEqual(beats, [10.0, 20.0])
        self.assertEqual(now[0], 21.0)
        self.assertLessEqual(len(waits), 42)
        self.assertEqual(engine.proc.stdin.getvalue(), "")
        self.assertEqual(engine.slot_busy, [True, True])
        self.assertEqual(engine.slot_claims, {0, 1})
        self.assertEqual((engine.waiting, engine.admitting), (0, 0))
        self.assertFalse(engine.ctl.locked())


class CancelOnAcquire:
    """A real lock whose newly admitted request loses its client at that boundary."""
    def __init__(self, cancel):
        self.lock, self.cancel = threading.Lock(), cancel

    def acquire(self, **kwargs):
        acquired = self.lock.acquire(**kwargs)
        if acquired:
            self.cancel.set()
        return acquired

    def release(self):
        self.lock.release()

    def locked(self):
        return self.lock.locked()


class BatchAdmissionCancel(unittest.TestCase):
    def run_cancelled(self, busy, max_new, *, during_acquire=False):
        engine = engine_with_slots(busy)
        # A bad admission can still finish, so a regression fails without hanging.
        engine.lines.put("DONE 0 3 1 0 cancel")
        engine.lines.put("BADM 1 0")
        cancel = threading.Event()
        if during_acquire:
            engine.ctl = CancelOnAcquire(cancel)
        else:
            cancel.set()
        self.assertEqual(list(engine.generate_batched([1, 2, 3], max_new, {}, cancel)), [])
        self.assertEqual(engine.proc.stdin.getvalue(), "", "a cancelled request must never reach the engine")
        self.assertEqual(engine.slot_busy, busy)
        self.assertEqual(engine.slot_claims, {slot for slot, used in enumerate(busy) if used})
        self.assertEqual((engine.waiting, engine.admitting), (0, 0))
        self.assertFalse(engine.ctl.locked())

    def test_a_pre_cancelled_request_is_not_sent(self):
        # Missing the check before an immediately successful lock admits dead work.
        for busy, max_new in (([False, False], 8), ([True, False], 1)):
            with self.subTest(busy=busy):
                self.run_cancelled(busy, max_new)

    def test_a_client_lost_during_control_acquisition_is_not_sent(self):
        # Checking only before acquire misses a disconnect while the lock is taken.
        for busy, max_new in (([False, False], 8), ([True, False], 1)):
            with self.subTest(busy=busy):
                self.run_cancelled(busy, max_new, during_acquire=True)

    def test_a_client_lost_while_claiming_a_slot_is_not_sent(self):
        # The send boundary must check again after selecting a free batch slot.
        engine = engine_with_slots([True, False])
        engine.lines.put("DONE 0 3 1 0 cancel")
        engine.lines.put("BADM 1 0")
        cancel, pick_slot = threading.Event(), engine.pick_slot

        def disconnect(prompt):
            slot = pick_slot(prompt)
            cancel.set()
            return slot

        with mock.patch.object(engine, "pick_slot", side_effect=disconnect):
            self.assertEqual(list(engine.generate_batched([1, 2, 3], 1, {}, cancel)), [])
        self.assertEqual(engine.proc.stdin.getvalue(), "")
        self.assertEqual(engine.slot_busy, [True, False])
        self.assertEqual(engine.slot_claims, {0})
        self.assertFalse(engine.ctl.locked())


class BatchDecodeWait(unittest.TestCase):
    def admitted_engine(self):
        engine = engine_with_slots([True, False])
        for line in ("T 65", "DONE 1 3 1 1 length", "BADM 1 1"):
            engine.lines.put(line)
        return engine

    def test_a_quiet_slot_stops_promptly_and_stays_owned_until_bdone(self):
        # A ten-second slot read delays BSTOP; freeing on BSTOP would reuse live GPU state.
        for cancel_before_wait in (False, True):
            with self.subTest(cancel_before_wait=cancel_before_wait):
                engine = self.admitted_engine()
                cancel, now, stop_seen_at = threading.Event(), [0.0], []

                def read_slot(timeout):
                    self.assertGreater(timeout, 0, "quiet decoding must not spin")
                    self.assertTrue(engine.slot_busy[1], "BSTOP does not free a slot before BDONE")
                    self.assertIn(1, engine.slot_claims)
                    if "BSTOP 1\n" in engine.proc.stdin.getvalue() and not stop_seen_at:
                        stop_seen_at.append(now[0])
                    now[0] += timeout
                    if now[0] >= 0.2:
                        cancel.set()
                    if now[0] >= 1.5 and stop_seen_at:
                        return "BDONE 1 1 cancel 1.0"
                    raise queue.Empty

                with mock.patch.object(server.time, "monotonic", lambda: now[0]), \
                        mock.patch.object(engine.slot_q[1], "get", side_effect=read_slot):
                    run = engine.generate_batched([1, 2, 3], 8, {}, cancel)
                    self.assertEqual(next(run), 65)
                    if cancel_before_wait:
                        cancel.set()
                    list(run)
                self.assertEqual(len(stop_seen_at), 1)
                self.assertLessEqual(stop_seen_at[0], 0.7, "disconnect must reach BSTOP within one short poll")
                self.assertEqual(engine.proc.stdin.getvalue(), "BGEN 1 8 1,2,3\nBSTOP 1\n")
                self.assertEqual(engine.slot_busy, [True, False])
                self.assertEqual(engine.slot_claims, {0})
                self.assertIsNone(engine.slot_live[1])
                self.assertEqual(engine.last["finish"], "cancel")

    def test_short_polls_keep_ten_second_heartbeats(self):
        # Yielding on every short poll floods the HTTP stream with keep-alives.
        engine = self.admitted_engine()
        now, polls, beats = [0.0], [], []

        def read_slot(timeout):
            self.assertGreater(timeout, 0)
            polls.append(timeout)
            now[0] = min(21.0, now[0] + timeout)
            if now[0] >= 21.0:
                return "BDONE 1 1 length 1.0"
            raise queue.Empty

        with mock.patch.object(server.time, "monotonic", lambda: now[0]), \
                mock.patch.object(engine.slot_q[1], "get", side_effect=read_slot):
            for token in engine.generate_batched([1, 2, 3], 8, {}, threading.Event()):
                if token is None:
                    beats.append(now[0])
                else:
                    self.assertEqual(token, 65)
        self.assertEqual(beats, [10.0, 20.0])
        self.assertLessEqual(len(polls), 44)
        self.assertEqual(engine.proc.stdin.getvalue(), "BGEN 1 8 1,2,3\n")
        self.assertEqual(engine.slot_busy, [True, False])


if __name__ == "__main__":
    unittest.main()
