"""Request priority uses the shared slots and their existing control lock."""
import threading
import time
import unittest
from unittest import mock

from serve import server
from serve.test_admit_order import engine_without_process
from serve import test_responses


class RequestPriorityTest(unittest.TestCase):
    def setUp(self):
        self.e = engine_without_process()

    def until(self, condition):
        deadline = time.monotonic() + 5
        while not condition() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(condition())

    def test_priority_takes_next_free_slot(self):
        e = self.e
        e.slot_busy[:] = [True] * e.batch
        e.slot_claims.update(range(e.batch))
        e.ctl.acquire()
        order = []
        threads = []

        def enqueue(priority):
            cancel = threading.Event()
            cancel._strata_priority = priority

            def run():
                acquired = yield_from_control(e, cancel, new=True)
                if acquired:
                    with e.slot_cv:
                        order.append((priority, not all(e.slot_busy)))
                        e.slot_busy[:] = [False] * e.batch
                        e.slot_cv.notify_all()
                    e.ctl.release()

            thread = threading.Thread(target=run, daemon=True)
            threads.append(thread)
            thread.start()
            self.until(lambda: e.waiting == len(threads))

        with mock.patch.object(server, "REQUEST_PRIORITY", True, create=True):
            enqueue(0)
            enqueue(-1)
            e.ctl.release()
            with e.slot_cv:
                e.slot_busy[-1] = False
                e.slot_cv.notify_all()
            for thread in threads:
                thread.join(5)
        self.assertEqual(order, [(-1, True), (0, True)])

    def test_suspended_request_can_resume_while_urgent_request_needs_a_slot(self):
        e = self.e
        e.slot_busy[:] = [True] * e.batch
        e.slot_claims.update(range(e.batch))
        urgent = threading.Event()
        urgent._strata_priority = -1
        urgent_acquired = threading.Event()

        def run():
            if yield_from_control(e, urgent, new=True):
                urgent_acquired.set()
                e.ctl.release()

        with mock.patch.object(server, "REQUEST_PRIORITY", True):
            waiter = threading.Thread(target=run, daemon=True)
            waiter.start()
            try:
                self.until(lambda: e.waiting == 1)
                resumed = threading.Event()
                resumed._strata_queue_deadline = time.monotonic() + 3
                self.assertTrue(yield_from_control(e, resumed, after_epoch=e.ctl_epoch))
                try:
                    self.assertFalse(urgent_acquired.is_set())
                    with e.slot_cv:
                        e.slot_busy[-1] = False
                        e.slot_cv.notify_all()
                finally:
                    e.ctl.release()
                self.assertTrue(urgent_acquired.wait(3))
            finally:
                urgent.set()
                waiter.join(5)

    def test_expired_or_canceled_waiter_does_not_trigger_yield(self):
        with mock.patch.object(server, "REQUEST_PRIORITY", True):
            urgent = threading.Event()
            urgent._strata_priority = -1
            urgent._strata_queue_deadline = time.monotonic() - 1
            self.e.wait_lens = [[64, time.monotonic(), True, None, urgent, None]]
            cancel = threading.Event()
            self.assertFalse(self.e._should_yield(64, cancel))
            urgent._strata_queue_deadline = None
            self.assertTrue(self.e._should_yield(64, cancel))
            urgent.set()
            self.assertFalse(self.e._should_yield(64, cancel))


def yield_from_control(engine, cancel, **options):
    control = engine._take_control(cancel, 64, **options)
    try:
        while True:
            next(control)
    except StopIteration as done:
        return done.value


class PriorityOverHttp(test_responses.Server):
    def test_priority_reaches_the_engine_on_each_route(self):
        message = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 32}
        routes = {"/v1/messages": message, "/v1/chat/completions": message,
                  "/v1/responses": {"model": "m", "input": "hi"}}
        observed = []
        generate = self.engine.generate

        def record(ids, max_new, sampling, cancel, *args, **kwargs):
            observed.append(getattr(cancel, "_strata_priority", None))
            yield from generate(ids, max_new, sampling, cancel, *args, **kwargs)

        with mock.patch.object(self.engine, "generate", record):
            for path, body in routes.items():
                with self.subTest(path=path), mock.patch.object(server, "REQUEST_PRIORITY", True):
                    code, reply = self.post(body, path=path, headers={"X-Strata-Priority": "-7"})
                    self.assertEqual(code, 200, reply)
                    self.assertEqual(observed[-1], -7)
                    code, reply = self.post(body, path=path, headers={"X-Strata-Priority": "2147483648"})
                    self.assertEqual(code, 400, reply)
                    code, reply = self.post(body, path=path)
                    self.assertEqual(code, 200, reply)
                    self.assertEqual(observed[-1], 0)
            with mock.patch.object(server, "REQUEST_PRIORITY", False):
                code, reply = self.post(routes["/v1/responses"], headers={"X-Strata-Priority": "ignored"})
                self.assertEqual(code, 200, reply)
                self.assertIsNone(observed[-1])
