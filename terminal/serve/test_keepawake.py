"""prevent_sleep: shared lifecycle and injected backend selection.

Linux D-Bus and optional desktop checks are in test_keepawake_linux.

    python -m unittest serve.test_keepawake -v
"""
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from serve import keepawake
from serve.frontend import ChatTemplate
from serve.keepawake import ES_CONTINUOUS, ES_SYSTEM_REQUIRED, KeepAwake
from serve.server import ByteTokenizer, MockEngine, Service

ROOT = Path(__file__).resolve().parents[1]
ON = ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def wait_for(cond, t=3.0):
    end = time.time() + t
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


class Recorder:
    def __init__(self):
        self.calls, self.threads = [], set()

    def __call__(self, flags):
        self.calls.append(flags)
        self.threads.add(threading.get_ident())
        return ES_CONTINUOUS


class Inhibitor:
    name = "test inhibitor"

    def __init__(self, failures=0):
        self.calls = []
        self.held = False
        self.failures = failures

    def test(self) -> bool:
        return True

    def acquire(self, cancel):
        if self.held:
            return True
        self.calls.append("on")
        if self.failures:
            self.failures -= 1
            raise OSError("controlled failure")
        self.held = True
        return True

    def release(self):
        if self.held:
            self.calls.append("off")
            self.held = False


def windows_controller(rec):
    return KeepAwake(keepawake._WindowsInhibitor(rec), say=mock.Mock())


class KeepAwakeTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.object(keepawake, "GRACE_S", 0.0))

    def test_acquire_attempts_the_hold_before_returning(self):
        rec = Recorder()
        ka = windows_controller(rec)
        self.addCleanup(ka.close)
        # The worker cannot enter until acquire releases this recursive lock
        # to wait for the actual ON attempt, rather than just thread startup.
        with ka._cond:
            ka.acquire()
            self.assertEqual(rec.calls, [ON])
        ka.release()

    def test_held_while_busy_and_released_when_idle(self):
        rec = Recorder()
        ka = windows_controller(rec)
        self.addCleanup(ka.close)
        self.assertEqual(rec.calls, [])                       # nothing until a request
        ka.acquire()
        ka.acquire()                                          # two requests: still one hold
        self.assertTrue(wait_for(lambda: rec.calls == [ON]))
        ka.release()
        time.sleep(0.1)
        self.assertEqual(rec.calls, [ON])                     # one still running
        ka.release()
        self.assertTrue(wait_for(lambda: rec.calls == [ON, ES_CONTINUOUS]))
        ka.close()
        self.assertEqual(rec.calls[-1], ES_CONTINUOUS)
        self.assertEqual(len(rec.threads), 1)                 # set and cleared by the same thread
        self.assertFalse(ON & 0x2)                            # never ES_DISPLAY_REQUIRED

    def test_exit_clears_a_held_state(self):
        rec = Recorder()
        ka = windows_controller(rec)
        self.addCleanup(ka.close)
        ka.acquire()
        self.assertTrue(wait_for(lambda: rec.calls == [ON]))
        ka.close()
        self.assertEqual(rec.calls, [ON, ES_CONTINUOUS])

    def test_a_failing_call_never_breaks_a_request(self):
        def boom(flags):
            raise OSError("no")
        ka = windows_controller(boom)
        ka.acquire()
        ka.release()
        ka.close()

    def test_off_and_unsupported_platform(self):
        say = mock.Mock()
        self.assertIsNone(KeepAwake.create(False, say))
        say.assert_not_called()
        with mock.patch("serve.keepawake.sys.platform", "darwin"):
            self.assertIsNone(KeepAwake.create(True, say))
        self.assertIn("no available inhibitor", say.call_args[0][0])

    def test_service_holds_for_the_whole_request_including_early_close(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        rec = Recorder()
        svc.keep_awake = ka = windows_controller(rec)
        self.addCleanup(ka.close)
        gen = svc.run(tok.encode("hi"), False, None, 8, {}, threading.Event())
        self.addCleanup(gen.close)
        first_kind, _ = next(gen)
        # Keep the mock request open until the asynchronous worker observes it.
        self.assertTrue(wait_for(lambda: rec.calls == [ON]))
        kinds = [first_kind] + [k for k, _ in gen]
        self.assertIn("done", kinds)
        self.assertTrue(wait_for(lambda: rec.calls == [ON, ES_CONTINUOUS]))
        gen = svc.run(tok.encode("hi"), False, None, 8, {}, threading.Event())
        self.addCleanup(gen.close)
        next(gen)
        self.assertTrue(wait_for(lambda: rec.calls[-1] == ON))
        gen.close()                                           # the client left
        self.assertTrue(wait_for(lambda: rec.calls[-1] == ES_CONTINUOUS))
        ka.close()

    def test_default_service_has_none(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.assertIsNone(svc.keep_awake)

    def test_queued_request_is_protected_and_cancellation_releases_it(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        rec = Recorder()
        svc.keep_awake = ka = windows_controller(rec)
        self.addCleanup(ka.close)
        cancel, errors, events = threading.Event(), [], []

        def request():
            try:
                events.extend(svc.run(tok.encode("hi"), False, None, 8, {}, cancel))
            except Exception as exc:
                errors.append(exc)

        with svc.fifo:
            thread = threading.Thread(target=request, daemon=True)
            thread.start()
            try:
                self.assertTrue(wait_for(lambda: svc.status["queued"] == 1))
                self.assertEqual(rec.calls, [ON])
            finally:
                cancel.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual([data["finish"] for kind, data in events if kind == "done"], ["cancel"])
        self.assertTrue(wait_for(lambda: rec.calls == [ON, ES_CONTINUOUS]))

    def test_request_exception_releases_its_hold(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        rec = Recorder()
        svc.keep_awake = ka = windows_controller(rec)
        self.addCleanup(ka.close)
        with mock.patch.object(svc, "_run", side_effect=RuntimeError("request failed")):
            with self.assertRaisesRegex(RuntimeError, "request failed"):
                list(svc.run(tok.encode("hi"), False, None, 8, {}, threading.Event()))
        self.assertTrue(wait_for(lambda: rec.calls == [ON, ES_CONTINUOUS]))

    def test_failed_acquisition_acknowledges_without_breaking_the_request(self):
        failing = Inhibitor(failures=2)
        ka = KeepAwake(failing, say=mock.Mock())
        self.addCleanup(ka.close)
        ka.acquire()
        self.assertFalse(failing.held)
        ka.release()

    def test_reacquisition_waits_for_in_flight_release(self):
        inhibitor = Inhibitor()
        off_entered, finish_off, acquired = (threading.Event() for _ in range(3))
        release = inhibitor.release

        def delayed_release():
            off_entered.set()
            finish_off.wait(2)
            release()

        inhibitor.release = delayed_release
        ka = KeepAwake(inhibitor, say=mock.Mock())
        self.addCleanup(ka.close)
        self.addCleanup(finish_off.set)
        ka.acquire()
        ka.release()
        self.assertTrue(off_entered.wait(1))
        request = threading.Thread(target=lambda: (ka.acquire(), acquired.set()), daemon=True)
        request.start()
        self.assertFalse(acquired.wait(0.05), "a stale ON acknowledgment let work start during OFF")
        finish_off.set()
        self.assertTrue(acquired.wait(1))
        request.join(1)
        self.assertEqual(inhibitor.calls, ["on", "off", "on"])
        self.assertTrue(inhibitor.held)
        ka.release()

    def test_close_cancels_startup_and_wakes_the_request(self):
        inhibitor = Inhibitor()
        entered, acquired = threading.Event(), threading.Event()

        def blocked_acquire(cancel):
            entered.set()
            cancel.wait(2)
            return False

        inhibitor.acquire = blocked_acquire
        ka = KeepAwake(inhibitor, say=mock.Mock())
        self.addCleanup(ka.close)
        request = threading.Thread(target=lambda: (ka.acquire(), acquired.set()), daemon=True)
        request.start()
        self.assertTrue(entered.wait(1))
        self.assertFalse(acquired.is_set())
        ka.close()
        self.assertTrue(acquired.wait(1))
        request.join(1)
        self.assertFalse(ka._thread.is_alive())
        ka.acquire()
        self.assertFalse(inhibitor.held)

    def test_last_release_cancels_pending_acquisition_without_grace(self):
        inhibitor = Inhibitor()
        entered, cancelled = threading.Event(), threading.Event()

        def blocked_acquire(cancel):
            entered.set()
            if cancel.wait(2):
                cancelled.set()
            return False

        inhibitor.acquire = blocked_acquire
        ka = KeepAwake(inhibitor, say=mock.Mock())
        self.addCleanup(ka.close)
        request = threading.Thread(target=ka.acquire, daemon=True)
        request.start()
        self.assertTrue(entered.wait(1))
        ka.release()
        self.assertTrue(cancelled.wait(1))
        request.join(1)
        self.assertFalse(request.is_alive())

    def test_retry_recovers_the_selected_backend_without_repeating_the_warning(self):
        failing = Inhibitor(failures=2)
        ka = KeepAwake(failing, say=mock.Mock())
        self.addCleanup(ka.close)
        with mock.patch.object(keepawake, "RETRY_S", 0.02, create=True):
            ka.acquire()
            self.assertTrue(wait_for(lambda: failing.held))
            self.assertEqual(ka._say.call_count, 1)
            ka.release()
            self.assertTrue(wait_for(lambda: not failing.held))

    def test_windows_zero_return_is_a_failed_hold(self):
        inhibitor = keepawake._WindowsInhibitor(lambda flags: 0)
        self.assertFalse(inhibitor.acquire(threading.Event()))

    def test_unavailable_logging_does_not_stop_backend_recovery(self):
        failing = Inhibitor(failures=1)
        ka = KeepAwake(failing, say=mock.Mock(side_effect=BrokenPipeError))
        self.addCleanup(ka.close)
        with mock.patch.object(keepawake, "RETRY_S", 0.02):
            ka.acquire()
            self.assertTrue(wait_for(lambda: failing.held))
            ka.release()

    def test_thread_start_failure_does_not_break_request_or_close(self):
        ka = KeepAwake(Inhibitor(), say=mock.Mock())
        with mock.patch.object(threading.Thread, "start", side_effect=RuntimeError("no threads")):
            ka.acquire()
        ka.release()
        ka.close()

    def test_failed_release_is_retried_while_idle(self):
        rec = Recorder()
        fail_off = [2]

        def call(flags):
            rec(flags)
            if flags == ES_CONTINUOUS and fail_off[0]:
                fail_off[0] -= 1
                return 0
            return ES_CONTINUOUS

        ka = windows_controller(call)
        self.addCleanup(ka.close)
        with mock.patch.object(keepawake, "RETRY_S", 0.02):
            ka.acquire()
            ka.release()
            self.assertTrue(wait_for(lambda: rec.calls == [ON, ES_CONTINUOUS, ES_CONTINUOUS, ES_CONTINUOUS]),
                            "the idle worker forgot a hold after a failed native clear")
            self.assertEqual(ka._say.call_count, 1, "repeated clear failures should report once until recovery")


class GracePeriodTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000.0
        self.enterContext(mock.patch.object(keepawake, "time", SimpleNamespace(monotonic=lambda: self.now)))
        self.inhibitor = Inhibitor()
        self.ka = KeepAwake(self.inhibitor, say=mock.Mock())
        self.addCleanup(self.ka.close)
        self.waiting = threading.Event()
        original_wait = self.ka._cond.wait

        def worker_wait(timeout=None):
            if threading.current_thread() is self.ka._thread:
                self.waiting.set()
            return original_wait(timeout)

        self.ka._cond.wait = worker_wait

    def advance(self, seconds=0, *, settle=True):
        # Wake the worker after moving its clock, and optionally wait for it
        # to finish processing before checking that a hold remains in place.
        with self.ka._cond:
            self.waiting.clear()
            self.now += seconds
            self.ka._cond.notify_all()
        if settle:
            self.assertTrue(self.waiting.wait(1), "the worker did not settle")

    def test_last_request_holds_for_two_minutes_then_releases(self):
        self.ka.acquire()
        self.ka.release()
        self.advance(119)
        self.assertTrue(self.inhibitor.held)
        self.advance(1)
        self.assertFalse(self.inhibitor.held)
        self.assertEqual(self.inhibitor.calls, ["on", "off"])

    def test_next_request_reuses_the_hold_and_restarts_grace_when_it_finishes(self):
        self.ka.acquire()
        self.ka.release()
        self.advance(90)
        self.ka.acquire()
        self.ka.release()
        self.advance(30)  # the previous batch's deadline
        self.assertEqual(self.inhibitor.calls, ["on"])
        self.advance(89)
        self.assertTrue(self.inhibitor.held)
        self.advance(1)
        self.assertEqual(self.inhibitor.calls, ["on", "off"])

    def test_overlapping_requests_start_grace_only_after_the_last_finishes(self):
        self.ka.acquire()
        self.ka.acquire()
        self.ka.release()
        self.advance(120)
        self.assertTrue(self.inhibitor.held)
        self.ka.release()
        self.advance(119)
        self.assertTrue(self.inhibitor.held)
        self.advance(1)
        self.assertFalse(self.inhibitor.held)

    def test_extra_idle_releases_do_not_extend_grace(self):
        self.ka.acquire()
        self.ka.release()
        self.advance(60)
        self.ka.release()
        self.advance(60)
        self.assertFalse(self.inhibitor.held)

    def test_acquisition_recovery_during_grace_does_not_extend_it(self):
        self.inhibitor.failures = 2
        self.ka.acquire()
        self.ka.release()
        self.advance(5)
        self.assertFalse(self.inhibitor.held)
        self.advance(5)
        self.assertTrue(self.inhibitor.held)
        self.advance(109)
        self.assertTrue(self.inhibitor.held)
        self.advance(1)
        self.assertFalse(self.inhibitor.held)
        self.assertEqual(self.inhibitor.calls, ["on", "on", "on", "off"])

    def test_shutdown_releases_without_waiting_for_grace(self):
        self.ka.acquire()
        self.ka.release()
        self.advance()
        self.assertTrue(self.inhibitor.held)
        self.ka.close()
        self.assertFalse(self.inhibitor.held)
        self.assertFalse(self.ka._thread.is_alive())

    def test_request_during_grace_waits_for_a_health_check_that_shutdown_can_cancel(self):
        self.ka.acquire()
        self.ka.release()
        entered, cancelled, acquired = (threading.Event() for _ in range(3))

        def blocked_check(cancel):
            entered.set()
            if cancel.wait(2):
                cancelled.set()
            return False

        self.inhibitor.acquire = blocked_check
        self.advance(5, settle=False)  # the idle worker starts a health check
        self.assertTrue(entered.wait(1))
        request = threading.Thread(target=lambda: (self.ka.acquire(), acquired.set()), daemon=True)
        request.start()
        self.assertTrue(wait_for(lambda: self.ka._count == 1))
        self.assertFalse(acquired.is_set(), "a stale acknowledgement bypassed the health check")
        self.ka.close()
        self.assertTrue(cancelled.is_set())
        self.assertTrue(acquired.wait(1))
        request.join(1)
        self.assertFalse(self.inhibitor.held)

    def test_acquisition_finishing_after_expiry_is_released(self):
        entered, finish, acquired = (threading.Event() for _ in range(3))
        self.addCleanup(finish.set)
        acquire = self.inhibitor.acquire

        def delayed_acquire(cancel):
            entered.set()
            finish.wait(2)
            return acquire(cancel)

        self.inhibitor.acquire = delayed_acquire
        request = threading.Thread(target=lambda: (self.ka.acquire(), acquired.set()), daemon=True)
        request.start()
        self.assertTrue(entered.wait(1))
        self.ka.release()
        self.advance(120, settle=False)
        finish.set()
        self.assertTrue(acquired.wait(1))
        request.join(1)
        self.advance()
        self.assertEqual(self.inhibitor.calls, ["on", "off"])

    def test_request_during_release_waits_for_reacquisition(self):
        entered, finish, acquired = (threading.Event() for _ in range(3))
        self.addCleanup(finish.set)
        release = self.inhibitor.release

        def delayed_release():
            entered.set()
            finish.wait(2)
            release()

        self.inhibitor.release = delayed_release
        self.ka.acquire()
        self.ka.release()
        self.advance(120, settle=False)
        self.assertTrue(entered.wait(1))
        request = threading.Thread(target=lambda: (self.ka.acquire(), acquired.set()), daemon=True)
        request.start()
        self.assertFalse(acquired.wait(0.05), "a stale acknowledgement let work start during release")
        finish.set()
        self.assertTrue(acquired.wait(1))
        request.join(1)
        self.assertEqual(self.inhibitor.calls, ["on", "off", "on"])
        self.ka.release()
        self.advance(119)
        self.assertTrue(self.inhibitor.held)
        self.advance(1)
        self.assertFalse(self.inhibitor.held)

    def test_service_terminal_paths_start_grace(self):
        for outcome in ("completed", "cancelled", "exception", "stream_closed"):
            with self.subTest(outcome=outcome):
                tok = ByteTokenizer()
                svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
                svc.keep_awake = self.ka
                cancel = threading.Event()
                if outcome == "cancelled":
                    cancel.set()
                gen = svc.run(tok.encode("hi"), False, None, 8, {}, cancel)
                if outcome == "exception":
                    with mock.patch.object(svc, "_run", side_effect=RuntimeError("request failed")):
                        with self.assertRaisesRegex(RuntimeError, "request failed"):
                            list(gen)
                elif outcome == "stream_closed":
                    next(gen)
                    gen.close()
                else:
                    list(gen)
                self.advance(119)
                self.assertTrue(self.inhibitor.held)
                self.advance(1)
                self.assertFalse(self.inhibitor.held)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.object(keepawake, "GRACE_S", 0.0))

    def test_factory_uses_injected_backends_and_stops_at_the_first_success(self):
        for winner in (0, 1, 2, None):
            with self.subTest(winner=winner):
                order, candidates, constructors = [], [], []
                for index in range(3):
                    candidate = Inhibitor()

                    def usable(index=index):
                        order.append(index)
                        return index == winner

                    candidate.test = usable
                    candidates.append(candidate)
                    constructors.append(mock.Mock(return_value=candidate))
                with mock.patch.object(KeepAwake, "_backends", constructors):
                    controller = KeepAwake.create(True, mock.Mock())
                if controller is not None:
                    self.addCleanup(controller.close)
                self.assertIs(controller._backend if controller else None,
                              candidates[winner] if winner is not None else None)
                self.assertEqual(order, [0, 1, 2] if winner is None else list(range(winner + 1)))
                for index, constructor in enumerate(constructors):
                    if winner is None or index <= winner:
                        constructor.assert_called_once_with()
                    else:
                        constructor.assert_not_called()

    def test_disabled_does_not_discover_or_create_resources(self):
        say = mock.Mock()
        backend = mock.Mock()
        with mock.patch.object(KeepAwake, "_backends", (backend,)), \
                mock.patch("threading.Thread") as thread, mock.patch.object(keepawake, "_bus_has_service") as service:
            self.assertIsNone(KeepAwake.create(False, say))
        backend.assert_not_called()
        service.assert_not_called()
        thread.assert_not_called()
        say.assert_not_called()

    def test_windows_backend_and_setup_failure(self):
        for binding in (Recorder(), OSError("no API")):
            with self.subTest(binding=type(binding).__name__), \
                    mock.patch.object(keepawake.sys, "platform", "win32"), \
                    mock.patch.object(keepawake._WindowsInhibitor, "_windows_call") as resolve:
                if isinstance(binding, Exception):
                    resolve.side_effect = binding
                else:
                    resolve.return_value = binding
                controller = KeepAwake.create(True, mock.Mock())
                self.assertEqual(resolve.call_count, 1)
                if isinstance(binding, Exception):
                    self.assertIsNone(controller)
                else:
                    self.addCleanup(controller.close)
                    for _ in range(2):
                        controller.acquire()
                        controller.release()
                        self.assertTrue(wait_for(lambda: binding.calls[-1] == ES_CONTINUOUS))
                    controller.close()
                    self.assertEqual(binding.calls, [ON, ES_CONTINUOUS, ON, ES_CONTINUOUS])
        self.assertFalse(ON & 0x2)                            # never ES_DISPLAY_REQUIRED


if __name__ == "__main__":
    unittest.main()
