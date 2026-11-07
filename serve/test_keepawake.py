"""#1727: prevent_sleep and keep_awake.  The Windows call is mocked; the Linux and macOS keeper runs a real child.

    python -m unittest serve.test_keepawake -v
"""
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.keepawake import ES_CONTINUOUS, ES_SYSTEM_REQUIRED, KeepAwake, ProcessKeepAwake
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
        return 0


class KeepAwakeTests(unittest.TestCase):
    def test_held_while_busy_and_released_when_idle(self):
        rec = Recorder()
        ka = KeepAwake(rec)
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
        ka = KeepAwake(rec)
        ka.acquire()
        self.assertTrue(wait_for(lambda: rec.calls == [ON]))
        ka.close()
        self.assertEqual(rec.calls, [ON, ES_CONTINUOUS])

    def test_a_failing_call_never_breaks_a_request(self):
        def boom(flags):
            raise OSError("no")
        ka = KeepAwake(boom)
        ka.acquire()
        ka.release()
        ka.close()

    def test_off_says_nothing(self):
        say = mock.Mock()
        self.assertIsNone(KeepAwake.create(False, say))
        say.assert_not_called()

    def test_linux_holds_with_systemd_inhibit(self):
        say = mock.Mock()
        with mock.patch("serve.keepawake.sys.platform", "linux"), \
                mock.patch("serve.keepawake.shutil.which",
                           lambda n: "/usr/bin/systemd-inhibit" if n == "systemd-inhibit" else None):
            ka = KeepAwake.create(True, say)                  # per request: no child until a request
        self.assertIsInstance(ka, ProcessKeepAwake)
        self.assertIsNone(ka._proc)
        self.assertIn("systemd-inhibit", ka._argv[0])
        self.assertIn("systemd-inhibit", say.call_args[0][0])
        ka.close()

    def test_linux_without_systemd_inhibit_says_it(self):
        say = mock.Mock()
        with mock.patch("serve.keepawake.sys.platform", "linux"), \
                mock.patch("serve.keepawake.shutil.which", lambda n: None):
            self.assertIsNone(KeepAwake.create(True, say))
        self.assertIn("ignored here", say.call_args[0][0])

    def test_macos_holds_with_caffeinate(self):
        say = mock.Mock()
        with mock.patch("serve.keepawake.sys.platform", "darwin"), \
                mock.patch("serve.keepawake.shutil.which",
                           lambda n: "/usr/bin/caffeinate" if n == "caffeinate" else None):
            ka = KeepAwake.create(True, say)
        self.assertIsInstance(ka, ProcessKeepAwake)
        self.assertIn("caffeinate", ka._argv[0])
        ka.close()

    def test_process_keeper_starts_with_a_request_and_stops_when_idle(self):
        ka = ProcessKeepAwake([sys.executable, "-c", "import time; time.sleep(60)"])
        ka.acquire()
        proc = ka._proc
        self.assertIsNotNone(proc)
        time.sleep(0.2)
        self.assertIsNone(proc.poll())
        ka.release()
        self.assertIsNotNone(proc.poll())                     # stopped when nothing needs it
        ka.close()

    def test_a_session_hold_is_not_dropped_by_a_request(self):
        ka = ProcessKeepAwake([sys.executable, "-c", "import time; time.sleep(60)"])
        ka.hold()
        proc = ka._proc
        ka.acquire()
        ka.release()
        time.sleep(0.2)
        self.assertIsNone(proc.poll())                        # the session hold stays
        ka.close()
        self.assertIsNotNone(proc.poll())

    def test_session_hold_keeps_through_requests_and_clears_at_close(self):
        rec = Recorder()
        say = mock.Mock()
        with mock.patch("serve.keepawake.sys.platform", "win32"), \
                mock.patch("serve.keepawake._windows_call", lambda: rec):
            ka = KeepAwake.create(True, say, session=True, label="keep_awake")
        self.assertIsNotNone(ka)
        self.assertTrue(wait_for(lambda: rec.calls == [ON]))
        self.assertIn("keep_awake", say.call_args[0][0])
        ka.acquire()                                          # a request comes ...
        ka.release()                                          # ... and goes: the session hold stays
        time.sleep(0.1)
        self.assertEqual(rec.calls, [ON])
        ka.close()
        self.assertEqual(rec.calls[-1], ES_CONTINUOUS)

    def test_service_holds_for_the_whole_request_including_early_close(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        rec = Recorder()
        svc.keep_awake = ka = KeepAwake(rec)
        kinds = [k for k, _ in svc.run(tok.encode("hi"), False, None, 8, {}, threading.Event())]
        self.assertIn("done", kinds)
        self.assertTrue(wait_for(lambda: rec.calls == [ON, ES_CONTINUOUS]))
        gen = svc.run(tok.encode("hi"), False, None, 8, {}, threading.Event())
        next(gen)
        self.assertTrue(wait_for(lambda: rec.calls[-1] == ON))
        gen.close()                                           # the client left
        self.assertTrue(wait_for(lambda: rec.calls[-1] == ES_CONTINUOUS))
        ka.close()

    def test_setup_keep_awake_obeys_the_env(self):
        import setup as strata_setup
        strata_setup._KEEP_AWAKE = None
        strata_setup._KEEP_AWAKE_TRIED = False
        try:
            with mock.patch.dict(os.environ, {"STRATA_KEEP_AWAKE": "0"}):
                self.assertIsNone(strata_setup.keep_awake_for_setup())
            strata_setup._KEEP_AWAKE_TRIED = False
            with mock.patch("serve.keepawake.KeepAwake.create") as create:
                create.return_value = "keeper"
                with mock.patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("STRATA_KEEP_AWAKE", None)
                    self.assertEqual(strata_setup.keep_awake_for_setup(), "keeper")
            create.assert_called_once()
            self.assertTrue(create.call_args.kwargs["session"])
            self.assertEqual(create.call_args.kwargs["label"], "keep_awake")
        finally:
            strata_setup._KEEP_AWAKE = None
            strata_setup._KEEP_AWAKE_TRIED = False

    def test_default_service_has_none(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.assertIsNone(svc.keep_awake)


if __name__ == "__main__":
    unittest.main()
