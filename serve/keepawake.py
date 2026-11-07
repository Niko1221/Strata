"""Opt-in "prevent_sleep" (#1727): while at least one request is running or queued, the PC is kept awake.
"keep_awake" holds the same state for the whole serving session, from the server's start until it exits.
Windows: SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) keeps the system from idling into sleep
(not the display: no ES_DISPLAY_REQUIRED); ES_CONTINUOUS alone puts the normal behaviour back.  The state belongs
to the thread that set it, so one daemon thread sets and clears it; it is cleared when idle and on exit.
Linux: a systemd-inhibit child (`--what=sleep:idle`, `--mode=block`, `cat` on a pipe kept open) holds the logind
inhibitor lock while it runs; stopping the child releases it.  A system without it (or without logind) gets one
startup line and nothing is held.
macOS: a caffeinate child (`-i`) holds the assertion while it runs; stopping it releases it.
Elsewhere this does nothing.  The keeper is best effort: it never fails a request.
"""
from __future__ import annotations

import atexit
import shutil
import subprocess
import sys
import threading
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _windows_call():
    import ctypes
    fn = ctypes.windll.kernel32.SetThreadExecutionState
    fn.argtypes = [ctypes.c_uint32]
    fn.restype = ctypes.c_uint32
    return fn


class KeepAwake:
    """Windows: SetThreadExecutionState on one daemon thread (the state belongs to the thread that set it)."""

    def __init__(self, call=None):
        self._call = call
        self._count = 0
        self._session = False
        self._cond = threading.Condition()
        self._closed = False
        self._thread = None

    @classmethod
    def create(cls, enabled: bool, say=print, session: bool = False, label: str = "prevent_sleep"):
        """The keeper for this system, or None when off (or the system has no way to hold the state: one line says
        so).  `session` (the config's "keep_awake"): the state is held from here until close(), not only while a
        request runs; a request's release() never drops it."""
        if not enabled:
            return None
        if sys.platform.startswith("win"):
            try:
                ka = cls(_windows_call())
            except (ImportError, AttributeError, OSError) as e:
                say(f'[strata] "{label}" could not be set up: {e}', flush=True)
                return None
            how = "Windows"
        else:
            ka = _platform_keeper(say, label)
            if ka is None:
                return None
            how = "systemd-inhibit" if sys.platform.startswith("linux") else "caffeinate"
        atexit.register(ka.close)
        if session:
            ka.hold()
            say(f"[strata] {label}: the system stays awake ({how}) while the server runs", flush=True)
        else:
            say(f"[strata] {label}: the system stays awake ({how}) while a request is running or queued", flush=True)
        return ka

    def _loop(self):
        held = False
        while True:
            with self._cond:
                while not self._closed and ((self._count > 0) or self._session) == held:
                    self._cond.wait()
                if self._closed:
                    break
                want = self._count > 0 or self._session
            try:
                self._call(ES_CONTINUOUS | ES_SYSTEM_REQUIRED if want else ES_CONTINUOUS)
                held = want
            except Exception:         # noqa: BLE001  never fail a request over this
                held = want
        try:
            self._call(ES_CONTINUOUS)
        except Exception:             # noqa: BLE001
            pass

    def _start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="keep-awake")
            self._thread.start()

    def hold(self):
        """The whole session (the config's "keep_awake"): held until close(), whatever the requests do."""
        with self._cond:
            if self._closed or self._session:
                return
            self._session = True
            self._start()
            self._cond.notify_all()

    def acquire(self):
        with self._cond:
            if self._closed:
                return
            self._count += 1
            self._start()
            self._cond.notify_all()

    def release(self):
        with self._cond:
            self._count = max(0, self._count - 1)
            self._cond.notify_all()

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2)


class ProcessKeepAwake:
    """Linux and macOS: a child process holds the inhibitor while it runs (systemd-inhibit, caffeinate).

    The child lives while a request runs or the session is held, and is stopped when neither is true (or at exit).
    A child that cannot start, or exits at once (no logind, no D-Bus), is said once and never retried."""

    def __init__(self, argv, say=print, label="keep_awake"):
        self._argv = list(argv)
        self._say = say
        self._label = label
        self._lock = threading.Lock()
        self._count = 0
        self._session = False
        self._proc = None
        self._disabled = False

    def _wanted(self) -> bool:
        return self._count > 0 or self._session

    def _ensure(self):
        if self._disabled or (self._proc is not None and self._proc.poll() is None):
            return
        try:
            self._proc = subprocess.Popen(self._argv, stdin=subprocess.PIPE,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as e:
            self._disabled = True
            self._say(f"[strata] {self._label}: could not start {self._argv[0]} ({e}); the PC may sleep", flush=True)
            return
        time.sleep(0.05)              # a refusal (no logind, no bus) shows at once
        if self._proc.poll() is not None:
            self._disabled = True
            self._close_stdin(self._proc)
            self._proc = None
            self._say(f"[strata] {self._label}: {self._argv[0]} refused to hold the lock; the PC may sleep", flush=True)

    @staticmethod
    def _close_stdin(p):
        try:
            if p.stdin is not None:
                p.stdin.close()
        except OSError:
            pass

    def _drop(self):
        p, self._proc = self._proc, None
        if p is not None:
            if p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        p.kill()
                    except OSError:
                        pass
            self._close_stdin(p)

    def _sync(self):
        if self._wanted():
            self._ensure()
        else:
            self._drop()

    def hold(self):
        """The whole session (the config's "keep_awake"): held until close(), whatever the requests do."""
        with self._lock:
            self._session = True
            self._sync()

    def acquire(self):
        with self._lock:
            self._count += 1
            self._sync()

    def release(self):
        with self._lock:
            self._count = max(0, self._count - 1)
            self._sync()

    def close(self):
        with self._lock:
            self._session = False
            self._count = 0
            self._drop()


def _platform_keeper(say, label):
    """Linux and macOS: the child that holds the inhibitor, or None with one line when the system has no way to."""
    if sys.platform.startswith("linux"):
        exe = shutil.which("systemd-inhibit")
        if not exe:
            say(f'[strata] "{label}" needs systemd-inhibit (Linux) or SetThreadExecutionState (Windows): '
                "ignored here", flush=True)
            return None
        return ProcessKeepAwake([exe, "--what=sleep:idle", "--why=Strata: the model is serving",
                                 "--mode=block", "cat"], say, label)
    if sys.platform == "darwin":
        exe = shutil.which("caffeinate")
        if not exe:
            say(f'[strata] "{label}" needs caffeinate (macOS) or SetThreadExecutionState (Windows): '
                "ignored here", flush=True)
            return None
        return ProcessKeepAwake([exe, "-i"], say, label)
    say(f'[strata] "{label}" is not available on this system: ignored here', flush=True)
    return None
