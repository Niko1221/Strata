"""Opt-in "prevent_sleep" (#1727): while at least one request is running or queued, the PC is kept awake.

Windows and macOS.  On a Mac the hold is a `caffeinate -i -w <this pid>` child (the system's own tool: no idle system sleep; the
display may still turn off), started when the first request comes and ended when the last one is done; it also ends by itself if
the server dies.  A Mac on battery otherwise idle-sleeps in the middle of a long request (seen: each reply took 550 or 1,050 s
instead of 50 s, in steps of one ~500 s maintenance sleep).

On Windows SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) keeps the system from idling into sleep
(not the display: no ES_DISPLAY_REQUIRED); ES_CONTINUOUS alone puts the normal behaviour back.  The state belongs
to the thread that set it, so one daemon thread sets and clears it; it is cleared when idle and on exit.
Elsewhere this does nothing.
"""
from __future__ import annotations

import atexit
import sys
import threading

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _windows_call():
    import ctypes
    fn = ctypes.windll.kernel32.SetThreadExecutionState
    fn.argtypes = [ctypes.c_uint32]
    fn.restype = ctypes.c_uint32
    return fn


def _mac_call(run=None):
    """The same call shape as Windows' (flags in, 0 out): ES_SYSTEM_REQUIRED starts the caffeinate child, its absence ends it."""
    import os
    import subprocess
    exe = "/usr/bin/caffeinate"
    if run is None and not os.path.exists(exe):
        raise OSError(f"{exe} is missing")
    start = run or (lambda: subprocess.Popen([exe, "-i", "-w", str(os.getpid())], stdin=subprocess.DEVNULL,
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    state = {"child": None}

    def call(flags: int) -> int:
        child = state["child"]
        if flags & ES_SYSTEM_REQUIRED:
            if child is None or child.poll() is not None:
                state["child"] = start()
        elif child is not None:
            state["child"] = None
            try:
                child.terminate()
                child.wait(timeout=2)
            except Exception:         # noqa: BLE001
                pass
        return 0
    return call


class KeepAwake:
    def __init__(self, call=None):
        self._call = call
        self._count = 0
        self._cond = threading.Condition()
        self._closed = False
        self._thread = None

    @classmethod
    def create(cls, enabled: bool, say=print):
        """None when off.  On a system without SetThreadExecutionState: one line, and None."""
        if not enabled:
            return None
        mac = sys.platform == "darwin"
        if not mac and not sys.platform.startswith("win"):
            say('[strata] "prevent_sleep" is for Windows and macOS only: ignored here', flush=True)
            return None
        try:
            ka = cls(_mac_call() if mac else _windows_call())
        except (ImportError, AttributeError, OSError) as e:
            say(f'[strata] "prevent_sleep" could not be set up: {e}', flush=True)
            return None
        atexit.register(ka.close)
        say(f'[strata] prevent_sleep: {"the Mac" if mac else "Windows"} stays awake while a request is running or queued', flush=True)
        return ka

    def _loop(self):
        held = False
        while True:
            with self._cond:
                while not self._closed and (self._count > 0) == held:
                    self._cond.wait()
                if self._closed:
                    break
                want = self._count > 0
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
