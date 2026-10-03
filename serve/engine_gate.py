"""One engine owner, FIFO waiters, and one overdue parked request's next period."""
from collections import deque
from contextlib import contextmanager
import threading
import time


class EngineGate:
    def __init__(self):
        self._cv = threading.Condition()
        self._held = False
        self._waiters = deque()

    def acquire(self, blocking=True, timeout=-1, *, priority=False):
        if not blocking and timeout != -1:
            raise ValueError("can't specify a timeout for a non-blocking call")
        deadline = None if timeout < 0 else time.monotonic() + timeout
        with self._cv:
            if not blocking:
                if self._held or self._waiters:
                    return False
                self._held = True
                return True
            ticket = object()
            # Only the single parked request may request priority, after its wait deadline.
            # It still waits for the current owner to release; decode is never interrupted.
            if priority:
                self._waiters.appendleft(ticket)
            else:
                self._waiters.append(ticket)
            try:
                while self._held or self._waiters[0] is not ticket:
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        return False
                    self._cv.wait(remaining)
                self._waiters.popleft()
                self._held = True
                return True
            finally:
                if ticket in self._waiters:
                    self._waiters.remove(ticket)
                    self._cv.notify_all()

    def release(self):
        with self._cv:
            if not self._held:
                raise RuntimeError("release unlocked engine gate")
            self._held = False
            self._cv.notify_all()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()

    @contextmanager
    def period(self, *, priority=False):
        self.acquire(priority=priority)
        try:
            yield self
        finally:
            self.release()
