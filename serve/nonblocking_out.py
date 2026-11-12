"""#4479 / #2481: the server's console output must not hold a request.

A console that stops taking output (a Windows console in QuickEdit/selection mode, a terminal that is paused, a launcher
that stopped reading the server's pipe) blocks `print()` in whatever thread called it - and some of those threads own the
engine's turn, a cancel, or an admission.  `NonBlockingOut` stands in for sys.stdout / sys.stderr while the server runs:
write() puts the text on a bounded queue and returns; one daemon thread writes it to the real stream, and may block there
as long as it likes.  When the queue is full the OLDEST text is dropped (the newest lines matter most) and counted in
`dropped_chunks` / `dropped_bytes`; nothing about a request waits for the console.

Only diagnostics go through here.  Protocol output (the engine's pipe, the HTTP replies) does not.
A finished line that starts with [strata] gets a UTC ISO-8601 time in front of its text.  A line that already
has one is left as it is.  There is no switch for this.  Other text is not changed.
"""
from __future__ import annotations

import collections
import contextlib
import datetime
import re
import sys
import threading

# A wall clock the engine already wrote after "[strata]" (local YYYY-MM-DD or UTC ISO-8601).
_ALREADY_STAMPED = re.compile(
    r"(?:\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"
    r"|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)"
)

_MAX_PARTIAL = 1 << 16                                   # an unfinished [strata] line with no "\n" yet
_STRATA_TAG = "[strata]"


def utc_iso_timestamp() -> str:
    """UTC ISO-8601 with millisecond precision, Z suffix (diagnostics exports)."""
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def stamp_strata_line(line: str, now: datetime.datetime | None = None) -> str:
    """Prefix one complete server diagnostic line with a wall clock when it starts with [strata]."""
    if not line.startswith("[strata]"):
        return line
    rest = line[len("[strata]"):].lstrip()
    if _ALREADY_STAMPED.match(rest):
        return line
    ts = (
        now.astimezone(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        if now is not None
        else utc_iso_timestamp()
    )
    return f"[strata] {ts} {rest}"


def _hold_strata_line_at_bol(s: str) -> bool:
    """At beginning-of-line, keep buffering until we know this is not a [strata] diagnostic line."""
    if not s:
        return False
    if s.startswith(_STRATA_TAG):
        return True
    return _STRATA_TAG.startswith(s)


class NonBlockingOut:
    def __init__(self, real, max_bytes: int = 1 << 20, max_chunks: int = 4000):
        self._real = real
        self.max_bytes, self.max_chunks = max_bytes, max_chunks
        self._q: collections.deque[str] = collections.deque()
        self._bytes = 0
        self._cv = threading.Condition()
        self._closing = False
        self._idle = True
        self._partial = ""                                   # start-of-line buffer (the writer thread and _cv only)
        self._at_line_start = True                           # after \n or init; a mid-line non-strata stream goes through
        self.dropped_chunks = 0
        self.dropped_bytes = 0
        self._thread = threading.Thread(target=self._run, name="strata-console", daemon=True)
        self._thread.start()

    def _drop_oldest_locked(self) -> None:
        while self._q and (self._bytes > self.max_bytes or len(self._q) > self.max_chunks):
            old = self._q.popleft()
            self._bytes -= len(old)
            self.dropped_chunks += 1
            self.dropped_bytes += len(old)

    def _enqueue_raw_locked(self, text: str) -> None:
        if not text:
            return
        self._q.append(text)
        self._bytes += len(text)
        self._drop_oldest_locked()

    def _emit_stamped_line_locked(self, line: str, now: datetime.datetime | None) -> None:
        stamped = stamp_strata_line(line, now=now) + "\n"
        self._enqueue_raw_locked(stamped)

    def _emit_complete_line_locked(self, line: str) -> None:
        if line.startswith("[strata]"):
            if line.endswith("\r"):
                line = line[:-1]
            self._emit_stamped_line_locked(line, None)
        else:
            self._enqueue_raw_locked(line + "\n")
        self._at_line_start = True

    def _flush_partial_locked(self) -> None:
        if not self._partial:
            return
        p = self._partial
        self._partial = ""
        if p.startswith("[strata]"):
            line = p.rstrip("\r")
            if line:
                self._emit_stamped_line_locked(line, None)
            self._at_line_start = True
        else:
            self._enqueue_raw_locked(p)
            self._at_line_start = "\n" not in p

    def _ingest_stamped_locked(self, text: str) -> None:
        if not self._at_line_start:
            cut = text.find("\n")
            if cut < 0:
                self._enqueue_raw_locked(text)
                return
            self._enqueue_raw_locked(text[: cut + 1])
            text = text[cut + 1 :]
            self._at_line_start = True
            if not text:
                return
        self._partial += text
        while True:
            cut = self._partial.find("\n")
            if cut >= 0:
                line = self._partial[:cut]
                self._partial = self._partial[cut + 1:]
                self._emit_complete_line_locked(line)
                if not self._partial:
                    return
                continue
            if _hold_strata_line_at_bol(self._partial):
                if len(self._partial) > _MAX_PARTIAL:
                    self._flush_partial_locked()
                return
            self._enqueue_raw_locked(self._partial)
            self._partial = ""
            self._at_line_start = False
            return

    # ---- what the program calls: never blocks on the console
    def write(self, text) -> int:
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return 0
        with self._cv:
            if self._closing:
                return len(text)
            self._ingest_stamped_locked(text)
            self._cv.notify()
        return len(text)

    def flush(self) -> None:
        return None                                    # the writer thread flushes the real stream

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        try:
            return bool(self._real.isatty())
        except (OSError, ValueError, AttributeError):
            return False

    def __getattr__(self, name):                        # encoding, errors, fileno, buffer ...: the real stream's
        return getattr(self._real, name)

    # ---- the one thread that may block
    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._q and not self._closing:
                    self._idle = True
                    self._cv.notify_all()
                    self._cv.wait()
                if not self._q:
                    self._idle = True
                    self._cv.notify_all()
                    return
                self._idle = False
                text = self._q.popleft()
                self._bytes -= len(text)
            try:
                self._real.write(text)
                self._real.flush()
            except (OSError, ValueError, UnicodeError):
                pass                                    # a closed or broken console: the text is lost, the server is not held

    def drain(self, timeout: float = 2.0) -> bool:
        """Wait (bounded) until everything queued has been handed to the real stream."""
        import time
        end = time.monotonic() + timeout
        with self._cv:
            while time.monotonic() < end:
                if self._partial:
                    self._flush_partial_locked()
                    self._cv.notify()
                if not self._q and self._idle and not self._partial:
                    return True
                if not self._q and not self._idle:
                    self._cv.wait(timeout=max(0.01, end - time.monotonic()))
                elif self._q or self._partial:
                    self._cv.wait(timeout=max(0.01, end - time.monotonic()))
                else:
                    return True
            return not self._q and self._idle and not self._partial

    def close(self, timeout: float = 2.0) -> None:
        with self._cv:
            if self._partial:
                self._flush_partial_locked()
                self._cv.notify_all()
        self.drain(timeout)
        with self._cv:
            self._closing = True
            self._cv.notify_all()


@contextlib.contextmanager
def decoupled_console():
    """While the server runs: stdout and stderr go through NonBlockingOut, when they are the process's own streams (a
    test runner's capture, or a stream somebody else replaced, is left alone).  Restored, and drained for a moment, at
    the end.  A finished [strata] line gets a UTC ISO-8601 wall clock.  Other bytes are unchanged."""
    installed = []
    for name in ("stdout", "stderr"):
        cur = getattr(sys, name, None)
        if cur is not None and cur is getattr(sys, "__" + name + "__", None):
            wrapper = NonBlockingOut(cur)
            setattr(sys, name, wrapper)
            installed.append((name, cur, wrapper))
    try:
        yield [w for _, _, w in installed]
    finally:
        for name, orig, wrapper in installed:
            wrapper.close()
            if getattr(sys, name, None) is wrapper:
                setattr(sys, name, orig)
