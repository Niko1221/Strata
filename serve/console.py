"""Best-effort diagnostics with one daemon writer and bounded pending storage.

The 64-KiB/128-record budget includes coalesced progress, excluding the one
in-flight record. Each record retains its owned destination. Invalidation uses
key identity and cannot retract output already handed to the sink.
"""
import atexit
from collections import deque
import io
import os
import sys
import threading


class _Sink:
    """Own an unbuffered duplicate, independent of the original stream's life."""

    def __init__(self, stream):
        fd = os.dup(stream.fileno())
        try:
            os.set_inheritable(fd, False)
            # Windows selects _WindowsConsoleIO for a console, FileIO for a
            # redirected file/pipe. Both own fd and have no buffered shutdown.
            self.raw = open(fd, "wb", buffering=0)
        except BaseException:
            os.close(fd)
            raise
        self.encoding = getattr(stream, "encoding", None) or "utf-8"
        if os.name == "nt" and not isinstance(self.raw, io.FileIO):
            self.encoding = "utf-8"

    def __del__(self):
        # Last cache/record reference releases the raw owner, never a borrowed
        # descriptor. Timed-out close leaves the in-flight reference intact.
        try:
            self.raw.close()
        except (AttributeError, OSError):
            pass

    def write(self, text):
        if os.name == "nt":
            text = text.replace("\n", "\r\n")
        data = text.encode(self.encoding, errors="replace")
        view = memoryview(data)
        while view:
            written = self.raw.write(view)
            if written is None or written <= 0:
                raise OSError("diagnostic write made no progress")
            view = view[written:]


class Console:
    def __init__(self):
        self._condition = threading.Condition()
        self._queue = deque()
        self._progress = None
        self._bytes = 0
        self._closed = False
        self._dropped = 0
        # Identity caches retain the stream to prevent id reuse. Records keep
        # replaced sinks alive; only the owned raw object closes its descriptor.
        self._sinks = [None, None]
        self._worker = threading.Thread(target=self._run, name="console-writer",
                                        daemon=True)
        self._worker.start()

    @staticmethod
    def _bounded(text):
        data = str(text)[:4097].encode("utf-8", "replace")
        if len(data) <= 4096:
            return data.decode("utf-8")
        return data[:4092].decode("utf-8", "ignore") + "...\n"

    def emit(self, text, progress_key=None, *, stderr=False):
        """Admit bounded text without waiting for writes; return admission."""
        text = self._bounded(text)
        size = len(text.encode("utf-8"))
        with self._condition:
            if self._closed:
                return False
            previous = self._progress if progress_key is not None else None
            count = len(self._queue) + (self._progress is not None)
            new_bytes = self._bytes + size - (previous[1] if previous else 0)
            if count + (previous is None) > 128 or new_bytes > 65536:
                self._dropped += 1
                return False
            stream = sys.stderr if stderr else sys.stdout
            cached = self._sinks[bool(stderr)]
            if cached is None or cached[0] is not stream:
                # Capture failure must not route to the previous destination.
                self._sinks[bool(stderr)] = None
                try:
                    sink = _Sink(stream)
                except (AttributeError, OSError, ValueError):
                    return False
                self._sinks[bool(stderr)] = (stream, sink)
            else:
                sink = cached[1]
            record = (text, size, progress_key, sink)
            if progress_key is None:
                self._queue.append(record)
            else:
                self._progress = record
            self._bytes = new_bytes
            self._condition.notify()
            return True

    def discard_progress(self, key):
        with self._condition:
            if self._progress is not None and self._progress[2] is key:
                self._bytes -= self._progress[1]
                self._progress = None

    def close(self, timeout=.2):
        with self._condition:
            self._closed = True
            self._sinks = [None, None]
            self._condition.notify()
        if threading.current_thread() is not self._worker:
            self._worker.join(max(0, timeout))
        return not self._worker.is_alive()

    def _run(self):
        while True:
            with self._condition:
                while not self._queue and self._progress is None:
                    if self._closed:
                        return
                    self._condition.wait()
                if self._queue:
                    record = self._queue.popleft()
                else:
                    record, self._progress = self._progress, None
                self._bytes -= record[1]
                dropped, self._dropped = self._dropped, 0
            # Neither sink writes nor the best-effort overflow report hold the
            # queue lock. Failure is local; the next record still gets a try.
            try:
                record[3].write(record[0])
                if dropped:
                    record[3].write(f"[console: dropped {dropped} diagnostics]\n")
            except (OSError, ValueError):
                pass
            del record


_console = None
_console_lock = threading.Lock()
_shutdown = False


def _get_console():
    global _console
    with _console_lock:
        if _console is None and not _shutdown:
            _console = Console()
        return _console


def console_print(*args, stderr=False):
    """Publish a space-separated diagnostic line to current stdout or stderr."""
    console = _get_console()
    if console is not None:
        console.emit(" ".join(map(str, args)) + "\n", stderr=stderr)


def progress(text, key):
    """Publish progress with a stable, non-None opaque request identity token."""
    if key is None:
        raise ValueError("progress key must not be None")
    console = _get_console()
    if console is not None:
        text = str(text)
        console.emit(text if text.endswith("\n") else text + "\n", progress_key=key)


def invalidate_progress(key):
    if _console is not None:
        _console.discard_progress(key)


def shutdown(timeout=.2):
    global _shutdown
    with _console_lock:
        _shutdown = True
        console = _console
    return console.close(timeout) if console is not None else True


atexit.register(shutdown)
