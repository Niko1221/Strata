# Console diagnostics

Request execution, recovery and shutdown publish diagnostics to one process-owned
writer. A paused terminal or full output pipe does not make those operations wait
for a diagnostic write. Configuration helpers retain synchronous output so their
callers can capture notes immediately. Engine protocol traffic, HTTP responses,
model tokens and structured lifecycle events keep their existing channels. The
human-readable ready banner is best-effort; automation should use the ready file,
structured lifecycle events or HTTP readiness checks.

The writer keeps at most 128 pending records and 64 KiB of pending UTF-8 text,
plus one record being written. Each record is limited to 4 KiB. Longer messages
end with a truncation marker. Progress keeps only the latest pending sample;
request cleanup removes its own sample. A line already being written can still
appear after that request ends. When the queue is full, new diagnostics are
dropped. An overflow count is reported best-effort when output resumes.

Each standard output destination is held through an independent, non-inheritable
file descriptor. Queued records retain their destination until consumed or
discarded, so closing the original stream does not redirect delayed writes to a
reused descriptor. The two current destinations are cached by stream identity;
stable stdout/stderr need at most two owned descriptors. Replacing a destination
does not close one still retained by queued output. A failed destination does not
disable a later healthy destination.

The writer uses unbuffered output, preserving Windows console Unicode and
platform-default newlines. It does not hold Python's standard buffered-stream
lock during writes. Shutdown waits at most 0.2 seconds for the writer; remaining
diagnostics may be lost. Cached destinations can stay open until replaced or the
writer exits. Custom stream modes and in-place descriptor rebinding are outside
this internal stdout/stderr interface.

`before_load` output goes to a temporary file, and its last 3072 bytes are
submitted when the hook ends. Its exit-code, timeout and strict-mode policy stay
the same. Capture failures do not replace the hook outcome. The temporary file
can grow with the hook's output, and hooks see a file on stdout/stderr.

Configured third-party logging handlers and commands opening their own console
remain separate output paths.
