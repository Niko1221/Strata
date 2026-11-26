"""A private nested Windows Job; assign a suspended child before it can run."""
from __future__ import annotations

import asyncio
import ctypes as c
from ctypes import wintypes as w
import os
from pathlib import Path
import socket
import subprocess
import time


class JobError(RuntimeError):
    pass


class _Security(c.Structure):
    _fields_ = [("size", w.DWORD), ("descriptor", w.LPVOID), ("inherit", w.BOOL)]


class _Startup(c.Structure):
    _fields_ = [("size", w.DWORD), ("reserved", w.LPWSTR), ("desktop", w.LPWSTR),
                ("title", w.LPWSTR), ("x", w.DWORD), ("y", w.DWORD),
                ("x_size", w.DWORD), ("y_size", w.DWORD), ("x_chars", w.DWORD),
                ("y_chars", w.DWORD), ("fill", w.DWORD), ("flags", w.DWORD),
                ("show", w.WORD), ("reserved_size", w.WORD), ("reserved_data", c.POINTER(c.c_ubyte)),
                ("stdin", w.HANDLE), ("stdout", w.HANDLE), ("stderr", w.HANDLE)]


class _Process(c.Structure):
    _fields_ = [("process", w.HANDLE), ("thread", w.HANDLE), ("pid", w.DWORD), ("tid", w.DWORD)]


class _BasicLimits(c.Structure):
    _fields_ = [("process_time", c.c_int64), ("job_time", c.c_int64), ("flags", w.DWORD),
                ("min_working_set", c.c_size_t), ("max_working_set", c.c_size_t),
                ("active_process_limit", w.DWORD), ("affinity", c.c_size_t),
                ("priority", w.DWORD), ("scheduling", w.DWORD)]


class _IoCounters(c.Structure):
    _fields_ = [(key, c.c_uint64) for key in ("reads", "writes", "other", "read_bytes", "write_bytes", "other_bytes")]


class _ExtendedLimits(c.Structure):
    _fields_ = [("basic", _BasicLimits), ("io", _IoCounters), ("process_memory", c.c_size_t),
                ("job_memory", c.c_size_t), ("peak_process_memory", c.c_size_t), ("peak_job_memory", c.c_size_t)]


def _kernel():
    if os.name != "nt":
        raise JobError("The production carrier requires Windows")
    k = c.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateJobObjectW": ([w.LPVOID, w.LPCWSTR], w.HANDLE),
        "GetCurrentProcess": ([], w.HANDLE),
        "DuplicateHandle": ([w.HANDLE, w.HANDLE, w.HANDLE, c.POINTER(w.HANDLE), w.DWORD, w.BOOL, w.DWORD], w.BOOL),
        "SetInformationJobObject": ([w.HANDLE, c.c_int, w.LPVOID, w.DWORD], w.BOOL),
        "CreateFileW": ([w.LPCWSTR, w.DWORD, w.DWORD, c.POINTER(_Security), w.DWORD, w.DWORD, w.HANDLE], w.HANDLE),
        "CreateProcessW": ([w.LPCWSTR, w.LPWSTR, w.LPVOID, w.LPVOID, w.BOOL, w.DWORD,
                            w.LPVOID, w.LPCWSTR, c.POINTER(_Startup), c.POINTER(_Process)], w.BOOL),
        "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
        "ResumeThread": ([w.HANDLE], w.DWORD),
        "TerminateProcess": ([w.HANDLE, w.UINT], w.BOOL),
        "TerminateJobObject": ([w.HANDLE, w.UINT], w.BOOL),
        "WaitForSingleObject": ([w.HANDLE, w.DWORD], w.DWORD),
        "CloseHandle": ([w.HANDLE], w.BOOL),
        "QueryInformationJobObject": ([w.HANDLE, c.c_int, w.LPVOID, w.DWORD, c.POINTER(w.DWORD)], w.BOOL),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(k, name)
        fn.argtypes, fn.restype = args, result
    return k


def assert_port_unused(port: int) -> None:
    """Never use a listener which was present before our child started."""
    with socket.socket() as sock:
        if os.name == "nt":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise JobError("The private upstream port is already occupied") from exc


class WindowsJob:
    def __init__(self):
        self.k = _kernel()
        self.job, self.process, self.pid = None, None, None
        self.job = self.k.CreateJobObjectW(None, None)
        if not self.job:
            raise JobError("CreateJobObject failed")
        limits = _ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, no breakaway.
        if not self.k.SetInformationJobObject(self.job, 9, c.byref(limits), c.sizeof(limits)):
            self.k.CloseHandle(self.job)
            self.job = None
            raise JobError("SetInformationJobObject failed")

    def attach_stdio_child(self, child: subprocess.Popen) -> None:
        """Attach our newly created stdio child before supplying work; never adopt a listener."""
        if self.process is not None or child.poll() is not None:
            raise JobError("Expected one newly created live stdio child")
        duplicate = w.HANDLE()
        owner = self.k.GetCurrentProcess()
        # Popen's handle pins this exact child; duplication avoids PID reuse and double close.
        if not self.k.DuplicateHandle(owner, int(child._handle), owner, c.byref(duplicate), 0, False, 2):
            raise JobError("Cannot retain the owned stdio child handle")
        if not self.k.AssignProcessToJobObject(self.job, duplicate):
            self.k.CloseHandle(duplicate)
            raise JobError("The owned stdio child could not enter the private nested Job")
        self.process, self.pid = duplicate.value, child.pid

    def stop_and_verify(self, timeout: float = 5) -> None:
        """Stop only this Job's descendants and confirm none remain before a handoff."""
        if not self.job:
            raise JobError("Cannot verify a closed Job")
        if not self.k.TerminateJobObject(self.job, 0):
            raise JobError("The owned stdio Job could not be stopped")
        deadline = time.monotonic() + timeout
        while self.process_ids():
            if time.monotonic() >= deadline:
                raise JobError("Owned stdio descendants have not stopped")
            time.sleep(0.05)
        self.close()

    def start(self, argv: list[str], cwd: Path, log: Path) -> None:
        if self.process is not None:
            raise JobError("This job already owns a process")
        security = _Security(c.sizeof(_Security), None, True)
        invalid = c.c_void_p(-1).value
        output = self.k.CreateFileW(str(log), 4, 3, c.byref(security), 4, 0x80, None)  # FILE_APPEND_DATA
        source = self.k.CreateFileW("NUL", 0x80000000, 3, c.byref(security), 3, 0x80, None)
        info = _Process()
        try:
            if output in (None, invalid) or source in (None, invalid):
                raise JobError("Cannot open the private server log")
            startup = _Startup()
            startup.size, startup.flags = c.sizeof(startup), 0x100  # STARTF_USESTDHANDLES
            startup.stdin, startup.stdout, startup.stderr = source, output, output
            command = c.create_unicode_buffer(subprocess.list2cmdline(argv))
            # Inherit the parent's Job too. Windows 8+ supports nested Jobs.
            if not self.k.CreateProcessW(None, command, None, None, True, 0x08000004,
                                         None, str(cwd), c.byref(startup), c.byref(info)):
                raise JobError("CreateProcess failed")
            if not self.k.AssignProcessToJobObject(self.job, info.process):
                self.k.TerminateProcess(info.process, 1)
                self.k.WaitForSingleObject(info.process, 5000)
                raise JobError("The child could not enter the private nested Job")
            self.process, self.pid = info.process, int(info.pid)
            if self.k.ResumeThread(info.thread) == 0xFFFFFFFF:
                self.k.TerminateJobObject(self.job, 1)
                raise JobError("ResumeThread failed")
        finally:
            if info.thread:
                self.k.CloseHandle(info.thread)
            if info.process and self.process != info.process:
                self.k.CloseHandle(info.process)
            for handle in (output, source):
                if handle not in (None, invalid):
                    self.k.CloseHandle(handle)

    @property
    def alive(self) -> bool:
        return bool(self.process) and self.k.WaitForSingleObject(self.process, 0) == 258

    def process_ids(self) -> set[int]:
        if not self.job:
            return set()
        size = 4096
        while size <= 1 << 20:
            buffer = c.create_string_buffer(size)
            needed = w.DWORD()
            if self.k.QueryInformationJobObject(self.job, 3, buffer, size, c.byref(needed)):
                count = w.DWORD.from_buffer(buffer, 4).value
                ptr_size = c.sizeof(c.c_size_t)
                return {c.c_size_t.from_buffer(buffer, 8 + i * ptr_size).value for i in range(count)}
            if c.get_last_error() != 234:  # ERROR_MORE_DATA
                raise JobError("Cannot verify the private Job membership")
            size *= 2
        raise JobError("The private Job membership exceeds its limit")

    def listener_owned(self, port: int) -> bool:
        if not self.alive:
            return False
        api = c.WinDLL("iphlpapi", use_last_error=True)
        api.GetExtendedTcpTable.argtypes = [w.LPVOID, c.POINTER(w.DWORD), w.BOOL, w.ULONG, c.c_int, w.ULONG]
        api.GetExtendedTcpTable.restype = w.DWORD
        size = w.DWORD()
        result = api.GetExtendedTcpTable(None, c.byref(size), False, 2, 3, 0)
        if result not in (0, 122):  # TCP_TABLE_OWNER_PID_LISTENER, AF_INET
            raise JobError("Cannot inspect the private listener owner")
        buffer = c.create_string_buffer(size.value)
        if api.GetExtendedTcpTable(buffer, c.byref(size), False, 2, 3, 0):
            raise JobError("Cannot inspect the private listener owner")
        owners = self.process_ids()
        count = w.DWORD.from_buffer(buffer).value
        for i in range(count):
            row = (w.DWORD * 6).from_buffer(buffer, 4 + i * 24)
            if row[0] == 2 and socket.ntohs(row[2] & 0xFFFF) == port:
                # Refuse a foreign wildcard listener as well as a loopback one.
                address = socket.inet_ntoa(int(row[1]).to_bytes(4, "little"))
                if address in ("127.0.0.1", "0.0.0.0"):
                    return address == "127.0.0.1" and row[5] in owners
        return False

    def close(self) -> None:
        if self.job:
            if not self.k.TerminateJobObject(self.job, 0):
                raise JobError("The owned server Job could not be stopped")
            if self.process and self.k.WaitForSingleObject(self.process, 30000) != 0:
                raise JobError("The owned server did not stop")
            # The root may exit before its descendants release inherited handles.
            deadline = time.monotonic() + 5
            while self.process_ids():
                if time.monotonic() >= deadline:
                    raise JobError("Owned server descendants have not stopped")
                time.sleep(0.05)
            self.k.CloseHandle(self.job)
            self.job = None
        if self.process:
            self.k.CloseHandle(self.process)
            self.process = None
        self.pid = None


class WindowsServer:
    def __init__(self):
        self.job: WindowsJob | None = None
        self.port: int | None = None

    @property
    def alive(self) -> bool:
        return self.job is not None and self.job.alive

    @property
    def pid(self) -> int | None:
        return self.job.pid if self.job else None

    def listener_owned(self, port: int) -> bool:
        return self.job is not None and self.job.listener_owned(port)

    async def start(self, argv: list[str], cwd: Path, log: Path, port: int) -> None:
        if self.job is not None:
            raise JobError("An owned Job must stop before another starts")
        assert_port_unused(port)
        self.job = WindowsJob()
        launch = asyncio.create_task(asyncio.to_thread(self.job.start, argv, cwd, log))
        try:
            await asyncio.shield(launch)
            self.port = port
        except BaseException:
            # The Windows API thread must finish assignment before the Job closes.
            await asyncio.gather(launch, return_exceptions=True)
            await asyncio.to_thread(self.job.close)
            self.job = None
            raise

    def _close(self) -> None:
        self.job.close()
        if self.port is None:
            return
        # Windows can report WSAEADDRINUSE briefly after the owned Job exits.
        # Wait only at this shutdown boundary; never retry a spawn or weaken the
        # exclusive bind check that refuses a foreign listener before start.
        deadline = time.monotonic() + 5
        while True:
            try:
                assert_port_unused(self.port)
                return
            except JobError as exc:
                cause = exc.__cause__
                if (not isinstance(cause, OSError) or getattr(cause, "winerror", None) != 10048
                        or time.monotonic() >= deadline):
                    raise
                time.sleep(0.05)

    async def stop(self) -> None:
        if self.job is not None:
            closing = asyncio.create_task(asyncio.to_thread(self._close))
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError:
                # Do not let cleanup race a still-running handle-close thread.
                await closing
                self.job = None
                self.port = None
                raise
            self.job = None
            self.port = None
