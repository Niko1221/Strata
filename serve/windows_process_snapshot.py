"""Bounded read-only Windows x64 process-counter snapshots.

One NtQuerySystemInformation(SystemProcessInformation) call replaces hundreds of
per-process handle queries. No command lines, executable paths or window titles
are collected. All pointer-like fields are checked as offsets into the returned
buffer before reading; none is dereferenced through ctypes.

API and format sources (layout facts, no third-party implementation copied):
https://learn.microsoft.com/en-us/windows/win32/api/winternl/nf-winternl-ntquerysysteminformation
https://github.com/giampaolo/psutil/blob/master/psutil/arch/windows/ntextapi.h
The latter cross-checks the public NT ABI against System Informer's phnt. This
API may change: unsupported platforms, malformed buffers and missing counters
raise an error, allowing the caller's conservative psutil fallback.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
import platform
import struct
import time
from types import SimpleNamespace

HEADER_BYTES = 256
MAX_BUFFER_BYTES = 16 * 1024 * 1024
FILETIME_EPOCH = 11644473600


@dataclass(frozen=True)
class ProcessSnapshot:
    processes: dict
    wall_time: float
    monotonic_time: float
    scan_seconds: float


def decode_process_snapshot(raw: bytes, base: int, max_processes: int = 4096) -> dict:
    """Decode x64 SYSTEM_PROCESS_INFORMATION after validating every read span."""
    if not raw or len(raw) > MAX_BUFFER_BYTES or base <= 0 or max_processes < 1:
        raise ValueError("invalid process snapshot envelope")
    processes, offset = {}, 0
    while True:
        if offset + HEADER_BYTES > len(raw) or len(processes) >= max_processes:
            raise ValueError("process snapshot exceeds count/header bounds")
        step = struct.unpack_from("<I", raw, offset)[0]
        created, user, kernel = struct.unpack_from("<qqq", raw, offset + 32)
        length, maximum = struct.unpack_from("<HH", raw, offset + 56)
        pointer = struct.unpack_from("<Q", raw, offset + 64)[0]
        pid, parent = struct.unpack_from("<QQ", raw, offset + 80)
        rss = struct.unpack_from("<Q", raw, offset + 144)[0]
        if pid in processes or pid > 0xffffffff or parent > 0xffffffff or min(created, user, kernel) < 0:
            raise ValueError("invalid or duplicate process identity/counter")
        if length % 2 or maximum % 2 or length > maximum:
            raise ValueError("invalid process name length")
        if length:
            relative = pointer - base
            if relative < 0 or relative + length > len(raw):
                raise ValueError("process name outside snapshot buffer")
            name = raw[relative:relative + length].decode("utf-16-le", errors="strict")
            if "\0" in name:
                raise ValueError("embedded process name terminator")
        else:
            name = "System Idle Process" if pid == 0 else ""
        processes[pid] = {"pid": pid, "ppid": parent, "name": name,
                          "create_time": created / 10_000_000 - FILETIME_EPOCH,
                          # Preserve exact 100ns identity; floating epoch seconds
                          # otherwise round distinct creation values together.
                          "create_time_ticks": created,
                          "cpu_times": SimpleNamespace(user=user / 10_000_000, system=kernel / 10_000_000),
                          "memory_info": SimpleNamespace(rss=rss)}
        if not step:
            return processes
        if step < HEADER_BYTES or step % 8 or offset + step > len(raw) - HEADER_BYTES:
            raise ValueError("invalid next process offset")
        offset += step


class WindowsProcessSnapshot:
    def __init__(self):
        if (os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8
                or platform.machine().lower() not in ("amd64", "x86_64")):
            raise OSError("Windows x64 process snapshot unavailable")
        query = ctypes.WinDLL("ntdll").NtQuerySystemInformation
        query.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,
                          ctypes.POINTER(ctypes.c_uint32)]
        query.restype = ctypes.c_int32
        self._query = query
        self._buffer = ctypes.create_string_buffer(256 * 1024)

    def sample(self, max_processes=4096, max_seconds=.75) -> ProcessSnapshot:
        started = time.monotonic()
        for _ in range(5):
            length = ctypes.c_uint32()
            before = time.monotonic()
            wall = time.time()
            status = self._query(5, self._buffer, len(self._buffer), ctypes.byref(length))
            after = time.monotonic()
            if after - started > max_seconds:
                raise TimeoutError("native process snapshot exceeded scan budget")
            unsigned = status & 0xffffffff
            if unsigned in (0xc0000004, 0xc0000023):  # LENGTH_MISMATCH / BUFFER_TOO_SMALL
                size = max(length.value + 64 * 1024, len(self._buffer) * 2)
                if size > MAX_BUFFER_BYTES:
                    raise ValueError("native process snapshot exceeds buffer cap")
                self._buffer = ctypes.create_string_buffer(size)
                continue
            if status != 0:
                raise OSError(f"NtQuerySystemInformation failed: 0x{unsigned:08x}")
            if not HEADER_BYTES <= length.value <= len(self._buffer):
                raise ValueError("invalid native process snapshot length")
            # Copy only returned bytes. Names remain offsets into this private
            # copy, so malformed native pointers cannot make arbitrary reads.
            raw = self._buffer.raw[:length.value]
            records = decode_process_snapshot(raw, ctypes.addressof(self._buffer), max_processes)
            if os.getpid() not in records:
                raise ValueError("native process snapshot omitted the sampler process")
            finished = time.monotonic()
            if finished - started > max_seconds:
                raise TimeoutError("native process decoding exceeded scan budget")
            middle = before + (after - before) / 2
            return ProcessSnapshot(records, wall + middle - before, middle, finished - started)
        raise RuntimeError("native process snapshot changed size too often")
