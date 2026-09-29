"""Opt-in resource supervisor: python -m serve.resource_guard --help.

Owns only the process group/job it launches. Never searches for other servers.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

MIB = 1024 ** 2


@dataclass(frozen=True)
class Limits:
    ram_mib: float = 0
    vram_mib: float = 0
    interval: float = 1

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.ram_mib, self.vram_mib, self.interval)):
            raise ValueError("limits must be finite")
        if min(self.ram_mib, self.vram_mib) < 0 or self.interval <= 0:
            raise ValueError("floors must be non-negative and interval must be positive")
        if not (self.ram_mib or self.vram_mib):
            raise ValueError("enable at least one memory floor")

    def violation(self, available):
        for key, floor in (("RAM", self.ram_mib), ("VRAM", self.vram_mib)):
            if not floor:
                continue
            value = available.get(key)
            if value is None or not math.isfinite(value) or value < 0:
                return f"{key} telemetry unavailable"
            if value < floor:
                return f"{key} available {value:.1f} MiB is below {floor:g} MiB"
        return None


class MemoryProbe:
    def __init__(self, limits, gpu_index):
        from serve.telemetry import _CpuRamFallback, _Nvml
        self.limits = limits
        self.ram = _CpuRamFallback() if limits.ram_mib else None
        self.gpu = _Nvml(gpu_index) if limits.vram_mib else None
        try:
            import psutil
            self.ps = psutil
        except ImportError:
            self.ps = None

    def __call__(self):
        out = {}
        if self.ram is not None:
            if self.ps is not None:
                out["RAM"] = self.ps.virtual_memory().available / MIB
            else:
                used, total = self.ram.ram()
                out["RAM"] = (total - used) / MIB if total is not None and used is not None else None
        if self.gpu is not None and self.gpu.ok():
            memory = self.gpu.Mem()
            if self.gpu.lib.nvmlDeviceGetMemoryInfo(self.gpu.dev, ctypes.byref(memory)) == 0:
                out["VRAM"] = memory.free / MIB
        return out


def _kernel32():
    from ctypes import wintypes as w
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateJobObjectW": ([ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
        "SetInformationJobObject": ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
        "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
        "TerminateJobObject": ([w.HANDLE, w.UINT], w.BOOL),
        "GetCurrentProcess": ([], w.HANDLE),
        "CloseHandle": ([w.HANDLE], w.BOOL),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(k, name)
        fn.argtypes, fn.restype = args, result
    return k


class _WindowsJob:
    def __init__(self):
        from ctypes import wintypes as w

        class Basic(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                        ("flags", w.DWORD), ("min_working_set", ctypes.c_size_t),
                        ("max_working_set", ctypes.c_size_t), ("active_limit", w.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]

        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", ctypes.c_ulonglong * 6),
                        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

        self.k = _kernel32()
        self.handle = self.k.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = Extended()
        info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.k.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def close(self):
        if self.handle:
            self.k.CloseHandle(self.handle)
            self.handle = None


def _worker(handle, command):
    """Join before spawning any server/engine, removing the child-assignment race."""
    k = _kernel32()
    try:
        if not k.AssignProcessToJobObject(handle, k.GetCurrentProcess()):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        k.CloseHandle(handle)
    return subprocess.call(command)


def supervise(command, limits, probe, *, report=print):
    """Return child status, 2 for denied admission, 3 for a runtime guard trip."""
    def check():
        try:
            return limits.violation(probe())
        except Exception as exc:
            return f"memory telemetry failed ({type(exc).__name__})"

    reason = check()
    if reason:
        report(f"[resource guard] not started: {reason}")
        return 2
    proc = job = None
    try:
        kwargs = {}
        if os.name == "nt":
            job = _WindowsJob()
            os.set_handle_inheritable(job.handle, True)
            startup = subprocess.STARTUPINFO()
            startup.lpAttributeList = {"handle_list": [job.handle]}
            kwargs.update(startupinfo=startup, close_fds=True,
                          creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            command = [sys.executable, str(Path(__file__).resolve()), "--_worker", str(job.handle), "--", *command]
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(command, **kwargs)
        if job is not None:
            os.set_handle_inheritable(job.handle, False)
        while True:
            try:
                return proc.wait(timeout=limits.interval)
            except subprocess.TimeoutExpired:
                reason = check()
                if reason:
                    report(f"[resource guard] stopping this server: {reason}")
                    return 3
    except KeyboardInterrupt:
        return 130
    finally:
        if proc is not None:
            if job is not None:
                # Include descendants even if the frontend already exited. TerminateJobObject does not affect
                # other applications; kill() also covers the worker before it has joined its inherited job.
                job.k.TerminateJobObject(job.handle, 3)
                if proc.poll() is None:
                    proc.kill()
            else:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            proc.wait()
        if job is not None:
            job.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Launch Strata with opt-in RAM/VRAM floors. Stops only its own server.")
    ap.add_argument("--ram-floor-mib", type=float, default=0)
    ap.add_argument("--vram-floor-mib", type=float, default=0)
    ap.add_argument("--interval", type=float, default=1)
    ap.add_argument("--gpu", type=int, default=0, help="physical NVIDIA device index; also passed to the server")
    ap.add_argument("--_worker", type=int, help=argparse.SUPPRESS)
    ap.add_argument("server_args", nargs=argparse.REMAINDER, help="server options after --")
    args = ap.parse_args(argv)
    forwarded = args.server_args[1:] if args.server_args[:1] == ["--"] else args.server_args
    if args._worker is not None:
        return _worker(args._worker, forwarded)
    try:
        limits = Limits(args.ram_floor_mib, args.vram_floor_mib, args.interval)
    except ValueError as exc:
        ap.error(str(exc))
    if args.gpu < 0:
        ap.error("--gpu must be non-negative")
    command = [sys.executable, "-m", "serve.server", *forwarded, "--gpu", str(args.gpu)]
    return supervise(command, limits, MemoryProbe(limits, args.gpu))


if __name__ == "__main__":
    sys.exit(main())
