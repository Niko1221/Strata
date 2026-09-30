"""serve/winjob.py - on Windows, tie the processes Strata starts to the server's own lifetime.

Closing the console window, Task Manager or a crash end the server without running its cleanup, and the engine,
the vision encoder and the MCP servers kept running on their own. `contain(proc)` puts a child in a job object that
is set to kill everything in it when its last handle closes - and the only handle is this process's, which the OS
closes however the server ends. Processes the child starts later join the same job. The server itself stays out of
the job, so a browser that `--open` starts is not tied to it. Elsewhere, and if the job cannot be made, a no-op.
"""
from __future__ import annotations

import os
import threading

_job = None
_lock = threading.Lock()

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount",
                                                       "OtherOperationCount", "ReadTransferCount",
                                                       "WriteTransferCount", "OtherTransferCount")]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JobObjectExtendedLimitInformation = 9

    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    _k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    _k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)

    def _make_job():
        job = _k32.CreateJobObjectW(None, None)     # no security attributes: the handle is not inherited
        if not job:
            return None
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _k32.SetInformationJobObject(job, _JobObjectExtendedLimitInformation, ctypes.byref(info),
                                            ctypes.sizeof(info)):
            _k32.CloseHandle(job)
            return None
        return job


def contain(proc) -> bool:
    """Put a started subprocess.Popen in the kill-on-close job. True if it is in; False (harmlessly) otherwise."""
    global _job
    if os.name != "nt" or proc is None:
        return False
    with _lock:
        if _job is None:
            _job = _make_job() or 0
        if not _job:
            return False
    try:
        return bool(_k32.AssignProcessToJobObject(_job, int(proc._handle)))
    except (AttributeError, OSError, ValueError):
        return False
