"""Windows-only test process containment. No engine imports or external PID operations.

Uses the same atomic JOB_LIST + suspended-create pattern as tools/verify_bounded_process.cs.
Only handles created in this invocation can be terminated. No breakaway permission.
"""

import ctypes as C
from ctypes import wintypes as W
import msvcrt
import os
from pathlib import Path
import subprocess
import time


class BasicLimits(C.Structure):
    _fields_ = [
        ("process_time", C.c_longlong),
        ("job_time", C.c_longlong),
        ("flags", W.DWORD),
        ("minimum", C.c_size_t),
        ("maximum", C.c_size_t),
        ("active", W.DWORD),
        ("affinity", C.c_size_t),
        ("priority", W.DWORD),
        ("scheduling", W.DWORD),
    ]


class Limits(C.Structure):
    _fields_ = [
        ("basic", BasicLimits),
        ("io", C.c_ulonglong * 6),
        ("process_memory", C.c_size_t),
        ("job_memory", C.c_size_t),
        ("peak_process", C.c_size_t),
        ("peak_job", C.c_size_t),
    ]


class Accounting(C.Structure):
    _fields_ = [
        ("times", C.c_longlong * 4),
        ("faults", W.DWORD),
        ("total", W.DWORD),
        ("active", W.DWORD),
        ("terminated", W.DWORD),
    ]


class Startup(C.Structure):
    _fields_ = [
        ("size", W.DWORD),
        ("reserved", W.LPWSTR),
        ("desktop", W.LPWSTR),
        ("title", W.LPWSTR),
        ("x", W.DWORD),
        ("y", W.DWORD),
        ("width", W.DWORD),
        ("height", W.DWORD),
        ("chars_x", W.DWORD),
        ("chars_y", W.DWORD),
        ("fill", W.DWORD),
        ("flags", W.DWORD),
        ("show", W.WORD),
        ("reserved_size", W.WORD),
        ("reserved_bytes", C.c_void_p),
        ("stdin", W.HANDLE),
        ("stdout", W.HANDLE),
        ("stderr", W.HANDLE),
    ]


class StartupEx(C.Structure):
    _fields_ = [("startup", Startup), ("attributes", C.c_void_p)]


class Process(C.Structure):
    _fields_ = [("process", W.HANDLE), ("thread", W.HANDLE), ("pid", W.DWORD), ("tid", W.DWORD)]


def api() -> C.WinDLL:
    """Declare pointer-width-safe Win32 signatures before any process creation."""
    kernel = C.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateJobObjectW": (W.HANDLE, [C.c_void_p, W.LPCWSTR]),
        "CloseHandle": (W.BOOL, [W.HANDLE]),
        "SetInformationJobObject": (W.BOOL, [W.HANDLE, C.c_int, C.c_void_p, W.DWORD]),
        "QueryInformationJobObject": (W.BOOL, [W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.c_void_p]),
        "InitializeProcThreadAttributeList": (W.BOOL, [C.c_void_p, W.DWORD, W.DWORD, C.c_void_p]),
        "UpdateProcThreadAttribute": (
            W.BOOL,
            [C.c_void_p, W.DWORD, C.c_size_t, C.c_void_p, C.c_size_t, C.c_void_p, C.c_void_p],
        ),
        "DeleteProcThreadAttributeList": (None, [C.c_void_p]),
        "CreateProcessW": (
            W.BOOL,
            [
                W.LPCWSTR,
                W.LPWSTR,
                C.c_void_p,
                C.c_void_p,
                W.BOOL,
                W.DWORD,
                C.c_void_p,
                W.LPCWSTR,
                C.c_void_p,
                C.c_void_p,
            ],
        ),
        "IsProcessInJob": (W.BOOL, [W.HANDLE, W.HANDLE, C.c_void_p]),
        "ResumeThread": (W.DWORD, [W.HANDLE]),
        "WaitForSingleObject": (W.DWORD, [W.HANDLE, W.DWORD]),
        "GetExitCodeProcess": (W.BOOL, [W.HANDLE, C.c_void_p]),
        "TerminateJobObject": (W.BOOL, [W.HANDLE, W.UINT]),
    }
    for name, (result, arguments) in signatures.items():
        function = getattr(kernel, name)
        function.restype, function.argtypes = result, arguments
    return kernel


def require(ok: bool, operation: str) -> None:
    if not ok:
        raise OSError(C.get_last_error(), operation)


def run_owned(command: list[str], root: Path, env: dict[str, str], raw: Path, timeout: float) -> dict:
    """Create contained and suspended, verify membership, then resume; drain job on every exit."""
    kernel = api()
    result = {
        "status": "BLOCKED",
        "exit_code": 77,
        "contained_before_resume": False,
        "termination_confirmed": False,
        "job_active_after": None,
    }
    job, attributes, initialized = None, None, False
    process = Process()
    try:
        job = kernel.CreateJobObjectW(None, None)
        require(bool(job), "create private job")
        limits = Limits()
        limits.basic.flags = 0x2000  # KILL_ON_JOB_CLOSE, never BREAKAWAY_OK.
        require(kernel.SetInformationJobObject(job, 9, C.byref(limits), C.sizeof(limits)), "set job limits")
        size = C.c_size_t()
        kernel.InitializeProcThreadAttributeList(None, 2, 0, C.byref(size))
        attributes = C.create_string_buffer(size.value)
        require(kernel.InitializeProcThreadAttributeList(attributes, 2, 0, C.byref(size)), "initialize attributes")
        initialized = True
        jobs = (W.HANDLE * 1)(job)
        require(
            kernel.UpdateProcThreadAttribute(attributes, 0, 0x2000D, jobs, C.sizeof(jobs), None, None),
            "atomic job assignment unavailable",
        )
        with raw.open("xb", buffering=0) as output, open(os.devnull, "rb", buffering=0) as null:
            out_handle, in_handle = msvcrt.get_osfhandle(output.fileno()), msvcrt.get_osfhandle(null.fileno())
            os.set_handle_inheritable(out_handle, True)
            os.set_handle_inheritable(in_handle, True)
            handles = (W.HANDLE * 2)(out_handle, in_handle)
            require(
                kernel.UpdateProcThreadAttribute(attributes, 0, 0x20002, handles, C.sizeof(handles), None, None),
                "restrict inherited handles",
            )
            startup = StartupEx()
            startup.startup.size = C.sizeof(startup)
            startup.startup.flags = 0x101  # USESTDHANDLES + USESHOWWINDOW(hidden)
            startup.startup.stdin, startup.startup.stdout, startup.startup.stderr = in_handle, out_handle, out_handle
            startup.attributes = C.cast(attributes, C.c_void_p)
            environment = C.create_unicode_buffer(
                "\0".join(f"{key}={value}" for key, value in sorted(env.items(), key=lambda pair: pair[0].upper()))
                + "\0\0"
            )
            executable = str(Path(command[0]).resolve())
            line = C.create_unicode_buffer(subprocess.list2cmdline([executable, *command[1:]]))
            require(
                kernel.CreateProcessW(
                    executable,
                    line,
                    None,
                    None,
                    True,
                    0x08080404,
                    environment,
                    str(root),
                    C.byref(startup),
                    C.byref(process),
                ),
                "create atomically contained suspended child",
            )
            result["owned_root_pid"] = process.pid
            in_job = W.BOOL()
            require(
                kernel.IsProcessInJob(process.process, job, C.byref(in_job)) and in_job.value,
                "verify suspended job membership",
            )
            result["contained_before_resume"] = True
            require(kernel.ResumeThread(process.thread) != 0xFFFFFFFF, "resume owned child")
            started = time.monotonic()
            while True:
                wait = kernel.WaitForSingleObject(process.process, 20)
                require(wait != 0xFFFFFFFF, "wait owned child")
                if wait == 0:
                    code = W.DWORD()
                    require(kernel.GetExitCodeProcess(process.process, C.byref(code)), "read child exit")
                    result.update(status="FINISHED", exit_code=code.value)
                    break
                if time.monotonic() - started >= timeout:
                    result.update(status="TIMEOUT", exit_code=124)
                    break
                if raw.stat().st_size > 8 * 1024**2:
                    result.update(status="OUTPUT_LIMIT", exit_code=125)
                    break
    except (OSError, ValueError) as error:
        result["error"] = type(error).__name__ + ":" + str(error)
    finally:
        if job:
            try:
                require(kernel.TerminateJobObject(job, 124), "terminate owned job")
                deadline = time.monotonic() + 5
                while True:
                    accounting = Accounting()
                    require(
                        kernel.QueryInformationJobObject(job, 1, C.byref(accounting), C.sizeof(accounting), None),
                        "query owned job accounting",
                    )
                    result["job_total_processes"] = accounting.total
                    result["job_active_after"] = accounting.active
                    if accounting.active == 0 or time.monotonic() >= deadline:
                        break
                    time.sleep(0.01)
                result["termination_confirmed"] = accounting.active == 0
                if not result["termination_confirmed"]:
                    result.update(status="CLEANUP_UNCONFIRMED", exit_code=126)
            except OSError as error:
                result.update(status="CLEANUP_UNCONFIRMED", exit_code=126, cleanup_error=str(error))
        for handle in (process.thread, process.process, job):
            if handle:
                kernel.CloseHandle(handle)
        if initialized:
            kernel.DeleteProcThreadAttributeList(attributes)
    return result
