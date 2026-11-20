"""The OS and machine facts Control needs, answered without asking WMI on Windows (#4975, #5185).

`platform.system()`, `.machine()`, `.release()`, `.version()`, `.processor()` and `.node()` all go through
`platform.uname()`. On Windows with Python 3.12 that calls `win32_ver()` and `_get_machine_win32()`, which query WMI
(`platform._wmi_query`) with no timeout: a stuck `winmgmt` service froze every Control command that resolves the engine
component ("Strata Control did not answer within 60 s (models list)"), although the interpreter, the environment and the
registry know the same facts. On Windows these functions read only those. Everywhere else they are the `platform` calls
(an `os.uname()` there, not a service), looked up at call time so a test can replace them.

Pure and tiny on purpose, like targets.py: the start path imports it.
"""

from __future__ import annotations

import os
import socket
import sys
import platform

_WINDOWS_MACHINES = {"amd64": "AMD64", "x86_64": "AMD64", "arm64": "ARM64", "aarch64": "ARM64", "x86": "x86", "i386": "x86",
                     "i686": "x86", "ia64": "ia64"}
_SYSINFO_KEYS = {"system", "machine", "release", "version", "processor", "node", "platform_string"}
_NATIVE_CACHE: dict[tuple[str, str, str, str, str], dict[str, str]] = {}


def _valid_sysinfo(value: object) -> bool:
    return isinstance(value, dict) and set(value) == _SYSINFO_KEYS and all(isinstance(v, str) for v in value.values())


def _native_sysinfo() -> dict[str, str] | None:
    if not (is_windows() and os.name == "nt"):
        return None
    if type(getattr(sys, "getwindowsversion", None)).__name__ != "builtin_function_or_method":
        return None
    if not (os.environ.get("PROCESSOR_ARCHITECTURE") or os.environ.get("PROCESSOR_ARCHITEW6432")):
        return None
    from strata_control import native_helpers
    exe = native_helpers.program("sysinfo")
    if exe is None:
        return None
    cache_key = (
        str(exe),
        os.environ.get("PROCESSOR_ARCHITECTURE", ""),
        os.environ.get("PROCESSOR_ARCHITEW6432", ""),
        os.environ.get("PROCESSOR_IDENTIFIER", ""),
        os.environ.get("COMPUTERNAME", ""),
    )
    if cache_key in _NATIVE_CACHE:
        return _NATIVE_CACHE[cache_key]
    try:
        res = native_helpers.call("sysinfo", "gather", [], validate=_valid_sysinfo)
    except native_helpers.Unavailable:
        return None
    _NATIVE_CACHE[cache_key] = res
    return res


def is_windows() -> bool:
    return sys.platform.startswith("win")


def block_wmi() -> bool:
    """Make sure no part of this process can ask WMI through CPython 3.12's `_wmi` helper (Windows only; True when it is blocked).

    `platform.system()` / `uname()` / `win32_ver()` / `machine()` query WMI through `_wmi.exec_query`. When WMI refuses the account (HRESULT
    0x80041003, as for a standard user over an ssh logon) the query takes seconds, `exec_query` gives up after 100 ms and returns, and its
    worker thread then reads a struct on the caller's gone stack: an access violation that ends the whole process (0xC0000005; the runtime
    host died this way, "runtime control is unavailable", DIAG-WINHOST-0216). Without `_wmi`, `platform` answers from the registry, `ver` and
    the environment, as it did before Python 3.12. Every Strata entry point puts `sys.modules["_wmi"] = None` first; this is the second guard
    for a process whose `platform` was imported before that (`platform._wmi_query` then raises the OSError `platform` already handles)."""
    if not is_windows():
        return False
    sys.modules["_wmi"] = None

    def _wmi_query(*keys):
        raise OSError("not supported")
    _wmi_query.strata_blocked = True
    current = getattr(platform, "_wmi_query", None)
    if current is not None and not getattr(current, "strata_blocked", False):
        platform._wmi_query = _wmi_query                 # (a platform without _wmi_query never asks WMI)
    return True


def system() -> str:
    """`platform.system()`: "Windows", "Linux", "Darwin" ..."""
    if (native := _native_sysinfo()) is not None:
        return native["system"]
    return "Windows" if is_windows() else platform.system()


def _windows_machine() -> str:
    """Windows names its architecture in the environment the loader gives every process: PROCESSOR_ARCHITEW6432 is the real
    one inside a 32-bit process on 64-bit Windows, PROCESSOR_ARCHITECTURE the process's own (which is also what an x64
    program emulated on ARM64 sees: the engine that fits is the x64 one). `platform.uname()` itself read these first on
    Python 3.11 and before. Without them (a stripped environment) the interpreter's own build tag answers."""
    for key in ("PROCESSOR_ARCHITEW6432", "PROCESSOR_ARCHITECTURE"):
        value = os.environ.get(key, "").strip()
        if value:
            return _WINDOWS_MACHINES.get(value.lower(), value)
    try:
        import sysconfig
        tag = sysconfig.get_platform().lower()
    except Exception:                                    # noqa: BLE001 - a fact lookup never fails a start
        tag = ""
    if "arm64" in tag:
        return "ARM64"
    if "amd64" in tag or "x86_64" in tag:
        return "AMD64"
    return "x86" if tag in ("win32", "") else tag


def machine() -> str:
    """`platform.machine()`: "AMD64", "ARM64" on Windows; "x86_64", "arm64", "aarch64" elsewhere."""
    if (native := _native_sysinfo()) is not None:
        return native["machine"]
    return _windows_machine() if is_windows() else platform.machine()


def release() -> str:
    """`platform.release()`: "10" or "11" on Windows (11 from build 22000), the kernel release elsewhere."""
    if (native := _native_sysinfo()) is not None:
        return native["release"]
    if is_windows():
        info = _windows_version()
        if info is None:
            return ""
        return "11" if info[0] == 10 and info[2] >= 22000 else str(info[0])
    return platform.release()


def version() -> str:
    """`platform.version()`: "10.0.19045" on Windows, the kernel version elsewhere."""
    if (native := _native_sysinfo()) is not None:
        return native["version"]
    if is_windows():
        info = _windows_version()
        return "" if info is None else f"{info[0]}.{info[1]}.{info[2]}"
    return platform.version()


def _windows_version() -> tuple[int, int, int] | None:
    try:
        info = sys.getwindowsversion()                   # type: ignore[attr-defined]
        return int(info.major), int(info.minor), int(info.build)
    except Exception:                                    # noqa: BLE001
        return None


def processor() -> str:
    """`platform.processor()`: the CPU's description (PROCESSOR_IDENTIFIER on Windows; the empty string when unknown)."""
    if (native := _native_sysinfo()) is not None:
        return native["processor"]
    if is_windows():
        return os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
    return platform.processor()


def node() -> str:
    """`platform.node()`: the computer's name."""
    if (native := _native_sysinfo()) is not None:
        return native["node"]
    if is_windows():
        try:
            return socket.gethostname()
        except OSError:
            return os.environ.get("COMPUTERNAME", "")
    return platform.node()


def platform_string() -> str:
    """A one-line description of the OS for reports (`platform.platform()` asks the same questions as `uname`)."""
    if (native := _native_sysinfo()) is not None:
        return native["platform_string"]
    if is_windows():
        return f"Windows-{release()}-{version()}"
    return platform.platform()
