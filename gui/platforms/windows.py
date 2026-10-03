"""Windows process handling for the Manager's launcher.

Spawn: a NEW CONSOLE process whose console window is hidden from birth (STARTUPINFO SW_HIDE).  The reason:
on Windows a console-subsystem child (the engine, strata.exe) spawned by a process with NO console gets a
brand-new VISIBLE console window - an extra blank terminal in front of the browser.  Giving the server a
hidden console to inherit removes the window entirely while keeping Strata's server/engine console-less from
the user's point of view (all their output still goes to the log files the Manager tails).

Stop: close the whole process tree with taskkill /T - Strata's documented "close the console window"
outcome (server, engine, vision encoder and MCP children end together).  taskkill without /F posts a WM_CLOSE
a windowless detached server cannot answer, so after a short grace the tree is force-closed (and the stop
result says so - the UI then shows it took the forced path).

No PowerShell, no .bat: only CreateProcess + taskkill.

Every short utility subprocess this module starts (taskkill) runs with CREATE_NO_WINDOW: launched from a
pythonw/no-console parent (the Manager), a console-subsystem child would otherwise flash a transient black
window; from a console parent the flag changes nothing (its output still goes where the caller redirected it).
"""

from __future__ import annotations

import subprocess
import time

from .base import Launcher

# Born hidden: the console exists (so console children inherit it) but no window ever shows.
_HIDDEN_CONSOLE = subprocess.STARTUPINFO()
_HIDDEN_CONSOLE.dwFlags = subprocess.STARTF_USESHOWWINDOW
_HIDDEN_CONSOLE.wShowWindow = 0                       # SW_HIDE

# Windows' soft taskkill (no /F) only posts a WM_CLOSE a windowless detached server cannot answer: treat it
# as the graceful attempt, cap the wait so a normal Stop stays snappy (<4 s), then force-close the tree.
_SOFT_GRACE_S = 3.0


class WindowsLauncher(Launcher):
    name = "windows"

    def spawn(self, cmd: list, cwd: str, log_file) -> int:
        with open(log_file, "a", encoding="utf-8") as f:
            p = subprocess.Popen(
                cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP,
                startupinfo=_HIDDEN_CONSOLE, close_fds=False)
        return p.pid

    def terminate(self, pid: int, grace_s: float) -> dict:
        subprocess.run(["taskkill", "/PID", str(pid), "/T"], capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW,
                       timeout=max(1, int(grace_s) + 1))
        if self._wait_dead(pid, min(grace_s, _SOFT_GRACE_S)):
            return {"stopped": True, "forced": False}
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW,
                       timeout=max(1, int(grace_s) + 1))
        self._wait_dead(pid, 8.0)
        return {"stopped": not self.alive(pid), "forced": True}

    def terminate_force(self, pid: int) -> dict:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=30,
                       creationflags=subprocess.CREATE_NO_WINDOW)
        self._wait_dead(pid, 8.0)
        return {"stopped": not self.alive(pid), "forced": True}

    @staticmethod
    def _wait_dead(pid: int, seconds: float) -> bool:
        deadline = time.time() + seconds
        while time.time() < deadline:
            if not WindowsLauncher.alive(pid):
                return True
            time.sleep(0.25)
        return not WindowsLauncher.alive(pid)

    @staticmethod
    def alive(pid: int) -> bool:
        """A reliable Windows existence check.  os.kill(pid, 0) is not one: it keeps reporting a
        process as alive for a while after TerminateProcess.  OpenProcess + GetExitCodeProcess
        (STILL_ACTIVE) answers immediately and handles Access Denied correctly."""
        if not pid:
            return False
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        k32.CloseHandle.argtypes = (wintypes.HANDLE,)
        # PROCESS_QUERY_LIMITED_INFORMATION: enough for GetExitCodeProcess, works for any user's processes
        h = k32.OpenProcess(0x1000, False, int(pid))
        if not h:
            return False                                # gone (or not accessible): not alive
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return False
            return code.value == 259                    # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)