"""Linux/Unix process handling for the Manager's launcher.

Spawn: a new session (start_new_session=True) - the server becomes a process-group leader, and everything it
starts (the engine, the vision encoder, the MCP servers: plain children) stays in that group.  Stop: SIGTERM
to the group first - serve/server.py's own signal handler turns it into a graceful shutdown (it sends QUIT to
the engine) - then SIGKILL to the group after the grace.  Force Stop: SIGKILL immediately.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time

from .base import Launcher


class LinuxLauncher(Launcher):
    name = "linux"

    def spawn(self, cmd: list, cwd: str, log_file) -> int:
        f = open(log_file, "a", encoding="utf-8")
        try:
            proc = subprocess.Popen(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        finally:
            f.close()
        return proc.pid

    def terminate(self, pid: int, grace_s: float) -> dict:
        try:
            os.killpg(int(pid), signal.SIGTERM)              # graceful: server.py QUITs the engine
        except OSError:
            pass
        if self._wait_dead(pid, grace_s):
            return {"stopped": True, "forced": False}
        try:
            os.killpg(int(pid), signal.SIGKILL)
        except OSError:
            pass
        self._wait_dead(pid, grace_s)
        return {"stopped": not self.alive(pid), "forced": True}

    def terminate_force(self, pid: int) -> dict:
        try:
            os.killpg(int(pid), signal.SIGKILL)
        except OSError:
            pass
        self._wait_dead(pid, 5.0)
        return {"stopped": not self.alive(pid), "forced": True}

    @staticmethod
    def _wait_dead(pid: int, seconds: float) -> bool:
        deadline = time.time() + seconds
        while time.time() < deadline:
            if not LinuxLauncher.alive(pid):
                return True
            time.sleep(0.25)
        return not LinuxLauncher.alive(pid)

    @staticmethod
    def alive(pid: int) -> bool:
        if not pid:
            return False
        try:
            os.kill(int(pid), 0)
            return True
        except OSError:
            return False