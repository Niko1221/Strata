"""The launcher interface: how a platform spawns and terminates the Strata server.  Small on purpose."""

from __future__ import annotations


class Launcher:
    """What every platform adapter provides.

    spawn(cmd, cwd, log_file) -> pid
        Start `cmd` detached, its stdout/stderr appended to `log_file`, working from `cwd`.  The returned
        process (or its group) must survive the Manager exiting, and everything Strata is expected to manage
        (the server itself and the engine / vision encoder / MCP servers it starts) must be reachable through
        terminate(pid, grace_s).  On Windows the server gets a hidden console so console children (the
        engine) inherit it instead of popping a new visible console window.

    terminate(pid, grace_s) -> dict
        Best-effort graceful stop of the whole Strata process tree; after `grace_s` it must be forced down.
        Returns {"stopped": bool, "forced": bool} - `forced` True when the hard path had to be used.

    terminate_force(pid) -> dict
        The hard path immediately (no grace): what the Manager's "Force Stop" button calls.
        Returns {"stopped": bool, "forced": True}.

    alive(pid) -> bool
        Is the process still running?  (os.kill(pid, 0): the platform-neutral existence probe.)
    """

    name = "base"

    def spawn(self, cmd: list, cwd: str, log_file):  # pragma: no cover - the interface
        raise NotImplementedError

    def terminate(self, pid: int, grace_s: float):  # pragma: no cover
        raise NotImplementedError

    def terminate_force(self, pid: int):  # pragma: no cover
        raise NotImplementedError

    def alive(self, pid: int) -> bool:  # pragma: no cover
        raise NotImplementedError