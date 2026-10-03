"""gui.platforms - the OS-specific half of the Manager's launcher.

Everything else in the Manager is platform-neutral (pathlib paths, no PowerShell/.bat in the core); only
spawning a detached server process and terminating its process tree differ between Windows and Linux, so
those two operations live here.  The rest of the supervisor (gui/launcher.py) calls `get_launcher()`
and never branches on the OS itself.

  - windows.py   spawn: DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP (no console window)
                 stop:  close the process tree, taskkill /T (force /F after a grace) - the same outcome as
                        closing Strata's own console window
  - linux.py     spawn: start_new_session=True (a new process group; the server, engine, vision encoder and
                 MCP children share it)
                 stop:  SIGTERM to the group (serve/server.py handles it gracefully: QUIT to the engine),
                        SIGKILL to the group after a grace
"""
from __future__ import annotations

import os

from .base import Launcher  # noqa: F401  (re-exported: the interface)


def get_launcher() -> Launcher:
    """The platform adapter for this OS."""
    if os.name == "nt":
        from .windows import WindowsLauncher
        return WindowsLauncher()
    from .linux import LinuxLauncher
    return LinuxLauncher()


def platform_name() -> str:
    return "windows" if os.name == "nt" else "linux"