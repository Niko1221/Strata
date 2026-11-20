"""Owned short-timeout fixture only. Its descendant never starts an engine."""

import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_bounded import Memory


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "leaf":
        time.sleep(30)
        return
    memory = Memory()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise OSError("child resource telemetry unavailable")
    free = shutil.disk_usage(Path(__file__).resolve().parents[2]).free
    if memory.avail_phys < 768 * 1024**2 or memory.avail_page < 1024**3 or free < 5 * 1024**3:
        raise RuntimeError("child fixture gate blocked")
    sys.stdout.write(
        json.dumps(
            {"child_available_ram": memory.avail_phys, "child_commit_headroom": memory.avail_page, "child_h_free": free}
        )
        + "\n"
    )
    sys.stdout.flush()
    child = subprocess.Popen([sys.executable, "-B", __file__, "leaf"], creationflags=subprocess.CREATE_NO_WINDOW)
    sys.stdout.write(f"owned fixture descendant created: {child.pid}\n")
    sys.stdout.flush()
    time.sleep(30)


if __name__ == "__main__":
    main()
