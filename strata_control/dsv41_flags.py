"""What a DeepSeek engine program accepts, asked of the program itself.  No network and no installer code: the server may import this (strata_control/dsv41_engine.py
reaches the network and serving must never import it)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_SEEN: dict = {}


def _env(exe: Path) -> dict:
    """The environment of the engine's own package folders first on the library path (as dsv41_engine.start_env)."""
    env = dict(os.environ)
    try:
        from strata_control import engine_component as ec
        dirs = [d for d in ec.package_lib_dirs(exe.parent) if Path(d).is_dir()]
    except Exception:                                    # noqa: BLE001 - the program's own folder is the fallback
        dirs = [str(exe.parent)]
    var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    if dirs:
        env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
    return env


def takes_parallel(exe, timeout: float = 15.0) -> bool | None:
    """Does this DeepSeek engine take `--parallel N` (several conversations at once)?  Asked of the engine itself: its usage text (`--help`,
    which needs no card and no model) names the flag from the version that has it.  True / False; None when it cannot be told (no such file,
    it would not start, no usage text came back) - then nothing is decided and the caller goes on as before.  Remembered per file and time."""
    try:
        path = Path(str(exe))
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
    except (OSError, ValueError, TypeError):
        return None
    if key in _SEEN:
        return _SEEN[key]
    answer: bool | None = None
    try:
        done = subprocess.run([str(path), "--help"], capture_output=True, text=True, errors="replace", timeout=timeout, cwd=str(path.parent),
                              env=_env(path), stdin=subprocess.DEVNULL)
        text = (done.stdout or "") + (done.stderr or "")
        if "--pack" in text and "--serve" in text:        # its own usage text (every version prints these two)
            answer = "--parallel" in text
    except (OSError, subprocess.SubprocessError):
        answer = None
    _SEEN[key] = answer
    return answer
