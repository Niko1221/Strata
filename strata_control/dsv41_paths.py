"""Where the DeepSeek engine package (strata-dsv41) is installed and which saved run configs name it: local only, no network, no download.

A saved profile's run config names the DeepSeek engine by a plain path into ``engine-dsv41/<version>-<platform>-<backend>[-<variant>]/``.  When an
engine update (the explicit `strata update`, the app's "Update engine", or `strata launch`) replaces that package, the old folder goes and the run
configs would name a program that is not there.  This module is what both sides share:

* the start path (strata_control/runtime.py) only REPAIRS: a run config whose folder is gone is pointed at the newest installed package of its own
  kind.  It never looks at the feed.  (Starting and serving never download anything: tests/terminal/test_no_acquisition.py.)
* the explicit update step (strata_control/updater.py update_dsv41) installs the newer package next to the old one, then points every saved run
  config at it with `repoint_older`, and removes older folders with `prune`, never one a running engine uses.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

NAME = "strata-dsv41"
DIR = "engine-dsv41"


def exe_name() -> str:
    return NAME + (".exe" if os.name == "nt" else "")


def version_key(folder_name: str) -> tuple:
    """The version at the front of a package folder's name ("0.1.55.1-windows-x86_64-hip-rdna3" -> (0, 1, 55, 1))."""
    head = str(folder_name).split("-")[0]
    return tuple(int(x) for x in head.split(".") if x.isdigit())


def kind_of(folder_name: str) -> str:
    """The platform-backend-variant part of a package folder's name ("0.1.55.1-windows-x86_64-hip-rdna3" -> "windows-x86_64-hip-rdna3")."""
    return str(folder_name).split("-", 1)[1] if "-" in str(folder_name) else ""


def engine_root(exe) -> Path | None:
    """The engine-dsv41 folder a plain run-config `exe` lies in, or None: a component reference, another engine, or an engine of your own
    (STRATA_DSV41_EXE)."""
    if not isinstance(exe, str) or not exe or exe.startswith("@") or os.environ.get("STRATA_DSV41_EXE"):
        return None
    root = Path(exe).parent.parent
    return root if root.name == DIR and Path(exe).name == exe_name() else None


def installed(root: Path, kind: str | None = None) -> list[Path]:
    """The strata-dsv41 programs installed in `root` (engine-dsv41/), oldest first; only packages of `kind` when it is given."""
    try:
        folders = [d for d in Path(root).iterdir() if d.is_dir() and (d / exe_name()).is_file() and (kind is None or kind_of(d.name) == kind)]
    except OSError:
        return []
    folders.sort(key=lambda d: (version_key(d.name), d.name))
    return [d / exe_name() for d in folders]


def lib_dirs(exe) -> list[str]:
    """The folders that must be on the library path when this engine starts (its own folder; a Windows HIP package also rocm/bin)."""
    from strata_control import engine_component as ec
    try:
        return ec.package_lib_dirs(Path(exe).parent)
    except OSError:
        return [str(Path(exe).parent)]


def in_use(folder) -> bool:
    """Is a program of this package folder running (an app's engine, a `strata launch` server)?  A folder in use is never removed: on Windows the
    removal would take its libraries from under the running engine.  psutil when it is there (any process whose program lies in the folder), and
    on Windows the program itself cannot be opened for writing while it runs."""
    folder = Path(folder)
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        here = os.path.normcase(os.path.abspath(folder)) + os.sep
        for p in psutil.process_iter(["exe"]):
            try:
                if os.path.normcase(os.path.abspath(p.info.get("exe") or "")).startswith(here):
                    return True
            except (psutil.Error, OSError, ValueError):
                continue
    exe = folder / exe_name()
    if os.name == "nt" and exe.is_file():
        try:
            with open(exe, "r+b"):
                pass
        except PermissionError:
            return True
        except OSError:
            return False
    return False


def prune(root: Path, keep: Path) -> list[str]:
    """Remove the older package folders of `keep`'s kind in `root` that no running engine uses; returns the names kept because they are in use."""
    busy = []
    kind = kind_of(keep.parent.name)
    try:
        folders = [d for d in Path(root).iterdir() if d.is_dir()]
    except OSError:
        return busy
    for old in folders:
        if old != keep.parent and kind_of(old.name) == kind and version_key(old.name) < version_key(keep.parent.name):
            if in_use(old):
                busy.append(old.name)
            else:
                shutil.rmtree(old, ignore_errors=True)
    return busy


def repoint(config, exe) -> bool:
    """Make the run config `config` name `exe` (a strata-dsv41 in an engine-dsv41 folder) and that package's library folders, nothing else
    changed; True when the file changed."""
    config = Path(config)
    cfg = json.loads(config.read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict) or not isinstance(cfg.get("exe"), str):
        return False
    libs = [d for d in (cfg.get("lib_dirs") or []) if DIR not in str(d)]
    wanted = {"exe": str(exe), "lib_dirs": [*libs, *lib_dirs(exe)]}
    if all(cfg.get(k) == v for k, v in wanted.items()):
        return False
    cfg.update(wanted)
    from strata_control import profiles
    profiles._atomic(config, json.dumps(cfg, indent=2).encode("utf-8"))
    return True


def saved_configs() -> list[Path]:
    """The run config of every saved profile (app-1, app-2, launch-...): the files a start reads."""
    from strata_control import profiles
    out = []
    try:
        for p in profiles.list_profiles():
            if isinstance(p.get("source_config"), str):
                out.append(Path(p["source_config"]))
    except Exception:                                    # noqa: BLE001 - informational
        pass
    return out


def _read(path) -> dict | None:
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return cfg if isinstance(cfg, dict) else None


def roots_of(configs) -> list[Path]:
    """The engine-dsv41 folders the run configs `configs` name (each once)."""
    roots: list[Path] = []
    for c in configs:
        cfg = _read(c)
        root = engine_root(cfg.get("exe")) if cfg else None
        if root is not None and root not in roots:
            roots.append(root)
    return roots


def repair_missing(configs) -> list[str]:
    """For a START: every run config in `configs` whose package folder is gone is pointed at the newest installed package of its own kind (nothing
    else changes, nothing is fetched).  Returns one plain line for each config it changed."""
    notes = []
    for c in configs:
        cfg = _read(c)
        exe = cfg.get("exe") if cfg else None
        root = engine_root(exe)
        if root is None or Path(exe).is_file() or not root.is_dir():
            continue
        found = installed(root, kind_of(Path(exe).parent.name))
        if not found:
            continue
        try:
            if repoint(c, found[-1]):
                notes.append(f"The DeepSeek engine {Path(exe).parent.name.split('-')[0]} was replaced by an update; {Path(c).name} now uses {found[-1].parent.name.split('-')[0]}.")
        except (OSError, ValueError):
            continue
    return notes


def repoint_older(configs, new_exe) -> int:
    """After an update installed `new_exe`: every run config in `configs` that names an older package of the same kind in the same folder (present or
    gone) is pointed at it.  Returns how many files changed."""
    new_exe = Path(new_exe)
    root, kind = new_exe.parent.parent, kind_of(new_exe.parent.name)
    changed = 0
    for c in configs:
        cfg = _read(c)
        exe = cfg.get("exe") if cfg else None
        if engine_root(exe) != root or Path(exe) == new_exe or kind_of(Path(exe).parent.name) != kind:
            continue
        if version_key(Path(exe).parent.name) >= version_key(new_exe.parent.name):
            continue
        try:
            changed += 1 if repoint(c, new_exe) else 0
        except (OSError, ValueError):
            continue
    return changed


def test_start(exe, timeout: float = 120.0) -> tuple[bool, str]:
    """`strata-dsv41 --version` runs from its own folder (the package's library folders first on the path) and names itself: it needs no graphics
    card and no model."""
    env = dict(os.environ)
    dirs = [d for d in lib_dirs(exe) if Path(d).is_dir()]
    var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    if dirs:
        env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
    try:
        done = subprocess.run([str(exe), "--version"], capture_output=True, text=True, errors="replace", timeout=timeout, cwd=str(Path(exe).parent), env=env,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"it could not be started ({exc})"
    first = ((done.stdout or "").strip().splitlines() or [""])[0]
    if done.returncode != 0 or not first.startswith(NAME):
        tail = ((done.stderr or "").strip().splitlines() or [first or f"exit {done.returncode}"])[-1]
        return False, f"it did not start correctly ({tail[:160]})"
    return True, first
