"""DeepSeek-V4.1's engine as its own package (feed member ``dsv41``), downloaded when DeepSeek is chosen.

It goes in ``<data>/engine-dsv41/<version>-<platform>-<backend>[-<variant>]/``, beside the Strata engine and never inside it, so the Qwen
engine's files are not touched.  The package is picked, downloaded (resumable), unpacked and checked with the same code as an engine update
(strata_control/updater.py: the signed feed, the archive's SHA-256, the manifest hash the feed names, the package signature, every file's hash);
there is no test start here (setup's own 16-token question follows the pack).
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Callable

NAME = "strata-dsv41"
DIR = "engine-dsv41"


def exe_name() -> str:
    return NAME + (".exe" if os.name == "nt" else "")


def version_key(folder_name: str) -> tuple:
    """The version at the front of a package folder's name ("0.1.49-windows-x86_64-cuda-cuda13" -> (0, 1, 49)), for choosing the newest."""
    head = str(folder_name).split("-")[0]
    return tuple(int(x) for x in head.split(".") if x.isdigit())


def find(dest: Path) -> Path | None:
    """The newest installed package's strata-dsv41 in `dest` (engine-dsv41/), or None."""
    try:
        folders = sorted((d for d in Path(dest).iterdir() if d.is_dir() and (d / exe_name()).is_file()), key=lambda d: (version_key(d.name), d.name))
    except OSError:
        return None
    return folders[-1] / exe_name() if folders else None


ARCHIVE_SUFFIXES = (".tar.gz", ".tar.xz", ".zip")
MAX_UNPACKED = 8 << 30                  # a DeepSeek engine package is under a GB; this only stops an archive that unpacks into something huge

LIB_VAR = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"


def lib_dirs(exe: str | os.PathLike) -> list[str]:
    """The folders that must be on the library path when this engine starts: its own package folder, then the folders the package's
    BUILD.json names.  A Windows HIP package keeps the ROCm DLLs (hipBLAS / rocBLAS and their imports) in ``rocm/bin``, which is also
    where rocBLAS's kernel folders and ``../.kpack`` are laid out from (strata_control/engine_component.py package_lib_dirs).  A CUDA
    package keeps everything beside the program, so this is its folder alone."""
    from strata_control import engine_component as ec
    try:
        return ec.package_lib_dirs(Path(exe).parent)
    except OSError:
        return [str(Path(exe).parent)]


def start_env(exe: str | os.PathLike, environ: dict | None = None) -> dict:
    """`environ` (the process's own by default) with this package's folders first on PATH (Windows) / LD_LIBRARY_PATH."""
    env = dict(os.environ if environ is None else environ)
    dirs = [d for d in lib_dirs(exe) if Path(d).is_dir()]
    if dirs:
        env[LIB_VAR] = os.pathsep.join(dirs + ([env[LIB_VAR]] if env.get(LIB_VAR) else []))
    return env


def pending(dest: Path) -> list[Path]:
    """Packages the person put in `dest` (engine-dsv41/) by hand: strata-dsv41-*.tar.gz / .zip, newest first."""
    try:
        files = [f for f in Path(dest).iterdir() if f.is_file() and f.name.startswith("strata-dsv41") and f.name.endswith(ARCHIVE_SUFFIXES)]
    except OSError:
        return []
    return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)


def install_package(dest: Path, package: str | os.PathLike, *, emit: Callable[[str], None] = print) -> Path | None:
    """Install a DeepSeek engine package the person downloaded from the release page (a .tar.gz / .zip, or the unpacked folder) into `dest`.
    Checked before anything is installed: it is a DeepSeek engine package for this system, its manifest is signed by a key Strata trusts, and every
    file matches the hash in the manifest.  Returns the new strata-dsv41, or None with a plain line through `emit`."""
    from strata_control import components, engine_component as ec, signing, updater
    from strata_control.targets import host_platform

    src, dest = Path(package).expanduser(), Path(dest)
    if not src.exists():
        emit(f"{src} is not there. Download the DeepSeek engine package for your system from the release page and give its path again.")
        return None
    work = dest.parent / (DIR + ".manual")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        if src.is_dir():
            root = work / "pkg"
            shutil.copytree(src, root)
        else:
            fmt = "zip" if src.name.endswith(".zip") else "tar.xz" if src.name.endswith(".tar.xz") else "tar.gz"
            if not src.name.endswith(ARCHIVE_SUFFIXES):
                emit(f"{src.name} is not a package file (.tar.gz or .zip). Download the DeepSeek engine package for your system from the release page.")
                return None
            root = work / "pkg"
            updater.extract(src, root, fmt, MAX_UNPACKED)
        if not (root / ec.MANIFEST).is_file():
            inner = next((d for d in root.iterdir() if d.is_dir() and (d / ec.MANIFEST).is_file()), None) if root.is_dir() else None
            if inner is None:
                emit(f"{src.name} is not an engine package (it has no {ec.MANIFEST}).")
                return None
            root = inner
        component = ec.read_manifest(root)
        if component.entrypoint != exe_name():
            emit(f"{src.name} is not the DeepSeek engine (its program is {component.entrypoint}, not {exe_name()}). "
                 "The Strata engine package is a different download.")
            return None
        if component.platform != host_platform():
            emit(f"{src.name} is for {component.platform}, and this computer is {host_platform()}. Download the one for your system.")
            return None
        outer = ec._verifier
        signing.install_component_verifier(signing.load_trusted_keys())
        try:
            ec.verify_signature(component, {**os.environ, ec.ENV_REQUIRE_SIGNED: "1"})
            ec.verify(component, full=True)
        except ec.ComponentProblem as exc:
            emit(f"{src.name} did not pass its check ({exc}). It may be damaged or not from the Strata release: download it again from the release page.")
            return None
        finally:
            ec.register_signature_verifier(outer)
        name = f"{component.version}-{component.platform}-{component.backend}" + (f"-{component.variant}" if component.variant else "")
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / name
        shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(root), str(target))
        if os.name != "nt":
            (target / exe_name()).chmod(0o755)
        for old in dest.iterdir():
            if old.is_dir() and old.name != target.name:
                shutil.rmtree(old, ignore_errors=True)
        emit(f"DeepSeek engine {component.version} ({component.backend}) checked (signed, every file matches) and installed: {target}")
        return target / exe_name()
    except (updater.UpdateError, ec.ComponentProblem, OSError) as exc:
        emit(f"{src.name} could not be installed: {exc}")
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def install(dest: Path, source: str, *, hardware: dict | None = None, emit: Callable[[str], None] = print, cancel=None) -> Path | None:
    """Fetch the DeepSeek engine package for this PC from the signed feed at `source` (a URL or a folder) into `dest`.
    Returns the new strata-dsv41, or None with a plain line through `emit` when there is none (not in the feed, nothing for this PC)."""
    from strata_control import feed as feedmod, signing, updater

    dest = Path(dest)
    work = dest.parent / (DIR + ".work")
    hw = hardware or updater.detect_hardware()
    state_dir = work / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    ctx = updater.Context(components=work, source=source, trusted=signing.load_trusted_keys(), hardware=hw, cancel=cancel,
                          emit=lambda event: emit(event["message"]) if event.get("event") == "phase" and event.get("message") else None)
    state = updater.State(work)
    try:
        feed = updater.open_feed(ctx, state)
        pkg, reasons = feedmod.select_engine(list(feed.dsv41), hw)
        if pkg is None:
            emit("There is no DeepSeek engine package for this computer in the update feed"
                 + (": " + "; ".join(reasons[:2]) if reasons else "") + ".")
            return None
        archive = updater._download(ctx, "dsv41", pkg.url, pkg.size, pkg.sha256, "the DeepSeek engine", feed)
        stage_root, component = updater.stage_engine(ctx, feed, pkg, archive)
        try:
            folder = stage_root / "engine" / updater.folder_name(pkg)
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / folder.name
            shutil.rmtree(target, ignore_errors=True)
            shutil.move(str(folder), str(target))
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)
        if os.name != "nt":
            (target / exe_name()).chmod(0o755)
        for old in dest.iterdir():                              # one generation: the older folders of this engine go (one a running engine uses stays)
            if old.is_dir() and old.name != target.name and not in_use(old):
                shutil.rmtree(old, ignore_errors=True)
        emit(f"DeepSeek engine {pkg.version} installed: {target}")
        shutil.rmtree(work, ignore_errors=True)               # (on a failure the partial download stays in `work`: the next try resumes it)
        return target / exe_name()
    except (updater.UpdateError, feedmod.FeedError, signing.SignatureError) as e:
        emit(f"The DeepSeek engine could not be fetched: {e}")
        return None


# ---------------------------------------------------------------------------------------------- upgrade of an installed package

def source_reachable(source: str, timeout: float = 10.0) -> bool:
    """Can the feed `source` be reached within `timeout` seconds?  A folder or file: it exists.  A URL: any HTTP answer counts (a 404 is an answer);
    no answer, a refused or timed-out connection does not.  One quick check instead of minutes of retries."""
    import urllib.error
    import urllib.parse
    import urllib.request
    from strata_control import transfer
    kind = transfer.kind_of(source)
    if kind in ("path", "file"):
        try:
            return transfer._local_path(source).exists()
        except (OSError, transfer.TransferError):
            return False
    parts = urllib.parse.urlsplit(source)
    try:
        urllib.request.urlopen(urllib.request.Request(f"{parts.scheme}://{parts.netloc}/", method="HEAD", headers={"User-Agent": "strata-launch"}),
                               timeout=timeout).close()
        return True
    except urllib.error.HTTPError:
        return True
    except (OSError, urllib.error.URLError, ValueError):
        return False


def _test_start(exe: Path, timeout: float = 120.0) -> tuple[bool, str]:
    """The new engine starts: `strata-dsv41 --version` runs, loads its libraries and names itself (it needs no graphics card and no model).
    The package's own folders go on the library path first: a Windows HIP package carries the ROCm runtime it loads in rocm/bin."""
    import subprocess
    try:
        done = subprocess.run([str(exe), "--version"], capture_output=True, text=True, errors="replace", timeout=timeout, cwd=str(Path(exe).parent),
                              env=start_env(exe))
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"it could not be started ({exc})"
    first = ((done.stdout or "").strip().splitlines() or [""])[0]
    if done.returncode != 0 or not first.startswith(NAME):
        tail = ((done.stderr or "").strip().splitlines() or [first or f"exit {done.returncode}"])[-1]
        return False, f"it did not start correctly ({tail[:160]})"
    return True, first


def upgrade(dest: Path, source: str, *, installed: Path | None = None, hardware: dict | None = None,
            emit: Callable[[str], None] = print, cancel=None, same_kind: bool = False) -> Path | None:
    """`_upgrade` (below), one at a time: a second caller (a `strata launch`, an app start, the background fetch) while one runs gets None at once, so two
    of them never write the same download and work folders.  `same_kind`: only a package of the installed one's own platform, backend and variant."""
    with _fetch_lock(Path(dest).parent) as got:
        if not got:
            return None
        return _upgrade(dest, source, installed=installed, hardware=hardware, emit=emit, cancel=cancel, same_kind=same_kind)


def _upgrade(dest: Path, source: str, *, installed: Path | None = None, hardware: dict | None = None,
             emit: Callable[[str], None] = print, cancel=None, same_kind: bool = False) -> Path | None:
    """The installed DeepSeek engine package in `dest` (engine-dsv41/) is older than the signed feed's `dsv41` member for this PC: fetch the newer one,
    check it as `install` does (signature, hashes, every file), put it NEXT TO the old one, start it once (`--version`), and only then remove the older
    folders (one generation).  Returns the new strata-dsv41, or None: nothing newer, the feed cannot be read (silent: a launch never waits on the
    network), or the newer one could not be installed (one plain line, the installed engine keeps running).  Versions compare numerically
    (0.1.49 < 0.1.49.1).  The caller does not call this for a --dsv41-package or a STRATA_DSV41_EXE."""
    from strata_control import feed as feedmod, signing, transfer, updater

    dest = Path(dest)
    current = installed or find(dest)
    if current is None:
        return None
    have_name = Path(current).parent.name
    have = version_key(have_name)
    shown = have_name.split("-")[0]
    work = dest.parent / (DIR + ".work")
    hw = hardware or updater.detect_hardware()
    try:
        (work / "state").mkdir(parents=True, exist_ok=True)
        ctx = updater.Context(components=work, source=source, trusted=signing.load_trusted_keys(), hardware=hw, cancel=cancel,
                              emit=lambda event: None)
        state = updater.State(work)
        try:
            feed = updater.open_feed(ctx, state)
        except feedmod.FeedError as exc:
            if "could not be read" in str(exc):
                return None                                       # no network, no feed here: silent, the launch goes on
            emit(f"A newer DeepSeek engine may be available but could not be checked: {exc}; still using {shown}.")
            return None
        except (updater.UpdateError, OSError):
            return None                                           # (offline mode and the like: silent)
        pkg, _reasons = feedmod.select_engine(list(feed.dsv41), hw)
        if pkg is None or version_key(pkg.version) <= have:
            return None
        if same_kind and kind_of(updater.folder_name(pkg)) != kind_of(have_name):
            return None                                           # (a package for another backend is not an update of this one)
        target = None
        try:
            archive = updater._download(ctx, "dsv41", pkg.url, pkg.size, pkg.sha256, "the DeepSeek engine", feed)
            stage_root, _component = updater.stage_engine(ctx, feed, pkg, archive)
            try:
                folder = stage_root / "engine" / updater.folder_name(pkg)
                target = dest / folder.name
                shutil.rmtree(target, ignore_errors=True)
                shutil.move(str(folder), str(target))
            finally:
                shutil.rmtree(stage_root, ignore_errors=True)
            if os.name != "nt":
                (target / exe_name()).chmod(0o755)
            ok, why = _test_start(target / exe_name())
            if not ok:
                raise updater.UpdateError(why)
        except (updater.UpdateError, feedmod.FeedError, signing.SignatureError, transfer.TransferError, OSError) as exc:
            if target is not None and target.name != have_name:
                shutil.rmtree(target, ignore_errors=True)         # (never the installed one)
            emit(f"A newer DeepSeek engine is available but could not be installed: {str(exc).rstrip('.')}; still using {shown}.")
            return None
        for old in dest.iterdir():                                # switched: the older generations go (one a running engine uses stays)
            if old.is_dir() and old.name != target.name and version_key(old.name) < version_key(target.name) and not in_use(old):
                shutil.rmtree(old, ignore_errors=True)
        emit(f"DeepSeek engine updated from {shown} to {pkg.version}.")
        shutil.rmtree(work, ignore_errors=True)
        return target / exe_name()
    except Exception as exc:                                      # noqa: BLE001 - an upgrade check must never stop a launch
        emit(f"A newer DeepSeek engine may be available but could not be checked ({type(exc).__name__}); still using {shown}.")
        return None


# ---------------------------------------------------------------------------------------------- keeping the engine current for the app

def kind_of(folder_name: str) -> str:
    """The platform-backend-variant part of a package folder's name ("0.1.55.1-windows-x86_64-hip-rdna3" -> "windows-x86_64-hip-rdna3")."""
    return str(folder_name).split("-", 1)[1] if "-" in str(folder_name) else ""


def in_use(folder: str | os.PathLike) -> bool:
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


@contextlib.contextmanager
def _fetch_lock(base: Path):
    """One DeepSeek engine fetch at a time (a lock file beside engine-dsv41/; yields False when a live one holds it).  A lock older than 45 minutes is stale."""
    path = Path(base) / (DIR + ".lock")
    got = False
    try:
        for _ in range(2):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                got = True
                break
            except FileExistsError:
                try:
                    if time.time() - path.stat().st_mtime < 45 * 60:
                        break
                    path.unlink()
                except OSError:
                    break
            except OSError:
                break
        yield got
    finally:
        if got:
            try:
                path.unlink()
            except OSError:
                pass


CHECK_FILE = "feed-check.json"
CHECK_EVERY = 6 * 3600           # a feed check that worked is not repeated for six hours
RETRY_EVERY = 30 * 60            # one that failed (offline, feed unreadable) is not repeated for half an hour
FETCH_WAIT = 90.0                # seconds a start waits for the fetch; after that it goes on with the engine it has and the fetch finishes in the background


def _read_check(dest: Path) -> dict:
    try:
        data = json.loads((Path(dest) / CHECK_FILE).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_check(dest: Path, **fields) -> None:
    try:
        data = {**_read_check(dest), **fields}
        (Path(dest) / CHECK_FILE).write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass


def fetch_due(dest: Path, now: float | None = None) -> bool:
    """Is a feed check due?  Never checked, the last good one more than CHECK_EVERY ago, or the last failed one more than RETRY_EVERY ago."""
    now = time.time() if now is None else now
    check = _read_check(dest)
    at = check.get("at")
    if not isinstance(at, (int, float)):
        return True
    return now - at >= (CHECK_EVERY if check.get("ok") else RETRY_EVERY)


def repoint(config: Path, exe: Path) -> bool:
    """Make the run config `config` name `exe` (a strata-dsv41 in an engine-dsv41 folder) and its library folders, nothing else; True when it changed."""
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


def _engine_root(exe) -> Path | None:
    """The engine-dsv41/ folder a plain run-config exe lies in, or None (a component reference, another engine, an engine of your own)."""
    if not isinstance(exe, str) or not exe or exe.startswith("@") or os.environ.get("STRATA_DSV41_EXE"):
        return None
    root = Path(exe).parent.parent
    return root if root.name == DIR and Path(exe).name == exe_name() else None


def _newest_of_kind(root: Path, kind: str) -> Path | None:
    try:
        same = [d for d in root.iterdir() if d.is_dir() and (d / exe_name()).is_file() and kind_of(d.name) == kind]
    except OSError:
        return None
    same.sort(key=lambda d: (version_key(d.name), d.name))
    return same[-1] / exe_name() if same else None


def _spawn_fetch(root: Path, source: str, installed: Path):
    import subprocess
    import sys
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0
    work = root.parent / (DIR + ".work")
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "fetch.log", "ab") as out:
        return subprocess.Popen([sys.executable, "-m", "strata_control.dsv41_engine", "--fetch", str(root), "--source", source, "--installed", str(installed)],
                                cwd=str(Path(__file__).resolve().parents[1]), stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                creationflags=flags, start_new_session=(os.name != "nt"))


def keep_current(config: str | os.PathLike, *, source: str | None = None, log: Callable[[str], None] = lambda m: None,
                 wait: float = FETCH_WAIT, configs: list | None = None) -> Path | None:
    """Called before the app starts a saved profile (strata_control.runtime.start); never raises, never holds a start for more than `wait` seconds.

    The run config `config` names a DeepSeek engine in engine-dsv41/.  (1) When the feed check is due (fetch_due: once in six hours, half an hour
    after a failure) a newer package is fetched by a separate background process (python -m strata_control.dsv41_engine --fetch); the start waits for
    it `wait` seconds, then goes on with the engine it has and the fetch ends in the background: the next start uses it.  (2) When a newer package of
    the same kind is installed (that fetch, a `strata launch` that updated it, or the folder the config names is gone), `config` and `configs` (the
    other saved profiles' run configs) are pointed at it: local, no network.  (3) Older package folders that no running engine uses go.
    Returns the engine this start will use, or None when this config is not one of ours to look after."""
    try:
        cfg = json.loads(Path(config).read_text(encoding="utf-8-sig"))
        root = _engine_root(cfg.get("exe") if isinstance(cfg, dict) else None)
        if root is None or not root.is_dir():
            return None
        mine = Path(cfg["exe"])
        kind = kind_of(mine.parent.name)
        if mine.is_file() and fetch_due(root):
            from strata_cli import update_cmd
            import argparse
            src = source or update_cmd._feed_source(argparse.Namespace(feed=None), root.parent)
            _write_check(root, at=time.time(), ok=False)         # (a second start in these seconds does not start a second fetch)
            child = _spawn_fetch(root, src, mine)
            try:
                child.wait(timeout=wait)
            except Exception:                                    # noqa: BLE001 - subprocess.TimeoutExpired
                log(f"A newer DeepSeek engine is downloading in the background ({wait:.0f} s passed): this start uses {mine.parent.name.split('-')[0]}, the next one the new engine.")
        best = _newest_of_kind(root, kind)
        if best is not None and best != mine and (not mine.is_file() or version_key(best.parent.name) > version_key(mine.parent.name)):
            for other in [config, *(configs or [])]:
                try:
                    o = json.loads(Path(other).read_text(encoding="utf-8-sig"))
                    if isinstance(o, dict) and _engine_root(o.get("exe")) == root and Path(o["exe"]) != best and (
                            not Path(o["exe"]).is_file() or version_key(Path(o["exe"]).parent.name) < version_key(best.parent.name)):
                        repoint(Path(other), best)
                except (OSError, ValueError):
                    continue
            log(f"DeepSeek engine {best.parent.name.split('-')[0]} is used from now on (it replaced {mine.parent.name.split('-')[0]}).")
            mine = best
        for old in root.iterdir():
            if old.is_dir() and old != mine.parent and kind_of(old.name) == kind and version_key(old.name) < version_key(mine.parent.name) and not in_use(old):
                shutil.rmtree(old, ignore_errors=True)
        return mine
    except Exception as exc:                                      # noqa: BLE001 - a start never fails because of this check
        log(f"The DeepSeek engine check was skipped ({type(exc).__name__}); the installed engine is used.")
        return None


def _fetch_main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="strata_control.dsv41_engine")
    ap.add_argument("--fetch", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--installed", required=True)
    a = ap.parse_args(argv)
    root = Path(a.fetch)
    lines: list[str] = []
    ok = False
    try:
        if not source_reachable(a.source, 3.0):
            lines.append("the update feed could not be reached")
        else:
            upgrade(root, a.source, installed=Path(a.installed), emit=lines.append, same_kind=True)
            ok = not any("could not" in x for x in lines)
    except Exception as exc:                                      # noqa: BLE001
        lines.append(f"{type(exc).__name__}: {exc}")
    _write_check(root, at=time.time(), ok=ok)
    for line in lines:
        print(line, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(_fetch_main(sys.argv[1:]))
