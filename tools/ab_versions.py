"""Strata release helpers for ab_engine's versions mode: find engine/control pairs, resolve version specs, and fetch packages.

Packages come only from a download into the bench cache. Nothing is ever read from the real Strata install."""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

VER = r"(\d+(?:\.\d+)*)"
CHUNK = 1 << 20
# The signed feed's size cap is not available for a release asset; the sha256 check is the integrity guard, this only stops a bomb.
MAX_UNPACKED = 1 << 40


def _control_on_path() -> None:
    """Make `strata_control` importable (repo root on sys.path); the import itself stays lazy."""
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


@dataclass(frozen=True)
class ReleasePair:
    version: str
    tag: str
    engine_name: str
    engine_url: str
    control_name: str
    control_url: str
    sums_url: str


def _key(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in version.split("."))


def pairs_from_releases(releases: list[dict], target: str, backend: str) -> dict[str, ReleasePair]:
    """Map version -> ReleasePair for `target`/`backend`; newest release first wins; engine and control both needed."""
    engine_re = re.compile(rf"strata-engine-{VER}-{re.escape(target)}-{re.escape(backend)}\.tar\.gz")
    control_re = re.compile(rf"strata-control-{VER}-{re.escape(target)}\.(?:zip|tar\.gz)")
    pairs: dict[str, ReleasePair] = {}
    for rel in releases:
        assets = rel.get("assets", [])
        sums = next((x for x in assets if x["name"] == "SHA256SUMS.txt"), None)
        engines = {m[1]: x for x in assets if (m := engine_re.fullmatch(x["name"]))}
        controls = {m[1]: x for x in assets if (m := control_re.fullmatch(x["name"]))}
        for ver in engines.keys() & controls.keys():
            if ver in pairs or sums is None:
                continue
            e, c = engines[ver], controls[ver]
            pairs[ver] = ReleasePair(ver, rel["tag_name"], e["name"], e["browser_download_url"],
                                     c["name"], c["browser_download_url"], sums["browser_download_url"])
    return pairs


def resolve_versions(spec: str, available: dict[str, ReleasePair]) -> list[str]:
    """`a,b` or `a..b` (inclusive) -> versions in ascending order; ValueError names versions that have no pair."""
    if ".." in spec:
        lo, hi = (_key(x.strip()) for x in spec.split("..", 1))
        found = [v for v in available if lo <= _key(v) <= hi]
        if not found:
            have = ", ".join(sorted(available, key=_key))
            raise ValueError(f"no release with engine and control in range {spec}; have: {have}")
    else:
        found = [v.strip() for v in spec.split(",") if v.strip()]
        missing = [v for v in found if v not in available]
        if missing:
            raise ValueError(f"no release with engine and control for: {', '.join(missing)}")
    return sorted(set(found), key=_key)


def list_releases(repo: str = "Niko1221/Strata", opener=urllib.request.urlopen) -> list[dict]:
    """All releases of `repo`, newest first, via the public API (no token)."""
    out: list[dict] = []
    page = 1
    while True:
        req = urllib.request.Request(f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}",
                                     headers={"Accept": "application/vnd.github+json", "User-Agent": "strata-ab-engine"})
        with opener(req, timeout=30) as r:
            batch = json.load(r)
        if not batch:
            return out
        out += batch
        page += 1


@dataclass(frozen=True)
class Packages:
    version: str
    engine_dir: Path
    control_dir: Path


def parse_sums(text: str) -> dict[str, str]:
    """`sha256sum` output -> {file name: sha256}; accepts the binary-mode `*` marker."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[1].strip().lstrip("*")] = parts[0].lower()
    return out


def _stem(name: str) -> str:
    return name[:-len(".tar.gz")] if name.endswith(".tar.gz") else name[:-len(".zip")]


def _fetch(url: str, dest: Path, opener) -> str:
    """Stream `url` into `dest`; returns its sha256. Removes `dest` if the transfer fails."""
    h = hashlib.sha256()
    try:
        with opener(url, timeout=60) as r, dest.open("wb") as f:
            while chunk := r.read(CHUNK):
                h.update(chunk)
                f.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return h.hexdigest()


def _ensure_one(name: str, url: str, sums: dict[str, str] | None, cache: Path, opener, *, allow_links: bool) -> Path:
    from strata_control import updater
    target = cache / _stem(name)
    if (target / ".ok").is_file():
        return target
    if target.exists():
        raise RuntimeError(f"{target} exists without a .ok marker; remove it and run again")
    part = cache / f"{name}.part"
    try:
        got = _fetch(url, part, opener)
        want = sums.get(name)
        if want is None:
            raise ValueError(f"no sha256 for {name} in SHA256SUMS.txt")
        if got != want:
            raise ValueError(f"sha256 mismatch for {name}: expected {want}, got {got}")
        tmp = cache / f"{target.name}.tmp"
        try:
            updater.extract(part, tmp, "zip" if name.endswith(".zip") else "tar.gz", MAX_UNPACKED, allow_links=allow_links)
            (tmp / ".ok").write_bytes(b"")          # the marker goes in before the rename, so a folder never shows up unmarked
            tmp.rename(target)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    finally:
        part.unlink(missing_ok=True)
    return target


def ensure_packages(pair: ReleasePair, cache: Path, opener=urllib.request.urlopen) -> Packages:
    """Engine and control of `pair` unpacked under `cache/<asset-stem>/`, downloaded and sha256-checked if not already there.
    A folder with its `.ok` marker is a cache hit and needs no network. A failed download or check leaves nothing behind."""
    _control_on_path()
    cache.mkdir(parents=True, exist_ok=True)
    needed = [n for n in (pair.engine_name, pair.control_name) if not (cache / _stem(n) / ".ok").is_file()]
    sums = None
    if needed:
        with opener(pair.sums_url, timeout=60) as r:
            sums = parse_sums(r.read().decode("utf-8"))
    engine = _ensure_one(pair.engine_name, pair.engine_url, sums, cache, opener, allow_links=False)
    control = _ensure_one(pair.control_name, pair.control_url, sums, cache, opener, allow_links=pair.control_name.endswith(".tar.gz"))
    return Packages(pair.version, engine, control)


def isolated_env(home: Path, base: Mapping[str, str]) -> dict[str, str]:
    """`base` with every per-user folder under `home` and every STRATA_ variable dropped. STRATA_NO_ACQUIRE is set, but
    `strata launch` drops it for its children (launch may download); version_problem is the real guard against another engine."""
    env = {k: v for k, v in base.items() if not k.startswith("STRATA_")}
    for key, sub in (("HOME", ""), ("USERPROFILE", ""), ("APPDATA", "AppData/Roaming"), ("LOCALAPPDATA", "AppData/Local"),
                     ("XDG_STATE_HOME", ".local/state"), ("XDG_DATA_HOME", ".local/share"),
                     ("XDG_CONFIG_HOME", ".config"), ("XDG_CACHE_HOME", ".cache")):
        env[key] = str(home / sub)
    env["STRATA_NO_ACQUIRE"] = "1"
    return env


def provision_home(home: Path, packages: Packages) -> Path:
    """Put the engine and control of `packages` where the release Control looks for them under `home`; returns the control folder.
    Idempotent: a folder that is already there is reused, nothing is ever deleted."""
    _control_on_path()
    from strata_control.engine_component import components_dir
    components = components_dir(isolated_env(home, {}))
    engine = components / "engine" / packages.engine_dir.name
    control = components / "control" / packages.version
    for src, dest in ((packages.engine_dir, engine), (packages.control_dir, control)):
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dest, ignore=shutil.ignore_patterns(".ok"))     # .ok is the download cache's marker, not package content
    (control.parent / "active.json").write_text(json.dumps({"version": packages.version, "folder": packages.version}), encoding="utf-8")
    return control


class Busy(RuntimeError):
    """A live Strata server or a busy port: running a bench now would disturb it."""


def assert_idle(ports: Iterable[int], serving=None) -> None:
    """Raise `Busy` if Strata is serving or something accepts connections on 127.0.0.1:<port> for any of `ports`."""
    if serving is None:
        _control_on_path()
        from strata_control import updater
        serving = updater.serving_now
    if serving():
        raise Busy("a Strata server is already serving; stop it first (`strata launch --stop`)")
    for port in ports:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
        except OSError:                                                     # refused (or unreachable) = nobody listens
            continue
        raise Busy(f"port {port} on 127.0.0.1 is in use")


def snapshot(roots: Iterable[Path]) -> dict[str, tuple[int, int]]:
    """Every file under `roots` -> (size, mtime_ns); keys are `<root>/<relative path>` so equal names in two roots never collide.
    A root that does not exist adds nothing."""
    out: dict[str, tuple[int, int]] = {}
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                path = Path(dirpath, name)
                try:
                    st = path.stat()
                except OSError:                                             # vanished between listing and stat
                    continue
                out[str(root / path.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return out


def changed(before: dict, after: dict) -> list[str]:
    """Differences between two snapshots: `+new`, `-gone`, `~modified`, each group sorted, in that order."""
    return ([f"+{k}" for k in sorted(after.keys() - before.keys())] + [f"-{k}" for k in sorted(before.keys() - after.keys())]
            + [f"~{k}" for k in sorted(k for k in before.keys() & after.keys() if before[k] != after[k])])


def live_roots(env: Mapping[str, str], model_path: Path) -> list[Path]:
    """Every real Strata folder a bench must leave alone, derived from `env`: the component folder, the state folder, the profile folder,
    `%APPDATA%/Strata` (Windows), `~/.strata`, and the Strata data folder that holds `model_path`. The data folder counts only when it
    looks like one (a `models` folder next to `packs` or `mtp`); a drive root or the home folder is never returned."""
    _control_on_path()
    from strata_control.engine_component import components_dir
    from strata_control.paths import state_base
    home = Path(env.get("USERPROFILE") or env.get("HOME") or Path.home())
    if os.name == "nt":
        profiles = Path(env.get("APPDATA") or home / "AppData" / "Roaming") / "Strata"
    else:
        profiles = Path(env.get("XDG_CONFIG_HOME") or home / ".config") / "strata"
    roots = [components_dir(env), state_base(env, home=home), profiles, home / ".strata"]
    if env.get("STRATA_PROFILE_DIR"):
        roots.append(Path(env["STRATA_PROFILE_DIR"]))
    for parent in Path(model_path).parents:
        if parent.name.lower() == "models":
            data = parent.parent
            if data != data.parent and data != home and any((data / sub).is_dir() for sub in ("packs", "mtp")):
                roots.append(data)
            break
    return list(dict.fromkeys(roots))


@dataclass
class LaunchResult:
    ok: bool
    seconds: float
    reason: str = ""


def control_command(control_dir: Path) -> list[str]:
    """How to start the release Control in `control_dir`. On Windows its own Python runs `strata.py` directly when the package
    carries one: `bin/strata.cmd` goes through cmd.exe, which mangles `&`, `%` and `^` in arguments (a model path is an argument)."""
    if os.name != "nt":
        return [str(control_dir / "bin" / "strata")]
    py = control_dir / "python" / "python.exe"
    return [str(py), str(control_dir / "strata.py")] if py.exists() else [str(control_dir / "bin" / "strata.cmd")]


def _run(control_dir: Path, env: Mapping[str, str], args: list[str], timeout: float) -> tuple[int | None, str]:
    """Run the Control with `args`; returns (exit code or None on timeout, stdout + stderr). Output goes to files, not pipes:
    `launch --no-tui` leaves a server running that could keep a pipe open."""
    env = {k: v for k, v in env.items() if k not in ("PYTHONHOME", "PYTHONPATH")} | {"PYTHONUTF8": "1", "PYTHONNOUSERSITE": "1"}
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            code = subprocess.run(control_command(control_dir) + args, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                  timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = None
        text = ""
        for f in (out, err):
            f.seek(0)
            text += f.read().decode("utf-8", "replace") + "\n"
    return code, text


def _last_line(text: str) -> str:
    """The error message of the last `--output json` failure line (stdout, before stderr noise), else the last non-empty line."""
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            continue
    return lines[-1] if lines else ""


def _engine_logs(home: Path) -> list[Path]:
    return sorted(home.rglob("*.engine.log"), key=lambda p: p.stat().st_mtime_ns, reverse=True)


def launch(control_dir: Path, env: dict, model_path: Path, data_dir: Path, port: int, tune: str | None, timeout: float = 7200,
           opener=urllib.request.urlopen) -> LaunchResult:
    """`strata launch` with the release's own defaults (and its tuning when `tune`); ok = exit 0 and /health says ok."""
    args = ["launch", "--model-path", str(model_path), "--data-dir", str(data_dir), "--port", str(port),
            "--yes", "--no-tui", "--output", "json"] + (["--tune", tune, "--retune"] if tune else [])
    start = time.monotonic()
    code, text = _run(control_dir, env, args, timeout)
    seconds = time.monotonic() - start
    if code == 0:
        try:
            with opener(f"http://127.0.0.1:{port}/health", timeout=30) as r:
                status = json.loads(r.read()).get("status")
        except (OSError, ValueError, AttributeError) as e:
            status = f"error {e}"
        if status == "ok":
            return LaunchResult(True, seconds)
        return LaunchResult(False, seconds, f"health check: {status}")
    reason = _last_line(text)
    if not reason:
        logs = _engine_logs(Path(env["HOME"]))
        reason = _last_line(logs[0].read_text(encoding="utf-8", errors="replace")) if logs else ""
    if code is None:
        reason = f"timed out after {timeout:.0f} s" + (f": {reason}" if reason else "")
    return LaunchResult(False, seconds, reason or f"exit {code}")


def stop(control_dir: Path, env: dict) -> None:
    """`strata launch --stop`; a failure is printed, never raised."""
    try:
        code, text = _run(control_dir, env, ["launch", "--stop"], 300)
    except OSError as e:
        code, text = -1, str(e)
    if code != 0:
        print(f"strata launch --stop failed ({code}): {_last_line(text)}", file=sys.stderr)


ENGINE_STARTED = re.compile(r'engine started: (?:"([^"]+)"|(.+?)) --serve')


def served_engine(home: Path) -> Path | None:
    """The folder of the engine the newest `engine started: <exe> --serve` line under `home` names (serve/server.py writes it)."""
    for log in _engine_logs(home):
        found = ENGINE_STARTED.findall(log.read_text(encoding="utf-8", errors="replace"))
        if found:
            return Path(found[-1][0] or found[-1][1]).parent
    return None


def version_problem(home: Path, packages: Packages) -> str | None:
    """None when the served engine is the one `packages` provisioned, else why the arm must be skipped."""
    served = served_engine(home)
    if served is not None and served.name == packages.engine_dir.name:
        return None
    return f"served engine {served.name if served else 'unknown'}, expected {packages.engine_dir.name}"


# Windows state is %LOCALAPPDATA%/Strata, POSIX ~/.local/state/strata; tuned presets and profiles live under %APPDATA%/Strata or
# ~/.config/strata; native-tune is written by the desktop host only. fnmatch patterns for "/" + the lower-cased relative path
# (`*` also crosses "/"), so no case-insensitive glob (Python 3.12) is needed.
CONFIG_GLOBS = ("*/strata/launch/*.json", "*/strata/tuning/*", "*/strata/presets/*", "*/strata/profiles/*",
                "*/.strata/data/native-tune/*")


def collect_configs(home: Path, dest: Path) -> list[Path]:
    """Copy every config file under `home` (CONFIG_GLOBS, no logs) to `dest` at its relative path; never overwrites
    (an existing copy gets the next `.1`, `.2` suffix). Returns the copies."""
    found = {p for p in home.rglob("*") if p.suffix != ".log" and p.is_file()
             and any(fnmatch.fnmatchcase("/" + p.relative_to(home).as_posix().lower(), pat) for pat in CONFIG_GLOBS)}
    copies = []
    for src in sorted(found):
        target = dest / src.relative_to(home)
        n, base = 0, target
        while target.exists():
            n += 1
            target = base.with_name(f"{base.name}.{n}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copies.append(target)
    return copies
