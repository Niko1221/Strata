"""gui/launcher - the Manager's supervisor: launching, stopping and restarting Strata's server.

Platform-neutral: everything OS-specific (spawning a detached server, terminating its process tree) goes
through `gui.platforms.get_launcher()`.  The manager never branches on the OS here; it only records state
(logs/manager.json next to the Strata folder) and decides when a start/stop is safe.

The launched command is the same one setup.py's generated run scripts run
(`python serve/server.py --engine strata --config <cfg> --port <port>`), so a Mac/Windows/Linux Strata
install behaves identically; on Windows the server is spawned without a console window, on Linux in its own
session.  The Manager only supervises servers it started itself.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from gui.platforms import get_launcher

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR_NAME = "logs"                 # runtime state next to the Strata folder (gitignored)
STATE_FILE = "manager.json"
SERVE_LOG_SUFFIX = ".serve.log"         # the Server's stdout when launched by the Manager (strata-*.log)
PROBE_TIMEOUT = 1.0                     # seconds for the "is the port answering?" checks
SHUTDOWN_GRACE_S = 15.0


class ServerState:
    """The Manager's runtime state file (<strata>/logs/manager.json, gitignored): which server process it
    launched, the shutdown/restart lifecycle and the last stop's measurements, and which setup.py helper
    (custom GGUF) it is running."""

    def __init__(self, root: Path = ROOT):
        self.path = root / STATE_DIR_NAME / STATE_FILE

    def read(self) -> dict:
        try:
            st = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st = {}
        return {k: st.get(k) for k in ("server", "tool", "lifecycle", "last_stop")}

    def write(self, st: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)


def set_lifecycle(root: Path, state: str, phase: str) -> None:
    """Record a shutdown/restart lifecycle in the runtime state so /api/status can report it (the UI shows
    Stopping/Restarting + the current phase instead of looking frozen)."""
    st = ServerState(root).read()
    st["lifecycle"] = {"state": state, "phase": phase, "since": time.time()}
    ServerState(root).write(st)


def clear_lifecycle(root: Path) -> None:
    st = ServerState(root).read()
    st.pop("lifecycle", None)
    ServerState(root).write(st)


def read_config(path: Path) -> dict:
    """A strata-*.json config, as JSON (BOM-tolerant, like setup.py)."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def gguf_paths(cfg: dict) -> list:
    """The GGUF shards the config points at (--native and --ple-gguf, in setup.py's order)."""
    args = cfg.get("args", [])
    return [p for p in (arg_val(args, "--native"), arg_val(args, "--ple-gguf")) if p]


def arg_val(args: list, key: str, default=None):
    try:
        i = args.index(key)
        return args[i + 1] if i >= 0 and i + 1 < len(args) else default
    except ValueError:
        return default


def probe_port(host: str, port: int, timeout: float = PROBE_TIMEOUT) -> bool:
    """Is anything listening on host:port?"""
    import socket
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def strata_health(port: int, host: str = "127.0.0.1", timeout: float = PROBE_TIMEOUT,
                  api_key: str = "") -> bool:
    """Strata's own identity check: GET /api/health answers HTTP 200 with JSON service == "strata".
    This is what makes a process on the port *Strata* and nothing else - LM Studio, llama.cpp or any
    other OpenAI-compatible server answers /v1/models the same way but never claims service=strata.
    The configured key is sent when the config has one, so a key-protected install is still
    recognised."""
    import json as _json
    import urllib.request
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    try:
        req = urllib.request.Request(f"http://{host}:{port}/api/health", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                return False
            try:
                body = _json.loads(r.read(65536))
            except ValueError:
                return False
            return body.get("service") == "strata"
    except (OSError, ValueError):
        return False


def strata_ready(port: int, host: str = "127.0.0.1", timeout: float = PROBE_TIMEOUT,
                 api_key: str = "") -> bool:
    """Readiness, not identity: an OpenAI-style GET /v1/models answers (Strata's own key sent when
    the config has one).  Used for the status's `ready` field - deciding that something IS Strata is
    strata_health()'s job, never this."""
    import urllib.request
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    try:
        req = urllib.request.Request(f"http://{host}:{port}/v1/models", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200 and b'"object"' in r.read(65536)
    except (OSError, ValueError):
        return False


def _config_for_port(root: Path, port: int) -> dict | None:
    """The installed config that listens on `port` (most recently used first) - for its API key."""
    configs = sorted(root.glob("strata-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in configs:
        try:
            cfg = read_config(p)
        except (OSError, ValueError):
            continue
        if cfg.get("port", 8080) == port:
            return cfg
    return None


def external_strata(root: Path = ROOT) -> dict | None:
    """A Strata server this Manager did NOT start (START-HERE.bat / run-*.bat): one of the installed
    configs' ports answers /api/health with service == "strata" (the config's key sent when it has
    one).  An arbitrary OpenAI-compatible server on the port (LM Studio, plain llama.cpp) is NOT
    classified as Strata.  Returns {"port", "config", "ready"} of the first match."""
    configs = sorted(root.glob("strata-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in configs:
        try:
            cfg = read_config(p)
            port = cfg.get("port", 8080)
        except (OSError, ValueError):
            continue
        if not probe_port("127.0.0.1", port):
            continue
        key = cfg.get("api_key", "") or ""
        if not strata_health(port, api_key=key):
            continue
        return {"port": port, "config": p.name, "ready": strata_ready(port, api_key=key)}
    return None


def pid_alive(pid: int) -> bool:
    return get_launcher().alive(pid)


def server_status(root: Path = ROOT) -> dict:
    """Lifecycle-aware status:
      stopped | starting | running | stopping | restarting | error.
    A shutdown/restart in progress (recorded by stop_server / restart_server) reports its own state and the
    current phase (graceful | forcing | starting | running) with the elapsed seconds, so the UI can show
    "Stopping… 4.2s" instead of guessing.  Otherwise the port is the ground truth (Strata answers it once the
    model is loaded); the remembered PID distinguishes "starting" from "stopped"; a server the Manager
    started that is gone without a clean stop is an "error" (it crashed).  A Strata the Manager did NOT
    start (its port answers /api/health with service == "strata") is "external": the Manager only
    supervises what it started itself - an arbitrary OpenAI-compatible server is never claimed."""
    st = ServerState(root).read()
    srv = st.get("server") or {}
    lc = st.get("lifecycle") or {}
    pid, port, name = srv.get("pid"), srv.get("port"), srv.get("config")
    if name:
        try:
            port = port or read_config(root / name).get("port", 8080)
        except (OSError, ValueError):
            pass
    now = time.time()
    if lc.get("state") in ("stopping", "restarting"):
        since = float(lc.get("since") or now)
        return {"state": lc["state"], "phase": lc.get("phase", "graceful"),
                "since": since, "elapsed": round(max(0.0, now - since), 1),
                "config": name, "port": port, "log": srv.get("log")}
    if port and probe_port("127.0.0.1", port):
        key = (_config_for_port(root, port) or {}).get("api_key", "") or ""
        if pid_alive(pid):
            return {"state": "running", "port": port, "pid": pid, "config": name,
                    "ready": strata_ready(port, api_key=key), "log": srv.get("log")}
        # our process is gone but the port still answers.  Only claim it when its own health identity
        # says it IS Strata (with the config's key); anything else falls through to the crash report.
        if strata_health(port, api_key=key):
            return {"state": "external", "port": port, "config": name,
                    "ready": strata_ready(port, api_key=key), "log": srv.get("log"),
                    "external": True}
    if pid_alive(pid):
        return {"state": "starting", "port": port, "pid": pid, "config": name, "ready": False,
                "log": srv.get("log")}
    if srv.get("pid"):                                   # started then died on its own (crash, import error...)
        # unless an external instance answered anyway (checked above on the recorded port) - else report the crash
        ext = external_strata(root)
        if ext:
            return {"state": "external", "port": ext["port"], "config": ext.get("config"),
                    "ready": True, "log": None, "external": True}
        return {"state": "error", "port": port, "config": name, "ready": False,
                "log": srv.get("log"),
                "error": "the server exited before answering (see its log)"}
    ext = external_strata(root)                           # no Manager-owned server: is Strata up anyway?
    if ext:
        return {"state": "external", "port": ext["port"], "config": ext.get("config"),
                "ready": True, "log": None, "external": True}
    return {"state": "stopped", "port": None, "config": None, "ready": False,
            "last_stop": st.get("last_stop")}


def start_server(name: str, port: int | None, root: Path = ROOT, open_chat: bool = False) -> dict:
    """Launch Strata's server for an installed model the way run-*.bat / run-*.sh do (same serve/server.py,
    same config), detached, with its stdout in strata-<m>.serve.log.  Port-lock checks first - Strata's
    server refuses to start twice by itself; this must not even try."""
    cfg_path = root / name
    try:
        cfg = read_config(cfg_path)
    except (OSError, ValueError):
        return {"ok": False, "error": f"cannot read {name} - is it a Strata config?"}
    port = port or cfg.get("port", 8080)
    status = server_status(root)
    if status["state"] in ("running", "starting"):
        return {"ok": False, "state": status["state"],
                "error": f"Strata is already {status['state']} (config {status.get('config')})."}
    if status["state"] == "external":
        return {"ok": False, "state": "external",
                "error": "Strata is already running, started outside the Manager - close its console "
                "first, the Manager never touches a process it did not start."}
    if probe_port("127.0.0.1", port):
        return {"ok": False, "error": f"something else is already listening on port {port}: another Strata "
                "console, or another local server. Close it first - the Manager must not launch a second "
                "instance."}
    if not Path(cfg.get("exe", "")).exists():
        return {"ok": False, "error": f"the engine in this config is missing: {cfg.get('exe')} - run "
                "SETUP.bat / setup.sh to repair the install."}
    missing = [p for p in gguf_paths(cfg) if not Path(p).exists()]
    if missing:
        return {"ok": False, "error": f"this config refers to missing model files: {missing[0]} - run "
                "the setup again to repair it."}
    cmd = [sys.executable, str(root / "serve" / "server.py"), "--engine", "strata",
           "--config", str(cfg_path), "--port", str(port)]
    log_path = root / (cfg_path.name[:-5] + SERVE_LOG_SUFFIX)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    launcher = get_launcher()
    pid = launcher.spawn(cmd, str(root), log_path)
    ServerState(root).write({**ServerState(root).read(), "server": {
        "pid": pid, "config": name, "port": port, "log": str(log_path),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cmd": cmd, "platform": launcher.name}})
    if open_chat:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{port}/")
    return {"ok": True, "state": "starting", "pid": pid, "port": port, "log": str(log_path)}


def stop_server(root: Path = ROOT) -> dict:
    """Stop the server the Manager started, through the platform adapter (Windows: close the process tree
    like closing Strata's console; Linux: graceful SIGTERM to the server's group, force after grace).
    Reports the measured shutdown time and whether the force path was needed."""
    st = ServerState(root).read()
    srv = st.get("server") or {}
    pid = srv.get("pid")
    if not pid:
        # not started by the Manager: maybe another console; only the port can tell
        for name in sorted(p.name for p in (root.glob("strata-*.json")) if p.is_file()):
            try:
                port = read_config(root / name).get("port", 8080)
            except (OSError, ValueError):
                continue
            if probe_port("127.0.0.1", port):
                return {"ok": True, "state": "running", "note": "Strata is running from another window; close "
                        "that window to stop it (the Manager only stops what it started)."}
        return {"ok": True, "state": "stopped", "note": "not running", "elapsed_s": 0.0, "forced": False}
    set_lifecycle(root, "stopping", "graceful")
    t0 = time.time()
    res = get_launcher().terminate(int(pid), SHUTDOWN_GRACE_S)
    elapsed = round(time.time() - t0, 2)
    _finish_stop(root, elapsed=elapsed, forced=bool(res.get("forced")))
    return {"ok": True, "state": "stopped", "elapsed_s": elapsed, "forced": bool(res.get("forced"))}


def force_stop_server(root: Path = ROOT) -> dict:
    """Force Stop: the hard process-tree termination immediately (the platform's terminate_force), skipping
    the graceful phase.  Only what the Manager started is touched."""
    st = ServerState(root).read()
    pid = (st.get("server") or {}).get("pid")
    if not pid:
        return {"ok": True, "state": "stopped", "note": "not running", "elapsed_s": 0.0, "forced": True}
    set_lifecycle(root, "stopping", "forcing")
    t0 = time.time()
    res = get_launcher().terminate_force(int(pid))
    elapsed = round(time.time() - t0, 2)
    _finish_stop(root, elapsed=elapsed, forced=True, stopped=bool(res.get("stopped")))
    return {"ok": True, "state": "stopped", "elapsed_s": elapsed, "forced": True,
            "stopped": bool(res.get("stopped"))}


def _finish_stop(root: Path, elapsed=None, forced=None, stopped=None) -> None:
    """After a server is gone: drop the server record and the lifecycle, remember the stop measurements."""
    st = ServerState(root).read()
    last = st.get("last_stop") or {}
    last.update({"elapsed_s": elapsed, "forced": forced, "stopped": stopped, "at": time.strftime("%H:%M:%S")})
    st.pop("server", None)
    st.pop("lifecycle", None)
    st["last_stop"] = last
    ServerState(root).write(st)


def restart_server(name: str, port: int | None, root: Path = ROOT, open_chat: bool = False) -> dict:
    """Restart = Stop then Start, with the lifecycle going Restarting(stopping) → Restarting(starting) →
    Starting → Running so the UI shows the whole thing instead of a frozen state."""
    set_lifecycle(root, "restarting", "stopping")
    t0 = time.time()
    _stop_internal(root)
    set_lifecycle(root, "restarting", "starting")
    started = start_server(name, port, root, open_chat=open_chat)
    if not started.get("ok"):
        clear_lifecycle(root)
        return {**started, "elapsed_s": round(time.time() - t0, 2), "forced": False}
    clear_lifecycle(root)                                 # the next status poll sees starting → running
    return {"ok": True, "state": "restarting", "elapsed_s": round(time.time() - t0, 2), "forced": False}


def _stop_internal(root: Path) -> None:
    """The stop half of a restart: same termination, without touching the restart lifecycle."""
    st = ServerState(root).read()
    pid = (st.get("server") or {}).get("pid")
    if not pid:
        return
    get_launcher().terminate(int(pid), SHUTDOWN_GRACE_S)
    _finish_stop(root)


def run_tool(cmd: list, log_path: Path, what: str, root: Path = ROOT) -> dict:
    """Launch a detached helper (today: setup.py's own custom-GGUF install), remember it in the runtime state,
    stream its output into a log the UI can tail.  Returns the recorded tool description."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid = get_launcher().spawn(cmd, str(root), log_path)
    st = ServerState(root).read()
    st["tool"] = {"pid": pid, "log": str(log_path), "what": what, "started": time.time()}
    ServerState(root).write(st)
    return st["tool"]


def tail_lines(path: str | Path, n: int = 200) -> str:
    """The last n lines of a log (server/engine logs are plain text)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 131072))                # at most 128 KB read
            data = f.read().decode("utf-8", "replace")
        return "\n".join(data.splitlines()[-n:])
    except OSError:
        return ""