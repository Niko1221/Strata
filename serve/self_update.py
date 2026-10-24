"""serve/self_update.py - the web page's Updates card: `strata update` from the running server, then the same server again.

    GET  /update                      what the card shows: can this server update itself, the last check, a running update
    POST /update {"action": "check"}  `strata update --check`: reads the signed feed, changes nothing
    POST /update {"action": "install"}

An install stops taking new requests (503, as a draining server does), waits for the running ones, unloads the model so
the update's test start has the GPU, and runs `strata update --model-config <this server's run config>`. When something
was updated, the server stops the usual way (QUIT to the engine, MCP servers ended) and starts again with the same
command line, environment and folder, so a script such as run-agents.sh needs no second run; the run config's
"component:strata-engine" then names the new engine. When nothing changed, or the update failed (the old version is
still installed), the server takes requests again and loads the model with the next one.

Only a server started directly (serve/server.py, a run-<model> script) on Linux or a Mac can do this: `strata serve`,
`strata launch` and the app start it from a process of their own, which a restart in place would not reach; on Windows
a process cannot start itself again in place.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

DRAIN_S = 300.0                 # how long running requests may take to finish before they are cancelled
CHECK_TIMEOUT_S = 120.0
LOG_LINES = 30


def find_strata() -> str | None:
    """The `strata` command (the Terminal's installer puts it in ~/.local/bin, which a service's PATH may lack)."""
    found = shutil.which("strata")
    if found:
        return found
    home = Path.home() / ".local" / "bin" / "strata"
    return str(home) if os.access(home, os.X_OK) else None


def unsupported_reason(*, supervised: bool, engine_kind: str, exe, strata: str | None, environ=None) -> str | None:
    """Why this server cannot update and restart itself, or None when it can."""
    from strata_control import engine_component
    environ = os.environ if environ is None else environ
    if os.name != "posix":
        return "On Windows, run `strata update` in a terminal, then start the server again."
    if supervised:
        return ("This server was started by `strata serve`, `strata launch` or Strata's app: stop it, run `strata update`, "
                "and start it again.")
    if engine_kind == "strata" and not engine_component.is_reference(exe):
        return (f"This server runs {exe}, not the engine package that `strata update` installs (\"exe\": "
                "\"component:strata-engine\" in the run config), so an update would not change it.")
    if not strata:
        return "The `strata` command is not installed (docs/TERMINAL.md)."
    if environ.get("STRATA_NO_ACQUIRE"):
        return "STRATA_NO_ACQUIRE is set for this server, so it downloads nothing."
    return None


def _summary(report: dict) -> dict:
    """The parts of `strata update --check --output json` the card shows."""
    eng, ctl = report.get("engine") or {}, report.get("control") or {}
    engine = {"installed": (eng.get("installed") or {}).get("version"), "offered": (eng.get("offered") or {}).get("version"),
              "status": eng.get("status"), "download_bytes": eng.get("download_bytes"), "note": eng.get("note")}
    control = {"installed": ctl.get("installed"), "offered": ctl.get("offered"), "status": ctl.get("status"),
               "download_bytes": ctl.get("download_bytes")}
    return {"engine": engine, "control": control,
            "available": engine["status"] in ("available", "install") or control["status"] == "available",
            "serving": [item.get("profile", "an engine") for item in report.get("serving") or []],
            "checked_at": time.time()}


def _error_of(stdout: str, stderr: str, code: int) -> str:
    """The message of a failed `strata update` (its JSON error, else its last line of stderr)."""
    for line in reversed(stdout.strip().splitlines()):
        try:
            body = json.loads(line)
        except ValueError:
            continue
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            return str(body["error"].get("message") or f"exit code {code}")
    tail = stderr.strip().splitlines()[-1:] or [f"exit code {code}"]
    return tail[0].removeprefix("strata: ")


class SelfUpdate:
    """One check or install at a time; the page polls view()."""

    def __init__(self, svc, *, strata: str | None, reason: str | None, config: str | None, popen=subprocess.Popen):
        self.svc, self.strata, self.reason, self.config = svc, strata, reason, config
        self.popen = popen
        self.lock = threading.Lock()
        self.state = "idle"                         # idle | checking | updating | restarting | failed
        self.message = ""
        self.log: list[str] = []
        self.check_result: dict | None = None
        self.results: list[dict] = []
        self.restart_wanted = False                 # main() watches it: the usual stop, then restart()

    def view(self) -> dict:
        with self.lock:
            return {"supported": self.reason is None, "reason": self.reason, "state": self.state, "message": self.message,
                    "log": list(self.log), "check": self.check_result, "results": list(self.results),
                    "instance": getattr(self.svc, "instance_id", None)}

    def _set(self, state: str | None = None, message: str | None = None) -> None:
        with self.lock:
            if state is not None:
                self.state = state
            if message is not None:
                self.message = message
                if message and (not self.log or self.log[-1] != message):
                    self.log = [*self.log, message][-LOG_LINES:]

    def begin(self, action: str) -> tuple[int, dict]:
        """POST /update: start a check or an install on a thread of its own. -> (HTTP status, body)."""
        if action not in ("check", "install"):
            return 400, {"error": {"type": "invalid_request_error", "message": "action: \"check\" or \"install\""}}
        if self.reason:
            return 409, {"error": {"message": self.reason}}
        with self.lock:
            if self.state in ("checking", "updating", "restarting"):
                return 409, {"error": {"message": f"an update step is already running ({self.state})"}}
            self.state = "checking" if action == "check" else "updating"
            self.log, self.results = [], []
            self.message = "Reading the update feed ..." if action == "check" else "Starting the update ..."
        threading.Thread(target=self._check if action == "check" else self._install, daemon=True,
                         name=f"strata-update-{action}").start()
        return 202, self.view()

    # ----------------------------------------------------------------------------------------------- check

    def _check(self) -> None:
        try:
            done = subprocess.run([self.strata, "update", "--check", "--output", "json"], capture_output=True, text=True,
                                  timeout=CHECK_TIMEOUT_S, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError) as e:
            self._set("failed", f"`strata update --check` could not run: {e}")
            return
        if done.returncode != 0:
            self._set("failed", f"The check failed: {_error_of(done.stdout, done.stderr, done.returncode)}")
            return
        try:
            summary = _summary(json.loads(done.stdout))
        except (ValueError, AttributeError):
            self._set("failed", "`strata update --check` answered something that is not its JSON report.")
            return
        with self.lock:
            self.check_result = summary
        self._set("idle", "An update is available." if summary["available"] else "Everything is up to date.")

    # ----------------------------------------------------------------------------------------------- install

    def _resume(self, state: str, message: str) -> None:
        self.svc.end_drain()
        self._set(state, message)

    def _install(self) -> None:
        svc = self.svc
        pending = svc.begin_drain()
        if pending:
            self._set(message=f"Waiting for {pending} running request(s) to finish (new ones get 503 until the update is over) ...")
            if svc.wait_for_drain(DRAIN_S):
                self._set(message=f"Still running after {DRAIN_S:.0f} s: cancelling them ...")
                svc.cancel_owned_requests()
                svc.wait_for_drain(10)
        self._set(message="Unloading the model so the update's test start has the GPU ...")
        try:
            unloaded = svc.unload_waiting(30.0)
        except Exception as e:                      # noqa: BLE001 - EngineStuck and the like: say it, keep serving
            self._resume("failed", f"The model could not be unloaded ({e}); nothing was updated.")
            return
        if unloaded == "busy":
            self._resume("failed", "A request is still using the model; nothing was updated. Try again when it is idle.")
            return
        args = [self.strata, "update", "--output", "jsonl"]
        if self.config:
            args += ["--model-config", str(Path(self.config).resolve())]
        try:
            proc = self.popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True,
                              bufsize=1)
        except OSError as e:
            self._resume("failed", f"`strata update` could not run: {e}")
            return
        stderr: list[str] = []
        reader = threading.Thread(target=lambda: stderr.extend(proc.stderr), daemon=True, name="strata-update-stderr")
        reader.start()
        lines, final = [], None
        for line in proc.stdout:
            lines.append(line)
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if "results" in event:
                final = event
            elif event.get("event") == "progress":
                self._set(message=f"Downloading {event.get('label', '')}: {event.get('percent', 0)}%")
            elif event.get("message"):
                self._set(message=str(event["message"]))
        code = proc.wait()
        reader.join(5)
        proc.stdout.close()
        proc.stderr.close()
        print(f"[strata] `strata update` exited {code}", flush=True)
        if final is None:
            self._resume("failed", f"The update did not finish: {_error_of(''.join(lines), ''.join(stderr), code)}. "
                                   "The installed version is still in use.")
            return
        results = [{"component": r.get("component"), "status": r.get("status"), "error": r.get("error"),
                    "before": _version(r.get("installed_before")), "after": _version(r.get("installed_after"))}
                   for r in final.get("results") or []]
        with self.lock:
            self.results = results
            self.check_result = None
        failed = [r for r in results if r["status"] == "failed"]
        if any(r["status"] == "updated" for r in results):
            self._set("restarting", "Updated. Starting the server again with the same settings ...")
            print("[strata] update installed; restarting the server", flush=True)
            self.restart_wanted = True
        elif failed:
            self._resume("failed", "Not updated: " + "; ".join(f"{r['component']}: {r['error']}" for r in failed)
                         + ". The installed version is still in use.")
        else:
            self._resume("idle", "Nothing to update. The model loads again with the next request.")


def _version(value) -> str | None:
    return value.get("version") if isinstance(value, dict) else value


def restart(environ: dict, cwd: str) -> None:
    """Replace this process with a new start of the same server: the interpreter, its arguments, the environment and
    the folder it began with. The PID stays, so a script or service that waits on it sees no exit."""
    atexit._run_exitfuncs()                          # exec skips them: the engines' lifeline, the MCP servers
    sys.stdout.flush()
    sys.stderr.flush()
    os.chdir(cwd)
    os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], environ)
