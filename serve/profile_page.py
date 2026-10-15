"""serve/profile_page.py - the Profile page (/profile): where the time goes, for the last request and on demand.

Opt-in with `"profile": true` in strata-<model>.json (or `serve/server.py --profile`).  It does two things:

  * the engine starts with `STRATA_PREFILL_TIMING=1` and `STRATA_DECODE_TIMING=1` in its environment (docs/DETAILS.md,
    "Where a decode window's time goes"), so every request writes its breakdown to the engine log: the prompt read's
    GPU time by phase and the decode window's milliseconds by what the host waited for.  A value the user set for
    either variable stays as it is.
  * GET /profile serves the page; GET /profile/last reads the last finished request's lines from the tail of the engine
    log (the same parsers as tools/strata_profile.py) and answers its breakdown; POST /profile/run (Strata's own page
    only, as /settings) runs tools/strata_profile.py against this server, GET /profile/status follows it and returns
    the written report (bench/results/<date>-profile-<gpu>-<model>/).
  * POST /profile/tune (own page only) unloads the model and runs setup's calibration (setup.calibrate_config:
    tools/calibrate.py measures the PCIe share, the draft floor, the CPU workers and the adaptive tier on this PC and
    writes the winners into strata-<model>.json, remembered per PC in the settings file) in a child process
    (`python -m serve.profile_page --calibrate <config>`); when it ends, the engine's start command gets the tuned
    arguments (apply_to_spawn), so the next load - the next request - runs with them.  The run takes the time setup's
    --calibrate takes (its own estimate: roughly 15-30 minutes, longer on a slow card).  While it runs the server does
    not load its engine beside the calibration's (a request gets 503 with Retry-After, as when the GPU is in use by
    another program), /profile/run and the Settings view's POST /config answer 409, and the calibration's engines
    write to the same engine log, so the page's last-request view can show their lines meanwhile.  Setup offers the
    tuning on NVIDIA only (each control is verified on HIP first), and not for replicas: so does the page.

Off (the default), nothing is set in the engine's environment and the five paths answer 404, so the default path is
unchanged.
The switches cost a few event marks per prompt chunk and a timestamp per window; the measured cost is in
docs/DETAILS.md beside the page's description.
"""
from __future__ import annotations

import collections
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from serve.winjob import contain

ROOT = Path(__file__).resolve().parents[1]
TIMING_ENV = {"STRATA_PREFILL_TIMING": "1", "STRATA_DECODE_TIMING": "1"}
READ_MAX = 262144                 # the engine log's tail that is read for the last request (its lines are a few KB)
TOOL = ROOT / "tools" / "strata_profile.py"


def timing_env(env: dict) -> dict:
    """The engine's environment with the two timing switches; one the user set keeps its value."""
    out = dict(env)
    for k, v in TIMING_ENV.items():
        out.setdefault(k, v)
    return out


_parsers = None


def parsers():
    """tools/strata_profile.py: its line parsers and grouping (imported on first use; it needs no GPU)."""
    global _parsers
    if _parsers is None:
        if str(ROOT / "tools") not in sys.path:
            sys.path.insert(0, str(ROOT / "tools"))
        import strata_profile
        _parsers = strata_profile
    return _parsers


def last_request(lines: list[str]) -> dict | None:
    """The breakdown of the last finished request among `lines` (engine log lines, oldest first), or None.

    A request's `strata serve: prompt ...` summary line ends its run: the prompt and decode timing lines come before
    it, the hit rate and KV streaming lines after it.  So the last summary's request is the lines between the summary
    before it and the end: its own lines before the summary, the hit rate and KV lines after.  The KV streaming line
    counts since the engine started, so the request's own share is the difference from the line before it.  A
    cancelled request writes no summary, so its timing lines count with the next request's (a prompt the next request
    reused whole then shows the cancelled read's phases): the page is a view of the log, not a ledger."""
    P = parsers()
    idx = [i for i, line in enumerate(lines) if P.parse_summary(line)]
    if not idx:
        return None
    s = idx[-1]
    p = idx[-2] if len(idx) > 1 else -1
    own = P.engine_facts(lines[p + 1:s + 1])
    after = P.engine_facts(lines[s + 1:])
    summary = own.get("summary") or {}
    out = {"summary": summary,
           "prompt": {"tokens": summary.get("prompt_tokens"), "reused": summary.get("reused"), "read": summary.get("read"),
                      "tok_s": summary.get("prompt_tok_s"), "ms": summary.get("prompt_ms"),
                      "groups": P.prompt_groups(own.get("prefill")), "host": own.get("prefill_host")},
           "decode": {"generated": summary.get("generated"), "tok_s": summary.get("decode_tok_s"), "ms": summary.get("decode_ms"),
                      "groups": P.decode_groups(own.get("decode")), "drafts_accepted": summary.get("drafts_accepted"),
                      "drafts_offered": summary.get("drafts_offered")},
           "hit": after.get("hit"), "kv": P.kv_request(after.get("kv"), own.get("kv")),
           "gpu_stages": own.get("gpu_stages")}
    if out["prompt"]["groups"]:
        out["prompt"]["groups"]["top"] = [list(t) for t in out["prompt"]["groups"]["top"]]
    return out


class LastRequest:
    """GET /profile/last: the last request's breakdown from the engine log's tail, parsed again only when the log grew."""

    def __init__(self):
        self.cached = (None, None)                      # (key, value), one assignment: handlers read it concurrently

    def read(self, path, start) -> dict | None:
        """`start`: where this engine run's lines begin in the log (None: no run yet - a lazy start - so nothing)."""
        if not path or start is None:
            return None
        try:
            size = os.path.getsize(path)
        except OSError:
            return None
        key = (path, start, size)
        cached_key, cached_value = self.cached
        if key == cached_key:
            return cached_value
        pos = max(int(start), size - READ_MAX, 0)
        try:
            with open(path, "rb") as f:
                f.seek(pos)
                data = f.read(size - pos)
        except OSError:
            return None
        if pos > int(start):                            # the tail starts inside a line: drop that line
            data = data[data.find(b"\n") + 1:]
        end = data.rfind(b"\n") + 1                    # whole lines only
        value = last_request(data[:end].decode("utf-8", "replace").splitlines())
        self.cached = (key, value)
        return value


class ProfileRun:
    """POST /profile/run and GET /profile/status: one run of tools/strata_profile.py at a time, its output kept,
    the report read back once it ends.  `on_done()` is called (off the lock, before the run is published as done)
    when a run ends with exit code 0; what it raises becomes the run's `note`.  `on_end()` is called after every end."""

    name = "profile"

    def __init__(self, base: str = "", api_key: str = "", config_path: str | None = None, out_dir: Path | None = None,
                 popen=subprocess.Popen, on_done=None, on_end=None):
        self.base, self.api_key, self.config_path = base, api_key, config_path
        self.out_dir = out_dir
        self.popen = popen
        self.on_done, self.on_end = on_done, on_end
        self.note = None                                # what on_done said (or raised) about the result
        self.lock = threading.Lock()
        self.proc = None
        self.lines: collections.deque = collections.deque(maxlen=400)
        self.state = "idle"                             # idle | running | done | failed
        self.started_at = None
        self.ended_at = None
        self.returncode = None
        self.report = None                              # {"path", "readme", "results"} once a run wrote one

    def command(self, engine_log: str | None, long: int | None) -> list[str]:
        cmd = [sys.executable, str(TOOL), "--url", self.base]
        if self.config_path:
            cmd += ["--config", str(self.config_path)]
        if engine_log:
            cmd += ["--engine-log", str(engine_log)]
        if long is not None:
            cmd += ["--long", str(int(long))]
        if self.out_dir:
            cmd += ["--out", str(self.out_dir)]
        return cmd

    def start(self, engine_log: str | None, long: int | None = None) -> str:
        """"started", or "running" when a run is still going."""
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return "running"
            env = dict(os.environ)
            if self.api_key:
                env["STRATA_KEY"] = self.api_key         # the tool's --api-key, out of the command line
            self.lines.clear()
            self.report, self.returncode, self.ended_at, self.note = None, None, None, None
            self.started_at = time.time()
            self.state = "running"
            try:
                self.proc = self.popen(self.command(engine_log, long), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
                                       cwd=str(ROOT), env=env)
            except OSError as e:
                self.state, self.ended_at = "failed", time.time()
                self.lines.append(f"the {self.name} could not start: {e}")
                return "failed"
            contain(self.proc)                          # Windows: ends with the server, as the engine does
            threading.Thread(target=self._pump, args=(self.proc,), daemon=True, name="profile-run").start()
            return "started"

    def stop(self):
        """At shutdown: end a run still going."""
        with self.lock:
            proc = self.proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(10)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    proc.kill()
                except OSError:
                    pass

    def _pump(self, proc):
        try:
            for line in proc.stdout:
                self.lines.append(line.rstrip("\n"))
        except (OSError, ValueError):
            pass
        finally:
            try:
                proc.stdout.close()
            except OSError:
                pass
        rc = proc.wait()
        with self.lock:
            if self.proc is not proc:                   # a newer run took over: its pump reports
                return
        note = None
        if rc == 0 and self.on_done is not None:        # before the run is published as done: a request must not slip in
            try:
                note = self.on_done()
            except Exception as e:                       # noqa: BLE001 - said in the run's note, never raised in a thread
                note = f"the {self.name}'s result could not be applied: {e}"
        with self.lock:
            if self.proc is not proc:
                return
            self.returncode, self.ended_at = rc, time.time()
            self.report = self.find_report(list(self.lines))
            self.note = note
            self.state = "done" if rc == 0 else "failed"
        if self.on_end is not None:
            self.on_end()

    def find_report(self, lines: list[str]) -> dict | None:
        """The tool's `Report: <folder>/README.md (home folders ...)` line -> the folder's README and results.  Only a
        README.md inside the results folder the run was given (bench/results by default) is read: the line is the
        tool's, but what the server reads back is bounded all the same."""
        parent = (self.out_dir or ROOT / "bench" / "results").resolve()
        for line in reversed(lines):
            if not line.startswith("Report: "):
                continue
            readme = Path(line[len("Report: "):].rsplit(" (home folders", 1)[0].strip())
            if not readme.is_absolute():
                readme = ROOT / readme
            readme = readme.resolve()
            if readme.name != "README.md" or readme.parent.parent != parent:
                return None
            try:
                results = json.loads((readme.parent / "results.json").read_text(encoding="utf-8"))
                return {"path": str(readme.parent), "readme": readme.read_text(encoding="utf-8"), "results": results}
            except (OSError, ValueError):
                return {"path": str(readme.parent), "readme": None, "results": None}
        return None

    def status(self) -> dict:
        with self.lock:
            now = time.time()
            return {"state": self.state, "lines": list(self.lines), "started_at": self.started_at,
                    "elapsed_s": round((self.ended_at or now) - self.started_at, 1) if self.started_at else None,
                    "returncode": self.returncode, "report": self.report, "note": self.note}

    def running(self) -> bool:
        with self.lock:
            return self.state == "running"


class TuneRun(ProfileRun):
    """POST /profile/tune: setup's calibration in a child process, the model unloaded for it (the caller unloads)."""

    name = "tuning"

    def command(self, engine_log: str | None, long: int | None) -> list[str]:
        return [sys.executable, "-m", "serve.profile_page", "--calibrate", str(self.config_path)]

    def find_report(self, lines: list[str]) -> dict | None:
        """The calibration's own summary line (setup's `tuned for this PC: ...`), if it said one."""
        for line in reversed(lines):
            t = line.strip()
            if "tuned for this PC" in t:
                return {"summary": t.split("tuned for this PC: ", 1)[-1].strip()}
        return None


def tuned_settings(cfg: dict) -> tuple[dict, dict]:
    """What a calibration wrote into the config: the calibrated flags' values (tools/calibrate.py's DEFAULTS; an
    absent flag is the engine's own default) and, of the config's "env", the variables a calibration manages
    (calibrate.ENV_DEFAULTS where the tool has them; None: the variable is absent)."""
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    import calibrate as CAL
    args = list(cfg.get("args") or [])
    settings = {flag: CAL.arg_value(args, flag) for flag in CAL.DEFAULTS}
    env = cfg.get("env") or {}
    managed = {k: (str(env[k]) if k in env else None) for k in getattr(CAL, "ENV_DEFAULTS", {})}
    return {k: v for k, v in settings.items() if v is not None}, managed


def apply_to_spawn(engine, cfg: dict) -> dict:
    """The engine's start command (StrataEngine.spawn: exe, args, cwd, log, env, lazy, cpus) with the calibrated
    flags and variables as the config now has them, so the next load runs with them.  Returns what was set (a
    variable taken away as None).  An engine without a start command of its own (replicas) is left alone: {}."""
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    import calibrate as CAL
    spawn = getattr(engine, "spawn", None)
    if not spawn or hasattr(engine, "engines"):
        return {}
    settings, managed = tuned_settings(cfg)
    exe, args, cwd, log, env, lazy, cpus = spawn
    env = dict(env or {})
    for k, v in managed.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    engine.spawn = (exe, CAL.apply(list(args), settings), cwd, log, env, lazy, cpus)
    return {**settings, **managed}


if __name__ == "__main__":                              # python -m serve.profile_page --calibrate <strata-*.json>
    if len(sys.argv) == 3 and sys.argv[1] == "--calibrate":
        sys.path.insert(0, str(ROOT))
        import setup

        def _term(signum, frame):                       # the server's stop(): end the calibration's engine on the way out
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, _term)
        sys.exit(0 if setup.calibrate_config(Path(sys.argv[2])) else 1)
    sys.exit("usage: python -m serve.profile_page --calibrate <strata-*.json>")
