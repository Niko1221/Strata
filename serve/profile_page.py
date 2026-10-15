"""serve/profile_page.py - the Profile tab (#profile in the web app): where the time goes, for the last request and on demand.

Opt-in with `"profile": true` in strata-<model>.json (or `serve/server.py --profile`).  It does these things:

  * the engine starts with `STRATA_PREFILL_TIMING=1` and `STRATA_DECODE_TIMING=1` in its environment (docs/DETAILS.md,
    "Where a decode window's time goes"), so every request writes its breakdown to the engine log: the prompt read's
    GPU time by phase and the decode window's milliseconds by what the host waited for.  A value the user set for
    either variable stays as it is.
  * GET /health says `"profile": true` (the key is absent when it is off), which shows the tab; GET /profile answers
    a redirect to the tab (`./#profile`).
  * GET /profile/last reads the last finished request's lines from the tail of the engine log (the same parsers as
    tools/strata_profile.py) and answers its breakdown; GET /profile/history answers the last 50 requests' breakdowns
    (HISTORY_MAX), kept in memory by ProfileHistory, which reads the log on from where it was read (the /metrics poll
    drives it); GET /profile/settings answers the speed settings as the running engine reports them, as the next
    load will use them and as the run config has them.
  * POST /profile/run (Strata's own page only, as /settings) runs tools/strata_profile.py against this server, GET
    /profile/status follows it and returns the written report (bench/results/<date>-profile-<gpu>-<model>/) with its
    findings and the settings each one names (finding_settings).
  * POST /profile/tune (own page only) unloads the model and runs setup's calibration (setup.calibrate_config:
    tools/calibrate.py measures the PCIe share, the draft floor, the CPU workers and the adaptive tier on this PC and
    writes the winners into strata-<model>.json, remembered per PC in the settings file) in a child process
    (`python -m serve.profile_page --calibrate <config>`); when it ends, the engine's start command gets the tuned
    arguments (apply_to_spawn), so the next load - the next request - runs with them, and the run's report says what
    each calibrated flag was before and is now.  The run takes the time setup's --calibrate takes (its own estimate:
    roughly 15-30 minutes, longer on a slow card).  While it runs the server does not load its engine beside the
    calibration's (a request gets 503 with Retry-After, as when the GPU is in use by another program),
    /profile/run and the Settings view's POST /config answer 409, and the calibration's engines write to the same
    engine log, so the tab's last-request view can show their lines meanwhile.  Setup offers the tuning on NVIDIA
    only (each control is verified on HIP first), and not for replicas: so does the tab.

Off (the default), nothing is set in the engine's environment, /health has no "profile" key, the history is never
read and the paths answer 404, so the default path is unchanged.  The history keeps timings only, no prompt or answer.
The switches cost a few event marks per prompt chunk and a timestamp per window; the measured cost is in
docs/DETAILS.md beside the tab's description.
"""
from __future__ import annotations

import collections
import json
import os
import re
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
HISTORY_MAX = 50                  # requests the history keeps: fills the sparkline at a readable bar width, keeps the poll small
HISTORY_READ_MAX = 1 << 20        # at most this many new log bytes per read of the history
HISTORY_BUF_LINES = 4000          # the lines kept between reads (the open request's own lines and what follows)


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


def _round(x, digits=1):
    return round(x, digits) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def compact(b: dict) -> dict:
    """One request's breakdown (last_request's output) as a history entry: the numbers the tab's sparklines and table
    show, floats rounded to 1 decimal (the rates to 3, the tokens per window to 2).  A value the lines did not give is
    None (no prefill timing line, no decode line, a prompt reused whole)."""
    s = b.get("summary") or {}
    pg = (b.get("prompt") or {}).get("groups") or {}
    dg = (b.get("decode") or {}).get("groups") or {}
    share = pg.get("share") or {}
    ms = dg.get("ms")
    hit, kv = b.get("hit") or {}, b.get("kv") or {}
    return {"prompt_tokens": s.get("prompt_tokens"), "reused": s.get("reused"), "read": s.get("read"),
            "prompt_tok_s": _round(s.get("prompt_tok_s")), "prompt_ms": _round(s.get("prompt_ms")),
            "prompt_gpu_ms": _round(pg.get("gpu_ms")),
            "prompt_top": max(share, key=share.get) if share else None,
            "generated": s.get("generated"), "decode_tok_s": _round(s.get("decode_tok_s")),
            "decode_ms": _round(s.get("decode_ms")), "ms_per_window": _round(dg.get("ms_per_window")),
            "tokens_per_window": _round(dg.get("tokens_per_window"), 2),
            "ms": {k: _round(v) for k, v in ms.items()} if ms else None,
            "hit_rate": _round(hit.get("hit_rate"), 3), "pcie_share": _round(hit.get("pcie_share"), 3),
            "kv_hit": _round(kv.get("kv_hit"), 3)}


class ProfileHistory:
    """GET /profile/history: the last HISTORY_MAX requests' breakdowns, kept in memory, read from the engine log on
    from where it was read (as ConvCacheLog.poll does), so each request is parsed once.  The /metrics poll and the
    route call read(); while any Strata page is open an entry is there within about a second of its request's end.

    A request's summary line ends its run (see last_request): its prompt and decode lines come before it, the hit
    rate and KV lines after.  So the newest entry is open - its hit rate and KV lines may still arrive - and is
    replaced by the next read that has more of its lines, and closed by the next request's summary.  Timings only:
    nothing of a prompt or an answer is read or kept."""

    def __init__(self):
        self.lock = threading.Lock()
        self.entries: collections.deque = collections.deque(maxlen=HISTORY_MAX)
        self.key = None                                 # (log path, where this engine run begins in it)
        self.ident = None                               # (st_ino, st_dev) of that log: a replaced file reads from its start
        self.pos = 0                                    # the log offset read up to (whole lines only)
        self.buf: list[str] = []                        # the lines from just after the last-but-one summary
        self.open_at = None                             # index in buf of the open entry's summary line
        self.n = 0                                      # entries made since the server started (= summaries consumed)
        self.engine_run = 0                             # engine starts seen (a new (path, start) adds one)
        self.read_bytes = 0                             # what the last read took from the log (tests)
        self.failed = False                             # said once when the lines could not be read

    def snapshot(self) -> dict:
        with self.lock:
            return self._snapshot()

    def _snapshot(self) -> dict:
        return {"max": HISTORY_MAX, "engine_run": self.engine_run, "requests": [dict(e) for e in self.entries]}

    def read(self, path, start) -> dict:
        """The history after the log's new lines; `start` is where the engine's current run begins in the log (None:
        no run yet - a lazy start - so nothing is read)."""
        with self.lock:
            self._read(path, start)
            return self._snapshot()

    def poll(self, path, start) -> None:
        """read() for a caller that has no use for the answer (the /metrics poll): no snapshot is built."""
        with self.lock:
            self._read(path, start)

    def skip_to_end(self, path) -> None:
        """Lines written so far are not requests of the user's (a tuning's engines write to the same log): the next
        read starts at the log's end, and the open entry is closed."""
        with self.lock:
            try:
                self.pos = os.path.getsize(path) if path else self.pos
            except OSError:
                return
            self.buf, self.open_at = [], None
            self._close_open()

    def _close_open(self):
        if self.entries and self.entries[-1]["open"]:
            self.entries[-1] = {**self.entries[-1], "open": False}

    def _reset(self, pos=0):
        self.pos, self.buf, self.open_at = pos, [], None
        self._close_open()

    def _read(self, path, start):
        self.read_bytes = 0
        if (path, start) != self.key:                   # another start of the engine: its log lines begin again
            self.key, self.ident = (path, start), None
            self._reset(start or 0)
            if path and start is not None:
                self.engine_run += 1
        if not path or start is None:
            return
        try:
            st = os.stat(path)
            size = st.st_size
            ident = (st.st_ino, st.st_dev)
            if size < self.pos or (self.ident is not None and ident != self.ident):   # emptied, or another file
                self._reset(0)
            self.ident = ident
            skip = size - self.pos > HISTORY_READ_MAX
            if skip:                                    # a burst of other lines: the last MiB is what is read
                self._reset(size - HISTORY_READ_MAX)
            with open(path, "rb") as f:
                f.seek(self.pos)
                data = f.read(HISTORY_READ_MAX)
        except OSError:
            return
        self.read_bytes = len(data)
        if skip:                                        # the tail starts inside a line: drop that line
            cut = data.find(b"\n") + 1
            if not cut:                                 # no line end in it at all: all of it is one half-written line
                self.pos += len(data)
                return
            self.pos += cut
            data = data[cut:]
        end = data.rfind(b"\n") + 1                     # whole lines only: the rest is read next time
        self.pos += end
        lines = data[:end].decode("utf-8", "replace").splitlines()
        if lines:
            try:
                self._add(lines)
            except Exception as e:                      # noqa: BLE001 - /metrics calls this: the Monitor must not break for it
                if not self.failed:                     # _add changed nothing: these lines are left out, the history goes on
                    self.failed = True
                    print(f"[strata] the Profile tab's history could not read the engine log: {e}", flush=True)

    def _add(self, lines: list[str]):
        """The new lines' requests.  Works on copies and assigns them at the end, so a line that cannot be parsed
        leaves the history as it was."""
        P = parsers()
        buf = self.buf + lines
        open_at = self.open_at
        if len(buf) > HISTORY_BUF_LINES:
            cut = len(buf) - HISTORY_BUF_LINES
            buf = buf[cut:]
            open_at = open_at - cut if open_at is not None and open_at >= cut else None
        idx = [i for i, line in enumerate(buf) if P.parse_summary(line)]
        now = time.time()
        entries = collections.deque(self.entries, maxlen=HISTORY_MAX)
        n = self.n
        for j, i in enumerate(idx):
            last = j == len(idx) - 1
            if len(idx) - j > HISTORY_MAX and i != open_at:
                n += 1                                  # would roll off the ring at once: counted, not parsed
                continue
            start = idx[j - 1] + 1 if j else 0          # its own lines, then what follows up to the next summary
            b = last_request(buf[start:idx[j + 1] if not last else None])
            entry = {"n": n + 1, "engine_run": self.engine_run, "seen_at": round(now, 1), "open": last, **compact(b)}
            if i == open_at and entries:                # the same summary as the last read: its lines grew
                old = entries[-1]
                entries[-1] = {**entry, "n": old["n"], "engine_run": old["engine_run"], "seen_at": old["seen_at"]}
            else:
                n += 1
                if entries and entries[-1]["open"]:     # a newer request's summary closes the one before it
                    entries[-1] = {**entries[-1], "open": False}
                entries.append(entry)
        if len(idx) >= 2:                               # keep the open request's own lines and what follows
            cut = idx[-2] + 1
            buf = buf[cut:]
            open_at = idx[-1] - cut
        elif idx:
            open_at = idx[0]
        self.buf, self.open_at, self.n, self.entries = buf, open_at, n, entries


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
                return {"path": str(readme.parent), "readme": readme.read_text(encoding="utf-8"), "results": results,
                        "findings": report_findings(results)}
            except (OSError, ValueError):
                return {"path": str(readme.parent), "readme": None, "results": None, "findings": []}
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
    TOK_S = re.compile(r"\(([\d.]+) tok/s\)\s*$")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.before = None                              # tune_snapshot(): taken by the caller before it unloads; start() keeps it

    def command(self, engine_log: str | None, long: int | None) -> list[str]:
        return [sys.executable, "-m", "serve.profile_page", "--calibrate", str(self.config_path)]

    def find_report(self, lines: list[str]) -> dict | None:
        """The calibration's own summary line (setup's `tuned for this PC: ...`) and, for a run that ended with exit
        code 0, what each calibrated flag was before and is now.  A run that failed has no report."""
        if self.returncode != 0:
            return None
        summary = tok_s = None
        for line in reversed(lines):
            t = line.strip()
            if "tuned for this PC" in t:
                summary = t.split("tuned for this PC: ", 1)[-1].strip()
                m = self.TOK_S.search(summary)
                tok_s = float(m.group(1)) if m else None
                break
        changes = self._changes()
        if summary is None and changes is None:
            return None
        return {"summary": summary, "tok_s": tok_s, "changes": changes}

    def _changes(self) -> list | None:
        """One row per calibrated flag: its value before (the run config's, None: the engine's own default), the value
        the running engine had then (None: not reported), its value in the run config now, and whether that changed
        (numbers are compared as numbers: 0.50 and 0.5 are the same)."""
        if not self.before or not self.config_path:
            return None
        try:
            after, _ = tuned_settings(json.loads(Path(self.config_path).read_text(encoding="utf-8-sig")))
        except (OSError, ValueError):
            return None
        rows = []
        for flag, before in self.before["args"].items():
            rows.append({"flag": flag, "before": before, "before_loaded": self.before["loaded"].get(flag),
                         "after": after.get(flag), "changed": not _same_value(before, after.get(flag))})
        return rows


def _same_value(a, b) -> bool:
    if a == b:
        return True
    try:
        return a is not None and b is not None and float(a) == float(b)
    except (TypeError, ValueError):
        return False


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


def tune_snapshot(cfg: dict, info: dict | None) -> dict:
    """What the calibrated flags are before a tuning: `args` the run config's value of each (None: not set, the engine's
    own default) and `loaded` the value the running engine reports for the three it reports (None: not reported)."""
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    import calibrate as CAL
    args = list(cfg.get("args") or []) if isinstance(cfg, dict) else []
    info = info or {}
    loaded = {}
    for flag, key in (("--pcie-frac", "pcie_frac"), ("--spec-min-p", "spec_min_p"), ("--pool-workers", "pool_workers")):
        v = info.get(key)
        loaded[flag] = None if v is None or v == "" else str(v)
    return {"args": {flag: CAL.arg_value(args, flag) for flag in CAL.DEFAULTS}, "loaded": loaded}


# ------------------------------------------------------------------------------------------ findings -> settings
# What a finding of tools/strata_profile.py names, by the words it names it with, in the order the tab lists them
# (a test reads the tool's own wording, so a change there fails it).  The ids are SPEED_SETTINGS' ids.
FINDING_SETTINGS = (
    ("STRATA_PREFILL_CPU_SHARE,", ("prefill_cpu_share",)),
    ("STRATA_PREFILL_CPU_SHARE_MAX", ("prefill_cpu_share_max",)),
    ("--max-context", ("max_context",)),
    ("--kv-resident", ("kv_resident",)),
    ("--vram-reserve-mib", ("vram_reserve_mib",)),
    ("the cache's slots", ("expert_cache",)),
    ("the PCIe share", ("pcie_frac",)),
    ("the adaptive tier", ("adapt_every", "adapt_swaps", "adapt_decay")),
    ("expert_profile_save", ("expert_profile_save",)),
    ("--spec-min-p", ("spec_min_p",)),
    ("--spec ", ("spec",)),
)


def finding_settings(text: str) -> list[str]:
    """The ids of the speed settings a finding's sentence names (none for a measurement sentence)."""
    out: list[str] = []
    for words, ids in FINDING_SETTINGS:
        if words in text:
            out += [i for i in ids if i not in out]
    return out


def report_findings(results) -> list[dict]:
    """A run's findings (results.json: results.findings) with the settings each one names."""
    found = ((results or {}).get("results") or {}).get("findings") if isinstance(results, dict) else None
    return [{"text": t, "settings": finding_settings(t)} for t in found or [] if isinstance(t, str)]


# ------------------------------------------------------------------------------------------------- the settings
def _text(v):
    return None if v is None or v == "" else str(v)[:80]


def _cache(info: dict):
    slots, mib = info.get("expert_slots"), info.get("expert_cache_mib")
    return f"{slots} experts ({mib} MiB)" if slots is not None and mib is not None else None


# id, name, kind ("arg": a flag in the start command's args, "env": a variable in its environment, "key": a key of the
# run config), where the running engine's INFO line has it (a key, a function of the INFO dict, or None: not reported),
# calibrated (setup's calibration writes it), edit ("about": the About tab's Model settings has the key), the sentence.
SPEED_SETTINGS = (
    ("pcie_frac", "--pcie-frac", "arg", "pcie_frac", True, None,
     "The share of the missed experts the card reads over PCIe instead of the CPU computing them."),
    ("spec_min_p", "--spec-min-p", "arg", "spec_min_p", True, None,
     "The draft floor: how sure a drafted token must be for the draft to go on."),
    ("spec", "--spec", "arg", None, False, None, "How many tokens the draft may propose."),
    ("pool_workers", "--pool-workers", "arg", "pool_workers", True, None,
     "The CPU threads that compute the experts not in VRAM."),
    ("adapt_every", "--adapt-every", "arg", None, True, None,
     "The adaptive expert tier: how often it swaps experts into VRAM, how many, how fast it forgets."),
    ("adapt_swaps", "--adapt-swaps", "arg", None, True, None,
     "The adaptive expert tier: how often it swaps experts into VRAM, how many, how fast it forgets."),
    ("adapt_decay", "--adapt-decay", "arg", None, True, None,
     "The adaptive expert tier: how often it swaps experts into VRAM, how many, how fast it forgets."),
    ("kv_resident", "--kv-resident", "arg", "kv_resident", False, None,
     "KV positions per layer kept in VRAM; the rest is streamed from RAM."),
    ("max_context", "--max-context", "arg", "context", False, None,
     "The context the KV cache is sized for; what it does not take is left to the expert cache."),
    ("expert_cache", "--expert-cache", "arg", _cache, False, None,
     "The VRAM expert cache; auto takes what is left after the rest."),
    ("vram_reserve_mib", "--vram-reserve-mib", "arg", None, False, "about",
     "VRAM left free for other programs; it is taken from the expert cache."),
    ("prefill_cpu_share", "STRATA_PREFILL_CPU_SHARE", "env", None, False, None,
     "The CPU's share of the experts in short prompt reads."),
    ("prefill_cpu_share_max", "STRATA_PREFILL_CPU_SHARE_MAX", "env", None, False, None,
     "The largest prompt chunk the CPU share works on (1,024 tokens by default)."),
    ("expert_profile_save", "expert_profile_save", "key", None, False, None,
     "Saves the adaptive tier's expert ranking so the next start fills the cache with it."),
)
SAVE_FLAG = "--expert-profile-save"


def speed_settings(engine, loaded: bool, cfg: dict | None, config_path=None) -> dict:
    """GET /profile/settings: the speed settings from three places.  `loaded_now`: the running engine's own report
    (its INFO line; None while it is not loaded or when it does not report the value).  `next_load`: the server's start
    command, which the next load uses and a tuning updates (without one - replicas, the mock engine - the run config).
    `run_config`: the config file as it is on disk (None without one).  Values are strings or None (not set: the
    engine's own default); `expert_profile_save` holds a path, so it is only ever "set" or None."""
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    import calibrate as CAL
    info = (getattr(engine, "info", None) or {}) if loaded else {}
    spawn = getattr(engine, "spawn", None)
    start = None
    if spawn and not hasattr(engine, "engines"):
        start = (list(spawn[1] or []), dict(spawn[4] or {}))
    cfg_args = [str(a) for a in (cfg.get("args") or [])] if isinstance(cfg, dict) and isinstance(cfg.get("args"), list) else []
    cfg_env = cfg.get("env") if isinstance(cfg, dict) and isinstance(cfg.get("env"), dict) else {}

    def value(kind, name, args, env, key_set):
        if kind == "arg":
            return _text(CAL.arg_value(args, name))
        if kind == "env":
            return _text(env.get(name))
        return "set" if key_set else None

    rows = []
    for rid, name, kind, info_key, calibrated, edit, about in SPEED_SETTINGS:
        now = info_key(info) if callable(info_key) else info.get(info_key) if info_key else None
        run_cfg = value(kind, name, cfg_args, cfg_env,
                        isinstance(cfg, dict) and isinstance(cfg.get(name), str) and bool(cfg.get(name).strip()))
        if start is not None:
            nxt = value(kind, name, start[0], start[1], SAVE_FLAG in start[0])
        else:
            nxt = run_cfg
        rows.append({"id": rid, "name": name, "kind": kind, "loaded_now": _text(now), "next_load": nxt,
                     "run_config": run_cfg if cfg is not None else None, "calibrated": calibrated, "edit": edit,
                     "about": about})
    return {"file": Path(config_path).name if config_path and cfg is not None else None, "loaded": bool(loaded),
            "settings": rows}



if __name__ == "__main__":                              # python -m serve.profile_page --calibrate <strata-*.json>
    if len(sys.argv) == 3 and sys.argv[1] == "--calibrate":
        sys.path.insert(0, str(ROOT))
        import setup

        def _term(signum, frame):                       # the server's stop(): end the calibration's engine on the way out
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, _term)
        sys.exit(0 if setup.calibrate_config(Path(sys.argv[2])) else 1)
    sys.exit("usage: python -m serve.profile_page --calibrate <strata-*.json>")
