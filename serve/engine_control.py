"""serve/engine_control.py - Start/Stop engine in the web app (opt-in, --engine-background): the web server runs without the model, and
the web app starts and stops the engine (POST /engine/start, /engine/stop) and shows how far a start has got
(GET /engine).

With --engine-background the server answers right away and loads the model on a thread; Stop keeps it unloaded - a
request then gets a 503 instead of loading it again - until Start, and the choice outlives a restart
(<workspace>/engine.json).  The progress comes from what the start writes to the engine's log, and while the experts
are read into RAM (the long part of a start) from the engine process's own memory against what the last start loaded.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

# (marker in the engine's log, phase, progress in %) - the order a start writes them in (engine 0.1.35, RTX 2080 Ti).
# The experts are read after the GPU line; "expert arena:" comes only once they are in RAM.
STEPS = [
    ("load_hparams", "vision", 2),
    ("strata-vision:", "vision", 3),
    ("native pack:", "weights", 5),
    ("weights loaded from", "weights", 7),
    ("draft layer loaded", "weights", 9),
    ("compute capability", "experts", 10),
    (" GiB at ", "experts_done", 85),
    ("expert cache ", "cache", 88),
    ("pre-filled", "cache_done", 93),
    ("session is up", "up", 96),
    ("with everything loaded", "up", 98),
]
# The experts' share of the bar: read into RAM (measured: the engine's memory against the last start's size), then
# locked for the GPU (no counter for that: timed against how long it took last time). On the RTX 2080 Ti PC with a warm
# disk cache the reading took about 30 s and the locking about 90 s (2026-10-03).
READ = (10, 50)
LOCK = (50, 85)
PIN_GUESS_S = 60.0                                # the locking's first estimate, before a start has measured it
LOADED = re.compile(r"loaded ([\d.]+) GiB at ([\d.]+) GiB/s")
MESSAGES = {"vision": "Starting the image encoder ...", "weights": "Reading the model's weights ...",
            "experts": "Loading the experts into RAM ...", "experts_done": "The experts are in RAM",
            "cache": "Filling the GPU's expert cache ...", "cache_done": "Expert cache filled",
            "up": "Almost ready ..."}


def _rss_gib(pid) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 2**20
    except (OSError, ValueError):
        pass
    return None


class EngineControl:
    def __init__(self, svc, state_dir, log_path=None):
        self.svc = svc
        self.file = Path(state_dir) / "engine.json"
        self.log_path = log_path
        self.lock = threading.Lock()
        self.thread = None
        self.cancelled = False
        self.error = None
        self.progress = None                          # while starting: phase, percent, message, line, ...
        self.ready_s = None                           # how long the last start took
        saved = self._load()
        self.expected_gib = saved.get("arena_gib") or self._last_arena_in_log()
        self.pin_s = saved.get("pin_s")

    # ------------------------------------------------------------------ the choice that outlives a restart
    def _load(self) -> dict:
        try:
            d = json.loads(self.file.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, **kw):
        d = {**self._load(), **kw}
        tmp = self.file.with_name(f"{self.file.name}.tmp-{os.getpid()}")
        try:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(d), encoding="utf-8")
            os.replace(tmp, self.file)
        except OSError as e:
            print(f"[strata] engine: {self.file} not written: {e}", flush=True)

    def wanted(self) -> bool:
        return self._load().get("run", True) is not False

    def _last_arena_in_log(self):
        """The experts' size from the last start in the log (its end), so the first bar already knows it."""
        if not self.log_path:
            return None
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 512 * 1024))
                tail = f.read().decode("utf-8", "replace")
        except OSError:
            return None
        found = LOADED.findall(tail)
        return float(found[-1][0]) if found else None

    # ------------------------------------------------------------------ state
    def starting(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def status(self) -> dict:
        loaded = self.svc.loaded() and not self.svc._vision_down()
        state = "starting" if self.starting() else "running" if loaded else "error" if self.error else "stopped"
        out = {"state": state, "wanted": self.wanted(), "held": bool(getattr(self.svc, "engine_hold", False)),
               "error": self.error, "ready_s": self.ready_s, "can": hasattr(self.svc.engine, "unload")}
        if state == "starting" and self.progress:
            out["progress"] = dict(self.progress, elapsed_s=round(time.time() - self.progress["t0"], 1))
            out["progress"].pop("t0", None)
        if state == "running":
            out["version"] = (getattr(self.svc.engine, "info", {}) or {}).get("version")
        return out

    # ------------------------------------------------------------------ start and stop
    def boot(self):
        """At the server's start: load in the background unless the engine was stopped last time."""
        if self.wanted():
            self.start(remember=False)
        else:
            self.svc.engine_hold = True
            print("[strata] the engine stays stopped (stopped in the web app; Start engine there loads it)", flush=True)

    def start(self, remember=True) -> dict:
        with self.lock:
            if remember:
                self._save(run=True)
            self.svc.engine_hold = False
            if self.starting() or (self.svc.loaded() and not self.svc._vision_down()):
                return self.status()
            self.error, self.cancelled = None, False
            self.progress = {"phase": "queued", "percent": 1, "message": "Starting the engine ...", "line": "",
                             "t0": time.time()}
            self.thread = threading.Thread(target=self._run, daemon=True, name="engine-start")
            self.thread.start()
        return self.status()

    def _run(self):
        offset = 0
        if self.log_path:
            try:
                offset = os.path.getsize(self.log_path)
            except OSError:
                pass
        done = threading.Event()
        watcher = threading.Thread(target=self._watch, args=(offset, done), daemon=True)
        watcher.start()
        t0 = time.time()
        try:
            self.svc.load()
            if not self.cancelled:
                self.ready_s = round(time.time() - t0, 1)
                print(f"[strata] engine started in {self.ready_s:.0f} s", flush=True)
        except Exception as e:                        # GpuBusy, the engine exiting, a cancel
            if not self.cancelled:
                self.error = str(e) or type(e).__name__
                print(f"[strata] engine start failed: {self.error}", flush=True)
        finally:
            done.set()
            watcher.join(timeout=2)

    def _watch(self, offset: int, done: threading.Event):
        """Follow the log of this start and the engine's memory while it reads the experts."""
        pos, phase_at, arena_rss0, rate, read_done = offset, {}, None, None, None
        grown, grown_at = 0.0, time.time()               # the memory's last growth: still reading, or done
        p = self.progress
        while not done.wait(0.4):
            chunk = b""
            if self.log_path:
                try:
                    with open(self.log_path, "rb") as f:
                        f.seek(pos)
                        chunk = f.read()
                except OSError:
                    pass
            if b"\n" in chunk:
                cut = chunk.rfind(b"\n") + 1
                pos += cut
                for line in chunk[:cut].decode("utf-8", "replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    p["line"] = line[:200]
                    for marker, phase, pct in STEPS:
                        if marker in line and pct >= p["percent"]:
                            if phase != p["phase"]:
                                phase_at[phase] = time.time()
                                p["message"] = MESSAGES.get(phase, p["message"])
                            p.update(phase=phase, percent=max(pct, p["percent"]))
                            m = LOADED.search(line)
                            if m:
                                self.expected_gib = float(m.group(1))
                                if read_done is not None:
                                    self.pin_s = round(time.time() - read_done, 1)
                                self._save(arena_gib=self.expected_gib, pin_s=self.pin_s)
                                p["message"] = f"The experts are in RAM: {m.group(1)} GiB at {m.group(2)} GiB/s"
                            break
            if p["phase"] == "experts":                 # the long part: reading the experts, then locking them
                proc = getattr(self.svc.engine, "proc", None)
                rss = _rss_gib(proc.pid) if proc is not None else None
                exp = self.expected_gib
                if rss is not None and read_done is None:
                    if arena_rss0 is None:
                        arena_rss0 = rss
                    got = max(0.0, rss - arena_rss0)
                    if got > grown + 0.05:
                        grown, grown_at = got, time.time()
                    spent = time.time() - phase_at.get("experts", time.time())
                    rate = got / spent if spent > 1 else rate
                    p.update(read_gib=round(got, 1), total_gib=exp, rate_gib_s=round(rate, 2) if rate else None)
                    if exp:
                        frac = min(1.0, got / exp)
                        p["percent"] = round(READ[0] + (READ[1] - READ[0]) * frac, 1)
                        p["message"] = "Loading the experts into RAM ..."   # the numbers: read_gib, total_gib, rate
                        left = (exp - got) / rate if rate and rate > 0.01 else None
                        p["eta_s"] = round(left + (self.pin_s or PIN_GUESS_S)) if left is not None else None
                        # all read: the memory is there, or it stopped growing near the end (a warm start ended at
                        # 46.1 of 46.84 GiB, the rest of the size is not resident memory)
                        if frac >= 0.99 or (frac >= 0.9 and time.time() - grown_at > 3):
                            read_done = time.time()
                    else:
                        p["message"] = "Loading the experts into RAM ..."
                if read_done is not None:               # in RAM: now the engine locks it for the GPU
                    t = time.time() - read_done
                    pin = self.pin_s or PIN_GUESS_S
                    frac = min(0.97, t / pin) if self.pin_s else 1 - 0.5 ** (t / PIN_GUESS_S)
                    p["percent"] = round(LOCK[0] + (LOCK[1] - LOCK[0]) * frac, 1)
                    p.update(read_gib=exp, total_gib=exp, rate_gib_s=None)
                    p["message"] = "Locking the experts in RAM for the GPU ..."
                    p["eta_s"] = round(max(1.0, pin - t)) if self.pin_s else None
            else:
                for k in ("eta_s", "read_gib", "rate_gib_s"):
                    p.pop(k, None)

    def stop(self) -> dict:
        """Unload now and keep it unloaded; during a start, cancel it. ("busy" while a request runs.)"""
        with self.lock:
            self._save(run=False)
            self.svc.engine_hold = True
            if self.starting():
                self.cancelled = True
                for proc in (getattr(self.svc.engine, "proc", None), getattr(self.svc.vision, "proc", None)):
                    try:
                        if proc is not None and proc.poll() is None:
                            proc.kill()
                    except OSError:
                        pass
                result = "cancelled"
            else:
                result = self.svc.unload()
                if result == "busy":
                    return {**self.status(), "result": "busy"}
            vision = self.svc.vision
            if vision is not None and hasattr(vision, "alive") and vision.alive() and not self.svc.loaded():
                vision.unload()                       # a cancelled start may have started the encoder already
        if result == "cancelled" and self.thread is not None:
            self.thread.join(timeout=30)
        self.error = None
        print(f"[strata] engine stopped in the web app ({result}); requests get a 503 until Start", flush=True)
        return {**self.status(), "result": result}


# ---------------------------------------------------------------------- for tests: a mock that starts like the engine
FAKE_START = r'''
import sys, time
log, seconds = open(sys.argv[1], "a", encoding="utf-8"), float(sys.argv[2])
lines = ["strata generate: native pack: mock experts", "strata generate: 1 MiB of weights loaded from mock",
         "strata mtp: draft layer loaded, 1 MiB of VRAM", "strata generate: GPU 0: mock, compute capability 0.0", None,
         "strata generate: expert arena: mock", "strata generate: loaded 0.53 GiB at 0.18 GiB/s", "strata generate: expert cache 10 slots, 0.1 GiB of VRAM",
         "strata generate: pre-filled 10 of 10 slots from the profile", "strata generate: session is up (engine mock)",
         "strata serve: 1 MiB of VRAM free with everything loaded"]
held = []
for line in lines:
    if line is None:                                   # the experts: 0.5 GiB into memory, step by step
        for _ in range(10):
            held.append(bytearray(54 * 2**20))
            time.sleep(seconds / 20)
        continue
    log.write(line + "\n")
    log.flush()
    time.sleep(seconds / 20)
print("READY", flush=True)
sys.stdin.read()                                       # up until it is told to end (stdin closes) or killed
'''


class MockStartable:
    """The mock engine's answers, with a start that takes `seconds` and writes the engine's log lines
    (STRATA_MOCK_START_S): the web app's Start and Stop and the progress can be tested without a model."""

    def __init__(self, inner, seconds: float, log_path: str | None = None):
        import tempfile
        self.inner, self.seconds = inner, seconds
        self.log_path = log_path or os.path.join(tempfile.mkdtemp(prefix="strata-mock-"), "engine.log")
        self.proc, self.unloaded = None, True
        self.info = dict(getattr(inner, "info", {}) or {}, version="mock")

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def alive(self) -> bool:
        return self.proc is not None and not self.unloaded and self.proc.poll() is None

    def exit_code(self):
        return self.proc.poll() if self.proc is not None else None

    def restart(self):
        import subprocess
        import sys
        self.unloaded = False
        self.proc = subprocess.Popen([sys.executable, "-c", FAKE_START, self.log_path, str(self.seconds)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        if not (self.proc.stdout.readline() or "").startswith("READY"):
            raise RuntimeError("the engine exited before it was ready")

    def unload(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.unloaded = True
