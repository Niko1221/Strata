"""Elastic peer pair (opt-in, the run config's "elastic" block): the helper card turns into its own engine on a second
request, and while both run each engine is the other's peer expert tier ("mutual help", --peer-link).

One engine process per GPU behind the one server, one request per process, as always:

* the LEAD engine owns the first card and, while it serves alone, uses the second card as its `--peer-device` tier
  (one request runs exactly as with `--peer-device`);
* the HELPER engine owns the second card.  While no request needs it, it sleeps: its expert cache is given back
  (`VRAM`, #533's segmented cache, `--vram-elastic`), it is optionally frozen with NVIDIA's cuda-checkpoint (its
  card is then completely free), and its CPU cores are the lead's.

When a request arrives while the lead is busy, the helper wakes:

    lead:   PEER_DETACH   (between two verify windows: the peer tier and its prompt buffers are freed; with
                           --peer-link the helper's card stays the lead's tier through the helper engine)
            POOL_HOLD n   (the lead's last n CPU workers sit out: the helper's cores)
    helper: thawed (if frozen), VRAM (its cache comes back from the shared host arena)

and serves the request beside the lead's.  The two engines compute each other's rows for the experts their cards hold
(peer_link.hpp), so neither falls back to the CPU for the experts the other card has.  After `hold_s` seconds without
a request the helper sleeps again and the lead takes its card back (POOL_HOLD 0, PEER_ATTACH - the attach waits for
the lead's current request to end).

The engines share the expert arena in host RAM (`--shared-expert-arena`), so the helper costs its own dense weights
and buffers, not a second copy of the experts.  `generate` picks a lane per call:

1. an idle awake lane whose last conversation shares the longest prefix with the prompt (its cache holds it);
2. any idle awake lane (the lead first);
3. the sleeping helper (woken first);
otherwise the call waits for a lane.

Config (the lead's run config):

    "elastic": {"helper": "strata-<model>-helper.json", "hold_s": 30, "helper_cores": "8-15,24-31",
                "hold_workers": 8, "checkpoint": null}

The helper config is a normal one-card run config ("gpu": the second card; its args with --peer-link FILE
--peer-link-role 1 --vram-elastic --shared-expert-arena FILE); the lead's args have --peer-device 1 --peer-link FILE
--peer-link-role 0 and the same --shared-expert-arena.  Without the block nothing here runs.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path


def _cores(spec) -> set[int] | None:
    """"8-15,24-31" -> {8, ..., 15, 24, ..., 31}; None/"" -> None (no restriction)."""
    if spec is None or spec == "":
        return None
    if isinstance(spec, (list, tuple)):
        return {int(x) for x in spec}
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def common_prefix(a, b) -> int:
    n = min(len(a), len(b))
    i = 0
    step = 256                                  # prompts are long: blocks first, then the tail
    while i + step <= n and a[i:i + step] == b[i:i + step]:
        i += step
    while i < n and a[i] == b[i]:
        i += 1
    return i


def log(msg: str) -> None:
    print(f"[elastic] {time.strftime('%H:%M:%S')} {msg}", flush=True)


class CtlError(RuntimeError):
    pass


_EMPTY: dict = {}
CTL_PREFIXES = ("CTL ", "VRAM ", "ERR VRAM")


class _Filtered:
    """The engine's stdout without the control answers (they go to their own queue, so a control command can be
    answered while a request streams on the same pipe)."""

    def __init__(self, f, ctl_q):
        self.f, self.ctl_q = f, ctl_q

    def __iter__(self):
        for line in self.f:
            if line.startswith(CTL_PREFIXES):
                self.ctl_q.put(line.strip())
                continue
            yield line

    def __getattr__(self, k):
        return getattr(self.f, k)


def make_engine_class(StrataEngine):
    class ElasticStrataEngine(StrataEngine):
        """A StrataEngine whose control answers have their own queue and whose process can start on given cores."""

        def __init__(self, exe, args, cwd=None, log=None, env=None, lazy=False, cores=None):
            self.ctl_q: queue.Queue = queue.Queue()
            self.cores = cores
            self.espawn = (exe, list(args), cwd, log, env, lazy, cores)
            prev = None
            if cores:   # the child inherits the CALLING thread's affinity at fork; the engine's pool reads it
                prev = os.sched_getaffinity(0)
                os.sched_setaffinity(0, cores)
            try:
                super().__init__(exe, args, cwd=cwd, log=log, env=env, lazy=lazy)
            finally:
                if prev is not None:
                    os.sched_setaffinity(0, prev)

        def _pump(self):
            proc, ctl = self.proc, self.ctl_q
            if proc is not None and not isinstance(proc.stdout, _Filtered):
                proc.stdout = _Filtered(proc.stdout, ctl)
            try:
                super()._pump()
            finally:
                ctl.put(None)

        def restart(self, tries: int = 3):
            exe, args, cwd, log_, env, lazy, cores = self.espawn
            prev = None
            if cores:
                prev = os.sched_getaffinity(0)
                os.sched_setaffinity(0, cores)
            try:
                self.ctl_q = queue.Queue()
                super().restart(tries)
            finally:
                if prev is not None:
                    os.sched_setaffinity(0, prev)

        def send(self, line: str) -> None:
            with self.wlock:
                self.proc.stdin.write(line + "\n")
                self.proc.stdin.flush()

        def control(self, command: str, expect: tuple[str, ...], timeout: float) -> str:
            """Send a control line and wait for its answer (a line starting with one of `expect`); answers to earlier
            fire-and-forget lines that arrive meanwhile are logged and skipped."""
            while True:
                try:
                    old = self.ctl_q.get_nowait()
                except queue.Empty:
                    break
                if old is not None:
                    log(f"(late control answer: {old})")
            if self.proc is None or self.proc.poll() is not None:
                raise CtlError(f"{command}: the engine is not running")
            self.send(command)
            deadline = time.monotonic() + timeout
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise CtlError(f"{command}: no answer within {timeout:.0f} s")
                try:
                    line = self.ctl_q.get(timeout=min(left, 2.0))
                except queue.Empty:
                    if self.proc is None or self.proc.poll() is not None:
                        raise CtlError(f"{command}: the engine ended") from None
                    continue
                if line is None:
                    raise CtlError(f"{command}: the engine ended")
                if line.startswith(expect):
                    return line
                if line.startswith(("ERR VRAM", "CTL PEER ERR")):
                    raise CtlError(f"{command}: {line}")
                log(f"(late control answer: {line})")

    return ElasticStrataEngine


def toggle(checkpoint, engine, what, want):
    """NVIDIA cuda-checkpoint on an idle engine process: its GPU state goes to host RAM (the card is free) or comes
    back at the same addresses.  `want` = "checkpointed" or "running"; nothing happens in that state already.  No-op
    without a checkpoint tool."""
    if not checkpoint or getattr(engine, "proc", None) is None:
        return
    st = subprocess.run([checkpoint, "--get-state", "--pid", str(engine.proc.pid)], capture_output=True, text=True,
                        timeout=60)
    if st.returncode == 0 and st.stdout.strip() == want:
        return
    t0 = time.monotonic()
    r = subprocess.run([checkpoint, "--toggle", "--pid", str(engine.proc.pid)], capture_output=True, text=True,
                       timeout=300)
    if r.returncode != 0:
        raise CtlError(f"cuda-checkpoint ({what}): {r.stderr.strip() or r.stdout.strip()}")
    log(f"{what} (cuda-checkpoint) in {time.monotonic() - t0:.2f} s")


class Lane:
    def __init__(self, name: str, engine, role: str):
        self.name, self.engine, self.role = name, engine, role
        self.busy = False
        self.awake = role == "lead"
        self.live: list[int] = []               # the conversation the engine holds: last prompt + its output
        self.idle_since = time.monotonic()
        self.requests = 0
        self.broken_until = 0.0                 # a helper whose wake failed is skipped until then


class ElasticEngine:
    """The engine Service talks to: two lanes behind one generate().  Unknown attributes are the lead's."""

    can_stop = True
    unloaded = False

    def __init__(self, lead, helper, hold_s: float = 30.0, hold_workers: int = 0, checkpoint: str | None = None):
        self.lead = Lane("lead", lead, "lead")
        self.helper = Lane("helper", helper, "helper")
        self.lanes = [self.lead, self.helper]
        self.batch = len(self.lanes)            # Service.run: requests run at once (one per lane), each its own figures
        self.hold_s = float(hold_s)
        self.hold_workers = int(hold_workers)
        self.checkpoint = checkpoint
        self.cv = threading.Condition()
        self.transition = threading.Lock()      # one wake / sleep at a time (they move the same card)
        self.tl = threading.local()
        self.waits = 0
        self.stats = {"wakes": 0, "sleeps": 0, "wake_s": [], "parallel_requests": 0}
        self._stop = False
        threading.Thread(target=self._sleeper, daemon=True, name="elastic-sleeper").start()

    def __getattr__(self, k):                   # only for what is not defined here: the lead engine's
        if k in ("lead", "helper", "lanes"):
            raise AttributeError(k)
        return getattr(self.lead.engine, k)

    # ---------------------------------------------------------------- the Engine surface Service uses
    @property
    def max_context(self):
        return min(l.engine.max_context for l in self.lanes)

    @property
    def info(self):
        i = dict(getattr(self.lead.engine, "info", {}) or {})
        i["elastic_awake"] = sum(1 for l in self.lanes if l.awake)
        return i

    @info.setter
    def info(self, v):
        self.lead.engine.info = v

    def _cur(self):
        return getattr(self.tl, "lane", None) or self.lead

    @property
    def last(self):
        # per request thread: the DONE figures of the lane THIS request ran on (a stable empty object until then)
        return getattr(self.tl, "last", _EMPTY)

    @last.setter
    def last(self, v):
        self.tl.last = v

    @property
    def progress(self):
        return getattr(self._cur().engine, "progress", None)

    @property
    def prefill_tok_s_mean(self):
        return getattr(self._cur().engine, "prefill_tok_s_mean", None)

    @property
    def silence_s(self):
        return getattr(self.lead.engine, "silence_s", 0)

    @silence_s.setter
    def silence_s(self, v):
        for l in self.lanes:
            l.engine.silence_s = v

    def alive(self) -> bool:
        return all(l.engine.alive() for l in self.lanes)

    def exit_code(self):
        for l in self.lanes:
            if not l.engine.alive():
                return l.engine.exit_code()
        return None

    def death_note(self) -> str:
        for l in self.lanes:
            if not l.engine.alive():
                return f"({l.name}) " + l.engine.death_note()
        return ""

    def restart(self, tries: int = 3):
        """Start every dead engine again: the helper while the lead has given its card back (then it sleeps), the
        lead only while the helper sleeps (its peer tier sizes on the free card)."""
        with self.transition:
            h = self.helper
            if not h.engine.alive():
                log("the helper had stopped: starting it again")
                try:
                    if self.lead.engine.alive():
                        self.lead.engine.control("PEER_DETACH", ("CTL PEER OFF",), 600)
                    h.engine.restart(tries)
                    h.awake, h.live = True, []
                    if not h.busy:
                        self._sleep_locked(h)
                except Exception as e:
                    log(f"helper: restart failed: {e}")
                    h.awake, h.broken_until = False, time.monotonic() + 60
                    self._lead_send("POOL_HOLD 0")
                    self._lead_send("PEER_ATTACH")
            if not self.lead.engine.alive():
                log("the lead had stopped: starting it again")
                if h.awake and not h.busy and h.engine.alive():
                    try:
                        self._sleep_locked(h, attach=False)
                    except Exception as e:
                        log(f"helper: could not put it to sleep before the lead restart: {e}")
                self.lead.engine.restart(tries)
                self.lead.live = []
                if h.awake:
                    self.lead.engine.control("PEER_DETACH", ("CTL PEER OFF",), 600)

    def unload(self):
        raise ValueError("the elastic pair does not unload")

    def vram(self, reserve_mib, timeout: float = 120.0):
        raise ValueError("the elastic pair moves the helper card's VRAM itself (not with POST /v1/vram)")

    def close(self):
        self._stop = True
        for l in self.lanes:
            try:
                l.engine.close()
            except Exception:
                pass

    # ---------------------------------------------------------------- transitions
    def _lead_send(self, cmd):
        try:
            self.lead.engine.send(cmd)
        except (OSError, AttributeError) as e:
            log(f"lead {cmd}: {e}")

    def _toggle(self, lane, what, want):
        toggle(self.checkpoint, lane.engine, f"{lane.name}: {what}", want)

    def _wake_locked(self, h: Lane):
        t0 = time.monotonic()
        r = self.lead.engine.control("PEER_DETACH", ("CTL PEER OFF",), 600)
        t1 = time.monotonic()
        if self.hold_workers > 0:
            self.lead.engine.control(f"POOL_HOLD {self.hold_workers}", ("CTL HOLD",), 600)
        self._toggle(h, "thawed", "running")
        t2 = time.monotonic()
        v = h.engine.control("VRAM", ("VRAM ",), 300)       # the reserve it started with: its whole cache again
        h.engine.control("LINK_SERVE 1", ("CTL LINK",), 60)  # the lead may use its card's cache again
        h.awake = True
        dt = time.monotonic() - t0
        self.stats["wakes"] += 1
        self.stats["wake_s"] = (self.stats["wake_s"] + [round(dt, 2)])[-20:]
        log(f"helper awake in {dt:.2f} s (the lead's peer tier off after {t1 - t0:.2f} s [{r}], thaw "
            f"{t2 - t1:.2f} s, cache {time.monotonic() - t2:.2f} s: {v})")

    def _sleep_locked(self, h: Lane, attach: bool = True):
        t0 = time.monotonic()
        h.engine.control("LINK_SERVE 0", ("CTL LINK",), 60)  # the lead plans nothing more for this card
        v = h.engine.control("VRAM 1000000", ("VRAM ",), 300)   # the cache shrinks to its smallest size
        self._toggle(h, "frozen", "checkpointed")
        h.awake = False
        if attach:
            self._lead_send("POOL_HOLD 0")
            self._lead_send("PEER_ATTACH")      # served when the lead is between requests
        self.stats["sleeps"] += 1
        log(f"helper asleep in {time.monotonic() - t0:.2f} s ({v}); the lead takes its card back")

    def _sleeper(self):
        while not self._stop:
            time.sleep(1.0)
            h = self.helper
            with self.cv:
                due = h.awake and not h.busy and time.monotonic() - h.idle_since >= self.hold_s
                if due:
                    h.busy = True                   # reserved: no request picks it while it goes to sleep
            if not due:
                continue
            try:
                with self.transition:
                    self._sleep_locked(h)
            except Exception as e:                  # it stays awake; the next round tries again
                log(f"helper: sleep failed: {e}")
            finally:
                with self.cv:
                    h.busy = False
                    h.idle_since = time.monotonic()
                    self.cv.notify_all()

    # ---------------------------------------------------------------- routing
    def _pick(self, ids):
        now = time.monotonic()
        idle = [l for l in self.lanes if not l.busy and (l.awake or l.broken_until <= now)]
        if not idle:
            return None
        awake = [l for l in idle if l.awake]
        if awake:
            # the lane holding the longest shared prefix (its conversation cache / live session), the lead on ties
            return max(awake, key=lambda l: (common_prefix(l.live, ids), l is self.lead))
        return idle[0]

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        ids = list(ids)
        with self.cv:
            lane = self._pick(ids)
            while lane is None:
                self.waits += 1
                self.cv.wait(timeout=1.0)
                if cancel.is_set():
                    return
                lane = self._pick(ids)
            lane.busy = True
            parallel = sum(1 for l in self.lanes if l.busy) > 1
        if parallel:
            self.stats["parallel_requests"] += 1
        self.tl.lane = lane
        out: list[int] = []
        last0 = getattr(lane.engine, "last", None)
        try:
            if not lane.awake:
                try:
                    with self.transition:
                        if not lane.awake:
                            log("a request while the lead is busy: waking the helper")
                            self._wake_locked(lane)
                except Exception as e:
                    # it stays asleep for a minute; this request waits for the lead instead
                    log(f"helper: wake failed ({e}) - the request waits for the lead")
                    try:
                        self._toggle(lane, "frozen again", "checkpointed")
                    except Exception as e2:
                        log(f"helper: could not freeze it again: {e2}")
                    self._lead_send("POOL_HOLD 0")
                    self._lead_send("PEER_ATTACH")
                    with self.cv:
                        lane.busy = False
                        lane.broken_until = time.monotonic() + 60
                        self.cv.notify_all()
                        while self.lead.busy:
                            self.cv.wait(timeout=1.0)
                            if cancel.is_set():
                                return
                        lane = self.lead
                        lane.busy = True
                    self.tl.lane = lane
            lane.requests += 1
            gen = lane.engine.generate(ids, max_new, sampling, cancel, embeddings=embeddings) if embeddings \
                else lane.engine.generate(ids, max_new, sampling, cancel)
            try:
                for t in gen:
                    if t is not None:
                        out.append(t)
                    yield t
            finally:
                gen.close()
        finally:
            cur = self.tl.lane
            done = getattr(cur.engine, "last", None)
            if done is not None and (cur is not lane or done is not last0):
                self.tl.last = done
            with self.cv:
                cur.live = ids + out
                cur.busy = False
                cur.idle_since = time.monotonic()
                self.cv.notify_all()

    def status(self) -> dict:
        with self.cv:
            return {"hold_s": self.hold_s, "checkpoint": bool(self.checkpoint), "waits": self.waits, **self.stats,
                    "lanes": [{"name": l.name, "awake": l.awake, "busy": l.busy, "requests": l.requests,
                               "live_tokens": len(l.live)} for l in self.lanes]}


def _arg(args, flag):
    return args[args.index(flag) + 1] if flag in args else None


def build(cfg: dict, StrataEngine, engine_args, child_env, exe_of, silence: float) -> ElasticEngine:
    """The pair from the lead's run config and its "elastic" block (see the module text).  The helper starts first
    (its cache sizes on an empty card), gives its cache back (and is frozen, with a checkpoint tool); only then the
    lead starts and takes the helper's card as its peer tier."""
    EE = make_engine_class(StrataEngine)
    el = cfg["elastic"]
    base = Path(cfg.get("cwd") or ".")
    p = Path(el["helper"])
    hcfg = json.loads((p if p.is_absolute() else base / p).read_text(encoding="utf-8-sig"))
    hargs, largs = engine_args(hcfg), engine_args(cfg)
    link = _arg(largs, "--peer-link")
    if link is None or _arg(hargs, "--peer-link") != link:
        raise SystemExit("[elastic] the lead and the helper need the same --peer-link FILE")
    if "--peer-device" not in largs or "--vram-elastic" not in hargs:
        raise SystemExit("[elastic] the lead needs --peer-device, the helper --vram-elastic")
    try:                                        # a stale link file from an earlier run: its counters are not ours
        os.unlink(link)
    except FileNotFoundError:
        pass
    checkpoint = el.get("checkpoint")
    log(f"starting the helper engine (GPU {hcfg.get('gpu')}, cores {el.get('helper_cores')}) ...")
    helper = EE(exe_of(hcfg), hargs, cwd=hcfg.get("cwd"), log=hcfg.get("log"), env=child_env(hcfg),
                cores=_cores(el.get("helper_cores")))
    helper.silence_s = silence
    helper.control("LINK_SERVE 0", ("CTL LINK",), 60)
    v = helper.control("VRAM 1000000", ("VRAM ",), 300)
    toggle(checkpoint, helper, "helper: frozen", "checkpointed")
    log(f"helper ready and asleep ({v})")
    log("starting the lead engine ...")
    lead = EE(exe_of(cfg), largs, cwd=cfg.get("cwd"), log=cfg.get("log"), env=child_env(cfg),
              cores=_cores(el.get("lead_cores")))
    lead.silence_s = silence
    hold = os.environ.get("STRATA_ELASTIC_HOLD_S")   # a start script's override of the config's hold time
    eng = ElasticEngine(lead, helper, hold_s=float(hold if hold else el.get("hold_s", 30)),
                        hold_workers=int(el.get("hold_workers", 0)), checkpoint=checkpoint)
    log(f"ready: lead + helper, the helper sleeps after {eng.hold_s:.0f} s without a request")
    return eng
