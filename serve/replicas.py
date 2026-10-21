"""Data-parallel replicas: N independent engines, each on its own group of cards, behind one server.

A model that fits two cards (IQ3_S on two R9700s) runs better as two 2-card engines than as one 4-card layer split:
the split runs one pipeline whose stages wait on each other, two replicas run two pipelines.  The server keeps its one
HTTP front, its one Service and its one set of endpoints; `ReplicaEngine` stands in for the engine and spreads the
requests over the replicas.

Opt-in, in the run config:

    "replicas": 2                                     the "gpu" list cut into 2 equal groups ([0,1] and [2,3])
    "replicas": [{"gpus": [0, 1]}, {"gpus": [2, 3]}]  the groups named; an entry may add "env" and "args"

or `serve/server.py --replicas 2`.  Nothing changes without it.

Port mode.  When every entry has a "port" ({"gpus": [1], "port": 8081}), each replica is a separate, ordinary server
on its own port instead (`run_port_mode`): this process only starts them one after another, pins their cores, ends
them when it stops (and they end with it), and deletes "replica_shared_arena" (one copy of the experts for all of
them, `--shared-expert-arena`).  Some entries with a port and some without refuse the start.

Routing.  Each request goes to the replica with the lowest cost, cost = prompt tokens the replica would have to read
again (it holds the start of a conversation in its cache) + LOAD_TOKENS for every request already running there.  A
conversation's next turn therefore goes back to its replica, until that replica is so much busier than another that
reading the prompt again is cheaper than waiting.  What each replica holds is tracked as hashes of 64-token blocks of
the prompts and answers it served (the engine itself checks the real prefix, as it always does).

CPU cores.  On Linux the replicas get disjoint physical cores (serve/cpu_affinity.py, from PR #1436 by agorevski), so
the host threads of the engines (the expert workers) do not compete.  "replica_cpus": false turns it off.
"""
from __future__ import annotations

import collections
import copy
import os
import threading
import time

BLOCK = 64                          # tokens per hashed block of a prompt
LOAD_TOKENS = 3000                  # the cost of one request already running on a replica, in prompt tokens
INDEX_MAX = 262144                  # block hashes kept (about 17M tokens of prompts)
HEAL_COOLDOWN_S = 60.0              # a replica that died is started again at most this often


class ReplicaConfigError(ValueError):
    pass


def gpu_groups(cfg: dict) -> list[list[int]]:
    """The cards of each replica, from the config's "replicas" (see the module doc); [] when there are none or one."""
    spec = cfg.get("replicas")
    if spec is None or spec is False:
        return []
    cards = _cards(cfg.get("gpu"))
    if isinstance(spec, bool) or (not isinstance(spec, (int, list))):
        raise ReplicaConfigError(f'"replicas" must be a number or a list of {{"gpus": [...]}}, not {spec!r}')
    if isinstance(spec, int):
        if spec <= 1:
            return []
        if not cards:
            raise ReplicaConfigError(f'"replicas": {spec} needs the "gpu" list to cut into groups (e.g. "gpu": [0, 1, 2, 3])')
        if len(cards) % spec:
            raise ReplicaConfigError(f'"replicas": {spec} does not divide the {len(cards)} cards {cards} evenly: name the '
                                     f'groups ("replicas": [{{"gpus": [0, 1]}}, ...]) or change the count')
        n = len(cards) // spec
        return [cards[i * n:(i + 1) * n] for i in range(spec)]
    groups = []
    for e in spec:
        g = e.get("gpus") if isinstance(e, dict) else e
        groups.append(_cards(g))
    if any(not g for g in groups):
        raise ReplicaConfigError('every replica needs its cards: {"gpus": [0, 1]}')
    flat = [c for g in groups for c in g]
    if len(set(flat)) != len(flat):
        raise ReplicaConfigError(f"a card is in two replicas: {groups}")
    if len(groups) < 2:
        return []
    return groups


def port_mode(cfg: dict) -> list[int] | None:
    """Port mode (see the module doc): the replicas' ports in entry order when every entry of a "replicas" list names
    one, None when none does.  Ports on some entries only, a repeated port, a port that is not a whole number in
    1..65535, or "replica_shared_arena" without ports are refused (ReplicaConfigError): what to do then is not clear."""
    spec = cfg.get("replicas")
    entries = spec if isinstance(spec, list) else []
    ports = [e.get("port") if isinstance(e, dict) else None for e in entries]
    named = [p for p in ports if p is not None]
    if not named:
        if cfg.get("replica_shared_arena"):
            raise ReplicaConfigError('"replica_shared_arena" needs replica ports: give every "replicas" entry a "port"')
        return None
    if len(named) != len(ports):
        missing = [i for i, p in enumerate(ports) if p is None]
        raise ReplicaConfigError(f'every replica needs a "port" once one has one (entries {missing} have none)')
    for i, p in enumerate(ports):
        if isinstance(p, bool) or not isinstance(p, int) or not 1 <= p <= 65535:
            raise ReplicaConfigError(f'replica {i}: "port" must be a whole number in 1..65535, not {p!r}')
    seen = [p for p in ports if ports.count(p) > 1]
    if seen:
        raise ReplicaConfigError(f"two replicas have the port {seen[0]}")
    return ports


def _cards(g) -> list[int]:
    if g is None or g == "":
        return []
    items = g if isinstance(g, (list, tuple)) else str(g).split(",")
    try:
        return [int(str(x).strip()) for x in items if str(x).strip() != ""]
    except ValueError:
        raise ReplicaConfigError(f"cards must be numbers, not {g!r}") from None


def _suffixed(path, i: int):
    if not isinstance(path, str) or not path:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}.r{i}{ext}"


def replica_cfg(cfg: dict, i: int, gpus: list[int]) -> dict:
    """The config of replica i: its own cards, log and learned-profile file; the entry's "env" and "args" added."""
    c = copy.deepcopy(cfg)
    spec = cfg.get("replicas")
    entry = spec[i] if isinstance(spec, list) and isinstance(spec[i], dict) else {}
    c.pop("replicas", None)
    c["gpu"] = gpus if len(gpus) > 1 else gpus[0]
    c.pop("hip_ordinal", None)                 # a one-card config's Windows ordinal: not for a group
    c["log"] = _suffixed(cfg.get("log"), i)
    if cfg.get("expert_profile_save"):
        c["expert_profile_save"] = _suffixed(cfg["expert_profile_save"], i)
    if entry.get("env"):
        c["env"] = {**(c.get("env") or {}), **{str(k): str(v) for k, v in entry["env"].items()}}
    if entry.get("args"):
        c["args"] = list(c["args"]) + [str(x) for x in entry["args"]]
    return c


def port_child_cfg(cfg: dict, i: int, gpus: list[int], port: int) -> dict:
    """Port mode: the run config of replica i's own server - replica_cfg's, with its port, and the shared arena (when
    "replica_shared_arena" is set) as the engine's --shared-expert-arena."""
    c = replica_cfg(cfg, i, gpus)
    arena = c.pop("replica_shared_arena", None)
    c["port"] = port
    if arena:
        c["args"] = list(c.get("args") or []) + ["--shared-expert-arena", str(arena)]
    return c


# the supervisor's own options, not passed to a replica's server (each gets its own --config and --port)
_PARENT_ONLY = ("--config", "--port", "--replicas", "--replica-shared-arena", "--gpu", "--ready-file", "--owner-fd")


def child_argv(launch_args: list[str], cfg_path: str, port: int) -> list[str]:
    """Port mode: a replica server's command-line options - the supervisor's own, less those in _PARENT_ONLY (as
    "--opt value" or "--opt=value"), with this replica's --config and --port."""
    out, skip = [], False
    for x in launch_args:
        if skip:
            skip = False
            continue
        name = x.split("=", 1)[0]
        if name in _PARENT_ONLY:
            skip = "=" not in x
            continue
        out.append(x)
    return out + ["--config", cfg_path, "--port", str(port)]


def cpu_sets(n: int, enabled: bool = True):
    """n disjoint sets of CPUs (whole physical cores, SMT siblings together), or None (off, not Linux, or the topology
    cannot be read: the engines then share the cores as they would without replicas)."""
    if not enabled or n < 2 or not hasattr(os, "sched_getaffinity"):
        return None
    try:
        from serve.cpu_affinity import partition_cpu_sets
        return partition_cpu_sets(n, set(os.sched_getaffinity(0)))
    except (ValueError, OSError, ImportError) as e:
        print(f"[strata] replicas: no separate CPU cores ({e})", flush=True)
        return None


def _loaded(host: str, port: int) -> bool:
    import json
    from strata_control import loopback               # never through a proxy of the environment
    probe = "127.0.0.1" if host in ("", "0.0.0.0", "::") else host
    try:
        with loopback.urlopen(f"http://{probe}:{port}/health", timeout=2) as r:
            return json.loads(r.read() or b"{}").get("loaded") is True
    except (OSError, ValueError):
        return False


def _port_free(host: str, port: int) -> bool:
    import socket
    s = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET)
    try:
        if os.name != "nt":                           # on Windows it would let a port already serving be bound
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def run_port_mode(cfg: dict, launch_args: list[str], host: str, *, server_py: str, cfg_path: str | None = None,
                  ready_timeout_s: float = 900.0) -> int:
    """Port mode: one ordinary server per replica, each on its own port, started one after another (the first fills a
    shared arena, the next ones attach). This process only supervises: it forwards a stop to the servers, ends them
    with itself (they get the parent-death signal) and deletes the shared arena when it stops. Returns the exit code.
    With `cfg_path`, each replica's run config is written next to it as <name>.replica<i>.json (removed at stop), so
    the replica's shared Chat settings (<name>.replica<i>.shared-settings.json) are kept there between starts."""
    import json
    import shutil
    import signal
    import subprocess
    import sys
    import tempfile
    from serve import lifeline

    ports = port_mode(cfg) or []
    if "--lazy" in launch_args or cfg.get("lazy_load") is True:
        print("[strata] replicas on their own ports start loaded: leave out --lazy / \"lazy_load\"", flush=True)
        return 2
    for i, port in enumerate(ports):
        if not _port_free(host, port):
            print(f"[strata] replica {i}: port {port} is in use", flush=True)
            return 2
    groups = gpu_groups(cfg)
    cpus = cpu_sets(len(ports), cfg.get("replica_cpus") is not False)
    ready_timeout_s = float(os.environ.get("STRATA_REPLICA_READY_S") or ready_timeout_s)
    stop = threading.Event()
    stopped_by: list[str] = []

    def on_signal(signum, frame):
        stopped_by.append(signal.Signals(signum).name)
        stop.set()
    old = {}
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is not None and not (name == "SIGHUP" and signal.getsignal(sig) == signal.SIG_IGN):
            old[sig] = signal.signal(sig, on_signal)

    tmp = tempfile.mkdtemp(prefix="strata-replicas-")
    written: list[str] = []
    children: list[subprocess.Popen] = []
    code = 0
    try:
        for i, (gpus, port) in enumerate(zip(groups, ports)):
            path = os.path.splitext(cfg_path)[0] + f".replica{i}.json" if cfg_path else os.path.join(tmp, f"replica{i}.json")
            written.append(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(port_child_cfg(cfg, i, gpus, port), f)
            pin = cpus[i] if cpus else None
            child = lifeline.popen([sys.executable, server_py, *child_argv(launch_args, path, port)],
                                   role="replica", start_new_session=True,
                                   preexec_fn=(lambda c=pin: os.sched_setaffinity(0, c)) if pin else None)
            children.append(child)
            deadline, why = time.monotonic() + ready_timeout_s, None
            while not _loaded(host, port):
                if child.poll() is not None:
                    why = f"exit {child.returncode}"
                elif stop.is_set():
                    why = "stopped"
                elif time.monotonic() > deadline:
                    why = f"not loaded after {ready_timeout_s:.0f} s"
                if why:
                    break
                stop.wait(0.5)
            if why:
                print(f"[strata] replica {i} did not start: {why}", flush=True)
                return 0 if stopped_by else 1
            print(f"[strata] replica {i} on cards {gpus}: http://{host}:{port}/v1", flush=True)
        print("[strata] replicas each on their own port; this process only supervises them (its own --port is "
              "not used)", flush=True)
        while not stop.is_set():
            for k, c in enumerate(children):
                if c.poll() is not None:
                    print(f"[strata] replica {k} stopped (exit {c.returncode})", flush=True)
                    code = code or c.returncode or 1
                    stop.set()
            stop.wait(1.0)
        return 0 if stopped_by else code
    finally:
        for c in children:
            if c.poll() is None:
                c.send_signal(signal.SIGTERM)
        for c in children:
            try:
                c.wait(60)
            except subprocess.TimeoutExpired:
                c.kill()
                c.wait()
        for sig, h in old.items():
            signal.signal(sig, h)
        arena = cfg.get("replica_shared_arena")
        if arena:
            try:
                os.unlink(arena)
            except OSError:
                pass
        for path in written:
            try:
                os.unlink(path)
            except OSError:
                pass
        shutil.rmtree(tmp, ignore_errors=True)


class Router:
    """Which replica holds the start of a prompt: chained hashes of its 64-token blocks, each with the set of replicas
    that served it (a bit mask)."""

    def __init__(self, n: int, cap: int = INDEX_MAX):
        self.n, self.cap = n, cap
        self.blocks: collections.OrderedDict[int, int] = collections.OrderedDict()
        self.lock = threading.Lock()

    @staticmethod
    def chain(ids) -> list[int]:
        out, h = [], 0
        for k in range(len(ids) // BLOCK):
            h = hash((h, tuple(ids[k * BLOCK:(k + 1) * BLOCK])))
            out.append(h)
        return out

    def matched(self, ids) -> list[int]:
        """Per replica: how many leading tokens of `ids` it holds (a multiple of BLOCK)."""
        got = [0] * self.n
        alive = (1 << self.n) - 1
        with self.lock:
            for k, h in enumerate(self.chain(ids)):
                m = self.blocks.get(h, 0) & alive
                if not m:
                    break
                alive = m
                for r in range(self.n):
                    if m >> r & 1:
                        got[r] = (k + 1) * BLOCK
        return got

    def note(self, r: int, ids) -> None:
        with self.lock:
            for h in self.chain(ids):
                self.blocks[h] = self.blocks.pop(h, 0) | (1 << r)
            while len(self.blocks) > self.cap:
                self.blocks.popitem(last=False)

    def forget(self, r: int) -> None:
        """Replica r lost its cache (it restarted)."""
        with self.lock:
            bit = ~(1 << r)
            for h in [h for h, m in self.blocks.items() if m >> r & 1]:
                m = self.blocks[h] & bit
                if m:
                    self.blocks[h] = m
                else:
                    del self.blocks[h]


class ReplicaEngine:
    """Stands in for one engine (the interface Service uses), over several."""

    NOT_HERE = {"session_file", "vram", "ctl", "slot_cv"}       # not offered: one conversation file / VRAM resize / control line
    PROPAGATE = {"silence_s", "gpu_busy"}

    def __init__(self, engines: list, load_tokens: int = LOAD_TOKENS):
        if len(engines) < 2:
            raise ValueError("ReplicaEngine needs two or more engines")
        d = self.__dict__
        d["engines"] = list(engines)
        d["router"] = Router(len(engines))
        d["load_tokens"] = load_tokens
        d["inflight"] = [0] * len(engines)
        d["run_locks"] = [threading.Lock() for _ in engines]    # an engine without batch slots: one request at a time
        d["lock"] = threading.Lock()
        d["tl"] = threading.local()
        d["healing"] = {}
        d["heal_at"] = [0.0] * len(engines)
        d["routed"] = [0] * len(engines)
        d["gen_seen"] = [getattr(e, "gen", 0) for e in engines]

    # ---- attribute plumbing
    @property
    def current(self):
        return getattr(self.tl, "cur", None) or self.engines[0]

    def __getattr__(self, name):
        if name in self.NOT_HERE or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.current, name)

    def __setattr__(self, name, value):
        if name in self.PROPAGATE:
            for e in self.engines:
                setattr(e, name, value)
        else:
            self.__dict__[name] = value

    @property
    def last(self):
        return self.current.last

    @last.setter
    def last(self, value):
        self.current.last = value

    @property
    def batch(self) -> int:
        """Parallel slots, in total: the Service lets this many requests run at once (it holds no FIFO).  An engine
        without slots counts one."""
        return sum(max(1, int(getattr(e, "batch", 0) or 0)) for e in self.engines)

    @property
    def waiting(self) -> int:
        return sum(int(getattr(e, "waiting", 0) or 0) for e in self.engines)

    @property
    def max_context(self) -> int:
        v = [e.max_context for e in self.engines if getattr(e, "max_context", 0) and e.alive()]
        return min(v) if v else 0

    @property
    def known_ctx(self) -> int:
        v = [getattr(e, "known_ctx", 0) for e in self.engines]
        return min(x for x in v if x) if any(v) else 0

    @property
    def unloaded(self) -> bool:
        return all(getattr(e, "unloaded", False) for e in self.engines)

    @property
    def starting(self) -> bool:
        return not self.alive() and any(getattr(e, "starting", False) for e in self.engines)

    @property
    def info(self) -> dict:
        i = dict(self.engines[0].info)
        i["replicas"] = len(self.engines)
        i["replicas_alive"] = sum(1 for e in self.engines if e.alive())
        return i

    @info.setter
    def info(self, value):
        pass

    @property
    def spawn(self):
        return self.engines[0].spawn

    @property
    def log_path(self):
        return self.engines[0].log_path

    @property
    def progress(self):
        c = getattr(self.tl, "cur", None)
        if c is not None:
            return c.progress
        return next((e.progress for e in self.engines if getattr(e, "progress", None)), None)

    @property
    def prefill_tok_s_mean(self):
        c = getattr(self.tl, "cur", None)
        if c is not None:
            return c.prefill_tok_s_mean
        return next((e.prefill_tok_s_mean for e in self.engines if getattr(e, "prefill_tok_s_mean", None)), None)

    def alive(self) -> bool:
        return any(e.alive() for e in self.engines)

    def exit_code(self):
        return next((c for c in (e.exit_code() for e in self.engines if not e.alive()) if c is not None), None)

    def death_note(self) -> str:
        return next((e.death_note() for e in self.engines if not e.alive()), "")

    def slots_view(self) -> list[dict]:
        out, base = [], 0
        for k, e in enumerate(self.engines):
            n = int(getattr(e, "batch", 0) or 0)
            view = e.slots_view() if n and hasattr(e, "slots_view") else \
                [{"slot": 0, "state": "busy" if self.inflight[k] else "idle", "held_tokens": 0}]
            for s in view:
                out.append({**s, "slot": base + s["slot"], "replica": k})
            base += max(1, n)
        return out

    def replica_view(self) -> list[dict]:
        """For /metrics: each replica's state, requests running, and requests routed to it so far."""
        return [{"replica": k, "alive": e.alive(), "running": self.inflight[k], "routed": self.routed[k],
                 "batch": int(getattr(e, "batch", 0) or 0)} for k, e in enumerate(self.engines)]

    # ---- life cycle
    def restart(self, tries: int = 3):
        dead = [k for k, e in enumerate(self.engines) if not e.alive()]
        errors: dict[int, BaseException] = {}

        def one(k):
            try:
                self.engines[k].restart(tries)
                self.router.forget(k)
            except BaseException as e:           # noqa: BLE001
                errors[k] = e
        ts = [threading.Thread(target=one, args=(k,), daemon=True) for k in dead]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if errors and not self.alive():
            raise next(iter(errors.values()))
        for k, e in errors.items():
            print(f"[strata] replica {k} did not start ({e}); the others serve", flush=True)

    def unload(self):
        for e in self.engines:
            e.unload()
        for k in range(len(self.engines)):
            self.router.forget(k)

    def close(self):
        err = None
        for e in self.engines:
            try:
                e.close()
            except BaseException as x:           # noqa: BLE001
                err = err or x
        if err:
            raise err

    # ---- routing
    def route(self, ids) -> int:
        """The replica for this prompt (and its running count raised): lowest cost among the living ones."""
        got = self.router.matched(ids)
        with self.lock:
            alive = [k for k, e in enumerate(self.engines) if e.alive()]
            if not alive:
                return -1
            cost = {k: (len(ids) - got[k]) + self.load_tokens * self.inflight[k] for k in alive}
            # the least loaded wins a tie of cost (equal cards, nothing cached): spread, do not pile on replica 0
            k = min(alive, key=lambda k: (cost[k], self.inflight[k], self.routed[k]))
            self.inflight[k] += 1
            self.routed[k] += 1
        if len(alive) < len(self.engines):
            self.heal()
        return k

    def heal(self) -> None:
        """Start a replica that died again, in the background (the others keep serving)."""
        now = time.monotonic()
        for k, e in enumerate(self.engines):
            if e.alive() or getattr(e, "unloaded", False) or getattr(e, "starting", False) or self.healing.get(k):
                continue
            if now < self.heal_at[k]:
                continue
            self.heal_at[k] = now + HEAL_COOLDOWN_S
            self.healing[k] = True

            def run(k=k):
                try:
                    print(f"[strata] replica {k} had stopped; starting it again", flush=True)
                    self.engines[k].restart()
                    self.router.forget(k)
                    print(f"[strata] replica {k} is running again", flush=True)
                except BaseException as x:       # noqa: BLE001
                    print(f"[strata] replica {k} did not start again: {x}", flush=True)
                finally:
                    self.healing[k] = False
            threading.Thread(target=run, daemon=True).start()

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        # imported here: serve.server imports this module
        from serve.server import EngineDied, EngineNonFinite
        k = self.route(ids)
        if k < 0:
            raise EngineDied("no replica is running; this request was not sent")
        e = self.engines[k]
        self.tl.cur = e
        produced: list[int] = []
        held = False
        nonfinite = False
        try:
            selected = getattr(cancel, "_strata_replica_selected", None)
            if selected is not None:
                selected(k)
            self.router.note(k, ids)
            if not int(getattr(e, "batch", 0) or 0):          # no slots: this replica takes one request at a time
                lock = self.run_locks[k]
                while not lock.acquire(timeout=0.25):
                    if cancel is not None and cancel.is_set():
                        return
                held = True
            if embeddings is not None:
                gen = e.generate(ids, max_new, sampling, cancel, embeddings=embeddings)
            else:
                gen = e.generate(ids, max_new, sampling, cancel)
            for t in gen:
                if t is not None:
                    produced.append(t)
                yield t
        except EngineNonFinite:
            nonfinite = True
            raise
        finally:
            if held:
                self.run_locks[k].release()
            with self.lock:
                self.inflight[k] -= 1
            if nonfinite:                       # #879: the engine dropped every cached conversation
                self.router.forget(k)
            elif produced:
                self.router.note(k, list(ids) + produced)
