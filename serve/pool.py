"""serve/pool.py - the pool: one model across several PCs, configured from the web app's Pool tab.

A pool is one COORDINATOR - the PC whose server you chat with; it runs the first layers, the output head, the sampler
and the draft layer - and one or more WORKERS, each of which runs a contiguous range of the remaining layers on its
own GPU, with that range's experts in its own RAM and VRAM and that range's share of the context (KV and recurrent
state).  The engines talk to each other directly over TCP (include/strata/pool/protocol.hpp); this module is the
part around them:

- `PoolConfig`: the role of this PC and its peers, kept next to the run config as `<config>.pool.json`.
- `WorkerSupervisor`: on a worker, keeps `strata --serve --pool-listen` running (it waits for a coordinator, loads the
  layers that coordinator assigns, serves it, and stays loaded for the next one), and reads its `POOL ...` lines.
- `Discovery`: a UDP beacon on the LAN, so the Pool tab can list the Strata PCs with a pool that it sees.
- `PoolManager`: the server's side - the engine arguments for the role, switching roles without restarting the
  server, and GET /pool's state.

The other mode, ROUTING (role "router", serve/route.py), keeps the whole model on every PC and sends each chat request
to the PC that suits it: the one that holds the conversation, else an idle one.  Its peers are the other PCs' web
apps (host:8080), in `route_peers`.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import secrets
import socket
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path

ENGINE_PORT = 7701          # a worker's engine listens here for its coordinator (TCP)
DISCOVERY_PORT = 7702       # beacons (UDP broadcast)
BEACON_EVERY_S = 3.0
BEACON_FRESH_S = 15.0
ROLES = ("off", "router", "coordinator", "worker")
HTTP_PORT = 8080
WIRES = ("f32", "f16", "bf16")


def hostname() -> str:
    return os.environ.get("COMPUTERNAME") or socket.gethostname() or "this-pc"


# ------------------------------------------------------------------------------------------------ the config
class PoolConfig:
    """This PC's part in a pool.  `role`: off (a normal single-PC Strata), router (routing mode: the whole model here,
    requests shared with the PCs in `route_peers`), coordinator (layer split: uses the workers in `peers`) or worker
    (lends its GPU to a coordinator).  Every PC of a pool shares `secret`."""

    DEFAULTS = {"role": "off", "name": "", "secret": "", "worker_port": ENGINE_PORT, "peers": [], "split": "auto",
                "wire": "f32", "draft_wire": "f16", "discovery": True,
                "route_peers": [], "primary": False, "move_tokens": 8000}

    def __init__(self, path: str | None = None, data: dict | None = None):
        self.path = path
        self.data = self.clean({**copy.deepcopy(self.DEFAULTS), **(data or {})})

    @classmethod
    def load(cls, path: str | None) -> "PoolConfig":
        data = {}
        if path and Path(path).is_file():
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
            except (OSError, ValueError) as e:
                print(f"[pool] {path} could not be read ({e}); starting with the defaults", flush=True)
                data = {}
        try:
            pc = cls(path, data)
        except ValueError as e:
            print(f"[pool] {path}: {e}; starting with the defaults", flush=True)
            pc = cls(path, {})
        if not pc.data["secret"]:                     # every pool needs one; make it once, keep it
            pc.data["secret"] = secrets.token_urlsafe(18)
            pc.save()
        return pc

    def save(self) -> None:
        if not self.path:
            return
        tmp = Path(str(self.path) + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def __getitem__(self, k):
        return self.data[k]

    @property
    def role(self) -> str:
        return self.data["role"]

    @staticmethod
    def clean(d: dict) -> dict:
        """The config with every field checked (ValueError names a bad one)."""
        if not isinstance(d, dict):
            raise ValueError("the pool config must be an object")
        out = copy.deepcopy(PoolConfig.DEFAULTS)
        for k, v in d.items():
            if k not in out:
                continue                              # unknown keys are dropped
            out[k] = v
        if out["role"] not in ROLES:
            raise ValueError(f"role: one of {', '.join(ROLES)}")
        if not isinstance(out["name"], str) or len(out["name"]) > 64:
            raise ValueError("name: up to 64 characters")
        if not isinstance(out["secret"], str) or len(out["secret"]) > 200 or any(c in out["secret"] for c in "\r\n"):
            raise ValueError("secret: up to 200 characters on one line")
        p = out["worker_port"]
        if isinstance(p, str) and p.isdigit():
            p = int(p)
        if not isinstance(p, int) or isinstance(p, bool) or not 1 <= p <= 65535:
            raise ValueError("worker_port: 1..65535")
        out["worker_port"] = p
        split = str(out["split"]).strip().replace(" ", "")
        if split != "auto" and not re.fullmatch(r"\d+(,\d+)*", split):
            raise ValueError("split: auto, or the first layer of each worker's share, e.g. 24 or 16,32")
        out["split"] = split
        for k in ("wire", "draft_wire"):
            if out[k] not in WIRES:
                raise ValueError(f"{k}: one of {', '.join(WIRES)}")
        out["discovery"] = bool(out["discovery"])
        peers = out["peers"]
        if not isinstance(peers, list) or len(peers) > 7:
            raise ValueError("peers: a list of up to 7 workers")
        clean_peers, seen = [], set()
        for x in peers:
            if isinstance(x, str):
                x = {"addr": x}
            if not isinstance(x, dict):
                raise ValueError("peers: each is {addr, enabled, label, http}")
            addr = normalize_addr(str(x.get("addr", "")), out["worker_port"])
            if addr is None:
                raise ValueError(f"peers: '{x.get('addr')}' is not host or host:port")
            if addr in seen:
                continue
            seen.add(addr)
            http = x.get("http")
            if http not in (None, ""):
                http = normalize_addr(str(http), 8080)
                if http is None:
                    raise ValueError(f"peers: '{x.get('http')}' is not a host:port for the worker's web app")
            label = str(x.get("label") or "")[:64]
            clean_peers.append({"addr": addr, "enabled": bool(x.get("enabled", True)), "label": label,
                                "http": http or None})
        out["peers"] = clean_peers
        rp = out["route_peers"]
        if not isinstance(rp, list) or len(rp) > 7:
            raise ValueError("route_peers: a list of up to 7 PCs")
        clean_rp, seen = [], set()
        for x in rp:
            if isinstance(x, str):
                x = {"addr": x}
            if not isinstance(x, dict):
                raise ValueError("route_peers: each is {addr, enabled, label}")
            addr = normalize_addr(str(x.get("addr", "")), HTTP_PORT)
            if addr is None:
                raise ValueError(f"route_peers: '{x.get('addr')}' is not host or host:port (its web app)")
            if addr in seen:
                continue
            seen.add(addr)
            clean_rp.append({"addr": addr, "enabled": bool(x.get("enabled", True)), "label": str(x.get("label") or "")[:64]})
        out["route_peers"] = clean_rp
        out["primary"] = bool(out["primary"])
        mt = out["move_tokens"]
        if isinstance(mt, str) and mt.isdigit():
            mt = int(mt)
        if not isinstance(mt, int) or isinstance(mt, bool) or not 0 <= mt <= 10_000_000:
            raise ValueError("move_tokens: 0..10000000")
        out["move_tokens"] = mt
        if out["split"] != "auto":
            n = len([p for p in clean_peers if p["enabled"]])
            ks = [int(k) for k in out["split"].split(",")]
            if out["role"] == "coordinator" and n and len(ks) != n:
                raise ValueError(f"split: {n} enabled worker(s) need {n} split point(s)")
            if any(b <= a for a, b in zip(ks, ks[1:])) or ks[0] < 2:
                raise ValueError("split: rising layer numbers from 2")
        return out

    def enabled_peers(self) -> list[dict]:
        return [p for p in self.data["peers"] if p["enabled"]]

    def public(self, reveal_secret: bool) -> dict:
        d = copy.deepcopy(self.data)
        d["secret_set"] = bool(d["secret"])
        if not reveal_secret:
            d["secret"] = ""
        return d


def normalize_addr(s: str, default_port: int) -> str | None:
    s = s.strip()
    if not s:
        return None
    if s.startswith("["):
        m = re.fullmatch(r"\[([0-9a-fA-F:.]+)\](?::(\d+))?", s)
        if not m:
            return None
        port = int(m.group(2) or default_port)
        return f"[{m.group(1)}]:{port}" if 0 < port < 65536 else None
    if s.count(":") > 1:                              # a bare v6 address
        return f"[{s}]:{default_port}"
    host, _, port = s.partition(":")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", host or "") or (port and not port.isdigit()):
        return None
    port = int(port or default_port)
    return f"{host}:{port}" if 0 < port < 65536 else None


def coordinator_args(pc: PoolConfig) -> list[str]:
    peers = [p["addr"] for p in pc.enabled_peers()]
    if not peers:
        return []
    return ["--pool-peers", ",".join(peers), "--pool-split", pc["split"], "--pool-wire", pc["wire"],
            "--pool-draft-wire", pc["draft_wire"]]


def worker_args(pc: PoolConfig) -> list[str]:
    return ["--pool-listen", f"0.0.0.0:{pc['worker_port']}"]


def pool_env(pc: PoolConfig, env: dict) -> dict:
    """The secret travels in the environment, not on the command line (which other users of the PC can read)."""
    env = dict(env)
    env["STRATA_POOL_SECRET"] = pc["secret"]
    env["STRATA_POOL_NAME"] = pc["name"] or hostname()
    return env


def strip_pool_args(args: list[str]) -> list[str]:
    """The engine arguments without any --pool-* (the role adds its own)."""
    out, skip = [], False
    for x in args:
        if skip:
            skip = False
            continue
        if x.startswith("--pool-"):
            skip = "=" not in x
            continue
        out.append(x)
    return out


def engine_has_pool(exe: str | None) -> bool | None:
    """Whether the engine binary has the layer split's network code (its --pool-listen flag); None when it cannot
    be read.  A ready-made engine older than the pool, or one built without it, has not."""
    if not exe:
        return None
    try:
        with open(exe, "rb") as f:
            data = f.read()
    except OSError:
        return None
    return b"--pool-listen" in data


def config_problems(cfg: dict, pc: PoolConfig) -> list[str]:
    """What in the run config a pool cannot do (the engine refuses these too; the Pool tab says so first)."""
    if pc.role in ("off", "router"):           # routing: every PC runs the whole model as it would alone
        return []
    args = cfg.get("args") or []
    out = []
    exe = cfg.get("exe")
    if exe and not os.path.isabs(exe):
        exe = os.path.join(cfg.get("cwd") or ".", exe)
    if engine_has_pool(exe) is False:
        out.append("this PC's engine was built without the layer split (an older or ready-made build): compile this "
                   "version's engine with START-HERE.bat --build, or use Share requests, which works with any engine")
    if cfg.get("vision"):
        out.append("images (vision) are on: a pool runs text only for now - run setup again with images off")
    if any(a in ("--control-vector", "--control-vector-scaled") for a in args):
        out.append("the experimental speed projection is on: a pool does not support it yet")
    if any(a in ("--mmap-experts", "--resident-experts", "--resident-cpu-experts") for a in args):
        out.append("the low-RAM mode is on: a pool node holds only its own layers' experts instead")
    if len([g for g in str(cfg.get("gpu", "")).split(",") if g.strip()]) > 1:
        out.append("this PC is set up for several GPUs: a pool node uses one (run setup with one GPU)")
    return out


# ------------------------------------------------------------------------------------------------ the worker
class WorkerSupervisor:
    """Keeps the worker engine running and knows what it is doing from its stdout:

        POOL LISTEN <port>              waiting for a coordinator
        POOL COORD <addr>               a coordinator connected (it sends the layer range next)
        POOL CONFIG <lb> <le> <ctx>     loading layers lb..le-1
        POOL READY lb= le= slots= ...   serving
        POOL REQ windows= ...           a request ended
        POOL IDLE <why>                 the coordinator left; the layers stay loaded
        POOL RELOAD                     the next coordinator wants other layers: restarting
        POOL REFUSED <addr> secret      a coordinator with another secret
        POOL FAILED                     an error while serving (it restarts)
    """

    def __init__(self, exe: str, args: list[str], cwd=None, log: str | None = None, env: dict | None = None,
                 echo=print):
        self.exe, self.args, self.cwd, self.log_path, self.env = exe, list(args), cwd, log, env
        self.echo = echo
        self.proc = None
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.spawn_lock = threading.Lock()
        self.status = {"state": "starting", "since": time.time(), "coordinator": None, "lb": None, "le": None,
                       "ready": {}, "requests": 0, "last_request": None, "restarts": 0, "refused": None,
                       "exit_code": None, "error": None, "port": None}
        self.thread = None

    def start(self):
        self.stopping.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self, timeout=20.0):
        with self.spawn_lock:                        # a start in progress finishes (or sees `stopping`) first
            self.stopping.set()
            p = self.proc
        if p is not None and p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=timeout)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    p.kill()
                except OSError:
                    pass
        if self.thread is not None:
            self.thread.join(timeout=timeout)
        self._set(state="stopped")

    def snapshot(self) -> dict:
        with self.lock:
            return copy.deepcopy(self.status)

    def _set(self, **kw):
        with self.lock:
            if "state" in kw and kw["state"] != self.status.get("state"):
                self.status["since"] = time.time()
            self.status.update(kw)

    def _run(self):
        backoff = 3.0
        while not self.stopping.is_set():
            log = open(self.log_path, "a", encoding="utf-8") if self.log_path else subprocess.DEVNULL
            try:
                with self.spawn_lock:
                    if self.stopping.is_set():
                        return
                    self.proc = subprocess.Popen([self.exe, "--serve", *self.args], cwd=self.cwd,
                                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log,
                                                 text=True, encoding="utf-8", errors="replace", bufsize=1,
                                                 env=self.env)
            except OSError as e:
                self._set(state="error", error=f"the engine could not start: {e}")
                self.echo(f"[pool] the worker engine could not start: {e}")
                if self.stopping.wait(30):
                    return
                continue
            try:
                from serve.winjob import contain
                contain(self.proc)
            except Exception:  # noqa: BLE001 - containment is a nicety
                pass
            self._set(state="loading weights", error=None, exit_code=None, coordinator=None)
            t0 = time.time()
            for line in self.proc.stdout:
                self._line(line.strip())
            code = self.proc.wait()
            self.proc.stdout.close()
            if log is not subprocess.DEVNULL:
                log.close()
            if self.stopping.is_set():
                return
            with self.lock:
                self.status["restarts"] += 1
                self.status["exit_code"] = code
            if code == 3:                              # it restarts to load another range: at once
                backoff = 3.0
                self._set(state="reloading")
                continue
            note = self._log_tail_reason()
            self._set(state="error", error=f"the worker engine exited (code {code}){': ' + note if note else ''}")
            self.echo(f"[pool] the worker engine exited (code {code}){': ' + note if note else ''}; "
                      f"starting it again in {backoff:.0f} s")
            if time.time() - t0 > 120:
                backoff = 3.0
            if self.stopping.wait(backoff):
                return
            backoff = min(backoff * 2, 60.0)

    def _log_tail_reason(self) -> str:
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 4096))
                tail = f.read().decode("utf-8", "replace")
        except (OSError, TypeError):
            return ""
        for line in reversed(tail.splitlines()):
            s = line.strip()
            if s.startswith("strata pool:") or s.startswith("strata generate:") or s.startswith("strata serve:"):
                return s
        return ""

    @staticmethod
    def _kv(text: str) -> dict:
        out = {}
        for kv in text.split():
            k, _, v = kv.partition("=")
            if v:
                out[k] = int(v) if v.lstrip("-").isdigit() else (float(v) if re.fullmatch(r"-?\d+\.\d*", v) else v)
        return out

    def _line(self, line: str):
        if not line.startswith("POOL "):
            return
        f = line.split(" ", 2)
        what, rest = f[1], f[2] if len(f) > 2 else ""
        if what == "LISTEN":
            self._set(state="waiting", port=int(rest) if rest.isdigit() else None)
            self.echo(f"[pool] worker ready to join a pool (port {rest}); waiting for a coordinator")
        elif what == "COORD":
            with self.lock:
                loaded = self.status.get("state") in ("idle", "serving")
            self._set(state="connected" if loaded else "loading", coordinator=rest)
            self.echo(f"[pool] coordinator {rest} connected")
        elif what == "CONFIG":
            p = rest.split()
            if len(p) >= 2:
                self._set(state="loading", lb=int(p[0]), le=int(p[1]))
                self.echo(f"[pool] loading layers {p[0]}-{int(p[1]) - 1} for the coordinator "
                          "(a minute or two: the experts of these layers go into RAM)")
        elif what == "READY":
            kv = self._kv(rest)
            self._set(state="serving", ready=kv, lb=kv.get("lb"), le=kv.get("le"))
            self.echo(f"[pool] serving layers {kv.get('lb')}-{(kv.get('le') or 1) - 1}: "
                      f"{kv.get('slots')} experts in VRAM, {kv.get('arena_mib', 0) / 1024:.1f} GB of experts in RAM")
        elif what == "REQ":
            kv = self._kv(rest)
            with self.lock:
                self.status["requests"] += 1
                self.status["last_request"] = {**kv, "at": time.time()}
        elif what == "IDLE":
            self._set(state="idle", coordinator=None)
            self.echo(f"[pool] {rest}; layers stay loaded for the next coordinator")
        elif what == "RELOAD":
            self._set(state="reloading")
        elif what == "REFUSED":
            self._set(refused={"from": rest.split()[0] if rest else "?", "at": time.time()})
            self.echo(f"[pool] refused {rest.split()[0] if rest else 'a coordinator'}: its pool secret differs")
        elif what == "FAILED":
            self._set(state="error", error="the worker failed while serving (see the log)")


# ------------------------------------------------------------------------------------------------ discovery
class Discovery:
    """LAN beacons: every few seconds each Strata server with a pool broadcasts who it is (name, role, model, GPU, ports);
    the Pool tab lists the others it hears.  No secret is sent, and nothing is trusted from a beacon: it only fills a
    list the user picks from.  Fails quietly where UDP broadcast is blocked."""

    def __init__(self, beacon, port: int = DISCOVERY_PORT, dests=("255.255.255.255", "<broadcast>")):
        self.beacon, self.port, self.dests = beacon, port, tuple(dests)
        self.id = uuid.uuid4().hex
        self.seen: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.listening = False
        self.error = None

    def start(self):
        threading.Thread(target=self._send, daemon=True).start()
        threading.Thread(target=self._recv, daemon=True).start()

    def stop(self):
        self.stopping.set()

    def _send(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        while not self.stopping.is_set():
            try:
                msg = json.dumps({"app": "strata-pool", "v": 1, "id": self.id, **self.beacon()}).encode()
                for dst in self.dests:
                    try:
                        s.sendto(msg, (dst, self.port))
                        break
                    except OSError:
                        continue
            except Exception:  # noqa: BLE001 - beacons are best effort
                pass
            self.stopping.wait(BEACON_EVERY_S)
        s.close()

    def _recv(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", self.port))
            s.settimeout(1.0)
            self.listening = True
        except OSError as e:
            self.error = f"cannot listen for other PCs on UDP {self.port}: {e}"
            return
        while not self.stopping.is_set():
            try:
                data, (ip, _) = s.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                continue
            try:
                b = json.loads(data.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(b, dict) or b.get("app") != "strata-pool" or b.get("id") == self.id:
                continue
            info = {k: b.get(k) for k in ("name", "role", "model", "gpu", "vram_mib", "ram_mib", "http_port",
                                           "worker_port", "state", "version", "layers", "max_context", "primary")}
            info["ip"] = ip
            info["seen"] = time.time()
            with self.lock:
                self.seen[f"{ip}:{b.get('http_port')}"] = info
        s.close()

    def peers(self) -> list[dict]:
        now = time.time()
        with self.lock:
            for k in [k for k, v in self.seen.items() if now - v["seen"] > BEACON_FRESH_S * 4]:
                del self.seen[k]
            return [dict(v, fresh=now - v["seen"] <= BEACON_FRESH_S) for v in self.seen.values()]


def probe(http_addr: str, timeout: float = 2.0) -> dict | None:
    """A worker's public facts from its own server (GET /pool/node), or None when it does not answer."""
    try:
        with urllib.request.urlopen(f"http://{http_addr}/pool/node", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - offline, firewalled, not a Strata pool
        return None


def parse_engine_pool(info: dict) -> dict | None:
    """The coordinator engine's INFO pool_* facts -> {layers, workers: [{addr, lb, le, slots}], wire, slots_here}."""
    if info.get("pool") != "coordinator":
        return None
    workers = []
    for item in str(info.get("pool_workers") or "").split(","):
        m = re.fullmatch(r"(.+)@(\d+)-(\d+)/(\d+)", item.strip())
        if m:
            workers.append({"addr": m.group(1), "lb": int(m.group(2)), "le": int(m.group(3)) + 1,
                            "slots": int(m.group(4))})
    layers = str(info.get("pool_layers") or "")
    m = re.fullmatch(r"(\d+)-(\d+)", layers)
    return {"lb": int(m.group(1)) if m else 0, "le": int(m.group(2)) + 1 if m else None, "workers": workers,
            "wire": info.get("pool_wire"), "slots_here": info.get("pool_slots_here")}


# ------------------------------------------------------------------------------------------------ the manager
class PoolManager:
    """The server's pool side: which role this PC has, the engine arguments for it, switching roles while the server
    runs (the chat answers 503 while this PC is a worker), and what GET /pool reports."""

    def __init__(self, svc, cfg: dict, pc: PoolConfig, base_args: list[str], env: dict, http_port: int,
                 echo=lambda m: print(m, flush=True)):
        self.svc, self.cfg, self.pc = svc, cfg, pc
        self.base_args = strip_pool_args(base_args)
        self.base_env = dict(env)
        self.http_port = http_port
        self.echo = echo
        self.worker: WorkerSupervisor | None = None
        self.router = None                           # serve/route.py's Router in routing mode
        self.discovery: Discovery | None = None
        self.lock = threading.Lock()
        self.applying = None                         # what an apply in progress is doing
        self.last_error = None
        self.peer_cache: dict[str, dict] = {}
        self.peer_cache_at = 0.0
        self.addresses = lambda: []                  # the server sets its lan_addresses here
        svc.pool_role = pc.role

    # ---- the engine's command line for a role
    def engine_args(self, role: str | None = None) -> list[str]:
        role = role or self.pc.role
        if role == "coordinator":
            return self.base_args + coordinator_args(self.pc)
        if role == "worker":
            return self.base_args + worker_args(self.pc)
        return list(self.base_args)

    def engine_env(self) -> dict:
        return pool_env(self.pc, self.base_env)

    def start(self):
        """At server start: the worker engine for a worker (the chat engine is not started); discovery."""
        if self.pc.role == "worker":
            self._start_worker()
        if self.pc.role == "router":
            self._start_router()
        if self.pc["discovery"]:
            self.discovery = Discovery(self.beacon)
            self.discovery.start()

    def _start_router(self):
        from serve.route import Router
        if self.router is None:
            self.router = Router(self.svc, self, echo=self.echo)
            self.router.start()
            n = len([p for p in self.pc["route_peers"] if p["enabled"]])
            self.echo(f"[pool] routing mode: this PC runs the whole model and shares requests with {n} other PC(s)")

    def _stop_router(self):
        if self.router is not None:
            self.router.stop()
            self.router = None

    def routing(self):
        """The Router while this PC is in routing mode, else None."""
        return self.router if self.pc.role == "router" else None

    def _start_worker(self):
        self.worker = WorkerSupervisor(self.cfg["exe"], self.engine_args("worker"), cwd=self.cfg.get("cwd"),
                                       log=self.cfg.get("log"), env=self.engine_env(), echo=self.echo)
        self.worker.start()

    def load_in_background(self):
        """The chat engine (standalone or coordinator) loads on a thread: the web app stays up meanwhile and shows
        why it fails (a worker that is offline, a different model on a worker) instead of the server exiting."""
        def run():
            try:
                self.applying = "loading the model" + (" and connecting to the workers"
                                                       if self.pc.role == "coordinator" else "")
                self.svc.load()
                self.last_error = None
            except Exception as e:  # noqa: BLE001 - shown in the Pool tab
                note = ""
                if hasattr(self.svc.engine, "death_note"):
                    note = self._engine_pool_error()
                self.last_error = (note or str(e))
                self.echo(f"[pool] the model did not start: {self.last_error}")
            finally:
                self.applying = None
        threading.Thread(target=run, daemon=True).start()

    def _engine_pool_error(self) -> str:
        """The engine's own last 'strata pool:' line (a worker that is offline, a model that differs, ...)."""
        path = self.cfg.get("log")
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 16384))
                tail = f.read().decode("utf-8", "replace").splitlines()
        except (OSError, TypeError):
            return ""
        for line in reversed(tail):
            if line.startswith("strata pool:"):
                return line[len("strata pool:"):].strip()
            if line.startswith("strata generate: ") and ("failed" in line or "cannot" in line):
                return line[len("strata generate: "):].strip()
        return ""

    # ---- switching
    def apply(self, new: dict) -> dict:
        """Save `new` (the Pool tab's form) and move this PC into its role.  Raises ValueError on a bad config."""
        data = PoolConfig.clean({**self.pc.data, **new, "secret": new.get("secret") or self.pc["secret"]})
        if data["role"] == "coordinator" and not [p for p in data["peers"] if p["enabled"]]:
            raise ValueError("a coordinator needs at least one enabled worker")
        if data["role"] == "router" and not [p for p in data["route_peers"] if p["enabled"]]:
            raise ValueError("routing needs at least one other PC")
        problems = config_problems(self.cfg, PoolConfig(None, data))
        if problems:
            raise ValueError("; ".join(problems))
        with self.lock:
            if self.applying:
                raise ValueError("a change is still being applied: " + self.applying)
            old_role = self.pc.role
            self.pc.data = data
            self.pc.save()
            self.applying = "applying"
            threading.Thread(target=self._switch, args=(old_role,), daemon=True).start()
        return {"status": "applying"}

    def _switch(self, old_role: str):
        role = self.pc.role
        self.last_error = None
        self.applying = "stopping the current engine"
        try:
            if role != "router":
                self._stop_router()
            if self.worker is not None:
                self.worker.stop()
                self.worker = None
            eng = self.svc.engine
            # off <-> routing: the same engine (the whole model on this PC), so it keeps running
            if (role in ("off", "router") and old_role in ("off", "router") and hasattr(eng, "spawn") and
                    eng.spawn[1] == self.engine_args() and self.svc.loaded()):
                self.svc.pool_role = role
                if role == "router":
                    self._start_router()
                self.echo("[pool] " + ("routing mode on" if role == "router" else "routing mode off") +
                          ": the model keeps running")
                self.applying = None
                return
            with self.svc.fifo:                      # between requests
                if hasattr(eng, "alive") and eng.alive():
                    self.echo("[pool] stopping the model to change this PC's pool role ...")
                    eng.unload()
                self.svc.pool_role = role
                if hasattr(eng, "spawn"):
                    exe, _, cwd, log, _ = eng.spawn
                    eng.spawn = (exe, self.engine_args(), cwd, log, self.engine_env())
            if role == "worker":
                self.applying = None
                self.echo("[pool] this PC is now a pool worker: the chat here is off; the coordinator uses its GPU")
                self._start_worker()
            else:
                self.echo("[pool] starting the model" + (" with the pool's workers" if role == "coordinator"
                                                               else " on this PC alone"))
                self.applying = None
                if role == "router":
                    self._start_router()
                self.load_in_background()
        except Exception as e:  # noqa: BLE001 - shown in the Pool tab
            self.last_error = str(e)
            self.applying = None

    def restart_engine(self):
        """POST /pool/restart: the chat engine again (after a worker came online)."""
        if self.pc.role == "worker":
            if self.worker is not None:
                self.worker.stop()
            self._start_worker()
            return
        threading.Thread(target=self._switch, args=(self.pc.role,), daemon=True).start()

    # ---- what the Pool tab and other PCs see
    def node(self) -> dict:
        """This PC's public facts (GET /pool/node; also the beacon): no secret."""
        tel = getattr(self.svc, "telemetry", None)
        snap = tel.snapshot() if tel else {"now": {}, "static": {}}
        hw, st = snap.get("now") or {}, snap.get("static") or {}
        info = dict(getattr(self.svc.engine, "info", {}) or {})
        state = None
        if self.pc.role == "worker" and self.worker is not None:
            state = self.worker.snapshot()["state"]
        elif self.applying:
            state = "starting"
        else:
            state = "running" if self.svc.loaded() else "stopped"
        layers = None
        if self.pc.role == "worker" and self.worker is not None:
            w = self.worker.snapshot()
            if w.get("lb") is not None and w.get("le"):
                layers = f"{w['lb']}-{w['le'] - 1}"
        from serve.route import secret_id
        settings = {"sampling": getattr(self.svc, "sampling_defaults", {}), "shared": getattr(self.svc, "shared", {}),
                    "reasoning_budget_tokens": getattr(self.svc, "reasoning_budget_tokens", 0)}
        busy = False
        lock = getattr(self.svc, "status_lock", None)
        if lock is not None:
            with lock:
                busy = bool(self.svc.status.get("busy")) or bool(self.svc.status.get("queued"))
        return {"name": self.pc["name"] or hostname(), "role": self.pc.role, "model": self.svc.model,
                "max_context": int(getattr(self.svc.engine, "max_context", 0) or 0), "busy": busy,
                "loaded": bool(self.svc.loaded()), "primary": bool(self.pc["primary"]),
                "pool_id": secret_id(self.pc["secret"]),
                "settings": hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:10],
                "gpu": st.get("gpu_name"),
                "vram_mib": int(hw["gpu_mem_total"] / 2**20) if hw.get("gpu_mem_total") else None,
                "ram_mib": int(hw["ram_total"] / 2**20) if hw.get("ram_total") else None,
                "http_port": self.http_port, "worker_port": self.pc["worker_port"], "state": state,
                "version": info.get("version"), "layers": layers, "app": "strata-pool"}

    def beacon(self) -> dict:
        return self.node()

    def peers_status(self) -> list[dict]:
        """Each configured worker: its own server's facts when it has one we can reach (cached a few seconds)."""
        now = time.time()
        if now - self.peer_cache_at > 4.0:
            self.peer_cache_at = now
            disc = {(d["ip"], d.get("worker_port")): d for d in (self.discovery.peers() if self.discovery else [])}
            results = {}

            def one(p):
                host = p["addr"].rsplit(":", 1)[0].strip("[]")
                http = p.get("http")
                d = None
                for (ip, wp), info in disc.items():
                    if (ip == host or info.get("name") == host) and str(wp) == p["addr"].rsplit(":", 1)[1]:
                        d = info
                        http = http or f"{ip}:{info.get('http_port')}"
                facts = probe(http, 1.5) if http else None
                results[p["addr"]] = {"facts": facts or d, "reachable": facts is not None or (d or {}).get("fresh", False)}
            threads = [threading.Thread(target=one, args=(p,), daemon=True) for p in self.pc["peers"]]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=3.0)
            self.peer_cache = results
        return [{**p, **self.peer_cache.get(p["addr"], {"facts": None, "reachable": False})} for p in self.pc["peers"]]

    def state(self, reveal_secret: bool) -> dict:
        info = dict(getattr(self.svc.engine, "info", {}) or {})
        return {"config": self.pc.public(reveal_secret), "node": self.node(),
                "worker": self.worker.snapshot() if self.worker else None,
                "engine": {"loaded": self.svc.loaded(), "pool": parse_engine_pool(info),
                           "expert_slots": info.get("expert_slots"), "n_layers": 48},
                "peers": self.peers_status() if self.pc["peers"] else [],
                "discovered": self.discovery.peers() if self.discovery else [],
                "discovery": {"on": self.discovery is not None,
                              "listening": bool(self.discovery and self.discovery.listening),
                              "error": self.discovery.error if self.discovery else None},
                "router": self.router.state() if self.router else None,
                "applying": self.applying, "error": self.last_error,
                "problems": config_problems(self.cfg, PoolConfig(None, {**self.pc.data, "role": "coordinator"})),
                "addresses": self.addresses()}

    def close(self):
        self._stop_router()
        if self.worker is not None:
            self.worker.stop()
        if self.discovery is not None:
            self.discovery.stop()

