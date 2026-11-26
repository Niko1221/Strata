"""Several models behind one server (opt-in: `server.py --models FILE`).

One process, one port, one API key, and one complete Service per model: each member is built by server.py's own
`_main` from its ordinary run config (so every per-model setting is exactly what a one-model server of that config
would have), with its engine not started.  Only one member owns the GPU at a time.  A switch waits for the running
requests of the member that holds it (never cuts a stream), unloads it, then loads the other one; a failed load brings
the previous member back.

Requests go to the member their JSON "model" names (name or alias, any case); anything else goes to the member that
holds the GPU, whose own model check answers exactly as a one-model server does.  GET routes take `?model=NAME`.
A request for a member that does not hold the GPU gets 409 model_not_loaded unless the models file says
"auto_swap": true, which switches to it first.

The models file:

    {"models": [{"config": "strata-qwen.json"}, {"config": "strata-inkling.json", "args": ["--parallel", "2"]}],
     "default": "qwen3.8-flash-next-iq3_s", "auto_swap": false, "swap_wait_s": 120, "settle_s": 0}

"config" paths are relative to the models file; "args" are server arguments for that member only; "default" is the
member loaded at start (null: none, the first request or POST /v1/load {"model": ...} loads one; absent: the first
member).  Process arguments (--host, --port, --api-key, --offline, ...) apply to every member.
"""
from __future__ import annotations

import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

FILE_KEYS = {"models", "default", "auto_swap", "swap_wait_s", "settle_s"}
MEMBER_KEYS = {"config", "args"}
# server arguments that belong to the process, not to one member
PROCESS_ONLY = {"--models": 1, "--default-model": 1, "--owner-fd": 1, "--ready-file": 1, "--config": 1, "--open": 0,
                "--lazy": 0}
SWITCH_RETRY_S = 5


class ModelsFileError(ValueError):
    pass


def _number(raw, key, lo, hi, default):
    if raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not lo <= raw <= hi:
        raise ModelsFileError(f'"{key}" must be a number of seconds in {lo}..{hi}, not {raw!r}')
    return float(raw)


def load_models_file(path) -> dict:
    """The models file, checked: {"models": [{"config": abs path, "args": [...]}, ...], "default", "auto_swap",
    "swap_wait_s", "settle_s"}.  ModelsFileError with the reason for anything else."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as e:
        raise ModelsFileError(f"cannot read the models file {p}: {e}") from None
    if not isinstance(data, dict):
        raise ModelsFileError("the models file must be a JSON object")
    unknown = sorted(set(data) - FILE_KEYS)
    if unknown:
        raise ModelsFileError(f"the models file has unknown keys: {', '.join(unknown)}")
    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise ModelsFileError('"models" must be a non-empty list')
    out = []
    for i, m in enumerate(models):
        if not isinstance(m, dict) or not isinstance(m.get("config"), str) or not m["config"].strip():
            raise ModelsFileError(f'models entry {i} needs "config": the path of a run config')
        bad = sorted(set(m) - MEMBER_KEYS)
        if bad:
            raise ModelsFileError(f"models entry {i} has unknown keys: {', '.join(bad)}")
        args = m.get("args", [])
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise ModelsFileError(f'models entry {i}: "args" must be a list of strings')
        for a in args:
            if a.split("=", 1)[0] in PROCESS_ONLY or a in ("--host", "--port", "--api-key"):
                raise ModelsFileError(f"models entry {i}: {a} belongs to the whole server, not to one model")
        cfg = Path(m["config"])
        out.append({"config": str(cfg if cfg.is_absolute() else (p.parent / cfg)), "args": list(args)})
    default = data.get("default", ...)
    if default is not ... and default is not None and not isinstance(default, str):
        raise ModelsFileError('"default" must be a model name or null')
    auto = data.get("auto_swap", False)
    if not isinstance(auto, bool):
        raise ModelsFileError('"auto_swap" must be true or false')
    return {"models": out, "default": default, "auto_swap": auto,
            "swap_wait_s": _number(data.get("swap_wait_s"), "swap_wait_s", 0, 3600, 120.0),
            "settle_s": _number(data.get("settle_s"), "settle_s", 0, 600, 0.0)}


def member_argv(argv: list[str], member: dict) -> list[str]:
    """The server arguments one member is built with: the process's own minus the process-only ones, its config,
    --lazy (no engine starts while the members are built) and the member's own "args"."""
    out, skip = [], 0
    for a in argv:
        if skip:
            skip -= 1
            continue
        key = a.split("=", 1)[0]
        if key in PROCESS_ONLY:
            skip = PROCESS_ONLY[key] if "=" not in a else 0
            continue
        out.append(a)
    return [*out, "--config", member["config"], "--lazy", *member["args"]]


class Member:
    def __init__(self, svc, handler_class, closers=(), activate=None):
        self.svc = svc
        self.name = svc.model
        self.handler_class = handler_class
        self.closers = list(closers)
        self.inflight = 0                               # requests the router handed to this member and not yet back
        # the serve settings its run config keeps in modules (server.py _process_state: the family's effort words,
        # env_to_user, engine_ready_s), put back each time this member takes the GPU; one process holds them all
        self.activate = activate or (lambda: None)

    def names(self) -> set[str]:
        return {n.strip().lower() for n in [self.svc.model, *(self.svc.aliases or [])] if isinstance(n, str) and n.strip()}


class SwitchError(Exception):
    """A switch that did not happen: the HTTP status, a code and the sentence."""

    def __init__(self, status: int, code: str, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.status, self.code, self.retry_after = status, code, retry_after


class ModelHost:
    """The members and which one holds the GPU.  `active` is the member that owns the cards (loaded, loading, or
    unloaded by its own idle timer); `switching` the member a switch is bringing up."""

    def __init__(self, members: list[Member], *, default=..., auto_swap: bool = False, swap_wait_s: float = 120.0,
                 settle_s: float = 0.0):
        if not members:
            raise ModelsFileError("no models")
        self.members: list[Member] = []
        seen: dict[str, str] = {}
        for m in members:
            for n in m.names():
                if n in seen:
                    raise ModelsFileError(f"two models answer to the name {n!r} ({seen[n]} and {m.name})")
                seen[n] = m.name
            self.members.append(m)
        keys = {repr(sorted(_keys(m.svc.api_key))) for m in members}
        if len(keys) > 1:
            raise ModelsFileError("every model must use the same API key(s): give the key to the server "
                                  "(--api-key / STRATA_API_KEY), not in one model's config")
        if default is ...:
            default = self.members[0].name
        if default is not None and self.find(default) is None:
            raise ModelsFileError(f"default model {default!r} is not in the models file "
                                  f"({', '.join(m.name for m in self.members)})")
        self.default = self.find(default).name if default is not None else None
        self.auto_swap, self.swap_wait_s, self.settle_s = auto_swap, swap_wait_s, settle_s
        self.cond = threading.Condition()
        self.active: str | None = None
        self.switching: str | None = None
        self.last_switch: dict = {}

    # ------------------------------------------------------------------ lookup
    def find(self, name) -> Member | None:
        if not isinstance(name, str) or not name.strip():
            return None
        n = name.strip().lower()
        return next((m for m in self.members if n in m.names()), None)

    def member(self, name: str) -> Member:
        return next(m for m in self.members if m.name == name)

    def current(self) -> Member:
        """The member a request without a model of its own goes to: the one a switch brings up, else the GPU's owner,
        else the default one, else the first."""
        name = self.switching or self.active or self.default
        return self.member(name) if name else self.members[0]

    def route(self, asked) -> Member:
        return self.find(asked) or self.current()

    # ------------------------------------------------------------------ admission
    def admit(self, m: Member, deadline_s: float | None = None) -> None:
        """Before a request goes to member m: count it in, or raise SwitchError.  A request for the member a switch is
        bringing up waits for it; one for any other member is refused while a switch runs."""
        deadline = time.monotonic() + (deadline_s if deadline_s is not None else self.swap_wait_s + 1800.0)
        while True:
            with self.cond:
                while self.switching is not None:
                    if self.switching != m.name:
                        raise SwitchError(503, "model_switching",
                                          f"the server is switching to {self.switching}; retry shortly",
                                          retry_after=SWITCH_RETRY_S)
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise SwitchError(503, "model_switching", f"{m.name} is still loading; retry shortly",
                                          retry_after=SWITCH_RETRY_S)
                    self.cond.wait(min(1.0, left))
                if self.active in (None, m.name):
                    if self.active is None:
                        m.activate()
                    self.active = m.name
                    m.inflight += 1
                    return
                if not self.auto_swap:
                    raise SwitchError(409, "model_not_loaded",
                                      f'model {m.name} is configured but not loaded ({self.active} is); load it with '
                                      f'POST /v1/load {{"model": "{m.name}"}}')
            self.switch(m.name)                         # auto_swap: this request brings its model up, then goes in

    def leave(self, m: Member) -> None:
        with self.cond:
            m.inflight -= 1
            self.cond.notify_all()

    # ------------------------------------------------------------------ switching
    def switch(self, name: str, wait_s: float | None = None) -> dict:
        """Make member `name` the one that holds the GPU and load it.  -> {"status": "loaded", "model", "from",
        "seconds"}; SwitchError when it cannot (busy, another switch, a load that failed: the previous member is
        loaded back first)."""
        target = self.find(name)
        if target is None:
            raise SwitchError(404, "model_not_found", f"model {name!r} is not one of this server's models")
        wait_s = self.swap_wait_s if wait_s is None else max(0.0, wait_s)
        t0 = time.monotonic()
        with self.cond:
            while self.switching == target.name:      # the same switch is already on its way: wait for its outcome
                self.cond.wait(1.0)
            if self.switching is not None:
                raise SwitchError(409, "model_switching", f"the server is already switching to {self.switching}",
                                  retry_after=SWITCH_RETRY_S)
            old = self.member(self.active) if self.active not in (None, target.name) else None
            self.switching = target.name
            self.cond.notify_all()
            deadline = time.monotonic() + wait_s
            while old is not None and old.inflight > 0:
                left = deadline - time.monotonic()
                if left <= 0:
                    self.switching = None
                    self.cond.notify_all()
                    raise SwitchError(409, "model_busy", f"{old.name} still has {old.inflight} request(s) running "
                                      f"after {wait_s:g} s; try again when they have finished")
                self.cond.wait(min(0.25, left))
        ok = False
        try:
            if old is not None:
                r = old.svc.unload_waiting(max(0.0, deadline - time.monotonic()))
                if r == "busy":
                    raise SwitchError(409, "model_busy", f"{old.name} is busy; try again when it has finished")
                if r == "unsupported":
                    raise SwitchError(409, "model_busy", f"{old.name} cannot be unloaded by this server")
                print(f"[strata] models: {old.name} {r} ({time.monotonic() - t0:.1f} s)", flush=True)
                if self.settle_s:
                    time.sleep(self.settle_s)
            try:
                target.activate()
                target.svc.load()
            except Exception as e:                      # noqa: BLE001 - every failure takes the rollback below
                self._rollback(target, old, e)
            ok = True
        finally:
            with self.cond:
                self.switching = None
                if ok:
                    self.active = target.name
                self.cond.notify_all()
        out = {"status": "loaded", "model": target.name, "from": old.name if old else None,
               "seconds": round(time.monotonic() - t0, 2)}
        self.last_switch = {**out, "at": time.time()}
        print(f"[strata] models: {target.name} loaded{f' (was {old.name})' if old else ''} in {out['seconds']} s",
              flush=True)
        return out

    def _rollback(self, target: Member, old: Member | None, e: Exception):
        print(f"[strata] models: {target.name} could not be loaded ({e})"
              + (f"; loading {old.name} again" if old else ""), flush=True)
        try:
            target.svc.unload()
        except Exception:                               # noqa: BLE001 - the engine may be gone already
            pass
        back = None
        if old is not None:
            try:
                old.activate()
                old.svc.load()
                back = old.name
            except Exception as e2:                     # noqa: BLE001
                print(f"[strata] models: {old.name} could not be loaded again either ({e2}); no model is loaded",
                      flush=True)
        with self.cond:
            self.active = back
        self.last_switch = {"status": "failed", "model": target.name, "error": str(e), "back": back,
                            "at": time.time()}
        raise SwitchError(503, "model_load_failed", f"{target.name} could not be loaded: {e}"
                          + (f"; {back} is loaded again" if back else "; no model is loaded")) from e

    def unload(self, name: str, wait_s: float = 0.0) -> str:
        """POST /v1/unload {"model": name}: the member's own unload; it keeps the GPU's ownership only while loaded."""
        m = self.find(name)
        if m is None:
            raise SwitchError(404, "model_not_found", f"model {name!r} is not one of this server's models")
        with self.cond:
            if self.switching is not None:
                raise SwitchError(409, "model_switching", f"the server is switching to {self.switching}",
                                  retry_after=SWITCH_RETRY_S)
        r = m.svc.unload_waiting(wait_s) if wait_s else m.svc.unload()
        if r in ("unloaded", "not loaded", "asleep"):
            with self.cond:
                if self.active == m.name and m.inflight == 0:
                    self.active = None
        return r

    # ------------------------------------------------------------------ status
    def model_list(self) -> list[dict]:
        data = []
        for m in self.members:
            entries = m.svc.model_list() or [{"id": m.name, "object": "model", "owned_by": "strata"}]
            state = ("loading" if self.switching == m.name else
                     "loaded" if self.active == m.name and m.svc.loaded() else "unloaded")
            for e in entries:
                if e.get("id") == m.name or e.get("alias_of") == m.name:     # not the image model a config may add
                    e = {**e, "status": {"value": state}}
                data.append(e)
        return data

    def status(self) -> dict:
        return {"active": self.active, "switching": self.switching, "default": self.default,
                "auto_swap": self.auto_swap, "models": [m.name for m in self.members],
                "last_switch": self.last_switch}

    def close(self) -> None:
        for m in self.members:
            for close in m.closers:
                try:
                    close()
                except Exception as e:                  # noqa: BLE001 - the other members still close
                    print(f"[strata] models: closing {m.name}: {e}", flush=True)


def _keys(key) -> list[str]:
    if not key:
        return []
    if isinstance(key, (list, tuple)):
        return [str(k) for k in key]
    return [k.strip() for k in str(key).split(",") if k.strip()]


def make_router_handler(host: ModelHost, bad_body=Exception):
    """The HTTP handler of a models server: reads the request line and headers with the first member's handler (its
    Host check, its limits), picks the member, and lets that member's own handler answer the request."""
    base = host.members[0].handler_class

    class Router(BaseHTTPRequestHandler):
        timeout = getattr(base, "timeout", None)
        protocol_version = "HTTP/1.0"

        def log_message(self, fmt, *args):
            pass

        def _delegate(self, member: Member, parsed=None, body: bytes | None = None):
            cls = member.handler_class
            d = cls.__new__(cls)
            for k in ("request", "client_address", "server", "connection", "rfile", "wfile"):
                setattr(d, k, getattr(self, k))
            d.close_connection = True
            d.answer_started = False
            if parsed is not None:
                for k in ("raw_requestline", "requestline", "command", "path", "request_version", "headers"):
                    setattr(d, k, getattr(parsed, k))
            if body is not None:                        # the body the router read, handed over from memory
                d.rfile = io.BytesIO(body)
                for h in ("Transfer-Encoding", "Content-Length"):
                    del d.headers[h]
                d.headers["Content-Length"] = str(len(body))
            return d

        def handle(self):
            self.close_connection = True
            p = self._delegate(host.current())
            p.raw_requestline = p.rfile.readline(65537)
            if len(p.raw_requestline) > 65536:
                p.requestline, p.request_version, p.command = "", "", ""
                p.send_error(414)
                return
            if not p.raw_requestline:
                return
            if not p.parse_request():                   # the member's own check (Host header, malformed lines)
                return
            try:
                self._route(p)
            finally:
                try:
                    p.wfile.flush()
                except OSError:
                    pass

        def _dispatch(self, d):
            method = getattr(d, "do_" + d.command, None)
            if method is None:
                d.send_error(501, f"Unsupported method ({d.command!r})")
                return
            method()
            d._drain_body()

        def _route(self, p):
            path = p.path.split("?")[0].rstrip("/")
            query = parse_qs(urlsplit(p.path).query)
            if p.command == "POST":
                if not p._authorized():
                    return
                try:
                    body = p._body()
                except bad_body as e:
                    p._json(e.status, {"error": {"type": "invalid_request_error", "message": str(e)}})
                    return
                try:
                    req = json.loads(body or b"{}")
                except ValueError:
                    req = None
                asked = req.get("model") if isinstance(req, dict) else None
                if path in ("/v1/load", "/v1/unload") and isinstance(asked, str) and asked.strip():
                    self._load_unload(p, path, req)
                    return
                m = host.route(asked)
                try:
                    host.admit(m)
                except SwitchError as e:
                    self._refuse(p, path, e)
                    return
                try:
                    self._dispatch(self._delegate(m, p, body))
                finally:
                    host.leave(m)
                return
            if p.command == "GET" and (path in ("/v1/models", "/models") or path.startswith(("/v1/models/", "/models/"))):
                if path.startswith("/v1/") and p._foreign_page(body=False):
                    return
                if not p._authorized():
                    return
                data = host.model_list()
                if path in ("/v1/models", "/models"):
                    p._json(200, {"object": "list", "data": data, "active": host.active, "switching": host.switching,
                                  "last_switch": host.last_switch})
                    return
                asked = unquote(path[len("/v1/models/" if path.startswith("/v1/") else "/models/"):])
                item = next((x for x in data if x.get("id") == asked), None)
                if item is None:
                    p._json(404, {"error": {"message": f"The model '{asked}' does not exist",
                                            "type": "invalid_request_error", "param": "model", "code": "model_not_found"}})
                else:
                    p._json(200, item)
                return
            m = host.find((query.get("model") or [None])[0]) or host.current()
            self._dispatch(self._delegate(m, p))

        def _refuse(self, p, path, e: SwitchError):
            error = {"type": "server_error" if e.status >= 500 else "invalid_request_error", "code": e.code,
                     "message": str(e)}
            if path == "/v1/responses":
                body = {"error": {**error, "param": None}}
            else:
                body = {"error": error}
            p._json(e.status, body, retry_after=e.retry_after)

        def _load_unload(self, p, path, req):
            if p._foreign_page():
                return
            if not p._own_page("the model can be loaded or unloaded"):
                return
            try:
                wait = parse_qs(urlsplit(p.path).query).get("wait", [None])[0]
                wait_s = float(wait) if wait not in (None, "", "0") else 0.0
                if path == "/v1/load":
                    out = host.switch(req["model"], wait_s=wait_s or None)
                    m = host.member(out["model"])
                    p._json(200, {**out, **m.svc.v1_status()})
                else:
                    r = host.unload(req["model"], wait_s)
                    if r == "busy":
                        raise SwitchError(409, "model_busy", "a request is running or queued")
                    p._json(200, {"status": r, "model": host.find(req["model"]).name})
            except SwitchError as e:
                self._refuse(p, path, e)
            except ValueError:
                p._json(400, {"error": {"type": "invalid_request_error", "message": "wait must be a number of seconds"}})

    return Router
