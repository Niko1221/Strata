"""serve/route.py - the pool's routing mode: every PC runs the whole model on its own, and a chat request that
reaches any of them runs on whichever PC suits it best.

A request is sent to another PC only when that helps:

- A conversation stays on the PC that holds its context (the engine's prompt cache and parked chats are per PC):
  moving it means reading the whole prompt again there, which takes minutes for a long chat.  The router knows which
  PC has served a conversation from a chain of hashes over its messages (each PC keeps the chains of what it ran and
  answers a probe with how much of a new request it has seen).
- A conversation moves only when its PC is busy, another is idle, and the part the other PC would have to read again
  is short (`move_tokens`).
- A new conversation goes to an idle PC: the primary first, then the PC it arrived at, then any other.
- A request goes only to a PC whose context window holds it.  Each PC keeps its own window and says so to its own
  clients; a chat longer than a peer's window simply never goes there.
- When the other PCs cannot be reached, each PC is an ordinary Strata server and keeps asking for them every few
  seconds; nothing waits on them.

PCs of a pool prove themselves to each other with the pool secret: a routed request and a probe carry an HMAC over
the time, a nonce, the path and the body's hash.  A peer's public facts (GET /pool/node) need no secret.
"""
from __future__ import annotations

import collections
import hashlib
import hmac
import http.client
import json
import secrets
import threading
import time
import urllib.request

AUTH_HEADER = "X-Strata-Pool-Auth"
ROUTED_HEADER = "X-Strata-Pool-Routed"
AUTH_SKEW_S = 120.0
HEARTBEAT_S = 3.0
PROBE_TIMEOUT_S = 1.5
CHARS_PER_TOKEN = 3.5        # a rough count of the prompt: enough to compare with a context window and move_tokens
TABLE_MAX = 50000            # conversation chain hashes each PC remembers (newest kept)
FORWARD_HEADERS = ("Content-Type", "Accept", "anthropic-version", "anthropic-beta", "User-Agent")


# ------------------------------------------------------------------------------------------------ auth
def sign(secret: str, path: str, body: bytes, now: float | None = None) -> str:
    ts = str(int(now if now is not None else time.time()))
    nonce = secrets.token_hex(8)
    mac = hmac.new(secret.encode(), f"{ts}|{nonce}|{path}|".encode() + hashlib.sha256(body).digest(),
                   hashlib.sha256).hexdigest()
    return f"{ts}.{nonce}.{mac}"


class Verifier:
    """Checks AUTH_HEADER values: the pool's secret, a recent time, a nonce not seen before."""

    def __init__(self, secret_fn):
        self.secret_fn = secret_fn
        self.seen: collections.OrderedDict[str, float] = collections.OrderedDict()
        self.lock = threading.Lock()

    def ok(self, value: str | None, path: str, body: bytes, now: float | None = None) -> bool:
        secret = self.secret_fn()
        if not value or not secret:
            return False
        try:
            ts, nonce, mac = value.split(".")
            t = int(ts)
        except ValueError:
            return False
        now = now if now is not None else time.time()
        if abs(now - t) > AUTH_SKEW_S:
            return False
        want = hmac.new(secret.encode(), f"{ts}|{nonce}|{path}|".encode() + hashlib.sha256(body).digest(),
                        hashlib.sha256).hexdigest()
        if not hmac.compare_digest(want, mac):
            return False
        with self.lock:
            for k in [k for k, v in self.seen.items() if now - v > AUTH_SKEW_S * 2]:
                del self.seen[k]
            if nonce in self.seen:
                return False                          # a replay
            self.seen[nonce] = now
        return True


# ------------------------------------------------------------------------------------------------ conversations
def _strip(o):
    """A message as the router compares it: without the prompt-caching markers clients move from turn to turn."""
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k != "cache_control"}
    if isinstance(o, list):
        return [_strip(x) for x in o]
    return o


def _text_len(o) -> int:
    if isinstance(o, str):
        return len(o)
    if isinstance(o, dict):
        return sum(_text_len(v) for v in o.values())
    if isinstance(o, list):
        return sum(_text_len(x) for x in o)
    return 0


def chain(path: str, req: dict) -> tuple[list[str], list[int]]:
    """The request's conversation as a chain of hashes - one per message, each over everything before it - and the
    characters up to each.  Two requests of one conversation share the chain's start."""
    head = {"path": path, "tools": _strip(req.get("tools")), "system": _strip(req.get("system"))}
    h = hashlib.sha256(json.dumps(head, sort_keys=True, ensure_ascii=False).encode()).digest()
    total = _text_len(head["tools"]) + _text_len(head["system"])
    hashes, chars = [], []
    for m in req.get("messages") or []:
        h = hashlib.sha256(h + json.dumps(_strip(m), sort_keys=True, ensure_ascii=False).encode()).digest()
        total += _text_len(m)
        hashes.append(h.hex()[:24])
        chars.append(total)
    return hashes, chars


def est_tokens(chars: int) -> int:
    return int(chars / CHARS_PER_TOKEN)


class Table:
    """The conversations this PC ran (their chain hashes, newest kept)."""

    def __init__(self, cap: int = TABLE_MAX):
        self.cap = cap
        self.d: collections.OrderedDict[str, float] = collections.OrderedDict()
        self.lock = threading.Lock()

    def add(self, hashes: list[str]):
        now = time.time()
        with self.lock:
            for h in hashes:
                self.d[h] = now
                self.d.move_to_end(h)
            while len(self.d) > self.cap:
                self.d.popitem(last=False)

    def match(self, hashes: list[str]) -> int:
        """How many messages from the start of `hashes` this PC has seen (the conversation's longest known prefix)."""
        with self.lock:
            for i in range(len(hashes) - 1, -1, -1):
                if hashes[i] in self.d:
                    return i + 1
        return 0


# ------------------------------------------------------------------------------------------------ the router
class Router:
    """Routing mode's server side: the peers' state (a heartbeat), the choice for each request, and the counters the
    Pool tab shows.  `svc` is the server's Service, `pm` its PoolManager (config, name)."""

    def __init__(self, svc, pm, echo=print):
        self.svc, self.pm, self.echo = svc, pm, echo
        self.table = Table()
        self.verifier = Verifier(lambda: self.pm.pc["secret"])
        self.lock = threading.Lock()
        self.peers: dict[str, dict] = {}            # http addr -> {facts, reachable, since, inflight, error}
        self.local_active = 0                       # requests this PC is running now (its own and routed in)
        self.counts = {"local": 0, "routed_out": 0, "routed_in": 0, "fallback": 0}
        self.recent = collections.deque(maxlen=20)   # the last decisions (Pool tab)
        self.stopping = threading.Event()
        self.thread = None

    # ---- configuration
    def peer_addrs(self) -> list[str]:
        return [p["addr"] for p in self.pm.pc["route_peers"] if p["enabled"]]

    def start(self):
        self.stopping.clear()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def stop(self):
        self.stopping.set()

    def _heartbeat(self):
        while not self.stopping.is_set():
            addrs = self.peer_addrs()
            threads = [threading.Thread(target=self._check, args=(a,), daemon=True) for a in addrs]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=PROBE_TIMEOUT_S + 1)
            with self.lock:
                for a in [a for a in self.peers if a not in addrs]:
                    del self.peers[a]
            self.stopping.wait(HEARTBEAT_S)

    def _check(self, addr: str):
        facts, err = None, None
        try:
            with urllib.request.urlopen(f"http://{addr}/pool/node", timeout=PROBE_TIMEOUT_S) as r:
                facts = json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001 - offline, firewalled, not a Strata pool
            err = str(getattr(e, "reason", e)) or type(e).__name__
        usable, why = self.usable(facts)
        with self.lock:
            p = self.peers.setdefault(addr, {"inflight": 0, "reachable": False, "since": time.time(), "facts": None})
            was = p["reachable"] and p.get("usable")
            if bool(facts) != p["reachable"]:
                p["since"] = time.time()
            p.update(facts=facts, reachable=facts is not None, usable=usable, why=why if facts else err,
                     checked=time.time())
            now_ok = p["reachable"] and usable
        if now_ok != was:
            name = (facts or {}).get("name") or addr
            self.echo(f"[pool] routing: {name} ({addr}) " + ("joined: requests can run there" if now_ok else
                      f"is out of reach ({why if facts else err}): this PC serves its own requests"))

    def usable(self, facts: dict | None) -> tuple[bool, str]:
        if not facts:
            return False, "no answer"
        if facts.get("app") != "strata-pool":
            return False, "not a Strata pool server"
        if facts.get("role") != "router":
            return False, f"its pool role is {facts.get('role') or 'off'}, not routing"
        if facts.get("model") != self.svc.model:
            return False, f"another model ({facts.get('model')})"
        if facts.get("pool_id") and facts.get("pool_id") != secret_id(self.pm.pc["secret"]):
            return False, "another pool secret"
        return True, ""

    # ---- what a probe answers (this PC's side)
    def local_view(self) -> dict:
        with self.svc.status_lock:
            s = dict(self.svc.status)
        return {"busy": bool(s.get("busy")) or self.local_active > 0, "queued": int(s.get("queued") or 0),
                "active": self.local_active, "max_context": int(getattr(self.svc.engine, "max_context", 0) or 0),
                "loaded": bool(self.svc.loaded())}

    def answer_probe(self, body: dict) -> dict:
        hashes = [str(h) for h in (body.get("hashes") or [])][:4096]
        return {**self.local_view(), "match": self.table.match(hashes), "name": self.pm.pc["name"],
                "primary": bool(self.pm.pc["primary"]), "model": self.svc.model}

    def note_local(self, path: str, req: dict, routed_in: bool):
        hashes, _ = chain(path, req)
        self.table.add(hashes)
        with self.lock:
            self.local_active += 1
            self.counts["routed_in" if routed_in else "local"] += 1

    def done_local(self):
        with self.lock:
            self.local_active = max(0, self.local_active - 1)

    # ---- the choice
    def _probe(self, addr: str, hashes: list[str], out: dict):
        body = json.dumps({"hashes": hashes, "from": self.pm.pc["name"]}).encode()
        try:
            host, _, port = addr.rpartition(":")
            c = http.client.HTTPConnection(host.strip("[]"), int(port), timeout=PROBE_TIMEOUT_S)
            c.request("POST", "/pool/route/probe", body, {"Content-Type": "application/json",
                                                          AUTH_HEADER: sign(self.pm.pc["secret"], "/pool/route/probe", body)})
            r = c.getresponse()
            data = r.read()
            c.close()
            if r.status == 200:
                out[addr] = json.loads(data.decode("utf-8"))
        except Exception:  # noqa: BLE001 - it went away since the heartbeat: not a candidate
            pass

    def choose(self, path: str, req: dict) -> dict | None:
        """The peer to run this request on ({addr, facts, ...}), or None for this PC."""
        hashes, chars = chain(path, req)
        total = chars[-1] if chars else 0
        tokens = est_tokens(total)
        with self.lock:
            live = {a: dict(p) for a, p in self.peers.items() if p.get("reachable") and p.get("usable")}
        if not live:
            return None
        probes: dict[str, dict] = {}
        threads = [threading.Thread(target=self._probe, args=(a, hashes, probes), daemon=True) for a in live]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=PROBE_TIMEOUT_S + 0.5)
        me = self.local_view()
        cands = [{"addr": None, "match": self.table.match(hashes), **me, "primary": bool(self.pm.pc["primary"]),
                  "local": True}]
        for a, pr in probes.items():
            with self.lock:
                infl = self.peers.get(a, {}).get("inflight", 0)
            cands.append({"addr": a, **pr, "busy": bool(pr.get("busy")) or infl > 0, "local": False,
                          "facts": live[a].get("facts")})
        decision = pick(cands, chars, tokens, int(self.pm.pc["move_tokens"]))
        why = decision.pop("why", "")
        target = None if decision.get("local") else decision
        with self.lock:
            self.recent.append({"at": time.time(), "to": "this PC" if target is None else
                                ((target.get("facts") or {}).get("name") or target["addr"]),
                                "why": why, "tokens": tokens})
        return target

    def begin_forward(self, addr: str):
        with self.lock:
            self.peers.setdefault(addr, {"inflight": 0})["inflight"] = self.peers.get(addr, {}).get("inflight", 0) + 1
            self.counts["routed_out"] += 1

    def end_forward(self, addr: str, fell_back: bool = False):
        with self.lock:
            p = self.peers.get(addr)
            if p is not None:
                p["inflight"] = max(0, p.get("inflight", 0) - 1)
            if fell_back:
                self.counts["routed_out"] -= 1
                self.counts["fallback"] += 1

    def state(self) -> dict:
        with self.lock:
            peers = {a: {k: v for k, v in p.items()} for a, p in self.peers.items()}
            counts, recent = dict(self.counts), list(self.recent)
        return {"peers": peers, "counts": counts, "recent": recent, "local": self.local_view(),
                "islanded": bool(self.peer_addrs()) and not any(p.get("reachable") and p.get("usable")
                                                                for p in peers.values())}


def secret_id(secret: str) -> str:
    """A short public fingerprint of the pool secret: two PCs can tell they hold different secrets, nobody learns
    the secret (a salted hash, truncated)."""
    return hashlib.sha256(b"strata-pool-secret-id|" + secret.encode()).hexdigest()[:8] if secret else ""


def pick(cands: list[dict], chars: list[int], tokens: int, move_tokens: int) -> dict:
    """The decision (pure, for tests).  `cands[0]` is this PC.  Each: match (messages seen), busy, queued, max_context,
    loaded, primary, local.  Returns the chosen candidate with a `why`."""
    total = chars[-1] if chars else 0

    def fits(c):
        mc = c.get("max_context") or 0
        return mc <= 0 or tokens <= 0.95 * mc

    def seen_chars(c):
        m = c.get("match") or 0
        return chars[m - 1] if m > 0 and m <= len(chars) else 0

    def free(c):
        return not c.get("busy") and not c.get("queued") and c.get("loaded", True)

    def pref(c):        # among equals: the primary, then this PC
        return (bool(c.get("primary")), bool(c.get("local")))

    me = cands[0]
    ok = [c for c in cands if fits(c)]
    if not ok:
        return {**me, "why": "no PC's context window holds it: this PC answers (and reports it)"}
    best_seen = max(ok, key=lambda c: (seen_chars(c), free(c), pref(c)))
    if seen_chars(best_seen) > 0:
        if free(best_seen):
            return {**best_seen, "why": "the PC that holds this conversation"}
        idle = [c for c in ok if free(c)]
        if idle:
            alt = max(idle, key=lambda c: (seen_chars(c), pref(c)))
            reread = est_tokens(total - seen_chars(alt))
            if reread <= move_tokens:
                return {**alt, "why": f"its PC is busy; this one reads ~{reread} tokens again"}
        return {**best_seen, "why": "the PC that holds this conversation (busy: it waits there, a move would read "
                                    "the whole conversation again)"}
    idle = [c for c in ok if free(c)]
    if idle:
        return {**max(idle, key=pref), "why": "a new conversation, to an idle PC"}
    return {**min(ok, key=lambda c: (int(c.get("queued") or 0) + (1 if c.get("busy") else 0),
                                     not c.get("primary"), not c.get("local"))),
            "why": "every PC is busy: the shortest queue"}
